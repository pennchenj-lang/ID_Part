from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import CharacterPartDataset, class_weights
from .losses import LossWeights, hpid_split_loss
from .model import HPIDSplitNet
from .taxonomy import Taxonomy


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _batchnorm_eval(module: nn.Module) -> None:
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()


def _parent_weights(fine_weights: torch.Tensor, taxonomy: Taxonomy) -> torch.Tensor:
    values = torch.ones(taxonomy.num_parent_classes)
    for parent_id in range(taxonomy.num_parent_classes):
        children = [
            fine_id
            for fine_id, mapped_parent in enumerate(taxonomy.fine_to_parent)
            if mapped_parent == parent_id
        ]
        values[parent_id] = fine_weights[children].mean()
    values[0] = min(float(values[0]), 0.18)
    return values


def train(
    image_dir: Path,
    label_dir: Path,
    role_ids: list[str],
    taxonomy: Taxonomy,
    checkpoint_path: Path,
    *,
    epochs: int = 20,
    repeats: int = 8,
    batch_size: int = 2,
    device: str = "cuda",
    seed: int = 20260811,
    focus_probability: float = 0.65,
    loss_weights: LossWeights | None = None,
) -> pd.DataFrame:
    loss_weights = loss_weights or LossWeights()
    _set_seed(seed)
    dataset = CharacterPartDataset(
        image_dir,
        label_dir,
        role_ids,
        taxonomy,
        repeats=repeats,
        focus_probability=focus_probability,
        seed=seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )
    model = HPIDSplitNet(taxonomy, pretrained_backbone=True).to(device)
    cached = [(dataset.cache[role][0], dataset.cache[role][1]) for role in role_ids]
    fine_weights = class_weights(cached, taxonomy.num_fine_classes).to(device)
    parent_weights = _parent_weights(fine_weights.cpu(), taxonomy).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": 1.2e-4},
            {"params": model.heads.parameters(), "lr": 8e-4},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    rows: list[dict[str, float | int]] = []
    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        model.train()
        model.apply(_batchnorm_eval)
        batches: list[dict[str, float]] = []
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                loss, components = hpid_split_loss(
                    model(images),
                    labels,
                    taxonomy,
                    fine_class_weights=fine_weights,
                    parent_class_weights=parent_weights,
                    weights=loss_weights,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            batches.append(components)
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            **{
                key: float(np.mean([batch[key] for batch in batches]))
                for key in batches[0]
            },
        }
        rows.append(row)
        print(f"epoch={epoch + 1}/{epochs} loss={row['loss']:.5f}", flush=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "taxonomy": taxonomy.to_dict(),
            "role_ids": role_ids,
            "epochs": epochs,
            "repeats": repeats,
            "seed": seed,
            "loss_weights": asdict(loss_weights),
            "inference_uses_ground_truth": False,
        },
        checkpoint_path,
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(checkpoint_path.with_suffix(".history.csv"), index=False)
    return frame


def load_checkpoint(
    path: Path, device: str
) -> tuple[HPIDSplitNet, Taxonomy, dict[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    taxonomy = Taxonomy.from_dict(payload["taxonomy"])
    model = HPIDSplitNet(taxonomy, pretrained_backbone=False).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, taxonomy, payload
