from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from hpid_split.conditional_parts import (
    GroupedConditionalPartCollator,
    GroupedConditionalPartDataset,
    grouped_conditional_part_loss,
    load_conditional_part_samples,
    split_conditional_samples,
    threshold_metrics,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move(values: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in values.items()}


def _evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    device: str,
    thresholds: tuple[float, ...],
) -> dict[str, object]:
    model.eval()
    losses: list[dict[str, float]] = []
    probabilities: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    roots: list[torch.Tensor] = []
    positives: list[torch.Tensor] = []
    overlap_by_threshold: dict[float, list[float]] = {
        threshold: [] for threshold in thresholds
    }
    with torch.inference_mode():
        for batch in loader:
            inputs = _move(batch.model_inputs, device)
            target = batch.targets.to(device, non_blocking=True)
            root = batch.root_masks.to(device, non_blocking=True)
            positive = batch.positive.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
                logits = model(**inputs).logits
                _, components = grouped_conditional_part_loss(
                    logits, target, root, positive
                )
            probability = logits.sigmoid() * root
            losses.append(components)
            probabilities.append(probability.cpu())
            targets.append(target.cpu())
            roots.append(root.cpu())
            positives.append(positive.cpu())
            physical_root = root[0].bool()
            root_area = max(1, int(physical_root.sum()))
            for threshold in thresholds:
                predicted = (probability >= threshold) & root.bool()
                surplus = (predicted.sum(dim=0) - 1).clamp_min(0)
                overlap_by_threshold[threshold].append(
                    float(surplus[physical_root].sum().cpu()) / root_area
                )
    metrics = threshold_metrics(
        torch.cat(probabilities),
        torch.cat(targets),
        torch.cat(roots),
        torch.cat(positives),
        thresholds,
    )
    for threshold, row in metrics.items():
        overlap = float(np.mean(overlap_by_threshold[threshold]))
        row["sibling_overlap_root_fraction"] = overlap
        row["selection_score"] = (
            row["positive_mean_iou"]
            - 0.25 * row["negative_false_area_fraction"]
            - 0.15 * overlap
        )
    best_threshold, best = max(
        metrics.items(), key=lambda item: item[1]["selection_score"]
    )
    return {
        "loss": float(np.mean([row["loss"] for row in losses])),
        "competition": float(np.mean([row["competition"] for row in losses])),
        "overlap_loss": float(np.mean([row["overlap"] for row in losses])),
        "best_threshold": best_threshold,
        **best,
        "thresholds": {str(key): value for key, value in metrics.items()},
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune conditional part masks jointly per asset so sibling IDs "
            "compete for visible ownership."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--maximum-queries", type=int, default=12)
    parser.add_argument("--negative-ratio", type=float, default=0.35)
    parser.add_argument("--validation-fraction", type=float, default=0.17)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    _seed_everything(args.seed)

    samples = load_conditional_part_samples(
        args.manifest,
        negative_ratio=args.negative_ratio,
        seed=args.seed,
    )
    train_samples, validation_samples = split_conditional_samples(
        samples,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    train_assets = {sample.asset_id for sample in train_samples}
    validation_assets = {sample.asset_id for sample in validation_samples}
    if train_assets & validation_assets:
        raise RuntimeError("asset leakage detected between grouped splits")

    try:
        from transformers import AutoProcessor, CLIPSegForImageSegmentation
    except ImportError as error:
        raise RuntimeError("Install the foundation extra before training") from error
    processor = AutoProcessor.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )
    model = CLIPSegForImageSegmentation.from_pretrained(
        args.model, local_files_only=args.local_files_only
    ).to(args.device)
    for parameter in model.clip.parameters():
        parameter.requires_grad_(False)
    trainable = [
        parameter for parameter in model.decoder.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.device == "cuda")

    train_dataset = GroupedConditionalPartDataset(
        train_samples,
        training=True,
        seed=args.seed,
        maximum_queries=args.maximum_queries,
    )
    validation_dataset = GroupedConditionalPartDataset(
        validation_samples,
        training=False,
        seed=args.seed,
        maximum_queries=args.maximum_queries,
        root_suppression_probability=0.0,
    )
    collator = GroupedConditionalPartCollator(processor)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        generator=generator,
        collate_fn=collator,
        num_workers=0,
        pin_memory=args.device == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=args.device == "cuda",
    )
    thresholds = tuple(float(value) for value in np.arange(0.25, 0.61, 0.05))
    args.output.mkdir(parents=True, exist_ok=True)
    split = {
        "format": "HPID grouped conditional part split",
        "format_version": "0.1.0",
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": _sha256(args.manifest),
        "initial_model": args.model,
        "seed": args.seed,
        "maximum_queries": args.maximum_queries,
        "train_asset_count": len(train_assets),
        "validation_asset_count": len(validation_assets),
        "train_validation_asset_overlap": [],
    }
    (args.output / "split_manifest.json").write_text(
        json.dumps(split, indent=2), encoding="utf-8"
    )

    baseline = _evaluate(
        model, validation_loader, device=args.device, thresholds=thresholds
    )
    (args.output / "baseline_validation.json").write_text(
        json.dumps(baseline, indent=2), encoding="utf-8"
    )
    print(
        f"baseline iou={baseline['positive_mean_iou']:.4f} "
        f"negative={baseline['negative_false_area_fraction']:.4f} "
        f"overlap={baseline['sibling_overlap_root_fraction']:.4f} "
        f"threshold={baseline['best_threshold']:.2f}",
        flush=True,
    )

    history: list[dict[str, object]] = []
    best_score = -float("inf")
    model_dir = args.output / "model"
    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        model.clip.eval()
        training_rows: list[dict[str, float]] = []
        for batch in train_loader:
            inputs = _move(batch.model_inputs, args.device)
            target = batch.targets.to(args.device, non_blocking=True)
            root = batch.root_masks.to(args.device, non_blocking=True)
            positive = batch.positive.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.device == "cuda"):
                logits = model(**inputs).logits
                loss, components = grouped_conditional_part_loss(
                    logits, target, root, positive
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            training_rows.append(components)
        scheduler.step()
        validation = _evaluate(
            model,
            validation_loader,
            device=args.device,
            thresholds=thresholds,
        )
        row: dict[str, object] = {
            "epoch": epoch + 1,
            "train_loss": float(
                np.mean([value["loss"] for value in training_rows])
            ),
            "train_competition": float(
                np.mean([value["competition"] for value in training_rows])
            ),
            "train_overlap": float(
                np.mean([value["overlap"] for value in training_rows])
            ),
            "validation_positive_mean_iou": validation["positive_mean_iou"],
            "validation_negative_false_area_fraction": validation[
                "negative_false_area_fraction"
            ],
            "validation_sibling_overlap_root_fraction": validation[
                "sibling_overlap_root_fraction"
            ],
            "validation_selection_score": validation["selection_score"],
            "validation_best_threshold": validation["best_threshold"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"epoch={epoch + 1}/{args.epochs} loss={row['train_loss']:.5f} "
            f"iou={row['validation_positive_mean_iou']:.4f} "
            f"negative={row['validation_negative_false_area_fraction']:.4f} "
            f"overlap={row['validation_sibling_overlap_root_fraction']:.4f} "
            f"threshold={row['validation_best_threshold']:.2f}",
            flush=True,
        )
        score = float(validation["selection_score"])
        if score > best_score:
            best_score = score
            model.config.hpid_conditional_parts = {
                "format_version": "0.2.0",
                "training_stage": "grouped sibling competition",
                "initial_model": args.model,
                "frozen_clip_encoders": True,
                "trained_component": "CLIPSeg decoder",
                "calibrated_threshold": validation["best_threshold"],
                "validation_selection_score": score,
                "training_manifest_sha256": _sha256(args.manifest),
                "seed": args.seed,
            }
            model.save_pretrained(model_dir, safe_serialization=True)
            processor.save_pretrained(model_dir)
            (args.output / "best_validation.json").write_text(
                json.dumps(validation, indent=2), encoding="utf-8"
            )
        _write_csv(args.output / "training_history.csv", history)

    summary = {
        "format": "HPID grouped conditional part training",
        "format_version": "0.1.0",
        "initial_model": args.model,
        "frozen_clip_encoders": True,
        "trained_component": "CLIPSeg decoder",
        "best_epoch": max(
            history, key=lambda row: float(row["validation_selection_score"])
        )["epoch"],
        "best_selection_score": best_score,
        "baseline_selection_score": baseline["selection_score"],
        "model_directory": str(model_dir.resolve()),
        "model_weights_sha256": _sha256(model_dir / "model.safetensors"),
        "ground_truth_used_for_training": True,
        "ground_truth_available_at_inference": False,
    }
    (args.output / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
