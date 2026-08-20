from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from hpid_split.conditional_parts import (
    ConditionalPartCollator,
    ConditionalPartDataset,
    balanced_sample_weights,
    conditional_part_loss,
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
    with torch.inference_mode():
        for batch in loader:
            model_inputs = _move(batch.model_inputs, device)
            target = batch.targets.to(device, non_blocking=True)
            root = batch.root_masks.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
                logits = model(**model_inputs).logits
                _, components = conditional_part_loss(logits, target, root)
            losses.append(components)
            probabilities.append((logits.sigmoid() * root).cpu())
            targets.append(target.cpu())
            roots.append(root.cpu())
            positives.append(batch.positive.cpu())
    metrics = threshold_metrics(
        torch.cat(probabilities),
        torch.cat(targets),
        torch.cat(roots),
        torch.cat(positives),
        thresholds,
    )
    best_threshold, best = max(
        metrics.items(), key=lambda item: item[1]["selection_score"]
    )
    return {
        "loss": float(np.mean([row["loss"] for row in losses])),
        "bce": float(np.mean([row["bce"] for row in losses])),
        "dice": float(np.mean([row["dice"] for row in losses])),
        "best_threshold": best_threshold,
        **best,
        "thresholds": {str(key): value for key, value in metrics.items()},
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a text-conditioned CLIPSeg decoder on leakage-safe PACO "
            "part masks. CLIP image/text encoders remain frozen."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model", default="CIDAS/clipseg-rd64-refined"
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--negative-ratio", type=float, default=0.25)
    parser.add_argument("--validation-fraction", type=float, default=0.17)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 and not args.inspect_only:
        parser.error("--epochs must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
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
    train_assets = sorted({sample.asset_id for sample in train_samples})
    validation_assets = sorted({sample.asset_id for sample in validation_samples})
    if set(train_assets) & set(validation_assets):
        raise RuntimeError("asset leakage detected between train and validation")
    split = {
        "format": "HPID conditional part split",
        "format_version": "0.1.0",
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": _sha256(args.manifest),
        "seed": args.seed,
        "negative_ratio": args.negative_ratio,
        "validation_fraction": args.validation_fraction,
        "sample_count": len(samples),
        "positive_sample_count": sum(sample.positive for sample in samples),
        "negative_sample_count": sum(not sample.positive for sample in samples),
        "semantic_count": len({sample.semantic_name for sample in samples}),
        "asset_count": len({sample.asset_id for sample in samples}),
        "train_asset_count": len(train_assets),
        "validation_asset_count": len(validation_assets),
        "train_sample_count": len(train_samples),
        "validation_sample_count": len(validation_samples),
        "train_assets": train_assets,
        "validation_assets": validation_assets,
        "train_validation_asset_overlap": [],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "split_manifest.json").write_text(
        json.dumps(split, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in split.items() if not key.endswith("assets")}, indent=2), flush=True)
    if args.inspect_only:
        return 0

    try:
        from transformers import AutoProcessor, CLIPSegForImageSegmentation
    except ImportError as error:
        raise RuntimeError("Install the foundation extra before training") from error
    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    model = CLIPSegForImageSegmentation.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    ).to(args.device)
    for parameter in model.clip.parameters():
        parameter.requires_grad_(False)
    trainable = [parameter for parameter in model.decoder.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.device == "cuda")

    train_dataset = ConditionalPartDataset(
        train_samples,
        training=True,
        seed=args.seed,
    )
    validation_dataset = ConditionalPartDataset(
        validation_samples,
        training=False,
        seed=args.seed,
        root_suppression_probability=0.0,
    )
    collator = ConditionalPartCollator(processor)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        balanced_sample_weights(train_samples),
        num_samples=len(train_samples),
        replacement=True,
        generator=generator,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        collate_fn=collator,
        num_workers=0,
        pin_memory=args.device == "cuda",
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=args.device == "cuda",
    )
    thresholds = tuple(float(value) for value in np.arange(0.20, 0.81, 0.05))
    baseline_validation = _evaluate(
        model,
        validation_loader,
        device=args.device,
        thresholds=thresholds,
    )
    (args.output / "baseline_validation.json").write_text(
        json.dumps(baseline_validation, indent=2), encoding="utf-8"
    )
    print(
        "baseline "
        f"val_iou={baseline_validation['positive_mean_iou']:.4f} "
        f"val_negative={baseline_validation['negative_false_area_fraction']:.4f} "
        f"threshold={baseline_validation['best_threshold']:.2f}",
        flush=True,
    )
    history: list[dict[str, object]] = []
    best_score = -float("inf")
    model_dir = args.output / "model"
    for epoch in range(args.epochs):
        train_dataset.set_epoch(epoch)
        model.train()
        model.clip.eval()
        epoch_rows: list[dict[str, float]] = []
        for batch in train_loader:
            model_inputs = _move(batch.model_inputs, args.device)
            target = batch.targets.to(args.device, non_blocking=True)
            root = batch.root_masks.to(args.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.device == "cuda"):
                logits = model(**model_inputs).logits
                loss, components = conditional_part_loss(logits, target, root)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            epoch_rows.append(components)
        scheduler.step()
        validation = _evaluate(
            model,
            validation_loader,
            device=args.device,
            thresholds=thresholds,
        )
        row: dict[str, object] = {
            "epoch": epoch + 1,
            "train_loss": float(np.mean([value["loss"] for value in epoch_rows])),
            "train_bce": float(np.mean([value["bce"] for value in epoch_rows])),
            "train_dice": float(np.mean([value["dice"] for value in epoch_rows])),
            "validation_loss": validation["loss"],
            "validation_positive_mean_iou": validation["positive_mean_iou"],
            "validation_negative_false_area_fraction": validation[
                "negative_false_area_fraction"
            ],
            "validation_selection_score": validation["selection_score"],
            "validation_best_threshold": validation["best_threshold"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"train_loss={row['train_loss']:.5f} "
            f"val_iou={row['validation_positive_mean_iou']:.4f} "
            f"val_negative={row['validation_negative_false_area_fraction']:.4f} "
            f"threshold={row['validation_best_threshold']:.2f}",
            flush=True,
        )
        score = float(validation["selection_score"])
        if score > best_score:
            best_score = score
            model.config.hpid_conditional_parts = {
                "format_version": "0.1.0",
                "source_model": args.model,
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
        "format": "HPID conditional part training",
        "format_version": "0.1.0",
        "model": args.model,
        "frozen_clip_encoders": True,
        "trained_component": "CLIPSeg decoder",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "best_selection_score": best_score,
        "baseline_selection_score": baseline_validation["selection_score"],
        "baseline_positive_mean_iou": baseline_validation["positive_mean_iou"],
        "baseline_negative_false_area_fraction": baseline_validation[
            "negative_false_area_fraction"
        ],
        "best_epoch": max(
            history,
            key=lambda row: float(row["validation_selection_score"]),
        )["epoch"],
        "model_directory": str(model_dir.resolve()),
        "model_config_sha256": _sha256(model_dir / "config.json"),
        "model_weights_sha256": _sha256(model_dir / "model.safetensors"),
        "train_semantic_distribution": dict(
            sorted(Counter(sample.semantic_name for sample in train_samples).items())
        ),
        "ground_truth_used_for_training": True,
        "ground_truth_available_at_inference": False,
    }
    (args.output / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
