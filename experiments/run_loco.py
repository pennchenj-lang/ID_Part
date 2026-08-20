from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import wilcoxon
from torch import nn
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from hpid_split.data import CharacterPartDataset, class_weights, load_label_map
from hpid_split.inference import predict
from hpid_split.losses import LossWeights, hpid_split_loss
from hpid_split.metrics import evaluate_semantic
from hpid_split.model import HPIDSplitNet
from hpid_split.taxonomy import Taxonomy

SEED = 20260811
METRICS = (
    "foreground_iou",
    "coarse_miou",
    "semantic_miou",
    "boundary_f1",
    "small_part_recall",
    "component_f1",
    "part_count_abs_error",
)

TRAINING_MODES: dict[str, tuple[float, LossWeights]] = {
    "flat": (
        0.0,
        LossWeights(
            fine_ce=1.0,
            parent_ce=0.0,
            detail_tversky=0.0,
            boundary_bce=0.0,
            hierarchy=0.0,
        ),
    ),
    "detail": (
        0.65,
        LossWeights(
            fine_ce=1.0,
            parent_ce=0.0,
            detail_tversky=0.55,
            boundary_bce=0.0,
            hierarchy=0.0,
        ),
    ),
    "full": (0.65, LossWeights()),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_batchnorm_eval(module: nn.Module) -> None:
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()


def parent_weights_from_fine(
    fine_weights: torch.Tensor, taxonomy: Taxonomy
) -> torch.Tensor:
    output = torch.ones(taxonomy.num_parent_classes, dtype=torch.float32)
    for parent_id in range(taxonomy.num_parent_classes):
        children = [
            fine_id
            for fine_id, mapped_parent in enumerate(taxonomy.fine_to_parent)
            if mapped_parent == parent_id
        ]
        output[parent_id] = fine_weights[children].mean()
    output[0] = min(float(output[0]), 0.18)
    return output


def train_fold(
    mode: str,
    held_out: str,
    train_roles: list[str],
    image_dir: Path,
    label_dir: Path,
    taxonomy: Taxonomy,
    output_dir: Path,
    *,
    epochs: int,
    repeats: int,
    batch_size: int,
    device: str,
) -> tuple[HPIDSplitNet, pd.DataFrame]:
    focus_probability, loss_weights = TRAINING_MODES[mode]
    set_seed(SEED + int(held_out) * 101 + sum(ord(value) for value in mode))
    dataset = CharacterPartDataset(
        image_dir,
        label_dir,
        train_roles,
        taxonomy,
        repeats=repeats,
        focus_probability=focus_probability,
        seed=SEED + int(held_out) * 1009,
    )
    loader_generator = torch.Generator().manual_seed(SEED + int(held_out))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device == "cuda",
        generator=loader_generator,
    )
    model = HPIDSplitNet(taxonomy, pretrained_backbone=True).to(device)
    train_data = [
        (dataset.cache[role][0], dataset.cache[role][1]) for role in train_roles
    ]
    fine_weights = class_weights(train_data, taxonomy.num_fine_classes).to(device)
    parent_weights = parent_weights_from_fine(fine_weights.cpu(), taxonomy).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": 1.2e-4},
            {"params": model.heads.parameters(), "lr": 8.0e-4},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    history: list[dict[str, object]] = []
    for epoch in range(epochs):
        dataset.set_epoch(epoch)
        model.train()
        model.apply(set_batchnorm_eval)
        components: list[dict[str, float]] = []
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                outputs = model(images)
                loss, values = hpid_split_loss(
                    outputs,
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
            components.append(values)
        scheduler.step()
        row: dict[str, object] = {
            "mode": mode,
            "held_out_role": held_out,
            "epoch": epoch + 1,
            "focus_probability": focus_probability,
            **{
                key: float(np.mean([item[key] for item in components]))
                for key in components[0]
            },
        }
        history.append(row)
        print(
            json.dumps(
                {
                    "stage": "train",
                    "mode": mode,
                    "held_out": held_out,
                    "epoch": epoch + 1,
                    "epochs": epochs,
                    "loss": round(float(row["loss"]), 5),
                }
            ),
            flush=True,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "taxonomy": taxonomy.to_dict(),
            "mode": mode,
            "held_out_role": held_out,
            "train_roles": train_roles,
            "epochs": epochs,
            "repeats": repeats,
            "seed": SEED,
            "loss_weights": asdict(loss_weights),
        },
        output_dir / "model.pt",
    )
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(output_dir / "training_history.csv", index=False)
    return model, history_frame


def colorize(labels: np.ndarray) -> Image.Image:
    rng = np.random.default_rng(17)
    colors = np.vstack(
        [
            np.zeros((1, 3), np.uint8),
            rng.integers(35, 240, size=(255, 3), dtype=np.uint8),
        ]
    )
    return Image.fromarray(colors[labels])


def evaluate_fold(
    model: HPIDSplitNet,
    mode: str,
    role_id: str,
    image_dir: Path,
    label_dir: Path,
    taxonomy: Taxonomy,
    output_dir: Path,
    device: str,
) -> pd.DataFrame:
    image = Image.open(image_dir / f"{role_id}.png").convert("RGB")
    label_path = label_dir / f"{role_id}_final.png"
    if not label_path.exists():
        label_path = label_dir / f"{role_id}.png"
    truth = load_label_map(label_path, taxonomy)
    if mode == "flat":
        variants = {
            "flat_same_backbone": {
                "recursive": False,
                "use_parent_conditioning": False,
                "use_boundary_refinement": False,
            }
        }
    elif mode == "detail":
        variants = {
            "detail_sampling": {
                "recursive": False,
                "use_parent_conditioning": False,
                "use_boundary_refinement": False,
            }
        }
    else:
        variants = {
            "hpid_parent": {
                "recursive": False,
                "use_parent_conditioning": True,
                "use_boundary_refinement": False,
            },
            "hpid_parent_recursive": {
                "recursive": True,
                "use_parent_conditioning": True,
                "use_boundary_refinement": False,
            },
            "hpid_split_full": {
                "recursive": True,
                "use_parent_conditioning": True,
                "use_boundary_refinement": True,
            },
        }
    rows: list[dict[str, object]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for method, options in variants.items():
        started = time.perf_counter()
        result = predict(
            model,
            image,
            taxonomy,
            device=device,
            evaluation_height=768,
            **options,
        )
        inference_sec = time.perf_counter() - started
        values = evaluate_semantic(
            result.semantic_map,
            truth,
            taxonomy,
            boundary_tolerance=3,
            component_iou_threshold=0.25,
            small_part_fraction=0.01,
        )
        rows.append(
            {
                "role": role_id,
                "method": method,
                "training_mode": mode,
                "inference_sec": inference_sec,
                "inference_uses_ground_truth": False,
                **values,
            }
        )
        colorize(result.semantic_map).save(output_dir / f"{method}.png")
        Image.fromarray(result.instance_map).save(
            output_dir / f"{method}_instances.tiff"
        )
        (output_dir / f"{method}_instances.json").write_text(
            json.dumps([record.to_dict() for record in result.instances], indent=2),
            encoding="utf-8",
        )
    colorize(truth).save(output_dir / "ground_truth.png")
    image.save(output_dir / "source.png")
    return pd.DataFrame(rows)


def cluster_bootstrap(
    values: pd.Series, seed: int, samples: int = 5000
) -> tuple[float, float, float]:
    array = values.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, (0.025, 0.975))
    return float(array.mean()), float(low), float(high)


def summarize(cases: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, object]] = []
    for method, group in cases.groupby("method"):
        for metric in METRICS:
            mean, low, high = cluster_bootstrap(group[metric], SEED)
            summary_rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "role_count": group["role"].nunique(),
                    "mean": mean,
                    "median": float(group[metric].median()),
                    "std": float(group[metric].std(ddof=1)),
                    "ci95_low": low,
                    "ci95_high": high,
                }
            )
    tests: list[dict[str, object]] = []
    pivot_source = cases.pivot(index="role", columns="method", values=list(METRICS))
    baseline = "flat_same_backbone"
    for method in sorted(cases["method"].unique()):
        if method == baseline:
            continue
        for metric in METRICS:
            left = pivot_source[(metric, method)]
            right = pivot_source[(metric, baseline)]
            statistic, p_value = wilcoxon(left, right, zero_method="zsplit")
            difference = left - right
            standardizer = difference.std(ddof=1)
            effect = (
                difference.mean() / standardizer if standardizer > 0 else float("inf")
            )
            tests.append(
                {
                    "comparison": f"{method} vs {baseline}",
                    "metric": metric,
                    "role_count": len(difference),
                    "mean_difference": float(difference.mean()),
                    "median_difference": float(difference.median()),
                    "paired_standardized_effect": float(effect),
                    "wilcoxon_statistic": float(statistic),
                    "p_value_two_sided": float(p_value),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(tests)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strict LOCO benchmark for HPID-Split."
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "character_parts.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--roles", default=",".join(f"{index:04d}" for index in range(1, 11))
    )
    parser.add_argument("--modes", default="flat,detail,full")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    taxonomy = Taxonomy.from_json(args.taxonomy)
    roles = [value.strip() for value in args.roles.split(",") if value.strip()]
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    unknown = set(modes) - set(TRAINING_MODES)
    if unknown:
        raise ValueError(f"Unknown modes: {sorted(unknown)}")
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device == "auto":
        device = "cpu"
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    all_cases: list[pd.DataFrame] = []
    all_history: list[pd.DataFrame] = []
    for held_out in roles[: args.folds]:
        train_roles = [role for role in roles if role != held_out]
        for mode in modes:
            fold_dir = args.output / "folds" / held_out / mode
            model, history = train_fold(
                mode,
                held_out,
                train_roles,
                args.image_dir,
                args.label_dir,
                taxonomy,
                fold_dir,
                epochs=args.epochs,
                repeats=args.repeats,
                batch_size=args.batch_size,
                device=device,
            )
            cases = evaluate_fold(
                model,
                mode,
                held_out,
                args.image_dir,
                args.label_dir,
                taxonomy,
                fold_dir / "predictions",
                device,
            )
            all_cases.append(cases)
            all_history.append(history)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
    case_frame = pd.concat(all_cases, ignore_index=True)
    history_frame = pd.concat(all_history, ignore_index=True)
    case_frame.to_csv(args.output / "case_metrics.csv", index=False)
    history_frame.to_csv(args.output / "training_history.csv", index=False)
    if args.folds == len(roles) and set(modes) == set(TRAINING_MODES):
        summary, tests = summarize(case_frame)
        summary.to_csv(args.output / "summary.csv", index=False)
        tests.to_csv(args.output / "paired_tests.csv", index=False)
    manifest = {
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "HPID-Split 0.1.0",
        "roles": roles[: args.folds],
        "modes": modes,
        "folds": args.folds,
        "epochs": args.epochs,
        "repeats": args.repeats,
        "batch_size": args.batch_size,
        "device": device,
        "seed": SEED,
        "small_part_fraction": 0.01,
        "component_iou_threshold": 0.25,
        "boundary_tolerance": 3,
        "inference_uses_ground_truth": False,
        "truth_usage": "training labels and post-inference scoring only",
        "elapsed_sec": time.perf_counter() - started,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
