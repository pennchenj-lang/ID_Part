from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from hpid_split.benchmarking import evaluate_amodal_case, make_edge_occlusion
from hpid_split.fusion import FusionConfig, MaskCandidate, fuse_candidates
from hpid_split.instances import PartInstance
from hpid_split.restoration import CompletionRequest, load_completion_backend


def _load_candidates(path: Path) -> list[MaskCandidate]:
    payload = np.load(path, allow_pickle=False)
    metadata = json.loads(str(payload["metadata"]))
    return [
        MaskCandidate(mask=mask.astype(bool), **item)
        for mask, item in zip(payload["masks"], metadata, strict=True)
    ]


def _geometry(
    mask: np.ndarray,
) -> tuple[tuple[int, int, int, int], tuple[float, float], int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("cannot describe an empty benchmark mask")
    return (
        (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)),
        (float(xs.mean()), float(ys.mean())),
        len(xs),
    )


def _target_priority(record: PartInstance, image_area: int) -> tuple[float, float]:
    name = record.semantic_name
    preferred = (
        ("upper_clothing", 8.0),
        ("lower_clothing", 7.5),
        ("hair", 7.0),
        ("torso", 6.5),
        ("sleeve", 6.0),
        ("shoe", 5.5),
        ("arm", 5.0),
        ("leg", 5.0),
        ("head", 4.5),
    )
    semantic_score = next((score for token, score in preferred if token in name), 1.0)
    area_ratio = record.area_px / max(1, image_area)
    useful_scale = 1.0 - min(1.0, abs(area_ratio - 0.055) / 0.055)
    return semantic_score + useful_scale, area_ratio


def _select_target(
    records: tuple[PartInstance, ...],
    image_shape: tuple[int, int],
    selection_offset: int,
) -> PartInstance:
    image_area = image_shape[0] * image_shape[1]
    eligible: list[PartInstance] = []
    for record in records:
        if record.semantic_name == record.semantic_parent:
            continue
        x0, y0, x1, y1 = record.bbox_xyxy
        if record.area_px < max(900, round(image_area * 0.0015)):
            continue
        if min(x1 - x0, y1 - y0) < 24:
            continue
        if any(
            token in record.semantic_name
            for token in ("eye", "lace", "cuff", "hem", "sole", "heel")
        ):
            continue
        eligible.append(record)
    if not eligible:
        raise RuntimeError("no sufficiently large non-root Part-ID for benchmark")
    ordered = sorted(
        eligible,
        key=lambda item: _target_priority(item, image_area),
        reverse=True,
    )
    semantic_pool: list[PartInstance] = []
    seen_semantics: set[str] = set()
    for record in ordered:
        if record.semantic_name in seen_semantics:
            continue
        seen_semantics.add(record.semantic_name)
        semantic_pool.append(record)
        if len(semantic_pool) >= 5:
            break
    return semantic_pool[selection_offset % len(semantic_pool)]


def _synthetic_input(
    image: Image.Image, occluder: np.ndarray, case_key: str
) -> Image.Image:
    rgba = np.asarray(image.convert("RGBA")).copy()
    digest = hashlib.sha256(case_key.encode("utf-8")).digest()
    first = np.array([35 + digest[0] % 80, 40 + digest[1] % 70, 45, 255])
    second = np.array([205, 210 - digest[2] % 70, 60 + digest[3] % 90, 255])
    yy, xx = np.indices(occluder.shape)
    checker = ((xx // 18 + yy // 18) % 2) == 0
    rgba[occluder & checker] = first
    rgba[occluder & ~checker] = second
    return Image.fromarray(rgba, mode="RGBA").convert("RGB")


def _benchmark_records(
    record: PartInstance,
    visible: np.ndarray,
    occluder: np.ndarray,
) -> tuple[np.ndarray, PartInstance, tuple[PartInstance, ...]]:
    target_bbox, target_centroid, target_area = _geometry(visible)
    occluder_bbox, occluder_centroid, occluder_area = _geometry(occluder)
    target = PartInstance(
        part_id=record.part_id,
        semantic_name=record.semantic_name,
        semantic_parent=record.semantic_parent,
        instance_index=1,
        side=record.side,
        bbox_xyxy=target_bbox,
        centroid_xy=target_centroid,
        area_px=target_area,
        assembly_parent_id=None,
    )
    blocker = PartInstance(
        part_id="benchmark/synthetic_occluder/center/01",
        semantic_name="synthetic_occluder",
        semantic_parent="synthetic_occluder",
        instance_index=2,
        side="center",
        bbox_xyxy=occluder_bbox,
        centroid_xy=occluder_centroid,
        area_px=occluder_area,
        assembly_parent_id=None,
    )
    instance_map = np.zeros(visible.shape, dtype=np.uint16)
    instance_map[visible] = 1
    instance_map[occluder] = 2
    return instance_map, target, (target, blocker)


def _crop_box(mask: np.ndarray, padding: int = 24) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0, 0, mask.shape[1], mask.shape[0]
    return (
        max(0, int(xs.min()) - padding),
        max(0, int(ys.min()) - padding),
        min(mask.shape[1], int(xs.max() + 1) + padding),
        min(mask.shape[0], int(ys.max() + 1) + padding),
    )


def _save_case_audit(
    output: Path,
    source: Image.Image,
    synthetic: Image.Image,
    full: np.ndarray,
    visible: np.ndarray,
    occluder: np.ndarray,
    completed: np.ndarray,
    metadata: dict[str, object],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    x0, y0, x1, y1 = _crop_box(full | occluder | completed)
    source.crop((x0, y0, x1, y1)).save(output / "source_crop.png")
    synthetic.crop((x0, y0, x1, y1)).save(output / "synthetic_input_crop.png")
    masks = {
        "pseudo_full_mask.png": full,
        "visible_after_occlusion.png": visible,
        "synthetic_occluder.png": occluder,
        "completed_mask.png": completed,
    }
    for name, mask in masks.items():
        Image.fromarray(mask[y0:y1, x0:x1].astype(np.uint8) * 255).save(output / name)
    overlay = np.zeros((*full.shape, 4), dtype=np.uint8)
    overlay[completed & full] = (40, 190, 95, 220)
    overlay[full & ~completed] = (50, 110, 230, 240)
    overlay[completed & ~full] = (225, 55, 55, 240)
    Image.fromarray(overlay[y0:y1, x0:x1], mode="RGBA").save(
        output / "error_overlay_tp_green_fn_blue_fp_red.png"
    )
    (output / "case.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _mean(rows: list[dict[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def run(args: argparse.Namespace) -> int:
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_roles = (
        [value.strip() for value in args.roles.split(",") if value.strip()]
        if args.roles
        else sorted(path.stem for path in args.candidate_cache_dir.glob("*.npz"))
    )
    roles = [value.zfill(4) if value.isdigit() else value for value in raw_roles]
    roles = roles[: args.max_cases]
    if not roles:
        raise ValueError("no benchmark roles were selected")
    for role in roles:
        image_path = args.image_dir / f"{role}.png"
        cache_path = args.candidate_cache_dir / f"{role}.npz"
        if not image_path.is_file() or not cache_path.is_file():
            raise FileNotFoundError(f"missing image or candidate cache for {role}")

    args.output.mkdir(parents=True, exist_ok=True)
    backend = load_completion_backend(args.backend_config, device=device)
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for case_index, role in enumerate(roles):
        image_path = args.image_dir / f"{role}.png"
        cache_path = args.candidate_cache_dir / f"{role}.npz"
        source = Image.open(image_path).convert("RGB")
        candidates = _load_candidates(cache_path)
        fused = fuse_candidates(
            candidates,
            image_shape=(source.height, source.width),
            config=FusionConfig(
                use_parent_envelope=True,
                use_transitive_residual=True,
            ),
        )
        selected = _select_target(
            fused.instances,
            (source.height, source.width),
            case_index,
        )
        full = fused.instance_map == selected.instance_index
        synthetic_case = make_edge_occlusion(
            full,
            target_hidden_fraction=args.hidden_fraction,
            direction_offset=case_index,
        )
        synthetic = _synthetic_input(
            source, synthetic_case.occluder_mask, f"{role}:{selected.part_id}"
        )
        instance_map, target, records = _benchmark_records(
            selected,
            synthetic_case.visible_mask,
            synthetic_case.occluder_mask,
        )
        case_started = time.perf_counter()
        result = backend.complete(
            CompletionRequest(synthetic, instance_map, target, records)
        )
        elapsed = time.perf_counter() - case_started
        completed = result.full_mask.astype(bool) | synthetic_case.visible_mask
        metrics = evaluate_amodal_case(
            synthetic_case.full_mask,
            synthetic_case.visible_mask,
            completed,
        )
        status = str(result.metadata.get("status", "unknown"))
        row: dict[str, object] = {
            "role": role,
            "part_id": selected.part_id,
            "semantic_name": selected.semantic_name,
            "direction": synthetic_case.direction,
            "hidden_fraction": synthetic_case.hidden_fraction,
            **asdict(metrics),
            "iou_gain": metrics.completed_iou - metrics.visible_only_iou,
            "status": status,
            "completion_confidence": result.confidence,
            "elapsed_seconds": elapsed,
            "ground_truth_used_in_inference": False,
        }
        rows.append(row)
        case_metadata = {
            **row,
            "evaluation_target": (
                "synthetically occluded HPID visible mask; not natural amodal truth"
            ),
            "backend": asdict(result.provenance),
            "completion_metadata": result.metadata,
        }
        _save_case_audit(
            args.output / "cases" / f"{case_index + 1:02d}_{role}",
            source,
            synthetic,
            synthetic_case.full_mask,
            synthetic_case.visible_mask,
            synthetic_case.occluder_mask,
            completed,
            case_metadata,
        )
        print(
            f"{role} {selected.semantic_name} status={status} "
            f"IoU {metrics.visible_only_iou:.4f}->{metrics.completed_iou:.4f}"
        )

    fieldnames = list(rows[0])
    with (args.output / "case_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_count": len(rows),
        "backend": asdict(backend.provenance),
        "mean_visible_only_iou": _mean(rows, "visible_only_iou"),
        "mean_completed_iou": _mean(rows, "completed_iou"),
        "mean_iou_gain": _mean(rows, "iou_gain"),
        "mean_hidden_recall": _mean(rows, "hidden_recall"),
        "mean_added_precision": _mean(rows, "added_precision"),
        "mean_false_added_ratio": _mean(rows, "false_added_ratio"),
        "visible_lock_minimum_recall": min(
            float(row["visible_recall"]) for row in rows
        ),
        "model_supported_case_count": sum(
            row["status"] == "model_supported_completion" for row in rows
        ),
        "evaluation_scope": (
            "Synthetic self-occlusion recovery of already predicted visible "
            "Part-ID masks. This is not natural amodal ground truth, physical "
            "interior recovery, or 3D validation."
        ),
        "candidate_cache_dir": str(args.candidate_cache_dir.resolve()),
        "image_dir": str(args.image_dir.resolve()),
        "backend_config": str(args.backend_config.resolve()),
        "inference_uses_ground_truth": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an inference-independent synthetic amodal sanity check."
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--candidate-cache-dir", type=Path, required=True)
    parser.add_argument("--backend-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--roles", default="")
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--hidden-fraction", type=float, default=0.28)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
