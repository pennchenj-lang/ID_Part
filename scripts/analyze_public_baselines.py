from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from hpid_split.paco_semantics import canonical_part_token, normalize_paco_name
from hpid_split.paper_eval import DEFAULT_IOU_THRESHOLDS, evaluate_part_predictions

SEED = 20260822
METHODS = (
    "sam2_raw",
    "sam2_nms",
    "sam2_max_ownership",
    "grounded_sam2_same_inventory",
    "hpid_split_a3",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) >= 128


def _load_sam_cache(path: Path) -> tuple[list[np.ndarray], list[float]]:
    with np.load(path) as payload:
        width = int(payload["width"][0])
        unpacked = np.unpackbits(payload["packed_masks"], axis=2)[:, :, :width]
        scores = payload["scores"].astype(np.float32).tolist()
    return [row.astype(bool) for row in unpacked], [float(row) for row in scores]


def _overlap(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    intersection = int(np.count_nonzero(first & second))
    if not intersection:
        return 0.0, 0.0
    first_area = int(np.count_nonzero(first))
    second_area = int(np.count_nonzero(second))
    return (
        intersection / max(1, first_area + second_area - intersection),
        intersection / max(1, min(first_area, second_area)),
    )


def _nms(
    masks: list[np.ndarray],
    scores: list[float],
    semantics: list[str | None],
    *,
    iou_threshold: float,
    containment_threshold: float,
    semantic_aware: bool,
) -> tuple[list[np.ndarray], list[float], list[str | None]]:
    order = sorted(range(len(masks)), key=lambda index: scores[index], reverse=True)
    kept: list[int] = []
    for index in order:
        reject = False
        for previous in kept:
            if semantic_aware and semantics[index] != semantics[previous]:
                continue
            iou, containment = _overlap(masks[index], masks[previous])
            if iou >= iou_threshold or containment >= containment_threshold:
                reject = True
                break
        if not reject:
            kept.append(index)
    return (
        [masks[index] for index in kept],
        [scores[index] for index in kept],
        [semantics[index] for index in kept],
    )


def _exclusive_ownership(
    masks: list[np.ndarray],
    scores: list[float],
    semantics: list[str | None],
    *,
    minimum_area: int = 6,
) -> tuple[list[np.ndarray], list[str | None]]:
    if not masks:
        return [], []
    stack = np.stack(masks, axis=0)
    weighted = np.where(
        stack,
        np.asarray(scores, dtype=np.float32)[:, None, None],
        -np.inf,
    )
    winner = np.argmax(weighted, axis=0)
    foreground = stack.any(axis=0)
    output_masks: list[np.ndarray] = []
    output_semantics: list[str | None] = []
    for index, semantic in enumerate(semantics):
        owned = foreground & (winner == index)
        if int(np.count_nonzero(owned)) < minimum_area:
            continue
        output_masks.append(owned)
        output_semantics.append(semantic)
    return output_masks, output_semantics


def _truth(
    case_path: Path,
    expected_domain: str,
) -> tuple[dict[str, object], list[np.ndarray], list[str], np.ndarray]:
    case = json.loads(case_path.read_text(encoding="utf-8"))
    category = str(case["object_category"])
    rows = list(case["parts"])
    masks = [_load_mask(case_path.parent / str(row["mask_crop"])) for row in rows]
    semantics = [
        canonical_part_token(
            str(row["part_name"]),
            expected_domain,
            object_category=category,
        )
        for row in rows
    ]
    return case, masks, semantics, _load_mask(case_path.parent / "object_mask_crop.png")


def _predicted_token(
    value: str,
    expected_domain: str,
    *,
    object_category: str,
) -> str:
    normalized = normalize_paco_name(value).removeprefix(f"{expected_domain}_")
    return canonical_part_token(
        normalized,
        expected_domain,
        object_category=object_category,
    )


def _grounded_predictions(
    package_dir: Path,
    expected_domain: str,
    object_category: str,
    *,
    nms_iou: float,
    nms_containment: float,
) -> tuple[list[np.ndarray], list[str | None]]:
    rows = json.loads((package_dir / "candidates.json").read_text(encoding="utf-8"))
    selected = [
        row
        for row in rows
        if str(row.get("source", "")).startswith("grounded-sam2[")
        and str(row.get("semantic_name", "")) != expected_domain
    ]
    masks = [_load_mask(package_dir / str(row["mask_path"])) for row in selected]
    scores = [
        float(row.get("score", 0.0)) * float(row.get("source_reliability", 1.0))
        for row in selected
    ]
    semantics: list[str | None] = [
        _predicted_token(
            str(row["semantic_name"]),
            expected_domain,
            object_category=object_category,
        )
        for row in selected
    ]
    masks, scores, semantics = _nms(
        masks,
        scores,
        semantics,
        iou_threshold=nms_iou,
        containment_threshold=nms_containment,
        semantic_aware=True,
    )
    return _exclusive_ownership(masks, scores, semantics)


def _hpid_predictions(
    package_dir: Path,
    expected_domain: str,
    object_category: str,
    truth_semantics: list[str],
) -> tuple[list[np.ndarray], list[str | None]]:
    rows = json.loads((package_dir / "parts.json").read_text(encoding="utf-8"))
    include_root = "body" in truth_semantics
    selected = [
        row
        for row in rows
        if str(row.get("semantic_name", "")) != expected_domain or include_root
    ]
    masks = [_load_mask(package_dir / str(row["mask_visible_path"])) for row in selected]
    semantics = [
        (
            "body"
            if str(row["semantic_name"]) == expected_domain
            else _predicted_token(
                str(row["semantic_name"]),
                expected_domain,
                object_category=object_category,
            )
        )
        for row in selected
    ]
    return masks, semantics


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap(values: list[float], seed: int, iterations: int) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    samples = generator.integers(0, len(array), size=(iterations, len(array)))
    means = array[samples].mean(axis=1)
    return (
        float(array.mean()),
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate external proposal baselines and HPID-Split uniformly."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--sam-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nms-iou", type=float, default=0.80)
    parser.add_argument("--nms-containment", type=float, default=0.92)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    benchmark = json.loads(
        (args.benchmark_root / "benchmark_summary.json").read_text(encoding="utf-8")
    )
    benchmark_rows = {
        str(row["case_id"]): row
        for row in benchmark.get("cases", [])
        if int(row.get("return_code", 1)) == 0
    }
    cases = [row for row in manifest.get("cases", []) if row.get("case_path")]
    rows: list[dict[str, object]] = []
    for case_index, manifest_case in enumerate(cases, start=1):
        case_id = str(manifest_case["case_id"])
        if case_id not in benchmark_rows:
            continue
        expected_domain = str(manifest_case["expected_domain"])
        case_path = Path(str(manifest_case["case_path"]))
        case, truth_masks, truth_semantics, truth_object = _truth(
            case_path, expected_domain
        )
        object_category = str(case["object_category"])
        package_dir = args.benchmark_root / case_id
        raw_masks, raw_scores = _load_sam_cache(
            args.sam_cache / "cases" / f"{case_id}.npz"
        )
        raw_semantics: list[str | None] = [None] * len(raw_masks)
        nms_masks, nms_scores, nms_semantics = _nms(
            raw_masks,
            raw_scores,
            raw_semantics,
            iou_threshold=args.nms_iou,
            containment_threshold=args.nms_containment,
            semantic_aware=False,
        )
        owned_masks, owned_semantics = _exclusive_ownership(
            nms_masks, nms_scores, nms_semantics
        )
        grounded_masks, grounded_semantics = _grounded_predictions(
            package_dir,
            expected_domain,
            object_category,
            nms_iou=args.nms_iou,
            nms_containment=args.nms_containment,
        )
        hpid_masks, hpid_semantics = _hpid_predictions(
            package_dir,
            expected_domain,
            object_category,
            truth_semantics,
        )
        predictions = {
            "sam2_raw": (raw_masks, raw_semantics),
            "sam2_nms": (nms_masks, nms_semantics),
            "sam2_max_ownership": (owned_masks, owned_semantics),
            "grounded_sam2_same_inventory": (
                grounded_masks,
                grounded_semantics,
            ),
            "hpid_split_a3": (hpid_masks, hpid_semantics),
        }
        for method, (prediction_masks, prediction_semantics) in predictions.items():
            metrics = evaluate_part_predictions(
                truth_masks=truth_masks,
                truth_semantics=truth_semantics,
                prediction_masks=prediction_masks,
                prediction_semantics=prediction_semantics,
                truth_object_mask=truth_object,
                thresholds=DEFAULT_IOU_THRESHOLDS,
            )
            rows.append(
                {
                    "case_id": case_id,
                    "object_category": case["object_category"],
                    "expected_domain": expected_domain,
                    "method": method,
                    "ground_truth_used_in_prediction": False,
                    **metrics,
                }
            )
        print(f"[{case_index}/{len(cases)}] {case_id}", flush=True)

    metric_names = sorted(
        key
        for key in rows[0]
        if key
        not in {
            "case_id",
            "object_category",
            "expected_domain",
            "method",
            "ground_truth_used_in_prediction",
        }
    )
    summary_rows: list[dict[str, object]] = []
    by_domain_rows: list[dict[str, object]] = []
    for method_index, method in enumerate(METHODS):
        selected = [row for row in rows if row["method"] == method]
        summary: dict[str, object] = {"method": method, "case_count": len(selected)}
        for metric_index, metric in enumerate(metric_names):
            mean, low, high = _bootstrap(
                [float(row[metric]) for row in selected],
                SEED + method_index * 1000 + metric_index,
                args.bootstrap_iterations,
            )
            summary[f"mean_{metric}"] = mean
            summary[f"ci95_low_{metric}"] = low
            summary[f"ci95_high_{metric}"] = high
        summary_rows.append(summary)
        by_domain: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in selected:
            by_domain[str(row["expected_domain"])].append(row)
        for domain, domain_rows in sorted(by_domain.items()):
            output: dict[str, object] = {
                "method": method,
                "expected_domain": domain,
                "case_count": len(domain_rows),
            }
            output.update(
                {
                    f"mean_{metric}": float(
                        np.mean([float(row[metric]) for row in domain_rows])
                    )
                    for metric in metric_names
                }
            )
            by_domain_rows.append(output)

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "baseline_cases.csv", rows)
    _write_csv(args.output / "baseline_summary.csv", summary_rows)
    _write_csv(args.output / "baseline_by_domain.csv", by_domain_rows)
    report = {
        "format": "HPID-Split public object-conditioned baseline evaluation",
        "format_version": "1.0.0",
        "case_count": len({str(row["case_id"]) for row in rows}),
        "methods": list(METHODS),
        "iou_thresholds": list(DEFAULT_IOU_THRESHOLDS),
        "nms_iou": args.nms_iou,
        "nms_containment": args.nms_containment,
        "manifest_sha256": _sha256(args.manifest),
        "benchmark_summary_sha256": _sha256(
            args.benchmark_root / "benchmark_summary.json"
        ),
        "sam_cache_manifest_sha256": _sha256(args.sam_cache / "manifest.json"),
        "scope": "object-conditioned PACO-LVIS ground-truth bounding-box crops",
        "ground_truth_usage": (
            "Predictions are loaded or generated before PACO part masks are read. "
            "The crop itself is derived from the ground-truth object box."
        ),
        "grounded_baseline_note": (
            "Grounded-SAM2 uses the same category-conditioned semantic inventory "
            "and exported pre-fusion grounded candidates, followed only by NMS "
            "and score-max ownership."
        ),
        "seed": SEED,
    }
    (args.output / "baseline_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
