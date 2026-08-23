from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from hpid_split.fusion import FusionConfig, MaskCandidate, fuse_candidates
from hpid_split.metrics import binary_iou
from hpid_split.paco_eval import (
    _normalize,
)
from hpid_split.paper_eval import (
    DEFAULT_IOU_THRESHOLDS,
    evaluate_part_predictions,
)

SEED = 20260821
METRICS = (
    "root_foreground_iou",
    "root_foreground_precision",
    "root_foreground_recall",
    "editable_part_coverage_iou",
    "editable_part_coverage_precision",
    "editable_part_coverage_recall",
    "part_precision_at_025",
    "part_recall_at_025",
    "part_f1_at_025",
    "part_precision_at_050",
    "part_recall_at_050",
    "part_f1_at_050",
    "part_precision_at_075",
    "part_recall_at_075",
    "part_f1_at_075",
    "part_f1_mean_025_075",
    "mean_matched_iou_at_025",
    "mean_matched_iou_at_050",
    "mean_matched_iou_at_075",
    "mean_matched_boundary_f1_at_025",
    "mean_matched_boundary_f1_at_050",
    "mean_matched_boundary_f1_at_075",
    "semantic_precision_at_025",
    "semantic_recall_at_025",
    "semantic_f1_at_025",
    "semantic_precision_at_050",
    "semantic_recall_at_050",
    "semantic_f1_at_050",
    "semantic_precision_at_075",
    "semantic_recall_at_075",
    "semantic_f1_at_075",
    "semantic_f1_mean_025_075",
    "oversegmentation_ratio",
    "predicted_part_count",
    "fusion_seconds",
)


def _variants() -> dict[str, FusionConfig]:
    unconstrained = {
        "use_parent_support": False,
        "use_parent_residual": False,
        "use_root_coverage_conservation": False,
        "use_direct_gate": False,
        "use_specificity_ownership": False,
        "use_hierarchical_duplicate_suppression": False,
        "use_remainder_attachment": False,
        "detail_bonus": 0.0,
    }
    return {
        "A0_independent_max": FusionConfig(
            use_consensus=False,
            **unconstrained,
        ),
        "A1_cross_source_consensus": FusionConfig(
            use_consensus=True,
            **unconstrained,
        ),
        "A2_consensus_hierarchy": FusionConfig(
            use_specificity_ownership=False,
        ),
        "A3_full_fusion": FusionConfig(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) >= 128


def _load_candidates(package_dir: Path) -> list[MaskCandidate]:
    rows = json.loads(
        (package_dir / "candidates.json").read_text(encoding="utf-8")
    )
    return [
        MaskCandidate(
            semantic_name=str(row["semantic_name"]),
            semantic_parent=str(row["semantic_parent"]),
            mask=_load_mask(package_dir / str(row["mask_path"])),
            score=float(row["score"]),
            source=str(row["source"]),
            prompt=str(row.get("prompt", "")),
            source_reliability=float(row.get("source_reliability", 1.0)),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in rows
    ]


def _coverage_metrics(
    masks: list[np.ndarray],
    truth_object: np.ndarray,
) -> tuple[float, float, float]:
    if masks:
        prediction_union = np.logical_or.reduce(masks)
    else:
        prediction_union = np.zeros(truth_object.shape, dtype=bool)
    inside = int(np.count_nonzero(prediction_union & truth_object))
    predicted = int(np.count_nonzero(prediction_union))
    truth = int(np.count_nonzero(truth_object))
    return (
        binary_iou(prediction_union, truth_object),
        inside / max(1, predicted),
        inside / max(1, truth),
    )


def _evaluate_result(
    *,
    result,
    case: dict[str, object],
    case_dir: Path,
    expected_domain: str,
) -> dict[str, float]:
    object_category = str(case["object_category"])
    truth_rows = list(case["parts"])
    truth_masks = [
        _load_mask(case_dir / str(row["mask_crop"])) for row in truth_rows
    ]
    truth_object = _load_mask(case_dir / "object_mask_crop.png")

    all_prediction_rows = [record.to_dict() for record in result.instances]
    all_prediction_masks = [
        result.instance_map == record.instance_index for record in result.instances
    ]
    truth_names = {
        _normalize(
            str(row["part_name"]),
            expected_domain,
            object_category=object_category,
        )
        for row in truth_rows
    }
    include_root_body = "body" in truth_names
    selected = [
        (row, mask)
        for row, mask in zip(
            all_prediction_rows, all_prediction_masks, strict=True
        )
        if str(row["semantic_name"]) != expected_domain or include_root_body
    ]
    prediction_rows = [row for row, _mask in selected]
    prediction_masks = [mask for _row, mask in selected]
    truth_semantics = [
        _normalize(
            str(row["part_name"]),
            expected_domain,
            object_category=object_category,
        )
        for row in truth_rows
    ]
    prediction_semantics = [
        (
            "body"
            if str(row["semantic_name"]) == expected_domain
            else _normalize(
                str(row["semantic_name"]).removeprefix(
                    f"{expected_domain}_"
                ),
                expected_domain,
                object_category=object_category,
            )
        )
        for row in prediction_rows
    ]
    metrics = evaluate_part_predictions(
        truth_masks=truth_masks,
        truth_semantics=truth_semantics,
        prediction_masks=prediction_masks,
        prediction_semantics=prediction_semantics,
        truth_object_mask=truth_object,
        thresholds=DEFAULT_IOU_THRESHOLDS,
    )
    part_iou = metrics.pop("object_iou")
    part_precision = metrics.pop("object_precision")
    part_recall = metrics.pop("object_recall")
    root_iou, root_precision, root_recall = _coverage_metrics(
        all_prediction_masks,
        truth_object,
    )
    metrics.update(
        {
            "root_foreground_iou": root_iou,
            "root_foreground_precision": root_precision,
            "root_foreground_recall": root_recall,
            "editable_part_coverage_iou": part_iou,
            "editable_part_coverage_precision": part_precision,
            "editable_part_coverage_recall": part_recall,
        }
    )
    return {key: float(value) for key, value in metrics.items()}


def _bootstrap_mean(
    values: list[float],
    *,
    seed: int,
    iterations: int,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indexes = generator.integers(0, len(array), size=(iterations, len(array)))
    means = array[indexes].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summaries(
    rows: list[dict[str, object]],
    *,
    iterations: int,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    variants = list(_variants())
    for variant_index, variant in enumerate(variants):
        selected = [row for row in rows if row["variant"] == variant]
        summary: dict[str, object] = {
            "variant": variant,
            "case_count": len(selected),
        }
        for metric_index, metric in enumerate(METRICS):
            interval = _bootstrap_mean(
                [float(row[metric]) for row in selected],
                seed=SEED + variant_index * 100 + metric_index,
                iterations=iterations,
            )
            summary[f"mean_{metric}"] = interval["mean"]
            summary[f"ci95_low_{metric}"] = interval["ci95_low"]
            summary[f"ci95_high_{metric}"] = interval["ci95_high"]
        output.append(summary)
    return output


def _domain_summaries(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    domains = sorted({str(row["expected_domain"]) for row in rows})
    for domain in domains:
        for variant in _variants():
            selected = [
                row
                for row in rows
                if row["variant"] == variant
                and row["expected_domain"] == domain
            ]
            summary: dict[str, object] = {
                "expected_domain": domain,
                "variant": variant,
                "case_count": len(selected),
            }
            summary.update(
                {
                    f"mean_{metric}": float(
                        np.mean([float(row[metric]) for row in selected])
                    )
                    for metric in METRICS
                }
            )
            output.append(summary)
    return output


def _paired_deltas(
    rows: list[dict[str, object]],
    *,
    iterations: int,
) -> list[dict[str, object]]:
    indexed = {
        (str(row["case_id"]), str(row["variant"])): row for row in rows
    }
    case_ids = sorted({str(row["case_id"]) for row in rows})
    output: list[dict[str, object]] = []
    for variant_index, variant in enumerate(_variants()):
        if variant == "A3_full_fusion":
            continue
        for metric_index, metric in enumerate(METRICS):
            differences = [
                float(indexed[(case_id, "A3_full_fusion")][metric])
                - float(indexed[(case_id, variant)][metric])
                for case_id in case_ids
            ]
            interval = _bootstrap_mean(
                differences,
                seed=SEED + 1000 + variant_index * 100 + metric_index,
                iterations=iterations,
            )
            output.append(
                {
                    "comparison": f"A3_full_fusion_minus_{variant}",
                    "metric": metric,
                    "case_count": len(case_ids),
                    "mean_paired_difference": interval["mean"],
                    "ci95_low": interval["ci95_low"],
                    "ci95_high": interval["ci95_high"],
                }
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run cross-domain fusion ablations on frozen HPID candidates."
        )
    )
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()

    benchmark_path = args.benchmark_root / "benchmark_summary.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    manifest_path = args.manifest or Path(str(benchmark["source_manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_cases = {
        str(row["case_id"]): row
        for row in manifest["cases"]
        if row.get("case_path")
    }
    variants = _variants()
    rows: list[dict[str, object]] = []
    reproduction_rows: list[dict[str, object]] = []

    for raw in benchmark["cases"]:
        if int(raw.get("return_code", 1)) != 0:
            continue
        case_id = str(raw["case_id"])
        package_dir = args.benchmark_root / case_id
        candidates = _load_candidates(package_dir)
        stored_map = np.asarray(Image.open(package_dir / "part_id_map.tiff"))

        # Every variant is inferred before evaluation labels are opened.
        predictions = {}
        timings = {}
        for variant, config in variants.items():
            started = time.perf_counter()
            predictions[variant] = fuse_candidates(
                candidates,
                image_shape=stored_map.shape,
                config=config,
            )
            timings[variant] = time.perf_counter() - started

        full = predictions["A3_full_fusion"]
        reproduction_rows.append(
            {
                "case_id": case_id,
                "stored_part_count": int(stored_map.max()),
                "rerun_part_count": len(full.instances),
                "foreground_exact": bool(
                    np.array_equal(stored_map > 0, full.instance_map > 0)
                ),
            }
        )

        case_path = Path(str(manifest_cases[case_id]["case_path"]))
        case = json.loads(case_path.read_text(encoding="utf-8"))
        expected_domain = str(raw["expected_domain"])
        for variant, result in predictions.items():
            metrics = _evaluate_result(
                result=result,
                case=case,
                case_dir=case_path.parent,
                expected_domain=expected_domain,
            )
            rows.append(
                {
                    "case_id": case_id,
                    "object_category": raw["object_category"],
                    "expected_domain": expected_domain,
                    "variant": variant,
                    "candidate_count": len(candidates),
                    "fusion_seconds": timings[variant],
                    "ground_truth_used_in_inference": False,
                    **metrics,
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    summary_rows = _summaries(
        rows,
        iterations=args.bootstrap_iterations,
    )
    domain_rows = _domain_summaries(rows)
    paired_rows = _paired_deltas(
        rows,
        iterations=args.bootstrap_iterations,
    )
    _write_csv(args.output / "fusion_ablation_cases.csv", rows)
    _write_csv(args.output / "fusion_ablation_summary.csv", summary_rows)
    _write_csv(args.output / "fusion_ablation_by_domain.csv", domain_rows)
    _write_csv(args.output / "fusion_ablation_paired_deltas.csv", paired_rows)
    _write_csv(args.output / "full_fusion_reproduction.csv", reproduction_rows)

    report = {
        "format": "HPID-Split cross-domain frozen-candidate fusion ablation",
        "format_version": "1.0.0",
        "release_version": "0.3.0",
        "case_count": len(reproduction_rows),
        "domain_count": len({row["expected_domain"] for row in rows}),
        "domains": sorted({str(row["expected_domain"]) for row in rows}),
        "variants": {name: asdict(config) for name, config in variants.items()},
        "full_fusion_exact_part_count_cases": sum(
            row["stored_part_count"] == row["rerun_part_count"]
            for row in reproduction_rows
        ),
        "full_fusion_exact_foreground_cases": sum(
            bool(row["foreground_exact"]) for row in reproduction_rows
        ),
        "benchmark_sha256": _sha256(benchmark_path),
        "manifest_sha256": _sha256(manifest_path),
        "candidate_scope": (
            "frozen accepted proposal masks from the original automatic run"
        ),
        "evaluation_scope": (
            "end-to-end fusion and Part-ID assignment after frozen proposal "
            "generation; not a proposal-generator ablation"
        ),
        "coverage_metric_definitions": {
            "root_foreground": (
                "union of every exported ID, including root residuals, "
                "against the object mask"
            ),
            "editable_part_coverage": (
                "union of non-root editable part IDs against the object mask"
            ),
        },
        "ground_truth_usage": (
            "all four predictions are completed before labels are loaded"
        ),
        "inference_uses_ground_truth": False,
        "seed": SEED,
    }
    (args.output / "fusion_ablation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
