from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from hpid_split.metrics import binary_iou, boundary_f1

METRICS = (
    "object_iou",
    "object_precision",
    "object_recall",
    "part_discovery_precision_at_025",
    "part_discovery_recall_at_025",
    "part_discovery_f1_at_025",
    "mean_matched_iou",
    "mean_matched_boundary_f1",
    "semantic_part_recall",
    "oversegmentation_ratio",
)

GATE_CONDITIONS = (
    ("C1_semantic", "stage_1_semantic"),
    ("C2_semantic_structure", "stage_2_structure"),
    ("C3_full_serial_gate", "accepted"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) >= 128


def _bootstrap_mean(
    values: list[float],
    *,
    seed: int,
    iterations: int,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0}
    generator = np.random.default_rng(seed)
    indexes = generator.integers(0, len(array), size=(iterations, len(array)))
    means = array[indexes].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _group_count(path: Path) -> int:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("groups", [])
    return len(rows)


def _part_count(path: Path) -> int:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("parts", [])
    return len(rows)


def _stage_seconds(path: Path) -> float:
    diagnostics = json.loads(path.read_text(encoding="utf-8"))
    timings = diagnostics.get("stage_timings_seconds") or {}
    return float(sum(float(value) for value in timings.values()))


def _failure_class(row: dict[str, object]) -> str:
    if not bool(row.get("domain_correct")):
        return "routing_error"
    object_iou = float(row["object_iou"])
    part_f1 = float(row["part_discovery_f1_at_025"])
    semantic_recall = float(row["semantic_part_recall"])
    if object_iou < 0.25:
        return "root_failure"
    if object_iou < 0.50:
        return "weak_root"
    if part_f1 < 0.10:
        return "part_discovery_failure"
    if semantic_recall < 0.10:
        return "semantic_assignment_failure"
    return "partial_or_strong_success"


def _find_gate_audit(diagnostics: dict[str, object]) -> dict[str, object] | None:
    grouping = diagnostics.get("physical_grouping")
    if not isinstance(grouping, dict):
        return None
    audit = grouping.get("three_stage_candidate_verification")
    return audit if isinstance(audit, dict) else None


def _gate_verified(row: dict[str, object], key: str) -> bool:
    if key == "accepted":
        return bool(row.get("accepted"))
    stage = row.get(key)
    return bool(stage.get("verified")) if isinstance(stage, dict) else False


def _match_metrics(
    truth_masks: list[np.ndarray],
    prediction_masks: list[np.ndarray],
    *,
    threshold: float,
    boundary_tolerance: int,
) -> dict[str, float]:
    matrix = np.zeros(
        (len(truth_masks), len(prediction_masks)), dtype=np.float32
    )
    for truth_index, truth in enumerate(truth_masks):
        for prediction_index, prediction in enumerate(prediction_masks):
            matrix[truth_index, prediction_index] = binary_iou(
                truth, prediction
            )
    if matrix.size:
        truth_indexes, prediction_indexes = linear_sum_assignment(1.0 - matrix)
        matches = [
            (
                int(truth_index),
                int(prediction_index),
                float(matrix[truth_index, prediction_index]),
            )
            for truth_index, prediction_index in zip(
                truth_indexes, prediction_indexes, strict=True
            )
        ]
    else:
        matches = []
    accepted = [row for row in matches if row[2] >= threshold]
    precision = len(accepted) / max(1, len(prediction_masks))
    recall = len(accepted) / max(1, len(truth_masks))
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    boundaries = [
        boundary_f1(
            prediction_masks[prediction_index],
            truth_masks[truth_index],
            tolerance=boundary_tolerance,
        )
        for truth_index, prediction_index, _overlap in accepted
    ]
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": (
            float(np.mean([row[2] for row in accepted])) if accepted else 0.0
        ),
        "mean_matched_boundary_f1": (
            float(np.mean(boundaries)) if boundaries else 0.0
        ),
    }


def _candidate_by_key(package_dir: Path) -> dict[str, dict[str, object]]:
    rows = json.loads((package_dir / "candidates.json").read_text("utf-8"))
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        metadata = row.get("metadata") or {}
        key = metadata.get("candidate_key")
        if key:
            result[str(key)] = row
    return result


def _gate_case_rows(
    *,
    package_dir: Path,
    case_path: Path,
    case_id: str,
    expected_domain: str,
) -> list[dict[str, object]]:
    diagnostics = json.loads(
        (package_dir / "inference_diagnostics.json").read_text("utf-8")
    )
    audit = _find_gate_audit(diagnostics)
    if audit is None:
        return []
    gate_rows = [
        row for row in audit.get("candidates", []) if isinstance(row, dict)
    ]
    candidate_map = _candidate_by_key(package_dir)
    case = json.loads(case_path.read_text(encoding="utf-8"))
    truth_masks = [
        _load_mask(case_path.parent / str(row["mask_crop"]))
        for row in case["parts"]
    ]
    output: list[dict[str, object]] = []
    for condition, gate_key in GATE_CONDITIONS:
        selected: list[dict[str, object]] = []
        missing_keys = 0
        for gate_row in gate_rows:
            if not _gate_verified(gate_row, gate_key):
                continue
            key = str(gate_row.get("candidate_key", ""))
            candidate = candidate_map.get(key)
            if candidate is None:
                missing_keys += 1
                continue
            if str(candidate.get("semantic_name", "")) == expected_domain:
                continue
            selected.append(candidate)
        prediction_masks = [
            _load_mask(package_dir / str(row["mask_path"])) for row in selected
        ]
        at_025 = _match_metrics(
            truth_masks,
            prediction_masks,
            threshold=0.25,
            boundary_tolerance=3,
        )
        at_050 = _match_metrics(
            truth_masks,
            prediction_masks,
            threshold=0.50,
            boundary_tolerance=3,
        )
        output.append(
            {
                "case_id": case_id,
                "expected_domain": expected_domain,
                "condition": condition,
                "candidate_count": len(prediction_masks),
                "truth_part_count": len(truth_masks),
                "missing_candidate_keys": missing_keys,
                "precision_at_025": at_025["precision"],
                "recall_at_025": at_025["recall"],
                "f1_at_025": at_025["f1"],
                "precision_at_050": at_050["precision"],
                "recall_at_050": at_050["recall"],
                "f1_at_050": at_050["f1"],
                "mean_matched_iou_at_025": at_025["mean_matched_iou"],
                "mean_matched_boundary_f1_at_025": at_025[
                    "mean_matched_boundary_f1"
                ],
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create leakage-aware paper tables from a frozen benchmark."
    )
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_cases = {
        str(row["case_id"]): row
        for row in manifest.get("cases", [])
        if row.get("case_path")
    }
    package_root = args.benchmark.parent
    cases: list[dict[str, object]] = []
    gate_cases: list[dict[str, object]] = []
    for raw in benchmark.get("cases", []):
        if int(raw.get("return_code", 1)) != 0:
            continue
        case_id = str(raw["case_id"])
        manifest_row = manifest_cases[case_id]
        package_dir = package_root / case_id
        quality = json.loads(
            (package_dir / "quality_report.json").read_text(encoding="utf-8")
        )
        row = {
            key: raw.get(key)
            for key in (
                "case_id",
                "object_category",
                "expected_domain",
                "expected_profile",
                "selected_domain",
                "domain_correct",
                "profile_correct",
                *METRICS,
            )
        }
        row.update(
            {
                "part_count": _part_count(package_dir / "parts.json"),
                "editable_group_count": _group_count(
                    package_dir / "groups.json"
                ),
                "recorded_stage_seconds": _stage_seconds(
                    package_dir / "inference_diagnostics.json"
                ),
                "quality_status": quality.get("status"),
                "evidence_grade": quality.get("evidence_grade"),
            }
        )
        row["failure_class"] = _failure_class(row)
        cases.append(row)
        gate_cases.extend(
            _gate_case_rows(
                package_dir=package_dir,
                case_path=Path(str(manifest_row["case_path"])),
                case_id=case_id,
                expected_domain=str(manifest_row["expected_domain"]),
            )
        )

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "automatic_holdout_cases.csv", cases)
    _write_csv(args.output / "candidate_gate_ablation_cases.csv", gate_cases)

    domain_rows: list[dict[str, object]] = []
    by_domain: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in cases:
        by_domain[str(row["expected_domain"])].append(row)
    for domain, rows in sorted(by_domain.items()):
        domain_row: dict[str, object] = {
            "expected_domain": domain,
            "case_count": len(rows),
            "domain_accuracy": float(
                np.mean([bool(row["domain_correct"]) for row in rows])
            ),
            "zero_part_f1_count": sum(
                float(row["part_discovery_f1_at_025"]) == 0.0 for row in rows
            ),
            "mean_recorded_stage_seconds": float(
                np.mean([float(row["recorded_stage_seconds"]) for row in rows])
            ),
        }
        domain_row.update(
            {
                f"mean_{metric}": float(
                    np.mean([float(row[metric]) for row in rows])
                )
                for metric in METRICS
            }
        )
        domain_rows.append(domain_row)
    _write_csv(args.output / "automatic_holdout_by_domain.csv", domain_rows)

    overall = {
        metric: _bootstrap_mean(
            [float(row[metric]) for row in cases],
            seed=args.seed + index,
            iterations=args.bootstrap_iterations,
        )
        for index, metric in enumerate(METRICS)
    }
    failure_counts = {
        label: sum(row["failure_class"] == label for row in cases)
        for label in sorted({str(row["failure_class"]) for row in cases})
    }

    gate_summary_rows: list[dict[str, object]] = []
    for condition, _gate_key in GATE_CONDITIONS:
        rows = [row for row in gate_cases if row["condition"] == condition]
        summary_row: dict[str, object] = {
            "condition": condition,
            "case_count": len(rows),
            "mean_candidate_count": float(
                np.mean([float(row["candidate_count"]) for row in rows])
            )
            if rows
            else 0.0,
            "missing_candidate_key_count": int(
                sum(int(row["missing_candidate_keys"]) for row in rows)
            ),
        }
        for metric in (
            "precision_at_025",
            "recall_at_025",
            "f1_at_025",
            "precision_at_050",
            "recall_at_050",
            "f1_at_050",
            "mean_matched_iou_at_025",
            "mean_matched_boundary_f1_at_025",
        ):
            interval = _bootstrap_mean(
                [float(row[metric]) for row in rows],
                seed=args.seed + len(gate_summary_rows) * 100 + len(metric),
                iterations=args.bootstrap_iterations,
            )
            summary_row[f"mean_{metric}"] = interval["mean"]
            summary_row[f"ci95_low_{metric}"] = interval["ci95_low"]
            summary_row[f"ci95_high_{metric}"] = interval["ci95_high"]
        gate_summary_rows.append(summary_row)
    _write_csv(
        args.output / "candidate_gate_ablation_summary.csv", gate_summary_rows
    )

    gate_by_condition = {
        condition: {
            str(row["case_id"]): row
            for row in gate_cases
            if row["condition"] == condition
        }
        for condition, _gate_key in GATE_CONDITIONS
    }
    paired_rows: list[dict[str, object]] = []
    comparisons = (
        ("C2_minus_C1", "C2_semantic_structure", "C1_semantic"),
        ("C3_minus_C2", "C3_full_serial_gate", "C2_semantic_structure"),
        ("C3_minus_C1", "C3_full_serial_gate", "C1_semantic"),
    )
    paired_metrics = (
        "candidate_count",
        "precision_at_025",
        "recall_at_025",
        "f1_at_025",
        "precision_at_050",
        "recall_at_050",
        "f1_at_050",
        "mean_matched_iou_at_025",
        "mean_matched_boundary_f1_at_025",
    )
    for comparison, left, right in comparisons:
        shared = sorted(
            set(gate_by_condition[left]) & set(gate_by_condition[right])
        )
        for metric_index, metric in enumerate(paired_metrics):
            differences = [
                float(gate_by_condition[left][case_id][metric])
                - float(gate_by_condition[right][case_id][metric])
                for case_id in shared
            ]
            interval = _bootstrap_mean(
                differences,
                seed=args.seed + 1000 + len(paired_rows) + metric_index,
                iterations=args.bootstrap_iterations,
            )
            paired_rows.append(
                {
                    "comparison": comparison,
                    "left_condition": left,
                    "right_condition": right,
                    "metric": metric,
                    "case_count": len(shared),
                    "mean_paired_difference": interval["mean"],
                    "ci95_low": interval["ci95_low"],
                    "ci95_high": interval["ci95_high"],
                }
            )
    _write_csv(args.output / "candidate_gate_paired_deltas.csv", paired_rows)

    incomplete_cases = [
        str(row.get("case_id"))
        for row in manifest.get("cases", [])
        if not row.get("case_path")
    ]
    quality_status_counts = {
        label: sum(row["quality_status"] == label for row in cases)
        for label in sorted({str(row["quality_status"]) for row in cases})
    }
    recorded_stage_seconds = np.asarray(
        [float(row["recorded_stage_seconds"]) for row in cases],
        dtype=np.float64,
    )
    report = {
        "format": "HPID-Split frozen paper result analysis",
        "format_version": "1.0.0",
        "release_version": "0.3.0",
        "benchmark_sha256": _sha256(args.benchmark),
        "manifest_sha256": _sha256(args.manifest),
        "manifest_listed_case_count": len(manifest.get("cases", [])),
        "materialized_case_count": len(manifest_cases),
        "evaluated_case_count": len(cases),
        "incomplete_manifest_cases": incomplete_cases,
        "failed_process_count": sum(
            int(row.get("return_code", 1)) != 0
            for row in benchmark.get("cases", [])
        ),
        "domain_accuracy": float(
            np.mean([bool(row["domain_correct"]) for row in cases])
        ),
        "profile_accuracy": float(
            np.mean([bool(row["profile_correct"]) for row in cases])
        ),
        "zero_part_f1_case_count": sum(
            float(row["part_discovery_f1_at_025"]) == 0.0 for row in cases
        ),
        "quality_status_counts": quality_status_counts,
        "quality_ready_rate": quality_status_counts.get("ready", 0)
        / max(1, len(cases)),
        "recorded_stage_seconds": {
            "mean": float(recorded_stage_seconds.mean()),
            "median": float(np.median(recorded_stage_seconds)),
            "p90": float(np.quantile(recorded_stage_seconds, 0.90)),
            "minimum": float(recorded_stage_seconds.min()),
            "maximum": float(recorded_stage_seconds.max()),
        },
        "failure_class_counts": failure_counts,
        "macro_metrics_with_case_bootstrap_ci95": overall,
        "gate_ablation_scope": (
            "candidate-retention audit on frozen proposals; not final grouped "
            "end-to-end segmentation"
        ),
        "gate_ablation_case_count": len(
            {str(row["case_id"]) for row in gate_cases}
        ),
        "ground_truth_usage": (
            "ground truth is loaded by this post-inference evaluator only"
        ),
        "inference_uses_ground_truth": False,
    }
    (args.output / "paper_result_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
