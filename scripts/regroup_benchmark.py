from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from scripts.regroup_package import regroup_package
except ModuleNotFoundError:  # Direct execution adds scripts/ rather than the repo root.
    from regroup_package import regroup_package

from hpid_split import __version__
from hpid_split.paco_eval import evaluate_paco_package


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _register_benchmark_evaluation(package: Path) -> None:
    """Refresh package hashes after adding the ground-truth-only evaluation."""

    manifest_path = package / "package_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["benchmark_evaluation_path"] = "paco_evaluation.json"
    manifest["benchmark_evaluation_uses_ground_truth"] = True
    payloads = sorted(
        path
        for path in package.rglob("*")
        if path.is_file() and path.name != "package_manifest.json"
    )
    manifest["files"] = [
        {
            "path": path.relative_to(package).as_posix(),
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in payloads
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    metric_names = (
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
    metrics = {
        name: float(np.mean([float(row[name]) for row in rows])) if rows else 0.0
        for name in metric_names
    }
    group_rows = [
        row["editable_group_metrics"]
        for row in rows
        if isinstance(row.get("editable_group_metrics"), dict)
    ]
    group_metrics = {
        name: float(np.mean([float(row[name]) for row in group_rows]))
        if group_rows
        else 0.0
        for name in metric_names[3:]
    }
    return {
        "case_count": len(rows),
        "successful_case_count": len(rows),
        "failed_case_count": 0,
        "domain_accuracy": float(np.mean([bool(row["domain_correct"]) for row in rows])),
        "profile_accuracy": float(
            np.mean(
                [
                    bool(row["profile_correct"])
                    for row in rows
                    if row.get("profile_correct") is not None
                ]
            )
        ),
        "macro_metrics": metrics,
        "editable_group_case_count": len(group_rows),
        "editable_group_macro_metrics": group_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild and evaluate editable groups for every frozen package in an "
            "existing PACO benchmark without rerunning proposal inference."
        )
    )
    parser.add_argument("--source-benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_benchmark.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_summary_path = source / "benchmark_summary.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    manifest_path = Path(str(source_summary["source_manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_cases = {str(row["case_id"]): row for row in manifest["cases"]}

    rows: list[dict[str, object]] = []
    for source_row in source_summary["cases"]:
        case_id = str(source_row["case_id"])
        case = manifest_cases[case_id]
        source_package = source / case_id
        output_package = output / case_id
        regroup_package(source_package, output_package)
        evaluation = evaluate_paco_package(
            output_package,
            Path(str(case["case_path"])),
            expected_domain=str(case["expected_domain"]),
            expected_profile=(
                str(case["expected_profile"])
                if case.get("expected_profile") is not None
                else None
            ),
        )
        (output_package / "paco_evaluation.json").write_text(
            json.dumps(evaluation, indent=2), encoding="utf-8"
        )
        _register_benchmark_evaluation(output_package)
        row = dict(source_row)
        row["return_code"] = 0
        row["regroup_source_package"] = str(source_package)
        row["candidate_generation_rerun"] = False
        row["fine_part_map_changed"] = False
        row.update(
            {
                key: value
                for key, value in evaluation.items()
                if key
                not in {
                    "format",
                    "format_version",
                    "package",
                    "case",
                    "matches",
                    "semantic_matches",
                }
            }
        )
        rows.append(row)
        group_metrics = evaluation.get("editable_group_metrics")
        group_f1 = (
            float(group_metrics["part_discovery_f1_at_025"])
            if isinstance(group_metrics, dict)
            else float("nan")
        )
        print(f"{case_id}: group_f1={group_f1:.4f}", flush=True)

    summary = {
        key: value
        for key, value in source_summary.items()
        if key not in {"aggregate", "cases"}
    }
    summary.update(
        {
            "format": "HPID PACO frozen-package regroup benchmark",
            "format_version": "0.1.0",
            "source_benchmark": str(source),
            "source_benchmark_summary_sha256": _sha256(source_summary_path),
            "candidate_generation_rerun": False,
            "fine_part_map_changed": False,
            "regroup_algorithm": {
                "name": "HPID-Split",
                "version": __version__,
                "stage": "editable_physical_group_regeneration",
                "ground_truth_used": False,
            },
            "evaluation_reads_ground_truth_after_regrouping": True,
            "aggregate": _aggregate(rows),
            "cases": rows,
        }
    )
    (output / "benchmark_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
