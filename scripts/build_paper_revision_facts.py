from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

EXPECTED_TEST_CASES = 226
EXPECTED_DOMAINS = 6
EXPECTED_VARIANTS = {
    "A0_independent_max",
    "A1_cross_source_consensus",
    "A2_consensus_hierarchy",
    "A3_full_fusion",
}
EXPECTED_BASELINES = {
    "sam2_raw",
    "sam2_nms",
    "sam2_max_ownership",
    "grounded_sam2_same_inventory",
    "hpid_split_a3",
}
PRIMARY_METRICS = (
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
    "mean_matched_iou_at_050",
    "mean_matched_boundary_f1_at_050",
    "semantic_precision_at_025",
    "semantic_recall_at_025",
    "semantic_f1_at_025",
    "semantic_f1_mean_025_075",
    "object_iou",
    "oversegmentation_ratio",
    "predicted_part_count",
    "truth_part_count",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def _summary_metrics(row: dict[str, str]) -> dict[str, float]:
    output: dict[str, float] = {}
    for metric in PRIMARY_METRICS:
        key = f"mean_{metric}"
        if key in row and row[key] != "":
            output[metric] = float(row[key])
    return output


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validate_case_count(rows: list[dict[str, str]], label: str) -> None:
    counts = {int(float(row["case_count"])) for row in rows}
    if counts != {EXPECTED_TEST_CASES}:
        raise RuntimeError(f"{label} case counts are {sorted(counts)}, expected 226")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the manuscript's machine-readable fact inventory."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--clipseg-evaluation", type=Path, required=True)
    parser.add_argument("--ablation-dir", type=Path, required=True)
    parser.add_argument("--quality-dir", type=Path, required=True)
    parser.add_argument("--sensitivity-dir", type=Path, required=True)
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--dev-ablation-dir", type=Path, required=True)
    parser.add_argument("--dev-gate-dir", type=Path, required=True)
    parser.add_argument("--parameter-dir", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    manifest_cases = [row for row in manifest.get("cases", []) if row.get("case_path")]
    if len(manifest_cases) != EXPECTED_TEST_CASES:
        raise RuntimeError(
            f"materialized test cases={len(manifest_cases)}, expected {EXPECTED_TEST_CASES}"
        )
    case_ids = [str(row["case_id"]) for row in manifest_cases]
    if len(set(case_ids)) != len(case_ids):
        raise RuntimeError("duplicate case IDs in test manifest")
    domains = Counter(str(row["expected_domain"]) for row in manifest_cases)
    if len(domains) != EXPECTED_DOMAINS:
        raise RuntimeError(f"domain count={len(domains)}, expected {EXPECTED_DOMAINS}")

    benchmark_path = args.benchmark_root / "benchmark_summary.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    successful = [
        row
        for row in benchmark.get("cases", [])
        if int(row.get("return_code", 1)) == 0
    ]
    if len(successful) != EXPECTED_TEST_CASES:
        raise RuntimeError(
            f"successful HPID cases={len(successful)}, expected {EXPECTED_TEST_CASES}"
        )
    if {str(row["case_id"]) for row in successful} != set(case_ids):
        raise RuntimeError("benchmark and manifest case IDs differ")

    baseline_rows = _read_csv(args.baseline_dir / "baseline_summary.csv")
    _validate_case_count(baseline_rows, "external baseline")
    methods = {row["method"] for row in baseline_rows}
    if methods != EXPECTED_BASELINES:
        raise RuntimeError(
            f"baseline methods={sorted(methods)}, expected={sorted(EXPECTED_BASELINES)}"
        )
    baselines = {
        row["method"]: {
            "case_count": int(float(row["case_count"])),
            "metrics": _summary_metrics(row),
            "ci95": {
                metric: [
                    _float(row, f"ci95_low_{metric}"),
                    _float(row, f"ci95_high_{metric}"),
                ]
                for metric in PRIMARY_METRICS
                if f"ci95_low_{metric}" in row
            },
        }
        for row in baseline_rows
    }

    clipseg = json.loads(args.clipseg_evaluation.read_text(encoding="utf-8"))
    clip_summary = dict(clipseg["summary"])
    if int(clip_summary["case_count"]) != EXPECTED_TEST_CASES:
        raise RuntimeError("CLIPSeg baseline does not cover all 226 cases")
    clipseg_metrics = {
        metric: float(clip_summary[metric])
        for metric in PRIMARY_METRICS
        if metric in clip_summary
    }

    ablation_rows = _read_csv(args.ablation_dir / "fusion_ablation_summary.csv")
    _validate_case_count(ablation_rows, "fusion ablation")
    variants = {row["variant"] for row in ablation_rows}
    if variants != EXPECTED_VARIANTS:
        raise RuntimeError(
            f"ablation variants={sorted(variants)}, expected={sorted(EXPECTED_VARIANTS)}"
        )
    ablations = {
        row["variant"]: {
            "case_count": int(float(row["case_count"])),
            "metrics": _summary_metrics(row),
            "ci95": {
                metric: [
                    _float(row, f"ci95_low_{metric}"),
                    _float(row, f"ci95_high_{metric}"),
                ]
                for metric in PRIMARY_METRICS
                if f"ci95_low_{metric}" in row
            },
        }
        for row in ablation_rows
    }
    paired_deltas = _read_csv(
        args.ablation_dir / "fusion_ablation_paired_deltas.csv"
    )
    ablation_report = json.loads(
        (args.ablation_dir / "fusion_ablation_report.json").read_text(
            encoding="utf-8"
        )
    )
    if int(ablation_report["full_fusion_exact_part_count_cases"]) != EXPECTED_TEST_CASES:
        raise RuntimeError("A3 part-count reproduction is incomplete")
    if int(ablation_report["full_fusion_exact_foreground_cases"]) != EXPECTED_TEST_CASES:
        raise RuntimeError("A3 foreground reproduction is incomplete")

    quality_report = json.loads(
        (args.quality_dir / "quality_exit_report.json").read_text(encoding="utf-8")
    )
    if int(quality_report["case_count"]) != EXPECTED_TEST_CASES:
        raise RuntimeError("quality-exit audit does not cover all 226 cases")
    quality_by_status = _read_csv(args.quality_dir / "quality_exit_by_status.csv")
    quality_coverage = _read_csv(args.quality_dir / "quality_exit_coverage.csv")
    quality_error_impact = _read_csv(
        args.quality_dir / "quality_exit_error_impact.csv"
    )

    sensitivity_rows = _read_csv(
        args.sensitivity_dir / "fusion_sensitivity_summary.csv"
    )
    if len(sensitivity_rows) != 25:
        raise RuntimeError(f"sensitivity rows={len(sensitivity_rows)}, expected 25")
    _validate_case_count(sensitivity_rows, "sensitivity")
    sensitivity_report = json.loads(
        (args.sensitivity_dir / "fusion_sensitivity_report.json").read_text(
            encoding="utf-8"
        )
    )

    runtime = json.loads(args.runtime_report.read_text(encoding="utf-8"))
    if int(runtime["successful_case_count"]) != EXPECTED_TEST_CASES:
        raise RuntimeError("runtime audit does not cover all 226 cases")

    dev_ablation_report = json.loads(
        (args.dev_ablation_dir / "fusion_ablation_report.json").read_text(
            encoding="utf-8"
        )
    )
    dev_gate_report = json.loads(
        (args.dev_gate_dir / "gate_to_final_report.json").read_text(encoding="utf-8")
    )
    parameter_report = json.loads(
        (args.parameter_dir / "parameter_registry_report.json").read_text(
            encoding="utf-8"
        )
    )

    payload = {
        "format": "HPID-Split manuscript fact inventory",
        "format_version": "1.0.0",
        "evaluation_scope": (
            "Object-conditioned part decomposition with oracle ground-truth object "
            "box/root; no test part mask, part name, or part count is used in prediction."
        ),
        "test": {
            "case_count": EXPECTED_TEST_CASES,
            "domain_count": EXPECTED_DOMAINS,
            "domains": dict(sorted(domains.items())),
            "object_category_count": len(
                {str(row["object_category"]) for row in manifest_cases}
            ),
            "unique_image_count": len(
                {str(row.get("image_id")) for row in manifest_cases}
            ),
            "unique_object_count": len(
                {str(row.get("object_annotation_id")) for row in manifest_cases}
            ),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": _sha256(args.manifest),
            "benchmark_sha256": _sha256(benchmark_path),
        },
        "development": {
            "case_count": int(dev_ablation_report["case_count"]),
            "manifest_sha256": dev_ablation_report["manifest_sha256"],
            "gate_to_final": dev_gate_report,
        },
        "external_baselines": baselines,
        "clipseg_ovparts_style": {
            "case_count": int(clip_summary["case_count"]),
            "successful_case_count": int(clip_summary["successful_case_count"]),
            "metrics": clipseg_metrics,
            "evaluation_sha256": _sha256(args.clipseg_evaluation),
            "interpretation": clipseg["baseline_interpretation"],
        },
        "fusion_ablation": {
            "variants": ablations,
            "paired_deltas": paired_deltas,
            "report": ablation_report,
        },
        "quality_exit": {
            "report": quality_report,
            "by_status": quality_by_status,
            "coverage": quality_coverage,
            "error_impact": quality_error_impact,
        },
        "sensitivity": {
            "rows": sensitivity_rows,
            "report": sensitivity_report,
        },
        "runtime": runtime,
        "parameter_registry": parameter_report,
        "code": {
            "repository": "https://github.com/pennchenj-lang/ID_Part",
            "head_commit": _git(args.repo, "rev-parse", "HEAD"),
            "release_tag": "v0.3.0",
            "release_tag_commit": _git(args.repo, "rev-list", "-n", "1", "v0.3.0"),
            "working_tree_clean": _git(args.repo, "status", "--porcelain") == "",
        },
        "source_files": {
            "baseline_report_sha256": _sha256(
                args.baseline_dir / "baseline_report.json"
            ),
            "ablation_report_sha256": _sha256(
                args.ablation_dir / "fusion_ablation_report.json"
            ),
            "quality_report_sha256": _sha256(
                args.quality_dir / "quality_exit_report.json"
            ),
            "sensitivity_report_sha256": _sha256(
                args.sensitivity_dir / "fusion_sensitivity_report.json"
            ),
            "runtime_report_sha256": _sha256(args.runtime_report),
        },
    }

    args.output.mkdir(parents=True, exist_ok=True)
    output_path = args.output / "paper_revision_facts.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.output / "paper_revision_facts.sha256").write_text(
        f"{_sha256(output_path)}  {output_path.name}\n", encoding="ascii"
    )
    print(json.dumps({"output": str(output_path), "sha256": _sha256(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
