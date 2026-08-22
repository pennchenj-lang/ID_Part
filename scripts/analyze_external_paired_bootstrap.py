from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from hpid_split.paper_audit import paired_bootstrap_interval

SEED = 20260822
EXPECTED_CASES = 226
METRICS = (
    "part_f1_at_025",
    "part_f1_at_050",
    "part_f1_at_075",
    "part_f1_mean_025_075",
    "mean_matched_boundary_f1_at_050",
    "semantic_f1_at_025",
    "object_iou",
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


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute case-paired HPID-Split minus external-baseline intervals."
    )
    parser.add_argument("--baseline-cases", type=Path, required=True)
    parser.add_argument("--clipseg-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=20_000)
    args = parser.parse_args()

    baseline_rows = _read_csv(args.baseline_cases)
    indexed = {
        (str(row["case_id"]), str(row["method"])): row for row in baseline_rows
    }
    hpid_case_ids = sorted(
        case_id
        for case_id, method in indexed
        if method == "hpid_split_a3"
    )
    if len(hpid_case_ids) != EXPECTED_CASES:
        raise RuntimeError(
            f"HPID case count is {len(hpid_case_ids)}, expected {EXPECTED_CASES}"
        )

    clipseg_payload = json.loads(
        args.clipseg_evaluation.read_text(encoding="utf-8")
    )
    clipseg_rows = {
        str(row["case_id"]): row for row in clipseg_payload.get("cases", [])
    }
    if set(clipseg_rows) != set(hpid_case_ids):
        raise RuntimeError("CLIPSeg and HPID-Split case identities differ")

    methods = sorted(
        {method for _case_id, method in indexed if method != "hpid_split_a3"}
    )
    methods.append("clipseg_ovparts_style")
    output_rows: list[dict[str, object]] = []
    for method_index, method in enumerate(methods):
        if method == "clipseg_ovparts_style":
            comparison = clipseg_rows
        else:
            comparison = {
                case_id: indexed[(case_id, method)] for case_id in hpid_case_ids
            }
        if set(comparison) != set(hpid_case_ids):
            raise RuntimeError(f"{method} does not cover the common 226 cases")
        for metric_index, metric in enumerate(METRICS):
            differences = [
                float(indexed[(case_id, "hpid_split_a3")][metric])
                - float(comparison[case_id][metric])
                for case_id in hpid_case_ids
            ]
            interval = paired_bootstrap_interval(
                differences,
                seed=SEED + method_index * 100 + metric_index,
                iterations=args.bootstrap_iterations,
            )
            output_rows.append(
                {
                    "comparison": f"hpid_split_a3_minus_{method}",
                    "baseline": method,
                    "metric": metric,
                    **interval,
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    output_csv = args.output / "external_paired_differences.csv"
    _write_csv(output_csv, output_rows)
    report = {
        "format": "HPID-Split external case-paired bootstrap audit",
        "format_version": "1.0.0",
        "case_count": EXPECTED_CASES,
        "methods": methods,
        "metrics": list(METRICS),
        "bootstrap_iterations": args.bootstrap_iterations,
        "seed": SEED,
        "unit_of_resampling": "independent object case",
        "interpretation": (
            "Intervals estimate HPID-Split A3 minus each baseline. An interval "
            "including zero is reported as statistical comparability, not superiority."
        ),
        "baseline_cases_sha256": _sha256(args.baseline_cases),
        "clipseg_evaluation_sha256": _sha256(args.clipseg_evaluation),
        "output_csv_sha256": _sha256(output_csv),
    }
    (args.output / "external_paired_bootstrap_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
