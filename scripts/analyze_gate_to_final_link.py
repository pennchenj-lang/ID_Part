from __future__ import annotations

import argparse
import csv
import hashlib
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

CONDITIONS = (
    "C1_semantic",
    "C2_semantic_structure",
    "C3_full_serial_gate",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _spearman(first: list[float], second: list[float]) -> tuple[float, float]:
    if len(set(first)) < 2 or len(set(second)) < 2:
        return 0.0, 1.0
    result = spearmanr(first, second)
    return float(result.statistic), float(result.pvalue)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Relate the frozen candidate-gate audit to final A3 Part-ID output."
        )
    )
    parser.add_argument("--gate-cases", type=Path, required=True)
    parser.add_argument("--fusion-cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    gate_rows = _read_csv(args.gate_cases)
    final_rows = {
        str(row["case_id"]): row
        for row in _read_csv(args.fusion_cases)
        if row["variant"] == "A3_full_fusion"
    }
    aligned = [row for row in gate_rows if row["case_id"] in final_rows]

    summary_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    for condition in CONDITIONS:
        selected = [row for row in aligned if row["condition"] == condition]
        summary_rows.append(
            {
                "condition": condition,
                "case_count": len(selected),
                "mean_candidate_count": float(
                    np.mean([float(row["candidate_count"]) for row in selected])
                ),
                "mean_candidate_precision_at_025": float(
                    np.mean([float(row["precision_at_025"]) for row in selected])
                ),
                "mean_candidate_recall_at_025": float(
                    np.mean([float(row["recall_at_025"]) for row in selected])
                ),
                "mean_candidate_f1_at_025": float(
                    np.mean([float(row["f1_at_025"]) for row in selected])
                ),
                "mean_candidate_precision_at_050": float(
                    np.mean([float(row["precision_at_050"]) for row in selected])
                ),
                "mean_candidate_recall_at_050": float(
                    np.mean([float(row["recall_at_050"]) for row in selected])
                ),
                "mean_candidate_f1_at_050": float(
                    np.mean([float(row["f1_at_050"]) for row in selected])
                ),
                "zero_candidate_cases": sum(
                    int(float(row["candidate_count"])) == 0 for row in selected
                ),
            }
        )
        pairs = (
            ("candidate_f1_at_025", "f1_at_025", "final_part_f1_at_025", "part_f1_at_025"),
            ("candidate_f1_at_050", "f1_at_050", "final_part_f1_at_050", "part_f1_at_050"),
            ("candidate_recall_at_025", "recall_at_025", "final_part_recall_at_025", "part_recall_at_025"),
            ("candidate_precision_at_050", "precision_at_050", "final_part_precision_at_050", "part_precision_at_050"),
            ("candidate_count", "candidate_count", "final_oversegmentation_ratio", "oversegmentation_ratio"),
        )
        for source_label, source_key, target_label, target_key in pairs:
            source = [float(row[source_key]) for row in selected]
            target = [
                float(final_rows[str(row["case_id"])][target_key])
                for row in selected
            ]
            coefficient, p_value = _spearman(source, target)
            correlation_rows.append(
                {
                    "condition": condition,
                    "case_count": len(selected),
                    "source_metric": source_label,
                    "target_metric": target_label,
                    "spearman_rho": coefficient,
                    "two_sided_p_value": p_value,
                }
            )

    indexed = {
        (str(row["case_id"]), str(row["condition"])): row for row in aligned
    }
    transition_rows: list[dict[str, object]] = []
    for earlier, later in pairwise(CONDITIONS):
        case_ids = sorted(
            {
                case_id
                for case_id, condition in indexed
                if condition == earlier and (case_id, later) in indexed
            }
        )
        for metric in ("candidate_count", "f1_at_025", "f1_at_050"):
            deltas = np.asarray(
                [
                    float(indexed[(case_id, later)][metric])
                    - float(indexed[(case_id, earlier)][metric])
                    for case_id in case_ids
                ],
                dtype=np.float64,
            )
            for direction, selector in (
                ("decreased", deltas < -1e-12),
                ("unchanged", np.abs(deltas) <= 1e-12),
                ("increased", deltas > 1e-12),
            ):
                selected_ids = [
                    case_id
                    for case_id, keep in zip(case_ids, selector, strict=True)
                    if bool(keep)
                ]
                transition_rows.append(
                    {
                        "transition": f"{earlier}_to_{later}",
                        "candidate_metric": metric,
                        "change_direction": direction,
                        "case_count": len(selected_ids),
                        "mean_candidate_delta": float(deltas[selector].mean())
                        if selected_ids
                        else 0.0,
                        "mean_final_part_f1_at_025": float(
                            np.mean(
                                [
                                    float(final_rows[case_id]["part_f1_at_025"])
                                    for case_id in selected_ids
                                ]
                            )
                        )
                        if selected_ids
                        else 0.0,
                        "mean_final_part_f1_at_050": float(
                            np.mean(
                                [
                                    float(final_rows[case_id]["part_f1_at_050"])
                                    for case_id in selected_ids
                                ]
                            )
                        )
                        if selected_ids
                        else 0.0,
                    }
                )

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "gate_summary.csv", summary_rows)
    _write_csv(args.output / "gate_final_correlations.csv", correlation_rows)
    _write_csv(args.output / "gate_transition_strata.csv", transition_rows)
    report = {
        "format": "HPID-Split candidate-gate to final-output linkage audit",
        "format_version": "1.0.0",
        "case_count": len(final_rows),
        "gate_conditions": list(CONDITIONS),
        "gate_cases_sha256": _sha256(args.gate_cases),
        "fusion_cases_sha256": _sha256(args.fusion_cases),
        "interpretation": (
            "C1-C3 measure candidate retention before fusion. A0-A3 use the "
            "same frozen C3 candidate set and measure final identity fusion. "
            "Correlations describe propagation, not an additional causal "
            "ablation of the appearance gate."
        ),
    }
    (args.output / "gate_to_final_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
