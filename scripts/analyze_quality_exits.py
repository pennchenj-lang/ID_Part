from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from analyze_public_baselines import _hpid_predictions, _truth

from hpid_split.paper_eval import DEFAULT_IOU_THRESHOLDS, evaluate_part_predictions

FAILURE_PART_F1_050 = 0.20
FAILURE_SEMANTIC_F1_025 = 0.20
METRICS = (
    "part_f1_at_025",
    "part_f1_at_050",
    "part_f1_at_075",
    "part_f1_mean_025_075",
    "mean_matched_iou_at_050",
    "mean_matched_boundary_f1_at_050",
    "semantic_recall_at_025",
    "semantic_f1_at_025",
    "semantic_f1_mean_025_075",
    "object_iou",
    "oversegmentation_ratio",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _quality_features(package_dir: Path) -> dict[str, object]:
    report = json.loads(
        (package_dir / "quality_report.json").read_text(encoding="utf-8")
    )
    diagnostics = json.loads(
        (package_dir / "inference_diagnostics.json").read_text(
            encoding="utf-8"
        )
    )
    structure = dict(report.get("part_structure") or {})
    uncertainty = dict(report.get("asset_domain_uncertainty") or {})
    return {
        "quality_status": str(report.get("status") or "unknown"),
        "quality_evidence_grade": str(report.get("evidence_grade") or ""),
        "original_review_flag": str(report.get("status") or "unknown")
        != "ready",
        "generic_name_ratio": float(structure.get("generic_name_ratio") or 0.0),
        "root_residual_ratio": float(
            structure.get("root_residual_ratio") or 0.0
        ),
        "part_count": int(structure.get("part_count") or 0),
        "confirmed_profile_count": len(
            list(structure.get("confirmed_profiles") or [])
        ),
        "unresolved_disagreement_count": int(
            uncertainty.get("unresolved_disagreement_count") or 0
        ),
        "weak_selected_root_evidence": bool(
            uncertainty.get("weak_selected_root_evidence") or False
        ),
        "candidate_source_count": int(
            diagnostics.get("candidate_source_count") or 0
        ),
    }


def _detection_metrics(
    rows: list[dict[str, object]], flag_name: str
) -> dict[str, float]:
    true_positive = sum(bool(row[flag_name]) and bool(row["failure"]) for row in rows)
    false_positive = sum(bool(row[flag_name]) and not bool(row["failure"]) for row in rows)
    false_negative = sum(not bool(row[flag_name]) and bool(row["failure"]) for row in rows)
    true_negative = len(rows) - true_positive - false_positive - false_negative
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "true_positive": float(true_positive),
        "false_positive": float(false_positive),
        "false_negative": float(false_negative),
        "true_negative": float(true_negative),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "review_rate": sum(bool(row[flag_name]) for row in rows) / max(1, len(rows)),
    }


def _candidate_rules() -> list[dict[str, object]]:
    rules: list[dict[str, object]] = [
        {"feature": "confirmed_profile_count", "operator": "<=", "value": 0},
        {
            "feature": "unresolved_disagreement_count",
            "operator": ">",
            "value": 0,
        },
        {
            "feature": "weak_selected_root_evidence",
            "operator": "is_true",
            "value": True,
        },
    ]
    for threshold in (0.05, 0.10, 0.20):
        rules.append(
            {
                "feature": "generic_name_ratio",
                "operator": ">=",
                "value": threshold,
            }
        )
    for threshold in (0.10, 0.20, 0.40, 0.55, 0.70):
        rules.append(
            {
                "feature": "root_residual_ratio",
                "operator": ">=",
                "value": threshold,
            }
        )
    for threshold in (2, 3, 4):
        rules.append(
            {"feature": "part_count", "operator": "<=", "value": threshold}
        )
    for threshold in (10, 15, 20):
        rules.append(
            {"feature": "part_count", "operator": ">=", "value": threshold}
        )
    for threshold in (1, 2):
        rules.append(
            {
                "feature": "candidate_source_count",
                "operator": "<=",
                "value": threshold,
            }
        )
    return rules


def _applies(row: dict[str, object], rule: dict[str, object]) -> bool:
    value = row[str(rule["feature"])]
    threshold = rule["value"]
    operator = str(rule["operator"])
    if operator == "is_true":
        return bool(value)
    if operator == "<=":
        return float(value) <= float(threshold)
    if operator == ">=":
        return float(value) >= float(threshold)
    if operator == ">":
        return float(value) > float(threshold)
    raise ValueError(f"unsupported operator: {operator}")


def _derive_rules(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    baseline = _detection_metrics(rows, "original_review_flag")["f1"]
    for _step in range(2):
        best_rule: dict[str, object] | None = None
        best_f1 = baseline
        best_rate = 1.0
        for rule in _candidate_rules():
            if rule in selected:
                continue
            trial = [*selected, rule]
            for row in rows:
                row["trial_review_flag"] = bool(row["original_review_flag"]) or any(
                    _applies(row, item) for item in trial
                )
            result = _detection_metrics(rows, "trial_review_flag")
            if result["f1"] > best_f1 + 1e-12 or (
                abs(result["f1"] - best_f1) <= 1e-12
                and result["review_rate"] < best_rate
            ):
                best_rule = rule
                best_f1 = result["f1"]
                best_rate = result["review_rate"]
        if best_rule is None or best_f1 < baseline + 0.02:
            break
        selected.append(best_rule)
        baseline = best_f1
    for row in rows:
        row.pop("trial_review_flag", None)
    return selected


def _group_summary(
    rows: list[dict[str, object]], key: str
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for value in sorted({str(row[key]) for row in rows}):
        selected = [row for row in rows if str(row[key]) == value]
        summary: dict[str, object] = {
            key: value,
            "case_count": len(selected),
            "failure_rate": float(
                np.mean([float(bool(row["failure"])) for row in selected])
            ),
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


def _coverage_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    tiers = (
        ("original_ready_only", lambda row: not bool(row["original_review_flag"])),
        ("revised_ready_only", lambda row: not bool(row["revised_review_flag"])),
        (
            "original_ready_plus_review",
            lambda row: str(row["quality_status"])
            in {"ready", "review_recommended"},
        ),
        ("all_outputs", lambda _row: True),
    )
    output: list[dict[str, object]] = []
    for label, predicate in tiers:
        selected = [row for row in rows if predicate(row)]
        output.append(
            {
                "acceptance_policy": label,
                "accepted_case_count": len(selected),
                "coverage": len(selected) / max(1, len(rows)),
                "failure_rate": float(
                    np.mean([float(bool(row["failure"])) for row in selected])
                )
                if selected
                else 0.0,
                **{
                    f"mean_{metric}": float(
                        np.mean([float(row[metric]) for row in selected])
                    )
                    if selected
                    else 0.0
                    for metric in METRICS
                },
            }
        )
    return output


def _error_impact(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    error_names = (
        "routing_error",
        "profile_error",
        "weak_root",
        "part_discovery_failure",
        "semantic_assignment_failure",
    )
    output: list[dict[str, object]] = []
    for error_name in error_names:
        affected = [row for row in rows if bool(row[error_name])]
        unaffected = [row for row in rows if not bool(row[error_name])]
        summary: dict[str, object] = {
            "error_type": error_name,
            "affected_count": len(affected),
            "unaffected_count": len(unaffected),
            "review_capture_rate": (
                float(
                    np.mean(
                        [float(bool(row["revised_review_flag"])) for row in affected]
                    )
                )
                if affected
                else 0.0
            ),
        }
        for metric in (
            "part_f1_at_050",
            "mean_matched_boundary_f1_at_050",
            "semantic_recall_at_025",
            "semantic_f1_at_025",
        ):
            affected_mean = (
                float(np.mean([float(row[metric]) for row in affected]))
                if affected
                else 0.0
            )
            unaffected_mean = (
                float(np.mean([float(row[metric]) for row in unaffected]))
                if unaffected
                else 0.0
            )
            summary[f"affected_mean_{metric}"] = affected_mean
            summary[f"unaffected_mean_{metric}"] = unaffected_mean
            summary[f"affected_minus_unaffected_{metric}"] = (
                affected_mean - unaffected_mean
            )
        output.append(summary)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate HPID-Split quality exits and review flags."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--derive-rule", action="store_true")
    group.add_argument("--rule", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    manifest_cases = {
        str(row["case_id"]): row
        for row in manifest.get("cases", [])
        if row.get("case_path")
    }
    benchmark_path = args.benchmark_root / "benchmark_summary.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for raw in benchmark.get("cases", []):
        if int(raw.get("return_code", 1)) != 0:
            continue
        case_id = str(raw["case_id"])
        package_dir = args.benchmark_root / case_id
        expected_domain = str(raw["expected_domain"])
        case_path = Path(str(manifest_cases[case_id]["case_path"]))
        case, truth_masks, truth_semantics, truth_object = _truth(
            case_path, expected_domain
        )
        prediction_masks, prediction_semantics = _hpid_predictions(
            package_dir, expected_domain, truth_semantics
        )
        metrics = evaluate_part_predictions(
            truth_masks=truth_masks,
            truth_semantics=truth_semantics,
            prediction_masks=prediction_masks,
            prediction_semantics=prediction_semantics,
            truth_object_mask=truth_object,
            thresholds=DEFAULT_IOU_THRESHOLDS,
        )
        part_failure = metrics["part_f1_at_050"] < FAILURE_PART_F1_050
        semantic_failure = (
            metrics["semantic_f1_at_025"] < FAILURE_SEMANTIC_F1_025
        )
        rows.append(
            {
                "case_id": case_id,
                "object_category": case["object_category"],
                "expected_domain": expected_domain,
                "routing_error": not bool(raw.get("domain_correct", False)),
                "profile_error": not bool(raw.get("profile_correct", False)),
                "weak_root": metrics["object_iou"] < 0.50,
                "part_discovery_failure": part_failure,
                "semantic_assignment_failure": semantic_failure,
                "failure": part_failure or semantic_failure,
                **_quality_features(package_dir),
                **metrics,
            }
        )

    if args.derive_rule:
        rules = _derive_rules(rows)
        rule_source = "derived_once_on_this_development_set"
    else:
        payload = json.loads(args.rule.read_text(encoding="utf-8"))
        rules = list(payload["additional_review_rules"])
        rule_source = str(args.rule.resolve())
    for row in rows:
        row["revised_review_flag"] = bool(row["original_review_flag"]) or any(
            _applies(row, rule) for rule in rules
        )

    args.output.mkdir(parents=True, exist_ok=True)
    status_rows = _group_summary(rows, "quality_status")
    coverage_rows = _coverage_rows(rows)
    error_rows = _error_impact(rows)
    _write_csv(args.output / "quality_exit_cases.csv", rows)
    _write_csv(args.output / "quality_exit_by_status.csv", status_rows)
    _write_csv(args.output / "quality_exit_coverage.csv", coverage_rows)
    _write_csv(args.output / "quality_exit_error_impact.csv", error_rows)
    rule_payload = {
        "format": "HPID-Split review-rule registry",
        "format_version": "1.0.0",
        "development_manifest_sha256": _sha256(args.manifest)
        if args.derive_rule
        else None,
        "base_rule": "quality_status != ready",
        "additional_review_rules": rules,
        "failure_definition": {
            "part_f1_at_050_below": FAILURE_PART_F1_050,
            "or_semantic_f1_at_025_below": FAILURE_SEMANTIC_F1_025,
        },
        "selection_note": (
            "At most two observable additions were selected greedily on the "
            "development set, each requiring at least +0.02 absolute failure-"
            "detection F1. The rule is frozen before test evaluation."
        ),
    }
    (args.output / "review_rule.json").write_text(
        json.dumps(rule_payload, indent=2), encoding="utf-8"
    )
    report = {
        "format": "HPID-Split quality-exit evaluation",
        "format_version": "1.0.0",
        "case_count": len(rows),
        "manifest_sha256": _sha256(args.manifest),
        "benchmark_sha256": _sha256(benchmark_path),
        "rule_source": rule_source,
        "additional_review_rules": rules,
        "original_review_detection": _detection_metrics(
            rows, "original_review_flag"
        ),
        "revised_review_detection": _detection_metrics(
            rows, "revised_review_flag"
        ),
        "ready_semantic_failure_count": sum(
            not bool(row["original_review_flag"])
            and bool(row["semantic_assignment_failure"])
            for row in rows
        ),
        "ground_truth_usage": (
            "Quality features and review flags are produced without labels; "
            "PACO labels are read afterward to score failure detection."
        ),
    }
    (args.output / "quality_exit_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
