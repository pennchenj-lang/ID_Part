from __future__ import annotations

import json

from scripts.run_paco_benchmark import _aggregate, _write_summary


def test_write_summary_includes_trailing_resumed_rows(tmp_path) -> None:
    path = tmp_path / "benchmark_summary.json"
    rows = [
        {"case_id": "executed", "return_code": 1},
        {"case_id": "resumed_after_execution", "return_code": 1},
    ]

    _write_summary(path, {"format": "test"}, rows)

    summary = json.loads(path.read_text(encoding="utf-8"))
    assert [row["case_id"] for row in summary["cases"]] == [
        "executed",
        "resumed_after_execution",
    ]
    assert summary["aggregate"]["case_count"] == 2


def test_aggregate_keeps_fine_and_editable_group_metrics_separate() -> None:
    fine = {
        "object_iou": 0.8,
        "object_precision": 0.9,
        "object_recall": 0.85,
        "part_discovery_precision_at_025": 0.2,
        "part_discovery_recall_at_025": 0.3,
        "part_discovery_f1_at_025": 0.24,
        "mean_matched_iou": 0.4,
        "mean_matched_boundary_f1": 0.5,
        "semantic_part_recall": 0.25,
        "oversegmentation_ratio": 2.0,
    }
    groups = {
        "part_discovery_precision_at_025": 0.8,
        "part_discovery_recall_at_025": 0.8,
        "part_discovery_f1_at_025": 0.8,
        "mean_matched_iou": 0.75,
        "mean_matched_boundary_f1": 0.7,
        "semantic_part_recall": 0.8,
        "oversegmentation_ratio": 1.0,
    }
    row = {
        "return_code": 0,
        "domain_correct": True,
        "profile_correct": True,
        **fine,
        "editable_group_metrics": groups,
    }

    aggregate = _aggregate([row])

    assert aggregate["macro_metrics"]["part_discovery_f1_at_025"] == 0.24
    assert (
        aggregate["editable_group_macro_metrics"][
            "part_discovery_f1_at_025"
        ]
        == 0.8
    )
