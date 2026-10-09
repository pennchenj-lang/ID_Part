from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scripts.run_semantic_completion_frontend_benchmark import (
    METHODS,
    SEED,
    _mask_iou,
    _normalize,
    _normalized_name,
    _predictions,
)


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) >= 128


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap(values: list[float], seed: int, iterations: int) -> dict[str, float]:
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit target-routing correctness on the frozen completion cases."
    )
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()

    manifest = json.loads(args.target_manifest.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for target in manifest["selected_cases"]:
        case_id = str(target["case_id"])
        object_category = str(target["object_category"])
        expected_domain = str(target["expected_domain"])
        target_name = str(target["target_part_name"])
        truth = _load_mask(
            args.reference_root / str(target["target_mask_relative_path"])
        )
        predictions = _predictions(args.inference_root / case_id)
        normalized_target = _normalize(
            target_name,
            expected_domain,
            object_category=object_category,
        )

        for method in METHODS:
            prediction = predictions[method]
            matches = [
                instance
                for instance in prediction.instances
                if _normalized_name(
                    instance,
                    expected_domain=expected_domain,
                    object_category=object_category,
                )
                == normalized_target
            ]
            selected = (
                max(
                    matches,
                    key=lambda item: (
                        float(item.confidence),
                        int(np.count_nonzero(item.mask)),
                        item.identity,
                    ),
                )
                if matches
                else None
            )
            match_ious = [_mask_iou(instance.mask, truth) for instance in matches]
            selected_iou = _mask_iou(selected.mask, truth) if selected else 0.0
            best_iou = max(match_ious, default=0.0)
            match_count = len(matches)
            unique = match_count == 1
            ambiguous = match_count > 1
            unresolved = match_count == 0
            rows.append(
                {
                    "case_id": case_id,
                    "method": method,
                    "target_part_name": target_name,
                    "query_match_count": match_count,
                    "query_unique": float(unique),
                    "query_ambiguous": float(ambiguous),
                    "query_unresolved": float(unresolved),
                    "review_option_count": (
                        match_count if matches else len(prediction.instances)
                    ),
                    "selected_part_iou": selected_iou,
                    "best_review_match_iou": best_iou,
                    "unique_and_correct_025": float(unique and selected_iou >= 0.25),
                    "unique_and_correct_050": float(unique and selected_iou >= 0.50),
                    "wrong_unique_025": float(unique and selected_iou < 0.25),
                    "wrong_unique_050": float(unique and selected_iou < 0.50),
                    "correct_in_review_set_025": float(ambiguous and best_iou >= 0.25),
                    "correct_in_review_set_050": float(ambiguous and best_iou >= 0.50),
                }
            )

    summary_rows: list[dict[str, object]] = []
    summary_metrics = (
        "query_unique",
        "query_ambiguous",
        "query_unresolved",
        "review_option_count",
        "unique_and_correct_025",
        "unique_and_correct_050",
        "wrong_unique_025",
        "wrong_unique_050",
        "selected_part_iou",
        "best_review_match_iou",
    )
    for method_index, method in enumerate(METHODS):
        selected_rows = [row for row in rows if row["method"] == method]
        summary: dict[str, object] = {
            "method": method,
            "case_count": len(selected_rows),
        }
        for metric_index, metric in enumerate(summary_metrics):
            interval = _bootstrap(
                [float(row[metric]) for row in selected_rows],
                SEED + method_index * 100 + metric_index,
                args.bootstrap_iterations,
            )
            for key, value in interval.items():
                summary[f"{key}_{metric}"] = value

        ambiguous_rows = [
            row for row in selected_rows if float(row["query_ambiguous"]) == 1.0
        ]
        summary["ambiguous_case_count"] = len(ambiguous_rows)
        for threshold in ("025", "050"):
            metric = f"correct_in_review_set_{threshold}"
            values = [float(row[metric]) for row in ambiguous_rows]
            interval = _bootstrap(
                values,
                SEED + method_index * 100 + 50 + int(threshold),
                args.bootstrap_iterations,
            )
            for key, value in interval.items():
                summary[f"{key}_{metric}_given_ambiguous"] = value
        summary_rows.append(summary)

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "routing_correctness_cases.csv", rows)
    _write_csv(args.output / "routing_correctness_summary.csv", summary_rows)
    report = {
        "format": "HPID-Split frozen routing-correctness audit",
        "format_version": "1.0.0",
        "case_count": int(manifest["case_count"]),
        "methods": list(METHODS),
        "thresholds": [0.25, 0.50],
        "definitions": {
            "unique_and_correct": (
                "Exactly one semantic match and its mask reaches the IoU threshold."
            ),
            "wrong_unique": (
                "Exactly one semantic match but its mask is below the IoU threshold."
            ),
            "correct_in_review_set_given_ambiguous": (
                "Among queries with multiple semantic matches, at least one match reaches "
                "the IoU threshold."
            ),
            "unresolved": "No exported identity matches the semantic query.",
        },
        "reference_usage": (
            "Reference part masks are used only for this post-hoc routing audit and were "
            "not supplied to proposal generation, HPID fusion, or semantic query resolution."
        ),
    }
    (args.output / "routing_correctness_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
