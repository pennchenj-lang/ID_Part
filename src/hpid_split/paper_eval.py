from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .metrics import binary_iou, boundary_f1

DEFAULT_IOU_THRESHOLDS = tuple(
    round(float(value), 2) for value in np.arange(0.25, 0.751, 0.05)
)


def _threshold_key(value: float) -> str:
    return f"{round(value * 100):03d}"


def _hungarian(
    truth_masks: Sequence[np.ndarray],
    prediction_masks: Sequence[np.ndarray],
) -> list[tuple[int, int, float]]:
    matrix = np.zeros(
        (len(truth_masks), len(prediction_masks)), dtype=np.float32
    )
    for truth_index, truth in enumerate(truth_masks):
        for prediction_index, prediction in enumerate(prediction_masks):
            matrix[truth_index, prediction_index] = binary_iou(
                truth, prediction
            )
    if not matrix.size:
        return []
    truth_indexes, prediction_indexes = linear_sum_assignment(1.0 - matrix)
    return [
        (
            int(truth_index),
            int(prediction_index),
            float(matrix[truth_index, prediction_index]),
        )
        for truth_index, prediction_index in zip(
            truth_indexes, prediction_indexes, strict=True
        )
    ]


def _semantic_hungarian(
    truth_masks: Sequence[np.ndarray],
    truth_semantics: Sequence[str],
    prediction_masks: Sequence[np.ndarray],
    prediction_semantics: Sequence[str | None],
) -> list[tuple[int, int, float]]:
    truth_groups: dict[str, list[int]] = defaultdict(list)
    prediction_groups: dict[str, list[int]] = defaultdict(list)
    for index, semantic in enumerate(truth_semantics):
        truth_groups[str(semantic)].append(index)
    for index, semantic in enumerate(prediction_semantics):
        if semantic:
            prediction_groups[str(semantic)].append(index)
    matches: list[tuple[int, int, float]] = []
    for semantic, truth_indexes in truth_groups.items():
        prediction_indexes = prediction_groups.get(semantic, [])
        local = _hungarian(
            [truth_masks[index] for index in truth_indexes],
            [prediction_masks[index] for index in prediction_indexes],
        )
        matches.extend(
            (
                truth_indexes[truth_index],
                prediction_indexes[prediction_index],
                overlap,
            )
            for truth_index, prediction_index, overlap in local
        )
    return matches


def _prf(
    accepted_count: int,
    *,
    truth_count: int,
    prediction_count: int,
) -> tuple[float, float, float]:
    precision = accepted_count / max(1, prediction_count)
    recall = accepted_count / max(1, truth_count)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, f1


def evaluate_part_predictions(
    *,
    truth_masks: Sequence[np.ndarray],
    truth_semantics: Sequence[str],
    prediction_masks: Sequence[np.ndarray],
    prediction_semantics: Sequence[str | None],
    truth_object_mask: np.ndarray,
    thresholds: Sequence[float] = DEFAULT_IOU_THRESHOLDS,
    boundary_tolerance: int = 3,
) -> dict[str, float]:
    """Evaluate instance masks with class-agnostic and semantic matching.

    Predictions may overlap. Semantic precision uses every predicted mask as
    its denominator, so unlabeled proposals are not silently ignored.
    """

    if len(truth_masks) != len(truth_semantics):
        raise ValueError("truth masks and semantics must have equal length")
    if len(prediction_masks) != len(prediction_semantics):
        raise ValueError(
            "prediction masks and semantics must have equal length"
        )
    normalized_truth = [np.asarray(mask, dtype=bool) for mask in truth_masks]
    normalized_predictions = [
        np.asarray(mask, dtype=bool) for mask in prediction_masks
    ]
    object_mask = np.asarray(truth_object_mask, dtype=bool)
    if normalized_predictions:
        prediction_union = np.logical_or.reduce(normalized_predictions)
    else:
        prediction_union = np.zeros(object_mask.shape, dtype=bool)
    if prediction_union.shape != object_mask.shape:
        raise ValueError("prediction and truth object masks differ in shape")

    class_matches = _hungarian(normalized_truth, normalized_predictions)
    semantic_matches = _semantic_hungarian(
        normalized_truth,
        truth_semantics,
        normalized_predictions,
        prediction_semantics,
    )
    result: dict[str, float] = {
        "truth_part_count": float(len(normalized_truth)),
        "predicted_part_count": float(len(normalized_predictions)),
        "oversegmentation_ratio": len(normalized_predictions)
        / max(1, len(normalized_truth)),
        "object_iou": binary_iou(prediction_union, object_mask),
        "object_precision": float(
            np.count_nonzero(prediction_union & object_mask)
            / max(1, np.count_nonzero(prediction_union))
        ),
        "object_recall": float(
            np.count_nonzero(prediction_union & object_mask)
            / max(1, np.count_nonzero(object_mask))
        ),
    }
    part_f1_values: list[float] = []
    semantic_f1_values: list[float] = []
    for threshold in thresholds:
        key = _threshold_key(float(threshold))
        accepted = [row for row in class_matches if row[2] >= threshold]
        precision, recall, f1 = _prf(
            len(accepted),
            truth_count=len(normalized_truth),
            prediction_count=len(normalized_predictions),
        )
        semantic_accepted = [
            row for row in semantic_matches if row[2] >= threshold
        ]
        semantic_precision, semantic_recall, semantic_f1 = _prf(
            len(semantic_accepted),
            truth_count=len(normalized_truth),
            prediction_count=len(normalized_predictions),
        )
        boundaries = [
            boundary_f1(
                normalized_predictions[prediction_index],
                normalized_truth[truth_index],
                tolerance=boundary_tolerance,
            )
            for truth_index, prediction_index, _overlap in accepted
        ]
        result.update(
            {
                f"part_precision_at_{key}": precision,
                f"part_recall_at_{key}": recall,
                f"part_f1_at_{key}": f1,
                f"mean_matched_iou_at_{key}": float(
                    np.mean([row[2] for row in accepted])
                )
                if accepted
                else 0.0,
                f"mean_matched_boundary_f1_at_{key}": float(
                    np.mean(boundaries)
                )
                if boundaries
                else 0.0,
                f"semantic_precision_at_{key}": semantic_precision,
                f"semantic_recall_at_{key}": semantic_recall,
                f"semantic_f1_at_{key}": semantic_f1,
            }
        )
        part_f1_values.append(f1)
        semantic_f1_values.append(semantic_f1)
    result["part_f1_mean_025_075"] = float(np.mean(part_f1_values))
    result["semantic_f1_mean_025_075"] = float(
        np.mean(semantic_f1_values)
    )
    return result
