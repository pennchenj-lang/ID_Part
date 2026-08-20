from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

from .taxonomy import Taxonomy


def binary_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.count_nonzero(left | right)
    return float(np.count_nonzero(left & right) / union) if union else 1.0


def boundary_f1(prediction: np.ndarray, truth: np.ndarray, tolerance: int = 3) -> float:
    kernel = np.ones((3, 3), np.uint8)
    pred_boundary = prediction ^ (cv2.erode(prediction.astype(np.uint8), kernel) > 0)
    truth_boundary = truth ^ (cv2.erode(truth.astype(np.uint8), kernel) > 0)
    if not pred_boundary.any() and not truth_boundary.any():
        return 1.0
    if not pred_boundary.any() or not truth_boundary.any():
        return 0.0
    dilation = np.ones((2 * tolerance + 1, 2 * tolerance + 1), np.uint8)
    truth_near = cv2.dilate(truth_boundary.astype(np.uint8), dilation) > 0
    pred_near = cv2.dilate(pred_boundary.astype(np.uint8), dilation) > 0
    precision = np.count_nonzero(pred_boundary & truth_near) / np.count_nonzero(
        pred_boundary
    )
    recall = np.count_nonzero(truth_boundary & pred_near) / np.count_nonzero(
        truth_boundary
    )
    return (
        float(2 * precision * recall / (precision + recall))
        if precision + recall
        else 0.0
    )


def _components(mask: np.ndarray, minimum_area: int = 12) -> list[np.ndarray]:
    labels, count = ndi.label(mask)
    return [
        labels == index
        for index in range(1, count + 1)
        if np.count_nonzero(labels == index) >= minimum_area
    ]


def evaluate_semantic(
    prediction: np.ndarray,
    truth: np.ndarray,
    taxonomy: Taxonomy,
    *,
    boundary_tolerance: int = 3,
    component_iou_threshold: float = 0.25,
    small_part_fraction: float = 0.01,
) -> dict[str, float | int]:
    class_ious: list[float] = []
    class_boundaries: list[float] = []
    foreground = max(1, np.count_nonzero(truth > 0))
    component_true_positive = 0
    component_prediction_count = 0
    component_truth_count = 0
    classwise_part_count_abs_error = 0
    small_count = 0
    small_matched = 0
    for class_id in range(1, taxonomy.num_fine_classes):
        truth_mask = truth == class_id
        if not truth_mask.any():
            continue
        prediction_mask = prediction == class_id
        class_ious.append(binary_iou(prediction_mask, truth_mask))
        class_boundaries.append(
            boundary_f1(prediction_mask, truth_mask, boundary_tolerance)
        )
        true_components = _components(truth_mask)
        pred_components = _components(prediction_mask)
        component_truth_count += len(true_components)
        component_prediction_count += len(pred_components)
        classwise_part_count_abs_error += abs(
            len(pred_components) - len(true_components)
        )
        matrix = np.zeros(
            (len(true_components), len(pred_components)), dtype=np.float32
        )
        for row, true_component in enumerate(true_components):
            for column, pred_component in enumerate(pred_components):
                matrix[row, column] = binary_iou(true_component, pred_component)
        matched_rows: set[int] = set()
        if matrix.size:
            rows, columns = linear_sum_assignment(1.0 - matrix)
            for row, column in zip(rows, columns):
                if matrix[row, column] >= component_iou_threshold:
                    component_true_positive += 1
                    matched_rows.add(int(row))
        for row, component in enumerate(true_components):
            if np.count_nonzero(component) / foreground <= small_part_fraction:
                small_count += 1
                small_matched += int(row in matched_rows)
    precision = (
        component_true_positive / component_prediction_count
        if component_prediction_count
        else 0.0
    )
    recall = (
        component_true_positive / component_truth_count
        if component_truth_count
        else 0.0
    )
    component_f1 = (
        2 * precision * recall / (precision + recall) if precision + recall else 0.0
    )

    mapping = np.asarray(taxonomy.fine_to_parent)
    coarse_prediction = mapping[prediction]
    coarse_truth = mapping[truth]
    coarse_ious = [
        binary_iou(coarse_prediction == parent_id, coarse_truth == parent_id)
        for parent_id in range(1, taxonomy.num_parent_classes)
        if np.any(coarse_truth == parent_id)
    ]
    return {
        "foreground_iou": binary_iou(prediction > 0, truth > 0),
        "coarse_miou": float(np.mean(coarse_ious)) if coarse_ious else 0.0,
        "semantic_miou": float(np.mean(class_ious)) if class_ious else 0.0,
        "boundary_f1": float(np.mean(class_boundaries)) if class_boundaries else 0.0,
        "small_part_recall": small_matched / small_count if small_count else 1.0,
        "component_f1": component_f1,
        "component_precision": precision,
        "component_recall": recall,
        "component_truth_count": component_truth_count,
        "component_prediction_count": component_prediction_count,
        "part_count_abs_error": classwise_part_count_abs_error,
    }
