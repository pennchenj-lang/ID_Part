from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SyntheticOcclusion:
    full_mask: np.ndarray
    visible_mask: np.ndarray
    occluder_mask: np.ndarray
    hidden_mask: np.ndarray
    direction: str
    hidden_fraction: float


@dataclass(frozen=True)
class AmodalCaseMetrics:
    visible_only_iou: float
    completed_iou: float
    hidden_recall: float
    added_precision: float
    false_added_ratio: float
    visible_recall: float


def _largest_component_fraction(mask: np.ndarray) -> float:
    area = int(np.count_nonzero(mask))
    if not area:
        return 0.0
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return 0.0
    largest = int(stats[1:, cv2.CC_STAT_AREA].max(initial=0))
    return largest / area


def make_edge_occlusion(
    full_mask: np.ndarray,
    *,
    target_hidden_fraction: float = 0.28,
    direction_offset: int = 0,
    margin_ratio: float = 0.12,
) -> SyntheticOcclusion:
    """Hide one edge of a known mask for an inference-independent sanity test."""
    full = full_mask.astype(bool)
    ys, xs = np.nonzero(full)
    if len(xs) < 64:
        raise ValueError("synthetic amodal cases require at least 64 mask pixels")
    if not 0.10 <= target_hidden_fraction <= 0.45:
        raise ValueError("target_hidden_fraction must be between 0.10 and 0.45")
    height, width = full.shape
    x0, x1 = int(xs.min()), int(xs.max() + 1)
    y0, y1 = int(ys.min()), int(ys.max() + 1)
    box_width = x1 - x0
    box_height = y1 - y0
    margin = max(3, round(min(box_width, box_height) * margin_ratio))
    directions = ("left", "right", "top", "bottom")
    proposals: list[tuple[float, float, int, str, np.ndarray]] = []

    for order, direction in enumerate(directions):
        occluder = np.zeros_like(full)
        if direction == "left":
            boundary = int(np.quantile(xs, target_hidden_fraction))
            occluder[
                max(0, y0 - margin) : min(height, y1 + margin),
                max(0, x0 - margin) : min(width, boundary + 1),
            ] = True
        elif direction == "right":
            boundary = int(np.quantile(xs, 1.0 - target_hidden_fraction))
            occluder[
                max(0, y0 - margin) : min(height, y1 + margin),
                max(0, boundary) : min(width, x1 + margin),
            ] = True
        elif direction == "top":
            boundary = int(np.quantile(ys, target_hidden_fraction))
            occluder[
                max(0, y0 - margin) : min(height, boundary + 1),
                max(0, x0 - margin) : min(width, x1 + margin),
            ] = True
        else:
            boundary = int(np.quantile(ys, 1.0 - target_hidden_fraction))
            occluder[
                max(0, boundary) : min(height, y1 + margin),
                max(0, x0 - margin) : min(width, x1 + margin),
            ] = True

        hidden = full & occluder
        visible = full & ~occluder
        hidden_fraction = np.count_nonzero(hidden) / len(xs)
        if not visible.any() or not hidden.any():
            continue
        contact = (
            cv2.dilate(visible.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            & hidden
        )
        if not contact.any():
            continue
        fraction_error = abs(hidden_fraction - target_hidden_fraction)
        fragmentation_penalty = 1.0 - _largest_component_fraction(visible)
        rotated_order = (order - direction_offset) % len(directions)
        proposals.append(
            (
                fraction_error,
                fragmentation_penalty,
                rotated_order,
                direction,
                occluder,
            )
        )

    if not proposals:
        raise ValueError("could not construct a connected synthetic occlusion")
    _, _, _, direction, occluder = min(proposals, key=lambda item: item[:3])
    hidden = full & occluder
    visible = full & ~occluder
    return SyntheticOcclusion(
        full_mask=full,
        visible_mask=visible,
        occluder_mask=occluder,
        hidden_mask=hidden,
        direction=direction,
        hidden_fraction=float(np.count_nonzero(hidden) / len(xs)),
    )


def evaluate_amodal_case(
    full_mask: np.ndarray,
    visible_mask: np.ndarray,
    completed_mask: np.ndarray,
) -> AmodalCaseMetrics:
    full = full_mask.astype(bool)
    visible = visible_mask.astype(bool)
    completed = completed_mask.astype(bool)
    if not (full.shape == visible.shape == completed.shape):
        raise ValueError("amodal metric masks must have equal shapes")
    if np.any(visible & ~full):
        raise ValueError("visible mask must be a subset of the known full mask")

    hidden = full & ~visible
    added = completed & ~visible
    intersection = int(np.count_nonzero(completed & full))
    union = int(np.count_nonzero(completed | full))
    visible_intersection = int(np.count_nonzero(visible & full))
    visible_union = int(np.count_nonzero(visible | full))
    false_added = int(np.count_nonzero(added & ~full))
    return AmodalCaseMetrics(
        visible_only_iou=(
            visible_intersection / visible_union if visible_union else 1.0
        ),
        completed_iou=intersection / union if union else 1.0,
        hidden_recall=(
            np.count_nonzero(completed & hidden) / np.count_nonzero(hidden)
            if hidden.any()
            else 1.0
        ),
        added_precision=(
            np.count_nonzero(added & hidden) / np.count_nonzero(added)
            if added.any()
            else 1.0
        ),
        false_added_ratio=false_added / max(1, int(np.count_nonzero(full))),
        visible_recall=(
            np.count_nonzero(completed & visible) / np.count_nonzero(visible)
            if visible.any()
            else 1.0
        ),
    )
