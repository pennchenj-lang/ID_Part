from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .instances import PartInstance


@dataclass(frozen=True)
class AmodalPart:
    record: PartInstance
    visible_mask: np.ndarray
    full_mask: np.ndarray
    completed_rgba: np.ndarray
    completion_confidence: float
    occluder_instance_indices: tuple[int, ...]
    added_area_px: int


def _convex_candidate(mask: np.ndarray) -> np.ndarray:
    points = np.column_stack(np.nonzero(mask))[:, ::-1].astype(np.int32)
    if len(points) < 3:
        return mask.copy()
    hull = cv2.convexHull(points)
    output = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillConvexPoly(output, hull, 1)
    return output.astype(bool)


def _ellipse_candidate(mask: np.ndarray) -> np.ndarray:
    points = np.column_stack(np.nonzero(mask))[:, ::-1].astype(np.float32)
    if len(points) < 12:
        return mask.copy()
    center = points.mean(axis=0)
    covariance = np.cov(points.T)
    values, vectors = np.linalg.eigh(covariance)
    values = np.maximum(values, 1.0)
    axes = np.sqrt(values) * 2.45
    angle = float(np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1])))
    output = np.zeros_like(mask, dtype=np.uint8)
    cv2.ellipse(
        output,
        tuple(round(value) for value in center),
        tuple(max(1, round(value)) for value in axes),
        angle,
        0,
        360,
        1,
        -1,
    )
    return output.astype(bool)


def _reflection_candidate(mask: np.ndarray) -> np.ndarray:
    y, x = np.nonzero(mask)
    if len(x) < 8:
        return mask.copy()
    points = np.column_stack([x, y]).astype(np.float64)
    center = points.mean(axis=0)
    covariance = np.cov((points - center).T)
    _, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, 1]
    relative = points - center
    projection = relative @ axis
    reflected = center + (2.0 * projection[:, None] * axis - relative)
    reflected = np.rint(reflected).astype(np.int32)
    valid = (
        (reflected[:, 0] >= 0)
        & (reflected[:, 0] < mask.shape[1])
        & (reflected[:, 1] >= 0)
        & (reflected[:, 1] < mask.shape[0])
    )
    output = mask.copy()
    output[reflected[valid, 1], reflected[valid, 0]] = True
    return cv2.morphologyEx(
        output.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)
    ).astype(bool)


def _closing_candidates(mask: np.ndarray) -> list[np.ndarray]:
    y, x = np.nonzero(mask)
    if not len(x):
        return [mask.copy()]
    width = int(x.max() - x.min() + 1)
    height = int(y.max() - y.min() + 1)
    radius = max(2, round(min(width, height) * 0.10))
    candidates = []
    for factor in (1.0, 1.8):
        size = max(3, 2 * round(radius * factor) + 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        candidates.append(
            cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(
                bool
            )
        )
    return candidates


def _limit_expansion(
    candidate: np.ndarray, visible: np.ndarray, maximum_ratio: float = 2.5
) -> np.ndarray:
    visible_area = max(1, np.count_nonzero(visible))
    maximum_area = round(visible_area * maximum_ratio)
    candidate = candidate | visible
    if np.count_nonzero(candidate) <= maximum_area:
        return candidate
    distance = ndi.distance_transform_edt(~visible)
    allowed = np.argpartition(distance.ravel(), maximum_area - 1)[:maximum_area]
    output = np.zeros_like(visible)
    output.ravel()[allowed] = candidate.ravel()[allowed]
    return output | visible


def _complete_mask(
    visible: np.ndarray,
    occupied: np.ndarray,
) -> tuple[np.ndarray, float, tuple[int, ...]]:
    boundary_ring = (
        cv2.dilate(
            visible.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1
        ).astype(bool)
        & ~visible
    )
    contact = boundary_ring & occupied
    if np.count_nonzero(contact) < max(3, round(np.count_nonzero(visible) * 0.002)):
        return visible.copy(), 1.0, ()

    candidates = [
        visible.copy(),
        *_closing_candidates(visible),
        _convex_candidate(visible),
        _ellipse_candidate(visible),
        _reflection_candidate(visible),
    ]
    candidates = [_limit_expansion(candidate, visible) for candidate in candidates]
    stack = np.stack(candidates, axis=0)
    votes = stack.sum(axis=0)
    support = cv2.dilate(
        visible.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
        iterations=1,
    ).astype(bool)
    consensus = visible | ((votes >= 3) & support)
    added = consensus & ~visible
    if not added.any():
        scores = []
        for candidate in candidates[1:]:
            extension = candidate & ~visible
            overlap = np.count_nonzero(extension & occupied) / max(
                1, np.count_nonzero(extension)
            )
            ratio = np.count_nonzero(candidate) / max(1, np.count_nonzero(visible))
            scores.append(overlap - 0.15 * abs(ratio - 1.35))
        consensus = candidates[1 + int(np.argmax(scores))]
        added = consensus & ~visible
    union = stack.any(axis=0) & ~visible
    disagreement = (
        np.mean(np.var(stack[:, union].astype(np.float32), axis=0))
        if union.any()
        else 0.0
    )
    contact_support = np.count_nonzero(added & occupied) / max(
        1, np.count_nonzero(added)
    )
    confidence = float(
        np.clip(0.70 * (1.0 - 4.0 * disagreement) + 0.30 * contact_support, 0.05, 0.95)
    )
    return consensus, confidence, ()


def _nearest_visible_texture(
    image: np.ndarray, visible: np.ndarray, full: np.ndarray
) -> np.ndarray:
    output = image.copy()
    hidden = full & ~visible
    if not hidden.any() or not visible.any():
        output[..., 3] = full.astype(np.uint8) * 255
        return output
    _, indices = ndi.distance_transform_edt(~visible, return_indices=True)
    output[hidden, :3] = output[indices[0][hidden], indices[1][hidden], :3]
    output[..., 3] = full.astype(np.uint8) * 255
    return output


def complete_instances(
    image: Image.Image,
    instance_map: np.ndarray,
    records: list[PartInstance],
) -> tuple[list[AmodalPart], list[dict[str, object]]]:
    source = np.asarray(image.convert("RGBA")).copy()
    foreground = instance_map > 0
    completed: list[AmodalPart] = []
    edges: list[dict[str, object]] = []
    for record in records:
        visible = instance_map == record.instance_index
        other = foreground & ~visible
        full, confidence, _ = _complete_mask(visible, other)
        contact_ring = (
            cv2.dilate(visible.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            & other
        )
        occluders = tuple(
            int(value)
            for value in np.unique(instance_map[contact_ring])
            if int(value) not in {0, record.instance_index}
        )
        for occluder in occluders:
            edges.append(
                {
                    "occluder_instance_index": occluder,
                    "occluded_instance_index": record.instance_index,
                    "relation": "adjacency-supported occlusion hypothesis",
                    "confidence": confidence,
                }
            )
        completed.append(
            AmodalPart(
                record=record,
                visible_mask=visible,
                full_mask=full,
                completed_rgba=_nearest_visible_texture(source, visible, full),
                completion_confidence=confidence,
                occluder_instance_indices=occluders,
                added_area_px=int(np.count_nonzero(full & ~visible)),
            )
        )
    return completed, edges
