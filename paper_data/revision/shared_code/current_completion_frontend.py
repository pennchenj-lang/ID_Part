from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ControlledDefect:
    mask: np.ndarray
    mode: str
    requested_fraction: float
    actual_fraction: float
    seed: int


@dataclass(frozen=True)
class CompletionHandoffMetrics:
    request_recall: float
    request_precision: float
    target_mask_iou: float
    visible_lock_exact: bool
    changed_outside_request_fraction: float
    hidden_region_mae: float
    hidden_region_psnr: float


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    union = int(np.count_nonzero(first | second))
    return intersection / union if union else 0.0


def _select_nearest_pixels(
    target: np.ndarray,
    center_xy: tuple[float, float],
    count: int,
) -> np.ndarray:
    y, x = np.nonzero(target)
    output = np.zeros(target.shape, dtype=bool)
    if not len(x) or count <= 0:
        return output
    distances = (x - center_xy[0]) ** 2 + (y - center_xy[1]) ** 2
    count = min(count, len(x))
    indexes = np.argpartition(distances, count - 1)[:count]
    output[y[indexes], x[indexes]] = True
    return output


def make_controlled_defect(
    target_mask: np.ndarray,
    *,
    seed: int,
    fraction: float = 0.18,
    mode: str = "interior",
) -> ControlledDefect:
    """Create a deterministic, evaluation-only missing region inside one part."""
    target = np.asarray(target_mask, dtype=bool)
    area = int(np.count_nonzero(target))
    if area < 24:
        raise ValueError("target mask is too small for controlled completion")
    if not 0.05 <= fraction <= 0.45:
        raise ValueError("controlled defect fraction must lie in [0.05, 0.45]")
    if mode not in {"interior", "boundary"}:
        raise ValueError("controlled defect mode must be interior or boundary")

    desired = int(np.clip(round(area * fraction), 12, max(12, area - 12)))
    y, x = np.nonzero(target)
    generator = np.random.default_rng(seed)
    if mode == "interior":
        distance = cv2.distanceTransform(target.astype(np.uint8), cv2.DIST_L2, 5)
        maximum = float(distance.max())
        eligible = np.column_stack(np.nonzero(distance >= maximum * 0.82))
        chosen = eligible[int(generator.integers(0, len(eligible)))]
        center = (float(chosen[1]), float(chosen[0]))
    else:
        directions = (
            (float(x.min()), float(np.median(y[x == x.min()]))),
            (float(x.max()), float(np.median(y[x == x.max()]))),
            (float(np.median(x[y == y.min()])), float(y.min())),
            (float(np.median(x[y == y.max()])), float(y.max())),
        )
        center = directions[int(generator.integers(0, len(directions)))]

    defect = _select_nearest_pixels(target, center, desired)
    # Keep a single connected region. Nearest-pixel selection is usually already
    # connected, but thin or holed parts can create small detached islands.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        defect.astype(np.uint8), connectivity=8
    )
    if count > 2:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        defect = labels == largest
    return ControlledDefect(
        mask=defect,
        mode=mode,
        requested_fraction=fraction,
        actual_fraction=float(np.count_nonzero(defect) / area),
        seed=seed,
    )


def corrupt_region(image: Image.Image, defect_mask: np.ndarray) -> Image.Image:
    source = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    defect = np.asarray(defect_mask, dtype=bool)
    if defect.shape != source.shape[:2]:
        raise ValueError("defect mask must match the source image size")
    context = source[~defect]
    fill = (
        np.median(context, axis=0).astype(np.uint8)
        if len(context)
        else np.array([127, 127, 127], dtype=np.uint8)
    )
    source[defect] = fill
    return Image.fromarray(source, mode="RGB")


def visible_lock_region(
    corrupted: Image.Image,
    generated_rgba: np.ndarray,
    request_mask: np.ndarray,
) -> np.ndarray:
    source = np.asarray(corrupted.convert("RGBA"), dtype=np.uint8)
    generated = np.asarray(generated_rgba, dtype=np.uint8)
    request = np.asarray(request_mask, dtype=bool)
    if generated.shape != source.shape:
        raise ValueError("generated image must match the corrupted image size")
    if request.shape != source.shape[:2]:
        raise ValueError("request mask must match the corrupted image size")
    output = source.copy()
    output[request] = generated[request]
    return output


def evaluate_completion_handoff(
    *,
    original: Image.Image,
    corrupted: Image.Image,
    completed_rgba: np.ndarray,
    requested_mask: np.ndarray,
    truth_defect_mask: np.ndarray,
    selected_part_mask: np.ndarray,
    truth_part_mask: np.ndarray,
) -> CompletionHandoffMetrics:
    original_rgb = np.asarray(original.convert("RGB"), dtype=np.uint8)
    corrupted_rgb = np.asarray(corrupted.convert("RGB"), dtype=np.uint8)
    completed_rgb = np.asarray(completed_rgba, dtype=np.uint8)[..., :3]
    request = np.asarray(requested_mask, dtype=bool)
    truth_defect = np.asarray(truth_defect_mask, dtype=bool)
    selected_part = np.asarray(selected_part_mask, dtype=bool)
    truth_part = np.asarray(truth_part_mask, dtype=bool)
    for mask in (request, truth_defect, selected_part, truth_part):
        if mask.shape != original_rgb.shape[:2]:
            raise ValueError("completion masks must match the source image size")

    intersection = int(np.count_nonzero(request & truth_defect))
    request_area = int(np.count_nonzero(request))
    truth_area = int(np.count_nonzero(truth_defect))
    outside = ~request
    changed_outside = np.any(completed_rgb != corrupted_rgb, axis=2) & outside
    visible_lock_exact = not bool(np.any(changed_outside))
    changed_outside_fraction = float(
        np.count_nonzero(changed_outside) / max(1, np.count_nonzero(outside))
    )
    if truth_area:
        difference = (
            completed_rgb[truth_defect].astype(np.float32)
            - original_rgb[truth_defect].astype(np.float32)
        )
        mae = float(np.mean(np.abs(difference)))
        mse = float(np.mean(difference**2))
        psnr = float(20.0 * np.log10(255.0 / np.sqrt(max(mse, 1e-12))))
    else:
        mae = 0.0
        psnr = float("inf")
    return CompletionHandoffMetrics(
        request_recall=float(intersection / max(1, truth_area)),
        request_precision=float(intersection / max(1, request_area)),
        target_mask_iou=_mask_iou(selected_part, truth_part),
        visible_lock_exact=visible_lock_exact,
        changed_outside_request_fraction=changed_outside_fraction,
        hidden_region_mae=mae,
        hidden_region_psnr=psnr,
    )
