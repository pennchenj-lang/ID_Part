from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ScopeRoutingResult:
    requested_scope: str
    resolved_scope: str
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class ForegroundExtractionResult:
    mask: np.ndarray | None
    diagnostics: dict[str, object]


def _resize_for_analysis(image: Image.Image, maximum_dimension: int = 384) -> np.ndarray:
    image = image.convert("RGB")
    scale = min(1.0, maximum_dimension / max(image.width, image.height))
    width = max(32, round(image.width * scale))
    height = max(32, round(image.height * scale))
    return cv2.resize(
        np.asarray(image),
        (width, height),
        interpolation=cv2.INTER_AREA,
    )


def _border_connected_background(lab: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    height, width = lab.shape[:2]
    band = max(2, round(min(height, width) * 0.035))
    border = np.zeros((height, width), dtype=bool)
    border[:band] = True
    border[-band:] = True
    border[:, :band] = True
    border[:, -band:] = True
    pixels = lab[border]
    center = np.median(pixels, axis=0)
    distances = np.linalg.norm(pixels - center, axis=1)
    q50, q80, q95 = (float(np.quantile(distances, value)) for value in (0.5, 0.8, 0.95))
    threshold = float(np.clip(max(10.0, q80 + 1.35 * (q95 - q50)), 10.0, 42.0))
    similar = np.linalg.norm(lab - center, axis=2) <= threshold

    count, labels = cv2.connectedComponents(similar.astype(np.uint8), connectivity=8)
    connected = np.zeros_like(similar)
    for label in np.unique(labels[border]):
        if label > 0 and label < count:
            connected |= labels == label
    return connected, {
        "border_distance_q50": q50,
        "border_distance_q80": q80,
        "border_distance_q95": q95,
        "background_threshold": threshold,
    }


def extract_primary_foreground(
    image: Image.Image,
    *,
    maximum_dimension: int = 640,
) -> ForegroundExtractionResult:
    """Extract one isolated foreground asset without category-specific prompts."""

    source = image.convert("RGB")
    rgb = _resize_for_analysis(source, maximum_dimension)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    background, border_diagnostics = _border_connected_background(lab)
    foreground = ~background
    kernel = np.ones((3, 3), dtype=np.uint8)
    foreground = cv2.morphologyEx(
        foreground.astype(np.uint8), cv2.MORPH_OPEN, kernel
    ).astype(bool)
    foreground = cv2.morphologyEx(
        foreground.astype(np.uint8), cv2.MORPH_CLOSE, kernel
    ).astype(bool)
    image_area = max(1, foreground.size)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8), connectivity=8
    )
    components = sorted(
        (
            (int(stats[index, cv2.CC_STAT_AREA]), index)
            for index in range(1, count)
            if int(stats[index, cv2.CC_STAT_AREA])
            >= max(24, round(0.001 * image_area))
        ),
        reverse=True,
    )
    total_area = sum(area for area, _ in components)
    dominant_area = components[0][0] if components else 0
    dominant_fraction = dominant_area / max(1, total_area)
    foreground_fraction = dominant_area / image_area
    uniform_border = border_diagnostics["border_distance_q80"] <= 24.0
    reliable = bool(
        components
        and uniform_border
        and 0.025 <= foreground_fraction <= 0.72
        and dominant_fraction >= 0.72
    )
    diagnostics: dict[str, object] = {
        "algorithm": "hpid-border-foreground-v1",
        "status": "accepted" if reliable else "rejected",
        "analysis_size": [int(rgb.shape[1]), int(rgb.shape[0])],
        "foreground_fraction": float(foreground_fraction),
        "dominant_component_fraction": float(dominant_fraction),
        "component_count": len(components),
        "uniform_border": bool(uniform_border),
        **border_diagnostics,
        "ground_truth_used": False,
    }
    if not reliable:
        return ForegroundExtractionResult(None, diagnostics)

    selected = labels == components[0][1]
    grabcut_mask = np.full(selected.shape, cv2.GC_BGD, dtype=np.uint8)
    grabcut_mask[selected] = cv2.GC_PR_FGD
    core = cv2.erode(
        selected.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ).astype(bool)
    grabcut_mask[core] = cv2.GC_FGD
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            grabcut_mask,
            None,
            background_model,
            foreground_model,
            1,
            cv2.GC_INIT_WITH_MASK,
        )
        refined = np.isin(grabcut_mask, (cv2.GC_FGD, cv2.GC_PR_FGD))
        overlap = np.count_nonzero(refined & selected) / max(1, dominant_area)
        if overlap >= 0.82:
            selected = refined
            diagnostics["grabcut_refined"] = True
        else:
            diagnostics["grabcut_refined"] = False
            diagnostics["grabcut_rejection"] = "foreground_overlap"
    except cv2.error:
        diagnostics["grabcut_refined"] = False
        diagnostics["grabcut_rejection"] = "opencv_error"

    # GrabCut may revive disconnected border strokes that were excluded by the
    # initial component selection.  Keep the component that overlaps the
    # pre-GrabCut subject most strongly before returning the source-size mask.
    component_count, component_labels = cv2.connectedComponents(
        selected.astype(np.uint8), connectivity=8
    )
    if component_count > 2:
        selected_component = max(
            range(1, component_count),
            key=lambda index: int(
                np.count_nonzero((component_labels == index) & core)
            ),
        )
        removed_area = int(
            np.count_nonzero(selected & (component_labels != selected_component))
        )
        selected = component_labels == selected_component
        diagnostics["post_grabcut_component_cleanup"] = True
        diagnostics["post_grabcut_removed_area_px"] = removed_area
    else:
        diagnostics["post_grabcut_component_cleanup"] = False
    full_mask = cv2.resize(
        selected.astype(np.uint8),
        (source.width, source.height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    diagnostics["output_fraction"] = float(_area_fraction(full_mask))
    return ForegroundExtractionResult(full_mask, diagnostics)


def _area_fraction(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask) / max(1, mask.size))


def route_extraction_scope(
    image: Image.Image,
    requested_scope: str,
) -> ScopeRoutingResult:
    """Resolve a scene request to primary when the image has one clear subject.

    This is deliberately category independent. It uses only border-connected
    background evidence and foreground component geometry, so the same decision
    applies to characters, props, devices, and other isolated assets.
    """

    if requested_scope != "Entire scene":
        return ScopeRoutingResult(
            requested_scope,
            requested_scope,
            {
                "algorithm": "hpid-scope-router-v1",
                "status": "kept_explicit_scope",
                "ground_truth_used": False,
            },
        )

    rgb = _resize_for_analysis(image)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    background, border_diagnostics = _border_connected_background(lab)
    foreground = ~background
    kernel = np.ones((3, 3), dtype=np.uint8)
    foreground = cv2.morphologyEx(
        foreground.astype(np.uint8), cv2.MORPH_OPEN, kernel
    ).astype(bool)
    foreground = cv2.morphologyEx(
        foreground.astype(np.uint8), cv2.MORPH_CLOSE, kernel
    ).astype(bool)

    image_area = max(1, foreground.size)
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground.astype(np.uint8), connectivity=8
    )
    component_areas = sorted(
        (
            int(stats[index, cv2.CC_STAT_AREA])
            for index in range(1, count)
            if int(stats[index, cv2.CC_STAT_AREA]) >= max(18, round(0.001 * image_area))
        ),
        reverse=True,
    )
    foreground_area = sum(component_areas)
    foreground_fraction = foreground_area / image_area
    dominant_fraction = (
        component_areas[0] / max(1, foreground_area) if component_areas else 0.0
    )
    second_to_first = (
        component_areas[1] / max(1, component_areas[0])
        if len(component_areas) > 1
        else 0.0
    )
    uniform_border = border_diagnostics["border_distance_q80"] <= 24.0
    single_subject = bool(
        uniform_border
        and 0.035 <= foreground_fraction <= 0.46
        and dominant_fraction >= 0.88
        and second_to_first <= 0.16
    )
    resolved_scope = "Primary asset" if single_subject else requested_scope
    return ScopeRoutingResult(
        requested_scope,
        resolved_scope,
        {
            "algorithm": "hpid-scope-router-v1",
            "status": (
                "resolved_single_subject" if single_subject else "kept_scene_scope"
            ),
            "analysis_size": [int(rgb.shape[1]), int(rgb.shape[0])],
            "foreground_fraction": float(foreground_fraction),
            "dominant_component_fraction": float(dominant_fraction),
            "second_to_first_component_ratio": float(second_to_first),
            "component_count": len(component_areas),
            "uniform_border": bool(uniform_border),
            **border_diagnostics,
            "ground_truth_used": False,
        },
    )
