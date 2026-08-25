from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .asset_routing import AssetRoute
from .fusion import MaskCandidate, mask_iou
from .prompt_bank import PromptBank
from .scene_instances import partition_scene_instances
from .scope_routing import extract_primary_foreground, route_extraction_scope
from .visual_regions import VisualMaskProposal


@dataclass(frozen=True)
class ProposalFirstConfig:
    root_mode: str = "primary"
    maximum_roots: int = 48
    minimum_primary_fraction: float = 0.008
    minimum_scene_fraction: float = 0.0008
    maximum_root_fraction: float = 0.92
    maximum_classified_proposals: int = 72
    minimum_scene_rootness: float = 0.34
    part_containment: float = 0.90
    maximum_parent_child_area_ratio: float = 12.0


@dataclass(frozen=True)
class ProposalFirstResult:
    roots: tuple[MaskCandidate, ...]
    proposals: tuple[VisualMaskProposal, ...]
    diagnostics: dict[str, object]


def route_proposal_first_execution(
    image: Image.Image,
    *,
    requested: bool,
    root_mode: str,
    target_point_xy: tuple[float, float] | None = None,
    minimum_isolated_fraction: float = 0.20,
    maximum_isolated_fraction: float = 0.50,
) -> tuple[bool, dict[str, object]]:
    """Choose one root-acquisition backend before loading either heavy model."""

    if not requested:
        return False, {
            "algorithm": "hpid-adaptive-root-backend-v1",
            "selected_backend": "detector_first",
            "reason": "proposal_first_not_requested",
            "ground_truth_used": False,
        }
    if root_mode == "scene":
        return True, {
            "algorithm": "hpid-adaptive-root-backend-v1",
            "selected_backend": "proposal_first",
            "reason": "scene_reuses_global_proposals",
            "ground_truth_used": False,
        }
    if target_point_xy is not None:
        return True, {
            "algorithm": "hpid-adaptive-root-backend-v1",
            "selected_backend": "proposal_first",
            "reason": "explicit_target_point",
            "ground_truth_used": False,
        }
    foreground = extract_primary_foreground(image)
    output_fraction = float(
        foreground.diagnostics.get(
            "output_fraction",
            foreground.diagnostics.get("foreground_fraction", 0.0),
        )
    )
    use_proposal_first = bool(
        foreground.mask is not None
        and minimum_isolated_fraction <= output_fraction <= maximum_isolated_fraction
    )
    return use_proposal_first, {
        "algorithm": "hpid-adaptive-root-backend-v1",
        "selected_backend": (
            "proposal_first" if use_proposal_first else "detector_first"
        ),
        "reason": (
            "reliable_isolated_foreground"
            if use_proposal_first
            else "nonisolated_or_tight_crop"
        ),
        "minimum_isolated_fraction": minimum_isolated_fraction,
        "maximum_isolated_fraction": maximum_isolated_fraction,
        "foreground_preflight": foreground.diagnostics,
        "ground_truth_used": False,
    }


def _area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def _box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _border_touches(mask: np.ndarray) -> int:
    return sum(
        (
            bool(np.any(mask[0])),
            bool(np.any(mask[-1])),
            bool(np.any(mask[:, 0])),
            bool(np.any(mask[:, -1])),
        )
    )


def _saliency_contrast(lab: np.ndarray, mask: np.ndarray) -> float:
    height, width = mask.shape
    band = max(2, round(min(height, width) * 0.025))
    frame = np.zeros_like(mask)
    frame[:band] = True
    frame[-band:] = True
    frame[:, :band] = True
    frame[:, -band:] = True
    outside = frame & ~mask
    if _area(mask) < 8 or _area(outside) < 8:
        return 0.0
    inside_median = np.median(lab[mask], axis=0)
    outside_median = np.median(lab[outside], axis=0)
    return float(
        np.clip(np.linalg.norm(inside_median - outside_median) / 105.0, 0.0, 1.0)
    )


def _compactness(mask: np.ndarray) -> float:
    x0, y0, x1, y1 = _box(mask)
    box_area = max(1, (x1 - x0) * (y1 - y0))
    return float(np.clip(_area(mask) / box_area, 0.0, 1.0))


def _area_prior(fraction: float, *, scene: bool) -> float:
    if scene:
        return float(np.clip(np.sqrt(fraction / 0.08), 0.0, 1.0))
    distance = abs(float(np.log(max(1e-6, fraction) / 0.32)))
    return float(np.clip(1.0 - distance / 3.2, 0.0, 1.0))


def _center_support(mask: np.ndarray, point: tuple[float, float] | None) -> float:
    height, width = mask.shape
    if point is None:
        x, y = width // 2, height // 2
    else:
        x = int(np.clip(round(point[0]), 0, width - 1))
        y = int(np.clip(round(point[1]), 0, height - 1))
    if mask[y, x]:
        return 1.0
    distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    diagonal = max(1.0, float(np.hypot(height, width)))
    return float(np.clip(1.0 - distance[y, x] / (0.22 * diagonal), 0.0, 1.0))


def _nested_region_support(
    parent: VisualMaskProposal,
    proposals: list[VisualMaskProposal],
) -> tuple[float, int]:
    """Measure whether a root explains several independently proposed subregions."""

    parent_mask = parent.mask.astype(bool)
    parent_area = max(1, _area(parent_mask))
    evidence = 0.0
    count = 0
    for child in proposals:
        if child is parent:
            continue
        child_mask = child.mask.astype(bool)
        child_area = _area(child_mask)
        fraction = child_area / parent_area
        if child_area < 24 or not 0.002 <= fraction <= 0.72:
            continue
        containment = _area(child_mask & parent_mask) / max(1, child_area)
        if containment < 0.88:
            continue
        count += 1
        evidence += float(child.score) * min(1.0, np.sqrt(fraction) * 2.8)
    return float(np.clip(evidence / 1.8, 0.0, 1.0)), count


def _primary_root_quality(
    selected: list[tuple[float, VisualMaskProposal, dict[str, object], str, float, float]],
    rows: list[dict[str, object]],
    foreground_diagnostics: dict[str, object],
) -> dict[str, object]:
    if len(selected) != 1:
        return {
            "status": "not_applicable",
            "fallback_recommended": False,
            "reasons": [],
        }
    row = selected[0][2]
    fraction = float(row.get("area_fraction", 0.0))
    contrast = float(row.get("saliency_contrast", 0.0))
    boundary = float(row.get("boundary_alignment", 0.0))
    border_touches = int(row.get("border_touches", 0))
    larger_structured_candidate = any(
        candidate is not row
        and candidate.get("rejection") is None
        and float(candidate.get("area_fraction", 0.0)) >= 0.25
        and int(candidate.get("nested_region_count", 0)) >= 3
        for candidate in rows
    )
    reasons: list[str] = []
    if fraction < 0.14 and larger_structured_candidate:
        reasons.append("selected_fragment_despite_larger_structured_candidate")
    if (
        fraction >= 0.32
        and border_touches >= 1
        and contrast < 0.08
        and boundary < 0.60
    ):
        reasons.append("selected_low_contrast_background_region")
    estimated_foreground_fraction = float(
        foreground_diagnostics.get("foreground_fraction", 0.0)
    )
    dominant_component_fraction = float(
        foreground_diagnostics.get("dominant_component_fraction", 0.0)
    )
    dominant_foreground = bool(
        foreground_diagnostics.get("uniform_border", False)
        and dominant_component_fraction >= 0.82
    )
    foreground_coverage_ratio = fraction / max(1e-6, estimated_foreground_fraction)
    if (
        estimated_foreground_fraction >= 0.12
        and dominant_component_fraction >= 0.72
        and fraction < 0.30
        and foreground_coverage_ratio < 0.58
    ):
        reasons.append("selected_root_under_covers_foreground_envelope")
    if (
        estimated_foreground_fraction >= 0.08
        and fraction >= 0.28
        and border_touches >= 2
        and foreground_coverage_ratio > 1.70
    ):
        reasons.append("selected_root_over_covers_foreground_envelope")
    return {
        "status": "fallback_recommended" if reasons else "accepted",
        "fallback_recommended": bool(reasons),
        "reasons": reasons,
        "selected_area_fraction": fraction,
        "selected_saliency_contrast": contrast,
        "selected_boundary_alignment": boundary,
        "selected_border_touches": border_touches,
        "larger_structured_candidate": larger_structured_candidate,
        "estimated_foreground_fraction": estimated_foreground_fraction,
        "foreground_coverage_ratio": float(foreground_coverage_ratio),
        "dominant_foreground": dominant_foreground,
        "dominant_component_fraction": dominant_component_fraction,
        "ground_truth_used": False,
    }


def _foreground_from_background(
    background: np.ndarray,
    *,
    target_point_xy: tuple[float, float] | None,
) -> np.ndarray | None:
    background = background.astype(bool)
    foreground = ~background
    height, width = foreground.shape
    minimum_area = max(32, round(height * width * 0.00035))
    labels: np.ndarray | None = None
    candidates: list[tuple[float, int]] = []
    enclosed_candidates: list[tuple[float, int]] = []

    def collect(candidate_foreground: np.ndarray) -> None:
        nonlocal labels, candidates, enclosed_candidates
        count, current_labels, stats, _ = cv2.connectedComponentsWithStats(
            candidate_foreground.astype(np.uint8), connectivity=8
        )
        current_candidates: list[tuple[float, int]] = []
        current_enclosed: list[tuple[float, int]] = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < minimum_area:
                continue
            component = current_labels == label
            touches_frame = _border_touches(component) > 0
            center = _center_support(component, target_point_xy)
            score = area * (0.78 + 0.22 * center)
            current_candidates.append((score, label))
            if not touches_frame:
                current_enclosed.append((score, label))
        labels = current_labels
        candidates = current_candidates
        enclosed_candidates = current_enclosed

    collect(foreground)
    if not enclosed_candidates:
        # Object-surrounding SAM backgrounds can retain a narrow opening near
        # an eave, wheel, or handle.  Close only such small gaps, then recover
        # the large enclosed component.  The caller still requires nested
        # independent proposals before treating this as a scene object.
        radius = int(np.clip(round(np.hypot(height, width) * 0.010), 3, 28))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        panel_background = background.copy()
        ys, xs = np.nonzero(background)
        if len(xs):
            x0, x1 = int(xs.min()), int(xs.max() + 1)
            y0, y1 = int(ys.min()), int(ys.max() + 1)
            span_x = max(1, x1 - x0)
            span_y = max(1, y1 - y0)
            x0 = 0 if x0 <= 0.12 * width else max(0, x0 - round(0.01 * span_x))
            x1 = (
                width
                if width - x1 <= 0.12 * width
                else min(width, x1 + round(0.01 * span_x))
            )
            y0 = 0 if y0 <= 0.12 * height else max(0, y0 - round(0.01 * span_y))
            y1 = (
                height
                if height - y1 <= 0.12 * height
                else min(height, y1 + round(0.01 * span_y))
            )
            panel_background[:y0] = True
            panel_background[y1:] = True
            panel_background[:, :x0] = True
            panel_background[:, x1:] = True
            rim = max(2, min(10, radius // 3))
            panel_background[y0 : y0 + rim, x0:x1] = True
            panel_background[y1 - rim : y1, x0:x1] = True
            panel_background[y0:y1, x0 : x0 + rim] = True
            panel_background[y0:y1, x1 - rim : x1] = True
        closed_background = cv2.morphologyEx(
            panel_background.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)
        collect(~closed_background)
    if not candidates:
        return None
    assert labels is not None
    selected_label = max(enclosed_candidates or candidates)[1]
    selected = labels == selected_label
    radius = int(np.clip(round(np.hypot(height, width) * 0.0025), 1, 4))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    closed = cv2.morphologyEx(
        selected.astype(np.uint8), cv2.MORPH_CLOSE, kernel
    ).astype(bool)
    fraction = _area(closed) / max(1, height * width)
    return closed if 0.008 <= fraction <= 0.78 else None


def _is_surrounding_scene_background(
    background: np.ndarray,
    foreground: np.ndarray | None,
    *,
    compactness: float,
    proposals: list[VisualMaskProposal],
) -> bool:
    """Identify a panel background that surrounds one complete scene object."""

    if foreground is None or compactness > 0.72:
        return False
    image_area = max(1, background.size)
    background_fraction = _area(background) / image_area
    foreground_fraction = _area(foreground) / image_area
    if not (
        0.06 <= background_fraction <= 0.58
        and 0.025 <= foreground_fraction <= 0.52
    ):
        return False
    if _border_touches(foreground) > 0:
        return False
    foreground_proposal = VisualMaskProposal(foreground, 1.0)
    _support, nested_count = _nested_region_support(
        foreground_proposal,
        proposals,
    )
    return nested_count >= 2


def _box_iou(first: np.ndarray, second: np.ndarray) -> float:
    ax0, ay0, ax1, ay1 = _box(first)
    bx0, by0, bx1, by1 = _box(second)
    intersection = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(
        0, min(ay1, by1) - max(ay0, by0)
    )
    union = max(1, (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersection)
    return float(intersection / union)


def _scene_object_envelope_quality(row: dict[str, object]) -> float:
    compactness = float(row.get("compactness", 0.0))
    area_fraction = float(row.get("area_fraction", 0.0))
    shape_support = float(np.clip(1.0 - abs(compactness - 0.64) / 0.64, 0.0, 1.0))
    panel_fill_penalty = float(
        np.clip((compactness - 0.88) / 0.12, 0.0, 1.0)
        * np.clip((area_fraction - 0.24) / 0.18, 0.0, 1.0)
    )
    return float(
        float(row.get("rootness", 0.0))
        + 0.08 * float(row.get("saliency_contrast", 0.0))
        + 0.07 * shape_support
        - 0.24 * panel_fill_penalty
    )


def _prune_duplicate_scene_object_envelopes(
    viable: list[tuple[float, VisualMaskProposal, dict[str, object]]],
) -> tuple[
    list[tuple[float, VisualMaskProposal, dict[str, object]]],
    list[dict[str, object]],
]:
    envelopes = [
        (index, item)
        for index, item in enumerate(viable)
        if bool(item[2].get("derived_scene_object_envelope"))
    ]
    rejected: set[int] = set()
    rows: list[dict[str, object]] = []
    for offset, (first_index, first) in enumerate(envelopes):
        if first_index in rejected:
            continue
        for second_index, second in envelopes[offset + 1 :]:
            if second_index in rejected:
                continue
            overlap = _box_iou(first[1].mask, second[1].mask)
            if overlap < 0.68:
                continue
            first_quality = _scene_object_envelope_quality(first[2])
            second_quality = _scene_object_envelope_quality(second[2])
            if (first_quality, -_area(first[1].mask)) >= (
                second_quality,
                -_area(second[1].mask),
            ):
                removed_index = second_index
                kept, removed = first, second
            else:
                removed_index = first_index
                kept, removed = second, first
            rejected.add(removed_index)
            removed[2]["rejection"] = "duplicate_scene_object_envelope"
            rows.append(
                {
                    "kept_proposal_index": kept[2].get("proposal_index"),
                    "removed_proposal_index": removed[2].get("proposal_index"),
                    "bbox_iou": overlap,
                    "kept_quality": _scene_object_envelope_quality(kept[2]),
                    "removed_quality": _scene_object_envelope_quality(removed[2]),
                }
            )
            if removed_index == first_index:
                break
    return [item for index, item in enumerate(viable) if index not in rejected], rows


def _classify_domains(
    image: Image.Image,
    proposals: list[VisualMaskProposal],
    prompt_bank: PromptBank,
    dense_proposer: Any | None,
) -> tuple[list[tuple[str, float, float]], dict[str, object]]:
    domains = list(prompt_bank.domains)
    if not domains:
        raise ValueError("proposal-first decomposition requires at least one domain")
    if dense_proposer is None:
        fallback = domains[0].name
        return (
            [(fallback, 0.0, 0.0) for _ in proposals],
            {
                "algorithm": "proposal-first-domain-fallback-v1",
                "reason": "dense_semantic_backend_unavailable",
                "ground_truth_used": False,
            },
        )
    regions = [
        (f"proposal:{index}", proposal.mask.astype(bool))
        for index, proposal in enumerate(proposals)
    ]
    labels = [
        (
            domain.name,
            domain.classifier_prompt
            or f"one {domain.name.replace('_', ' ')} object",
        )
        for domain in domains
    ]
    rankings = dense_proposer.rank_regions_labels(
        image,
        regions,
        labels,
        masked_weight=0.90,
    )
    classified: list[tuple[str, float, float]] = []
    rows: list[dict[str, object]] = []
    for index in range(len(proposals)):
        key = f"proposal:{index}"
        scores = rankings[key]
        ordered = sorted(
            scores.items(),
            key=lambda item: float(item[1]["combined_similarity"]),
            reverse=True,
        )
        best_name, best = ordered[0]
        margin = float(best["combined_similarity"]) - float(
            ordered[1][1]["combined_similarity"] if len(ordered) > 1 else 0.0
        )
        classified.append((best_name, float(best["probability"]), margin))
        rows.append(
            {
                "proposal_index": index,
                "domain": best_name,
                "probability": float(best["probability"]),
                "margin": margin,
            }
        )
    return classified, {
        "algorithm": "clipseg-masked-proposal-domain-routing-v1",
        "proposal_count": len(proposals),
        "domain_count": len(domains),
        "rows": rows,
        "ground_truth_used": False,
    }


def generate_proposal_first_roots(
    image: Image.Image,
    proposals: list[VisualMaskProposal],
    prompt_bank: PromptBank,
    dense_proposer: Any | None,
    *,
    preferred_route: AssetRoute | None = None,
    target_point_xy: tuple[float, float] | None = None,
    config: ProposalFirstConfig | None = None,
) -> ProposalFirstResult:
    """Choose asset roots from one reusable image-driven proposal pool."""

    config = config or ProposalFirstConfig()
    if config.root_mode not in {"primary", "scene"}:
        raise ValueError("proposal-first root mode must be primary or scene")
    image = image.convert("RGB")
    image_area = max(1, image.width * image.height)
    lab = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2LAB).astype(np.float32)
    scene = config.root_mode == "scene"
    border_foreground = (
        extract_primary_foreground(image) if not scene else None
    )
    proposal_pool = list(proposals)
    border_foreground_index: int | None = None
    if border_foreground is not None and border_foreground.mask is not None:
        border_foreground_index = len(proposal_pool)
        proposal_pool.append(
            VisualMaskProposal(
                mask=border_foreground.mask,
                score=0.90,
                bbox_xyxy=_box(border_foreground.mask),
                scale_level=-10,
                view_id="border-foreground",
                support_views=("border-background",),
                support_levels=(-10,),
                boundary_alignment=0.72,
                source="hpid-border-foreground/grabcut",
                geometric_support=0.82,
            )
        )
    minimum_fraction = (
        config.minimum_scene_fraction if scene else config.minimum_primary_fraction
    )
    rows: list[dict[str, object]] = []
    viable: list[tuple[float, VisualMaskProposal, dict[str, object]]] = []
    foreground_complements: list[tuple[int, VisualMaskProposal]] = []
    for proposal_index, proposal in enumerate(proposal_pool):
        mask = proposal.mask.astype(bool)
        area = _area(mask)
        fraction = area / image_area
        touches = _border_touches(mask)
        contrast = _saliency_contrast(lab, mask)
        compactness = _compactness(mask)
        boundary = float(proposal.boundary_alignment)
        center = _center_support(mask, target_point_xy)
        rootness = float(
            np.clip(
                0.36 * proposal.score
                + 0.18 * boundary
                + 0.16 * contrast
                + 0.12 * compactness
                + 0.12 * _area_prior(fraction, scene=scene)
                + 0.06 * center,
                0.0,
                1.0,
            )
        )
        rejection = None
        row_foreground: float | None = None
        if area < 32 or not minimum_fraction <= fraction <= config.maximum_root_fraction:
            rejection = "root_scale"
        else:
            primary_background = (
                not scene
                and touches >= 3
                and fraction >= 0.28
                and contrast < 0.18
            )
            scene_canvas_background = (
                scene
                and touches == 4
                and 0.32 <= fraction <= 0.82
                and contrast < 0.12
                and compactness < 0.88
            )
            possible_scene_foreground = (
                _foreground_from_background(
                    mask,
                    target_point_xy=target_point_xy,
                )
                if scene and 0.06 <= fraction <= 0.58 and compactness <= 0.72
                else None
            )
            surrounding_scene_background = bool(
                scene
                and _is_surrounding_scene_background(
                    mask,
                    possible_scene_foreground,
                    compactness=compactness,
                    proposals=proposals,
                )
            )
            background_kind = (
                "background_like_border_surface"
                if primary_background
                else (
                    "background_surrounding_scene_object"
                    if surrounding_scene_background
                    else "background_like_scene_canvas"
                )
            )
            foreground = _foreground_from_background(
                mask,
                target_point_xy=target_point_xy,
            ) if (
                primary_background
                or scene_canvas_background
                or surrounding_scene_background
            ) else None
            if foreground is not None and (
                primary_background
                or surrounding_scene_background
                or _nested_region_support(
                    VisualMaskProposal(foreground, proposal.score), proposals
                )[1]
                >= 2
            ):
                rejection = background_kind
                foreground_complements.append(
                    (
                        proposal_index,
                        VisualMaskProposal(
                            mask=foreground,
                            score=float(np.clip(0.96 * proposal.score, 0.0, 1.0)),
                            bbox_xyxy=_box(foreground),
                            scale_level=proposal.scale_level,
                            view_id=f"{proposal.view_id}-foreground-complement",
                            support_views=proposal.support_views,
                            support_levels=proposal.support_levels,
                            best_view_iou=proposal.best_view_iou,
                            boundary_alignment=proposal.boundary_alignment,
                        ),
                    )
                )
                row_foreground = _area(foreground) / image_area
        row = {
            "proposal_index": proposal_index,
            "area_fraction": fraction,
            "sam_score": float(proposal.score),
            "boundary_alignment": boundary,
            "saliency_contrast": contrast,
            "compactness": compactness,
            "center_support": center,
            "border_touches": touches,
            "rootness": rootness,
            "rejection": rejection,
            "derived_from_border_foreground": (
                proposal_index == border_foreground_index
            ),
            "derived_scene_object_envelope": bool(
                rejection == "background_surrounding_scene_object"
            ),
        }
        if rejection in {
            "background_like_border_surface",
            "background_like_scene_canvas",
        }:
            row["foreground_complement_fraction"] = row_foreground
        rows.append(row)
        if rejection is None:
            viable.append((rootness, proposal, row))
    for source_index, proposal in foreground_complements:
        mask = proposal.mask.astype(bool)
        fraction = _area(mask) / image_area
        contrast = _saliency_contrast(lab, mask)
        compactness = _compactness(mask)
        center = _center_support(mask, target_point_xy)
        rootness = float(
            np.clip(
                0.34 * proposal.score
                + 0.20 * float(proposal.boundary_alignment)
                + 0.18 * contrast
                + 0.08 * compactness
                + 0.10 * _area_prior(fraction, scene=False)
                + 0.10 * center,
                0.0,
                1.0,
            )
        )
        row = {
            "proposal_index": f"background-complement:{source_index}",
            "derived_from_proposal_index": source_index,
            "derived_from_background_complement": True,
            "derived_scene_object_envelope": bool(
                rows[source_index].get("derived_scene_object_envelope", False)
            ),
            "area_fraction": fraction,
            "sam_score": float(proposal.score),
            "boundary_alignment": float(proposal.boundary_alignment),
            "saliency_contrast": contrast,
            "compactness": compactness,
            "center_support": center,
            "border_touches": _border_touches(mask),
            "rootness": rootness,
            "rejection": None,
        }
        rows.append(row)
        viable.append((rootness, proposal, row))
    augmented_proposals = list(proposal_pool)
    scene_instance_diagnostics: dict[str, object] | None = None
    if scene:
        viable, envelope_deduplication = _prune_duplicate_scene_object_envelopes(
            viable
        )
        envelope_items = [
            item
            for item in viable
            if bool(item[2].get("derived_from_background_complement"))
            and 0.24 <= _area(item[1].mask) / image_area <= 0.94
            and _border_touches(item[1].mask.astype(bool)) <= 1
        ]
        if envelope_items:
            envelope_item = max(envelope_items, key=lambda item: _area(item[1].mask))
            envelope_item[2]["scene_instance_envelope_selected"] = True
            seed_items = [
                item
                for item in viable
                if item is not envelope_item
                and not bool(item[2].get("derived_from_background_complement"))
            ]
            instance_result = partition_scene_instances(
                image,
                envelope_item[1],
                [item[1] for item in seed_items],
            )
            scene_instance_diagnostics = instance_result.diagnostics
            if instance_result.partitions:
                partition_union = np.zeros(
                    envelope_item[1].mask.shape, dtype=bool
                )
                for partition in instance_result.partitions:
                    partition_union |= partition.proposal.mask.astype(bool)
                replaced_items: set[int] = set()
                for seed_index, (_, seed, row) in enumerate(seed_items):
                    coverage = _area(seed.mask.astype(bool) & partition_union) / max(
                        1, _area(seed.mask)
                    )
                    if (
                        seed_index in instance_result.replaced_seed_indices
                        or coverage >= 0.70
                    ):
                        replaced_items.add(id(seed))
                        row["rejection"] = "replaced_by_scene_instance_partition"
                        row["scene_partition_coverage"] = float(coverage)
                viable = [
                    item for item in viable if id(item[1]) not in replaced_items
                ]
                for partition_index, partition in enumerate(
                    instance_result.partitions, start=1
                ):
                    seed_row = (
                        seed_items[partition.seed_index][2]
                        if partition.seed_index is not None
                        else None
                    )
                    proposal = partition.proposal
                    rootness = float(
                        np.clip(0.42 + 0.50 * proposal.score, 0.0, 0.94)
                    )
                    row = {
                        "proposal_index": f"scene-instance:{partition_index}",
                        "derived_from_proposal_index": (
                            seed_row.get("proposal_index")
                            if seed_row is not None
                            else None
                        ),
                        "scene_instance_partition": True,
                        "area_fraction": _area(proposal.mask) / image_area,
                        "sam_score": float(proposal.score),
                        "boundary_alignment": float(
                            proposal.boundary_alignment
                        ),
                        "saliency_contrast": _saliency_contrast(
                            lab, proposal.mask.astype(bool)
                        ),
                        "compactness": _compactness(proposal.mask.astype(bool)),
                        "center_support": _center_support(
                            proposal.mask.astype(bool), target_point_xy
                        ),
                        "border_touches": _border_touches(
                            proposal.mask.astype(bool)
                        ),
                        "rootness": rootness,
                        "rejection": None,
                    }
                    rows.append(row)
                    viable.append((rootness, proposal, row))
                    augmented_proposals.append(proposal)
    else:
        envelope_deduplication = []
    viable.sort(key=lambda item: (item[0], _area(item[1].mask)), reverse=True)
    viable = viable[: config.maximum_classified_proposals]
    if not viable:
        raise RuntimeError("SAM2 did not produce a plausible asset root")

    classifications, classification_diagnostics = _classify_domains(
        image,
        [item[1] for item in viable],
        prompt_bank,
        dense_proposer,
    )
    enriched: list[
        tuple[float, VisualMaskProposal, dict[str, object], str, float, float]
    ] = []
    preferred_primary = bool(
        not scene
        and preferred_route is not None
        and preferred_route.accepted
        and preferred_route.asset_domain
    )
    for item, classification in zip(viable, classifications, strict=True):
        rootness, proposal, row = item
        domain, probability, margin = classification
        if preferred_primary:
            domain = str(preferred_route.asset_domain)
        row["classified_domain"] = domain
        row["domain_probability"] = probability
        row["domain_margin"] = margin
        enriched.append((rootness, proposal, row, domain, probability, margin))

    selected: list[
        tuple[float, VisualMaskProposal, dict[str, object], str, float, float]
    ] = []
    if not scene:
        if target_point_xy is not None:
            x = int(np.clip(round(target_point_xy[0]), 0, image.width - 1))
            y = int(np.clip(round(target_point_xy[1]), 0, image.height - 1))
            containing = [item for item in enriched if item[1].mask[y, x]]
        else:
            containing = []
        pool = containing or enriched
        for item in pool:
            nested_support, nested_count = _nested_region_support(
                item[1], [candidate[1] for candidate in enriched]
            )
            fraction = _area(item[1].mask) / image_area
            item[2]["nested_region_support"] = nested_support
            item[2]["nested_region_count"] = nested_count
            item[2]["complete_extent_support"] = float(
                np.clip(np.sqrt(fraction / 0.45), 0.0, 1.0)
            )
            item[2]["primary_selection_score"] = float(
                item[0]
                + 0.08 * _area_prior(fraction, scene=False)
                + 0.14 * nested_support
                + 0.045 * min(nested_count, 5)
                + 0.14 * float(item[2]["complete_extent_support"])
                + (
                    0.18
                    if bool(item[2].get("derived_from_background_complement"))
                    and nested_count >= 2
                    else 0.0
                )
                + 0.08
                * float(item[2].get("derived_from_border_foreground", False))
            )
        selected = [
            max(
                pool,
                key=lambda item: (
                    float(item[2]["primary_selection_score"]),
                    _area(item[1].mask),
                ),
            )
        ]
    else:
        suppressed_parts: set[int] = set()
        for child_index, child in enumerate(enriched):
            if bool(child[2].get("scene_instance_partition")):
                continue
            child_area = max(1, _area(child[1].mask))
            for parent_index, parent in enumerate(enriched):
                if parent_index == child_index:
                    continue
                parent_area = _area(parent[1].mask)
                if parent_area <= child_area:
                    continue
                ratio = parent_area / child_area
                if ratio > config.maximum_parent_child_area_ratio:
                    continue
                containment = _area(child[1].mask & parent[1].mask) / child_area
                parent_fraction = parent_area / image_area
                parent_is_layer = (
                    parent[3] in {"terrain", "structure"} and parent_fraction >= 0.24
                ) or parent_fraction >= 0.62 or bool(
                    parent[2].get("derived_from_background_complement")
                    and not parent[2].get("derived_scene_object_envelope")
                )
                if (
                    containment >= config.part_containment
                    and not parent_is_layer
                    and parent[0] >= child[0] - 0.08
                ):
                    suppressed_parts.add(child_index)
                    child[2]["rejection"] = "contained_part_not_scene_root"
                    break
        for index, item in enumerate(enriched):
            if index in suppressed_parts or item[0] < config.minimum_scene_rootness:
                continue
            if any(mask_iou(item[1].mask, kept[1].mask) >= 0.66 for kept in selected):
                item[2]["rejection"] = "duplicate_scene_root"
                continue
            selected.append(item)
            if len(selected) >= config.maximum_roots:
                break
    if not selected:
        selected = [enriched[0]]
        selected[0][2]["fallback_selected"] = True

    domain_by_name = {domain.name: domain for domain in prompt_bank.domains}
    selected.sort(key=lambda item: (_box(item[1].mask)[1], _box(item[1].mask)[0]))
    roots: list[MaskCandidate] = []
    for root_index, (rootness, proposal, row, domain_name, probability, margin) in enumerate(
        selected, start=1
    ):
        domain = domain_by_name.get(domain_name) or prompt_bank.domains[0]
        profile = (
            preferred_route.asset_profile
            if preferred_primary and preferred_route is not None
            else None
        )
        label = (
            str(preferred_route.asset_label)
            if preferred_primary
            and preferred_route is not None
            and preferred_route.asset_label
            else domain.name.replace("_", " ")
        )
        roots.append(
            MaskCandidate(
                semantic_name=domain.name,
                semantic_parent=domain.name,
                mask=proposal.mask.astype(bool),
                score=rootness,
                source="hpid-proposal-first/sam2-root",
                prompt=label,
                source_reliability=0.78 + 0.18 * float(proposal.score),
                metadata={
                    "source_family": "hpid-proposal-first-v2",
                    "root_origin": "hpid-proposal-first[sam2-amg]",
                    "root_index": root_index,
                    "candidate_key": f"root:{root_index}",
                    "parent_candidate_key": None,
                    "sam_quality": float(proposal.score),
                    "box_xyxy": list(proposal.bbox_xyxy or _box(proposal.mask)),
                    "root_label_specificity": domain.root_label_specificity(label),
                    "part_profile_specificity": (
                        1.0 if profile is not None else 0.0
                    ),
                    "selected_part_profile": profile,
                    "root_query_mode": "proposal_first_cross_cue",
                    "root_model_label": label,
                    "profile_hint_source": (
                        "global_asset_proposal" if profile is not None else None
                    ),
                    "profile_resolution_status": (
                        "accepted" if profile is not None else None
                    ),
                    "domain_evidence_score": probability,
                    "domain_evidence_contrast": margin,
                    "proposal_first_evidence": row,
                    "atomic_scene_instance": bool(
                        row.get("scene_instance_partition")
                        or row.get("derived_scene_object_envelope")
                    ),
                    "ground_truth_used": False,
                },
            )
        )
        row["selected_root_index"] = root_index

    primary_root_quality = (
        _primary_root_quality(
            selected,
            rows,
            route_extraction_scope(image, "Entire scene").diagnostics,
        )
        if not scene
        else {
            "status": "not_applicable",
            "fallback_recommended": False,
            "reasons": [],
            "ground_truth_used": False,
        }
    )
    return ProposalFirstResult(
        tuple(roots),
        tuple(augmented_proposals),
        {
            "algorithm": "hpid-proposal-first-cross-cue-roots-v2",
            "root_mode": config.root_mode,
            "input_proposal_count": len(proposals),
            "border_foreground": (
                border_foreground.diagnostics
                if border_foreground is not None
                else None
            ),
            "viable_proposal_count": len(viable),
            "selected_root_count": len(roots),
            "preferred_route_used": preferred_primary,
            "proposal_rows": rows,
            "domain_classification": classification_diagnostics,
            "scene_instance_partition": scene_instance_diagnostics,
            "scene_object_envelope_deduplication": envelope_deduplication,
            "primary_root_quality": primary_root_quality,
            "ground_truth_used": False,
        },
    )
