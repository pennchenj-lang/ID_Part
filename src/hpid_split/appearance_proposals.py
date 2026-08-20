from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .fusion import MaskCandidate, mask_iou
from .visual_regions import VisualMaskProposal


@dataclass(frozen=True)
class AppearanceProposalConfig:
    """Image-driven proposal settings shared by every supported asset domain."""

    analysis_maximum_dimension: int = 640
    graph_scales: tuple[float, ...] = (70.0, 150.0, 300.0)
    graph_sigma: float = 0.8
    graph_minimum_size: int = 28
    minimum_root_fraction: float = 0.0015
    maximum_root_fraction: float = 0.58
    maximum_regions_per_root: int = 28
    duplicate_iou: float = 0.82
    detail_region_limit: int = 4
    small_region_limit: int = 5
    medium_region_limit: int = 6
    large_region_limit: int = 3
    use_graph_regions: bool = True
    use_closed_contours: bool = True
    use_enclosed_interiors: bool = True


@dataclass(frozen=True)
class AppearanceProposalResult:
    proposals: tuple[VisualMaskProposal, ...]
    diagnostics: dict[str, object]


def _area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def _box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _root_key(root: MaskCandidate) -> str:
    return (
        f"{root.metadata.get('root_origin', 'legacy')}::"
        f"{root.metadata.get('root_index', 'unknown')}"
    )


def _analysis_size(width: int, height: int, maximum: int) -> tuple[int, int]:
    scale = min(1.0, maximum / max(width, height))
    return max(32, round(width * scale)), max(32, round(height * scale))


def _boundary_evidence(
    mask: np.ndarray,
    root: np.ndarray,
    lab: np.ndarray,
    gradient: np.ndarray,
    texture: np.ndarray,
) -> tuple[float, float, float, float]:
    radius = int(np.clip(round(np.sqrt(max(1, _area(root))) * 0.003), 1, 3))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    eroded = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    inner = mask & ~eroded
    outer = dilated & root & ~mask
    boundary = inner | outer
    reference_values = gradient[root]
    reference = (
        float(np.quantile(reference_values, 0.82))
        if reference_values.size
        else 0.0
    )
    boundary_values = gradient[boundary]
    alignment = float(
        np.clip(
            np.median(boundary_values) / max(14.0, reference)
            if boundary_values.size
            else 0.0,
            0.0,
            1.0,
        )
    )
    closure = float(
        np.mean(boundary_values >= max(12.0, 0.50 * reference))
        if boundary_values.size
        else 0.0
    )
    if not np.any(inner) or not np.any(outer):
        return alignment, closure, 0.0, 0.0
    inside_lab = np.median(lab[inner], axis=0)
    outside_lab = np.median(lab[outer], axis=0)
    chroma = float(
        np.clip(np.linalg.norm(inside_lab[1:] - outside_lab[1:]) / 58.0, 0.0, 1.0)
    )
    texture_delta = float(
        np.clip(
            abs(float(np.median(texture[inner]) - np.median(texture[outer]))) / 25.0,
            0.0,
            1.0,
        )
    )
    return alignment, closure, chroma, texture_delta


def _area_stratum(fraction: float) -> str:
    if fraction < 0.01:
        return "detail"
    if fraction < 0.05:
        return "small"
    if fraction < 0.20:
        return "medium"
    return "large"


def _proposal_fraction(proposal: VisualMaskProposal, root_area: int) -> float:
    return _area(proposal.mask) / max(1, root_area)


def _balanced_select(
    proposals: list[VisualMaskProposal],
    *,
    root_area: int,
    config: AppearanceProposalConfig,
) -> tuple[list[VisualMaskProposal], dict[str, int]]:
    """Retain structural regions without allowing tiny details to monopolize slots."""

    ordered = sorted(
        proposals,
        key=lambda proposal: (
            proposal.score + 0.05 * proposal.geometric_support,
            _area(proposal.mask),
        ),
        reverse=True,
    )
    unique: list[VisualMaskProposal] = []
    for proposal in ordered:
        if any(
            mask_iou(proposal.mask, incumbent.mask) >= config.duplicate_iou
            for incumbent in unique
        ):
            continue
        unique.append(proposal)

    limits = {
        "detail": config.detail_region_limit,
        "small": config.small_region_limit,
        "medium": config.medium_region_limit,
        "large": config.large_region_limit,
    }
    selected: list[VisualMaskProposal] = []
    counts = {key: 0 for key in limits}
    for stratum in ("large", "medium", "small", "detail"):
        candidates = [
            proposal
            for proposal in unique
            if _area_stratum(_proposal_fraction(proposal, root_area)) == stratum
        ]
        for proposal in candidates[: limits[stratum]]:
            if len(selected) >= config.maximum_regions_per_root:
                break
            selected.append(proposal)
            counts[stratum] += 1

    if len(selected) < config.maximum_regions_per_root:
        selected_ids = {id(proposal) for proposal in selected}
        for proposal in unique:
            if id(proposal) in selected_ids:
                continue
            selected.append(proposal)
            stratum = _area_stratum(_proposal_fraction(proposal, root_area))
            counts[stratum] += 1
            if len(selected) >= config.maximum_regions_per_root:
                break
    return selected, counts


def _closed_contour_proposals(
    *,
    root_mask: np.ndarray,
    root_key: str,
    gray: np.ndarray,
    lab: np.ndarray,
    gradient: np.ndarray,
    texture: np.ndarray,
    output_size: tuple[int, int],
    config: AppearanceProposalConfig,
) -> tuple[list[VisualMaskProposal], list[dict[str, object]]]:
    """Find closed structural faces such as screens and mechanical panels."""

    root_area = max(1, _area(root_mask))
    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    smooth = cv2.GaussianBlur(gray_u8, (3, 3), 0.7)
    root_values = gradient[root_mask]
    reference = float(np.quantile(root_values, 0.72)) if root_values.size else 30.0
    low = int(np.clip(0.45 * reference, 18, 72))
    high = int(np.clip(1.25 * reference, low + 20, 180))
    edges = cv2.Canny(smooth, low, high)
    edges[~cv2.dilate(root_mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)] = 0
    proposals: list[VisualMaskProposal] = []
    rows: list[dict[str, object]] = []
    output_width, output_height = output_size
    for close_size in (3, 5, 7):
        closed_edges = cv2.morphologyEx(
            edges,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (close_size, close_size)
            ),
            iterations=1,
        )
        contours, _ = cv2.findContours(
            closed_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            contour_area = float(cv2.contourArea(contour))
            fraction = contour_area / root_area
            if not (
                max(0.008, config.minimum_root_fraction) <= fraction
                <= config.maximum_root_fraction
            ):
                continue
            perimeter = float(cv2.arcLength(contour, True))
            if perimeter < 16.0:
                continue
            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
            rectangle = cv2.minAreaRect(contour)
            rectangle_area = float(rectangle[1][0] * rectangle[1][1])
            solidity = contour_area / max(1.0, hull_area)
            rectangularity = contour_area / max(1.0, rectangle_area)
            vertices = len(cv2.approxPolyDP(contour, 0.018 * perimeter, True))
            shape_support = float(
                np.clip(0.55 * rectangularity + 0.45 * solidity, 0.0, 1.0)
            )
            if vertices > 18 or shape_support < 0.58:
                continue
            contour_mask = np.zeros_like(root_mask, dtype=np.uint8)
            convex_regularized = bool(
                fraction >= 0.12
                and rectangularity >= 0.74
                and solidity >= 0.78
            )
            fill_contour = cv2.convexHull(contour) if convex_regularized else contour
            cv2.drawContours(
                contour_mask, [fill_contour], -1, 1, thickness=cv2.FILLED
            )
            contour_bool = contour_mask.astype(bool)
            inside_fraction = _area(contour_bool & root_mask) / max(
                1, _area(contour_bool)
            )
            contour_bool &= root_mask
            if inside_fraction < 0.88 or _area(contour_bool) < 12:
                continue
            alignment, closure, chroma, texture_delta = _boundary_evidence(
                contour_bool, root_mask, lab, gradient, texture
            )
            if max(alignment, closure) < 0.20 and max(chroma, texture_delta) < 0.12:
                continue
            full_mask = cv2.resize(
                contour_bool.astype(np.uint8),
                (output_width, output_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            score = float(
                np.clip(
                    0.43
                    + 0.16 * alignment
                    + 0.14 * closure
                    + 0.08 * chroma
                    + 0.06 * texture_delta
                    + 0.13 * shape_support,
                    0.0,
                    0.95,
                )
            )
            proposal = VisualMaskProposal(
                mask=full_mask,
                score=score,
                bbox_xyxy=_box(full_mask),
                scale_level=-close_size,
                view_id=f"appearance-closed-contour-{close_size}",
                support_views=(f"closed-edge-{close_size}",),
                support_levels=(-close_size,),
                best_view_iou=0.0,
                boundary_alignment=alignment,
                target_root_key=root_key,
                source="hpid-appearance-contour/closed-edge",
                geometric_support=shape_support,
            )
            proposals.append(proposal)
            rows.append(
                {
                    "root_key": root_key,
                    "proposal_kind": "closed_contour",
                    "closing_size": close_size,
                    "area_fraction": _area(contour_bool) / root_area,
                    "boundary_alignment": alignment,
                    "boundary_closure": closure,
                    "chroma_contrast": chroma,
                    "texture_contrast": texture_delta,
                    "rectangularity": rectangularity,
                    "solidity": solidity,
                    "vertices": vertices,
                    "convex_regularized": convex_regularized,
                    "score": score,
                }
            )
    return proposals, rows


def _enclosed_interior_proposals(
    *,
    candidates: list[VisualMaskProposal],
    root_mask: np.ndarray,
    root_key: str,
    lab: np.ndarray,
    gradient: np.ndarray,
    texture: np.ndarray,
    output_size: tuple[int, int],
    config: AppearanceProposalConfig,
) -> tuple[list[VisualMaskProposal], list[dict[str, object]]]:
    """Recover root-supported interiors enclosed by a visible frame or bezel."""

    analysis_height, analysis_width = root_mask.shape
    output_width, output_height = output_size
    root_area = max(1, _area(root_mask))
    proposals: list[VisualMaskProposal] = []
    rows: list[dict[str, object]] = []
    radius = int(np.clip(round(np.sqrt(root_area) * 0.008), 1, 4))
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    surround_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for candidate in candidates:
        candidate_small = cv2.resize(
            candidate.mask.astype(np.uint8),
            (analysis_width, analysis_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        candidate_fraction = _area(candidate_small) / root_area
        if not 0.025 <= candidate_fraction <= config.maximum_root_fraction:
            continue
        x1, y1, x2, y2 = _box(candidate_small)
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        closed = cv2.morphologyEx(
            candidate_small.astype(np.uint8),
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=1,
        ).astype(bool)
        crop = closed[y1:y2, x1:x2]
        component_count, labels = cv2.connectedComponents(
            (~crop).astype(np.uint8), connectivity=8
        )
        border_labels = set(labels[0, :])
        border_labels.update(labels[-1, :])
        border_labels.update(labels[:, 0])
        border_labels.update(labels[:, -1])
        for component in range(1, component_count):
            if component in border_labels:
                continue
            local_hole = labels == component
            hole = np.zeros_like(root_mask, dtype=bool)
            hole[y1:y2, x1:x2] = local_hole
            hole &= root_mask & ~candidate_small
            hole_area = _area(hole)
            fraction = hole_area / root_area
            if not 0.018 <= fraction <= config.maximum_root_fraction:
                continue
            surrounding = (
                cv2.dilate(hole.astype(np.uint8), surround_kernel).astype(bool)
                & ~hole
            )
            surround_support = _area(surrounding & closed) / max(
                1, _area(surrounding)
            )
            if surround_support < 0.34:
                continue
            alignment, closure, chroma, texture_delta = _boundary_evidence(
                hole, root_mask, lab, gradient, texture
            )
            if max(alignment, closure, surround_support) < 0.42:
                continue
            full_mask = cv2.resize(
                hole.astype(np.uint8),
                (output_width, output_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            score = float(
                np.clip(
                    0.44
                    + 0.18 * surround_support
                    + 0.14 * alignment
                    + 0.12 * closure
                    + 0.06 * chroma
                    + 0.04 * texture_delta,
                    0.0,
                    0.95,
                )
            )
            proposals.append(
                VisualMaskProposal(
                    mask=full_mask,
                    score=score,
                    bbox_xyxy=_box(full_mask),
                    scale_level=-2,
                    view_id="appearance-enclosed-interior",
                    support_views=("enclosure",),
                    support_levels=(-2,),
                    best_view_iou=0.0,
                    boundary_alignment=alignment,
                    target_root_key=root_key,
                    source="hpid-appearance-enclosure/interior",
                    geometric_support=surround_support,
                )
            )
            rows.append(
                {
                    "root_key": root_key,
                    "proposal_kind": "enclosed_interior",
                    "area_fraction": fraction,
                    "boundary_alignment": alignment,
                    "boundary_closure": closure,
                    "chroma_contrast": chroma,
                    "texture_contrast": texture_delta,
                    "surround_support": surround_support,
                    "score": score,
                }
            )
    return proposals, rows


def propose_appearance_regions(
    image: Image.Image,
    roots: list[MaskCandidate],
    *,
    config: AppearanceProposalConfig | None = None,
) -> AppearanceProposalResult:
    """Generate class-agnostic regions from graph-based appearance boundaries.

    The proposals are visual hypotheses, not material labels. Final Part IDs are
    still decided by the cross-cue ownership and hierarchy stages.
    """

    config = config or AppearanceProposalConfig()
    if not roots:
        return AppearanceProposalResult(
            (),
            {
                "algorithm": "hpid-multiscale-appearance-proposals-v2",
                "proposal_count": 0,
                "reason": "no_roots",
                "material_claim": "appearance_proxy_only",
                "ground_truth_used": False,
            },
        )
    try:
        from skimage.segmentation import felzenszwalb
    except ImportError as error:
        raise RuntimeError(
            "multiscale appearance proposals require scikit-image"
        ) from error

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width = rgb.shape[:2]
    analysis_width, analysis_height = _analysis_size(
        width, height, config.analysis_maximum_dimension
    )
    analysis = cv2.resize(
        rgb,
        (analysis_width, analysis_height),
        interpolation=cv2.INTER_AREA,
    )
    lab = cv2.cvtColor(analysis, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(analysis, cv2.COLOR_RGB2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (0, 0), 0.8)
    dx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(dx, dy)
    mean = cv2.boxFilter(gray, cv2.CV_32F, (7, 7), normalize=True)
    square_mean = cv2.boxFilter(gray * gray, cv2.CV_32F, (7, 7), normalize=True)
    texture = np.sqrt(np.maximum(0.0, square_mean - mean * mean))

    segmentations = (
        [
            felzenszwalb(
                analysis,
                scale=scale,
                sigma=config.graph_sigma,
                min_size=config.graph_minimum_size,
                channel_axis=-1,
            )
            for scale in config.graph_scales
        ]
        if config.use_graph_regions
        else []
    )
    selected: list[VisualMaskProposal] = []
    rows: list[dict[str, object]] = []
    selected_strata = {"detail": 0, "small": 0, "medium": 0, "large": 0}
    closed_contour_count = 0
    enclosed_interior_count = 0
    for root in roots:
        root_mask = root.mask.astype(bool)
        root_key = _root_key(root)
        root_candidates: list[VisualMaskProposal] = []
        root_small = cv2.resize(
            root_mask.astype(np.uint8),
            (analysis_width, analysis_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        root_area = max(1, _area(root_small))
        for scale_index, labels in enumerate(segmentations):
            active_labels = np.unique(labels[root_small])
            for label in active_labels:
                small_mask = (labels == label) & root_small
                if _area(small_mask) < 8:
                    continue
                component_count, component_labels, stats, _ = (
                    cv2.connectedComponentsWithStats(
                        small_mask.astype(np.uint8), connectivity=8
                    )
                )
                for component in range(1, component_count):
                    component_area = int(stats[component, cv2.CC_STAT_AREA])
                    fraction = component_area / root_area
                    if not (
                        config.minimum_root_fraction
                        <= fraction
                        <= config.maximum_root_fraction
                    ):
                        continue
                    component_small = component_labels == component
                    alignment, closure, chroma, texture_delta = _boundary_evidence(
                        component_small, root_small, lab, gradient, texture
                    )
                    cue_count = sum(
                        (
                            alignment >= 0.22 and closure >= 0.16,
                            chroma >= 0.08,
                            texture_delta >= 0.10,
                        )
                    )
                    if cue_count == 0:
                        continue
                    component_mask = cv2.resize(
                        component_small.astype(np.uint8),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                    component_mask &= root_mask
                    if _area(component_mask) < 8:
                        continue
                    score = float(
                        np.clip(
                            0.42
                            + 0.22 * alignment
                            + 0.14 * closure
                            + 0.12 * chroma
                            + 0.10 * texture_delta,
                            0.0,
                            0.94,
                        )
                    )
                    proposal = VisualMaskProposal(
                        mask=component_mask,
                        score=score,
                        bbox_xyxy=_box(component_mask),
                        scale_level=scale_index,
                        view_id=f"appearance-graph-scale-{scale_index}",
                        support_views=(f"appearance-scale-{scale_index}",),
                        support_levels=(scale_index,),
                        best_view_iou=0.0,
                        boundary_alignment=alignment,
                        target_root_key=root_key,
                        source="hpid-appearance-graph/felzenszwalb",
                    )
                    root_candidates.append(proposal)
                    rows.append(
                        {
                            "root_key": root_key,
                            "proposal_kind": "appearance_graph",
                            "scale": config.graph_scales[scale_index],
                            "area_fraction": fraction,
                            "boundary_alignment": alignment,
                            "boundary_closure": closure,
                            "chroma_contrast": chroma,
                            "texture_contrast": texture_delta,
                            "cue_count": cue_count,
                            "score": score,
                        }
                    )
        contour_proposals, contour_rows = (
            _closed_contour_proposals(
                root_mask=root_small,
                root_key=root_key,
                gray=gray,
                lab=lab,
                gradient=gradient,
                texture=texture,
                output_size=(width, height),
                config=config,
            )
            if config.use_closed_contours
            else ([], [])
        )
        root_candidates.extend(contour_proposals)
        rows.extend(contour_rows)
        closed_contour_count += len(contour_proposals)
        enclosure_proposals, enclosure_rows = (
            _enclosed_interior_proposals(
                candidates=root_candidates,
                root_mask=root_small,
                root_key=root_key,
                lab=lab,
                gradient=gradient,
                texture=texture,
                output_size=(width, height),
                config=config,
            )
            if config.use_enclosed_interiors
            else ([], [])
        )
        root_candidates.extend(enclosure_proposals)
        rows.extend(enclosure_rows)
        enclosed_interior_count += len(enclosure_proposals)
        root_selected, root_strata = _balanced_select(
            root_candidates,
            root_area=max(1, _area(root_mask)),
            config=config,
        )
        selected.extend(root_selected)
        for key, count in root_strata.items():
            selected_strata[key] += count

    return AppearanceProposalResult(
        tuple(selected),
        {
            "algorithm": "hpid-multiscale-appearance-proposals-v2",
            "analysis_size": [analysis_width, analysis_height],
            "root_count": len(roots),
            "graph_scales": list(config.graph_scales),
            "graph_regions_enabled": config.use_graph_regions,
            "closed_contours_enabled": config.use_closed_contours,
            "enclosed_interiors_enabled": config.use_enclosed_interiors,
            "proposal_count": len(selected),
            "closed_contour_candidate_count": closed_contour_count,
            "enclosed_interior_candidate_count": enclosed_interior_count,
            "selected_area_strata": selected_strata,
            "candidate_rows": rows,
            "appearance_cues": [
                "chromatic_contrast",
                "luminance_structure",
                "texture_proxy",
                "edge_alignment",
                "boundary_closure",
            ],
            "material_claim": "appearance_proxy_only",
            "ground_truth_used": False,
        },
    )
