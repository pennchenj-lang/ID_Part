from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np
from PIL import Image

from .fusion import MaskCandidate, mask_iou


@dataclass(frozen=True)
class AppearanceGraphConfig:
    """Conservative, category-independent evidence for visible part boundaries."""

    minimum_generic_utility: float = 0.38
    minimum_detail_utility: float = 0.40
    minimum_semantic_utility: float = 0.30
    minimum_boundary_evidence: float = 0.18
    duplicate_iou: float = 0.72
    duplicate_containment: float = 0.92
    duplicate_maximum_area_ratio: float = 1.55
    partial_conflict_containment: float = 0.35
    partial_conflict_iou: float = 0.12
    laminar_containment: float = 0.88
    strong_conflict_utility: float = 0.54


@dataclass(frozen=True)
class AppearanceGraphResult:
    candidates: tuple[MaskCandidate, ...]
    diagnostics: dict[str, object]


def annotate_appearance_evidence(
    image: Image.Image,
    candidates: list[MaskCandidate] | tuple[MaskCandidate, ...],
    roots: list[MaskCandidate] | tuple[MaskCandidate, ...],
) -> AppearanceGraphResult:
    """Attach photometric boundary evidence without selecting or naming regions.

    The normal appearance graph is built only from generic visual proposals.
    Semantic backends can therefore otherwise reach physical grouping without
    being checked for highlight- or shadow-only boundaries.  This audit runs on
    every candidate after root cleanup, preserves every mask and label, and only
    supplies evidence to the later physical-region gate.
    """

    roots_by_key = {_root_key(root): root for root in roots}
    lab, gradient, texture = _feature_maps(image)
    output: list[MaskCandidate] = []
    rows: list[dict[str, object]] = []
    missing_root_count = 0
    for candidate in candidates:
        root = roots_by_key.get(_root_key(candidate))
        if root is None:
            output.append(candidate)
            missing_root_count += 1
            continue
        evidence = _appearance_evidence(
            candidate,
            root,
            lab,
            gradient,
            texture,
        )
        output.append(
            replace(
                candidate,
                metadata={
                    **candidate.metadata,
                    "appearance_graph_evidence": evidence,
                    "photometric_boundary_audit": {
                        "algorithm": "hpid-all-candidate-photometric-audit-v1",
                        "evidence_only": True,
                        "can_create_id": False,
                        "ground_truth_used": False,
                    },
                },
            )
        )
        rows.append(
            {
                "candidate_key": _candidate_key(candidate),
                "semantic_name": candidate.semantic_name,
                "source": candidate.source,
                **evidence,
            }
        )
    return AppearanceGraphResult(
        tuple(output),
        {
            "algorithm": "hpid-all-candidate-photometric-audit-v1",
            "input_candidate_count": len(candidates),
            "audited_candidate_count": len(rows),
            "missing_root_count": missing_root_count,
            "evidence_only": True,
            "appearance_can_create_ids": False,
            "evidence": rows,
            "ground_truth_used": False,
        },
    )


def _root_key(candidate: MaskCandidate) -> str:
    return (
        f"{candidate.metadata.get('root_origin', 'legacy')}::"
        f"{candidate.metadata.get('root_index', 'unknown')}"
    )


def _candidate_key(candidate: MaskCandidate) -> str:
    return str(
        candidate.metadata.get(
            "candidate_key",
            f"{_root_key(candidate)}::{candidate.semantic_name}",
        )
    )


def _source_family(candidate: MaskCandidate) -> str:
    return str(
        candidate.metadata.get(
            "source_family", candidate.source.rsplit("/", maxsplit=1)[0]
        )
    )


def _area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def _robust_median(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.median(values))


def _feature_maps(image: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    blurred = cv2.GaussianBlur(gray, (0, 0), 0.8)
    dx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(dx, dy)
    mean = cv2.boxFilter(gray, cv2.CV_32F, (7, 7), normalize=True)
    square_mean = cv2.boxFilter(gray * gray, cv2.CV_32F, (7, 7), normalize=True)
    texture = np.sqrt(np.maximum(0.0, square_mean - mean * mean))
    return lab, gradient, texture


def _appearance_evidence(
    candidate: MaskCandidate,
    root: MaskCandidate,
    lab: np.ndarray,
    gradient: np.ndarray,
    texture: np.ndarray,
) -> dict[str, float | int | bool]:
    root_mask = root.mask.astype(bool)
    mask = candidate.mask.astype(bool) & root_mask
    area = _area(mask)
    root_area = max(1, _area(root_mask))
    if area == 0:
        return {
            "utility": 0.0,
            "boundary_alignment": 0.0,
            "boundary_closure": 0.0,
            "chroma_contrast": 0.0,
            "luminance_contrast": 0.0,
            "texture_contrast": 0.0,
            "shading_only_penalty": 0.0,
            "illumination_region": "none",
            "signed_luminance_delta": 0.0,
            "root_boundary_contact": 0.0,
            "nestedness": 1.0,
            "independent_cue_count": 0,
            "multi_view_confirmed": False,
            "geometric_support": 0.0,
        }

    radius = int(np.clip(round(np.sqrt(root_area) * 0.004), 1, 4))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    eroded = cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    dilated = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    inner = mask & ~eroded
    outer = dilated & root_mask & ~mask
    boundary = inner | outer

    root_eroded = cv2.erode(root_mask.astype(np.uint8), kernel).astype(bool)
    root_boundary = root_mask & ~root_eroded
    candidate_boundary = inner
    candidate_boundary_area = max(1, _area(candidate_boundary))
    root_boundary_contact = float(
        np.count_nonzero(candidate_boundary & root_boundary)
        / candidate_boundary_area
    )

    root_gradients = gradient[root_mask]
    edge_reference = (
        float(np.quantile(root_gradients, 0.85))
        if root_gradients.size
        else 0.0
    )
    boundary_gradient = gradient[boundary]
    effective_edge_reference = max(16.0, edge_reference)
    boundary_alignment = float(
        np.clip(
            _robust_median(boundary_gradient) / effective_edge_reference,
            0.0,
            1.0,
        )
    )
    boundary_closure = float(
        np.mean(boundary_gradient >= max(14.0, 0.55 * edge_reference))
        if boundary_gradient.size
        else 0.0
    )

    if np.any(inner) and np.any(outer):
        inner_lab = np.median(lab[inner], axis=0)
        outer_lab = np.median(lab[outer], axis=0)
        chroma_contrast = float(
            np.clip(np.linalg.norm(inner_lab[1:] - outer_lab[1:]) / 58.0, 0.0, 1.0)
        )
        signed_luminance_delta = float(inner_lab[0] - outer_lab[0])
        luminance_contrast = float(
            np.clip(abs(signed_luminance_delta) / 72.0, 0.0, 1.0)
        )
        texture_contrast = float(
            np.clip(
                abs(_robust_median(texture[inner]) - _robust_median(texture[outer]))
                / 26.0,
                0.0,
                1.0,
            )
        )
    else:
        chroma_contrast = 0.0
        luminance_contrast = 0.0
        texture_contrast = 0.0
        signed_luminance_delta = 0.0

    # Closed contours are deliberately not treated as material evidence here:
    # cast shadows and specular highlights can both be closed. A nested region
    # whose boundary changes mainly in L, while chroma and local texture remain
    # stable, is illumination evidence rather than a new part. Root silhouette
    # contact reduces (but never removes) that penalty because many physical
    # parts terminate on the object outline.
    material_change_support = max(
        1.20 * chroma_contrast,
        0.90 * texture_contrast,
    )
    illumination_strength = float(
        np.clip(
            (luminance_contrast - material_change_support - 0.055) / 0.30,
            0.0,
            1.0,
        )
    )
    nestedness = float(np.clip(1.0 - root_boundary_contact / 0.18, 0.0, 1.0))
    shading_only_penalty = float(
        np.clip(
            illumination_strength * (0.58 + 0.42 * nestedness),
            0.0,
            1.0,
        )
    )
    illumination_region = (
        "highlight"
        if shading_only_penalty >= 0.42 and signed_luminance_delta > 0.0
        else "shadow"
        if shading_only_penalty >= 0.42 and signed_luminance_delta < 0.0
        else "none"
    )
    multi_view = bool(candidate.metadata.get("multi_view_confirmed"))
    geometric_support = float(
        np.clip(candidate.metadata.get("geometric_support", 0.0), 0.0, 1.0)
    )
    semantic = not bool(candidate.metadata.get("generic_visual_region", True))
    sam_quality = float(
        np.clip(candidate.metadata.get("sam_quality", candidate.score), 0.0, 1.0)
    )
    independent_cue_count = sum(
        (
            boundary_alignment >= 0.28 and boundary_closure >= 0.18,
            chroma_contrast >= 0.12,
            texture_contrast >= 0.12,
            multi_view,
            geometric_support >= 0.48,
            semantic,
        )
    )
    utility = float(
        np.clip(
            0.31 * sam_quality
            + 0.18 * boundary_alignment
            + 0.15 * boundary_closure
            + 0.13 * chroma_contrast
            + 0.08 * texture_contrast
            + 0.08 * float(multi_view)
            + 0.10 * geometric_support
            + 0.07 * float(semantic)
            - 0.18 * shading_only_penalty,
            0.0,
            1.0,
        )
    )
    return {
        "utility": utility,
        "boundary_alignment": boundary_alignment,
        "boundary_closure": boundary_closure,
        "chroma_contrast": chroma_contrast,
        "luminance_contrast": luminance_contrast,
        "texture_contrast": texture_contrast,
        "shading_only_penalty": shading_only_penalty,
        "illumination_region": illumination_region,
        "signed_luminance_delta": signed_luminance_delta,
        "root_boundary_contact": root_boundary_contact,
        "nestedness": nestedness,
        "independent_cue_count": int(independent_cue_count),
        "multi_view_confirmed": multi_view,
        "geometric_support": geometric_support,
        "root_area_fraction": area / root_area,
    }


def _accept(
    candidate: MaskCandidate,
    evidence: dict[str, float | int | bool],
    config: AppearanceGraphConfig,
) -> tuple[bool, str]:
    utility = float(evidence["utility"])
    semantic = not bool(candidate.metadata.get("generic_visual_region", True))
    multi_view = bool(evidence["multi_view_confirmed"])
    cue_count = int(evidence["independent_cue_count"])
    boundary = max(
        float(evidence["boundary_alignment"]),
        float(evidence["boundary_closure"]),
    )
    if semantic:
        return utility >= config.minimum_semantic_utility, "semantic_evidence"
    kind = str(candidate.metadata.get("visual_region_kind", "panel"))
    large_structural_surface = bool(
        candidate.source.startswith("hpid-appearance-graph/")
        and kind == "panel"
        and float(evidence.get("root_area_fraction", 0.0)) >= 0.24
        and float(evidence["boundary_closure"]) >= 0.24
        and float(evidence["texture_contrast"]) >= 0.32
        and cue_count >= 1
    )
    if large_structural_surface:
        return True, "large_structural_surface"
    threshold = (
        config.minimum_detail_utility
        if kind == "detail"
        else config.minimum_generic_utility
    )
    effective_threshold = threshold - (0.08 if multi_view else 0.0)
    if utility < effective_threshold:
        return False, "low_cross_cue_utility"
    if cue_count == 0:
        return False, "single_model_without_visible_boundary"
    if kind == "detail" and not (
        multi_view
        or boundary >= config.minimum_boundary_evidence
        or float(evidence["chroma_contrast"]) >= 0.16
        or float(evidence["texture_contrast"]) >= 0.16
    ):
        return False, "unsupported_detail"
    return True, "cross_cue_supported"


def _priority(candidate: MaskCandidate) -> tuple[float, float, int, str]:
    evidence = candidate.metadata["appearance_graph_evidence"]
    semantic = not bool(candidate.metadata.get("generic_visual_region", True))
    utility = float(evidence["utility"])
    source_bonus = (
        0.06
        if candidate.source.startswith("sam2-amg[")
        else -0.04
        if candidate.source.startswith("hpid-shape-bottleneck/")
        else 0.0
    )
    area = _area(candidate.mask)
    return (
        utility + 0.08 * float(semantic) + source_bonus,
        candidate.score * candidate.source_reliability,
        area,
        _candidate_key(candidate),
    )


def _regularize_large_surface(
    candidate: MaskCandidate,
    root: MaskCandidate,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    mask = candidate.mask.astype(bool)
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None or len(points) < 3:
        return mask, {"applied": False, "convex_support": 0.0}
    hull = cv2.convexHull(points)
    envelope = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillConvexPoly(envelope, hull, 1)
    envelope_mask = envelope.astype(bool) & root.mask.astype(bool)
    convex_support = _area(mask) / max(1, _area(envelope_mask))
    root_fraction = _area(envelope_mask) / max(1, _area(root.mask))
    applied = bool(convex_support >= 0.72 and root_fraction <= 0.78)
    return (
        envelope_mask if applied else mask,
        {
            "applied": applied,
            "convex_support": float(convex_support),
            "regularized_root_fraction": float(root_fraction),
        },
    )
def _resolve_conflicts(
    candidates: list[MaskCandidate],
    config: AppearanceGraphConfig,
) -> tuple[list[MaskCandidate], list[dict[str, object]], int]:
    selected: list[MaskCandidate] = []
    rejected: list[dict[str, object]] = []
    cross_source_confirmations = 0
    for candidate in sorted(candidates, key=_priority, reverse=True):
        area = max(1, _area(candidate.mask))
        conflict: tuple[str, int, MaskCandidate] | None = None
        for incumbent_index, incumbent in enumerate(selected):
            incumbent_area = max(1, _area(incumbent.mask))
            intersection = _area(candidate.mask & incumbent.mask)
            if not intersection:
                continue
            iou = mask_iou(candidate.mask, incumbent.mask)
            containment = intersection / min(area, incumbent_area)
            area_ratio = max(area, incumbent_area) / min(area, incumbent_area)
            if iou >= config.duplicate_iou or (
                containment >= config.duplicate_containment
                and area_ratio <= config.duplicate_maximum_area_ratio
            ):
                conflict = ("duplicate_region", incumbent_index, incumbent)
                break
            if containment >= config.laminar_containment:
                continue
            if (
                containment >= config.partial_conflict_containment
                and iou >= config.partial_conflict_iou
            ):
                candidate_semantic = not bool(
                    candidate.metadata.get("generic_visual_region", True)
                )
                incumbent_semantic = not bool(
                    incumbent.metadata.get("generic_visual_region", True)
                )
                candidate_utility = float(
                    candidate.metadata["appearance_graph_evidence"]["utility"]
                )
                incumbent_utility = float(
                    incumbent.metadata["appearance_graph_evidence"]["utility"]
                )
                if (
                    candidate_semantic
                    and incumbent_semantic
                    and candidate.semantic_name != incumbent.semantic_name
                    and min(candidate_utility, incumbent_utility)
                    >= config.strong_conflict_utility
                ):
                    continue
                conflict = ("non_laminar_overlap", incumbent_index, incumbent)
                break
        if conflict is None:
            selected.append(candidate)
            continue
        if (
            conflict[0] == "duplicate_region"
            and _source_family(candidate) != _source_family(conflict[2])
        ):
            incumbent = conflict[2]
            families = {
                *incumbent.metadata.get("supporting_source_families", ()),
                _source_family(incumbent),
                _source_family(candidate),
            }
            selected[conflict[1]] = replace(
                incumbent,
                score=max(incumbent.score, candidate.score),
                metadata={
                    **incumbent.metadata,
                    "cross_source_confirmed": True,
                    "supporting_source_families": sorted(families),
                },
            )
            cross_source_confirmations += 1
        rejected.append(
            {
                "candidate_key": _candidate_key(candidate),
                "reason": conflict[0],
                "kept_candidate_key": _candidate_key(conflict[2]),
            }
        )
    return selected, rejected, cross_source_confirmations


def optimize_appearance_graph(
    image: Image.Image,
    candidates: list[MaskCandidate],
    roots: list[MaskCandidate],
    *,
    config: AppearanceGraphConfig | None = None,
) -> AppearanceGraphResult:
    """Select a clean laminar region graph from model proposals.

    RGB appearance is treated as evidence only. The method never claims true
    material recovery and never reads benchmark annotations.
    """

    config = config or AppearanceGraphConfig()
    roots_by_key = {_root_key(root): root for root in roots}
    lab, gradient, texture = _feature_maps(image)
    accepted_by_root: dict[str, list[MaskCandidate]] = {}
    rejected_rows: list[dict[str, object]] = []
    evidence_rows: list[dict[str, object]] = []
    for candidate in candidates:
        root_key = _root_key(candidate)
        root = roots_by_key.get(root_key)
        if root is None:
            rejected_rows.append(
                {"candidate_key": _candidate_key(candidate), "reason": "missing_root"}
            )
            continue
        evidence = _appearance_evidence(candidate, root, lab, gradient, texture)
        accepted, reason = _accept(candidate, evidence, config)
        evidence_rows.append(
            {
                "candidate_key": _candidate_key(candidate),
                "root_key": root_key,
                "accepted_before_graph": accepted,
                "decision": reason,
                **evidence,
            }
        )
        if not accepted:
            rejected_rows.append(
                {
                    "candidate_key": _candidate_key(candidate),
                    "reason": reason,
                }
            )
            continue
        adjusted_score = float(
            np.clip(0.62 * candidate.score + 0.38 * float(evidence["utility"]), 0.0, 1.0)
        )
        updated_mask = candidate.mask
        surface_regularization: dict[str, float | bool] | None = None
        if reason == "large_structural_surface":
            updated_mask, surface_regularization = _regularize_large_surface(
                candidate, root
            )
        updated = replace(
            candidate,
            mask=updated_mask,
            score=adjusted_score,
            metadata={
                **candidate.metadata,
                "appearance_graph_evidence": evidence,
                "appearance_graph_algorithm": "hpid-cross-cue-region-graph-v1",
                "large_surface_regularization": surface_regularization,
                "ground_truth_used": False,
            },
        )
        accepted_by_root.setdefault(root_key, []).append(updated)

    selected: list[MaskCandidate] = []
    conflict_rows: list[dict[str, object]] = []
    cross_source_confirmation_count = 0
    for root_key in sorted(accepted_by_root):
        root_selected, root_conflicts, root_confirmations = _resolve_conflicts(
            accepted_by_root[root_key], config
        )
        selected.extend(root_selected)
        cross_source_confirmation_count += root_confirmations
        conflict_rows.extend(
            {"root_key": root_key, **row} for row in root_conflicts
        )
    rejected_rows.extend(conflict_rows)
    selected.sort(key=lambda candidate: (_root_key(candidate), _candidate_key(candidate)))
    return AppearanceGraphResult(
        tuple(selected),
        {
            "algorithm": "hpid-cross-cue-region-graph-v1",
            "input_candidate_count": len(candidates),
            "accepted_candidate_count": len(selected),
            "rejected_candidate_count": len(rejected_rows),
            "cross_source_confirmation_count": cross_source_confirmation_count,
            "evidence": evidence_rows,
            "rejections": rejected_rows,
            "appearance_cues": [
                "chromatic_contrast",
                "luminance_contrast",
                "texture_proxy",
                "boundary_alignment",
                "boundary_closure",
                "multi_view_consensus",
                "semantic_support",
            ],
            "material_claim": "appearance_proxy_only",
            "ground_truth_used": False,
        },
    )
