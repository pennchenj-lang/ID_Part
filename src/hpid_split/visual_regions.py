from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .fusion import MaskCandidate, mask_iou


@dataclass(frozen=True)
class VisualRegionConfig:
    points_per_crop: int = 20
    points_per_batch: int = 32
    crops_n_layers: int = 0
    crop_n_points_downscale_factor: int = 2
    predicted_iou_threshold: float = 0.78
    stability_score_threshold: float = 0.82
    crops_nms_threshold: float = 0.65
    minimum_root_containment: float = 0.68
    minimum_root_area_fraction: float = 0.00035
    maximum_root_area_fraction: float = 0.58
    semantic_match_iou: float = 0.28
    semantic_match_containment: float = 0.72
    semantic_minimum_candidate_score: float = 0.30
    semantic_family_minimum_candidate_score: float = 0.22
    allow_dense_semantic_relabel: bool = False
    maximum_regions_per_root: int = 32
    hierarchy_minimum_containment: float = 0.90
    hierarchy_maximum_child_fraction: float = 0.72
    consensus_match_iou: float = 0.52
    consensus_match_containment: float = 0.90
    consensus_maximum_area_ratio: float = 1.55
    isolated_crop_score_scale: float = 0.93
    use_isolated_root_crops: bool = False
    isolated_root_points_per_crop: int = 10
    isolated_root_crop_padding: float = 0.08
    maximum_isolated_root_crops: int = 48
    minimum_isolated_root_area_px: int = 96
    maximum_isolated_root_image_fraction: float = 0.48


@dataclass(frozen=True)
class VisualMaskProposal:
    mask: np.ndarray
    score: float
    bbox_xyxy: tuple[int, int, int, int] | None = None
    scale_level: int = 0
    view_id: str = "global"
    support_views: tuple[str, ...] = ()
    support_levels: tuple[int, ...] = ()
    best_view_iou: float = 0.0
    boundary_alignment: float = 0.0
    target_root_key: str | None = None
    source: str | None = None
    geometric_support: float = 0.0


@dataclass(frozen=True)
class VisualRegionGeneration:
    candidates: tuple[MaskCandidate, ...]
    diagnostics: dict[str, object]


def _area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def _box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _overlap_metrics(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    intersection = int(np.count_nonzero(first & second))
    if not intersection:
        return 0.0, 0.0
    first_area = _area(first)
    second_area = _area(second)
    union = first_area + second_area - intersection
    return (
        intersection / max(1, union),
        intersection / max(1, min(first_area, second_area)),
    )


def _edge_strength(image_bgr: np.ndarray) -> np.ndarray:
    smoothed = cv2.GaussianBlur(image_bgr, (0, 0), 0.65)
    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB).astype(np.float32)
    gradients: list[np.ndarray] = []
    for channel in cv2.split(lab):
        dx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        gradients.append(cv2.magnitude(dx, dy))
    return np.maximum.reduce(gradients)


def _boundary_alignment(edge_strength: np.ndarray, mask: np.ndarray) -> float:
    """Measure whether a proposal boundary follows a visible image edge."""

    if not np.any(mask):
        return 0.0
    kernel = np.ones((3, 3), dtype=np.uint8)
    binary = mask.astype(np.uint8)
    boundary = cv2.dilate(binary, kernel).astype(bool) ^ cv2.erode(
        binary, kernel
    ).astype(bool)
    if not np.any(boundary):
        return 0.0
    reference = float(np.quantile(edge_strength, 0.90))
    if reference <= 1e-6:
        return 0.0
    return float(
        np.clip(float(edge_strength[boundary].mean()) / reference, 0.0, 1.0)
    )


def _consolidate_multiview_proposals(
    image: Image.Image,
    proposals: list[VisualMaskProposal],
    config: VisualRegionConfig,
) -> tuple[list[VisualMaskProposal], dict[str, object]]:
    """Collapse masks that independently recur across scales or crop views.

    SAM masks are proposals, not identities. Overlapping crop windows frequently
    return slightly different masks for one physical part. Treating all of them
    as independent Part IDs creates the duplicate panels and noisy ownership
    seen in difficult assets. This stage clusters only near-equal masks from
    independent views and keeps the member with the strongest joint SAM,
    cross-view, and image-boundary evidence.
    """

    if not proposals:
        return proposals, {
            "raw_proposal_count": 0,
            "consolidated_proposal_count": 0,
            "multi_view_cluster_count": 0,
            "duplicate_view_proposals_removed": 0,
        }

    parent = list(range(len(proposals)))
    cluster_min_area = [_area(proposal.mask) for proposal in proposals]
    cluster_max_area = cluster_min_area.copy()

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        minimum_area = min(
            cluster_min_area[first_root], cluster_min_area[second_root]
        )
        maximum_area = max(
            cluster_max_area[first_root], cluster_max_area[second_root]
        )
        if maximum_area / max(1, minimum_area) > config.consensus_maximum_area_ratio:
            return
        parent[second_root] = first_root
        cluster_min_area[first_root] = minimum_area
        cluster_max_area[first_root] = maximum_area

    for first_index, first in enumerate(proposals):
        first_area = _area(first.mask)
        if not first_area:
            continue
        for second_index in range(first_index + 1, len(proposals)):
            second = proposals[second_index]
            if first.view_id == second.view_id:
                continue
            if (
                first.target_root_key is not None
                and second.target_root_key is not None
                and first.target_root_key != second.target_root_key
            ):
                continue
            second_area = _area(second.mask)
            if not second_area:
                continue
            area_ratio = max(first_area, second_area) / max(
                1, min(first_area, second_area)
            )
            iou, containment = _overlap_metrics(first.mask, second.mask)
            if iou >= config.consensus_match_iou or (
                containment >= config.consensus_match_containment
                and area_ratio <= config.consensus_maximum_area_ratio
            ):
                union(first_index, second_index)

    groups: dict[int, list[int]] = {}
    for index in range(len(proposals)):
        groups.setdefault(find(index), []).append(index)

    image_bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    edge_strength = _edge_strength(image_bgr)
    consolidated: list[VisualMaskProposal] = []
    multi_view_clusters = 0
    for indices in groups.values():
        views = tuple(sorted({proposals[index].view_id for index in indices}))
        levels = tuple(sorted({proposals[index].scale_level for index in indices}))
        targeted_roots = {
            proposals[index].target_root_key
            for index in indices
            if proposals[index].target_root_key is not None
        }
        if len(views) > 1:
            multi_view_clusters += 1
        ranked: list[tuple[float, float, float, int]] = []
        for index in indices:
            proposal = proposals[index]
            agreements = [
                _overlap_metrics(proposal.mask, proposals[other].mask)[0]
                for other in indices
                if other != index and proposal.view_id != proposals[other].view_id
            ]
            best_iou = max(agreements, default=0.0)
            edge_score = _boundary_alignment(edge_strength, proposal.mask)
            rank = 0.60 * proposal.score + 0.24 * best_iou + 0.16 * edge_score
            ranked.append((rank, best_iou, edge_score, index))
        _, best_iou, edge_score, representative_index = max(ranked)
        representative = proposals[representative_index]
        score = representative.score
        if len(views) > 1:
            score = min(1.0, score * (0.96 + 0.08 * best_iou))
        elif representative.scale_level > 0:
            score *= config.isolated_crop_score_scale
        consolidated.append(
            VisualMaskProposal(
                mask=representative.mask,
                score=float(score),
                bbox_xyxy=representative.bbox_xyxy,
                scale_level=representative.scale_level,
                view_id=representative.view_id,
                support_views=views,
                support_levels=levels,
                best_view_iou=float(best_iou),
                boundary_alignment=float(edge_score),
                target_root_key=(
                    next(iter(targeted_roots))
                    if len(targeted_roots) == 1
                    else representative.target_root_key
                ),
            )
        )
    consolidated.sort(
        key=lambda proposal: (-proposal.score, -_area(proposal.mask), _box(proposal.mask))
    )
    return consolidated, {
        "raw_proposal_count": len(proposals),
        "consolidated_proposal_count": len(consolidated),
        "multi_view_cluster_count": multi_view_clusters,
        "duplicate_view_proposals_removed": len(proposals) - len(consolidated),
    }


def _root_key(candidate: MaskCandidate) -> str:
    origin = candidate.metadata.get("root_origin", "legacy")
    index = candidate.metadata.get("root_index", "unknown")
    return f"{origin}::{index}"


def _root_semantic_namespaces(roots: list[MaskCandidate]) -> dict[str, str]:
    """Give repeated same-domain roots distinct, deterministic child namespaces."""

    grouped: dict[str, list[MaskCandidate]] = {}
    for root in roots:
        grouped.setdefault(root.semantic_name, []).append(root)

    namespaces: dict[str, str] = {}
    for semantic_name, group in grouped.items():
        ordered = sorted(group, key=lambda candidate: (_box(candidate.mask), _root_key(candidate)))
        if len(ordered) == 1:
            namespaces[_root_key(ordered[0])] = semantic_name
            continue
        for ordinal, root in enumerate(ordered, start=1):
            namespaces[_root_key(root)] = f"{semantic_name}_asset_{ordinal:02d}"
    return namespaces


def _region_kind(mask: np.ndarray, root_area: int) -> str:
    x0, y0, x1, y1 = _box(mask)
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    aspect = max(width / height, height / width)
    fraction = _area(mask) / max(1, root_area)
    if fraction <= 0.012:
        return "detail"
    # Axis-aligned boxes make diagonal handles, rails, cables, and blades look
    # deceptively square.  PCA measures the occupied pixels themselves, so a
    # slanted thin part remains a strip regardless of image orientation.
    ys, xs = np.nonzero(mask)
    principal_elongation = 1.0
    if len(xs) >= 8:
        points = np.column_stack((xs, ys)).astype(np.float64)
        eigenvalues = np.linalg.eigvalsh(np.cov(points.T))
        principal_elongation = float(
            eigenvalues[-1] / max(1e-6, eigenvalues[0])
        )
    if aspect >= 3.2 or principal_elongation >= 10.0:
        return "strip"
    return "panel"


def _semantic_match(
    mask: np.ndarray,
    semantic_candidates: list[MaskCandidate],
    config: VisualRegionConfig,
) -> MaskCandidate | None:
    area = _area(mask)
    matches: list[tuple[float, MaskCandidate]] = []
    for candidate in semantic_candidates:
        if candidate.semantic_name == candidate.semantic_parent:
            continue
        guided = bool(candidate.metadata.get("guided_prompt"))
        if (
            bool(candidate.metadata.get("dense_semantic_fallback"))
            and not config.allow_dense_semantic_relabel
            and not guided
        ):
            continue
        if candidate.score < config.semantic_minimum_candidate_score and not guided:
            continue
        candidate_area = _area(candidate.mask)
        intersection = int(np.count_nonzero(mask & candidate.mask))
        if not intersection:
            continue
        overlap = mask_iou(mask, candidate.mask)
        containment = intersection / max(1, min(area, candidate_area))
        size_ratio = max(area, candidate_area) / max(1, min(area, candidate_area))
        if overlap >= config.semantic_match_iou or (
            containment >= config.semantic_match_containment and size_ratio <= 3.0
        ):
            matches.append((max(overlap, 0.85 * containment), candidate))
    if matches:
        return max(matches, key=lambda item: item[0])[1]

    grouped: dict[str, list[MaskCandidate]] = {}
    for candidate in semantic_candidates:
        if candidate.semantic_name == candidate.semantic_parent:
            continue
        guided = bool(candidate.metadata.get("guided_prompt"))
        if (
            bool(candidate.metadata.get("dense_semantic_fallback"))
            and not config.allow_dense_semantic_relabel
            and not guided
        ):
            continue
        if (
            candidate.score < config.semantic_family_minimum_candidate_score
            and not guided
        ):
            continue
        candidate_area = _area(candidate.mask)
        intersection = int(np.count_nonzero(mask & candidate.mask))
        if intersection / max(1, candidate_area) < 0.72:
            continue
        grouped.setdefault(candidate.semantic_name, []).append(candidate)
    family_matches: list[tuple[float, MaskCandidate]] = []
    for group in grouped.values():
        if len(group) < 2:
            continue
        support_union = np.zeros_like(mask, dtype=bool)
        for candidate in group:
            support_union |= candidate.mask.astype(bool)
        support_area = _area(support_union & mask)
        support_fraction = support_area / max(1, area)
        expansion_ratio = area / max(1, support_area)
        if support_fraction < 0.16 or expansion_ratio > 6.0:
            continue
        representative = max(group, key=lambda candidate: candidate.score)
        family_matches.append(
            (support_fraction + 0.03 * min(len(group), 4), representative)
        )
    return max(family_matches, key=lambda item: item[0])[1] if family_matches else None


def _stabilize_and_attach_hierarchy(
    accepted: list[tuple[MaskCandidate, int, str]],
    roots: list[MaskCandidate],
    config: VisualRegionConfig,
    candidate_namespace: str | None = None,
) -> list[MaskCandidate]:
    """Assign spatially stable names and infer containment-based visual parents."""

    roots_by_key = {_root_key(root): root for root in roots}
    root_namespaces = _root_semantic_namespaces(roots)
    ordered = sorted(
        accepted,
        key=lambda item: (
            item[2],
            _box(item[0].mask)[1],
            _box(item[0].mask)[0],
            -item[1],
        ),
    )
    ordinals: dict[tuple[str, str], int] = {}
    region_ordinals: dict[str, int] = {}
    stabilized: list[MaskCandidate] = []
    for candidate, _, root_key in ordered:
        root = roots_by_key[root_key]
        region_ordinals[root_key] = region_ordinals.get(root_key, 0) + 1
        region_ordinal = region_ordinals[root_key]
        root_candidate_key = str(
            root.metadata.get(
                "candidate_key", f"root:{root.metadata.get('root_index')}"
            )
        )
        metadata = {
            **candidate.metadata,
            "candidate_key": (
                f"{root_candidate_key}/"
                f"{candidate_namespace + '-' if candidate_namespace else ''}"
                f"visual-region:{region_ordinal:02d}"
            ),
            "parent_candidate_key": root_candidate_key,
            "assembly_parent_candidate_key": root_candidate_key,
            "hierarchy_depth": 1,
        }
        maximum_instances = candidate.metadata.get("maximum_instances")
        if maximum_instances is not None:
            metadata["maximum_instances"] = int(maximum_instances)
        if bool(candidate.metadata["generic_visual_region"]):
            kind = str(candidate.metadata["visual_region_kind"])
            ordinal_key = (root_key, kind)
            ordinals[ordinal_key] = ordinals.get(ordinal_key, 0) + 1
            semantic_name = (
                f"{root_namespaces[root_key]}_visual_"
                f"{candidate_namespace + '_' if candidate_namespace else ''}"
                f"{kind}_"
                f"{ordinals[ordinal_key]:02d}"
            )
            stabilized.append(
                replace(
                    candidate,
                    semantic_name=semantic_name,
                    semantic_parent=root.semantic_name,
                    metadata=metadata,
                )
            )
        else:
            stabilized.append(replace(candidate, metadata=metadata))

    hierarchical: list[MaskCandidate] = []
    for candidate in stabilized:
        if not bool(candidate.metadata["generic_visual_region"]):
            hierarchical.append(candidate)
            continue
        candidate_area = _area(candidate.mask)
        root_key = _root_key(candidate)
        possible_parents: list[tuple[int, MaskCandidate]] = []
        for parent in stabilized:
            if parent is candidate or _root_key(parent) != root_key:
                continue
            if not bool(parent.metadata["generic_visual_region"]):
                continue
            parent_area = _area(parent.mask)
            if parent_area <= candidate_area:
                continue
            child_fraction = candidate_area / max(1, parent_area)
            if child_fraction > config.hierarchy_maximum_child_fraction:
                continue
            containment = int(np.count_nonzero(candidate.mask & parent.mask)) / max(
                1, candidate_area
            )
            if containment < config.hierarchy_minimum_containment:
                continue
            possible_parents.append((parent_area, parent))
        if not possible_parents:
            hierarchical.append(candidate)
            continue
        parent = min(possible_parents, key=lambda item: item[0])[1]
        metadata = {
            **candidate.metadata,
            "parent_candidate_key": str(parent.metadata["candidate_key"]),
            "assembly_parent_semantic": parent.semantic_name,
            "assembly_parent_candidate_key": str(parent.metadata["candidate_key"]),
            "hierarchy_depth": int(parent.metadata.get("hierarchy_depth", 1)) + 1,
            "visual_parent_containment": float(
                np.count_nonzero(candidate.mask & parent.mask)
                / max(1, candidate_area)
            ),
        }
        hierarchical.append(
            replace(
                candidate,
                semantic_parent=parent.semantic_name,
                metadata=metadata,
            )
        )
    return hierarchical


def visual_region_candidates_from_masks(
    proposals: list[VisualMaskProposal],
    roots: list[MaskCandidate],
    semantic_candidates: list[MaskCandidate],
    *,
    config: VisualRegionConfig | None = None,
    source: str = "sam2-amg/point-grid",
    candidate_namespace: str | None = None,
) -> VisualRegionGeneration:
    """Convert label-free SAM regions into hierarchy-aware HPID candidates."""

    config = config or VisualRegionConfig()
    accepted: list[tuple[MaskCandidate, int, str]] = []
    rejected_outside = 0
    rejected_scale = 0
    rejected_duplicate = 0
    rejected_selected_root = 0
    rejected_atomic_scene_instance = 0
    per_root_count: dict[str, int] = {}
    ordered = sorted(
        proposals,
        key=lambda proposal: (
            -proposal.score,
            -_area(proposal.mask),
            _box(proposal.mask),
        ),
    )
    for proposal in ordered:
        mask = proposal.mask.astype(bool)
        area = _area(mask)
        if area < 8:
            rejected_scale += 1
            continue
        duplicates_selected_root = False
        for selected_root in roots:
            selected_area = _area(selected_root.mask)
            intersection = int(np.count_nonzero(mask & selected_root.mask))
            if not intersection:
                continue
            union = area + selected_area - intersection
            overlap = intersection / max(1, union)
            containment = intersection / max(1, min(area, selected_area))
            area_coherence = min(area, selected_area) / max(area, selected_area)
            if overlap >= 0.82 or (
                containment >= 0.95 and area_coherence >= 0.82
            ):
                duplicates_selected_root = True
                break
        if duplicates_selected_root:
            rejected_selected_root += 1
            continue
        overlaps_atomic_scene_instance = any(
            bool(root.metadata.get("atomic_scene_instance"))
            and np.count_nonzero(mask & root.mask) / max(1, area)
            >= config.minimum_root_containment
            for root in roots
        )
        if overlaps_atomic_scene_instance:
            rejected_atomic_scene_instance += 1
            continue
        root_matches: list[tuple[float, MaskCandidate]] = []
        for root in roots:
            if (
                proposal.target_root_key is not None
                and _root_key(root) != proposal.target_root_key
            ):
                continue
            root_area = _area(root.mask)
            intersection = int(np.count_nonzero(mask & root.mask))
            containment = intersection / max(1, area)
            fraction = area / max(1, root_area)
            if containment < config.minimum_root_containment:
                continue
            if not (
                config.minimum_root_area_fraction
                <= fraction
                <= config.maximum_root_area_fraction
            ):
                continue
            specificity = float(np.sqrt(np.clip(fraction, 0.0, 1.0)))
            routing_score = containment + 0.18 * specificity + 0.02 * root.score
            root_matches.append((routing_score, root))
        if not root_matches:
            rejected_outside += 1
            continue
        _, root = max(root_matches, key=lambda item: item[0])
        root_key = _root_key(root)
        if per_root_count.get(root_key, 0) >= config.maximum_regions_per_root:
            rejected_scale += 1
            continue
        if any(
            existing_root_key == root_key
            and mask_iou(mask, existing.mask) >= 0.86
            for existing, _, existing_root_key in accepted
        ):
            rejected_duplicate += 1
            continue
        matched = _semantic_match(mask, semantic_candidates, config)
        root_area = _area(root.mask)
        if matched is not None:
            semantic_name = matched.semantic_name
            semantic_parent = matched.semantic_parent
            prompt = f"visual support for {matched.semantic_name}"
            assembly_parent = matched.metadata.get(
                "assembly_parent_semantic", matched.semantic_parent
            )
            generic = False
        else:
            kind = _region_kind(mask, root_area)
            semantic_name = f"{root.semantic_name}_visual_{kind}_pending"
            semantic_parent = root.semantic_name
            prompt = "automatic visual region"
            assembly_parent = root.semantic_name
            generic = True
        root_candidate_key = str(
            root.metadata.get("candidate_key", f"root:{root.metadata.get('root_index')}")
        )
        proposal_source = proposal.source or source
        candidate = MaskCandidate(
            semantic_name=semantic_name,
            semantic_parent=semantic_parent,
            mask=mask,
            score=float(np.clip(proposal.score, 0.0, 1.0)),
            source=proposal_source,
            prompt=prompt,
            source_reliability=0.62 if generic else 0.68,
            metadata={
                "source_family": proposal_source.rsplit("/", maxsplit=1)[0],
                "root_origin": root.metadata.get("root_origin"),
                "root_index": root.metadata.get("root_index"),
                "candidate_key": (
                    f"{root_candidate_key}/visual-region:"
                    f"{per_root_count.get(root_key, 0) + 1:02d}"
                ),
                "parent_candidate_key": root_candidate_key,
                "assembly_parent_semantic": str(assembly_parent),
                "assembly_parent_candidate_key": root_candidate_key,
                "visual_region": True,
                "generic_visual_region": generic,
                "visual_region_kind": _region_kind(mask, root_area),
                "sam_quality": float(proposal.score),
                "proposal_scale_level": int(proposal.scale_level),
                "proposal_view_id": proposal.view_id,
                "proposal_support_views": list(proposal.support_views),
                "proposal_support_levels": list(proposal.support_levels),
                "proposal_best_view_iou": float(proposal.best_view_iou),
                "proposal_boundary_alignment": float(
                    proposal.boundary_alignment
                ),
                "proposal_target_root_key": proposal.target_root_key,
                "geometric_support": float(proposal.geometric_support),
                "multi_view_confirmed": len(proposal.support_views) > 1,
                "root_containment": float(
                    np.count_nonzero(mask & root.mask) / max(1, area)
                ),
                "root_area_fraction": float(area / max(1, root_area)),
                "box_xyxy": list(proposal.bbox_xyxy or _box(mask)),
                "ground_truth_used": False,
                **{
                    key: root.metadata[key]
                    for key in (
                        "scene_object_id",
                        "physical_group_id",
                        "scene_role",
                    )
                    if root.metadata.get(key) is not None
                },
                **(
                    {
                        "semantic_support_candidate_key": matched.metadata.get(
                            "candidate_key"
                        ),
                        "maximum_instances": int(
                            matched.metadata["maximum_instances"]
                        )
                    }
                    if matched is not None
                    and matched.metadata.get("maximum_instances") is not None
                    else {}
                ),
                **(
                    {
                        "semantic_support_candidate_key": matched.metadata.get(
                            "candidate_key"
                        )
                    }
                    if matched is not None
                    and matched.metadata.get("maximum_instances") is None
                    else {}
                ),
            },
        )
        accepted.append((candidate, area, root_key))
        per_root_count[root_key] = per_root_count.get(root_key, 0) + 1

    candidates = _stabilize_and_attach_hierarchy(
        accepted,
        roots,
        config,
        candidate_namespace=candidate_namespace,
    )
    return VisualRegionGeneration(
        tuple(candidates),
        {
            "algorithm": "hpid-open-set-visual-regions-v2",
            "proposal_count": len(proposals),
            "accepted_candidate_count": len(candidates),
            "generic_candidate_count": sum(
                bool(candidate.metadata["generic_visual_region"])
                for candidate in candidates
            ),
            "semantic_support_candidate_count": sum(
                not bool(candidate.metadata["generic_visual_region"])
                for candidate in candidates
            ),
            "rejected_outside_root_count": rejected_outside,
            "rejected_scale_count": rejected_scale,
            "rejected_duplicate_count": rejected_duplicate,
            "rejected_selected_root_count": rejected_selected_root,
            "rejected_atomic_scene_instance_count": (
                rejected_atomic_scene_instance
            ),
            "per_root_candidate_count": per_root_count,
            "candidate_namespace": candidate_namespace,
            "ground_truth_used": False,
        },
    )


class Sam2VisualRegionProposer:
    """Run SAM2 automatic mask generation with the already-loaded model."""

    def __init__(
        self,
        sam_processor: Any,
        sam_model: Any,
        *,
        segmentation_model: str,
        device: str,
        config: VisualRegionConfig | None = None,
    ) -> None:
        try:
            from transformers.pipelines import MaskGenerationPipeline
        except ImportError as error:
            raise RuntimeError(
                "Install the foundation extra: pip install 'hpid-split[foundation]'"
            ) from error
        self.config = config or VisualRegionConfig()
        self.source = f"sam2-amg[{segmentation_model}]/point-grid"
        self.pipeline = MaskGenerationPipeline(
            model=sam_model,
            image_processor=sam_processor.image_processor,
            device=device,
        )

    def propose_global(
        self, image: Image.Image
    ) -> tuple[list[VisualMaskProposal], dict[str, object]]:
        """Generate one reusable label-free proposal pool for roots and parts."""

        raw = self._pipeline_proposals(
            image,
            points_per_crop=self.config.points_per_crop,
            offset_xy=(0, 0),
            output_shape=(image.height, image.width),
            scale_level=0,
            view_id="global",
        )
        proposals, consensus = _consolidate_multiview_proposals(
            image, raw, self.config
        )
        return proposals, {
            "algorithm": "sam2-global-proposal-pool-v1",
            "raw_proposal_count": len(raw),
            "proposal_count": len(proposals),
            "multi_view_consensus": consensus,
            "ground_truth_used": False,
        }

    def _pipeline_proposals(
        self,
        image: Image.Image,
        *,
        points_per_crop: int,
        offset_xy: tuple[int, int],
        output_shape: tuple[int, int],
        scale_level: int,
        view_id: str,
        target_root_key: str | None = None,
    ) -> list[VisualMaskProposal]:
        result = self.pipeline(
            image.convert("RGB"),
            points_per_crop=points_per_crop,
            points_per_batch=self.config.points_per_batch,
            pred_iou_thresh=self.config.predicted_iou_threshold,
            stability_score_thresh=self.config.stability_score_threshold,
            crops_n_layers=0,
            crops_nms_thresh=self.config.crops_nms_threshold,
            output_bboxes_mask=True,
            max_hole_area=64,
            max_sprinkle_area=32,
        )
        boxes = result.get("bounding_boxes", [None] * len(result["masks"]))
        offset_x, offset_y = offset_xy
        output_height, output_width = output_shape
        proposals: list[VisualMaskProposal] = []
        for mask, score, box in zip(
            result["masks"], result["scores"], boxes, strict=True
        ):
            local_mask = np.asarray(mask).astype(bool)
            full_mask = np.zeros((output_height, output_width), dtype=bool)
            y1 = min(output_height, offset_y + local_mask.shape[0])
            x1 = min(output_width, offset_x + local_mask.shape[1])
            full_mask[offset_y:y1, offset_x:x1] = local_mask[
                : y1 - offset_y, : x1 - offset_x
            ]
            if box is None:
                full_box = _box(full_mask)
            else:
                local_box = tuple(round(float(value)) for value in box)
                full_box = (
                    local_box[0] + offset_x,
                    local_box[1] + offset_y,
                    local_box[2] + offset_x,
                    local_box[3] + offset_y,
                )
            proposals.append(
                VisualMaskProposal(
                    mask=full_mask,
                    score=float(score),
                    bbox_xyxy=full_box,
                    scale_level=scale_level,
                    view_id=view_id,
                    target_root_key=target_root_key,
                )
            )
        return proposals

    @staticmethod
    def _layer_crop_boxes(
        width: int, height: int, layer: int
    ) -> list[tuple[int, int, int, int]]:
        divisions = 2**layer
        tile_width = width / divisions
        tile_height = height / divisions
        overlap_x = round(tile_width * 0.14)
        overlap_y = round(tile_height * 0.14)
        boxes: list[tuple[int, int, int, int]] = []
        for row in range(divisions):
            for column in range(divisions):
                x0 = max(0, round(column * tile_width) - overlap_x)
                y0 = max(0, round(row * tile_height) - overlap_y)
                x1 = min(width, round((column + 1) * tile_width) + overlap_x)
                y1 = min(height, round((row + 1) * tile_height) + overlap_y)
                boxes.append((x0, y0, x1, y1))
        return boxes

    def generate(
        self,
        image: Image.Image,
        candidates: list[MaskCandidate],
    ) -> VisualRegionGeneration:
        roots = [
            candidate
            for candidate in candidates
            if candidate.semantic_name == candidate.semantic_parent
            and candidate.metadata.get("root_index") is not None
        ]
        if not roots:
            return VisualRegionGeneration(
                (),
                {
                    "algorithm": "hpid-open-set-visual-regions-v1",
                    "proposal_count": 0,
                    "accepted_candidate_count": 0,
                    "reason": "no_routed_root",
                    "ground_truth_used": False,
                },
            )
        proposals = self._pipeline_proposals(
            image,
            points_per_crop=self.config.points_per_crop,
            offset_xy=(0, 0),
            output_shape=(image.height, image.width),
            scale_level=0,
            view_id="global",
        )
        crop_count = 0
        for layer in range(1, self.config.crops_n_layers + 1):
            crop_points = max(
                8,
                round(
                    self.config.points_per_crop
                    / (self.config.crop_n_points_downscale_factor**layer)
                ),
            )
            for crop_index, (x0, y0, x1, y1) in enumerate(
                self._layer_crop_boxes(image.width, image.height, layer)
            ):
                proposals.extend(
                    self._pipeline_proposals(
                        image.crop((x0, y0, x1, y1)),
                        points_per_crop=crop_points,
                        offset_xy=(x0, y0),
                        output_shape=(image.height, image.width),
                        scale_level=layer,
                        view_id=f"layer-{layer}-crop-{crop_index}",
                    )
                )
                crop_count += 1
        isolated_root_crop_count = 0
        isolated_root_points = max(8, self.config.isolated_root_points_per_crop)
        if self.config.use_isolated_root_crops:
            image_area = max(1, image.width * image.height)
            eligible_roots = [
                root
                for root in roots
                if _area(root.mask) >= self.config.minimum_isolated_root_area_px
                and _area(root.mask) / image_area
                <= self.config.maximum_isolated_root_image_fraction
                and root.metadata.get("scene_role") != "scene_layer"
            ]
            eligible_roots.sort(
                key=lambda root: (-_area(root.mask), _box(root.mask), _root_key(root))
            )
            for root in eligible_roots[: self.config.maximum_isolated_root_crops]:
                rx0, ry0, rx1, ry1 = _box(root.mask)
                width = max(1, rx1 - rx0)
                height = max(1, ry1 - ry0)
                padding_x = max(
                    2, round(width * self.config.isolated_root_crop_padding)
                )
                padding_y = max(
                    2, round(height * self.config.isolated_root_crop_padding)
                )
                x0 = max(0, rx0 - padding_x)
                y0 = max(0, ry0 - padding_y)
                x1 = min(image.width, rx1 + padding_x)
                y1 = min(image.height, ry1 + padding_y)
                if x1 <= x0 or y1 <= y0:
                    continue
                root_key = _root_key(root)
                proposals.extend(
                    self._pipeline_proposals(
                        image.crop((x0, y0, x1, y1)),
                        points_per_crop=isolated_root_points,
                        offset_xy=(x0, y0),
                        output_shape=(image.height, image.width),
                        scale_level=self.config.crops_n_layers + 1,
                        view_id=f"isolated-root-{root_key}",
                        target_root_key=root_key,
                    )
                )
                isolated_root_crop_count += 1
        proposals, consensus_diagnostics = _consolidate_multiview_proposals(
            image, proposals, self.config
        )
        generated = visual_region_candidates_from_masks(
            proposals,
            roots,
            candidates,
            config=self.config,
            source=self.source,
        )
        diagnostics = {
            **generated.diagnostics,
            "multi_view_consensus": consensus_diagnostics,
            "models": {
                "segmentation_model": self.source,
                "points_per_crop": self.config.points_per_crop,
                "crops_n_layers": self.config.crops_n_layers,
                "crop_n_points_downscale_factor": (
                    self.config.crop_n_points_downscale_factor
                ),
                "external_multiscale_crop_count": crop_count,
                "isolated_root_crop_count": isolated_root_crop_count,
                "isolated_root_points_per_crop": isolated_root_points,
                "predicted_iou_threshold": self.config.predicted_iou_threshold,
                "stability_score_threshold": self.config.stability_score_threshold,
            },
        }
        return VisualRegionGeneration(generated.candidates, diagnostics)
