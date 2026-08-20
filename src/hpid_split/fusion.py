from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes

from .instances import PartInstance, semantic_to_part_ids
from .taxonomy import Taxonomy


@dataclass(frozen=True)
class MaskCandidate:
    """One visible-part hypothesis produced without access to ground truth."""

    semantic_name: str
    semantic_parent: str
    mask: np.ndarray
    score: float
    source: str
    prompt: str = ""
    source_reliability: float = 1.0
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mask.ndim != 2:
            raise ValueError("candidate masks must be two-dimensional")
        if not self.semantic_name or not self.semantic_parent:
            raise ValueError("candidate semantic names must not be empty")
        if not np.isfinite(self.score):
            raise ValueError("candidate score must be finite")


@dataclass(frozen=True)
class FusionConfig:
    minimum_area_px: int = 20
    same_source_nms_iou: float = 0.82
    same_source_nms_containment: float = 0.90
    hierarchy_strength: float = 0.65
    orphan_support: float = 0.12
    parent_dilation_ratio: float = 0.018
    dense_parent_dilation_ratio: float = 0.06
    direct_gate_threshold: float = 0.38
    direct_gate_margin: float = 0.82
    detail_bonus: float = 0.08
    identity_match_iou: float = 0.38
    identity_match_containment: float = 0.72
    identity_match_centroid_ratio: float = 0.45
    identity_strong_containment: float = 0.90
    identity_strong_containment_coherence: float = 0.86
    minimum_identity_visible_fraction: float = 0.012
    maximum_identity_sliver_image_fraction: float = 0.001
    standard_component_fraction: float = 0.00005
    detail_component_fraction: float = 0.00001
    cleanup_passes: int = 3
    uncorroborated_source_penalty: float = 1.0
    full_agreement_overlap: float = 0.55
    use_remainder_attachment: bool = True
    remainder_merge_distance_ratio: float = 0.025
    use_consensus: bool = True
    use_parent_support: bool = True
    use_parent_envelope: bool = False
    use_parent_residual: bool = True
    use_root_coverage_conservation: bool = True
    use_transitive_residual: bool = False
    transitive_residual_dense_only: bool = True
    use_direct_gate: bool = True
    use_specificity_ownership: bool = True
    use_hierarchical_duplicate_suppression: bool = True
    hierarchical_duplicate_iou: float = 0.90
    hierarchical_duplicate_containment: float = 0.98
    hierarchical_duplicate_area_coherence: float = 0.90
    hierarchical_duplicate_child_strength_ratio: float = 0.90
    specificity_minimum_containment: float = 0.80
    specificity_minimum_host_fraction: float = 0.012
    semantic_detail_minimum_host_fraction: float = 0.0002
    specificity_maximum_host_fraction: float = 0.78
    structural_specificity_maximum_host_fraction: float = 0.92
    specificity_minimum_candidate_strength: float = 0.24
    specificity_root_minimum_candidate_score: float = 0.36
    specificity_host_suppression: float = 0.28
    specificity_child_evidence_floor_ratio: float = 0.90
    scene_layer_fallback_weight: float = 0.72
    # This remains available as an experimental ablation. On the frozen
    # development set it fragmented components and reduced semantic/boundary
    # scores, so the release path deliberately leaves it disabled.
    use_boundary_ownership: bool = False


@dataclass(frozen=True)
class FusionResult:
    semantic_map: np.ndarray
    instance_map: np.ndarray
    instances: tuple[PartInstance, ...]
    taxonomy: Taxonomy
    evidence: np.ndarray
    accepted_candidates: tuple[MaskCandidate, ...]
    diagnostics: dict[str, object]


@dataclass
class _IdentityAssignment:
    visible_mask: np.ndarray
    identity_mask: np.ndarray
    representative: MaskCandidate
    support_score: float


def taxonomy_from_candidates(candidates: list[MaskCandidate]) -> Taxonomy:
    """Create a deterministic hierarchy when a prompt bank is used at inference."""
    parent_names = ["background"]
    parent_names.extend(sorted({candidate.semantic_parent for candidate in candidates}))
    mapping: dict[str, str] = {}
    for candidate in candidates:
        previous = mapping.setdefault(
            candidate.semantic_name, candidate.semantic_parent
        )
        if previous != candidate.semantic_parent:
            raise ValueError(
                f"semantic name {candidate.semantic_name!r} belongs to both "
                f"{previous!r} and {candidate.semantic_parent!r}; namespace it"
            )
    # A parent fallback class keeps pixels that are not claimed by a child.
    for parent_name in parent_names[1:]:
        mapping.setdefault(parent_name, parent_name)
    fine_names = ["background", *sorted(mapping)]
    parent_lookup = {name: index for index, name in enumerate(parent_names)}
    fine_to_parent = [0]
    fine_to_parent.extend(parent_lookup[mapping[name]] for name in fine_names[1:])
    detail_names = tuple(name for name in fine_names[1:] if name != mapping[name])
    return Taxonomy(
        fine_names=tuple(fine_names),
        parent_names=tuple(parent_names),
        fine_to_parent=tuple(fine_to_parent),
        detail_names=detail_names,
    )


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    union = int(np.count_nonzero(first | second))
    return intersection / union if union else 0.0


def _mask_containment(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    smaller = min(int(np.count_nonzero(first)), int(np.count_nonzero(second)))
    return intersection / smaller if smaller else 0.0


def _source_family(candidate: MaskCandidate) -> str:
    """Collapse correlated stages from one model stack into one evidence source."""
    explicit = candidate.metadata.get("source_family")
    if explicit is not None and str(explicit).strip():
        family = str(explicit).strip()
    else:
        # Stage is the final path-like suffix. Model identifiers inside brackets
        # can themselves contain '/', for example IDEA-Research/grounding-dino-tiny.
        family = candidate.source.rsplit("/", maxsplit=1)[0]
    # Tiny/base checkpoints from one Grounding DINO architecture are correlated
    # proposal sources, not independent consensus votes. Keep the repository and
    # downstream segmenter visible while collapsing only the checkpoint scale.
    family = re.sub(
        r"grounding-dino-(?:tiny|base)",
        "grounding-dino",
        family,
        flags=re.IGNORECASE,
    )
    return family


def _root_key(candidate: MaskCandidate) -> str | None:
    root_index = candidate.metadata.get("root_index")
    if root_index is None:
        return None
    origin = candidate.metadata.get("root_origin")
    return f"{origin if origin is not None else 'legacy'}::{root_index}"


def _is_broad_ownership_host(semantic_name: str) -> bool:
    return semantic_name.endswith(
        (
            "_base_panel",
            "_body",
            "_frame",
            "_housing",
            "_panel",
            "_receiver",
            "_shell",
            "_surface",
            "_torso",
        )
    )


def _identity_scope_key(candidate: MaskCandidate) -> str:
    """Scope per-object instance quotas without separating model evidence."""

    scene_object_id = candidate.metadata.get("scene_object_id")
    if scene_object_id is not None:
        return f"scene::{scene_object_id}"
    return _root_key(candidate) or "unscoped"


def _is_broad_scene_layer(candidate: MaskCandidate) -> bool:
    if candidate.metadata.get("scene_role") != "scene_layer":
        return False
    evidence = candidate.metadata.get("proposal_first_evidence")
    if not isinstance(evidence, dict):
        return False
    return bool(evidence.get("derived_from_background_complement")) or float(
        evidence.get("area_fraction", 0.0)
    ) >= 0.24


def _candidate_dedup_quality(candidate: MaskCandidate) -> float:
    """Rank alternate masks from one correlated model family.

    Detector logits are useful for finding boxes but are not calibrated boundary
    quality. When a candidate passed an independent dense semantic gate, its
    region-text score and contrast therefore participate in representative-mask
    selection. This keeps checkpoint ensembles from replacing a cleaner mask
    merely because one detector assigned a slightly higher box score.
    """

    calibrated = 0.45 + 0.55 * float(np.clip(candidate.score, 0.0, 1.0))
    base = float(np.clip(calibrated * candidate.source_reliability, 0.0, 1.0))
    sam_quality = float(np.clip(candidate.metadata.get("sam_quality", 0.5), 0.0, 1.0))
    dense_top_values = [
        candidate.metadata.get(name)
        for name in (
            "profile_dense_top_mean",
            "guided_dense_top_mean",
            "topology_dense_top_mean",
        )
        if candidate.metadata.get(name) is not None
    ]
    dense_contrast_values = [
        candidate.metadata.get(name)
        for name in (
            "profile_dense_contrast",
            "guided_dense_contrast",
            "topology_dense_contrast",
        )
        if candidate.metadata.get(name) is not None
    ]
    if not dense_top_values:
        return 0.72 * base + 0.28 * sam_quality
    dense_top = float(np.clip(max(map(float, dense_top_values)), 0.0, 1.0))
    dense_contrast = float(
        np.clip(max(map(float, dense_contrast_values), default=0.0) / 0.12, 0.0, 1.0)
    )
    return (
        0.34 * base
        + 0.22 * sam_quality
        + 0.22 * dense_top
        + 0.22 * dense_contrast
    )


def _deduplicate(
    candidates: list[MaskCandidate], config: FusionConfig
) -> tuple[list[MaskCandidate], int]:
    groups: dict[tuple[str, str, str, str, str], list[MaskCandidate]] = {}
    for candidate in candidates:
        if np.count_nonzero(candidate.mask) < config.minimum_area_px:
            continue
        key = (
            _source_family(candidate),
            candidate.semantic_parent,
            candidate.semantic_name,
            str(candidate.metadata.get("parent_candidate_key", "")),
            str(_root_key(candidate) or ""),
        )
        groups.setdefault(key, []).append(candidate)
    accepted: list[MaskCandidate] = []
    removed = 0
    for group in groups.values():
        kept: list[MaskCandidate] = []
        for candidate in sorted(
            group,
            key=lambda item: (_candidate_dedup_quality(item), item.score),
            reverse=True,
        ):
            if any(
                (
                    mask_iou(candidate.mask.astype(bool), item.mask.astype(bool))
                    >= config.same_source_nms_iou
                    or _mask_containment(
                        candidate.mask.astype(bool), item.mask.astype(bool)
                    )
                    >= config.same_source_nms_containment
                )
                for item in kept
            ):
                removed += 1
                continue
            kept.append(candidate)
        accepted.extend(kept)
    return accepted, removed


def _soft_membership(mask: np.ndarray) -> np.ndarray:
    """Turn a hard proposal into a boundary-aware ownership likelihood."""
    binary = mask.astype(np.uint8)
    if not binary.any():
        return np.zeros(mask.shape, dtype=np.float32)
    inside = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    scale = max(1.0, float(np.sqrt(np.count_nonzero(binary))) * 0.08)
    interior = np.clip(inside / scale, 0.0, 1.0)
    blurred = cv2.GaussianBlur(binary.astype(np.float32), (0, 0), 1.1)
    return np.where(binary > 0, 0.52 + 0.48 * interior, 0.12 * blurred).astype(
        np.float32
    )


def _combine_noisy_or(current: np.ndarray, proposal: np.ndarray) -> np.ndarray:
    return 1.0 - (1.0 - current) * (1.0 - proposal)


def _source_agreement_factors(
    candidates: list[MaskCandidate], config: FusionConfig
) -> tuple[list[float], int]:
    """Calibrate complementary sources without suppressing single-source classes."""
    factors = [1.0] * len(candidates)
    uncorroborated = 0
    by_semantic: dict[str, list[int]] = {}
    for index, candidate in enumerate(candidates):
        by_semantic.setdefault(candidate.semantic_name, []).append(index)
    for indices in by_semantic.values():
        families = {_source_family(candidates[index]) for index in indices}
        if len(families) < 2:
            continue
        for index in indices:
            candidate = candidates[index]
            family = _source_family(candidate)
            area = int(np.count_nonzero(candidate.mask))
            best_overlap = 0.0
            for other_index in indices:
                if other_index == index:
                    continue
                other = candidates[other_index]
                if _source_family(other) == family:
                    continue
                intersection = int(np.count_nonzero(candidate.mask & other.mask))
                if not intersection:
                    continue
                other_area = int(np.count_nonzero(other.mask))
                union = area + other_area - intersection
                overlap = max(
                    intersection / max(1, union),
                    intersection / max(1, min(area, other_area)),
                )
                best_overlap = max(best_overlap, overlap)
            agreement = min(1.0, best_overlap / config.full_agreement_overlap)
            factors[index] = config.uncorroborated_source_penalty + (
                1.0 - config.uncorroborated_source_penalty
            ) * agreement
            if best_overlap == 0.0:
                uncorroborated += 1
    return factors, uncorroborated


def _parent_support(
    evidence: np.ndarray,
    taxonomy: Taxonomy,
    accepted: list[MaskCandidate],
    name_to_id: dict[str, int],
) -> np.ndarray:
    support = np.zeros(
        (taxonomy.num_parent_classes, *evidence.shape[1:]), dtype=np.float32
    )
    for parent_id, parent_name in enumerate(taxonomy.parent_names[1:], start=1):
        if parent_name in name_to_id:
            support[parent_id] = evidence[name_to_id[parent_name]]
        child_ids = taxonomy.child_ids(parent_id, include_parent_fallback=False)
        if child_ids and not support[parent_id].any():
            # Child-derived support is only a fallback when the proposal front end
            # produced no root mask at all. Otherwise an orphan child would be
            # able to validate its own hierarchy constraint.
            support[parent_id] = np.maximum(
                support[parent_id], evidence[list(child_ids)].max(axis=0) * 0.72
            )
    # Root proposals are allowed to have an application label distinct from the
    # parent fallback class, so also aggregate candidates by semantic_parent.
    for candidate in accepted:
        if candidate.semantic_name != candidate.semantic_parent:
            continue
        parent_id = taxonomy.parent_names.index(candidate.semantic_parent)
        fine_id = name_to_id[candidate.semantic_name]
        support[parent_id] = np.maximum(support[parent_id], evidence[fine_id])
    # A profile can name an intermediate semantic parent without producing a
    # separate mask for that parent (for example, trigger -> body -> firearm).
    # In that case the explicit parent candidate key links the child to the
    # selected root.  Reuse that root envelope instead of treating the child as
    # an orphan and silently deleting a valid small semantic part.
    candidate_by_key = {
        str(candidate.metadata["candidate_key"]): candidate
        for candidate in accepted
        if candidate.metadata.get("candidate_key") is not None
    }
    for candidate in accepted:
        if candidate.semantic_name == candidate.semantic_parent:
            continue
        try:
            parent_id = taxonomy.parent_names.index(candidate.semantic_parent)
        except ValueError:
            continue
        if any(
            other.semantic_name == candidate.semantic_parent
            and _root_key(other) == _root_key(candidate)
            for other in accepted
        ):
            continue
        parent_key = candidate.metadata.get("parent_candidate_key")
        parent = candidate_by_key.get(str(parent_key))
        if parent is None or parent.semantic_name not in name_to_id:
            continue
        support[parent_id] = np.maximum(
            support[parent_id],
            evidence[name_to_id[parent.semantic_name]],
        )
    return support


def _hard_parent_support(
    taxonomy: Taxonomy,
    accepted: list[MaskCandidate],
    image_shape: tuple[int, int],
    *,
    fill_internal_holes: bool,
) -> np.ndarray:
    support = np.zeros((taxonomy.num_parent_classes, *image_shape), dtype=bool)
    parent_lookup = {name: index for index, name in enumerate(taxonomy.parent_names)}
    for candidate in accepted:
        if candidate.semantic_name not in parent_lookup:
            continue
        candidate_support = candidate.mask.astype(bool)
        if fill_internal_holes:
            candidate_support = binary_fill_holes(candidate_support).astype(bool)
        support[parent_lookup[candidate.semantic_name]] |= candidate_support
    candidate_by_key = {
        str(candidate.metadata["candidate_key"]): candidate
        for candidate in accepted
        if candidate.metadata.get("candidate_key") is not None
    }
    for candidate in accepted:
        if candidate.semantic_name == candidate.semantic_parent:
            continue
        parent_id = parent_lookup.get(candidate.semantic_parent)
        if parent_id is None or any(
            other.semantic_name == candidate.semantic_parent
            and _root_key(other) == _root_key(candidate)
            for other in accepted
        ):
            continue
        parent_key = candidate.metadata.get("parent_candidate_key")
        parent = candidate_by_key.get(str(parent_key))
        if parent is None:
            continue
        candidate_support = parent.mask.astype(bool)
        if fill_internal_holes:
            candidate_support = binary_fill_holes(candidate_support).astype(bool)
        support[parent_id] |= candidate_support
    return support


def _dilate_support(support: np.ndarray, ratio: float) -> np.ndarray:
    height, width = support.shape
    radius = max(1, round(min(height, width) * ratio))
    size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.dilate(support, kernel)


def _descendant_ids(taxonomy: Taxonomy, parent_name: str) -> tuple[int, ...]:
    parent_lookup = {name: index for index, name in enumerate(taxonomy.parent_names)}
    descendants: set[int] = set()

    def visit(name: str, active: frozenset[str] = frozenset()) -> None:
        if name in active:
            raise ValueError(f"semantic hierarchy contains a cycle at {name!r}")
        parent_id = parent_lookup.get(name)
        if parent_id is None:
            return
        direct = taxonomy.child_ids(parent_id, include_parent_fallback=False)
        for class_id in direct:
            if class_id in descendants:
                continue
            descendants.add(class_id)
            child_name = taxonomy.fine_names[class_id]
            if child_name != name:
                visit(child_name, active | {name})

    visit(parent_name)
    return tuple(sorted(descendants))


def _clean_labels(
    labels: np.ndarray,
    taxonomy: Taxonomy,
    minimum_area: int,
    config: FusionConfig,
) -> np.ndarray:
    cleaned = labels.copy()
    image_area = int(labels.size)
    parent_fallback = {
        parent_id: taxonomy.fine_names.index(parent_name)
        for parent_id, parent_name in enumerate(taxonomy.parent_names)
        if parent_name in taxonomy.fine_names
    }
    kernel = np.ones((3, 3), dtype=np.uint8)
    for _ in range(max(1, config.cleanup_passes)):
        changed = 0
        for class_id in range(1, taxonomy.num_fine_classes):
            count, components, stats, _ = cv2.connectedComponentsWithStats(
                (cleaned == class_id).astype(np.uint8), 8
            )
            parent_replacement = parent_fallback.get(
                taxonomy.fine_to_parent[class_id], 0
            )
            relative_fraction = (
                config.detail_component_fraction
                if class_id in taxonomy.detail_ids
                else config.standard_component_fraction
            )
            threshold = max(minimum_area, round(image_area * relative_fraction))
            for component_id in range(1, count):
                if int(stats[component_id, cv2.CC_STAT_AREA]) >= threshold:
                    continue
                component = components == component_id
                replacement = parent_replacement
                if replacement == class_id:
                    ring = (
                        cv2.dilate(component.astype(np.uint8), kernel).astype(bool)
                        & ~component
                    )
                    neighbors = cleaned[ring]
                    neighbors = neighbors[neighbors != class_id]
                    if len(neighbors):
                        values, counts = np.unique(neighbors, return_counts=True)
                        replacement = int(values[int(np.argmax(counts))])
                    else:
                        replacement = 0
                cleaned[component] = replacement
                changed += 1
        if not changed:
            break
    return cleaned


def _geometry(
    mask: np.ndarray,
) -> tuple[tuple[int, int, int, int], tuple[float, float], int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("cannot describe an empty part mask")
    return (
        (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)),
        (float(xs.mean()), float(ys.mean())),
        len(xs),
    )


def _mask_coherence(mask: np.ndarray) -> float:
    area = int(np.count_nonzero(mask))
    if area == 0:
        return 0.0
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    if count <= 1:
        return 0.0
    return float(stats[1:, cv2.CC_STAT_AREA].max()) / area


def _identity_affinity(
    first: MaskCandidate,
    second: MaskCandidate,
    config: FusionConfig,
) -> float | None:
    if first.semantic_name != second.semantic_name:
        return None
    intersection = int(np.count_nonzero(first.mask & second.mask))
    if not intersection:
        return None
    first_area = int(np.count_nonzero(first.mask))
    second_area = int(np.count_nonzero(second.mask))
    union = first_area + second_area - intersection
    iou = intersection / max(1, union)
    containment = intersection / max(1, min(first_area, second_area))
    first_center = _geometry(first.mask)[1]
    second_center = _geometry(second.mask)[1]
    centroid_distance = float(
        np.hypot(
            first_center[0] - second_center[0],
            first_center[1] - second_center[1],
        )
    )
    centroid_ratio = centroid_distance / max(1.0, np.sqrt(min(first_area, second_area)))
    if iou >= config.identity_match_iou:
        return iou
    larger_mask = first.mask if first_area >= second_area else second.mask
    larger_coherence = _mask_coherence(larger_mask)
    # Visual proposals may cover only one end of a connected garment or rigid
    # part. Strong containment then supports the same identity. A disconnected
    # broad mask (for example, both shoes) cannot collapse repeated instances.
    if (
        containment >= config.identity_strong_containment
        and larger_coherence >= config.identity_strong_containment_coherence
    ):
        return 0.55 + 0.45 * containment
    if (
        containment >= config.identity_match_containment
        and centroid_ratio <= config.identity_match_centroid_ratio
    ):
        return 0.5 * (containment + (1.0 - centroid_ratio))
    return None


def _cluster_identity_candidates(
    candidates: list[MaskCandidate], config: FusionConfig
) -> list[list[MaskCandidate]]:
    """Associate cross-source masks that describe the same physical part."""
    ordered = sorted(
        candidates,
        key=lambda item: (
            _mask_coherence(item.mask)
            >= config.identity_strong_containment_coherence,
            _mask_coherence(item.mask),
            int(np.count_nonzero(item.mask)),
            item.score * item.source_reliability,
        ),
        reverse=True,
    )
    groups: list[list[MaskCandidate]] = []
    for candidate in ordered:
        compatible: list[tuple[float, int]] = []
        for group_index, group in enumerate(groups):
            # The first member is a coherent, area-prioritized anchor. Matching
            # every pair fragments one object when two local supports lie at
            # opposite ends and do not overlap each other. Matching the anchor
            # avoids that while disconnected repeated parts form their own
            # anchor groups before a broad pair-mask is considered.
            affinity = _identity_affinity(candidate, group[0], config)
            if affinity is not None:
                compatible.append((affinity, group_index))
        if compatible:
            _, group_index = max(compatible)
            groups[group_index].append(candidate)
        else:
            groups.append([candidate])
    groups.sort(
        key=lambda group: _geometry(
            np.logical_or.reduce([candidate.mask for candidate in group])
        )[1]
    )
    return groups


def _limit_identity_groups(
    groups: list[list[MaskCandidate]], maximum_instances: int
) -> tuple[list[list[MaskCandidate]], int]:
    """Keep the strongest bounded identities without joining unrelated masks.

    An instance cap is an ontology constraint, not evidence that disconnected
    hypotheses belong to one physical part.  Overflow masks therefore remain
    rejected hypotheses; their pixels can still be owned by the enclosing root
    during semantic fusion, but they must not create a stitched Part ID.
    """

    if maximum_instances < 1 or len(groups) <= maximum_instances:
        return groups, 0
    ordered = sorted(
        groups,
        key=lambda group: max(
            candidate.score * candidate.source_reliability
            for candidate in group
        ),
        reverse=True,
    )
    kept = [list(group) for group in ordered[:maximum_instances]]
    return kept, len(groups) - len(kept)


def _scoped_identity_groups(
    candidates: list[MaskCandidate],
    config: FusionConfig,
) -> tuple[list[list[MaskCandidate]], int, int]:
    """Associate masks and apply semantic instance caps per physical root."""

    candidates_by_scope: dict[str, list[MaskCandidate]] = {}
    for candidate in candidates:
        candidates_by_scope.setdefault(_identity_scope_key(candidate), []).append(
            candidate
        )
    groups: list[list[MaskCandidate]] = []
    merged_by_cap = 0
    capped_scope_count = 0
    for scoped_candidates in candidates_by_scope.values():
        scoped_groups = _cluster_identity_candidates(scoped_candidates, config)
        instance_caps = [
            int(candidate.metadata["maximum_instances"])
            for candidate in scoped_candidates
            if candidate.metadata.get("maximum_instances") is not None
            and int(candidate.metadata["maximum_instances"]) > 0
        ]
        if instance_caps:
            capped_scope_count += 1
            scoped_groups, scoped_merged = _limit_identity_groups(
                scoped_groups,
                min(instance_caps),
            )
            merged_by_cap += scoped_merged
        groups.extend(scoped_groups)
    groups.sort(
        key=lambda group: _geometry(
            np.logical_or.reduce([candidate.mask for candidate in group])
        )[1]
    )
    return groups, merged_by_cap, capped_scope_count


def _filter_candidates_by_instance_caps(
    candidates: list[MaskCandidate],
    config: FusionConfig,
) -> tuple[list[MaskCandidate], int, int]:
    """Remove overflow identity hypotheses before semantic ownership fusion."""

    candidates_by_semantic: dict[str, list[MaskCandidate]] = {}
    for candidate in candidates:
        candidates_by_semantic.setdefault(candidate.semantic_name, []).append(
            candidate
        )
    kept_ids: set[int] = set()
    dropped_groups = 0
    capped_scopes = 0
    for semantic_candidates in candidates_by_semantic.values():
        groups, semantic_dropped, semantic_scopes = _scoped_identity_groups(
            semantic_candidates,
            config,
        )
        kept_ids.update(id(candidate) for group in groups for candidate in group)
        dropped_groups += semantic_dropped
        capped_scopes += semantic_scopes
    return (
        [candidate for candidate in candidates if id(candidate) in kept_ids],
        dropped_groups,
        capped_scopes,
    )


def _suppress_hierarchical_duplicates(
    candidates: list[MaskCandidate], config: FusionConfig
) -> tuple[list[MaskCandidate], int]:
    """Remove a parent proposal only when a stronger child is the same region."""

    if not config.use_hierarchical_duplicate_suppression:
        return candidates, 0
    dropped: set[int] = set()
    for parent in candidates:
        if parent.semantic_name == parent.semantic_parent:
            continue
        parent_root = _root_key(parent)
        if parent_root is None:
            continue
        parent_area = max(1, int(np.count_nonzero(parent.mask)))
        parent_strength = (
            0.45 + 0.55 * float(np.clip(parent.score, 0.0, 1.0))
        ) * parent.source_reliability
        for child in candidates:
            if (
                child is parent
                or child.semantic_parent != parent.semantic_name
                or _root_key(child) != parent_root
            ):
                continue
            child_area = max(1, int(np.count_nonzero(child.mask)))
            intersection = int(np.count_nonzero(parent.mask & child.mask))
            if not intersection:
                continue
            union = parent_area + child_area - intersection
            iou = intersection / max(1, union)
            containment = intersection / min(parent_area, child_area)
            area_coherence = min(parent_area, child_area) / max(
                parent_area, child_area
            )
            near_duplicate = iou >= config.hierarchical_duplicate_iou or (
                containment >= config.hierarchical_duplicate_containment
                and area_coherence
                >= config.hierarchical_duplicate_area_coherence
            )
            child_strength = (
                0.45 + 0.55 * float(np.clip(child.score, 0.0, 1.0))
            ) * child.source_reliability
            if near_duplicate and child_strength >= (
                parent_strength
                * config.hierarchical_duplicate_child_strength_ratio
            ):
                dropped.add(id(parent))
                break
    return [candidate for candidate in candidates if id(candidate) not in dropped], len(
        dropped
    )


def _side(
    centroid_x: float,
    reference_x: float,
    width: int,
    bbox_x: tuple[int, int] | None = None,
) -> str:
    if bbox_x is not None and bbox_x[0] <= reference_x < bbox_x[1]:
        return "center"
    dead_zone = max(2.0, width * 0.025)
    if centroid_x < reference_x - dead_zone:
        return "left"
    if centroid_x > reference_x + dead_zone:
        return "right"
    return "center"


def _is_identity_visibility_sliver(
    visible_area: int,
    identity_area: int,
    image_area: int,
    minimum_area: int,
    config: FusionConfig,
) -> bool:
    if visible_area <= 0 or identity_area <= 0:
        return False
    maximum_sliver_area = max(
        minimum_area * 4,
        round(image_area * config.maximum_identity_sliver_image_fraction),
    )
    return bool(
        visible_area < maximum_sliver_area
        and visible_area / identity_area
        < config.minimum_identity_visible_fraction
    )


def _hierarchical_part_ids(
    labels: np.ndarray,
    taxonomy: Taxonomy,
    candidates: list[MaskCandidate],
    minimum_area: int,
    config: FusionConfig,
) -> tuple[np.ndarray, list[PartInstance], dict[str, int]]:
    """Propagate candidate identity through an arbitrary-depth semantic DAG."""
    _, width = labels.shape
    foreground_x = np.nonzero(labels > 0)[1]
    reference_x = float(np.median(foreground_x)) if len(foreground_x) else width / 2.0
    instance_map = np.zeros(labels.shape, dtype=np.uint16)
    records: list[PartInstance] = []
    record_by_index: dict[int, PartInstance] = {}
    supports: list[
        tuple[
            str,
            np.ndarray,
            float,
            int,
            int | str | None,
            str | None,
            str,
        ]
    ] = []
    counters: dict[tuple[str, str, str], int] = {}
    numeric_id = 1
    identity_hypothesis_count = 0
    identity_group_count = 0
    identity_groups_dropped_by_instance_cap = 0
    identity_instance_cap_scope_count = 0
    remainder_components_merged = 0
    visibility_slivers_dropped = 0

    parent_by_name = {
        taxonomy.fine_names[class_id]: taxonomy.parent_names[
            taxonomy.fine_to_parent[class_id]
        ]
        for class_id in range(1, taxonomy.num_fine_classes)
    }
    dependencies: dict[str, set[str]] = {name: set() for name in parent_by_name}
    for name, parent in parent_by_name.items():
        if parent != name and parent in dependencies:
            dependencies[name].add(parent)
    for candidate in candidates:
        assembly_parent = candidate.metadata.get("assembly_parent_semantic")
        if (
            assembly_parent is not None
            and str(assembly_parent) != candidate.semantic_name
            and str(assembly_parent) in dependencies
        ):
            dependencies[candidate.semantic_name].add(str(assembly_parent))

    depth_cache: dict[str, int] = {}

    def semantic_depth(name: str, active: frozenset[str] = frozenset()) -> int:
        if name in depth_cache:
            return depth_cache[name]
        if name in active:
            raise ValueError(f"semantic hierarchy contains a cycle at {name!r}")
        parents = dependencies.get(name, set())
        if not parents:
            depth = 0
        else:
            depth = 1 + max(
                semantic_depth(parent, active | {name}) for parent in parents
            )
        depth_cache[name] = depth
        return depth

    class_order = sorted(
        range(1, taxonomy.num_fine_classes),
        key=lambda class_id: (
            semantic_depth(taxonomy.fine_names[class_id]),
            class_id,
        ),
    )

    def find_assembly_parent(
        support_mask: np.ndarray,
        parent_name: str,
        root_key: int | str | None,
        parent_candidate_key: str | None,
        asset_id: str,
    ) -> str | None:
        if parent_name not in parent_by_name:
            return None
        eligible = [
            item
            for item in supports
            if item[0] == parent_name and item[6] == asset_id
        ]
        if parent_candidate_key is not None:
            exact = [
                item
                for item in eligible
                if item[5] == parent_candidate_key
                and (root_key is None or item[4] == root_key)
            ]
            if exact:
                return record_by_index[max(exact, key=lambda item: item[2])[3]].part_id
        if root_key is not None:
            keyed = [item for item in eligible if item[4] == root_key]
            if keyed:
                eligible = keyed
        area = max(1, int(np.count_nonzero(support_mask)))
        scored: list[tuple[float, float, float, int]] = []
        for _, parent_support, support_score, instance_index, _, _, _ in eligible:
            overlap = np.count_nonzero(support_mask & parent_support) / area
            distance = cv2.distanceTransform(
                (~parent_support).astype(np.uint8), cv2.DIST_L2, 3
            )
            minimum_distance = float(distance[support_mask].min(initial=np.inf))
            parent_scale = max(3.0, np.sqrt(np.count_nonzero(parent_support)))
            normalized_distance = minimum_distance / parent_scale
            scored.append(
                (float(overlap), -normalized_distance, support_score, instance_index)
            )
        if not scored:
            return None
        overlap, negative_distance, _, parent_index = max(scored)
        if overlap < 0.02 and -negative_distance > 0.25:
            return None
        return record_by_index[parent_index].part_id

    def find_root_parent(
        candidate_key: str | None,
        root_key: int | str | None,
        asset_id: str,
    ) -> str | None:
        if candidate_key is None or "/visual-region:" not in candidate_key:
            return None
        root_candidate_key = candidate_key.split("/visual-region:", maxsplit=1)[0]
        eligible = [
            item
            for item in supports
            if item[5] == root_candidate_key
            and item[6] == asset_id
            and (root_key is None or item[4] == root_key)
        ]
        if not eligible:
            return None
        return record_by_index[max(eligible, key=lambda item: item[2])[3]].part_id

    def append_record(
        semantic_name: str,
        semantic_parent: str,
        visible_mask: np.ndarray,
        *,
        identity_mask: np.ndarray,
        support_score: float = 1.0,
        root_key: int | str | None = None,
        candidate_key: str | None = None,
        assembly_parent_semantic: str | None = None,
        assembly_parent_candidate_key: str | None = None,
        asset_id: str = "object_001",
    ) -> None:
        nonlocal numeric_id
        bbox, centroid, area = _geometry(visible_mask)
        identity_bbox, identity_centroid, _ = _geometry(identity_mask)
        assembly_semantic = assembly_parent_semantic or semantic_parent
        assembly_parent_id = (
            None
            if semantic_name == assembly_semantic
            else find_assembly_parent(
                identity_mask,
                assembly_semantic,
                root_key,
                assembly_parent_candidate_key,
                asset_id,
            )
        )
        if (
            assembly_parent_id is None
            and semantic_name != semantic_parent
            and assembly_semantic != semantic_parent
        ):
            assembly_parent_id = find_assembly_parent(
                identity_mask,
                semantic_parent,
                root_key,
                None,
                asset_id,
            )
        if assembly_parent_id is None and semantic_name != semantic_parent:
            assembly_parent_id = find_root_parent(
                candidate_key,
                root_key,
                asset_id,
            )
        parent_record = next(
            (
                record
                for record in records
                if record.part_id == assembly_parent_id
            ),
            None,
        )
        if (
            parent_record is not None
            and semantic_name != semantic_parent
            and not any(
                item[0] == semantic_parent
                and item[4] == root_key
                and item[6] == asset_id
                for item in supports
            )
        ):
            semantic_parent = parent_record.semantic_name
        side_reference_x = (
            parent_record.centroid_xy[0] if parent_record is not None else reference_x
        )
        side = _side(
            identity_centroid[0],
            side_reference_x,
            width,
            (identity_bbox[0], identity_bbox[2]),
        )
        key = (semantic_parent, semantic_name, side)
        counters[key] = counters.get(key, 0) + 1
        part_id = f"{semantic_parent}/{semantic_name}/{side}/{counters[key]:02d}"
        record = PartInstance(
            part_id=part_id,
            semantic_name=semantic_name,
            semantic_parent=semantic_parent,
            instance_index=numeric_id,
            side=side,
            bbox_xyxy=bbox,
            centroid_xy=centroid,
            area_px=area,
            asset_id=asset_id,
            assembly_parent_id=assembly_parent_id,
        )
        instance_map[visible_mask] = numeric_id
        records.append(record)
        record_by_index[numeric_id] = record
        supports.append(
            (
                semantic_name,
                identity_mask.astype(bool),
                support_score,
                numeric_id,
                root_key,
                candidate_key,
                asset_id,
            )
        )
        numeric_id += 1

    for class_id in class_order:
        semantic_name = taxonomy.fine_names[class_id]
        semantic_parent = parent_by_name[semantic_name]
        class_mask = labels == class_id
        class_candidates = [
            candidate
            for candidate in candidates
            if candidate.semantic_name == semantic_name
        ]
        claimed = np.zeros(labels.shape, dtype=bool)
        assignments: list[_IdentityAssignment] = []
        if class_candidates:
            identity_hypothesis_count += len(class_candidates)
            (
                identity_groups,
                merged_by_cap,
                capped_scope_count,
            ) = _scoped_identity_groups(class_candidates, config)
            identity_groups_dropped_by_instance_cap += merged_by_cap
            identity_instance_cap_scope_count += capped_scope_count
            identity_group_count += len(identity_groups)
            ownership_scores = np.stack(
                [
                    np.maximum.reduce(
                        [
                            _soft_membership(candidate.mask)
                            * (
                                0.45
                                + 0.55
                                * float(np.clip(candidate.score, 0.0, 1.0))
                            )
                            * candidate.source_reliability
                            * (
                                config.scene_layer_fallback_weight
                                if _is_broad_scene_layer(candidate)
                                else 1.0
                            )
                            * candidate.mask.astype(np.float32)
                            for candidate in group
                        ]
                    )
                    for group in identity_groups
                ],
                axis=0,
            )
            owner = ownership_scores.argmax(axis=0)
            supported = ownership_scores.max(axis=0) > 0.0
            for group_index, group in enumerate(identity_groups):
                representative = max(
                    group,
                    key=lambda item: item.score * item.source_reliability,
                )
                identity_mask = np.logical_or.reduce(
                    [candidate.mask for candidate in group]
                )
                visible = class_mask & supported & (owner == group_index) & ~claimed
                if not np.any(visible):
                    continue
                assignments.append(
                    _IdentityAssignment(
                        visible_mask=visible,
                        identity_mask=identity_mask,
                        representative=representative,
                        support_score=max(candidate.score for candidate in group),
                    )
                )
                claimed |= visible

        remainder = class_mask & ~claimed
        count, components, stats, _ = cv2.connectedComponentsWithStats(
            remainder.astype(np.uint8), 8
        )
        component_masks = [
            components == component_id
            for component_id in range(1, count)
            if int(stats[component_id, cv2.CC_STAT_AREA]) >= minimum_area
        ]
        component_masks.sort(key=lambda mask: _geometry(mask)[1])
        detached_components: list[np.ndarray] = []
        if assignments and config.use_remainder_attachment:
            maximum_distance = max(
                2.0,
                min(labels.shape) * config.remainder_merge_distance_ratio,
            )
            distance_maps = [
                cv2.distanceTransform(
                    (~assignment.visible_mask).astype(np.uint8),
                    cv2.DIST_L2,
                    3,
                )
                for assignment in assignments
            ]
            for component in component_masks:
                overlap_scores = [
                    int(np.count_nonzero(component & assignment.identity_mask))
                    for assignment in assignments
                ]
                if max(overlap_scores, default=0) > 0:
                    assignment_index = int(np.argmax(overlap_scores))
                else:
                    distances = [
                        float(distance[component].min(initial=np.inf))
                        for distance in distance_maps
                    ]
                    assignment_index = int(np.argmin(distances))
                    if distances[assignment_index] > maximum_distance:
                        detached_components.append(component)
                        continue
                assignments[assignment_index].visible_mask |= component
                remainder_components_merged += 1
        else:
            detached_components = component_masks

        for assignment in assignments:
            visible_area = int(np.count_nonzero(assignment.visible_mask))
            identity_area = int(np.count_nonzero(assignment.identity_mask))
            if visible_area < minimum_area:
                detached_components.append(assignment.visible_mask)
                continue
            if _is_identity_visibility_sliver(
                visible_area,
                identity_area,
                labels.size,
                minimum_area,
                config,
            ):
                visibility_slivers_dropped += 1
                continue
            representative = assignment.representative
            append_record(
                semantic_name,
                semantic_parent,
                assignment.visible_mask,
                identity_mask=assignment.identity_mask,
                support_score=assignment.support_score,
                root_key=_root_key(representative),
                candidate_key=(
                    str(representative.metadata["candidate_key"])
                    if representative.metadata.get("candidate_key") is not None
                    else None
                ),
                assembly_parent_semantic=(
                    str(representative.metadata["assembly_parent_semantic"])
                    if representative.metadata.get("assembly_parent_semantic")
                    is not None
                    else semantic_parent
                ),
                assembly_parent_candidate_key=(
                    str(representative.metadata["assembly_parent_candidate_key"])
                    if representative.metadata.get("assembly_parent_candidate_key")
                    is not None
                    else (
                        str(representative.metadata["parent_candidate_key"])
                        if representative.metadata.get("parent_candidate_key")
                        is not None
                        and representative.metadata.get(
                            "assembly_parent_semantic", semantic_parent
                        )
                        == semantic_parent
                        else None
                    )
                ),
                asset_id=str(
                    representative.metadata.get("scene_object_id") or "object_001"
                ),
            )

        for component in detached_components:
            if np.count_nonzero(component) < minimum_area:
                continue
            component_candidates = [
                candidate
                for candidate in class_candidates
                if np.any(candidate.mask & component)
            ]
            component_representative = (
                max(
                    component_candidates,
                    key=lambda candidate: int(
                        np.count_nonzero(candidate.mask & component)
                    ),
                )
                if component_candidates
                else None
            )
            append_record(
                semantic_name,
                semantic_parent,
                component,
                identity_mask=component,
                root_key=(
                    _root_key(component_representative)
                    if component_representative is not None
                    else None
                ),
                asset_id=(
                    str(
                        component_representative.metadata.get(
                            "scene_object_id"
                        )
                        or "object_001"
                    )
                    if component_representative is not None
                    else "object_001"
                ),
            )
    return instance_map, records, {
        "identity_hypothesis_count": identity_hypothesis_count,
        "identity_group_count": identity_group_count,
        "identity_hypotheses_merged": (
            identity_hypothesis_count - identity_group_count
        ),
        "identity_groups_dropped_by_instance_cap": (
            identity_groups_dropped_by_instance_cap
        ),
        # Kept as a compatibility field. Product inference no longer merges
        # disconnected identities merely to satisfy an ontology cap.
        "identity_groups_merged_by_instance_cap": 0,
        "identity_instance_cap_scope_count": identity_instance_cap_scope_count,
        "remainder_components_attached": remainder_components_merged,
        "identity_visibility_slivers_dropped": visibility_slivers_dropped,
    }


def fuse_candidates(
    candidates: list[MaskCandidate],
    *,
    image_shape: tuple[int, int] | None = None,
    taxonomy: Taxonomy | None = None,
    config: FusionConfig | None = None,
) -> FusionResult:
    """Fuse heterogeneous masks into exclusive, hierarchy-consistent Part IDs.

    This function is the model-independent HPID fusion path. It deliberately has
    no target/ground-truth parameter, which keeps evaluation labels outside the
    inference boundary.
    """
    config = config or FusionConfig()
    if not candidates and image_shape is None:
        raise ValueError("image_shape is required when no candidates are present")
    if image_shape is None:
        image_shape = tuple(int(value) for value in candidates[0].mask.shape)
    if any(candidate.mask.shape != image_shape for candidate in candidates):
        raise ValueError("all candidate masks must match image_shape")
    accepted, removed = _deduplicate(candidates, config)
    accepted, capped_groups_dropped, capped_scopes = (
        _filter_candidates_by_instance_caps(accepted, config)
    )
    accepted, hierarchical_duplicates_suppressed = (
        _suppress_hierarchical_duplicates(accepted, config)
    )
    taxonomy = taxonomy or taxonomy_from_candidates(accepted)
    name_to_id = {name: index for index, name in enumerate(taxonomy.fine_names)}
    unknown = sorted(
        {
            candidate.semantic_name
            for candidate in accepted
            if candidate.semantic_name not in name_to_id
        }
    )
    if unknown:
        raise ValueError(f"taxonomy is missing candidate classes: {unknown}")

    height, width = image_shape
    evidence = np.zeros((taxonomy.num_fine_classes, height, width), dtype=np.float32)
    direct = np.zeros_like(evidence)
    source_sets: dict[int, set[str]] = {
        class_id: set() for class_id in range(taxonomy.num_fine_classes)
    }
    source_evidence: dict[tuple[int, str], np.ndarray] = {}
    agreement_factors, uncorroborated_count = _source_agreement_factors(
        accepted, config
    )
    for candidate, agreement_factor in zip(
        accepted, agreement_factors, strict=True
    ):
        class_id = name_to_id[candidate.semantic_name]
        # Detector scores rank proposals but are not calibrated pixel
        # probabilities. A mask accepted by the segmenter therefore receives a
        # foreground floor while the detector score still controls conflicts.
        calibrated_score = 0.45 + 0.55 * float(np.clip(candidate.score, 0.0, 1.0))
        reliability = float(
            np.clip(
                calibrated_score
                * candidate.source_reliability
                * agreement_factor,
                0.01,
                0.995,
            )
        )
        if _is_broad_scene_layer(candidate):
            reliability = float(
                np.clip(
                    reliability * config.scene_layer_fallback_weight,
                    0.01,
                    0.995,
                )
            )
        proposal = _soft_membership(candidate.mask) * reliability
        direct[class_id] = np.maximum(direct[class_id], proposal)
        family = _source_family(candidate)
        key = (class_id, family)
        if key not in source_evidence:
            source_evidence[key] = proposal
        else:
            source_evidence[key] = np.maximum(source_evidence[key], proposal)
        source_sets[class_id].add(family)

    # Repeated prompts and hierarchy stages from one model are correlated.
    # They contribute their maximum support once; only independent model
    # families receive noisy-OR consensus credit.
    for (class_id, _), proposal in source_evidence.items():
        if config.use_consensus:
            evidence[class_id] = _combine_noisy_or(evidence[class_id], proposal)
        else:
            evidence[class_id] = np.maximum(evidence[class_id], proposal)

    parent_support = _parent_support(evidence, taxonomy, accepted, name_to_id)
    hard_parent_support = _hard_parent_support(
        taxonomy,
        accepted,
        image_shape,
        fill_internal_holes=False,
    )
    root_union = np.zeros(image_shape, dtype=bool)
    for candidate in accepted:
        if candidate.semantic_name == candidate.semantic_parent:
            root_union |= candidate.mask.astype(bool)
    envelope_parent_support = (
        _hard_parent_support(
            taxonomy,
            accepted,
            image_shape,
            fill_internal_holes=True,
        )
        if config.use_parent_envelope
        else hard_parent_support
    )
    dense_class_ids = {
        name_to_id[candidate.semantic_name]
        for candidate in accepted
        if bool(candidate.metadata.get("dense_semantic_fallback"))
    }
    if config.use_parent_support:
        for class_id in range(1, taxonomy.num_fine_classes):
            parent_id = taxonomy.fine_to_parent[class_id]
            if taxonomy.fine_names[class_id] == taxonomy.parent_names[parent_id]:
                continue
            class_parent_support = (
                envelope_parent_support[parent_id]
                if class_id in dense_class_ids
                else hard_parent_support[parent_id]
            )
            if class_parent_support.any():
                allowed = _dilate_support(
                    class_parent_support.astype(np.float32),
                    (
                        config.dense_parent_dilation_ratio
                        if class_id in dense_class_ids
                        else config.parent_dilation_ratio
                    ),
                ).astype(bool)
                outside_penalty = config.orphan_support**config.hierarchy_strength
                evidence[class_id] *= np.where(allowed, 1.0, outside_penalty)

    # A root proposal represents the whole asset before decomposition. Once a
    # child has credible evidence, the root becomes a residual/fallback label in
    # that region instead of competing with the child at its full box score.
    if config.use_parent_residual:
        for parent_id, parent_name in enumerate(taxonomy.parent_names[1:], start=1):
            if parent_name not in name_to_id:
                continue
            direct_child_ids = taxonomy.child_ids(
                parent_id, include_parent_fallback=False
            )
            if config.use_transitive_residual:
                descendant_ids = _descendant_ids(taxonomy, parent_name)
                if config.transitive_residual_dense_only:
                    descendant_ids = tuple(
                        class_id
                        for class_id in descendant_ids
                        if class_id in dense_class_ids
                    )
                child_ids = tuple(sorted({*direct_child_ids, *descendant_ids}))
            else:
                child_ids = direct_child_ids
            if not child_ids:
                continue
            strongest_child = evidence[list(child_ids)].max(axis=0)
            fallback_id = name_to_id[parent_name]
            evidence[fallback_id] *= 1.0 - 0.88 * strongest_child

    specificity_suppression_count = 0
    if config.use_specificity_ownership:
        suppression_masks: dict[int, np.ndarray] = {}
        child_evidence_floors: dict[int, np.ndarray] = {}
        broad_hosts = [
            candidate
            for candidate in accepted
            if (
                (
                    candidate.semantic_name == candidate.semantic_parent
                    and _root_key(candidate) is not None
                )
                or _is_broad_ownership_host(candidate.semantic_name)
            )
            and not bool(candidate.metadata.get("generic_visual_region"))
        ]
        for candidate in accepted:
            if (
                candidate.semantic_name == candidate.semantic_parent
                or bool(candidate.metadata.get("generic_visual_region"))
            ):
                continue
            calibrated = 0.45 + 0.55 * float(
                np.clip(candidate.score, 0.0, 1.0)
            )
            candidate_strength = calibrated * candidate.source_reliability
            if (
                candidate_strength
                < config.specificity_minimum_candidate_strength
            ):
                continue
            candidate_area = int(np.count_nonzero(candidate.mask))
            if candidate_area == 0:
                continue
            for host in broad_hosts:
                if (
                    host is candidate
                    or host.semantic_name == candidate.semantic_name
                    or _root_key(host) != _root_key(candidate)
                ):
                    continue
                host_is_root = host.semantic_name == host.semantic_parent
                child_id = name_to_id[candidate.semantic_name]
                structural_root_evidence = (
                    bool(candidate.metadata.get("structural_root_evidence"))
                    or candidate.metadata.get("structural_fusion_algorithm")
                    == "profile-planar-tile-union-v1"
                )
                vlm_audit = candidate.metadata.get("vlm_physicality_audit")
                vlm_root_evidence = bool(
                    isinstance(vlm_audit, dict)
                    and vlm_audit.get("decision") == "physical_supported"
                )
                independent_root_evidence = not bool(
                    candidate.metadata.get("visual_region")
                ) and not bool(candidate.metadata.get("structural_fusion"))
                semantic_inventory_evidence = bool(
                    candidate.metadata.get("semantic_reranked")
                    and not candidate.metadata.get("generic_visual_region")
                )
                if host_is_root and not (
                    structural_root_evidence
                    or vlm_root_evidence
                    or semantic_inventory_evidence
                    or (
                        independent_root_evidence
                        and candidate.score
                        >= config.specificity_root_minimum_candidate_score
                    )
                ):
                    continue
                host_area = int(np.count_nonzero(host.mask))
                if host_area <= candidate_area:
                    continue
                host_fraction = candidate_area / host_area
                maximum_host_fraction = (
                    config.structural_specificity_maximum_host_fraction
                    if structural_root_evidence
                    else config.specificity_maximum_host_fraction
                )
                minimum_host_fraction = (
                    config.semantic_detail_minimum_host_fraction
                    if semantic_inventory_evidence
                    and bool(candidate.metadata.get("detail"))
                    else config.specificity_minimum_host_fraction
                )
                if not (
                    minimum_host_fraction
                    <= host_fraction
                    <= maximum_host_fraction
                ):
                    continue
                overlap = candidate.mask.astype(bool) & host.mask.astype(bool)
                containment = int(np.count_nonzero(overlap)) / candidate_area
                if containment < config.specificity_minimum_containment:
                    continue
                host_id = name_to_id[host.semantic_name]
                if host_id not in suppression_masks:
                    suppression_masks[host_id] = overlap.copy()
                else:
                    suppression_masks[host_id] |= overlap
                floor = (
                    evidence[host_id]
                    * config.specificity_child_evidence_floor_ratio
                    * overlap.astype(np.float32)
                )
                if child_id not in child_evidence_floors:
                    child_evidence_floors[child_id] = floor
                else:
                    child_evidence_floors[child_id] = np.maximum(
                        child_evidence_floors[child_id],
                        floor,
                    )
                specificity_suppression_count += 1
        for child_id, evidence_floor in child_evidence_floors.items():
            evidence[child_id] = np.maximum(evidence[child_id], evidence_floor)
        for host_id, suppression_mask in suppression_masks.items():
            evidence[host_id, suppression_mask] *= (
                config.specificity_host_suppression
            )

    if config.detail_bonus:
        evidence[list(taxonomy.detail_ids)] += (
            direct[list(taxonomy.detail_ids)] > config.direct_gate_threshold
        ) * config.detail_bonus
    foreground = parent_support[1:].max(axis=0) if len(parent_support) > 1 else 0.0
    evidence[0] = np.clip(1.0 - foreground, 0.04, 0.98)
    if config.use_root_coverage_conservation:
        # Child masks may reassign root ownership, but decomposition must not
        # turn an accepted root pixel back into background. Soft membership is
        # weak at proposal boundaries, where the background complement could
        # otherwise erode an otherwise valid asset before Part-ID assignment.
        evidence[0, root_union] = np.minimum(evidence[0, root_union], 0.01)

    labels = evidence.argmax(axis=0).astype(np.int32)
    if config.use_direct_gate:
        best = evidence.max(axis=0)
        gated_evidence = np.full_like(evidence, -np.inf)
        for class_id in taxonomy.detail_ids:
            parent_id = taxonomy.fine_to_parent[class_id]
            if class_id in dense_class_ids and config.use_parent_envelope:
                direct_parent_support = envelope_parent_support[parent_id].astype(
                    np.float32
                )
                dilation_ratio = config.dense_parent_dilation_ratio
            else:
                direct_parent_support = parent_support[parent_id]
                dilation_ratio = config.parent_dilation_ratio
            support = _dilate_support(
                direct_parent_support, dilation_ratio
            )
            gate = (
                (direct[class_id] >= config.direct_gate_threshold)
                & (evidence[class_id] >= best * config.direct_gate_margin)
                & (support >= config.orphan_support)
            )
            gated_evidence[class_id, gate] = evidence[class_id, gate]
        gated_best = gated_evidence.max(axis=0)
        gated_owner = gated_evidence.argmax(axis=0)
        has_gate = np.isfinite(gated_best)
        labels[has_gate] = gated_owner[has_gate]

    if config.use_boundary_ownership:
        # Conflicts at proposal boundaries are reassigned to the class with the
        # strongest interior support, reducing jagged overwrite artifacts.
        ordered = np.partition(evidence, -2, axis=0)
        ambiguous = (ordered[-1] - ordered[-2]) < 0.055
        kernel = np.ones((3, 3), np.uint8)
        interior = np.zeros_like(direct)
        for class_id in range(1, taxonomy.num_fine_classes):
            core = cv2.erode(
                (direct[class_id] >= config.direct_gate_threshold).astype(np.uint8),
                kernel,
            ).astype(bool)
            interior[class_id] = direct[class_id] * core
        interior_owner = interior.argmax(axis=0)
        interior_supported = interior.max(axis=0) >= config.direct_gate_threshold
        replace = ambiguous & interior_supported
        labels[replace] = interior_owner[replace]

    labels = _clean_labels(labels, taxonomy, config.minimum_area_px, config)
    lost_root_pixels_before_conservation = int(
        np.count_nonzero(root_union & (labels == 0))
    )
    if config.use_root_coverage_conservation and lost_root_pixels_before_conservation:
        non_background_owner = evidence[1:].argmax(axis=0) + 1
        restore = root_union & (labels == 0)
        labels[restore] = non_background_owner[restore]
    lost_root_pixels_after_conservation = int(
        np.count_nonzero(root_union & (labels == 0))
    )
    if taxonomy.num_fine_classes <= 256:
        semantic_map = labels.astype(np.uint8)
    else:
        semantic_map = labels.astype(np.uint16)
    if any(
        candidate.semantic_name == candidate.semantic_parent for candidate in accepted
    ):
        instance_map, instances, identity_diagnostics = _hierarchical_part_ids(
            semantic_map,
            taxonomy,
            accepted,
            config.minimum_area_px,
            config,
        )
    else:
        instance_map, instances = semantic_to_part_ids(
            semantic_map, taxonomy, minimum_area=config.minimum_area_px
        )
        identity_diagnostics = {
            "identity_hypothesis_count": len(accepted),
            "identity_group_count": len(instances),
            "identity_hypotheses_merged": max(0, len(accepted) - len(instances)),
        }
    diagnostics: dict[str, object] = {
        "input_candidate_count": len(candidates),
        "accepted_candidate_count": len(accepted),
        "same_source_duplicates_removed": removed,
        "output_part_count": len(instances),
        **identity_diagnostics,
        "identity_groups_dropped_by_instance_cap": (
            int(
                identity_diagnostics.get(
                    "identity_groups_dropped_by_instance_cap", 0
                )
            )
            + capped_groups_dropped
        ),
        "identity_instance_cap_scope_count": max(
            int(identity_diagnostics.get("identity_instance_cap_scope_count", 0)),
            capped_scopes,
        ),
        "unresolved_assembly_parent_ids": [
            record.part_id
            for record in instances
            if record.semantic_name != record.semantic_parent
            and record.assembly_parent_id is None
        ],
        "classes_with_multiple_sources": [
            taxonomy.fine_names[class_id]
            for class_id, sources in source_sets.items()
            if len(sources) > 1
        ],
        "source_families": sorted(
            {family for families in source_sets.values() for family in families}
        ),
        "uncorroborated_cross_source_candidates": uncorroborated_count,
        "specificity_host_suppressions": specificity_suppression_count,
        "hierarchical_duplicates_suppressed": (
            hierarchical_duplicates_suppressed
        ),
        "mean_source_agreement_factor": (
            float(np.mean(agreement_factors)) if agreement_factors else 1.0
        ),
        "root_union_area_px": int(np.count_nonzero(root_union)),
        "lost_root_pixels_before_conservation": (
            lost_root_pixels_before_conservation
        ),
        "lost_root_pixels_after_conservation": lost_root_pixels_after_conservation,
        "ground_truth_used": False,
        "ablation": {
            "consensus": config.use_consensus,
            "parent_support": config.use_parent_support,
            "parent_envelope": config.use_parent_envelope,
            "parent_residual": config.use_parent_residual,
            "root_coverage_conservation": config.use_root_coverage_conservation,
            "transitive_residual": config.use_transitive_residual,
            "transitive_residual_dense_only": (
                config.transitive_residual_dense_only
            ),
            "direct_gate": config.use_direct_gate,
            "specificity_ownership": config.use_specificity_ownership,
            "hierarchical_duplicate_suppression": (
                config.use_hierarchical_duplicate_suppression
            ),
            "specificity_minimum_containment": (
                config.specificity_minimum_containment
            ),
            "semantic_detail_minimum_host_fraction": (
                config.semantic_detail_minimum_host_fraction
            ),
            "specificity_host_suppression": (
                config.specificity_host_suppression
            ),
            "specificity_root_minimum_candidate_score": (
                config.specificity_root_minimum_candidate_score
            ),
            "specificity_child_evidence_floor_ratio": (
                config.specificity_child_evidence_floor_ratio
            ),
            "scene_layer_fallback_weight": config.scene_layer_fallback_weight,
            "boundary_ownership": config.use_boundary_ownership,
            "standard_component_fraction": config.standard_component_fraction,
            "detail_component_fraction": config.detail_component_fraction,
            "cleanup_passes": config.cleanup_passes,
            "uncorroborated_source_penalty": (
                config.uncorroborated_source_penalty
            ),
            "full_agreement_overlap": config.full_agreement_overlap,
            "remainder_attachment": config.use_remainder_attachment,
            "remainder_merge_distance_ratio": (
                config.remainder_merge_distance_ratio
            ),
            "identity_strong_containment": config.identity_strong_containment,
            "identity_strong_containment_coherence": (
                config.identity_strong_containment_coherence
            ),
        },
    }
    return FusionResult(
        semantic_map=semantic_map,
        instance_map=instance_map,
        instances=tuple(instances),
        taxonomy=taxonomy,
        evidence=evidence,
        accepted_candidates=tuple(accepted),
        diagnostics=diagnostics,
    )
