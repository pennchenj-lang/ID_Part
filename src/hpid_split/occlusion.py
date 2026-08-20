from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .instances import PartInstance


@dataclass(frozen=True)
class OccluderHypothesis:
    instance_index: int
    contact_px: int
    contact_fraction: float
    search_mask: np.ndarray
    score: float


@dataclass(frozen=True)
class AmodalEvidence:
    accepted: bool
    full_mask: np.ndarray
    added_mask: np.ndarray
    score: float
    visible_recall: float
    search_precision: float
    orthogonal_precision: float
    orthogonal_span_ratio: float
    added_area_px: int
    reason: str


def structural_non_occluder_indices(
    target: PartInstance,
    records: tuple[PartInstance, ...] | list[PartInstance],
) -> frozenset[int]:
    """Return hierarchy/assembly supports that cannot occlude their child."""
    parents_by_semantic: dict[str, str] = {}
    for record in records:
        if record.semantic_name != record.semantic_parent:
            parents_by_semantic.setdefault(record.semantic_name, record.semantic_parent)
    ancestors: set[str] = set()
    current = target.semantic_parent
    while current and current not in ancestors and current != target.semantic_name:
        ancestors.add(current)
        current = parents_by_semantic.get(current, "")
    excluded = {target.instance_index}
    for record in records:
        if record.semantic_name in ancestors:
            excluded.add(record.instance_index)
        if (
            target.assembly_parent_id is not None
            and record.part_id == target.assembly_parent_id
        ):
            excluded.add(record.instance_index)
    return frozenset(excluded)


def _ellipse_kernel(radius: int) -> np.ndarray:
    size = 2 * max(1, radius) + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _mask_extent(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0, 0
    return int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)


def rank_occluder_hypotheses(
    visible: np.ndarray,
    instance_map: np.ndarray,
    target_instance_index: int,
    *,
    maximum_hypotheses: int = 3,
    search_radius_ratio: float = 0.45,
    excluded_instance_indices: frozenset[int] | set[int] = frozenset(),
) -> tuple[OccluderHypothesis, ...]:
    """Rank neighboring IDs without asserting which one is in front.

    A neighboring ID is only a hypothesis. The caller still has to remove it,
    run a learned completion model, and verify that the target reappears.
    """
    if visible.shape != instance_map.shape:
        raise ValueError("visible mask and instance map must have the same shape")
    if maximum_hypotheses < 1 or not visible.any():
        return ()
    width, height = _mask_extent(visible)
    short_side = max(1, min(width, height))
    contact_radius = max(2, round(short_side * 0.025))
    search_radius = max(5, round(short_side * search_radius_ratio))
    contact_ring = (
        cv2.dilate(visible.astype(np.uint8), _ellipse_kernel(contact_radius)).astype(
            bool
        )
        & ~visible
    )
    search_support = cv2.dilate(
        visible.astype(np.uint8), _ellipse_kernel(search_radius)
    ).astype(bool)
    visible_area = max(1, int(np.count_nonzero(visible)))
    minimum_contact = max(3, round(np.sqrt(visible_area) * 0.08))
    hypotheses: list[OccluderHypothesis] = []
    for raw_index in np.unique(instance_map[contact_ring]):
        index = int(raw_index)
        if index in {0, target_instance_index} or index in excluded_instance_indices:
            continue
        occluder = instance_map == index
        contact_px = int(np.count_nonzero(contact_ring & occluder))
        if contact_px < minimum_contact:
            continue
        search = occluder & search_support
        search_area = int(np.count_nonzero(search))
        if search_area < max(8, round(visible_area * 0.003)):
            continue
        contact_fraction = contact_px / max(
            1.0, np.sqrt(visible_area * np.count_nonzero(occluder))
        )
        reach = min(1.0, search_area / max(1.0, visible_area * 0.35))
        hypotheses.append(
            OccluderHypothesis(
                instance_index=index,
                contact_px=contact_px,
                contact_fraction=float(contact_fraction),
                search_mask=search,
                score=float(contact_fraction * (0.65 + 0.35 * reach)),
            )
        )
    hypotheses.sort(key=lambda item: item.score, reverse=True)
    return tuple(hypotheses[:maximum_hypotheses])


def _connected_hidden(hidden: np.ndarray, visible: np.ndarray) -> np.ndarray:
    if not hidden.any():
        return hidden.copy()
    count, components = cv2.connectedComponents(hidden.astype(np.uint8), 8)
    visible_ring = cv2.dilate(visible.astype(np.uint8), _ellipse_kernel(2)).astype(bool)
    kept = np.zeros_like(hidden, dtype=bool)
    for component_id in range(1, count):
        component = components == component_id
        if np.any(component & visible_ring):
            kept |= component
    return kept


def _orthogonal_continuation_geometry(
    visible: np.ndarray, added: np.ndarray
) -> tuple[float, float]:
    """Measure sideways spread relative to the inferred continuation axis."""
    if not added.any() or not visible.any():
        return 1.0, 1.0
    visible_y, visible_x = np.nonzero(visible)
    added_y, added_x = np.nonzero(added)
    visible_width = int(visible_x.max() - visible_x.min() + 1)
    visible_height = int(visible_y.max() - visible_y.min() + 1)
    delta_x = abs(float(added_x.mean() - visible_x.mean())) / max(1, visible_width)
    delta_y = abs(float(added_y.mean() - visible_y.mean())) / max(1, visible_height)

    allowed = np.zeros_like(added, dtype=bool)
    if delta_x >= delta_y:
        margin = max(2, round(visible_height * 0.08))
        low = max(0, int(visible_y.min()) - margin)
        high = min(added.shape[0], int(visible_y.max() + 1) + margin)
        allowed[low:high, :] = True
        added_span = int(added_y.max() - added_y.min() + 1)
        visible_span = visible_height
    else:
        margin = max(2, round(visible_width * 0.08))
        low = max(0, int(visible_x.min()) - margin)
        high = min(added.shape[1], int(visible_x.max() + 1) + margin)
        allowed[:, low:high] = True
        added_span = int(added_x.max() - added_x.min() + 1)
        visible_span = visible_width
    precision = float(np.count_nonzero(added & allowed) / np.count_nonzero(added))
    return precision, float(added_span / max(1, visible_span))


def validate_amodal_proposal(
    visible: np.ndarray,
    proposed: np.ndarray,
    search_mask: np.ndarray,
    model_quality: float,
    *,
    minimum_visible_recall: float = 0.82,
    minimum_search_precision: float = 0.65,
    minimum_model_quality: float = 0.30,
    minimum_evidence_score: float = 0.45,
    minimum_orthogonal_precision: float = 0.78,
    maximum_orthogonal_span_ratio: float = 1.30,
    maximum_added_ratio: float = 1.50,
) -> AmodalEvidence:
    """Accept only model evidence that preserves the visible target.

    The geometric search mask limits where an occluded continuation may be
    tested. It never supplies the final boundary: accepted pixels must come
    from the learned proposal and remain connected to the visible target.
    """
    if not (visible.shape == proposed.shape == search_mask.shape):
        raise ValueError("visible, proposed, and search masks must match")
    visible = visible.astype(bool)
    proposed = proposed.astype(bool)
    search_mask = search_mask.astype(bool) & ~visible
    visible_area = max(1, int(np.count_nonzero(visible)))
    visible_recall = float(np.count_nonzero(proposed & visible) / visible_area)
    raw_added = proposed & ~visible
    inside = raw_added & search_mask
    raw_added_area = int(np.count_nonzero(raw_added))
    search_precision = float(
        np.count_nonzero(inside) / raw_added_area if raw_added_area else 1.0
    )
    added = _connected_hidden(inside, visible)
    added_area = int(np.count_nonzero(added))
    minimum_added = max(6, round(visible_area * 0.003))
    orthogonal_precision, orthogonal_span_ratio = _orthogonal_continuation_geometry(
        visible, added
    )

    score = float(
        np.clip(
            model_quality
            * visible_recall
            * search_precision
            * orthogonal_precision
            * min(1.0, maximum_orthogonal_span_ratio / orthogonal_span_ratio)
            * min(1.0, added_area / max(1, minimum_added * 4)),
            0.0,
            1.0,
        )
    )
    reason = "accepted"
    if model_quality < minimum_model_quality:
        reason = "low_model_quality"
    elif visible_recall < minimum_visible_recall:
        reason = "visible_target_not_preserved"
    elif search_precision < minimum_search_precision:
        reason = "proposal_leaks_outside_occluder_search"
    elif added_area < minimum_added:
        reason = "no_supported_hidden_continuation"
    elif added_area > round(visible_area * maximum_added_ratio):
        reason = "implausible_expansion"
    elif orthogonal_precision < minimum_orthogonal_precision:
        reason = "implausible_lateral_spread"
    elif orthogonal_span_ratio > maximum_orthogonal_span_ratio:
        reason = "implausible_orthogonal_span"
    elif score < minimum_evidence_score:
        reason = "weak_combined_evidence"

    accepted = reason == "accepted"
    accepted_added = added if accepted else np.zeros_like(added)
    return AmodalEvidence(
        accepted=accepted,
        full_mask=visible | accepted_added,
        added_mask=accepted_added,
        score=score,
        visible_recall=visible_recall,
        search_precision=search_precision,
        orthogonal_precision=orthogonal_precision,
        orthogonal_span_ratio=orthogonal_span_ratio,
        added_area_px=int(np.count_nonzero(accepted_added)),
        reason=reason,
    )
