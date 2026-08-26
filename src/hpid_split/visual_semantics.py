from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Protocol

import cv2
import numpy as np
from PIL import Image

from .fusion import MaskCandidate, mask_iou
from .prompt_bank import DomainPrompt, PartPrompt


class RegionLabelRanker(Protocol):
    def rank_regions_labels(
        self,
        image: Image.Image,
        regions: list[tuple[str, np.ndarray]],
        labels: list[tuple[str, str]],
        *,
        masked_weight: float = 0.82,
        temperature: float = 0.035,
        image_batch_size: int = 8,
    ) -> dict[str, dict[str, dict[str, float | str | int]]]: ...


@dataclass(frozen=True)
class VisualSemanticConfig:
    minimum_probability: float = 0.16
    uniform_probability_multiplier: float = 1.28
    minimum_similarity_margin: float = 0.008
    accepted_profile_similarity_margin: float = 0.006
    maximum_regions_per_root: int = 24
    masked_weight: float = 0.82
    temperature: float = 0.035
    source_reliability: float = 0.66
    use_contextual_rescue: bool = True
    contextual_probability_increment: float = 0.02
    contextual_margin_multiplier: float = 2.0
    contextual_minimum_similarity_margin: float = 0.014
    use_repeated_instance_consensus: bool = True
    repeated_consensus_minimum_anchor_roots: int = 3
    repeated_consensus_probability_multiplier: float = 0.80
    repeated_consensus_minimum_margin: float = 0.0
    repeated_consensus_maximum_area_ratio: float = 4.0
    repeated_consensus_source_reliability: float = 0.62
    use_within_root_repetition_consensus: bool = True
    within_root_minimum_capacity: int = 4
    within_root_probability_multiplier: float = 0.75
    within_root_minimum_margin: float = 0.0
    within_root_maximum_area_ratio: float = 4.0
    within_root_confusion_maximum_similarity_drop: float = 0.035
    within_root_source_reliability: float = 0.61
    within_root_macro_minimum_independent_cues: int = 2
    within_root_macro_minimum_boundary_closure: float = 0.64
    within_root_macro_maximum_shading_penalty: float = 0.40
    within_root_macro_maximum_area_ratio: float = 2.6
    axis_similarity_bonus: float = 0.032
    axis_mismatch_penalty: float = 0.045
    minimum_axis_anchor_position: float = 0.55
    minimum_axis_orientation_margin: float = 0.05
    axis_rescue_minimum_probability: float = 0.07
    axis_rescue_uniform_probability_multiplier: float = 0.75
    axis_rescue_minimum_prior: float = 0.018
    axis_rescue_minimum_margin: float = 0.0
    use_capacity_aware_assignment: bool = True
    capacity_fallback_probability_multiplier: float = 0.55
    capacity_fallback_minimum_probability: float = 0.06
    capacity_fallback_maximum_similarity_drop: float = 0.055
    use_inventory_evidence_rescue: bool = True
    inventory_rescue_minimum_probability: float = 0.085
    inventory_rescue_minimum_margin: float = 0.002
    inventory_rescue_minimum_independent_cues: int = 2
    inventory_rescue_minimum_boundary_closure: float = 0.54
    inventory_rescue_minimum_geometric_support: float = 0.58
    inventory_rescue_maximum_shading_penalty: float = 0.50
    use_inventory_cluster_bootstrap: bool = True
    inventory_cluster_minimum_votes: int = 2
    inventory_cluster_minimum_probability: float = 0.065
    inventory_cluster_minimum_margin: float = -0.035
    inventory_cluster_maximum_promotions: int = 2


@dataclass(frozen=True)
class VisualSemanticResult:
    candidates: tuple[MaskCandidate, ...]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class AxisConsistencyConfig:
    """Conservative final gate for profile parts with an intrinsic long axis."""

    hard_rejection_margin: float = 0.16
    profile_refinement_hard_rejection_margin: float = 0.04
    preserve_guided_candidates: bool = True


@dataclass(frozen=True)
class AxisConsistencyResult:
    candidates: tuple[MaskCandidate, ...]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class PhysicalRegionGateConfig:
    """Keep unresolved masks only when they look like physical structure."""

    enabled: bool = True
    minimum_outer_boundary_contact: float = 0.08
    minimum_detail_outer_boundary_contact: float = 0.55
    minimum_detail_structural_area_fraction: float = 0.003
    require_multiview_for_detail_silhouette: bool = True
    allow_strong_single_view_detail_silhouette: bool = True
    minimum_strong_single_view_detail_outer_boundary_contact: float = 0.72
    minimum_strong_single_view_detail_structural_area_fraction: float = 0.008
    minimum_structural_area_fraction: float = 0.008
    maximum_profile_structural_fallbacks: int = 2
    minimum_shape_geometric_support: float = 0.50
    minimum_closed_contour_geometric_support: float = 0.78
    minimum_closed_contour_area_fraction: float = 0.04
    minimum_enclosure_geometric_support: float = 0.55
    minimum_enclosure_area_fraction: float = 0.04
    minimum_surface_host_area_fraction: float = 0.15
    nested_texture_minimum_containment: float = 0.88
    nested_texture_maximum_host_fraction: float = 0.25
    laminar_strip_maximum_area_fraction: float = 0.035
    laminar_strip_maximum_boundary_closure: float = 0.65
    laminar_strip_maximum_chroma_contrast: float = 0.10
    laminar_strip_minimum_luminance_contrast: float = 0.12
    named_host_minimum_containment: float = 0.72
    named_host_maximum_area_ratio: float = 2.50
    maximum_named_shading_only_penalty: float = 0.42
    minimum_named_physical_boundary_alignment: float = 0.58
    minimum_named_physical_boundary_closure: float = 0.46
    corroboration_iou: float = 0.25
    corroboration_containment: float = 0.65
    open_domains: tuple[str, ...] = (
        "character",
        "natural_object",
        "terrain",
        "structure",
    )
    generic_region_blocked_profiles: tuple[str, ...] = (
        "book",
        "flatware",
    )


@dataclass(frozen=True)
class PhysicalRegionGateResult:
    candidates: tuple[MaskCandidate, ...]
    diagnostics: dict[str, object]


_PHOTOMETRIC_PHYSICAL_SURFACE_TOKENS = {
    "display",
    "glass",
    "lens",
    "mirror",
    "screen",
    "visor",
    "window",
    "windshield",
}


@dataclass(frozen=True)
class _AxisContext:
    center_xy: tuple[float, float]
    direction_xy: tuple[float, float]
    projection_midpoint: float
    projection_half_extent: float
    orientation_sign: float
    anchor_count: int
    orientation_margin: float


_DETAIL_TOKENS = {
    "badge",
    "button",
    "camera",
    "claw",
    "dial",
    "eye",
    "flash",
    "gear",
    "key",
    "knob",
    "lens",
    "light",
    "logo",
    "microphone",
    "muzzle",
    "pedal",
    "pivot",
    "port",
    "screw",
    "sight",
    "switch",
    "tip",
    "trigger",
}
_STRIP_TOKENS = {
    "antenna",
    "arm",
    "band",
    "blade",
    "cable",
    "chain",
    "fingerboard",
    "handle",
    "leg",
    "mast",
    "neck",
    "rail",
    "rod",
    "shaft",
    "spindle",
    "stay",
    "stem",
    "stile",
    "strap",
    "stretcher",
    "tube",
    "wiper",
}
_PANEL_TOKENS = {
    "backrest",
    "base",
    "bezel",
    "cover",
    "door",
    "frame",
    "handguard",
    "hood",
    "lid",
    "magazine",
    "panel",
    "receiver",
    "screen",
    "seat",
    "shade",
    "stock",
    "top",
    "window",
}


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


def _part_prompt(part: PartPrompt, object_label: str = "") -> str:
    base = part.dense_phrases[0] if part.dense_phrases else part.phrases[0]
    context = " ".join(object_label.strip().split())
    if not context or context.casefold() in base.casefold():
        return base
    return f"{base} of a {context}"


def _normalized_profile_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _candidate_profile_for_root(
    root: MaskCandidate, domain: DomainPrompt
) -> tuple[str, str] | None:
    """Resolve one conservative inventory hint without claiming exact routing.

    A singleton router candidate may constrain which physical parts are
    plausible even when its confidence is below the threshold required to call
    the object identity accepted.  Ambiguous candidates deliberately keep the
    open-set path.
    """

    labels = root.metadata.get("asset_router_candidate_labels")
    candidate_domains = root.metadata.get("asset_router_candidate_domains")
    if not isinstance(labels, list) or not labels:
        return None
    if isinstance(candidate_domains, list) and candidate_domains:
        normalized_domains = {str(value).strip() for value in candidate_domains}
        if normalized_domains != {domain.name}:
            return None
    matches: dict[str, str] = {}
    for raw_label in labels:
        label = str(raw_label).strip()
        normalized = _normalized_profile_text(label)
        if not normalized:
            continue
        for profile in domain.part_profiles:
            aliases = {
                _normalized_profile_text(profile.name),
                *(
                    _normalized_profile_text(value)
                    for value in profile.root_hints
                ),
            }
            if normalized in aliases:
                matches[profile.name] = label
    if len(matches) != 1:
        return None
    return next(iter(matches.items()))


def _inventory_for_root(
    root: MaskCandidate, domain: DomainPrompt
) -> tuple[tuple[PartPrompt, ...], str | None, str, str, str | None]:
    object_label = str(
        root.metadata.get("resolved_object_label")
        or root.metadata.get("root_model_label")
        or root.prompt
        or domain.name
    )
    profile = root.metadata.get("selected_part_profile")
    if profile is not None and str(profile).strip():
        selected, resolved, diagnostics = domain.select_parts(
            object_label,
            profile_hint=str(profile),
            profile_hint_source="resolved_root_profile",
        )
        subtype_hints = diagnostics.get("subtype_root_hints", [])
        # The profile route is the inventory contract.  A weaker asset router
        # label can disagree inside the same broad domain (for example,
        # ``screwdriver`` versus the independently selected ``firearm``
        # profile).  Feeding that stale label back into the contextual text
        # scorer creates impossible prompts such as "magazine of a
        # screwdriver".  Once a profile has been accepted, use its canonical
        # name unless an explicit subtype supplies a more specific context.
        profile_definition = next(
            (item for item in domain.part_profiles if item.name == resolved),
            None,
        )
        normalized_object = _normalized_profile_text(object_label)
        profile_aliases = (
            {
                _normalized_profile_text(profile_definition.name),
                *(
                    _normalized_profile_text(value)
                    for value in profile_definition.root_hints
                ),
            }
            if profile_definition is not None
            else set()
        )
        object_matches_profile = any(
            alias
            and (
                alias in normalized_object
                or normalized_object in alias
            )
            for alias in profile_aliases
        )
        semantic_context = (
            str(subtype_hints[0])
            if isinstance(subtype_hints, list) and subtype_hints
            else object_label
            if object_matches_profile
            else str(resolved or object_label).replace("_", " ")
        )
        return (
            selected,
            resolved,
            "resolved_profile",
            semantic_context,
            (
                str(diagnostics["selected_subtype"])
                if diagnostics.get("selected_subtype") is not None
                else None
            ),
        )
    candidate_profile = _candidate_profile_for_root(root, domain)
    if candidate_profile is not None:
        profile_name, candidate_label = candidate_profile
        selected, resolved, diagnostics = domain.select_parts(
            candidate_label,
            profile_hint=profile_name,
            profile_hint_source="router_candidate_inventory",
        )
        return (
            selected,
            resolved,
            "router_candidate_inventory_review",
            candidate_label,
            (
                str(diagnostics["selected_subtype"])
                if diagnostics.get("selected_subtype") is not None
                else None
            ),
        )
    if domain.part_profiles:
        return (
            (),
            None,
            "unresolved_profile_preserves_visual_ids",
            object_label,
            None,
        )
    selected, resolved, diagnostics = domain.select_parts(domain.name)
    return (
        selected,
        resolved,
        str(diagnostics["selection_reason"]),
        object_label,
        None,
    )


def _eligible_parts(parts: tuple[PartPrompt, ...]) -> tuple[PartPrompt, ...]:
    return tuple(
        part
        for part in parts
        if not part.semantic_name.endswith("_body")
        and part.topology_relation is None
        and part.appearance_relation is None
    )


def _shape_prior(part: PartPrompt, region_kind: str) -> float:
    tokens = set(part.semantic_name.split("_"))
    if region_kind == "detail":
        if part.detail or tokens & _DETAIL_TOKENS:
            return 0.010
        if tokens & (_STRIP_TOKENS | _PANEL_TOKENS):
            return -0.008
    elif region_kind == "strip":
        if tokens & _STRIP_TOKENS:
            return 0.010
        if part.detail or tokens & _DETAIL_TOKENS:
            return -0.008
    elif region_kind == "panel":
        if tokens & _PANEL_TOKENS:
            return 0.006
        if part.detail and not tokens & _PANEL_TOKENS:
            return -0.010
    return 0.0


def _geometry_compatible(candidate: MaskCandidate, part: PartPrompt) -> bool:
    fraction = float(candidate.metadata.get("root_area_fraction", 0.0))
    lower = max(0.0, part.minimum_parent_fraction * 0.55)
    upper = min(1.0, part.maximum_parent_fraction * 1.25)
    if not lower <= fraction <= upper:
        return False
    return not (
        part.detail and fraction > min(0.24, part.maximum_parent_fraction * 1.15)
    )


def _mask_axis_coordinate(mask: np.ndarray, context: _AxisContext) -> float:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0.0
    center = np.asarray(context.center_xy, dtype=np.float64)
    direction = np.asarray(context.direction_xy, dtype=np.float64)
    centroid = np.asarray([float(xs.mean()), float(ys.mean())], dtype=np.float64)
    projection = float((centroid - center) @ direction)
    normalized = (
        projection - context.projection_midpoint
    ) / context.projection_half_extent
    return float(np.clip(context.orientation_sign * normalized, -1.25, 1.25))


def _build_axis_context(
    root: MaskCandidate,
    semantic_candidates: list[MaskCandidate],
    parts: tuple[PartPrompt, ...],
    config: VisualSemanticConfig,
) -> _AxisContext | None:
    ys, xs = np.nonzero(root.mask)
    if len(xs) < 8:
        return None
    points = np.column_stack((xs, ys)).astype(np.float64)
    center = points.mean(axis=0)
    covariance = np.cov((points - center).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    projections = (points - center) @ direction
    low, high = float(projections.min()), float(projections.max())
    half_extent = 0.5 * (high - low)
    if half_extent < 2.0:
        return None
    midpoint = 0.5 * (low + high)
    unoriented = _AxisContext(
        center_xy=(float(center[0]), float(center[1])),
        direction_xy=(float(direction[0]), float(direction[1])),
        projection_midpoint=midpoint,
        projection_half_extent=half_extent,
        orientation_sign=1.0,
        anchor_count=0,
        orientation_margin=0.0,
    )
    part_by_name = {part.semantic_name: part for part in parts}
    anchors: list[tuple[float, float, float]] = []
    for candidate in semantic_candidates:
        if (
            _root_key(candidate) != _root_key(root)
            or bool(candidate.metadata.get("generic_visual_region"))
        ):
            continue
        part = part_by_name.get(candidate.semantic_name)
        if (
            part is None
            or part.axis_position is None
            or abs(part.axis_position) < config.minimum_axis_anchor_position
            or not _geometry_compatible(candidate, part)
        ):
            continue
        coordinate = _mask_axis_coordinate(candidate.mask, unoriented)
        weight = max(
            1e-4,
            abs(part.axis_position)
            * candidate.score
            * candidate.source_reliability,
        )
        anchors.append((coordinate, float(part.axis_position), weight))
    if not anchors:
        return None
    positive_error = sum(
        weight * (coordinate - expected) ** 2
        for coordinate, expected, weight in anchors
    )
    negative_error = sum(
        weight * (-coordinate - expected) ** 2
        for coordinate, expected, weight in anchors
    )
    total_weight = sum(weight for _, _, weight in anchors)
    orientation_margin = abs(positive_error - negative_error) / max(
        1e-8, total_weight
    )
    if orientation_margin < config.minimum_axis_orientation_margin:
        return None
    return replace(
        unoriented,
        orientation_sign=1.0 if positive_error < negative_error else -1.0,
        anchor_count=len(anchors),
        orientation_margin=float(orientation_margin),
    )


def _axis_prior(
    candidate: MaskCandidate,
    part: PartPrompt,
    context: _AxisContext | None,
    config: VisualSemanticConfig,
) -> tuple[float, float | None]:
    if context is None or part.axis_position is None:
        return 0.0, None
    coordinate = _mask_axis_coordinate(candidate.mask, context)
    distance = abs(coordinate - part.axis_position)
    if distance <= part.axis_tolerance:
        prior = config.axis_similarity_bonus * (
            1.0 - distance / part.axis_tolerance
        )
    else:
        prior = -config.axis_mismatch_penalty * min(
            1.0,
            (distance - part.axis_tolerance) / part.axis_tolerance,
        )
    return float(prior), coordinate


def enforce_axis_consistency(
    candidates: list[MaskCandidate] | tuple[MaskCandidate, ...],
    roots: list[MaskCandidate] | tuple[MaskCandidate, ...],
    domains: dict[str, DomainPrompt],
    config: AxisConsistencyConfig | None = None,
) -> AxisConsistencyResult:
    """Reject semantically named parts that contradict an oriented object axis.

    The axis is inferred from the root mask and independently detected endpoint
    parts. The gate is deliberately inactive when orientation evidence is
    missing or ambiguous, and explicit guided candidates are never removed.
    """

    config = config or AxisConsistencyConfig()
    output_by_key = {_candidate_key(candidate): candidate for candidate in candidates}
    rejected_keys: set[str] = set()
    root_rows: list[dict[str, object]] = []
    evaluated_count = 0
    preserved_guided_count = 0

    for root in roots:
        domain = domains.get(root.semantic_name)
        if domain is None:
            continue
        parts, selected_profile, selection_reason, _, _ = _inventory_for_root(
            root, domain
        )
        part_by_name = {part.semantic_name: part for part in parts}
        root_key = _root_key(root)
        scoped = [
            candidate
            for candidate in candidates
            if _root_key(candidate) == root_key
        ]
        context = _build_axis_context(
            root,
            scoped,
            parts,
            VisualSemanticConfig(),
        )
        row: dict[str, object] = {
            "root_key": root_key,
            "root_semantic": root.semantic_name,
            "selected_profile": selected_profile,
            "selection_reason": selection_reason,
            "evaluated": [],
            "ground_truth_used": False,
        }
        if context is None:
            row["status"] = "orientation_evidence_unavailable"
            root_rows.append(row)
            continue

        row["axis_context"] = {
            "anchor_count": context.anchor_count,
            "orientation_margin": context.orientation_margin,
            "direction_xy": list(context.direction_xy),
        }
        for candidate in scoped:
            part = part_by_name.get(candidate.semantic_name)
            if part is None or part.axis_position is None:
                continue
            candidate_key = _candidate_key(candidate)
            coordinate = _mask_axis_coordinate(candidate.mask, context)
            distance = abs(coordinate - part.axis_position)
            profile_refinement = bool(
                candidate.metadata.get("profile_refinement")
            )
            rejection_margin = (
                config.profile_refinement_hard_rejection_margin
                if profile_refinement
                else config.hard_rejection_margin
            )
            hard_limit = min(2.0, part.axis_tolerance + rejection_margin)
            guided = bool(candidate.metadata.get("guided_prompt"))
            accepted = distance <= hard_limit or (
                config.preserve_guided_candidates and guided
            )
            evidence = {
                "candidate_key": candidate_key,
                "semantic_name": candidate.semantic_name,
                "axis_coordinate": float(coordinate),
                "axis_expected_position": float(part.axis_position),
                "axis_distance": float(distance),
                "axis_hard_limit": float(hard_limit),
                "axis_rejection_margin": float(rejection_margin),
                "profile_refinement_candidate": profile_refinement,
                "guided_candidate": guided,
                "accepted": accepted,
            }
            cast_rows = row["evaluated"]
            assert isinstance(cast_rows, list)
            cast_rows.append(evidence)
            evaluated_count += 1
            if guided and distance > hard_limit:
                preserved_guided_count += 1
            if not accepted:
                rejected_keys.add(candidate_key)
                continue
            output_by_key[candidate_key] = replace(
                candidate,
                metadata={
                    **candidate.metadata,
                    "axis_consistency_gate": evidence,
                },
            )
        row["status"] = "evaluated"
        row["rejected_count"] = sum(
            not bool(item["accepted"])
            for item in row["evaluated"]
            if isinstance(item, dict)
        )
        root_rows.append(row)

    filtered = tuple(
        output_by_key[_candidate_key(candidate)]
        for candidate in candidates
        if _candidate_key(candidate) not in rejected_keys
    )
    return AxisConsistencyResult(
        filtered,
        {
            "algorithm": "hpid-oriented-axis-consistency-gate-v1",
            "input_candidate_count": len(candidates),
            "output_candidate_count": len(filtered),
            "evaluated_candidate_count": evaluated_count,
            "rejected_candidate_count": len(rejected_keys),
            "preserved_guided_candidate_count": preserved_guided_count,
            "roots": root_rows,
            "ground_truth_used": False,
        },
    )


def _supports_axis_rescue(
    evidence: dict[str, object] | None,
    context: _AxisContext | None,
    probability_floor: float,
    config: VisualSemanticConfig,
) -> bool:
    return bool(
        evidence is not None
        and context is not None
        and bool(evidence.get("axis_rescue_allowed"))
        and evidence.get("axis_expected_position") is not None
        and float(evidence.get("axis_prior", 0.0))
        >= config.axis_rescue_minimum_prior
        and float(evidence["similarity_margin"])
        >= config.axis_rescue_minimum_margin
        and float(evidence["probability"]) >= probability_floor
    )


def _appearance_structure(candidate: MaskCandidate) -> dict[str, float | int | bool]:
    evidence = candidate.metadata.get("appearance_graph_evidence")
    evidence = evidence if isinstance(evidence, dict) else {}
    return {
        "cue_count": int(evidence.get("independent_cue_count", 0)),
        "closure": float(evidence.get("boundary_closure", 0.0)),
        "geometric_support": float(
            evidence.get(
                "geometric_support",
                candidate.metadata.get("geometric_support", 0.0),
            )
        ),
        "boundary_alignment": float(
            evidence.get(
                "boundary_alignment",
                candidate.metadata.get("proposal_boundary_alignment", 0.0),
            )
        ),
        "shading_penalty": float(evidence.get("shading_only_penalty", 0.0)),
        "multi_view": bool(
            evidence.get(
                "multi_view_confirmed",
                candidate.metadata.get("multi_view_confirmed", False),
            )
        ),
    }


def _supports_repeated_macro_component(
    candidate: MaskCandidate,
    config: VisualSemanticConfig,
) -> bool:
    """Require independent structure before cloning a repeated macro semantic."""

    if str(candidate.metadata.get("visual_region_kind", "panel")) not in {
        "panel",
        "strip",
    }:
        return False
    structure = _appearance_structure(candidate)
    return bool(
        int(structure["cue_count"])
        >= config.within_root_macro_minimum_independent_cues
        and float(structure["closure"])
        >= config.within_root_macro_minimum_boundary_closure
        and float(structure["shading_penalty"])
        <= config.within_root_macro_maximum_shading_penalty
        and (
            bool(structure["multi_view"])
            or float(structure["boundary_alignment"]) >= 0.60
        )
    )


def _supports_inventory_evidence_rescue(
    candidate: MaskCandidate,
    part: PartPrompt,
    evidence: dict[str, object] | None,
    *,
    selected_profile: str | None,
    config: VisualSemanticConfig,
) -> bool:
    """Accept a weak semantic score only with independent physical evidence."""

    if (
        not config.use_inventory_evidence_rescue
        or selected_profile is None
        or evidence is None
        or not _geometry_compatible(candidate, part)
        or float(evidence["probability"])
        < config.inventory_rescue_minimum_probability
        or float(evidence["similarity_margin"])
        < config.inventory_rescue_minimum_margin
    ):
        return False
    structure = _appearance_structure(candidate)
    if float(structure["shading_penalty"]) > config.inventory_rescue_maximum_shading_penalty:
        return False
    cross_cue = bool(
        int(structure["cue_count"])
        >= config.inventory_rescue_minimum_independent_cues
        and float(structure["closure"])
        >= config.inventory_rescue_minimum_boundary_closure
    )
    geometric = bool(
        float(structure["geometric_support"])
        >= config.inventory_rescue_minimum_geometric_support
        and float(structure["boundary_alignment"]) >= 0.20
    )
    return bool(cross_cue or geometric or structure["multi_view"])


def _route_alternatives(
    region_scores: dict[str, dict[str, object]],
    candidate: MaskCandidate,
    part_by_name: dict[str, PartPrompt],
    axis_context: _AxisContext | None,
    config: VisualSemanticConfig,
) -> list[dict[str, object]]:
    compatible: list[
        tuple[float, str, dict[str, object], float, float | None]
    ] = []
    all_ranked: list[tuple[float, str]] = []
    for semantic_name, raw_row in region_scores.items():
        if semantic_name not in part_by_name:
            continue
        part = part_by_name[semantic_name]
        row = dict(raw_row)
        axis_prior, axis_coordinate = _axis_prior(
            candidate, part, axis_context, config
        )
        adjusted = (
            float(row["combined_similarity"])
            + _shape_prior(
                part, str(candidate.metadata.get("visual_region_kind", "panel"))
            )
            + axis_prior
        )
        all_ranked.append((adjusted, semantic_name))
        if _geometry_compatible(candidate, part):
            compatible.append(
                (adjusted, semantic_name, row, axis_prior, axis_coordinate)
            )
    compatible.sort(key=lambda item: item[0], reverse=True)
    all_ranked.sort(key=lambda item: item[0], reverse=True)
    output: list[dict[str, object]] = []
    for item in compatible:
        runner_up = next(
            (
                score
                for score, semantic_name in all_ranked
                if semantic_name != item[1]
            ),
            item[0] - 1.0,
        )
        output.append(
            {
                "best_semantic": item[1],
                "adjusted_similarity": item[0],
                "similarity_margin": item[0] - runner_up,
                "probability": float(item[2]["probability"]),
                "ranker_row": item[2],
                "axis_prior": item[3],
                "axis_coordinate": item[4],
                "axis_expected_position": part_by_name[item[1]].axis_position,
                "axis_rescue_allowed": not part_by_name[item[1]].detail,
            }
        )
    return output


def _route_evidence(
    region_scores: dict[str, dict[str, object]],
    candidate: MaskCandidate,
    part_by_name: dict[str, PartPrompt],
    axis_context: _AxisContext | None,
    config: VisualSemanticConfig,
) -> dict[str, object] | None:
    alternatives = _route_alternatives(
        region_scores,
        candidate,
        part_by_name,
        axis_context,
        config,
    )
    return alternatives[0] if alternatives else None


def _specific_route_evidence(
    region_scores: dict[str, dict[str, object]],
    candidate: MaskCandidate,
    part: PartPrompt,
    axis_context: _AxisContext | None,
    config: VisualSemanticConfig,
) -> dict[str, float] | None:
    """Read one explicit label without allowing it to win by label count."""

    raw_row = region_scores.get(part.semantic_name)
    if raw_row is None or not _geometry_compatible(candidate, part):
        return None
    axis_prior, _ = _axis_prior(candidate, part, axis_context, config)
    adjusted = (
        float(raw_row["combined_similarity"])
        + _shape_prior(
            part, str(candidate.metadata.get("visual_region_kind", "panel"))
        )
        + axis_prior
    )
    return {
        "adjusted_similarity": float(adjusted),
        "probability": float(raw_row["probability"]),
    }


def _best_parent_key(
    candidate: MaskCandidate,
    semantic_name: str | None,
    evidence: list[MaskCandidate],
    root: MaskCandidate,
) -> str:
    if semantic_name is None or semantic_name == root.semantic_name:
        return _candidate_key(root)
    matches = [
        item
        for item in evidence
        if _root_key(item) == _root_key(root) and item.semantic_name == semantic_name
    ]
    if not matches:
        return _candidate_key(root)
    return _candidate_key(
        max(matches, key=lambda item: mask_iou(candidate.mask, item.mask))
    )


def _mask_containment(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    if not intersection:
        return 0.0
    return intersection / max(
        1,
        min(int(np.count_nonzero(first)), int(np.count_nonzero(second))),
    )


def _outer_boundary_contact(mask: np.ndarray, root_mask: np.ndarray) -> float:
    kernel = np.ones((3, 3), dtype=np.uint8)
    candidate_u8 = mask.astype(np.uint8)
    root_u8 = root_mask.astype(np.uint8)
    candidate_boundary = candidate_u8.astype(bool) & ~cv2.erode(
        candidate_u8, kernel
    ).astype(bool)
    root_boundary = root_u8.astype(bool) & ~cv2.erode(
        root_u8, kernel
    ).astype(bool)
    if not np.any(candidate_boundary) or not np.any(root_boundary):
        return 0.0
    near_root_boundary = cv2.dilate(root_boundary.astype(np.uint8), kernel).astype(
        bool
    )
    return float(
        np.count_nonzero(candidate_boundary & near_root_boundary)
        / max(1, np.count_nonzero(candidate_boundary))
    )


def _convex_host_envelope(mask: np.ndarray, root_mask: np.ndarray) -> np.ndarray:
    points = cv2.findNonZero(mask.astype(np.uint8))
    if points is None or len(points) < 3:
        return mask.astype(bool) & root_mask.astype(bool)
    hull = cv2.convexHull(points)
    envelope = np.zeros_like(mask, dtype=np.uint8)
    cv2.fillConvexPoly(envelope, hull, 1)
    return envelope.astype(bool) & root_mask.astype(bool)


def filter_unresolved_visual_regions(
    candidates: list[MaskCandidate] | tuple[MaskCandidate, ...],
    roots: list[MaskCandidate] | tuple[MaskCandidate, ...],
    domains: dict[str, DomainPrompt],
    config: PhysicalRegionGateConfig | None = None,
) -> PhysicalRegionGateResult:
    """Reject texture fragments in closed profiles without hiding real structure.

    Unresolved SAM regions need independent semantic support, outer-silhouette
    structure, or a bounded structural fallback. Open-world domains use the same
    evidence rule instead of exporting every texture fragment as a Part ID.
    """

    config = config or PhysicalRegionGateConfig()
    if not config.enabled:
        return PhysicalRegionGateResult(
            tuple(candidates),
            {
                "algorithm": "hpid-profile-physical-region-gate-v1",
                "status": "disabled",
                "input_candidate_count": len(candidates),
                "output_candidate_count": len(candidates),
                "ground_truth_used": False,
            },
        )
    roots_by_key = {_root_key(root): root for root in roots}
    named_by_root: dict[str, list[MaskCandidate]] = defaultdict(list)
    for candidate in candidates:
        if (
            not bool(candidate.metadata.get("generic_visual_region"))
            and candidate.semantic_name != candidate.semantic_parent
        ):
            named_by_root[_root_key(candidate)].append(candidate)
    policy_by_root: dict[str, tuple[bool, str | None, str]] = {}
    structural_fallback_keys: set[str] = set()
    generic_by_root: dict[str, list[MaskCandidate]] = defaultdict(list)
    for candidate in candidates:
        if bool(candidate.metadata.get("generic_visual_region")):
            generic_by_root[_root_key(candidate)].append(candidate)
    for root_key, generic_candidates in generic_by_root.items():
        ranked = sorted(
            (
                candidate
                for candidate in generic_candidates
                if str(candidate.metadata.get("visual_region_kind", "panel"))
                in {"panel", "strip"}
                and float(candidate.metadata.get("root_area_fraction", 0.0))
                >= config.minimum_structural_area_fraction
            ),
            key=lambda candidate: (
                candidate.score * candidate.source_reliability,
                float(candidate.metadata.get("root_area_fraction", 0.0)),
            ),
            reverse=True,
        )
        structural_fallback_keys.update(
            _candidate_key(candidate)
            for candidate in ranked[: config.maximum_profile_structural_fallbacks]
        )
    surface_hosts_by_root: dict[
        str, list[tuple[MaskCandidate, np.ndarray]]
    ] = defaultdict(list)
    for root_key, generic_candidates in generic_by_root.items():
        root = roots_by_key.get(root_key)
        if root is None:
            continue
        for candidate in generic_candidates:
            if (
                candidate.source.startswith("hpid-appearance-contour/")
                and str(candidate.metadata.get("visual_region_kind", "panel"))
                == "panel"
                and float(candidate.metadata.get("geometric_support", 0.0))
                >= config.minimum_closed_contour_geometric_support
                and float(candidate.metadata.get("root_area_fraction", 0.0))
                >= config.minimum_surface_host_area_fraction
            ):
                surface_hosts_by_root[root_key].append(
                    (candidate, _convex_host_envelope(candidate.mask, root.mask))
                )
    output: list[MaskCandidate] = []
    rows: list[dict[str, object]] = []
    rejected_count = 0
    corroborated_count = 0
    silhouette_count = 0
    profile_fallback_count = 0
    cross_source_structure_count = 0
    shape_structure_count = 0
    vlm_physical_count = 0
    vlm_nonphysical_count = 0
    nested_surface_texture_count = 0
    nested_named_region_count = 0
    laminar_surface_strip_count = 0
    photometric_only_named_count = 0
    for candidate in candidates:
        is_generic = bool(candidate.metadata.get("generic_visual_region"))
        vlm_audit = candidate.metadata.get("vlm_physicality_audit")
        vlm_decision = (
            str(vlm_audit.get("decision", ""))
            if isinstance(vlm_audit, dict)
            else ""
        )
        vlm_physical = vlm_decision == "physical_supported"
        vlm_nonphysical = vlm_decision == "nonphysical_supported"
        if not is_generic:
            appearance_evidence = candidate.metadata.get(
                "appearance_graph_evidence"
            )
            appearance_evidence = (
                appearance_evidence
                if isinstance(appearance_evidence, dict)
                else {}
            )
            shading_penalty = float(
                appearance_evidence.get("shading_only_penalty", 0.0)
            )
            boundary_alignment = float(
                appearance_evidence.get("boundary_alignment", 0.0)
            )
            boundary_closure = float(
                appearance_evidence.get("boundary_closure", 0.0)
            )
            root_boundary_contact = float(
                appearance_evidence.get("root_boundary_contact", 0.0)
            )
            cross_source = bool(
                candidate.metadata.get("cross_source_confirmed")
                or candidate.metadata.get("multi_view_confirmed")
            )
            physical_boundary = bool(
                boundary_alignment
                >= config.minimum_named_physical_boundary_alignment
                and boundary_closure
                >= config.minimum_named_physical_boundary_closure
            )
            semantic_tokens = set(
                re.findall(r"[a-z0-9]+", candidate.semantic_name.casefold())
            )
            bounded_photometric_surface = bool(
                physical_boundary
                and semantic_tokens & _PHOTOMETRIC_PHYSICAL_SURFACE_TOKENS
            )
            independent_physical_structure = bool(
                root_boundary_contact >= 0.10
                or float(candidate.metadata.get("geometric_support", 0.0))
                >= 0.72
                or candidate.metadata.get("topology_refinement")
                or candidate.source.startswith("hpid-shape-bottleneck/")
            )
            direct_semantic_source = any(
                token in candidate.source.casefold()
                for token in (
                    "/profile-refine",
                    "conditional-part",
                    "grounded-sam",
                    "grounding-dino",
                    "prototype-retrieval",
                )
            ) and "semantic-rerank" not in candidate.source.casefold()
            photometric_only = bool(
                candidate.semantic_name != candidate.semantic_parent
                and shading_penalty
                >= config.maximum_named_shading_only_penalty
                and not bounded_photometric_surface
                and not vlm_physical
                and not (
                    independent_physical_structure
                    and (cross_source or direct_semantic_source)
                )
            )
            if (
                bool(candidate.metadata.get("visual_region")) and vlm_nonphysical
            ) or photometric_only:
                rejected_count += 1
                vlm_nonphysical_count += int(vlm_nonphysical)
                photometric_only_named_count += int(photometric_only)
                rows.append(
                    {
                        "candidate_key": _candidate_key(candidate),
                        "root_key": _root_key(candidate),
                        "selected_profile": candidate.metadata.get(
                            "selected_part_profile"
                        ),
                        "accepted": False,
                        "named_visual_candidate": True,
                        "named_corroboration": False,
                        "silhouette_structure": False,
                        "profile_structural_fallback": False,
                        "vlm_physical_supported": False,
                        "vlm_nonphysical_supported": vlm_nonphysical,
                        "photometric_only_region": photometric_only,
                        "shading_only_penalty": shading_penalty,
                        "boundary_alignment": boundary_alignment,
                        "boundary_closure": boundary_closure,
                        "root_boundary_contact": root_boundary_contact,
                        "illumination_region": appearance_evidence.get(
                            "illumination_region", "none"
                        ),
                        "independent_physical_structure": (
                            independent_physical_structure
                        ),
                        "direct_semantic_source": direct_semantic_source,
                        "bounded_photometric_surface": (
                            bounded_photometric_surface
                        ),
                        "outer_boundary_contact": None,
                        "root_area_fraction": candidate.metadata.get(
                            "root_area_fraction"
                        ),
                        "visual_region_kind": candidate.metadata.get(
                            "visual_region_kind"
                        ),
                        "ground_truth_used": False,
                    }
                )
                continue
            output.append(candidate)
            continue
        root_key = _root_key(candidate)
        root = roots_by_key.get(root_key)
        if root is None:
            output.append(candidate)
            continue
        policy = policy_by_root.get(root_key)
        if policy is None:
            domain = domains.get(root.semantic_name)
            if domain is None:
                policy = (False, None, "domain_unavailable")
            else:
                _, selected_profile, _, _, _ = _inventory_for_root(root, domain)
                closed = bool(
                    selected_profile is not None
                    and root.semantic_name not in config.open_domains
                )
                reason = (
                    "closed_resolved_profile"
                    if closed
                    else "open_world_inventory"
                )
                policy = (closed, selected_profile, reason)
            policy_by_root[root_key] = policy
        closed, selected_profile, policy_reason = policy
        candidate_area = max(1, int(np.count_nonzero(candidate.mask)))
        corroborated = False
        nested_named_region = False
        for named in named_by_root.get(root_key, []):
            named_area = max(1, int(np.count_nonzero(named.mask)))
            intersection = int(np.count_nonzero(candidate.mask & named.mask))
            iou = intersection / max(1, candidate_area + named_area - intersection)
            candidate_containment = intersection / candidate_area
            area_ratio = max(candidate_area, named_area) / min(
                candidate_area, named_area
            )
            minimum_area_containment = intersection / min(
                candidate_area, named_area
            )
            if iou >= config.corroboration_iou or (
                candidate_containment >= config.corroboration_containment
                and area_ratio <= 3.0
            ):
                corroborated = True
            if (
                minimum_area_containment
                >= config.named_host_minimum_containment
                and area_ratio <= config.named_host_maximum_area_ratio
            ):
                nested_named_region = True
        root_area = max(1, int(np.count_nonzero(root.mask)))
        area_fraction = candidate_area / root_area
        boundary_contact = _outer_boundary_contact(candidate.mask, root.mask)
        region_kind = str(candidate.metadata.get("visual_region_kind", "panel"))
        silhouette_structure = bool(
            selected_profile not in config.generic_region_blocked_profiles
            and region_kind in {"panel", "strip"}
            and area_fraction >= config.minimum_structural_area_fraction
            and boundary_contact >= config.minimum_outer_boundary_contact
        )
        multi_view_detail = bool(candidate.metadata.get("multi_view_confirmed"))
        strong_single_view_detail = bool(
            config.allow_strong_single_view_detail_silhouette
            and not multi_view_detail
            and area_fraction
            >= config.minimum_strong_single_view_detail_structural_area_fraction
            and boundary_contact
            >= config.minimum_strong_single_view_detail_outer_boundary_contact
        )
        silhouette_detail_structure = bool(
            selected_profile not in config.generic_region_blocked_profiles
            and region_kind == "detail"
            and area_fraction >= config.minimum_detail_structural_area_fraction
            and boundary_contact >= config.minimum_detail_outer_boundary_contact
            and (
                not config.require_multiview_for_detail_silhouette
                or multi_view_detail
                or strong_single_view_detail
            )
        )
        profile_structural_fallback = bool(
            selected_profile not in config.generic_region_blocked_profiles
            and _candidate_key(candidate) in structural_fallback_keys
        )
        cross_source_structure = bool(
            selected_profile not in config.generic_region_blocked_profiles
            and candidate.metadata.get("cross_source_confirmed")
        )
        geometric_support = float(candidate.metadata.get("geometric_support", 0.0))
        shape_structure = bool(
            selected_profile not in config.generic_region_blocked_profiles
            and (
                (
                    candidate.source.startswith("hpid-shape-bottleneck/")
                    and geometric_support >= config.minimum_shape_geometric_support
                    and area_fraction >= config.minimum_structural_area_fraction
                )
                or (
                    candidate.source.startswith("hpid-appearance-contour/")
                    and region_kind in {"panel", "strip"}
                    and geometric_support
                    >= config.minimum_closed_contour_geometric_support
                    and area_fraction
                    >= config.minimum_closed_contour_area_fraction
                )
                or (
                    candidate.source.startswith("hpid-appearance-enclosure/")
                    and region_kind in {"panel", "strip"}
                    and geometric_support
                    >= config.minimum_enclosure_geometric_support
                    and area_fraction >= config.minimum_enclosure_area_fraction
                )
            )
        )
        appearance_evidence = candidate.metadata.get("appearance_graph_evidence")
        parent_candidate_key = str(
            candidate.metadata.get("parent_candidate_key") or ""
        )
        laminar_surface_strip = False
        if isinstance(appearance_evidence, dict):
            laminar_surface_strip = bool(
                region_kind == "strip"
                and "/visual-region:" in parent_candidate_key
                and not corroborated
                and not multi_view_detail
                and not cross_source_structure
                and geometric_support < config.minimum_shape_geometric_support
                and area_fraction <= config.laminar_strip_maximum_area_fraction
                and float(appearance_evidence.get("boundary_closure", 1.0))
                < config.laminar_strip_maximum_boundary_closure
                and float(appearance_evidence.get("chroma_contrast", 1.0))
                < config.laminar_strip_maximum_chroma_contrast
                and float(appearance_evidence.get("luminance_contrast", 0.0))
                >= config.laminar_strip_minimum_luminance_contrast
                and int(appearance_evidence.get("independent_cue_count", 99)) <= 2
            )
        nested_surface_texture = False
        for host, host_envelope in surface_hosts_by_root.get(root_key, []):
            if host is candidate:
                continue
            host_area = max(1, int(np.count_nonzero(host_envelope)))
            intersection = int(np.count_nonzero(candidate.mask & host_envelope))
            containment = intersection / candidate_area
            if (
                containment >= config.nested_texture_minimum_containment
                and candidate_area / host_area
                <= config.nested_texture_maximum_host_fraction
            ):
                nested_surface_texture = True
                break
        if selected_profile in config.generic_region_blocked_profiles:
            vlm_physical = False
        accepted = vlm_physical or (
            not vlm_nonphysical
            and not nested_named_region
            and not laminar_surface_strip
            and (
                corroborated
                or silhouette_structure
                or silhouette_detail_structure
                or shape_structure
                or (
                    not nested_surface_texture
                    and (profile_structural_fallback or cross_source_structure)
                )
            )
        )
        if accepted:
            evidence = {
                "algorithm": "hpid-profile-physical-region-gate-v1",
                "profile": selected_profile,
                "policy": policy_reason,
                "named_corroboration": corroborated,
                "silhouette_structure": silhouette_structure,
                "silhouette_detail_structure": silhouette_detail_structure,
                "strong_single_view_detail_structure": strong_single_view_detail,
                "profile_structural_fallback": profile_structural_fallback,
                "cross_source_structure": cross_source_structure,
                "shape_structure": shape_structure,
                "nested_surface_texture": nested_surface_texture,
                "nested_named_region": nested_named_region,
                "laminar_surface_strip": laminar_surface_strip,
                "vlm_physical_supported": vlm_physical,
                "vlm_nonphysical_supported": vlm_nonphysical,
                "outer_boundary_contact": boundary_contact,
                "root_area_fraction": area_fraction,
                "ground_truth_used": False,
            }
            output.append(
                replace(
                    candidate,
                    metadata={
                        **candidate.metadata,
                        "physical_region_gate": evidence,
                    },
                )
            )
            corroborated_count += int(corroborated)
            silhouette_count += int(silhouette_structure)
            silhouette_count += int(silhouette_detail_structure)
            profile_fallback_count += int(profile_structural_fallback)
            cross_source_structure_count += int(cross_source_structure)
            shape_structure_count += int(shape_structure)
            vlm_physical_count += int(vlm_physical)
        else:
            rejected_count += 1
            vlm_nonphysical_count += int(vlm_nonphysical)
            nested_surface_texture_count += int(nested_surface_texture)
            nested_named_region_count += int(nested_named_region)
            laminar_surface_strip_count += int(laminar_surface_strip)
        rows.append(
            {
                "candidate_key": _candidate_key(candidate),
                "root_key": root_key,
                "selected_profile": selected_profile,
                "accepted": accepted,
                "named_corroboration": corroborated,
                "silhouette_structure": silhouette_structure,
                "silhouette_detail_structure": silhouette_detail_structure,
                "strong_single_view_detail_structure": strong_single_view_detail,
                "profile_structural_fallback": profile_structural_fallback,
                "cross_source_structure": cross_source_structure,
                "shape_structure": shape_structure,
                "nested_surface_texture": nested_surface_texture,
                "nested_named_region": nested_named_region,
                "laminar_surface_strip": laminar_surface_strip,
                "vlm_physical_supported": vlm_physical,
                "vlm_nonphysical_supported": vlm_nonphysical,
                "outer_boundary_contact": boundary_contact,
                "root_area_fraction": area_fraction,
                "visual_region_kind": region_kind,
                "ground_truth_used": False,
            }
        )
    return PhysicalRegionGateResult(
        tuple(output),
        {
            "algorithm": "hpid-profile-physical-region-gate-v1",
            "status": "completed",
            "input_candidate_count": len(candidates),
            "output_candidate_count": len(output),
            "evaluated_candidate_count": len(rows),
            "rejected_texture_fragment_count": rejected_count,
            "named_corroboration_count": corroborated_count,
            "silhouette_structure_count": silhouette_count,
            "profile_structural_fallback_count": profile_fallback_count,
            "cross_source_structure_count": cross_source_structure_count,
            "shape_structure_count": shape_structure_count,
            "nested_surface_texture_rejected_count": nested_surface_texture_count,
            "nested_named_region_rejected_count": nested_named_region_count,
            "laminar_surface_strip_rejected_count": laminar_surface_strip_count,
            "photometric_only_named_rejected_count": photometric_only_named_count,
            "vlm_physical_supported_count": vlm_physical_count,
            "vlm_nonphysical_rejected_count": vlm_nonphysical_count,
            "candidates": rows,
            "ground_truth_used": False,
        },
    )


def _distinct_existing_candidates(
    candidates: list[MaskCandidate],
) -> list[MaskCandidate]:
    selected: list[MaskCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: item.score * item.source_reliability,
        reverse=True,
    ):
        if any(
            mask_iou(candidate.mask, existing.mask) >= 0.50
            or _mask_containment(candidate.mask, existing.mask) >= 0.85
            for existing in selected
        ):
            continue
        selected.append(candidate)
    return selected


def rerank_visual_candidates(
    image: Image.Image,
    visual_candidates: list[MaskCandidate],
    roots: list[MaskCandidate],
    semantic_candidates: list[MaskCandidate],
    domains: dict[str, DomainPrompt],
    ranker: RegionLabelRanker | None,
    *,
    config: VisualSemanticConfig | None = None,
    semantic_constraints: dict[str, dict[str, int]] | None = None,
) -> VisualSemanticResult:
    """Attach semantic Part IDs to label-free regions only when evidence agrees."""

    config = config or VisualSemanticConfig()
    semantic_constraints = semantic_constraints or {}
    if ranker is None:
        return VisualSemanticResult(
            tuple(visual_candidates),
            {
                "algorithm": "hpid-visual-semantic-reranker-v5",
                "status": "skipped",
                "reason": "semantic_ranker_unavailable",
                "candidate_count": len(visual_candidates),
                "ground_truth_used": False,
            },
        )

    output_by_key = {
        _candidate_key(candidate): candidate for candidate in visual_candidates
    }
    root_rows: list[dict[str, object]] = []
    accepted_total = 0
    direct_accepted_total = 0
    inventory_rescue_total = 0
    inventory_cluster_bootstrap_total = 0
    capacity_fallback_proposal_total = 0
    capacity_fallback_accepted_total = 0
    within_root_consensus_total = 0
    consensus_proposals: list[dict[str, object]] = []
    roots_by_key = {_root_key(root): root for root in roots}
    for root_key, root in roots_by_key.items():
        domain = domains.get(root.semantic_name)
        generic = [
            candidate
            for candidate in visual_candidates
            if _root_key(candidate) == root_key
            and bool(candidate.metadata.get("generic_visual_region"))
        ]
        generic = sorted(
            generic,
            key=lambda item: (
                -float(item.metadata.get("sam_quality", item.score)),
                -int(np.count_nonzero(item.mask)),
                _candidate_key(item),
            ),
        )[: config.maximum_regions_per_root]
        if domain is None or not generic:
            root_rows.append(
                {
                    "root_key": root_key,
                    "root_semantic": root.semantic_name,
                    "generic_region_count": len(generic),
                    "accepted_count": 0,
                    "status": "no_domain_or_regions",
                }
            )
            continue
        (
            selected_parts,
            selected_profile,
            selection_reason,
            object_context,
            selected_subtype,
        ) = _inventory_for_root(root, domain)
        parts = _eligible_parts(selected_parts)
        constrained_semantics = semantic_constraints.get(root_key)
        prototype_constraint_applied = bool(
            constrained_semantics and selected_profile is None
        )
        if prototype_constraint_applied:
            parts = tuple(
                part
                for part in parts
                if part.semantic_name in constrained_semantics
            )
        if len(parts) < 2:
            root_rows.append(
                {
                    "root_key": root_key,
                    "root_semantic": root.semantic_name,
                    "selected_profile": selected_profile,
                    "generic_region_count": len(generic),
                    "eligible_part_count": len(parts),
                    "prototype_inventory_available": bool(constrained_semantics),
                    "prototype_inventory_constrained": (
                        prototype_constraint_applied
                    ),
                    "prototype_inventory_advisory": bool(
                        constrained_semantics and selected_profile is not None
                    ),
                    "accepted_count": 0,
                    "status": "insufficient_label_contrast",
                }
            )
            continue

        base_labels = [(part.semantic_name, _part_prompt(part)) for part in parts]
        base_scores = ranker.rank_regions_labels(
            image,
            [(_candidate_key(candidate), candidate.mask) for candidate in generic],
            base_labels,
            masked_weight=config.masked_weight,
            temperature=config.temperature,
        )
        contextual_labels = [
            (part.semantic_name, _part_prompt(part, object_context)) for part in parts
        ]
        use_contextual_route = bool(
            config.use_contextual_rescue and contextual_labels != base_labels
        )
        contextual_scores = (
            ranker.rank_regions_labels(
                image,
                [(_candidate_key(candidate), candidate.mask) for candidate in generic],
                contextual_labels,
                masked_weight=config.masked_weight,
                temperature=config.temperature,
            )
            if use_contextual_route
            else {}
        )
        part_by_name = {part.semantic_name: part for part in parts}
        selected_profile_definition = next(
            (
                profile
                for profile in domain.part_profiles
                if profile.name == selected_profile
            ),
            None,
        )
        maximum_by_name = {part.semantic_name: part.maximum_instances for part in parts}
        if prototype_constraint_applied:
            maximum_by_name = {
                name: min(limit, int(constrained_semantics[name]))
                for name, limit in maximum_by_name.items()
            }
        existing_by_name: dict[str, list[MaskCandidate]] = defaultdict(list)
        for candidate in semantic_candidates:
            if (
                _root_key(candidate) == root_key
                and candidate.semantic_name in maximum_by_name
                and not bool(candidate.metadata.get("generic_visual_region"))
            ):
                existing_by_name[candidate.semantic_name].append(candidate)
        existing_by_name = {
            name: _distinct_existing_candidates(candidates)
            for name, candidates in existing_by_name.items()
        }
        provisional_axis_anchor_rows: dict[
            str, tuple[tuple[float, float, float], MaskCandidate]
        ] = {}
        for candidate in generic:
            key = _candidate_key(candidate)
            endpoint_routes: list[dict[str, object]] = []
            for score_table in (
                base_scores.get(key, {}),
                contextual_scores.get(key, {}),
            ):
                alternatives = _route_alternatives(
                    score_table,
                    candidate,
                    part_by_name,
                    None,
                    config,
                )
                if alternatives:
                    endpoint_routes.append(alternatives[0])
            if not endpoint_routes:
                continue
            evidence = max(
                endpoint_routes,
                key=lambda row: (
                    float(row["adjusted_similarity"]),
                    float(row["probability"]),
                ),
            )
            semantic_name = str(evidence["best_semantic"])
            part = part_by_name[semantic_name]
            if (
                part.axis_position is None
                or abs(part.axis_position) < config.minimum_axis_anchor_position
                or float(evidence["similarity_margin"])
                < config.accepted_profile_similarity_margin
                or float(evidence["probability"])
                < max(config.axis_rescue_minimum_probability, 0.12)
                or not _geometry_compatible(candidate, part)
            ):
                continue
            rank = (
                float(evidence["similarity_margin"]),
                float(evidence["probability"]),
                float(evidence["adjusted_similarity"]),
            )
            anchor = replace(
                candidate,
                semantic_name=semantic_name,
                metadata={
                    **candidate.metadata,
                    "generic_visual_region": False,
                    "provisional_axis_anchor": True,
                    "provisional_axis_probability": float(
                        evidence["probability"]
                    ),
                    "provisional_axis_margin": float(
                        evidence["similarity_margin"]
                    ),
                },
            )
            previous = provisional_axis_anchor_rows.get(semantic_name)
            if previous is None or rank > previous[0]:
                provisional_axis_anchor_rows[semantic_name] = (rank, anchor)
        provisional_axis_anchors = [
            row[1]
            for _, row in sorted(provisional_axis_anchor_rows.items())
        ]
        axis_context = _build_axis_context(
            root,
            [*semantic_candidates, *provisional_axis_anchors],
            parts,
            config,
        )
        accepted_by_name: dict[str, int] = {}
        candidate_rows: list[dict[str, object]] = []
        proposals: list[
            tuple[
                float,
                MaskCandidate,
                PartPrompt,
                dict[str, object],
                dict[str, object],
                str,
                bool,
            ]
        ] = []
        capacity_fallback_proposal_count = 0
        capacity_fallback_accepted_count = 0
        for candidate in generic:
            key = _candidate_key(candidate)
            base_alternatives = _route_alternatives(
                base_scores.get(key, {}),
                candidate,
                part_by_name,
                axis_context,
                config,
            )
            base = base_alternatives[0] if base_alternatives else None
            contextual_alternatives = (
                _route_alternatives(
                    contextual_scores.get(key, {}),
                    candidate,
                    part_by_name,
                    axis_context,
                    config,
                )
                if use_contextual_route
                else []
            )
            contextual = (
                contextual_alternatives[0] if contextual_alternatives else None
            )
            if base is None and contextual is None:
                candidate_rows.append(
                    {
                        "candidate_key": key,
                        "status": "unresolved",
                        "reason": "no_geometry_compatible_label",
                    }
                )
                continue
            probability_floor = max(
                config.minimum_probability,
                config.uniform_probability_multiplier / len(parts),
            )
            margin_floor = (
                config.accepted_profile_similarity_margin
                if selected_profile is not None
                else config.minimum_similarity_margin
            )
            base_accepted = bool(
                base is not None
                and float(base["similarity_margin"]) >= margin_floor
                and float(base["probability"]) >= probability_floor
            )
            contextual_accepted = bool(
                not base_accepted
                and contextual is not None
                and float(contextual["similarity_margin"])
                >= max(
                    config.contextual_minimum_similarity_margin,
                    config.contextual_margin_multiplier * margin_floor,
                )
                and float(contextual["probability"])
                >= min(
                    1.0,
                    probability_floor + config.contextual_probability_increment,
                )
                and (
                    base is None
                    or contextual["best_semantic"] == base["best_semantic"]
                    or float(base["similarity_margin"]) < margin_floor
                )
            )
            axis_probability_floor = max(
                config.axis_rescue_minimum_probability,
                config.axis_rescue_uniform_probability_multiplier / len(parts),
            )

            axis_routes = [
                (route_name, evidence)
                for route_name, evidence in (
                    ("axis_structure_rescue", base),
                    ("contextual_axis_structure_rescue", contextual),
                )
                if _supports_axis_rescue(
                    evidence,
                    axis_context,
                    axis_probability_floor,
                    config,
                )
            ]
            axis_route, axis_evidence = (
                max(
                    axis_routes,
                    key=lambda item: (
                        float(item[1]["adjusted_similarity"]),
                        float(item[1]["similarity_margin"]),
                    ),
                )
                if axis_routes
                else (None, None)
            )
            axis_accepted = bool(
                not base_accepted
                and not contextual_accepted
                and axis_evidence
            )
            unresolved_routes = [
                (route_name, evidence)
                for route_name, evidence in (
                    ("base_unresolved", base),
                    ("contextual_unresolved", contextual),
                )
                if evidence is not None
            ]
            unresolved_route, unresolved_evidence = max(
                unresolved_routes,
                key=lambda item: (
                    float(item[1]["adjusted_similarity"]),
                    float(item[1]["similarity_margin"]),
                    float(item[1]["probability"]),
                ),
            )
            selected_route = (
                "base"
                if base_accepted
                else "contextual_rescue"
                if contextual_accepted
                else str(axis_route)
                if axis_accepted
                else unresolved_route
            )
            selected_evidence = (
                axis_evidence
                if axis_accepted
                else contextual
                if selected_route.startswith("contextual")
                else base
                if not selected_route.endswith("_unresolved")
                else unresolved_evidence
            )
            assert selected_evidence is not None
            selected_part = part_by_name[str(selected_evidence["best_semantic"])]
            inventory_rescue_accepted = bool(
                not base_accepted
                and not contextual_accepted
                and not axis_accepted
                and _supports_inventory_evidence_rescue(
                    candidate,
                    selected_part,
                    selected_evidence,
                    selected_profile=selected_profile,
                    config=config,
                )
            )
            if inventory_rescue_accepted:
                selected_route = "semantic_inventory_evidence_rescue"
            accepted = (
                base_accepted
                or contextual_accepted
                or axis_accepted
                or inventory_rescue_accepted
            )
            row = {
                "candidate_key": key,
                "status": "proposed" if accepted else "unresolved",
                "selected_route": selected_route,
                "best_semantic": selected_evidence["best_semantic"],
                "adjusted_similarity": selected_evidence["adjusted_similarity"],
                "similarity_margin": selected_evidence["similarity_margin"],
                "probability": selected_evidence["probability"],
                "probability_floor": probability_floor,
                "axis_rescue_probability_floor": axis_probability_floor,
                "axis_structure_rescue": axis_accepted,
                "semantic_inventory_evidence_rescue": inventory_rescue_accepted,
                "axis_coordinate": selected_evidence.get("axis_coordinate"),
                "axis_expected_position": selected_evidence.get(
                    "axis_expected_position"
                ),
                "axis_prior": selected_evidence.get("axis_prior", 0.0),
                "margin_floor": margin_floor,
                "visual_region_kind": candidate.metadata.get("visual_region_kind"),
                "base_evidence": (
                    {key: value for key, value in base.items() if key != "ranker_row"}
                    if base is not None
                    else None
                ),
                "contextual_evidence": (
                    {
                        key: value
                        for key, value in contextual.items()
                        if key != "ranker_row"
                    }
                    if contextual is not None
                    else None
                ),
                "capacity_fallbacks": [],
            }
            candidate_rows.append(row)
            consensus_proposals.append(
                {
                    "candidate": candidate,
                    "root": root,
                    "part": part_by_name[str(selected_evidence["best_semantic"])],
                    "selected_profile": selected_profile,
                    "selection_reason": selection_reason,
                    "object_context": object_context,
                    "probability": float(selected_evidence["probability"]),
                    "similarity": float(selected_evidence["adjusted_similarity"]),
                    "margin": float(selected_evidence["similarity_margin"]),
                    "probability_floor": float(probability_floor),
                    "margin_floor": float(margin_floor),
                    "row": row,
                    "base_scores": base_scores.get(key, {}),
                    "contextual_scores": contextual_scores.get(key, {}),
                }
            )
            if accepted:
                proposals.append(
                    (
                        (2.0 if base_accepted else 1.0)
                        + float(selected_evidence["adjusted_similarity"])
                        + float(selected_evidence["similarity_margin"]),
                        candidate,
                        part_by_name[str(selected_evidence["best_semantic"])],
                        row,
                        selected_evidence,
                        selected_route,
                        False,
                    )
                )
                inventory_rescue_total += int(inventory_rescue_accepted)
                if (
                    config.use_capacity_aware_assignment
                    and selected_profile is not None
                ):
                    fallback_by_semantic: dict[str, tuple[str, dict[str, object]]] = {}
                    for route_name, alternatives in (
                        ("base", base_alternatives),
                        ("contextual", contextual_alternatives),
                    ):
                        for evidence in alternatives[1:]:
                            semantic_name = str(evidence["best_semantic"])
                            existing = fallback_by_semantic.get(semantic_name)
                            if existing is None or float(
                                evidence["adjusted_similarity"]
                            ) > float(existing[1]["adjusted_similarity"]):
                                fallback_by_semantic[semantic_name] = (
                                    route_name,
                                    evidence,
                                )
                    fallback_probability_floor = max(
                        config.capacity_fallback_minimum_probability,
                        probability_floor
                        * config.capacity_fallback_probability_multiplier,
                    )
                    for route_name, evidence in fallback_by_semantic.values():
                        similarity_drop = float(
                            selected_evidence["adjusted_similarity"]
                        ) - float(evidence["adjusted_similarity"])
                        if (
                            similarity_drop
                            > config.capacity_fallback_maximum_similarity_drop
                            or float(evidence["probability"])
                            < fallback_probability_floor
                        ):
                            continue
                        semantic_name = str(evidence["best_semantic"])
                        selected_part = part_by_name[
                            str(selected_evidence["best_semantic"])
                        ]
                        fallback_part = part_by_name[semantic_name]
                        if (
                            selected_part.detail
                            or fallback_part.detail
                            or selected_part.axis_position is None
                            or fallback_part.axis_position is None
                        ):
                            continue
                        fallback_row = {
                            "semantic_name": semantic_name,
                            "route": route_name,
                            "adjusted_similarity": float(
                                evidence["adjusted_similarity"]
                            ),
                            "probability": float(evidence["probability"]),
                            "similarity_drop": similarity_drop,
                            "probability_floor": fallback_probability_floor,
                            "status": "proposed",
                        }
                        row["capacity_fallbacks"].append(fallback_row)
                        proposals.append(
                            (
                                0.5
                                + float(evidence["adjusted_similarity"])
                                + 0.25 * float(evidence["probability"]),
                                candidate,
                                part_by_name[semantic_name],
                                row,
                                evidence,
                                f"capacity_fallback_{route_name}",
                                True,
                            )
                        )
                        capacity_fallback_proposal_count += 1

        proposals.sort(key=lambda item: item[0], reverse=True)
        accepted_keys: set[str] = set()
        for _, candidate, part, row, evidence, route_name, is_fallback in proposals:
            key = _candidate_key(candidate)
            if key in accepted_keys:
                continue
            current_count = accepted_by_name.get(part.semantic_name, 0)
            existing_count = len(existing_by_name.get(part.semantic_name, []))
            if existing_count + current_count >= maximum_by_name[part.semantic_name]:
                if is_fallback:
                    for fallback in row["capacity_fallbacks"]:
                        if fallback["semantic_name"] == part.semantic_name:
                            fallback["status"] = "capacity_reached"
                else:
                    row["reason"] = "existing_or_new_maximum_instances_reached"
                continue
            if any(
                mask_iou(candidate.mask, existing.mask) >= 0.72
                or _mask_containment(candidate.mask, existing.mask) >= 0.92
                for existing in existing_by_name.get(part.semantic_name, [])
            ):
                if is_fallback:
                    for fallback in row["capacity_fallbacks"]:
                        if fallback["semantic_name"] == part.semantic_name:
                            fallback["status"] = "semantic_region_duplicate"
                else:
                    row["reason"] = "existing_semantic_region_duplicate"
                continue
            if is_fallback:
                row.update(
                    {
                        "selected_route": route_name,
                        "best_semantic": evidence["best_semantic"],
                        "adjusted_similarity": evidence["adjusted_similarity"],
                        "similarity_margin": evidence["similarity_margin"],
                        "probability": evidence["probability"],
                        "axis_structure_rescue": False,
                        "axis_coordinate": evidence.get("axis_coordinate"),
                        "axis_expected_position": evidence.get(
                            "axis_expected_position"
                        ),
                        "axis_prior": evidence.get("axis_prior", 0.0),
                    }
                )
            semantic_parent = part.semantic_parent or root.semantic_name
            assembly_parent = part.assembly_parent or semantic_parent
            metadata = {
                **candidate.metadata,
                "generic_visual_region": False,
                "semantic_reranked": True,
                "semantic_rerank_algorithm": (
                    "dual-route-region-text-geometry-axis-consensus-v4"
                ),
                "semantic_rerank_route": row["selected_route"],
                "semantic_axis_structure_rescue": row["axis_structure_rescue"],
                "semantic_axis_coordinate": row["axis_coordinate"],
                "semantic_axis_expected_position": row["axis_expected_position"],
                "semantic_axis_prior": row["axis_prior"],
                "semantic_rerank_probability": row["probability"],
                "semantic_rerank_similarity": row["adjusted_similarity"],
                "semantic_rerank_margin": row["similarity_margin"],
                "semantic_rerank_profile": selected_profile,
                "semantic_rerank_inventory_reason": selection_reason,
                "maximum_instances": part.maximum_instances,
                "detail": part.detail,
                "parent_candidate_key": _best_parent_key(
                    candidate,
                    semantic_parent,
                    semantic_candidates,
                    root,
                ),
                "assembly_parent_semantic": assembly_parent,
                "assembly_parent_candidate_key": _best_parent_key(
                    candidate,
                    assembly_parent,
                    semantic_candidates,
                    root,
                ),
                "ground_truth_used": False,
            }
            output_by_key[key] = replace(
                candidate,
                semantic_name=part.semantic_name,
                semantic_parent=semantic_parent,
                score=float(
                    np.clip(
                        0.65 * candidate.score + 0.35 * float(row["probability"]),
                        0.0,
                        1.0,
                    )
                ),
                source=f"{candidate.source}/semantic-rerank",
                prompt=_part_prompt(part, object_context),
                source_reliability=max(
                    candidate.source_reliability, config.source_reliability
                ),
                metadata=metadata,
            )
            accepted_by_name[part.semantic_name] = current_count + 1
            accepted_keys.add(key)
            row["status"] = "accepted"
            accepted_total += 1
            if is_fallback:
                capacity_fallback_accepted_count += 1
                capacity_fallback_accepted_total += 1
                row["reason"] = "capacity_aware_semantic_fallback"
                for fallback in row["capacity_fallbacks"]:
                    if fallback["semantic_name"] == part.semantic_name:
                        fallback["status"] = "accepted"
            else:
                direct_accepted_total += 1

        inventory_cluster_bootstrap_count = 0
        inventory_cluster_bootstrap_rows: list[dict[str, object]] = []
        if config.use_inventory_cluster_bootstrap and selected_profile is not None:
            unresolved_by_semantic: dict[str, list[dict[str, object]]] = defaultdict(list)
            for proposal in consensus_proposals:
                proposal_root = proposal["root"]
                candidate = proposal["candidate"]
                part = proposal["part"]
                row = proposal["row"]
                assert isinstance(proposal_root, MaskCandidate)
                assert isinstance(candidate, MaskCandidate)
                assert isinstance(part, PartPrompt)
                assert isinstance(row, dict)
                if (
                    _root_key(proposal_root) != root_key
                    or not part.detail
                    or row.get("status") != "unresolved"
                    or float(proposal["probability"])
                    < config.inventory_cluster_minimum_probability
                    or float(proposal["margin"])
                    < config.inventory_cluster_minimum_margin
                    or not _geometry_compatible(candidate, part)
                ):
                    continue
                structure = _appearance_structure(candidate)
                structurally_supported = bool(
                    structure["multi_view"]
                    or (
                        int(structure["cue_count"])
                        >= config.inventory_rescue_minimum_independent_cues
                        and float(structure["closure"])
                        >= config.inventory_rescue_minimum_boundary_closure
                    )
                    or float(structure["geometric_support"])
                    >= config.inventory_rescue_minimum_geometric_support
                )
                if (
                    not structurally_supported
                    or float(structure["shading_penalty"])
                    > config.inventory_rescue_maximum_shading_penalty
                ):
                    continue
                unresolved_by_semantic[part.semantic_name].append(
                    {
                        "candidate": candidate,
                        "part": part,
                        "row": row,
                        "proposal": proposal,
                        "structure": structure,
                    }
                )

            for semantic_name, support in unresolved_by_semantic.items():
                if len(support) < config.inventory_cluster_minimum_votes:
                    continue
                current_count = accepted_by_name.get(semantic_name, 0)
                existing_count = len(existing_by_name.get(semantic_name, []))
                available = max(
                    0,
                    maximum_by_name[semantic_name] - existing_count - current_count,
                )
                promotion_limit = min(
                    available,
                    config.inventory_cluster_maximum_promotions,
                    max(1, len(support) // config.inventory_cluster_minimum_votes),
                )
                chosen: list[MaskCandidate] = []
                ordered_support = sorted(
                    support,
                    key=lambda item: (
                        float(item["proposal"]["probability"]),
                        float(item["proposal"]["margin"]),
                        int(item["structure"]["cue_count"]),
                        float(item["structure"]["closure"]),
                    ),
                    reverse=True,
                )
                for item in ordered_support:
                    if len(chosen) >= promotion_limit:
                        break
                    candidate = item["candidate"]
                    part = item["part"]
                    row = item["row"]
                    proposal = item["proposal"]
                    assert isinstance(candidate, MaskCandidate)
                    assert isinstance(part, PartPrompt)
                    assert isinstance(row, dict)
                    key = _candidate_key(candidate)
                    if key in accepted_keys or any(
                        mask_iou(candidate.mask, other.mask) >= 0.50
                        or _mask_containment(candidate.mask, other.mask) >= 0.85
                        for other in chosen
                    ):
                        continue
                    semantic_parent = part.semantic_parent or root.semantic_name
                    assembly_parent = part.assembly_parent or semantic_parent
                    probability = float(proposal["probability"])
                    margin = float(proposal["margin"])
                    metadata = {
                        **candidate.metadata,
                        "generic_visual_region": False,
                        "semantic_reranked": True,
                        "semantic_rerank_algorithm": (
                            "semantic-inventory-structure-appearance-consensus-v5"
                        ),
                        "semantic_rerank_route": "inventory_cluster_bootstrap",
                        "semantic_inventory_cluster_vote_count": len(support),
                        "semantic_rerank_probability": probability,
                        "semantic_rerank_similarity": float(proposal["similarity"]),
                        "semantic_rerank_margin": margin,
                        "semantic_rerank_profile": selected_profile,
                        "semantic_rerank_inventory_reason": selection_reason,
                        "maximum_instances": part.maximum_instances,
                        "detail": part.detail,
                        "parent_candidate_key": _best_parent_key(
                            candidate,
                            semantic_parent,
                            semantic_candidates,
                            root,
                        ),
                        "assembly_parent_semantic": assembly_parent,
                        "assembly_parent_candidate_key": _best_parent_key(
                            candidate,
                            assembly_parent,
                            semantic_candidates,
                            root,
                        ),
                        "ground_truth_used": False,
                    }
                    output_by_key[key] = replace(
                        candidate,
                        semantic_name=part.semantic_name,
                        semantic_parent=semantic_parent,
                        score=float(
                            np.clip(
                                0.68 * candidate.score + 0.32 * probability,
                                0.0,
                                1.0,
                            )
                        ),
                        source=(
                            f"{candidate.source}/semantic-inventory-cluster"
                        ),
                        prompt=_part_prompt(part, object_context),
                        source_reliability=max(
                            candidate.source_reliability,
                            config.within_root_source_reliability,
                        ),
                        metadata=metadata,
                    )
                    chosen.append(candidate)
                    accepted_keys.add(key)
                    accepted_by_name[semantic_name] = (
                        accepted_by_name.get(semantic_name, 0) + 1
                    )
                    row["status"] = "accepted"
                    row["reason"] = "semantic_inventory_cluster_bootstrap"
                    row["selected_route"] = "inventory_cluster_bootstrap"
                    accepted_total += 1
                    inventory_cluster_bootstrap_count += 1
                    inventory_cluster_bootstrap_total += 1
                    inventory_cluster_bootstrap_rows.append(
                        {
                            "candidate_key": key,
                            "semantic_name": semantic_name,
                            "vote_count": len(support),
                            "probability": probability,
                            "margin": margin,
                        }
                    )

        capacity_fallback_proposal_total += capacity_fallback_proposal_count
        for row in candidate_rows:
            if row.get("status") == "proposed":
                row["status"] = "unresolved"
                row.setdefault("reason", "no_capacity_compatible_assignment")

        within_root_consensus_count = 0
        within_root_consensus_rows: list[dict[str, object]] = []
        if config.use_within_root_repetition_consensus:
            root_proposals: list[dict[str, object]] = []
            for proposal in consensus_proposals:
                proposal_root = proposal["root"]
                assert isinstance(proposal_root, MaskCandidate)
                if _root_key(proposal_root) == root_key:
                    root_proposals.append(proposal)
            root_proposals.sort(
                key=lambda proposal: (
                    float(proposal["probability"]),
                    float(proposal["margin"]),
                    float(proposal["similarity"]),
                ),
                reverse=True,
            )
            anchors_by_semantic: dict[str, list[MaskCandidate]] = defaultdict(list)
            for assigned in output_by_key.values():
                if (
                    _root_key(assigned) == root_key
                    and bool(assigned.metadata.get("semantic_reranked"))
                    and not bool(
                        assigned.metadata.get(
                            "semantic_within_root_repetition_consensus"
                        )
                    )
                ):
                    anchors_by_semantic[assigned.semantic_name].append(assigned)
            for proposal in root_proposals:
                candidate = proposal["candidate"]
                part = proposal["part"]
                row = proposal["row"]
                assert isinstance(candidate, MaskCandidate)
                assert isinstance(part, PartPrompt)
                assert isinstance(row, dict)
                key = _candidate_key(candidate)
                current = output_by_key[key]
                anchors = anchors_by_semantic.get(part.semantic_name, [])
                confusion_reassigned_from: str | None = None
                confusion_evidence: dict[str, float] | None = None
                repeated_macro_supported = _supports_repeated_macro_component(
                    candidate,
                    config,
                )
                if not anchors and selected_profile_definition is not None and (
                    part.detail or repeated_macro_supported
                ):
                    confusion_group = set(
                        selected_profile_definition.confusion_group_for(
                            part.semantic_name
                        )
                    )
                    if (
                        part.semantic_parent is not None
                        and part.semantic_parent in part_by_name
                    ):
                        confusion_group.add(part.semantic_parent)
                    alternatives: list[
                        tuple[
                            int,
                            float,
                            PartPrompt,
                            list[MaskCandidate],
                            dict[str, float],
                        ]
                    ] = []
                    for semantic_name in sorted(confusion_group):
                        alternative = part_by_name.get(semantic_name)
                        alternative_anchors = anchors_by_semantic.get(
                            semantic_name, []
                        )
                        if (
                            alternative is None
                            or alternative.semantic_name == part.semantic_name
                            or alternative.maximum_instances
                            < config.within_root_minimum_capacity
                            or not alternative_anchors
                            or (
                                not alternative.detail
                                and not repeated_macro_supported
                            )
                        ):
                            continue
                        route_evidence = [
                            evidence
                            for evidence in (
                                _specific_route_evidence(
                                    proposal["base_scores"],
                                    candidate,
                                    alternative,
                                    axis_context,
                                    config,
                                ),
                                _specific_route_evidence(
                                    proposal["contextual_scores"],
                                    candidate,
                                    alternative,
                                    axis_context,
                                    config,
                                ),
                            )
                            if evidence is not None
                        ]
                        if not route_evidence:
                            continue
                        evidence = max(
                            route_evidence,
                            key=lambda item: (
                                item["adjusted_similarity"],
                                item["probability"],
                            ),
                        )
                        if (
                            evidence["probability"]
                            < float(proposal["probability_floor"])
                            * config.within_root_probability_multiplier
                            or evidence["adjusted_similarity"]
                            < float(proposal["similarity"])
                            - config.within_root_confusion_maximum_similarity_drop
                        ):
                            continue
                        alternatives.append(
                            (
                                len(alternative_anchors),
                                evidence["adjusted_similarity"],
                                alternative,
                                alternative_anchors,
                                evidence,
                            )
                        )
                    if alternatives:
                        (
                            _,
                            _,
                            alternative,
                            alternative_anchors,
                            confusion_evidence,
                        ) = max(
                            alternatives,
                            key=lambda item: (item[0], item[1]),
                        )
                        confusion_reassigned_from = part.semantic_name
                        part = alternative
                        anchors = alternative_anchors
                repeated_detail = bool(
                    part.detail
                    and candidate.metadata.get("visual_region_kind") == "detail"
                )
                repeated_macro = bool(
                    not part.detail and repeated_macro_supported
                )
                if (
                    not bool(current.metadata.get("generic_visual_region"))
                    or selected_profile is None
                    or part.maximum_instances < config.within_root_minimum_capacity
                    or not (repeated_detail or repeated_macro)
                    or not anchors
                    or float(proposal["probability"])
                    < float(proposal["probability_floor"])
                    * config.within_root_probability_multiplier
                    or float(proposal["margin"])
                    < config.within_root_minimum_margin
                ):
                    continue
                candidate_area = max(1, int(np.count_nonzero(candidate.mask)))
                anchor_areas = [
                    max(1, int(np.count_nonzero(anchor.mask))) for anchor in anchors
                ]
                median_anchor_area = float(np.median(anchor_areas))
                area_ratio = max(
                    candidate_area / median_anchor_area,
                    median_anchor_area / candidate_area,
                )
                maximum_area_ratio = (
                    config.within_root_macro_maximum_area_ratio
                    if repeated_macro
                    else config.within_root_maximum_area_ratio
                )
                if area_ratio > maximum_area_ratio:
                    continue
                current_count = accepted_by_name.get(part.semantic_name, 0)
                existing_count = len(existing_by_name.get(part.semantic_name, []))
                if existing_count + current_count >= maximum_by_name[part.semantic_name]:
                    continue
                if any(
                    mask_iou(candidate.mask, assigned.mask) >= 0.72
                    or _mask_containment(candidate.mask, assigned.mask) >= 0.92
                    for assigned in (
                        *existing_by_name.get(part.semantic_name, []),
                        *anchors,
                    )
                ):
                    continue
                semantic_parent = part.semantic_parent or root.semantic_name
                assembly_parent = part.assembly_parent or semantic_parent
                metadata = {
                    **candidate.metadata,
                    "generic_visual_region": False,
                    "semantic_reranked": True,
                    "semantic_rerank_algorithm": (
                        "dual-route-region-text-geometry-axis-consensus-v4"
                    ),
                    "semantic_rerank_route": "within_root_repetition_consensus",
                    "semantic_within_root_repetition_consensus": True,
                    "semantic_confusion_reassigned_from": (
                        confusion_reassigned_from
                    ),
                    "semantic_confusion_evidence": confusion_evidence,
                    "semantic_repetition_anchor_count": len(anchors),
                    "semantic_repetition_area_ratio": float(area_ratio),
                    "semantic_repetition_kind": (
                        "macro_component" if repeated_macro else "detail"
                    ),
                    "semantic_rerank_probability": float(proposal["probability"]),
                    "semantic_rerank_similarity": float(proposal["similarity"]),
                    "semantic_rerank_margin": float(proposal["margin"]),
                    "semantic_rerank_profile": selected_profile,
                    "semantic_rerank_inventory_reason": selection_reason,
                    "maximum_instances": part.maximum_instances,
                    "detail": part.detail,
                    "parent_candidate_key": _best_parent_key(
                        candidate,
                        semantic_parent,
                        semantic_candidates,
                        root,
                    ),
                    "assembly_parent_semantic": assembly_parent,
                    "assembly_parent_candidate_key": _best_parent_key(
                        candidate,
                        assembly_parent,
                        semantic_candidates,
                        root,
                    ),
                    "ground_truth_used": False,
                }
                assigned = replace(
                    candidate,
                    semantic_name=part.semantic_name,
                    semantic_parent=semantic_parent,
                    score=float(
                        np.clip(
                            0.68 * candidate.score
                            + 0.32 * float(proposal["probability"]),
                            0.0,
                            1.0,
                        )
                    ),
                    source=f"{candidate.source}/within-root-repetition-consensus",
                    prompt=_part_prompt(part, object_context),
                    source_reliability=max(
                        candidate.source_reliability,
                        config.within_root_source_reliability,
                    ),
                    metadata=metadata,
                )
                output_by_key[key] = assigned
                anchors_by_semantic[part.semantic_name].append(assigned)
                accepted_by_name[part.semantic_name] = current_count + 1
                accepted_keys.add(key)
                row["status"] = "accepted"
                row["reason"] = "within_root_repetition_consensus"
                within_root_consensus_count += 1
                within_root_consensus_total += 1
                accepted_total += 1
                within_root_consensus_rows.append(
                    {
                        "candidate_key": key,
                        "semantic_name": part.semantic_name,
                        "confusion_reassigned_from": confusion_reassigned_from,
                        "anchor_count": len(anchors),
                        "area_ratio": float(area_ratio),
                        "repetition_kind": (
                            "macro_component" if repeated_macro else "detail"
                        ),
                        "probability": float(proposal["probability"]),
                        "probability_floor": float(proposal["probability_floor"]),
                    }
                )

        root_rows.append(
            {
                "root_key": root_key,
                "root_semantic": root.semantic_name,
                "selected_profile": selected_profile,
                "selected_subtype": selected_subtype,
                "object_context": object_context,
                "contextual_rescue_enabled": use_contextual_route,
                "inventory_reason": selection_reason,
                "generic_region_count": len(generic),
                "eligible_part_count": len(parts),
                "prototype_inventory_available": bool(constrained_semantics),
                "prototype_inventory_constrained": prototype_constraint_applied,
                "prototype_inventory_advisory": bool(
                    constrained_semantics and selected_profile is not None
                ),
                "prototype_inventory_size": (
                    len(constrained_semantics) if constrained_semantics else 0
                ),
                "existing_by_semantic": {
                    name: len(candidates)
                    for name, candidates in sorted(existing_by_name.items())
                },
                "axis_context": (
                    {
                        "orientation_sign": axis_context.orientation_sign,
                        "anchor_count": axis_context.anchor_count,
                        "orientation_margin": axis_context.orientation_margin,
                        "provisional_anchor_count": len(
                            provisional_axis_anchors
                        ),
                    }
                    if axis_context is not None
                    else None
                ),
                "accepted_count": len(accepted_keys),
                "accepted_by_semantic": accepted_by_name,
                "capacity_fallback_proposal_count": (
                    capacity_fallback_proposal_count
                ),
                "capacity_fallback_accepted_count": (
                    capacity_fallback_accepted_count
                ),
                "inventory_cluster_bootstrap_count": (
                    inventory_cluster_bootstrap_count
                ),
                "inventory_cluster_bootstrap_rows": (
                    inventory_cluster_bootstrap_rows
                ),
                "within_root_repetition_consensus_count": (
                    within_root_consensus_count
                ),
                "within_root_repetition_consensus_rows": (
                    within_root_consensus_rows
                ),
                "status": "completed",
                "candidates": candidate_rows,
            }
        )

    repeated_consensus_count = 0
    repeated_consensus_rows: list[dict[str, object]] = []
    if config.use_repeated_instance_consensus and consensus_proposals:
        direct_anchors: dict[
            tuple[str, str, str], list[MaskCandidate]
        ] = defaultdict(list)
        for candidate in output_by_key.values():
            if not bool(candidate.metadata.get("semantic_reranked")):
                continue
            profile = candidate.metadata.get("semantic_rerank_profile")
            if profile is None:
                continue
            root = roots_by_key.get(_root_key(candidate))
            if root is None:
                continue
            direct_anchors[
                (root.semantic_name, str(profile), candidate.semantic_name)
            ].append(candidate)

        root_rows_by_key = {
            str(row["root_key"]): row
            for row in root_rows
            if row.get("root_key") is not None
        }
        accepted_masks_by_root_semantic: dict[
            tuple[str, str], list[MaskCandidate]
        ] = defaultdict(list)
        for candidate in semantic_candidates:
            if bool(candidate.metadata.get("generic_visual_region")):
                continue
            accepted_masks_by_root_semantic[
                (_root_key(candidate), candidate.semantic_name)
            ].append(candidate)
        for candidate in output_by_key.values():
            if bool(candidate.metadata.get("generic_visual_region")):
                continue
            accepted_masks_by_root_semantic[
                (_root_key(candidate), candidate.semantic_name)
            ].append(candidate)

        ranked_consensus = sorted(
            consensus_proposals,
            key=lambda item: (
                float(item["probability"]),
                float(item["margin"]),
                float(item["similarity"]),
            ),
            reverse=True,
        )
        for proposal in ranked_consensus:
            candidate = proposal["candidate"]
            assert isinstance(candidate, MaskCandidate)
            key = _candidate_key(candidate)
            current = output_by_key[key]
            if not bool(current.metadata.get("generic_visual_region")):
                continue
            root = proposal["root"]
            part = proposal["part"]
            assert isinstance(root, MaskCandidate)
            assert isinstance(part, PartPrompt)
            profile = proposal["selected_profile"]
            if profile is None:
                continue
            probability = float(proposal["probability"])
            probability_floor = float(proposal["probability_floor"])
            margin = float(proposal["margin"])
            if (
                probability
                < probability_floor
                * config.repeated_consensus_probability_multiplier
                or margin < config.repeated_consensus_minimum_margin
            ):
                continue
            anchor_key = (
                root.semantic_name,
                str(profile),
                part.semantic_name,
            )
            anchors = [
                anchor
                for anchor in direct_anchors.get(anchor_key, [])
                if _root_key(anchor) != _root_key(root)
            ]
            anchor_roots = sorted({_root_key(anchor) for anchor in anchors})
            if len(anchor_roots) < config.repeated_consensus_minimum_anchor_roots:
                continue
            candidate_kind = str(
                candidate.metadata.get("visual_region_kind", "panel")
            )
            anchor_kinds = {
                str(anchor.metadata.get("visual_region_kind", "panel"))
                for anchor in anchors
            }
            if candidate_kind not in anchor_kinds:
                continue
            anchor_fractions = [
                float(anchor.metadata.get("root_area_fraction", 0.0))
                for anchor in anchors
                if float(anchor.metadata.get("root_area_fraction", 0.0)) > 0.0
            ]
            candidate_fraction = float(
                candidate.metadata.get("root_area_fraction", 0.0)
            )
            if not anchor_fractions or candidate_fraction <= 0.0:
                continue
            median_fraction = float(np.median(anchor_fractions))
            area_ratio = max(candidate_fraction, median_fraction) / max(
                1e-9, min(candidate_fraction, median_fraction)
            )
            if area_ratio > config.repeated_consensus_maximum_area_ratio:
                continue
            existing = _distinct_existing_candidates(
                accepted_masks_by_root_semantic.get(
                    (_root_key(root), part.semantic_name), []
                )
            )
            if len(existing) >= part.maximum_instances:
                continue
            if any(
                mask_iou(candidate.mask, accepted.mask) >= 0.72
                or _mask_containment(candidate.mask, accepted.mask) >= 0.92
                for accepted in existing
            ):
                continue
            semantic_parent = part.semantic_parent or root.semantic_name
            assembly_parent = part.assembly_parent or semantic_parent
            metadata = {
                **candidate.metadata,
                "generic_visual_region": False,
                "semantic_reranked": True,
                "semantic_repeated_instance_consensus": True,
                "semantic_rerank_algorithm": (
                    "repeated-instance-part-consensus-v1"
                ),
                "semantic_rerank_probability": probability,
                "semantic_rerank_similarity": float(proposal["similarity"]),
                "semantic_rerank_margin": margin,
                "semantic_rerank_profile": profile,
                "semantic_rerank_inventory_reason": proposal[
                    "selection_reason"
                ],
                "semantic_consensus_anchor_roots": anchor_roots,
                "semantic_consensus_area_ratio": area_ratio,
                "maximum_instances": part.maximum_instances,
                "detail": part.detail,
                "parent_candidate_key": _best_parent_key(
                    candidate,
                    semantic_parent,
                    semantic_candidates,
                    root,
                ),
                "assembly_parent_semantic": assembly_parent,
                "assembly_parent_candidate_key": _best_parent_key(
                    candidate,
                    assembly_parent,
                    semantic_candidates,
                    root,
                ),
                "ground_truth_used": False,
            }
            resolved = replace(
                candidate,
                semantic_name=part.semantic_name,
                semantic_parent=semantic_parent,
                score=float(
                    np.clip(
                        0.72 * candidate.score + 0.28 * probability,
                        0.0,
                        1.0,
                    )
                ),
                source=f"{candidate.source}/semantic-consensus",
                prompt=_part_prompt(part, str(proposal["object_context"])),
                source_reliability=max(
                    candidate.source_reliability,
                    config.repeated_consensus_source_reliability,
                ),
                metadata=metadata,
            )
            output_by_key[key] = resolved
            accepted_masks_by_root_semantic[
                (_root_key(root), part.semantic_name)
            ].append(resolved)
            row = proposal["row"]
            assert isinstance(row, dict)
            row["status"] = "accepted_repeated_instance_consensus"
            row["consensus_anchor_root_count"] = len(anchor_roots)
            row["consensus_area_ratio"] = area_ratio
            root_row = root_rows_by_key.get(_root_key(root))
            if root_row is not None:
                root_row["accepted_count"] = int(root_row["accepted_count"]) + 1
                root_row["repeated_consensus_accepted_count"] = int(
                    root_row.get("repeated_consensus_accepted_count", 0)
                ) + 1
                accepted_by_semantic = root_row["accepted_by_semantic"]
                assert isinstance(accepted_by_semantic, dict)
                accepted_by_semantic[part.semantic_name] = int(
                    accepted_by_semantic.get(part.semantic_name, 0)
                ) + 1
            repeated_consensus_rows.append(
                {
                    "candidate_key": key,
                    "root_key": _root_key(root),
                    "semantic_name": part.semantic_name,
                    "anchor_root_count": len(anchor_roots),
                    "area_ratio": area_ratio,
                    "probability": probability,
                    "margin": margin,
                }
            )
            repeated_consensus_count += 1
            accepted_total += 1

    return VisualSemanticResult(
        tuple(
            output_by_key[_candidate_key(candidate)] for candidate in visual_candidates
        ),
        {
            "algorithm": "hpid-visual-semantic-reranker-v5",
            "status": "completed",
            "candidate_count": len(visual_candidates),
            "generic_input_count": sum(
                bool(candidate.metadata.get("generic_visual_region"))
                for candidate in visual_candidates
            ),
            "accepted_semantic_count": accepted_total,
            "direct_accepted_semantic_count": direct_accepted_total,
            "inventory_evidence_rescue_count": inventory_rescue_total,
            "inventory_cluster_bootstrap_count": (
                inventory_cluster_bootstrap_total
            ),
            "capacity_fallback_proposal_count": (
                capacity_fallback_proposal_total
            ),
            "capacity_fallback_accepted_count": (
                capacity_fallback_accepted_total
            ),
            "within_root_repetition_consensus_count": (
                within_root_consensus_total
            ),
            "repeated_instance_consensus_count": repeated_consensus_count,
            "repeated_instance_consensus_rows": repeated_consensus_rows,
            "unresolved_generic_count": sum(
                bool(candidate.metadata.get("generic_visual_region"))
                for candidate in output_by_key.values()
            ),
            "roots": root_rows,
            "ground_truth_used": False,
        },
    )
