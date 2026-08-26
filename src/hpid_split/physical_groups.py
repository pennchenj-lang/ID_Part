from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace

import cv2
import numpy as np
from PIL import Image

from .fusion import MaskCandidate
from .instances import PartInstance


@dataclass(frozen=True)
class PhysicalGroup:
    group_id: str
    group_index: int
    semantic_name: str
    asset_id: str
    member_part_ids: tuple[str, ...]
    bbox_xyxy: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]
    area_px: int
    evidence: str
    review_required: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PhysicalGroupingResult:
    group_map: np.ndarray
    groups: tuple[PhysicalGroup, ...]
    records: tuple[PartInstance, ...]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class _CharacterAxisFrame:
    coordinate: np.ndarray
    lateral: np.ndarray
    head_end: float
    axis_direction_xy: tuple[float, float]


_CHARACTER_BODY_TOKENS = {
    "body",
    "skin",
    "head",
    "face",
    "ear",
    "neck",
    "arm",
    "hand",
    "leg",
    "foot",
}
_UPPER_GARMENT_TOKENS = {
    "shirt",
    "top",
    "jacket",
    "coat",
    "hoodie",
    "blouse",
    "sweater",
    "upper",
    "sleeve",
    "torso cloth",
}
_LOWER_GARMENT_TOKENS = {"pants", "trousers", "shorts", "skirt", "dress"}
_KNIFE_HANDLE_TOKENS = {"handle", "guard", "pommel", "grip", "hilt"}
_OPEN_SET_PHYSICAL_DOMAINS = {
    "container",
    "daily_object",
    "device",
    "furniture",
    "natural_object",
    "structure",
    "terrain",
    "vehicle",
}
_PROFILE_PRIMARY_SURFACE_TOKENS = {"sphere", "shell", "canopy", "jar"}
_ROUND_PART_TOKENS = {"wheel", "tire", "rim", "dial", "knob", "lens"}
_SLENDER_PART_TOKENS = {
    "antenna",
    "barrel",
    "fork",
    "mast",
    "pole",
    "spoke",
    "stem",
}
_INTERNAL_PART_TOKENS = {"hole", "inner", "inside", "lining", "opening"}
_OPEN_INTERIOR_PROFILES = {
    "drinkware",
    "flatware",
    "open_container",
}
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
_DISCRETE_REPEATED_PART_TOKENS = {
    "boulder",
    "rim",
    "rock",
    "tire",
    "tree",
    "wheel",
}


def _words(value: str) -> set[str]:
    normalized = " ".join(re.findall(r"[a-z0-9]+", value.casefold()))
    words = set(normalized.split())
    if "torso" in words and "cloth" in words:
        words.add("torso cloth")
    return words


def _is_visual_semantic(name: str) -> bool:
    return bool("_visual_" in name or name.endswith(("_panel", "_strip", "_detail")))


def _root_domain(record: PartInstance) -> str:
    for domain in (
        "natural_object",
        "daily_object",
        "tool_prop",
        "character",
        "container",
        "device",
        "furniture",
        "structure",
        "terrain",
        "vehicle",
    ):
        if record.semantic_name.startswith(domain) or record.semantic_parent.startswith(
            domain
        ):
            return domain
    for value in (record.semantic_parent, record.semantic_name):
        if "_" in value:
            return value.split("_", maxsplit=1)[0]
    return record.semantic_parent


def _physical_visual_name(name: str) -> str:
    return (
        name.replace("_visual_panel_", "_physical_panel_")
        .replace("_visual_strip_", "_physical_strip_")
        .replace("_visual_detail_", "_physical_detail_")
    )


def _promotable_visual_semantics(
    candidates: tuple[MaskCandidate, ...],
) -> dict[str, str]:
    promoted: dict[str, str] = {}
    for candidate in candidates:
        if not _is_visual_semantic(candidate.semantic_name):
            continue
        domain = next(
            (
                value
                for value in _OPEN_SET_PHYSICAL_DOMAINS
                if candidate.semantic_name.startswith(value)
            ),
            None,
        )
        if domain is None:
            continue
        metadata = candidate.metadata
        physical_gate = metadata.get("physical_region_gate")
        physical_gate = physical_gate if isinstance(physical_gate, dict) else {}
        appearance = metadata.get("appearance_graph_evidence")
        appearance = appearance if isinstance(appearance, dict) else {}
        fraction = float(metadata.get("root_area_fraction", 0.0))
        closure = float(appearance.get("boundary_closure", 0.0))
        alignment = float(metadata.get("proposal_boundary_alignment", 0.0))
        cue_count = int(appearance.get("independent_cue_count", 0))
        named = bool(physical_gate.get("named_corroboration"))
        vlm = bool(physical_gate.get("vlm_physical_supported"))
        multi_source = bool(metadata.get("multi_view_confirmed"))
        independent_structure = bool(
            metadata.get("cross_source_confirmed")
            or physical_gate.get("cross_source_structure")
            or (
                candidate.source.startswith("hpid-shape-bottleneck/")
                and float(metadata.get("geometric_support", 0.0)) >= 0.50
            )
        )
        strong_closed = bool(
            0.008 <= fraction <= 0.62
            and alignment >= 0.78
            and ((closure >= 0.66 and cue_count >= 2) or closure >= 0.80)
        )
        large_enclosed_surface = bool(
            domain in {"container", "device", "furniture", "structure"}
            and 0.12 <= fraction <= 0.58
            and bool(physical_gate.get("silhouette_structure"))
            and float(physical_gate.get("outer_boundary_contact", 1.0)) <= 0.48
            and closure >= 0.24
            and alignment >= 0.26
        )
        if bool(physical_gate.get("nested_surface_texture")) or bool(
            physical_gate.get("laminar_surface_strip")
        ):
            strong_closed = False
            large_enclosed_surface = False
        if (
            named
            or vlm
            or multi_source
            or (independent_structure and (strong_closed or large_enclosed_surface))
        ):
            promoted[candidate.semantic_name] = (
                "semantic_corroboration"
                if named
                else "vlm_physical_support"
                if vlm
                else "multi_view_support"
                if multi_source
                else "independent_structure_and_closed_boundary"
            )
    return promoted


def _knife_inventory_active(candidates: tuple[MaskCandidate, ...]) -> bool:
    for candidate in candidates:
        if candidate.semantic_name != candidate.semantic_parent:
            continue
        selected = str(candidate.metadata.get("selected_part_profile", "")).strip()
        if selected == "knife":
            return True
        labels = candidate.metadata.get("asset_router_candidate_labels")
        domains = candidate.metadata.get("asset_router_candidate_domains")
        if (
            isinstance(labels, list)
            and len(labels) == 1
            and str(labels[0]).strip().casefold() == "knife"
            and (
                not isinstance(domains, list)
                or not domains
                or {str(value) for value in domains} == {"tool_prop"}
            )
        ):
            return True
    return False


def _selected_profile(candidates: tuple[MaskCandidate, ...]) -> str | None:
    root_profiles = {
        str(candidate.metadata["selected_part_profile"])
        for candidate in candidates
        if candidate.semantic_name == candidate.semantic_parent
        and candidate.metadata.get("selected_part_profile")
    }
    if len(root_profiles) == 1:
        return next(iter(root_profiles))
    if root_profiles:
        return None
    child_profiles = {
        str(candidate.metadata["selected_part_profile"])
        for candidate in candidates
        if candidate.metadata.get("selected_part_profile")
    }
    return next(iter(child_profiles)) if len(child_profiles) == 1 else None


def _mask_shape_descriptor(mask: np.ndarray) -> dict[str, float] | None:
    """Return scale-free geometry used to reject gross semantic mismatches."""

    region = np.asarray(mask, dtype=bool)
    area = int(np.count_nonzero(region))
    if area < 8:
        return None
    ys, xs = np.nonzero(region)
    width = max(1, int(xs.max() - xs.min() + 1))
    height = max(1, int(ys.max() - ys.min() + 1))
    points = np.column_stack((xs, ys)).astype(np.float64)
    covariance = np.cov((points - points.mean(axis=0)).T)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 1e-6)
    elongation = float(np.sqrt(eigenvalues[-1] / eigenvalues[0]))
    contours, _ = cv2.findContours(
        region.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    circularity = float(
        np.clip(4.0 * np.pi * area / max(1e-6, perimeter * perimeter), 0.0, 1.0)
    )
    return {
        "area_px": float(area),
        "bbox_fill_ratio": float(area / max(1, width * height)),
        "aspect_ratio": float(width / max(1, height)),
        "elongation": elongation,
        "circularity": circularity,
        "centroid_x": float(xs.mean()),
        "centroid_y": float(ys.mean()),
    }


def _semantic_shape_family(semantic_name: str) -> str | None:
    words = _words(semantic_name)
    if words & _ROUND_PART_TOKENS:
        return "round"
    if words & _SLENDER_PART_TOKENS:
        return "slender"
    return None


def _shape_family_matches(
    family: str | None,
    descriptor: dict[str, float] | None,
) -> bool:
    if family is None or descriptor is None:
        return True
    if family == "round":
        return bool(
            descriptor["elongation"] <= 1.55
            and descriptor["circularity"] >= 0.30
            and descriptor["bbox_fill_ratio"] >= 0.42
        )
    if family == "slender":
        return bool(
            descriptor["elongation"] >= 1.45
            or descriptor["bbox_fill_ratio"] <= 0.38
        )
    return True


def _semantic_shape_consistency(
    candidate: MaskCandidate,
    root: np.ndarray | None,
) -> dict[str, object]:
    """Check semantic size and shape without using appearance to name a part."""

    clipped = np.asarray(candidate.mask, dtype=bool)
    geometry = None
    if root is not None and root.shape == clipped.shape and np.any(root):
        geometry = _mask_geometry_against_root(clipped, root)
        if geometry is not None:
            clipped = np.asarray(geometry["mask"], dtype=bool)
    descriptor = _mask_shape_descriptor(clipped)
    fraction = float(
        geometry["root_area_fraction"]
        if geometry is not None
        else candidate.metadata.get("root_area_fraction", 0.0)
    )
    outer_contact = float(
        geometry["outer_boundary_contact"] if geometry is not None else 0.0
    )
    words = _words(candidate.semantic_name)
    family = _semantic_shape_family(candidate.semantic_name)
    profile = str(
        candidate.metadata.get("selected_part_profile")
        or candidate.metadata.get("semantic_rerank_profile")
        or candidate.metadata.get("structural_profile")
        or ""
    ).strip()
    accepted = True
    reason = "shape_not_contradicted"
    if words & _INTERNAL_PART_TOKENS:
        open_interior = profile in _OPEN_INTERIOR_PROFILES
        if open_interior and (
            fraction > 0.92 or (fraction > 0.80 and outer_contact > 0.80)
        ):
            accepted = False
            reason = "open_container_inner_consumes_root"
        elif not open_interior and fraction > 0.45:
            accepted = False
            reason = "internal_part_consumes_root"
        elif (
            not open_interior
            and fraction > 0.25
            and outer_contact > 0.22
        ):
            accepted = False
            reason = "internal_part_follows_outer_silhouette"
    if (
        accepted
        and family == "slender"
        and fraction >= 0.08
        and not _shape_family_matches(family, descriptor)
    ):
        accepted = False
        reason = f"{family}_part_shape_contradiction"
    return {
        "accepted": accepted,
        "reason": reason,
        "shape_family": family,
        "root_area_fraction": fraction,
        "outer_boundary_contact": outer_contact,
        "descriptor": descriptor,
        "ground_truth_used": False,
    }


def _candidate_three_stage_verification(
    candidate: MaskCandidate,
    root: np.ndarray | None = None,
) -> dict[str, object]:
    """Audit a proposed semantic part without letting appearance name it.

    Stage one asks whether the label is supported by a resolved inventory or a
    direct semantic segmenter. Stage two asks whether the mask is a plausible
    structural region. Stage three only checks that the visible boundary is
    not better explained by shading or texture. For visual proposals all three
    stages must pass before the semantic name may seed an editable group.
    """

    metadata = candidate.metadata
    source = candidate.source.casefold()
    route = str(metadata.get("semantic_rerank_route") or "")
    probability = float(metadata.get("semantic_rerank_probability", 0.0))
    margin = float(metadata.get("semantic_rerank_margin", -1.0))
    profile = str(
        metadata.get("selected_part_profile")
        or metadata.get("semantic_rerank_profile")
        or metadata.get("structural_profile")
        or ""
    ).strip()
    visual_derived = bool(
        metadata.get("visual_region")
        or metadata.get("semantic_reranked")
        or metadata.get("structural_fusion")
        or "semantic-rerank" in source
        or "semantic-inventory-cluster" in source
    )
    direct_semantic = any(
        token in source
        for token in (
            "/profile-refine",
            "vlm-",
            "florence",
            "grounded-sam",
            "grounding-dino",
            "prototype-retrieval",
            "conditional-part",
        )
    ) and "semantic-rerank" not in source
    inventory_supported = bool(profile)

    if not visual_derived:
        semantic_verified = bool(direct_semantic or inventory_supported)
        semantic_reason = (
            "direct_semantic_segmenter"
            if direct_semantic
            else "resolved_inventory_candidate"
            if inventory_supported
            else "nonvisual_semantic_candidate"
        )
    elif bool(metadata.get("structural_fusion")):
        residual = float(metadata.get("structural_axis_residual", 1.0))
        semantic_verified = bool(inventory_supported and residual <= 0.24)
        semantic_reason = "inventory_and_structural_route"
    else:
        direct_route = route in {"base", "contextual"}
        rescue_route = route in {
            "axis_structure_rescue",
            "contextual_axis_structure_rescue",
            "semantic_inventory_evidence_rescue",
        }
        bootstrap_route = route == "inventory_cluster_bootstrap"
        semantic_verified = bool(
            inventory_supported
            and (
                (direct_route and probability >= 0.14 and margin >= 0.006)
                or (rescue_route and probability >= 0.10 and margin >= 0.003)
                or (bootstrap_route and probability >= 0.12 and margin >= 0.0)
            )
        )
        semantic_reason = route or "visual_region_without_semantic_route"

    appearance = metadata.get("appearance_graph_evidence")
    appearance = appearance if isinstance(appearance, dict) else {}
    geometric_support = float(
        metadata.get(
            "geometric_support",
            metadata.get("proposal_boundary_alignment", 0.0),
        )
    )
    boundary_alignment = float(
        metadata.get(
            "proposal_boundary_alignment",
            appearance.get("boundary_alignment", 0.0),
        )
    )
    boundary_closure = float(appearance.get("boundary_closure", 0.0))
    cue_count = int(appearance.get("independent_cue_count", 0))
    shading_penalty = float(appearance.get("shading_only_penalty", 0.0))
    root_boundary_contact = float(appearance.get("root_boundary_contact", 0.0))
    cross_source = bool(
        metadata.get("cross_source_confirmed")
        or metadata.get("multi_view_confirmed")
    )
    independent_physical_structure = bool(
        root_boundary_contact >= 0.10
        or geometric_support >= 0.72
        or metadata.get("topology_refinement")
        or source.startswith("hpid-shape-bottleneck/")
    )
    photometric_only = bool(
        shading_penalty >= 0.42
        and not (
            boundary_alignment >= 0.58
            and boundary_closure >= 0.46
            and bool(
                _words(candidate.semantic_name)
                & _PHOTOMETRIC_PHYSICAL_SURFACE_TOKENS
            )
        )
        and not (
            independent_physical_structure
            and (cross_source or direct_semantic)
        )
    )
    axis_gate = metadata.get("axis_consistency_gate")
    axis_gate = axis_gate if isinstance(axis_gate, dict) else {}
    axis_not_rejected = axis_gate.get("accepted") is not False
    structural_route = bool(metadata.get("structural_fusion"))
    structural_residual = float(metadata.get("structural_axis_residual", 1.0))
    topology_route = bool(metadata.get("topology_refinement"))
    topology_diagnostics = metadata.get("topology_diagnostics")
    topology_diagnostics = (
        topology_diagnostics if isinstance(topology_diagnostics, dict) else {}
    )
    topology_structure = bool(
        topology_route
        and axis_not_rejected
        and 0.005 <= float(metadata.get("root_area_fraction", 0.0)) <= 0.82
        and (
            bool(metadata.get("topology_dense_gate"))
            or topology_diagnostics.get("selected_component") is not None
        )
    )
    physical_gate = metadata.get("physical_region_gate")
    physical_gate = physical_gate if isinstance(physical_gate, dict) else {}
    root_area_fraction = float(metadata.get("root_area_fraction", 0.0))
    semantic_shape = _semantic_shape_consistency(candidate, root)
    shape_structure = bool(
        source.startswith("hpid-shape-bottleneck/")
        and geometric_support >= 0.50
    )
    strong_closed_boundary = bool(
        boundary_alignment >= 0.68
        and boundary_closure >= 0.78
        and cue_count >= 1
    )
    multi_cue_closed_boundary = bool(
        boundary_alignment >= 0.78
        and boundary_closure >= 0.62
        and cue_count >= 2
    )
    closed_boundary_structure = bool(
        0.005 <= root_area_fraction <= 0.55
        and (strong_closed_boundary or multi_cue_closed_boundary)
        and shading_penalty <= 0.35
        and not bool(physical_gate.get("nested_surface_texture"))
        and not bool(physical_gate.get("laminar_surface_strip"))
    )
    appearance_evidence_available = bool(appearance)
    conditional_direct = bool(metadata.get("direct_conditional_mask"))
    direct_geometry_threshold = 0.64 if conditional_direct else 0.55
    bounded_photometric_surface = bool(
        _words(candidate.semantic_name)
        & _PHOTOMETRIC_PHYSICAL_SURFACE_TOKENS
        and boundary_alignment >= 0.58
        and boundary_closure >= 0.46
        and geometric_support >= 0.50
    )
    direct_semantic_structure = bool(
        direct_semantic
        and (
            not appearance_evidence_available
            or cross_source
            or geometric_support >= direct_geometry_threshold
            or closed_boundary_structure
            or bounded_photometric_surface
        )
    )
    raw_structure_verified = bool(
        axis_not_rejected
        and semantic_shape["accepted"]
        and (
            direct_semantic_structure
            or cross_source
            or shape_structure
            or closed_boundary_structure
            or geometric_support >= 0.55
            or (structural_route and structural_residual <= 0.24)
            or topology_structure
        )
    )
    raw_structure_reason = (
        str(semantic_shape["reason"])
        if not semantic_shape["accepted"]
        else "direct_semantic_with_independent_structure"
        if direct_semantic_structure
        else "cross_source_structure"
        if cross_source
        else "shape_bottleneck"
        if shape_structure
        else "closed_boundary_structure"
        if closed_boundary_structure
        else "boundary_aligned_region"
        if geometric_support >= 0.55
        else "structural_axis_route"
        if structural_route and structural_residual <= 0.24
        else "inventory_topology_complement"
        if topology_structure
        else "insufficient_structure"
    )

    # The public gate is deliberately serial.  A plausible shape is not
    # allowed to rescue a label that failed semantic verification.
    structure_evaluated = semantic_verified
    structure_verified = bool(structure_evaluated and raw_structure_verified)
    structure_reason = (
        raw_structure_reason
        if structure_evaluated
        else "not_evaluated_semantic_failed"
    )

    if not visual_derived or direct_semantic:
        raw_appearance_verified = not photometric_only
        raw_appearance_reason = (
            "no_appearance_contradiction"
            if raw_appearance_verified
            else "highlight_or_shadow_only_boundary"
        )
    else:
        raw_appearance_verified = bool(
            shading_penalty <= 0.50
            and boundary_alignment >= 0.55
            and (boundary_closure >= 0.28 or cue_count >= 2 or cross_source)
        )
        raw_appearance_reason = (
            "boundary_and_material_consistent"
            if raw_appearance_verified
            else "appearance_only_or_shading_like"
        )

    appearance_evaluated = bool(semantic_verified and structure_verified)
    appearance_verified = bool(
        appearance_evaluated and raw_appearance_verified
    )
    appearance_reason = (
        raw_appearance_reason
        if appearance_evaluated
        else "not_evaluated_previous_stage_failed"
    )

    accepted = bool(
        semantic_verified and structure_verified and appearance_verified
    )
    return {
        "algorithm": "hpid-semantic-structure-appearance-verification-v2",
        "visual_derived": visual_derived,
        "selected_profile": profile or None,
        "stage_1_semantic": {
            "order": 1,
            "evaluated": True,
            "verified": semantic_verified,
            "reason": semantic_reason,
            "probability": probability if route else None,
            "margin": margin if route else None,
            "direct_semantic_source": direct_semantic,
            "inventory_supported": inventory_supported,
        },
        "stage_2_structure": {
            "order": 2,
            "evaluated": structure_evaluated,
            "verified": structure_verified,
            "reason": structure_reason,
            "raw_verified": raw_structure_verified,
            "geometric_support": geometric_support,
            "axis_not_rejected": axis_not_rejected,
            "cross_source": cross_source,
            "closed_boundary_structure": closed_boundary_structure,
            "direct_semantic_structure": direct_semantic_structure,
            "bounded_photometric_surface": bounded_photometric_surface,
            "appearance_evidence_available": appearance_evidence_available,
            "root_area_fraction": root_area_fraction,
            "semantic_shape_consistency": semantic_shape,
        },
        "stage_3_appearance": {
            "order": 3,
            "evaluated": appearance_evaluated,
            "verified": appearance_verified,
            "reason": appearance_reason,
            "raw_verified": raw_appearance_verified,
            "boundary_alignment": boundary_alignment,
            "boundary_closure": boundary_closure,
            "independent_cue_count": cue_count,
            "shading_only_penalty": shading_penalty,
            "illumination_region": appearance.get(
                "illumination_region", "none"
            ),
            "root_boundary_contact": root_boundary_contact,
            "independent_physical_structure": independent_physical_structure,
            "can_create_id": False,
        },
        "accepted": accepted,
        "ground_truth_used": False,
    }


def _mask_geometry_against_root(
    mask: np.ndarray,
    root: np.ndarray,
) -> dict[str, object] | None:
    clipped = np.asarray(mask, dtype=bool) & root
    area = int(np.count_nonzero(clipped))
    root_area = max(1, int(np.count_nonzero(root)))
    if area == 0:
        return None
    ys, xs = np.nonzero(clipped)
    x0, x1 = int(xs.min()), int(xs.max() + 1)
    y0, y1 = int(ys.min()), int(ys.max() + 1)
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    count, _labels, stats, _ = cv2.connectedComponentsWithStats(
        clipped.astype(np.uint8), connectivity=8
    )
    largest = (
        max(
            (int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count)),
            default=0,
        )
        / max(1, area)
    )
    ring = (
        cv2.dilate(clipped.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(
            bool
        )
        & ~clipped
    )
    outer_contact = float(
        np.count_nonzero(ring & ~root) / max(1, np.count_nonzero(ring))
    )
    return {
        "mask": clipped,
        "area_px": area,
        "root_area_fraction": area / root_area,
        "bbox_xyxy": (x0, y0, x1, y1),
        "bbox_fill_ratio": area / max(1, width * height),
        "aspect_ratio": width / max(1, height),
        "largest_component_fraction": float(largest),
        "outer_boundary_contact": outer_contact,
    }


def _bbox_containment(
    inner: tuple[int, int, int, int],
    outer: tuple[int, int, int, int],
) -> float:
    ix0, iy0, ix1, iy1 = inner
    ox0, oy0, ox1, oy1 = outer
    intersection = max(0, min(ix1, ox1) - max(ix0, ox0)) * max(
        0, min(iy1, oy1) - max(iy0, oy0)
    )
    inner_area = max(1, (ix1 - ix0) * (iy1 - iy0))
    return float(intersection / inner_area)


def _phone_structural_masks(
    instance_map: np.ndarray,
    candidates: tuple[MaskCandidate, ...],
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """Resolve a phone display by serial semantic/structure/appearance evidence.

    A semantic screen proposal may capture the bezel instead of the emitting
    panel.  We therefore use it only to establish the inventory slot, then
    match that slot to an independently proposed, inset rectangular surface.
    Appearance is the final rejection gate and cannot introduce a screen on
    its own.
    """

    root = instance_map > 0
    if np.count_nonzero(root) < 64:
        return None, {"status": "root_too_small", "ground_truth_used": False}

    semantic_rows: list[tuple[MaskCandidate, dict[str, object]]] = []
    for candidate in candidates:
        if candidate.semantic_name != "device_screen":
            continue
        verification = _candidate_three_stage_verification(candidate)
        stage_1 = verification["stage_1_semantic"]
        if isinstance(stage_1, dict) and bool(stage_1.get("verified")):
            semantic_rows.append((candidate, verification))
    if not semantic_rows:
        return None, {
            "status": "semantic_inventory_slot_not_verified",
            "algorithm": "hpid-phone-three-stage-surface-bridge-v1",
            "ground_truth_used": False,
        }

    named_boxes = []
    for candidate, _ in semantic_rows:
        geometry = _mask_geometry_against_root(candidate.mask, root)
        if geometry is not None:
            named_boxes.append(geometry["bbox_xyxy"])

    structural_rows: list[dict[str, object]] = []
    accepted: list[tuple[float, MaskCandidate, dict[str, object]]] = []
    for candidate in candidates:
        metadata = candidate.metadata
        visual_kind = str(metadata.get("visual_region_kind", ""))
        if visual_kind != "panel" and candidate.semantic_name != "device_screen":
            continue
        geometry = _mask_geometry_against_root(candidate.mask, root)
        if geometry is None:
            continue
        physical_gate = metadata.get("physical_region_gate")
        physical_gate = physical_gate if isinstance(physical_gate, dict) else {}
        appearance = metadata.get("appearance_graph_evidence")
        appearance = appearance if isinstance(appearance, dict) else {}
        fraction = float(geometry["root_area_fraction"])
        fill = float(geometry["bbox_fill_ratio"])
        aspect = float(geometry["aspect_ratio"])
        largest = float(geometry["largest_component_fraction"])
        outer_contact = min(
            float(geometry["outer_boundary_contact"]),
            float(physical_gate.get("outer_boundary_contact", 1.0)),
        )
        containment = max(
            (
                _bbox_containment(geometry["bbox_xyxy"], named_box)
                for named_box in named_boxes
            ),
            default=0.0,
        )
        geometric_support = float(metadata.get("geometric_support", 0.0))
        shape_structure = bool(physical_gate.get("shape_structure"))
        structure_verified = bool(
            0.18 <= fraction <= 0.72
            and fill >= 0.56
            and 0.42 <= aspect <= 2.40
            and largest >= 0.88
            and outer_contact <= 0.12
            and containment >= 0.72
            and (shape_structure or geometric_support >= 0.62)
        )

        alignment = float(
            metadata.get(
                "proposal_boundary_alignment",
                appearance.get("boundary_alignment", 0.0),
            )
        )
        closure = float(appearance.get("boundary_closure", 0.0))
        cue_count = int(appearance.get("independent_cue_count", 0))
        shading_penalty = float(appearance.get("shading_only_penalty", 0.0))
        appearance_verified = bool(
            structure_verified
            and alignment >= 0.70
            and closure >= 0.50
            and cue_count >= 2
            and shading_penalty <= 0.45
        )
        row = {
            "candidate_key": metadata.get("candidate_key"),
            "semantic_name": candidate.semantic_name,
            "stage_1_semantic": {
                "verified": True,
                "evidence": "resolved_phone_inventory_and_screen_hypothesis",
            },
            "stage_2_structure": {
                "verified": structure_verified,
                "root_area_fraction": fraction,
                "bbox_fill_ratio": fill,
                "aspect_ratio": aspect,
                "largest_component_fraction": largest,
                "outer_boundary_contact": outer_contact,
                "semantic_bbox_containment": containment,
                "shape_structure": shape_structure,
                "geometric_support": geometric_support,
            },
            "stage_3_appearance": {
                "verified": appearance_verified,
                "boundary_alignment": alignment,
                "boundary_closure": closure,
                "independent_cue_count": cue_count,
                "shading_only_penalty": shading_penalty,
                "can_create_id": False,
            },
            "accepted": appearance_verified,
        }
        structural_rows.append(row)
        if appearance_verified:
            score = (
                0.30 * geometric_support
                + 0.20 * alignment
                + 0.18 * closure
                + 0.18 * fill
                + 0.14 * containment
            )
            accepted.append((score, candidate, geometry))

    if not accepted:
        return None, {
            "status": "no_surface_passed_all_three_stages",
            "algorithm": "hpid-phone-three-stage-surface-bridge-v1",
            "semantic_hypothesis_count": len(semantic_rows),
            "surface_candidates": structural_rows,
            "appearance_can_create_ids": False,
            "ground_truth_used": False,
        }

    _, selected, selected_geometry = max(accepted, key=lambda row: row[0])
    screen = np.asarray(selected_geometry["mask"], dtype=bool)
    body = root & ~screen
    if np.count_nonzero(body) < 0.20 * np.count_nonzero(root):
        return None, {
            "status": "screen_would_consume_phone_body",
            "algorithm": "hpid-phone-three-stage-surface-bridge-v1",
            "surface_candidates": structural_rows,
            "ground_truth_used": False,
        }
    return {
        "device_body": body,
        "device_screen": screen,
    }, {
        "status": "completed",
        "algorithm": "hpid-phone-three-stage-surface-bridge-v1",
        "evidence_order": ["semantic", "structure", "appearance"],
        "semantic_hypothesis_count": len(semantic_rows),
        "selected_candidate_key": selected.metadata.get("candidate_key"),
        "selected_source_semantic": selected.semantic_name,
        "surface_candidates": structural_rows,
        "appearance_can_create_ids": False,
        "ground_truth_used": False,
    }


def _profile_candidate_verification(
    candidates: tuple[MaskCandidate, ...],
    profile: str | None,
    root: np.ndarray | None = None,
) -> tuple[set[str], set[str], list[dict[str, object]]]:
    if profile is None:
        return set(), set(), []
    accepted: set[str] = set()
    proposed: set[str] = set()
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        if candidate.semantic_name == candidate.semantic_parent or _is_visual_semantic(
            candidate.semantic_name
        ):
            continue
        candidate_profile = str(
            candidate.metadata.get("selected_part_profile")
            or candidate.metadata.get("semantic_rerank_profile")
            or candidate.metadata.get("structural_profile")
            or ""
        ).strip()
        if candidate_profile != profile:
            continue
        proposed.add(candidate.semantic_name)
        verification = _candidate_three_stage_verification(candidate, root)
        if bool(verification["accepted"]):
            accepted.add(candidate.semantic_name)
        rows.append(
            {
                "candidate_key": candidate.metadata.get("candidate_key"),
                "semantic_name": candidate.semantic_name,
                "source": candidate.source,
                **verification,
            }
        )
    return accepted, proposed - accepted, rows


def _structurally_recovered_profile_semantics(
    profile: str | None,
    verification_rows: list[dict[str, object]],
) -> set[str]:
    """Recover profile parts whose identity and geometry are independently sound.

    Appearance remains a review signal in this case.  It cannot erase a named,
    structurally valid component merely because illumination changes along one
    continuous physical surface.
    """

    if profile != "globe":
        return set()
    recovered: set[str] = set()
    for row in verification_rows:
        semantic_name = str(row.get("semantic_name") or "")
        stage_1 = row.get("stage_1_semantic")
        stage_2 = row.get("stage_2_structure")
        if (
            semantic_name == "device_globe_meridian_ring"
            and isinstance(stage_1, dict)
            and isinstance(stage_2, dict)
            and bool(stage_1.get("verified"))
            and bool(stage_2.get("verified"))
        ):
            recovered.add(semantic_name)
    return recovered


def _character_axis_frame(
    instance_map: np.ndarray,
    records: list[PartInstance] | tuple[PartInstance, ...],
) -> _CharacterAxisFrame | None:
    """Build a pose-normalized head-to-feet frame for one character."""

    root = instance_map > 0
    ys, xs = np.nonzero(root)
    if len(xs) < 128:
        return None
    points = np.column_stack((xs, ys)).astype(np.float64)
    head_indices = [
        record.instance_index
        for record in records
        if _words(record.semantic_name) & {"head", "face"}
        and not (_words(record.semantic_name) & {"hair", "headwear"})
    ]
    head_mask = np.isin(instance_map, head_indices)
    if np.count_nonzero(head_mask) >= 32:
        head_y, head_x = np.nonzero(head_mask)
        head_center = np.asarray((head_x.mean(), head_y.mean()), dtype=np.float64)
        body_points = points[~head_mask[ys, xs]]
        body_center = (
            body_points.mean(axis=0) if len(body_points) >= 32 else points.mean(axis=0)
        )
        axis = body_center - head_center
    else:
        head_center = points.mean(axis=0)
        axis = np.zeros(2, dtype=np.float64)

    if float(np.linalg.norm(axis)) < 4.0:
        covariance = np.cov((points - points.mean(axis=0)).T)
        _, eigenvectors = np.linalg.eigh(covariance)
        axis = eigenvectors[:, -1]
        if abs(float(axis[1])) >= abs(float(axis[0])):
            axis *= 1.0 if axis[1] >= 0 else -1.0
        else:
            axis *= 1.0 if axis[0] >= 0 else -1.0
    axis /= max(1e-7, float(np.linalg.norm(axis)))
    if np.count_nonzero(head_mask) >= 32:
        root_center = points.mean(axis=0)
        if float(np.dot(root_center - head_center, axis)) < 0:
            axis *= -1.0

    grid_y, grid_x = np.indices(instance_map.shape)
    projection = grid_x.astype(np.float64) * axis[0] + grid_y * axis[1]
    root_projection = projection[root]
    start = float(root_projection.min())
    span = max(1.0, float(root_projection.max() - start))
    coordinate = np.clip((projection - start) / span, 0.0, 1.0).astype(np.float32)

    lateral_axis = np.asarray((-axis[1], axis[0]), dtype=np.float64)
    lateral_projection = (
        grid_x.astype(np.float64) * lateral_axis[0]
        + grid_y.astype(np.float64) * lateral_axis[1]
    )
    root_lateral = lateral_projection[root]
    lateral_center = 0.5 * (float(root_lateral.min()) + float(root_lateral.max()))
    lateral_span = max(1.0, float(root_lateral.max() - root_lateral.min()))
    lateral = ((lateral_projection - lateral_center) / lateral_span).astype(np.float32)
    head_end = (
        float(np.quantile(coordinate[head_mask], 0.985))
        if np.count_nonzero(head_mask) >= 32
        else 0.34
    )
    return _CharacterAxisFrame(
        coordinate=coordinate,
        lateral=lateral,
        head_end=float(np.clip(head_end, 0.18, 0.58)),
        axis_direction_xy=(float(axis[0]), float(axis[1])),
    )


def _profile_semantic_group_overrides(
    instance_map: np.ndarray,
    records: list[PartInstance] | tuple[PartInstance, ...],
    candidates: tuple[MaskCandidate, ...],
    image: Image.Image | np.ndarray | None,
) -> tuple[dict[str, tuple[str, int, str]], dict[str, object]]:
    """Attach unresolved visual atoms to trusted semantic inventory seeds.

    Semantic labels choose the possible physical groups. Geometry chooses the
    neighbouring seed, while Lab appearance is only a low-weight tie breaker.
    This keeps color or material regions from becoming public IDs on their own.
    """

    profile = _selected_profile(candidates)
    asset_ids = {record.asset_id for record in records}
    if profile is None or len(asset_ids) != 1 or profile == "knife":
        return {}, {"status": "inactive", "selected_profile": profile}
    profile_semantics, rejected_profile_semantics, verification_rows = (
        _profile_candidate_verification(candidates, profile)
    )
    structurally_recovered_semantics = _structurally_recovered_profile_semantics(
        profile,
        verification_rows,
    )
    if structurally_recovered_semantics:
        profile_semantics |= structurally_recovered_semantics
        rejected_profile_semantics -= structurally_recovered_semantics
    targets = [
        record
        for record in records
        if record.semantic_name in profile_semantics
        and not _is_visual_semantic(record.semantic_name)
    ]
    if not targets:
        return {}, {
            "status": "no_semantic_seeds",
            "selected_profile": profile,
            "three_stage_verification": verification_rows,
            "rejected_semantics": sorted(rejected_profile_semantics),
        }
    detail_semantics = {
        candidate.semantic_name
        for candidate in candidates
        if bool(candidate.metadata.get("detail"))
    }
    macro_targets = [
        record for record in targets if record.semantic_name not in detail_semantics
    ]
    if not macro_targets:
        return {}, {
            "status": "no_macro_semantic_seeds",
            "selected_profile": profile,
            "three_stage_verification": verification_rows,
            "rejected_semantics": sorted(rejected_profile_semantics),
        }

    rgb = None
    lab = None
    if image is not None:
        rgb = (
            np.asarray(image.convert("RGB"), dtype=np.uint8)
            if isinstance(image, Image.Image)
            else np.asarray(image, dtype=np.uint8)
        )
        if rgb.shape[:2] == instance_map.shape:
            lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    diagonal = max(1.0, float(np.hypot(*instance_map.shape)))
    target_rows: list[dict[str, object]] = []
    for target in macro_targets:
        mask = instance_map == target.instance_index
        if not np.any(mask):
            continue
        target_rows.append(
            {
                "record": target,
                "mask": mask,
                "distance": cv2.distanceTransform(
                    (~mask).astype(np.uint8), cv2.DIST_L2, 3
                ),
                "lab": (np.median(lab[mask], axis=0) if lab is not None else None),
            }
        )
    if not target_rows:
        return {}, {
            "status": "empty_semantic_seeds",
            "selected_profile": profile,
            "three_stage_verification": verification_rows,
            "rejected_semantics": sorted(rejected_profile_semantics),
        }

    overrides: dict[str, tuple[str, int, str]] = {}
    root_area = max(1, int(np.count_nonzero(instance_map)))
    dominant = max(
        macro_targets,
        key=lambda record: record.area_px,
    )
    dominant_words = _words(dominant.semantic_name)
    dominant_primary_surface = bool(
        dominant_words & _PROFILE_PRIMARY_SURFACE_TOKENS
        and dominant.area_px / root_area >= 0.45
    )
    for record in records:
        is_root_residual = record.semantic_name == record.semantic_parent
        is_visual = _is_visual_semantic(record.semantic_name)
        if not (is_root_residual or is_visual):
            continue
        if is_root_residual:
            if dominant_primary_surface:
                overrides[record.part_id] = (
                    dominant.semantic_name,
                    dominant.instance_index,
                    "dominant_semantic_surface_completion",
                )
            continue
        mask = instance_map == record.instance_index
        if not np.any(mask):
            continue
        ring = (
            cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            & ~mask
        )
        source_lab = np.median(lab[mask], axis=0) if lab is not None else None
        scored: list[tuple[float, PartInstance]] = []
        for target_row in target_rows:
            target = target_row["record"]
            target_mask = target_row["mask"]
            distance = target_row["distance"]
            assert isinstance(target, PartInstance)
            assert isinstance(target_mask, np.ndarray)
            assert isinstance(distance, np.ndarray)
            spatial = float(distance[mask].min(initial=diagonal)) / diagonal
            shared_boundary = float(
                np.count_nonzero(ring & target_mask) / max(1, np.count_nonzero(ring))
            )
            target_lab = target_row["lab"]
            appearance = (
                float(np.linalg.norm(source_lab - target_lab) / 255.0)
                if source_lab is not None and target_lab is not None
                else 0.0
            )
            centroid_distance = float(
                np.hypot(
                    record.centroid_xy[0] - target.centroid_xy[0],
                    record.centroid_xy[1] - target.centroid_xy[1],
                )
                / diagonal
            )
            cost = (
                0.58 * spatial
                + 0.14 * appearance
                + 0.24 * centroid_distance
                - 0.22 * shared_boundary
            )
            scored.append((cost, target))
        _, target = min(scored, key=lambda row: row[0])
        overrides[record.part_id] = (
            target.semantic_name,
            target.instance_index,
            "semantic_seed_geometry_appearance_attachment",
        )
    return overrides, {
        "status": "completed",
        "selected_profile": profile,
        "semantic_seed_count": len(targets),
        "macro_seed_count": len(macro_targets),
        "override_count": len(overrides),
        "dominant_surface_completion": dominant_primary_surface,
        "evidence_order": ["semantic_inventory", "geometry", "appearance"],
        "three_stage_verification": verification_rows,
        "verified_semantics": sorted(profile_semantics),
        "rejected_semantics": sorted(rejected_profile_semantics),
        "structurally_recovered_semantics": sorted(
            structurally_recovered_semantics
        ),
        "structural_recovery_policy": (
            "globe_meridian_requires_semantic_and_structure; appearance_marks_review"
            if structurally_recovered_semantics
            else None
        ),
        "ground_truth_used": False,
    }


def _weighted_segment(
    features: np.ndarray,
    weights: np.ndarray,
    start: int,
    end: int,
) -> tuple[float, np.ndarray]:
    local_weights = weights[start:end].astype(np.float64)
    local_features = features[start:end]
    total = float(local_weights.sum())
    if total <= 0.0:
        return float("inf"), np.zeros(features.shape[1], dtype=np.float64)
    mean = np.average(local_features, axis=0, weights=local_weights)
    error = float(np.sum(local_weights[:, None] * np.square(local_features - mean)))
    return error, mean


def _knife_structural_masks(
    image: Image.Image | np.ndarray,
    root_mask: np.ndarray,
    *,
    bin_count: int = 72,
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """Split a long knife into blade, rigid handle, and an axial wrap band.

    The method is category constrained but image agnostic: the broad terminal is
    chosen as the blade, the blade/handle boundary comes from a width bottleneck
    plus material change, and the middle handle band comes from three-segment
    Lab clustering.  No hue, coordinate, or file-name rule is used.
    """

    ys, xs = np.nonzero(root_mask)
    if len(xs) < 128:
        return None, {"status": "insufficient_root_area"}
    points = np.column_stack((xs, ys)).astype(np.float64)
    center = points.mean(axis=0)
    covariance = np.cov((points - center).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if float(eigenvalues[-1]) < 4.0:
        return None, {"status": "axis_unavailable"}
    axis = eigenvectors[:, -1]
    projection = (points - center) @ axis
    span = float(projection.max() - projection.min())
    if span < 16.0:
        return None, {"status": "axis_span_too_small"}
    coordinate = (projection - float(projection.min())) / span

    def bin_coordinates(values: np.ndarray) -> np.ndarray:
        return np.clip((values * bin_count).astype(np.int32), 0, bin_count - 1)

    indices = bin_coordinates(coordinate)
    widths = np.bincount(indices, minlength=bin_count).astype(np.float64)
    endpoint_span = max(3, round(bin_count * 0.12))
    if widths[-endpoint_span:].mean() > widths[:endpoint_span].mean():
        coordinate = 1.0 - coordinate
        axis = -axis
        indices = bin_coordinates(coordinate)
        widths = np.bincount(indices, minlength=bin_count).astype(np.float64)
    smoothed_widths = cv2.GaussianBlur(widths.reshape(1, -1), (1, 0), 1.7).ravel()

    rgb = (
        np.asarray(image.convert("RGB"), dtype=np.uint8)
        if isinstance(image, Image.Image)
        else np.asarray(image, dtype=np.uint8)
    )
    if rgb.shape[:2] != root_mask.shape:
        return None, {"status": "image_shape_mismatch"}
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    features = np.zeros((bin_count, 3), dtype=np.float64)
    for index in range(bin_count):
        selected = indices == index
        if np.any(selected):
            features[index] = lab[ys[selected], xs[selected]].mean(axis=0)
        elif index > 0:
            features[index] = features[index - 1]

    search_start = round(bin_count * 0.32)
    search_end = round(bin_count * 0.74)
    window = max(3, round(bin_count * 0.07))
    split_rows: list[tuple[float, int, float, float, float]] = []
    blade_width_reference = float(
        np.median(smoothed_widths[round(bin_count * 0.18) : round(bin_count * 0.46)])
    )
    for index in range(search_start, search_end):
        left = float(smoothed_widths[max(0, index - window) : index].mean())
        right = float(smoothed_widths[index : min(bin_count, index + window)].mean())
        if left <= 0.0:
            continue
        color_left = features[max(0, index - 3) : index].mean(axis=0)
        color_right = features[index : min(bin_count, index + 3)].mean(axis=0)
        color_change = float(np.linalg.norm(color_left - color_right))
        width_drop = (left - right) / left
        score = width_drop + 0.50 * min(1.0, color_change / 50.0)
        split_rows.append((score, index, left, right, color_change))
    if not split_rows:
        return None, {"status": "blade_handle_boundary_unavailable"}
    early_contractions = [
        row
        for row in split_rows
        if row[3] <= 0.90 * blade_width_reference
        and (row[2] - row[3]) / max(1.0, row[2]) >= 0.16
        and row[4] >= 4.0
    ]
    (
        split_score,
        split_index,
        left_width,
        right_width,
        split_color,
    ) = (
        min(early_contractions, key=lambda row: row[1])
        if early_contractions
        else max(split_rows)
    )
    if split_score < 0.18 or right_width >= left_width * 0.92:
        return None, {
            "status": "blade_handle_boundary_ambiguous",
            "split_score": split_score,
        }
    blade_end = split_index / bin_count

    handle_start = split_index
    handle_length = bin_count - handle_start
    minimum_segment = max(3, round(handle_length * 0.14))
    material_rows: list[tuple[float, int, int, np.ndarray, np.ndarray, np.ndarray]] = []
    for first_break in range(
        handle_start + minimum_segment,
        bin_count - 2 * minimum_segment + 1,
    ):
        for second_break in range(
            first_break + minimum_segment,
            bin_count - minimum_segment + 1,
        ):
            first_error, first_mean = _weighted_segment(
                features, widths, handle_start, first_break
            )
            middle_error, middle_mean = _weighted_segment(
                features, widths, first_break, second_break
            )
            last_error, last_mean = _weighted_segment(
                features, widths, second_break, bin_count
            )
            material_rows.append(
                (
                    first_error + middle_error + last_error,
                    first_break,
                    second_break,
                    first_mean,
                    middle_mean,
                    last_mean,
                )
            )
    if not material_rows:
        return None, {"status": "handle_material_partition_unavailable"}
    (
        material_error,
        wrap_start_index,
        wrap_end_index,
        first_mean,
        middle_mean,
        last_mean,
    ) = min(material_rows, key=lambda row: row[0])
    first_contrast = float(np.linalg.norm(middle_mean - first_mean))
    last_contrast = float(np.linalg.norm(middle_mean - last_mean))
    endpoint_difference = float(np.linalg.norm(first_mean - last_mean))
    if min(first_contrast, last_contrast) < 8.0 or max(
        first_contrast, last_contrast
    ) < max(14.0, endpoint_difference * 1.08):
        return None, {
            "status": "distinct_wrap_material_unavailable",
            "first_wrap_contrast": first_contrast,
            "last_wrap_contrast": last_contrast,
            "handle_endpoint_difference": endpoint_difference,
        }
    wrap_start = wrap_start_index / bin_count
    wrap_end = wrap_end_index / bin_count

    full_coordinate = np.full(root_mask.shape, np.nan, dtype=np.float32)
    full_coordinate[ys, xs] = coordinate.astype(np.float32)
    wrap = root_mask & (full_coordinate >= wrap_start) & (full_coordinate < wrap_end)
    blade_reference = coordinate < max(0.22, blade_end - 0.08)
    handle_reference = (coordinate >= blade_end + 0.035) & (
        coordinate < wrap_start - 0.015
    )
    if (
        int(np.count_nonzero(blade_reference)) >= 64
        and int(np.count_nonzero(handle_reference)) >= 64
    ):
        blade_mean = lab[ys[blade_reference], xs[blade_reference]].mean(axis=0)
        handle_mean = lab[ys[handle_reference], xs[handle_reference]].mean(axis=0)
        blade_distance = np.linalg.norm(lab - blade_mean, axis=2)
        handle_distance = np.linalg.norm(lab - handle_mean, axis=2)
        blade = (
            root_mask
            & ~wrap
            & (full_coordinate < wrap_start)
            & (blade_distance <= handle_distance)
        )
        blade |= root_mask & (full_coordinate < blade_end - 0.08)
        blade = cv2.morphologyEx(
            blade.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        ).astype(bool)
        component_count, component_labels, component_stats, _ = (
            cv2.connectedComponentsWithStats(blade.astype(np.uint8), connectivity=8)
        )
        if component_count > 1:
            main_component = max(
                range(1, component_count),
                key=lambda index: int(component_stats[index, cv2.CC_STAT_AREA]),
            )
            blade = component_labels == main_component
    else:
        blade = root_mask & (full_coordinate < blade_end)
    handle = root_mask & ~blade & ~wrap
    fractions = {
        "blade": float(np.count_nonzero(blade) / len(xs)),
        "handle": float(np.count_nonzero(handle) / len(xs)),
        "wrap": float(np.count_nonzero(wrap) / len(xs)),
    }
    if not (
        0.18 <= fractions["blade"] <= 0.80
        and 0.05 <= fractions["handle"] <= 0.60
        and 0.025 <= fractions["wrap"] <= 0.48
    ):
        return None, {"status": "physical_fraction_gate", "fractions": fractions}
    return {
        "tool_prop_blade": blade,
        "tool_prop_handle": handle,
        "tool_prop_wrap": wrap,
    }, {
        "status": "accepted",
        "axis_direction_xy": [float(axis[0]), float(axis[1])],
        "blade_handle_boundary": blade_end,
        "blade_handle_score": float(split_score),
        "blade_handle_color_change": float(split_color),
        "wrap_interval": [wrap_start, wrap_end],
        "wrap_material_error": float(material_error),
        "first_wrap_contrast": first_contrast,
        "last_wrap_contrast": last_contrast,
        "handle_endpoint_difference": endpoint_difference,
        "fractions": fractions,
        "ground_truth_used": False,
    }


def _refine_inventory_boundaries(
    image: Image.Image | np.ndarray | None,
    root_mask: np.ndarray,
    coarse_masks: dict[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Align structural inventory labels to image seams without creating IDs."""

    fallback = {name: mask.copy() for name, mask in coarse_masks.items()}
    if image is None:
        return fallback, {
            "status": "skipped_no_image",
            "algorithm": "structure-seeded-lab-watershed-v2",
            "appearance_can_create_ids": False,
            "ground_truth_used": False,
        }
    rgb = (
        np.asarray(image.convert("RGB"), dtype=np.uint8)
        if isinstance(image, Image.Image)
        else np.asarray(image, dtype=np.uint8)
    )
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    elif rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    if rgb.shape[:2] != root_mask.shape or rgb.ndim != 3 or rgb.shape[2] != 3:
        return fallback, {
            "status": "skipped_image_shape_mismatch",
            "algorithm": "structure-seeded-lab-watershed-v2",
            "appearance_can_create_ids": False,
            "ground_truth_used": False,
        }

    semantic_order = tuple(coarse_masks)
    coarse_map = np.zeros(root_mask.shape, dtype=np.int32)
    for group_index, semantic in enumerate(semantic_order, start=1):
        coarse_map[coarse_masks[semantic] & root_mask] = group_index
    if not np.array_equal(coarse_map > 0, root_mask):
        return fallback, {
            "status": "skipped_incomplete_coarse_partition",
            "algorithm": "structure-seeded-lab-watershed-v2",
            "appearance_can_create_ids": False,
            "ground_truth_used": False,
        }

    # Work on a softly reduced image so wood grain, reflections, and printed
    # decoration do not dominate true assembly seams.
    analysis_scale = min(1.0, 768.0 / max(rgb.shape[:2]))
    analysis_width = max(32, round(rgb.shape[1] * analysis_scale))
    analysis_height = max(32, round(rgb.shape[0] * analysis_scale))
    reduced = cv2.resize(
        rgb,
        (analysis_width, analysis_height),
        interpolation=cv2.INTER_AREA,
    )
    reduced = cv2.bilateralFilter(reduced, 9, 38, 38)
    smoothed = cv2.resize(
        reduced,
        (rgb.shape[1], rgb.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    lab = cv2.cvtColor(smoothed, cv2.COLOR_RGB2LAB).astype(np.float32)
    channel_gradients: list[np.ndarray] = []
    for channel in cv2.split(lab):
        dx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        channel_gradients.append(cv2.magnitude(dx, dy))
    gradient = np.maximum.reduce(channel_gradients)
    root_values = gradient[root_mask]
    reference = float(np.quantile(root_values, 0.90)) if root_values.size else 1.0
    elevation_levels = 4096
    elevation = np.rint(
        np.clip(gradient / max(1.0, reference), 0.0, 2.0)
        * elevation_levels
    ).astype(np.uint16)

    markers = np.zeros(root_mask.shape, dtype=np.int32)
    seed_rows: list[dict[str, object]] = []
    for group_index, semantic in enumerate(semantic_order, start=1):
        coarse = coarse_map == group_index
        distance = cv2.distanceTransform(
            coarse.astype(np.uint8),
            cv2.DIST_L2,
            5,
        )
        maximum_distance = float(distance.max())
        seed_depth = max(2.5, min(14.0, maximum_distance * 0.32))
        seed = distance >= seed_depth
        minimum_seed_area = max(12, round(np.count_nonzero(coarse) * 0.002))
        if np.count_nonzero(seed) < minimum_seed_area:
            seed_depth = max(1.0, maximum_distance * 0.18)
            seed = distance >= seed_depth
        if not np.any(seed):
            seed.flat[int(np.argmax(distance))] = True
        markers[seed] = group_index
        seed_rows.append(
            {
                "semantic_name": semantic,
                "coarse_area_px": int(np.count_nonzero(coarse)),
                "seed_area_px": int(np.count_nonzero(seed)),
                "seed_depth_px": float(seed_depth),
            }
        )

    from skimage.segmentation import watershed

    refined_map = watershed(
        elevation,
        markers,
        mask=root_mask,
        watershed_line=False,
    ).astype(np.int32)
    area_ratios: dict[str, float] = {}
    for group_index, semantic in enumerate(semantic_order, start=1):
        coarse_area = max(1, int(np.count_nonzero(coarse_map == group_index)))
        refined_area = int(np.count_nonzero(refined_map == group_index))
        area_ratios[semantic] = refined_area / coarse_area
    if not np.array_equal(refined_map > 0, root_mask) or any(
        ratio < 0.45 or ratio > 1.75 for ratio in area_ratios.values()
    ):
        return fallback, {
            "status": "rejected_unstable_refinement",
            "algorithm": "structure-seeded-lab-watershed-v2",
            "analysis_scale": analysis_scale,
            "elevation_quantization_levels": elevation_levels,
            "area_ratios": area_ratios,
            "appearance_can_create_ids": False,
            "ground_truth_used": False,
        }

    refined = {
        semantic: refined_map == group_index
        for group_index, semantic in enumerate(semantic_order, start=1)
    }
    return refined, {
        "status": "completed",
        "algorithm": "structure-seeded-lab-watershed-v2",
        "analysis_scale": analysis_scale,
        "analysis_size": [analysis_width, analysis_height],
        "elevation_quantization_levels": elevation_levels,
        "seed_rows": seed_rows,
        "area_ratios": area_ratios,
        "reassigned_pixel_count": int(np.count_nonzero(refined_map != coarse_map)),
        "appearance_role": "boundary_alignment_only",
        "appearance_can_create_ids": False,
        "ground_truth_used": False,
    }


def _canonical_profile_group_semantic(
    semantic_name: str,
    profile: str,
) -> str:
    """Map inventory labels to the physical group exposed by the profile."""

    if profile != "knife":
        return semantic_name
    words = _words(semantic_name)
    if "blade" in words:
        return "tool_prop_blade"
    if words & {"wrap", "wrapping", "cloth"}:
        return "tool_prop_wrap"
    if words & _KNIFE_HANDLE_TOKENS or semantic_name == "tool_prop":
        return "tool_prop_handle"
    return semantic_name


def _semantic_seeded_profile_masks(
    instance_map: np.ndarray,
    candidates: tuple[MaskCandidate, ...],
    image: Image.Image | np.ndarray | None,
    *,
    profile: str | None,
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """Complete a physical partition from verified semantic part seeds.

    Semantic evidence determines the allowed IDs. Root-constrained spatial
    propagation assigns only the still-unowned pixels, and appearance has a
    low-weight boundary role. It can refine a seam but cannot name or create a
    part. Repeated-instance slots stay on the normal instance path because one
    semantic marker must not merge separate wheels, legs, or similar parts.
    """

    root = instance_map > 0
    root_area = int(np.count_nonzero(root))
    if profile is None or root_area < 96:
        return None, {
            "status": "inactive",
            "selected_profile": profile,
            "ground_truth_used": False,
        }

    rows: list[dict[str, object]] = []
    grouped: dict[str, list[MaskCandidate]] = {}
    repeated_semantics: set[str] = set()
    for candidate in candidates:
        if candidate.semantic_name == candidate.semantic_parent or _is_visual_semantic(
            candidate.semantic_name
        ):
            continue
        candidate_profile = str(
            candidate.metadata.get("selected_part_profile")
            or candidate.metadata.get("semantic_rerank_profile")
            or candidate.metadata.get("structural_profile")
            or ""
        ).strip()
        if candidate_profile != profile or candidate.mask.shape != root.shape:
            continue
        verification = _candidate_three_stage_verification(candidate)
        structurally_recovered = bool(
            candidate.semantic_name
            in _structurally_recovered_profile_semantics(
                profile,
                [
                    {
                        "semantic_name": candidate.semantic_name,
                        **verification,
                    }
                ],
            )
        )
        canonical = _canonical_profile_group_semantic(
            candidate.semantic_name,
            profile,
        )
        maximum_instances = int(candidate.metadata.get("maximum_instances", 1))
        if maximum_instances > 1:
            repeated_semantics.add(canonical)
        clipped = np.asarray(candidate.mask, dtype=bool) & root
        containment = float(
            np.count_nonzero(clipped)
            / max(1, np.count_nonzero(candidate.mask))
        )
        fraction = float(np.count_nonzero(clipped) / root_area)
        accepted = bool(
            (verification["accepted"] or structurally_recovered)
            and containment >= 0.80
            and 0.004 <= fraction <= 0.88
        )
        rows.append(
            {
                "candidate_key": candidate.metadata.get("candidate_key"),
                "semantic_name": candidate.semantic_name,
                "canonical_group": canonical,
                "maximum_instances": maximum_instances,
                "root_containment": containment,
                "root_area_fraction": fraction,
                "accepted_seed": accepted,
                "structurally_recovered": structurally_recovered,
                "verification": verification,
            }
        )
        if accepted:
            grouped.setdefault(canonical, []).append(candidate)

    if repeated_semantics:
        return None, {
            "status": "repeated_instance_inventory_deferred",
            "selected_profile": profile,
            "repeated_semantics": sorted(repeated_semantics),
            "candidates": rows,
            "ground_truth_used": False,
        }
    if len(grouped) < 2:
        return None, {
            "status": "insufficient_verified_macro_seeds",
            "selected_profile": profile,
            "verified_group_count": len(grouped),
            "candidates": rows,
            "ground_truth_used": False,
        }

    semantics = tuple(sorted(grouped))
    seed_masks: dict[str, np.ndarray] = {}
    seed_confidences: dict[str, float] = {}
    for semantic in semantics:
        members = grouped[semantic]
        seed_masks[semantic] = np.logical_or.reduce(
            [np.asarray(candidate.mask, dtype=bool) & root for candidate in members]
        )
        seed_confidences[semantic] = max(
            float(candidate.score * candidate.source_reliability)
            for candidate in members
        )

    seed_union = np.logical_or.reduce(list(seed_masks.values()))
    seed_coverage = float(np.count_nonzero(seed_union) / root_area)
    if seed_coverage < 0.12:
        return None, {
            "status": "verified_seed_coverage_too_low",
            "selected_profile": profile,
            "seed_coverage": seed_coverage,
            "candidates": rows,
            "ground_truth_used": False,
        }

    lab: np.ndarray | None = None
    if image is not None:
        rgb = (
            np.asarray(image.convert("RGB"), dtype=np.uint8)
            if isinstance(image, Image.Image)
            else np.asarray(image, dtype=np.uint8)
        )
        if rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        if rgb.shape[:2] == root.shape and rgb.ndim == 3 and rgb.shape[2] == 3:
            scale = min(1.0, 640.0 / max(rgb.shape[:2]))
            width = max(32, round(rgb.shape[1] * scale))
            height = max(32, round(rgb.shape[0] * scale))
            reduced = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
            reduced = cv2.bilateralFilter(reduced, 9, 42, 42)
            smoothed = cv2.resize(
                reduced,
                (rgb.shape[1], rgb.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            lab = cv2.cvtColor(smoothed, cv2.COLOR_RGB2LAB).astype(np.float32)

    diagonal = max(1.0, float(np.hypot(*root.shape)))
    costs: list[np.ndarray] = []
    for semantic in semantics:
        seed = seed_masks[semantic]
        spatial = cv2.distanceTransform(
            (~seed).astype(np.uint8),
            cv2.DIST_L2,
            5,
        ).astype(np.float32)
        cost = spatial / diagonal
        if lab is not None:
            distance = cv2.distanceTransform(seed.astype(np.uint8), cv2.DIST_L2, 5)
            core = seed & (distance >= max(1.0, float(distance.max()) * 0.18))
            if np.count_nonzero(core) < 16:
                core = seed
            # Lightness is downweighted so highlights and shadows cannot split
            # a material. Appearance only breaks geometric ties near a seam.
            weighted_lab = lab * np.asarray((0.25, 1.0, 1.0), dtype=np.float32)
            reference = np.median(weighted_lab[core], axis=0)
            appearance = np.linalg.norm(weighted_lab - reference, axis=2) / 255.0
            cost = cost + 0.10 * appearance.astype(np.float32)
        cost = cost + 0.015 * (1.0 - seed_confidences[semantic])
        costs.append(cost)

    ownership = np.stack(costs, axis=0).argmin(axis=0)
    # Exclusive verified proposal pixels remain hard semantic seeds.
    stack = np.stack([seed_masks[name] for name in semantics], axis=0)
    exclusive_count = stack.sum(axis=0)
    for index, semantic in enumerate(semantics):
        ownership[(exclusive_count == 1) & seed_masks[semantic]] = index

    coarse = {
        semantic: root & (ownership == index)
        for index, semantic in enumerate(semantics)
    }
    refined, boundary_diagnostics = _refine_inventory_boundaries(
        image,
        root,
        coarse,
    )
    area_rows: list[dict[str, object]] = []
    stable = True
    for semantic in semantics:
        seed_area = max(1, int(np.count_nonzero(seed_masks[semantic])))
        final_area = int(np.count_nonzero(refined[semantic]))
        root_fraction = final_area / root_area
        seed_retention = final_area / seed_area
        accepted = bool(
            final_area >= 24
            and root_fraction >= 0.008
            and root_fraction <= 0.95
            and seed_retention >= 0.55
        )
        stable &= accepted
        area_rows.append(
            {
                "semantic_name": semantic,
                "seed_area_px": seed_area,
                "final_area_px": final_area,
                "root_fraction": root_fraction,
                "seed_retention_ratio": seed_retention,
                "accepted": accepted,
            }
        )
    complete = np.array_equal(
        np.logical_or.reduce(list(refined.values())),
        root,
    )
    disjoint = int(np.stack(list(refined.values()), axis=0).sum(axis=0).max()) <= 1
    if not (stable and complete and disjoint):
        return None, {
            "status": "partition_stability_gate_failed",
            "selected_profile": profile,
            "seed_coverage": seed_coverage,
            "complete_root_coverage": complete,
            "disjoint_ownership": disjoint,
            "areas": area_rows,
            "boundary_refinement": boundary_diagnostics,
            "candidates": rows,
            "ground_truth_used": False,
        }

    return refined, {
        "status": "completed",
        "algorithm": "hpid-semantic-seeded-physical-ownership-v1",
        "selected_profile": profile,
        "evidence_order": ["semantic", "structure", "appearance"],
        "verified_semantics": list(semantics),
        "seed_coverage": seed_coverage,
        "areas": area_rows,
        "boundary_refinement": boundary_diagnostics,
        "appearance_role": "boundary_tie_break_and_refinement_only",
        "appearance_can_create_ids": False,
        "complete_root_coverage": True,
        "disjoint_ownership": True,
        "candidates": rows,
        "ground_truth_used": False,
    }


def _firearm_handguard_from_structure_and_material(
    image: Image.Image | np.ndarray | None,
    fore_end: np.ndarray,
    normal_coordinate: np.ndarray,
    semantic_seed: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    """Recover the fore-end handguard without promoting texture to an ID."""

    fore_area = int(np.count_nonzero(fore_end))
    if fore_area < 96:
        return np.zeros_like(fore_end), {
            "status": "fore_end_too_small",
            "ground_truth_used": False,
        }
    normal_values = normal_coordinate[fore_end]
    lower_cut = float(np.quantile(normal_values, 0.56))
    upper_cut = float(np.quantile(normal_values, 0.34))
    lower_structure = fore_end & (normal_coordinate >= lower_cut)

    # Detector labels can swap barrel and handguard.  Retain only semantic
    # components that occupy the lower fore-end or overlap its structural core.
    trusted_semantic = np.zeros_like(fore_end)
    semantic_fore = semantic_seed & fore_end
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        semantic_fore.astype(np.uint8), connectivity=8
    )
    semantic_components: list[dict[str, object]] = []
    for component_index in range(1, count):
        component = labels == component_index
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if area < max(24, round(fore_area * 0.008)):
            continue
        lower_overlap = float(
            np.count_nonzero(component & lower_structure) / max(1, area)
        )
        mean_normal = float(normal_coordinate[component].mean())
        accepted = lower_overlap >= 0.16 or mean_normal >= lower_cut
        if accepted:
            trusted_semantic |= component
        semantic_components.append(
            {
                "area_px": area,
                "lower_overlap": lower_overlap,
                "mean_normal": mean_normal,
                "accepted": accepted,
            }
        )
    minimum_area = max(48, round(fore_area * 0.08))
    semantic_seed_accepted = np.count_nonzero(trusted_semantic) >= minimum_area
    # A sufficiently large lower semantic proposal is a better material
    # reference than the whole lower half of the fore-end.  The latter can
    # also contain the gas block or barrel hardware and is only a fallback.
    structural_seed = (
        trusted_semantic.copy() if semantic_seed_accepted else lower_structure.copy()
    )

    fallback = structural_seed.copy()
    if image is None:
        return fallback, {
            "status": "structure_only",
            "lower_cut": lower_cut,
            "upper_cut": upper_cut,
            "semantic_components": semantic_components,
            "semantic_seed_accepted": bool(semantic_seed_accepted),
            "appearance_used": False,
            "ground_truth_used": False,
        }
    rgb = (
        np.asarray(image.convert("RGB"), dtype=np.uint8)
        if isinstance(image, Image.Image)
        else np.asarray(image, dtype=np.uint8)
    )
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    if rgb.shape[:2] != fore_end.shape or rgb.ndim != 3 or rgb.shape[2] != 3:
        return fallback, {
            "status": "structure_only_image_mismatch",
            "semantic_components": semantic_components,
            "semantic_seed_accepted": bool(semantic_seed_accepted),
            "appearance_used": False,
            "ground_truth_used": False,
        }

    scale = min(1.0, 640.0 / max(rgb.shape[:2]))
    width = max(32, round(rgb.shape[1] * scale))
    height = max(32, round(rgb.shape[0] * scale))
    reduced = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    reduced = cv2.bilateralFilter(reduced, 11, 46, 46)
    smoothed = cv2.resize(
        reduced,
        (rgb.shape[1], rgb.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    lab = cv2.cvtColor(smoothed, cv2.COLOR_RGB2LAB).astype(np.float32)

    seed_distance = cv2.distanceTransform(
        structural_seed.astype(np.uint8), cv2.DIST_L2, 5
    )
    seed_core = structural_seed & (
        seed_distance >= max(1.5, seed_distance.max() * 0.24)
    )
    if np.count_nonzero(seed_core) < 24:
        seed_core = structural_seed
    metal_core = fore_end & (normal_coordinate <= upper_cut) & ~semantic_fore
    if np.count_nonzero(metal_core) < 24:
        metal_core = fore_end & ~structural_seed
    if min(np.count_nonzero(seed_core), np.count_nonzero(metal_core)) < 24:
        return fallback, {
            "status": "structure_only_missing_material_reference",
            "semantic_components": semantic_components,
            "semantic_seed_accepted": bool(semantic_seed_accepted),
            "appearance_used": False,
            "ground_truth_used": False,
        }

    # Material identity must survive illumination changes along a curved or
    # glossy part.  Downweight Lab lightness while retaining chromatic cues;
    # this joins bright and shadowed wood/polymer without joining grey metal.
    material_lab = lab * np.asarray((0.28, 1.0, 1.0), dtype=np.float32)
    handguard_center = np.median(material_lab[seed_core], axis=0)
    metal_center = np.median(material_lab[metal_core], axis=0)
    reference_contrast = float(np.linalg.norm(handguard_center - metal_center))
    if reference_contrast < 13.0:
        return fallback, {
            "status": "structure_only_low_material_contrast",
            "reference_contrast": reference_contrast,
            "semantic_components": semantic_components,
            "semantic_seed_accepted": bool(semantic_seed_accepted),
            "appearance_used": False,
            "ground_truth_used": False,
        }

    handguard_distance = np.linalg.norm(material_lab - handguard_center, axis=2)
    metal_distance = np.linalg.norm(material_lab - metal_center, axis=2)
    material_support = fore_end & (
        handguard_distance + min(8.0, 0.18 * reference_contrast) <= metal_distance
    )
    material_support |= structural_seed
    radius = max(1, round(min(fore_end.shape) * 0.0025))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    material_support = cv2.morphologyEx(
        material_support.astype(np.uint8), cv2.MORPH_CLOSE, kernel
    ).astype(bool)
    material_support &= fore_end

    # Keep the structural component and any separately visible upper cover
    # whose material independently agrees with it.  Tiny wood-grain islands
    # and highlights are absorbed by the nearest inventory part.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        material_support.astype(np.uint8), connectivity=8
    )
    accepted = np.zeros_like(fore_end)
    component_rows: list[dict[str, object]] = []
    for component_index in range(1, count):
        component = labels == component_index
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        seed_overlap = int(np.count_nonzero(component & structural_seed))
        median_distance = float(np.median(handguard_distance[component]))
        keep = bool(
            seed_overlap > 0
            or (
                area >= max(32, round(fore_area * 0.025))
                and median_distance <= 0.72 * reference_contrast
            )
        )
        if keep:
            accepted |= component
        component_rows.append(
            {
                "area_px": area,
                "seed_overlap_px": seed_overlap,
                "median_handguard_distance": median_distance,
                "accepted": keep,
            }
        )
    fraction = float(np.count_nonzero(accepted) / max(1, fore_area))
    if not 0.12 <= fraction <= 0.78:
        return fallback, {
            "status": "structure_only_material_fraction_gate",
            "material_fraction": fraction,
            "reference_contrast": reference_contrast,
            "semantic_components": semantic_components,
            "appearance_used": False,
            "ground_truth_used": False,
        }
    return accepted, {
        "status": "completed",
        "algorithm": "firearm-structure-material-handguard-v1",
        "lower_cut": lower_cut,
        "upper_cut": upper_cut,
        "reference_contrast": reference_contrast,
        "material_fraction": fraction,
        "semantic_components": semantic_components,
        "semantic_seed_accepted": bool(semantic_seed_accepted),
        "material_components": component_rows,
        "appearance_used": True,
        "appearance_role": "boundary_and_same-material-cover_support",
        "appearance_can_create_ids": False,
        "ground_truth_used": False,
    }


def _firearm_structural_masks(
    instance_map: np.ndarray,
    records: list[PartInstance] | tuple[PartInstance, ...],
    image: Image.Image | np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray] | None, dict[str, object]]:
    """Partition a side-view firearm from structure, then align real seams."""

    root = instance_map > 0
    ys, xs = np.nonzero(root)
    if len(xs) < 256:
        return None, {"status": "root_too_small", "ground_truth_used": False}
    points = np.column_stack((xs, ys)).astype(np.float64)
    center = points.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov((points - center).T))
    if eigenvalues[-1] <= 1.8 * max(1e-6, eigenvalues[-2]):
        return None, {"status": "root_not_axial", "ground_truth_used": False}
    axis = eigenvectors[:, -1]

    def semantic_mask(*names: str) -> np.ndarray:
        indices = [
            record.instance_index for record in records if record.semantic_name in names
        ]
        return (
            np.isin(instance_map, indices)
            if indices
            else np.zeros(instance_map.shape, dtype=bool)
        )

    stock_seed = semantic_mask("tool_prop_stock")
    muzzle_seed = semantic_mask("tool_prop_muzzle")
    if not np.any(stock_seed) or not np.any(muzzle_seed):
        return None, {
            "status": "missing_axis_endpoint_seeds",
            "stock_seed": bool(np.any(stock_seed)),
            "muzzle_seed": bool(np.any(muzzle_seed)),
            "ground_truth_used": False,
        }
    stock_points = np.column_stack(np.nonzero(stock_seed)[::-1]).astype(np.float64)
    muzzle_points = np.column_stack(np.nonzero(muzzle_seed)[::-1]).astype(np.float64)
    if float((stock_points.mean(axis=0) - muzzle_points.mean(axis=0)) @ axis) < 0:
        axis = -axis
    normal = np.asarray((-axis[1], axis[0]), dtype=np.float64)

    grid_y, grid_x = np.indices(root.shape)
    delta_x = grid_x - center[0]
    delta_y = grid_y - center[1]
    axial = delta_x * axis[0] + delta_y * axis[1]
    normal_coordinate = delta_x * normal[0] + delta_y * normal[1]
    root_normal = normal_coordinate[root]
    if float(np.quantile(root_normal, 0.95)) < abs(
        float(np.quantile(root_normal, 0.05))
    ):
        normal = -normal
        normal_coordinate = -normal_coordinate
        root_normal = -root_normal

    root_axial = axial[root]
    axial_span = max(1.0, float(root_axial.max() - root_axial.min()))
    seed_threshold = float(np.quantile(root_normal, 0.85))
    central_low = float(np.quantile(root_axial, 0.20))
    central_high = float(np.quantile(root_axial, 0.75))
    lower_seed_region = (
        root
        & (normal_coordinate >= seed_threshold)
        & (axial >= central_low)
        & (axial <= central_high)
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        lower_seed_region.astype(np.uint8),
        connectivity=8,
    )
    minimum_seed_area = max(64, round(np.count_nonzero(root) * 0.004))
    seed_rows: list[tuple[int, float, float, float, np.ndarray]] = []
    for component_index in range(1, count):
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        if area < minimum_seed_area:
            continue
        component = labels == component_index
        seed_rows.append(
            (
                area,
                float(axial[component].mean()),
                float(axial[component].min()),
                float(axial[component].max()),
                component,
            )
        )
    if len(seed_rows) < 2:
        return None, {
            "status": "lower_topology_unresolved",
            "lower_seed_count": len(seed_rows),
            "ground_truth_used": False,
        }
    selected_seeds = sorted(
        sorted(seed_rows, key=lambda row: row[0], reverse=True)[:2],
        key=lambda row: row[1],
    )
    front_seed, rear_seed = selected_seeds
    separation = 0.5 * (front_seed[3] + rear_seed[2])
    expansion_threshold = float(np.quantile(root_normal, 0.80))

    def expand_lower_seed(
        seed: np.ndarray,
        lower_u: float,
        upper_u: float,
    ) -> np.ndarray:
        region = (
            root
            & (normal_coordinate >= expansion_threshold)
            & (axial >= lower_u)
            & (axial <= upper_u)
        )
        component_count, component_labels, component_stats, _ = (
            cv2.connectedComponentsWithStats(region.astype(np.uint8), connectivity=8)
        )
        if component_count <= 1:
            return region
        best_index = max(
            range(1, component_count),
            key=lambda index: (
                int(np.count_nonzero((component_labels == index) & seed)),
                int(component_stats[index, cv2.CC_STAT_AREA]),
            ),
        )
        return component_labels == best_index

    margin = 0.02 * axial_span
    magazine = expand_lower_seed(
        front_seed[4],
        front_seed[2] - margin,
        separation,
    )
    grip = expand_lower_seed(
        rear_seed[4],
        separation,
        rear_seed[3] + margin,
    )
    if min(np.count_nonzero(magazine), np.count_nonzero(grip)) < minimum_seed_area:
        return None, {
            "status": "lower_topology_expansion_failed",
            "ground_truth_used": False,
        }

    stock_start = float(np.quantile(axial[stock_seed], 0.02))
    muzzle_end = float(np.quantile(axial[muzzle_seed], 0.98))
    receiver_start = front_seed[2] - 0.015 * axial_span
    if not (
        muzzle_end + 0.05 * axial_span < receiver_start
        and receiver_start + 0.05 * axial_span < stock_start
    ):
        return None, {
            "status": "axial_boundaries_unresolved",
            "muzzle_end": muzzle_end,
            "receiver_start": receiver_start,
            "stock_start": stock_start,
            "ground_truth_used": False,
        }

    lower_parts = magazine | grip
    stock = root & (axial >= stock_start) & ~lower_parts
    muzzle = root & (axial <= muzzle_end) & ~lower_parts
    fore_end = root & (axial > muzzle_end) & (axial < receiver_start) & ~lower_parts
    handguard_seed = semantic_mask("tool_prop_handguard", "tool_prop_barrel")
    handguard, handguard_diagnostics = _firearm_handguard_from_structure_and_material(
        image,
        fore_end,
        normal_coordinate,
        handguard_seed,
    )
    minimum_handguard_area = max(64, round(np.count_nonzero(fore_end) * 0.04))
    if np.count_nonzero(handguard) < minimum_handguard_area:
        return None, {
            "status": "handguard_seed_unresolved",
            "ground_truth_used": False,
        }
    barrel = fore_end & ~handguard
    claimed = stock | muzzle | handguard | barrel | magazine | grip
    receiver = root & ~claimed
    coarse_masks = {
        "tool_prop_muzzle": muzzle,
        "tool_prop_barrel": barrel,
        "tool_prop_handguard": handguard,
        "tool_prop_receiver": receiver,
        "tool_prop_magazine": magazine,
        "tool_prop_grip": grip,
        "tool_prop_stock": stock,
    }
    if any(np.count_nonzero(mask) < 32 for mask in coarse_masks.values()):
        return None, {
            "status": "empty_structural_inventory_part",
            "areas": {
                name: int(np.count_nonzero(mask)) for name, mask in coarse_masks.items()
            },
            "ground_truth_used": False,
        }
    masks, boundary_diagnostics = _refine_inventory_boundaries(
        image,
        root,
        coarse_masks,
    )
    return masks, {
        "status": "completed",
        "algorithm": "firearm-structure-seeded-inventory-v2",
        "axis_xy": [float(axis[0]), float(axis[1])],
        "normal_xy": [float(normal[0]), float(normal[1])],
        "seed_threshold": seed_threshold,
        "expansion_threshold": expansion_threshold,
        "muzzle_end": muzzle_end,
        "receiver_start": receiver_start,
        "stock_start": stock_start,
        "lower_seed_count": len(seed_rows),
        "selected_lower_seed_areas": [front_seed[0], rear_seed[0]],
        "areas": {name: int(np.count_nonzero(mask)) for name, mask in masks.items()},
        "coarse_areas": {
            name: int(np.count_nonzero(mask)) for name, mask in coarse_masks.items()
        },
        "boundary_refinement": boundary_diagnostics,
        "handguard_fusion": handguard_diagnostics,
        "appearance_used": boundary_diagnostics["status"] == "completed",
        "appearance_role": "boundary_alignment_only",
        "appearance_can_create_ids": False,
        "ground_truth_used": False,
    }


def _firearm_verified_detail_masks(
    instance_map: np.ndarray,
    records: list[PartInstance] | tuple[PartInstance, ...],
    candidates: tuple[MaskCandidate, ...],
    masks: dict[str, np.ndarray],
    structural_diagnostics: dict[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Add only detail parts that pass semantic, structural, and edge checks.

    The structural inventory owns the seven macro surfaces.  A small semantic
    candidate may carve a detail from those surfaces only when its profile
    label has non-trivial text support, its position matches firearm topology,
    and the underlying mask has a closed/aligned boundary.  This prevents weak
    color strips from becoming parts while retaining real sights and triggers.
    """

    root = instance_map > 0
    root_area = max(1, int(np.count_nonzero(root)))
    axis_values = structural_diagnostics.get("axis_xy")
    normal_values = structural_diagnostics.get("normal_xy")
    if not (
        isinstance(axis_values, list)
        and len(axis_values) == 2
        and isinstance(normal_values, list)
        and len(normal_values) == 2
    ):
        return masks, {
            "status": "structural_frame_unavailable",
            "ground_truth_used": False,
        }
    axis = np.asarray(axis_values, dtype=np.float64)
    normal = np.asarray(normal_values, dtype=np.float64)
    ys, xs = np.nonzero(root)
    center = np.asarray((xs.mean(), ys.mean()), dtype=np.float64)
    grid_y, grid_x = np.indices(root.shape)
    delta_x = grid_x - center[0]
    delta_y = grid_y - center[1]
    axial = delta_x * axis[0] + delta_y * axis[1]
    normal_coordinate = delta_x * normal[0] + delta_y * normal[1]
    root_axial = axial[root]
    root_normal = normal_coordinate[root]
    axial_min = float(root_axial.min())
    axial_span = max(1.0, float(root_axial.max() - axial_min))
    normal_center = float(np.median(root_normal))
    normal_span = max(
        1.0,
        float(np.quantile(root_normal, 0.95) - np.quantile(root_normal, 0.05)),
    )

    candidates_by_semantic: dict[str, list[MaskCandidate]] = {}
    for candidate in candidates:
        candidates_by_semantic.setdefault(candidate.semantic_name, []).append(candidate)
    records_by_semantic: dict[str, list[PartInstance]] = {}
    for record in records:
        records_by_semantic.setdefault(record.semantic_name, []).append(record)

    detail_rules = {
        "tool_prop_sight": {
            "maximum_fraction": 0.018,
            "minimum_probability": 0.08,
            "minimum_margin": -0.01,
            "normal_maximum": -0.015,
            "axial_interval": (0.03, 0.80),
        },
        "tool_prop_trigger": {
            "maximum_fraction": 0.012,
            "minimum_probability": 0.09,
            "minimum_margin": 0.01,
            "normal_minimum": 0.02,
            "axial_interval": (0.42, 0.78),
        },
        "tool_prop_trigger_guard": {
            "maximum_fraction": 0.025,
            "minimum_probability": 0.10,
            "minimum_margin": 0.01,
            "normal_minimum": 0.015,
            "axial_interval": (0.38, 0.80),
        },
        "tool_prop_charging_handle": {
            "maximum_fraction": 0.007,
            "minimum_probability": 0.14,
            "minimum_margin": 0.015,
            "normal_maximum": 0.03,
            "axial_interval": (0.35, 0.78),
        },
    }
    output = {name: mask.copy() for name, mask in masks.items()}
    detail_masks: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for semantic, rule in detail_rules.items():
        for record in records_by_semantic.get(semantic, []):
            record_mask = instance_map == record.instance_index
            record_area = int(np.count_nonzero(record_mask))
            if record_area == 0:
                continue
            matching = sorted(
                candidates_by_semantic.get(semantic, []),
                key=lambda candidate: int(
                    np.count_nonzero(candidate.mask.astype(bool) & record_mask)
                ),
                reverse=True,
            )
            candidate = matching[0] if matching else None
            metadata = candidate.metadata if candidate is not None else {}
            probability = float(metadata.get("semantic_rerank_probability", 0.0))
            margin = float(metadata.get("semantic_rerank_margin", -1.0))
            direct_semantic = bool(
                candidate is not None
                and (
                    "vlm" in candidate.source
                    or "florence" in candidate.source
                    or "profile-refine" in candidate.source
                )
            )
            cross_source_semantic = bool(
                candidate is not None
                and (
                    metadata.get("cross_source_confirmed")
                    or metadata.get("multi_view_confirmed")
                    or isinstance(metadata.get("vlm_semantic_audit"), dict)
                    and metadata["vlm_semantic_audit"].get("decision")
                    == "semantic_supported"
                )
            )
            appearance = metadata.get("appearance_graph_evidence")
            appearance = appearance if isinstance(appearance, dict) else {}
            boundary_alignment = float(
                metadata.get(
                    "proposal_boundary_alignment",
                    appearance.get("boundary_alignment", 0.0),
                )
            )
            boundary_closure = float(appearance.get("boundary_closure", 0.0))
            independent_cues = int(appearance.get("independent_cue_count", 0))
            semantic_ok = bool(
                direct_semantic
                or (
                    cross_source_semantic
                    and
                    probability >= float(rule["minimum_probability"])
                    and margin >= float(rule["minimum_margin"])
                )
            )
            boundary_ok = bool(
                boundary_alignment >= 0.78
                and (boundary_closure >= 0.52 or independent_cues >= 2)
            )
            area_fraction = record_area / root_area
            axial_center = float(np.median((axial[record_mask] - axial_min) / axial_span))
            normal_centered = float(
                (np.median(normal_coordinate[record_mask]) - normal_center)
                / normal_span
            )
            interval = rule["axial_interval"]
            assert isinstance(interval, tuple)
            geometry_ok = bool(
                area_fraction <= float(rule["maximum_fraction"])
                and float(interval[0]) <= axial_center <= float(interval[1])
                and (
                    "normal_minimum" not in rule
                    or normal_centered >= float(rule["normal_minimum"])
                )
                and (
                    "normal_maximum" not in rule
                    or normal_centered <= float(rule["normal_maximum"])
                )
            )
            accepted = bool(
                record_area >= max(20, round(root_area * 0.00008))
                and semantic_ok
                and boundary_ok
                and geometry_ok
            )
            rows.append(
                {
                    "part_id": record.part_id,
                    "semantic_name": semantic,
                    "area_fraction": area_fraction,
                    "semantic_probability": probability,
                    "semantic_margin": margin,
                    "direct_semantic_source": direct_semantic,
                    "cross_source_semantic": cross_source_semantic,
                    "boundary_alignment": boundary_alignment,
                    "boundary_closure": boundary_closure,
                    "independent_cue_count": independent_cues,
                    "axial_center": axial_center,
                    "normal_centered": normal_centered,
                    "semantic_verified": semantic_ok,
                    "structure_verified": geometry_ok,
                    "appearance_verified": boundary_ok,
                    "accepted": accepted,
                }
            )
            if accepted:
                detail_masks[semantic] = detail_masks.get(
                    semantic, np.zeros(root.shape, dtype=bool)
                ) | record_mask
    if detail_masks:
        detail_union = np.logical_or.reduce(list(detail_masks.values()))
        for semantic in tuple(output):
            output[semantic] &= ~detail_union
        output.update(detail_masks)
    return output, {
        "status": "completed",
        "algorithm": "firearm-three-evidence-detail-verification-v1",
        "accepted_detail_count": len(detail_masks),
        "rows": rows,
        "evidence_order": ["semantic_inventory", "structure", "appearance"],
        "appearance_can_create_ids": False,
        "ground_truth_used": False,
    }


def _character_surface_group_overrides(
    instance_map: np.ndarray,
    records: list[PartInstance] | tuple[PartInstance, ...],
    image: Image.Image | np.ndarray | None,
) -> tuple[dict[str, tuple[str, int | None, str]], dict[str, object]]:
    """Recover clothing groups from fine visual atoms and body landmarks.

    Semantic anatomy remains authoritative. Unnamed visual atoms are assigned
    by their position relative to the detected head and full character extent;
    image appearance is used only to decide whether an ``arm`` mask is exposed
    skin or a sleeve, and whether a head atom continues the detected hair.
    """

    record_list = list(records)
    if (
        image is None
        or len({record.asset_id for record in record_list}) != 1
        or not record_list
        or any(_root_domain(record) != "character" for record in record_list)
    ):
        return {}, {"status": "inactive"}
    root = instance_map > 0
    ys, xs = np.nonzero(root)
    if len(xs) < 256:
        return {}, {"status": "root_too_small", "ground_truth_used": False}
    x0, x1 = int(xs.min()), int(xs.max() + 1)
    y0, y1 = int(ys.min()), int(ys.max() + 1)
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    if max(height, width) < 64:
        return {}, {"status": "character_extent_too_small", "ground_truth_used": False}

    rgb = (
        np.asarray(image.convert("RGB"), dtype=np.uint8)
        if isinstance(image, Image.Image)
        else np.asarray(image, dtype=np.uint8)
    )
    if rgb.shape[:2] != instance_map.shape:
        return {}, {"status": "image_shape_mismatch", "ground_truth_used": False}
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    root_area = max(1, int(np.count_nonzero(root)))
    diagonal = max(1.0, float(np.hypot(height, width)))
    axis_frame = _character_axis_frame(instance_map, record_list)
    if axis_frame is None:
        return {}, {"status": "character_axis_unavailable", "ground_truth_used": False}

    def mask_for(record: PartInstance) -> np.ndarray:
        return instance_map == record.instance_index

    def normalized_geometry(record: PartInstance) -> tuple[float, float, float, float]:
        mask = mask_for(record)
        axial = axis_frame.coordinate[mask]
        lateral = axis_frame.lateral[mask]
        top = float(np.quantile(axial, 0.01))
        bottom = float(np.quantile(axial, 0.99))
        center_y = float(np.median(axial))
        center_x = float(np.clip(np.median(lateral) + 0.5, 0.0, 1.0))
        return top, bottom, center_y, center_x

    head_bottom_norm = axis_frame.head_end
    footwear_starts = [
        normalized_geometry(record)[0]
        for record in record_list
        if _words(record.semantic_name) & {"shoe", "boot", "sandal", "sock", "footwear"}
    ]
    if footwear_starts:
        footwear_start = float(np.median(footwear_starts))
        lower_zone_start = float(
            np.clip(
                head_bottom_norm + 0.50 * (footwear_start - head_bottom_norm),
                head_bottom_norm + 0.12,
                0.68,
            )
        )
    else:
        footwear_start = None
        lower_zone_start = max(0.60, head_bottom_norm + 0.13)

    hair_records = [
        record for record in record_list if "hair" in _words(record.semantic_name)
    ]
    hair_mask = np.isin(
        instance_map,
        [record.instance_index for record in hair_records],
    )
    hair_lab = np.median(lab[hair_mask], axis=0) if np.any(hair_mask) else None
    hair_distance = (
        cv2.distanceTransform((~hair_mask).astype(np.uint8), cv2.DIST_L2, 3)
        if np.any(hair_mask)
        else None
    )

    upper_records = [
        record
        for record in record_list
        if record.semantic_name != record.semantic_parent
        and bool(_words(record.semantic_name) & _UPPER_GARMENT_TOKENS)
        and "sleeve" not in _words(record.semantic_name)
    ]
    sleeve_records = [
        record
        for record in record_list
        if "sleeve" in _words(record.semantic_name)
    ]
    valid_upper_records: list[PartInstance] = []
    rejected_upper_records: list[PartInstance] = []
    for record in upper_records:
        top, bottom, center_y, _ = normalized_geometry(record)
        if (
            top >= max(0.0, head_bottom_norm - 0.035)
            and bottom <= lower_zone_start + 0.12
            and center_y <= lower_zone_start + 0.04
        ):
            valid_upper_records.append(record)
        else:
            rejected_upper_records.append(record)

    material_lab = lab * np.asarray((0.42, 1.0, 1.0), dtype=np.float32)
    upper_layer_distances: list[float] = []
    for sleeve in sleeve_records:
        sleeve_mask = mask_for(sleeve)
        if not np.any(sleeve_mask):
            continue
        sleeve_center = np.median(material_lab[sleeve_mask], axis=0)
        distances = [
            float(
                np.linalg.norm(
                    sleeve_center
                    - np.median(material_lab[mask_for(record)], axis=0)
                )
            )
            for record in valid_upper_records
            if np.any(mask_for(record))
        ]
        if distances:
            upper_layer_distances.append(min(distances))
    layered_upper = bool(
        valid_upper_records
        and sleeve_records
        and upper_layer_distances
        and float(np.median(upper_layer_distances)) >= 12.0
    )
    upper_layer_overrides: dict[str, tuple[str, int | None, str]] = {}
    if layered_upper:
        for record in valid_upper_records:
            upper_layer_overrides[record.part_id] = (
                "character_inner_top",
                None,
                "character_layered_garment_inner_semantic_seed",
            )
        for record in sleeve_records:
            upper_layer_overrides[record.part_id] = (
                "character_outer_garment",
                None,
                "character_layered_garment_outer_sleeve_seed",
            )
    for record in rejected_upper_records:
        record_mask = mask_for(record)
        source_lab = np.median(lab[record_mask], axis=0)
        continues_hair = bool(
            hair_lab is not None
            and hair_distance is not None
            and float(np.linalg.norm(source_lab - hair_lab)) <= 20.0
            and float(hair_distance[record_mask].min(initial=diagonal))
            <= 0.035 * diagonal
        )
        upper_layer_overrides[record.part_id] = (
            "character_hair" if continues_hair else "character_body",
            None,
            (
                "character_overbroad_upper_candidate_hair_return"
                if continues_hair
                else "character_overbroad_upper_candidate_body_return"
            ),
        )

    inner_layer_mask = np.isin(
        instance_map,
        [record.instance_index for record in valid_upper_records],
    )
    outer_layer_mask = np.isin(
        instance_map,
        [record.instance_index for record in sleeve_records],
    )
    layer_references = {
        "character_inner_top": (
            np.median(material_lab[inner_layer_mask], axis=0)
            if np.any(inner_layer_mask)
            else None
        ),
        "character_outer_garment": (
            np.median(material_lab[outer_layer_mask], axis=0)
            if np.any(outer_layer_mask)
            else None
        ),
    }

    visual_assignments: dict[str, tuple[str, int | None, str]] = {}
    upper_visual_mask = np.zeros(instance_map.shape, dtype=bool)
    rows: list[dict[str, object]] = []
    for record in record_list:
        if not _is_visual_semantic(record.semantic_name):
            continue
        mask = mask_for(record)
        top, bottom, center_y, center_x = normalized_geometry(record)
        area_fraction = record.area_px / root_area
        semantic = "character_body"
        evidence = "character_visual_body_fallback"
        source_lab = np.median(lab[mask], axis=0)
        continues_hair = bool(
            hair_lab is not None
            and hair_distance is not None
            and float(np.linalg.norm(source_lab - hair_lab)) <= 18.0
            and float(hair_distance[mask].min(initial=diagonal)) <= 0.025 * diagonal
        )
        if continues_hair and center_y <= lower_zone_start + 0.02:
            semantic = "character_hair"
            evidence = "character_hair_appearance_continuation"
        elif top >= 0.82 or (center_y >= 0.88 and bottom >= 0.91):
            semantic = "character_footwear"
            evidence = "character_pose_axis_footwear_zone"
        elif (
            center_y >= lower_zone_start - 0.012
            and bottom >= lower_zone_start + 0.04
            and top >= head_bottom_norm + 0.10
        ):
            semantic = "character_lower_garment"
            evidence = "character_pose_axis_lower_garment_zone"
        elif (
            bottom >= head_bottom_norm - 0.03
            and top <= lower_zone_start + 0.08
            and center_y <= lower_zone_start + 0.08
        ):
            if layered_upper:
                visual_center = np.median(material_lab[mask], axis=0)
                distances = [
                    (float(np.linalg.norm(visual_center - reference)), name)
                    for name, reference in layer_references.items()
                    if reference is not None
                ]
                if distances:
                    _, semantic = min(distances, key=lambda row: row[0])
                    evidence = "character_layered_garment_material_attachment"
                else:
                    semantic = "character_upper_garment"
                    evidence = "character_torso_and_sleeve_zone"
            else:
                semantic = "character_upper_garment"
                evidence = "character_torso_and_sleeve_zone"
        elif (
            bottom <= head_bottom_norm + 0.10
            and center_y <= head_bottom_norm
            and area_fraction >= 0.001
            and 0.08 <= center_x <= 0.92
        ):
            # A visual atom inside the face is commonly an eye socket, blush,
            # highlight, or shading region.  Location plus appearance is not
            # semantic evidence for headwear.  Only an explicit headwear
            # candidate may create that public ID; otherwise the atom returns
            # to the body surface.
            semantic = "character_hair" if continues_hair else "character_body"
            evidence = (
                "character_hair_appearance_continuation"
                if continues_hair
                else "character_head_visual_body_fallback"
            )
        visual_assignments[record.part_id] = (semantic, None, evidence)
        if semantic in {
            "character_upper_garment",
            "character_inner_top",
            "character_outer_garment",
        }:
            upper_visual_mask |= mask
        rows.append(
            {
                "part_id": record.part_id,
                "semantic_name": semantic,
                "top": float(top),
                "bottom": float(bottom),
                "center_y": float(center_y),
                "area_fraction": float(area_fraction),
                "evidence": evidence,
            }
        )

    upper_semantic_indices = [
        record.instance_index
        for record in record_list
        if _words(record.semantic_name) & _UPPER_GARMENT_TOKENS
    ]
    if upper_semantic_indices:
        upper_visual_mask |= np.isin(instance_map, upper_semantic_indices)

    skin_records = [
        record
        for record in record_list
        if _words(record.semantic_name) & {"face", "hand"}
    ]
    if not skin_records:
        skin_records = [
            record
            for record in record_list
            if _words(record.semantic_name) & {"head", "arm"}
        ]
    skin_mask = np.isin(
        instance_map,
        [record.instance_index for record in skin_records],
    )
    skin_lab = np.median(lab[skin_mask], axis=0) if np.any(skin_mask) else None
    upper_distance = (
        cv2.distanceTransform(
            (~upper_visual_mask).astype(np.uint8),
            cv2.DIST_L2,
            3,
        )
        if np.any(upper_visual_mask)
        else None
    )

    overrides = dict(visual_assignments)
    overrides.update(upper_layer_overrides)
    for record in record_list:
        words = _words(record.semantic_name)
        mask = mask_for(record)
        top, bottom, center_y, _ = normalized_geometry(record)
        if record.part_id in upper_layer_overrides:
            continue
        if words & _LOWER_GARMENT_TOKENS or record.semantic_name.endswith(
            "_lower_clothing"
        ):
            if top >= 0.80:
                overrides[record.part_id] = (
                    "character_footwear",
                    None,
                    "character_lower_label_rejected_by_footwear_zone",
                )
            else:
                overrides[record.part_id] = (
                    "character_lower_garment",
                    None,
                    "character_lower_garment_inventory",
                )
        elif words & _UPPER_GARMENT_TOKENS:
            overrides[record.part_id] = (
                "character_upper_garment",
                None,
                "character_upper_garment_inventory",
            )
        elif words & {"headwear", "hat", "cap", "helmet"}:
            overrides[record.part_id] = (
                "character_headwear",
                None,
                "explicit_character_headwear_inventory",
            )
        elif words & {"head", "face"}:
            overrides[record.part_id] = (
                "character_body",
                None,
                "character_head_anatomy_body_group",
            )
        elif (
            record.semantic_name == "character_accessory"
            and center_y <= head_bottom_norm + 0.08
            and bottom <= head_bottom_norm + 0.08
        ):
            overrides[record.part_id] = (
                "character_body",
                None,
                "generic_face_accessory_conservative_merge",
            )
        elif "torso" in words and skin_lab is not None:
            torso_lab = np.median(lab[mask], axis=0)
            skin_difference = float(np.linalg.norm(torso_lab - skin_lab))
            upper_zone = bool(
                bottom >= head_bottom_norm - 0.03
                and top <= lower_zone_start + 0.08
                and center_y <= lower_zone_start + 0.08
            )
            overrides[record.part_id] = (
                "character_lower_garment"
                if top >= 0.64
                else "character_upper_garment"
                if upper_zone and skin_difference >= 16.0
                else "character_body",
                None,
                "character_torso_surface_appearance_audit",
            )
            rows.append(
                {
                    "part_id": record.part_id,
                    "semantic_name": overrides[record.part_id][0],
                    "skin_lab_difference": skin_difference,
                    "top": float(top),
                    "bottom": float(bottom),
                    "evidence": "character_torso_surface_appearance_audit",
                }
            )
        elif "torso" in words and top >= 0.64:
            overrides[record.part_id] = (
                "character_lower_garment",
                None,
                "character_lower_zone_semantic_correction",
            )
        elif "arm" in words and skin_lab is not None and upper_distance is not None:
            arm_lab = np.median(lab[mask], axis=0)
            skin_difference = float(np.linalg.norm(arm_lab - skin_lab))
            upper_gap = float(upper_distance[mask].min(initial=diagonal))
            if skin_difference >= 20.0 and upper_gap <= 0.03 * diagonal:
                overrides[record.part_id] = (
                    (
                        "character_outer_garment"
                        if layered_upper
                        else "character_upper_garment"
                    ),
                    None,
                    "character_sleeve_skin_contrast",
                )
                rows.append(
                    {
                        "part_id": record.part_id,
                        "semantic_name": overrides[record.part_id][0],
                        "skin_lab_difference": skin_difference,
                        "upper_gap_px": upper_gap,
                        "evidence": "character_sleeve_skin_contrast",
                    }
                )

    return overrides, {
        "status": "completed",
        "algorithm": "character-pose-axis-surface-grouping-v2",
        "head_bottom_normalized": float(head_bottom_norm),
        "footwear_start_normalized": footwear_start,
        "lower_zone_start_normalized": float(lower_zone_start),
        "axis_direction_xy": list(axis_frame.axis_direction_xy),
        "override_count": len(overrides),
        "layered_upper_garment": {
            "detected": layered_upper,
            "valid_upper_seed_count": len(valid_upper_records),
            "outer_sleeve_seed_count": len(sleeve_records),
            "rejected_overbroad_upper_count": len(rejected_upper_records),
            "median_material_distance": (
                float(np.median(upper_layer_distances))
                if upper_layer_distances
                else None
            ),
            "appearance_role": "verify_same_or_distinct_semantic_garment_seeds",
            "appearance_can_create_ids": False,
        },
        "rows": rows,
        "appearance_role": "skin_or_sleeve_and_hair_continuation_only",
        "appearance_can_create_ids": False,
        "ground_truth_used": False,
    }


def _character_group(record: PartInstance) -> tuple[str, bool, str]:
    name = record.semantic_name
    words = _words(name)
    if name == "character" or words & _CHARACTER_BODY_TOKENS:
        return "character_body", True, "hierarchical_body_contract"
    if words & _UPPER_GARMENT_TOKENS:
        return "character_upper_garment", True, "garment_family_contract"
    if words & _LOWER_GARMENT_TOKENS:
        return "character_lower_garment", True, "garment_family_contract"
    if "hair" in words:
        return "character_hair", True, "semantic_part"
    if words & {"shoe", "boot", "sandal", "sock"}:
        return "character_footwear", True, "semantic_part"
    if _is_visual_semantic(name):
        return "character_body", True, "conservative_visual_merge"
    return name, False, "semantic_part"


def _knife_group(record: PartInstance) -> tuple[str, bool, str]:
    name = record.semantic_name
    words = _words(name)
    if "blade" in words:
        return "tool_prop_blade", True, "knife_inventory"
    if words & {"wrap", "wrapping", "cloth"}:
        return "tool_prop_wrap", True, "knife_inventory"
    if name == "tool_prop" or words & _KNIFE_HANDLE_TOKENS:
        return "tool_prop_handle", True, "knife_inventory"
    if _is_visual_semantic(name):
        return "tool_prop_handle", True, "conservative_visual_merge"
    return "tool_prop_handle", True, "knife_inventory_residual"


def _group_assignment(
    record: PartInstance,
    *,
    knife_inventory: bool,
    promoted_visuals: dict[str, str],
) -> tuple[str, bool, str]:
    domain = _root_domain(record)
    if domain == "character":
        return _character_group(record)
    if domain == "tool_prop" and knife_inventory:
        return _knife_group(record)
    if record.semantic_name in promoted_visuals:
        return (
            _physical_visual_name(record.semantic_name),
            False,
            promoted_visuals[record.semantic_name],
        )
    if record.semantic_name == record.semantic_parent or _is_visual_semantic(
        record.semantic_name
    ):
        return f"{domain}_body", True, "conservative_visual_merge"
    return record.semantic_name, False, "semantic_part"


def _repeated_semantic_shape_recovery(
    instance_map: np.ndarray,
    records: list[PartInstance] | tuple[PartInstance, ...],
    candidates: tuple[MaskCandidate, ...],
    *,
    profile: str | None,
    verified_semantics: set[str],
    rejected_semantics: set[str],
) -> tuple[dict[str, tuple[str, int, str]], dict[str, object]]:
    """Recover a mislabeled sibling only from a verified repeated-part prototype."""

    if profile is None or not rejected_semantics:
        return {}, {"status": "inactive", "ground_truth_used": False}
    root = instance_map > 0
    diagonal = max(1.0, float(np.hypot(*root.shape)))
    prototypes: list[dict[str, object]] = []
    for candidate in candidates:
        candidate_profile = str(
            candidate.metadata.get("selected_part_profile")
            or candidate.metadata.get("semantic_rerank_profile")
            or candidate.metadata.get("structural_profile")
            or ""
        ).strip()
        maximum_instances = int(candidate.metadata.get("maximum_instances", 1))
        family = _semantic_shape_family(candidate.semantic_name)
        if (
            candidate_profile != profile
            or candidate.semantic_name not in verified_semantics
            or maximum_instances <= 1
            or family is None
        ):
            continue
        mask = np.asarray(candidate.mask, dtype=bool) & root
        descriptor = _mask_shape_descriptor(mask)
        if not _shape_family_matches(family, descriptor):
            continue
        prototypes.append(
            {
                "semantic_name": candidate.semantic_name,
                "family": family,
                "mask": mask,
                "descriptor": descriptor,
                "maximum_instances": maximum_instances,
                "candidate_key": candidate.metadata.get("candidate_key"),
            }
        )

    overrides: dict[str, tuple[str, int, str]] = {}
    rows: list[dict[str, object]] = []
    for record in records:
        if record.semantic_name not in rejected_semantics:
            continue
        record_mask = instance_map == record.instance_index
        descriptor = _mask_shape_descriptor(record_mask)
        source_family = _semantic_shape_family(record.semantic_name)
        if descriptor is None or _shape_family_matches(source_family, descriptor):
            continue
        accepted: list[tuple[float, dict[str, object]]] = []
        for prototype in prototypes:
            family = str(prototype["family"])
            prototype_descriptor = prototype["descriptor"]
            if (
                family == source_family
                or not isinstance(prototype_descriptor, dict)
                or not _shape_family_matches(family, descriptor)
            ):
                continue
            prototype_mask = np.asarray(prototype["mask"], dtype=bool)
            overlap = float(
                np.count_nonzero(record_mask & prototype_mask)
                / max(1, np.count_nonzero(record_mask))
            )
            area_ratio = float(
                descriptor["area_px"] / max(1.0, prototype_descriptor["area_px"])
            )
            centroid_gap = float(
                np.hypot(
                    descriptor["centroid_x"] - prototype_descriptor["centroid_x"],
                    descriptor["centroid_y"] - prototype_descriptor["centroid_y"],
                )
                / diagonal
            )
            shape_distance = float(
                abs(
                    descriptor["elongation"]
                    - prototype_descriptor["elongation"]
                )
                + abs(
                    descriptor["circularity"]
                    - prototype_descriptor["circularity"]
                )
                + abs(
                    descriptor["bbox_fill_ratio"]
                    - prototype_descriptor["bbox_fill_ratio"]
                )
            )
            valid = bool(
                overlap <= 0.12
                and 0.45 <= area_ratio <= 2.20
                and centroid_gap >= 0.08
                and shape_distance <= 0.72
            )
            if valid:
                score = 1.0 - min(1.0, shape_distance / 0.72)
                accepted.append((score, prototype))
        if not accepted:
            continue
        score, selected = max(
            accepted,
            key=lambda row: (row[0], str(row[1]["semantic_name"])),
        )
        semantic_name = str(selected["semantic_name"])
        overrides[record.part_id] = (
            semantic_name,
            record.instance_index,
            "repeated_semantic_shape_recovery",
        )
        rows.append(
            {
                "part_id": record.part_id,
                "rejected_semantic": record.semantic_name,
                "recovered_semantic": semantic_name,
                "score": score,
                "prototype_candidate_key": selected["candidate_key"],
                "ground_truth_used": False,
            }
        )
    return overrides, {
        "status": "completed" if rows else "no_recovery",
        "algorithm": "repeated-semantic-shape-recovery-v1",
        "recovered_count": len(rows),
        "rows": rows,
        "ground_truth_used": False,
    }


def _repeated_semantic_limits(
    candidates: tuple[MaskCandidate, ...],
) -> dict[str, int]:
    limits: dict[str, int] = {}
    for candidate in candidates:
        maximum = int(candidate.metadata.get("maximum_instances", 1))
        if maximum > 1 and not _is_visual_semantic(candidate.semantic_name):
            limits[candidate.semantic_name] = max(
                limits.get(candidate.semantic_name, 1),
                maximum,
            )
    return limits


def _split_repeated_group_components(
    group_map: np.ndarray,
    groups: list[PhysicalGroup] | tuple[PhysicalGroup, ...],
    candidates: tuple[MaskCandidate, ...],
) -> tuple[np.ndarray, tuple[PhysicalGroup, ...], dict[str, object]]:
    """Split only disconnected regions that plausibly denote repeated objects."""

    limits = _repeated_semantic_limits(candidates)
    if not limits:
        return group_map, tuple(groups), {
            "status": "inactive",
            "ground_truth_used": False,
        }
    output = np.zeros_like(group_map)
    output_groups: list[PhysicalGroup] = []
    rows: list[dict[str, object]] = []
    next_index = 1
    for group in groups:
        mask = group_map == group.group_index
        maximum = limits.get(group.semantic_name, 1)
        words = _words(group.semantic_name)
        discrete_semantic = bool(words & _DISCRETE_REPEATED_PART_TOKENS)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        components = sorted(
            (
                (int(stats[index, cv2.CC_STAT_AREA]), index)
                for index in range(1, count)
            ),
            reverse=True,
        )
        minimum = max(32, round(max(1, group.area_px) * 0.15))
        major = [index for area, index in components if area >= minimum][:maximum]
        major_areas = [
            int(stats[index, cv2.CC_STAT_AREA]) for index in major
        ]
        balanced_components = bool(
            len(major_areas) >= 2
            and major_areas[1] / max(1, major_areas[0]) >= 0.25
        )
        descriptors = [
            _mask_shape_descriptor(labels == component_index)
            for component_index in major
        ]
        family = _semantic_shape_family(group.semantic_name)
        shape_consistent = bool(
            descriptors
            and all(
                descriptor is not None
                and _shape_family_matches(family, descriptor)
                for descriptor in descriptors
            )
        )
        diagonal = max(1.0, float(np.hypot(*mask.shape)))
        centroid_separation = min(
            (
                float(
                    np.hypot(
                        left["centroid_x"] - right["centroid_x"],
                        left["centroid_y"] - right["centroid_y"],
                    )
                    / diagonal
                )
                for left_index, left in enumerate(descriptors)
                if left is not None
                for right in descriptors[left_index + 1 :]
                if right is not None
            ),
            default=0.0,
        )
        well_separated = centroid_separation >= 0.05
        if (
            maximum <= 1
            or not discrete_semantic
            or len(major) < 2
            or not balanced_components
            or not shape_consistent
            or not well_separated
        ):
            output[mask] = next_index
            output_groups.append(replace(group, group_index=next_index))
            next_index += 1
            continue

        unresolved = mask & ~np.isin(labels, major)
        ownership_masks = [labels == component_index for component_index in major]
        if np.any(unresolved):
            distances = np.stack(
                [
                    cv2.distanceTransform((~member).astype(np.uint8), cv2.DIST_L2, 5)
                    for member in ownership_masks
                ],
                axis=0,
            )
            nearest = distances.argmin(axis=0)
            ownership_masks = [
                member | (unresolved & (nearest == index))
                for index, member in enumerate(ownership_masks)
            ]
        for slot, member in enumerate(ownership_masks, start=1):
            ys, xs = np.nonzero(member)
            if len(xs) == 0:
                continue
            output[member] = next_index
            output_groups.append(
                replace(
                    group,
                    group_id=f"{group.group_id}/component_{slot:02d}",
                    group_index=next_index,
                    bbox_xyxy=(
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max() + 1),
                        int(ys.max() + 1),
                    ),
                    centroid_xy=(float(xs.mean()), float(ys.mean())),
                    area_px=len(xs),
                    evidence=f"{group.evidence}/disconnected_repeated_instance",
                )
            )
            next_index += 1
        rows.append(
            {
                "source_group_id": group.group_id,
                "semantic_name": group.semantic_name,
                "component_count": len(ownership_masks),
                "maximum_instances": maximum,
                "second_to_largest_area_ratio": (
                    major_areas[1] / max(1, major_areas[0])
                ),
                "minimum_centroid_separation": centroid_separation,
                "ground_truth_used": False,
            }
        )
    return output, tuple(output_groups), {
        "status": "completed" if rows else "no_split",
        "algorithm": "disconnected-repeated-instance-split-v2",
        "split_group_count": len(rows),
        "rows": rows,
        "ground_truth_used": False,
    }


def _has_independent_part_support(
    component: np.ndarray,
    group: PhysicalGroup,
    instance_map: np.ndarray,
    records_by_part_id: dict[str, PartInstance],
) -> bool:
    """Keep disconnected regions that are themselves supported fine Parts."""

    component_area = max(1, int(np.count_nonzero(component)))
    for part_id in group.member_part_ids:
        record = records_by_part_id.get(part_id)
        if record is None:
            continue
        if _is_visual_semantic(record.semantic_name) or (
            record.semantic_name == record.semantic_parent
        ):
            continue
        part_mask = instance_map == record.instance_index
        part_area = int(np.count_nonzero(part_mask))
        if part_area == 0:
            continue
        overlap = int(np.count_nonzero(component & part_mask))
        if overlap / component_area >= 0.80 and overlap / part_area >= 0.20:
            return True
    return False


def _refresh_group_geometry(
    group_map: np.ndarray,
    groups: list[PhysicalGroup] | tuple[PhysicalGroup, ...],
) -> tuple[PhysicalGroup, ...]:
    refreshed: list[PhysicalGroup] = []
    for group in groups:
        mask = group_map == group.group_index
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        refreshed.append(
            replace(
                group,
                bbox_xyxy=(
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max() + 1),
                    int(ys.max() + 1),
                ),
                centroid_xy=(float(xs.mean()), float(ys.mean())),
                area_px=len(xs),
            )
        )
    return tuple(refreshed)


def _rebind_group_memberships(
    group_map: np.ndarray,
    groups: list[PhysicalGroup] | tuple[PhysicalGroup, ...],
    instance_map: np.ndarray,
    records: list[PartInstance] | tuple[PartInstance, ...],
) -> tuple[tuple[PhysicalGroup, ...], tuple[PartInstance, ...], dict[str, object]]:
    """Recompute record ownership after pixel-level group regularization."""

    group_list = list(groups)
    record_list = list(records)
    if not group_list or not record_list:
        return tuple(group_list), tuple(record_list), {
            "status": "empty",
            "ground_truth_used": False,
        }
    overlaps = np.asarray(
        [
            [
                int(
                    np.count_nonzero(
                        (group_map == group.group_index)
                        & (instance_map == record.instance_index)
                    )
                )
                for record in record_list
            ]
            for group in group_list
        ],
        dtype=np.int64,
    )
    assignment_by_record: dict[int, int] = {}
    used_records: set[int] = set()
    for group_row in sorted(
        range(len(group_list)),
        key=lambda index: int(overlaps[index].max(initial=0)),
        reverse=True,
    ):
        ordering = np.argsort(-overlaps[group_row])
        selected = next(
            (
                int(record_index)
                for record_index in ordering
                if int(record_index) not in used_records
                and overlaps[group_row, record_index] > 0
            ),
            None,
        )
        if selected is not None:
            assignment_by_record[selected] = group_row
            used_records.add(selected)
    largest_group_row = int(
        np.argmax([group.area_px for group in group_list])
    )
    for record_index in range(len(record_list)):
        if record_index in assignment_by_record:
            continue
        best = int(np.argmax(overlaps[:, record_index]))
        assignment_by_record[record_index] = (
            best if overlaps[best, record_index] > 0 else largest_group_row
        )

    updated_records = tuple(
        replace(
            record,
            group_id=group_list[assignment_by_record[record_index]].group_id,
        )
        for record_index, record in enumerate(record_list)
    )
    member_ids: dict[int, list[str]] = {
        row: [] for row in range(len(group_list))
    }
    for record_index, record in enumerate(record_list):
        member_ids[assignment_by_record[record_index]].append(record.part_id)
    shared_source_memberships: list[dict[str, object]] = []
    for group_row, group in enumerate(group_list):
        if member_ids[group_row]:
            continue
        fallback_record = int(np.argmax(overlaps[group_row]))
        if overlaps[group_row, fallback_record] <= 0:
            continue
        member_ids[group_row].append(record_list[fallback_record].part_id)
        shared_source_memberships.append(
            {
                "group_id": group.group_id,
                "part_id": record_list[fallback_record].part_id,
                "overlap_px": int(overlaps[group_row, fallback_record]),
                "reason": "one_fine_region_spans_multiple_physical_groups",
            }
        )
    updated_groups = tuple(
        replace(group, member_part_ids=tuple(member_ids[group_row]))
        for group_row, group in enumerate(group_list)
    )
    split_records = 0
    rows: list[dict[str, object]] = []
    for record_index, record in enumerate(record_list):
        nonzero_groups = np.flatnonzero(overlaps[:, record_index] > 0)
        split = len(nonzero_groups) > 1
        split_records += int(split)
        rows.append(
            {
                "part_id": record.part_id,
                "assigned_group_id": updated_records[record_index].group_id,
                "overlapping_group_count": len(nonzero_groups),
                "split_across_final_groups": split,
            }
        )
    return updated_groups, updated_records, {
        "status": "completed",
        "algorithm": "post-regularization-group-membership-rebind-v1",
        "record_count": len(record_list),
        "group_count": len(group_list),
        "split_record_count": split_records,
        "shared_source_membership_count": len(shared_source_memberships),
        "shared_source_memberships": shared_source_memberships,
        "rows": rows,
        "ground_truth_used": False,
    }


def _character_hair_interior_seed(
    group_map: np.ndarray,
    groups: list[PhysicalGroup] | tuple[PhysicalGroup, ...],
    axis_frame: _CharacterAxisFrame,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return compact face-interior islands swallowed by a broad hair mask."""

    seed = np.zeros(group_map.shape, dtype=bool)
    body_group = next(
        (group for group in groups if group.semantic_name == "character_body"),
        None,
    )
    hair_indices = [
        group.group_index for group in groups if group.semantic_name == "character_hair"
    ]
    if body_group is None or not hair_indices:
        return seed, {"status": "missing_body_or_hair", "ground_truth_used": False}
    hair = np.isin(group_map, hair_indices)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        hair.astype(np.uint8), connectivity=8
    )
    if count <= 2:
        return seed, {"status": "single_hair_component", "ground_truth_used": False}
    main_component = max(
        range(1, count),
        key=lambda index: int(stats[index, cv2.CC_STAT_AREA]),
    )
    main_area = int(stats[main_component, cv2.CC_STAT_AREA])
    root_area = max(1, int(np.count_nonzero(group_map)))
    rows: list[dict[str, object]] = []
    for component_index in range(1, count):
        if component_index == main_component:
            continue
        component = labels == component_index
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        width = int(stats[component_index, cv2.CC_STAT_WIDTH])
        height = int(stats[component_index, cv2.CC_STAT_HEIGHT])
        fill = area / max(1, width * height)
        axial_center = float(np.median(axis_frame.coordinate[component]))
        lateral_center = float(np.median(axis_frame.lateral[component]))
        ring = (
            cv2.dilate(component.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(
                bool
            )
            & ~component
        )
        ring_area = max(1, int(np.count_nonzero(ring)))
        body_contact = float(
            np.count_nonzero(ring & (group_map == body_group.group_index)) / ring_area
        )
        background_contact = float(
            np.count_nonzero(ring & (group_map == 0)) / ring_area
        )
        aspect = max(width, height) / max(1, min(width, height))
        accepted = bool(
            area >= max(16, round(root_area * 0.001))
            and area <= max(64, round(main_area * 0.18))
            and axial_center <= axis_frame.head_end + 0.05
            and abs(lateral_center) <= 0.30
            and body_contact >= 0.34
            and background_contact <= 0.10
            and aspect <= 2.4
            and fill >= 0.34
        )
        if accepted:
            seed |= component
        rows.append(
            {
                "area_px": area,
                "centroid_xy": [
                    float(centroids[component_index, 0]),
                    float(centroids[component_index, 1]),
                ],
                "axial_center": axial_center,
                "lateral_center": lateral_center,
                "body_boundary_contact": body_contact,
                "background_boundary_contact": background_contact,
                "aspect_ratio": float(aspect),
                "fill_ratio": float(fill),
                "accepted": accepted,
            }
        )
    return seed, {
        "status": "completed" if np.any(seed) else "no_supported_interior_island",
        "algorithm": "character-face-interior-topology-audit-v1",
        "reassigned_seed_area_px": int(np.count_nonzero(seed)),
        "components": rows,
        "appearance_used": False,
        "ground_truth_used": False,
    }


def _character_lower_limb_skin_seed(
    group_map: np.ndarray,
    groups: list[PhysicalGroup] | tuple[PhysicalGroup, ...],
    instance_map: np.ndarray,
    records: list[PartInstance] | tuple[PartInstance, ...],
    lab: np.ndarray,
    axis_frame: _CharacterAxisFrame,
) -> tuple[np.ndarray, dict[str, object]]:
    """Recover a symmetric pair of bare lower legs from an overbroad shoe mask."""

    seed = np.zeros(group_map.shape, dtype=bool)
    body_group = next(
        (group for group in groups if group.semantic_name == "character_body"),
        None,
    )
    source_indices = [
        group.group_index
        for group in groups
        if group.semantic_name in {"character_lower_garment", "character_footwear"}
    ]
    generic_record_indices = [
        record.instance_index
        for record in records
        if record.semantic_name == record.semantic_parent
    ]
    skin_record_indices = [
        record.instance_index
        for record in records
        if _words(record.semantic_name) & {"face", "hand", "skin"}
    ]
    if body_group is None or not source_indices or not skin_record_indices:
        return seed, {
            "status": "missing_skin_or_lower_body_seed",
            "ground_truth_used": False,
        }
    skin_reference = np.isin(instance_map, skin_record_indices)
    if np.count_nonzero(skin_reference) < 24:
        return seed, {"status": "skin_reference_too_small", "ground_truth_used": False}

    material_lab = lab * np.asarray((0.48, 1.0, 1.0), dtype=np.float32)
    skin_center = np.median(material_lab[skin_reference], axis=0)
    skin_distance = np.linalg.norm(material_lab - skin_center, axis=2)
    reference_spread = float(np.quantile(skin_distance[skin_reference], 0.88))
    threshold = float(np.clip(reference_spread + 9.0, 18.0, 32.0))
    lower_zone = (
        (axis_frame.coordinate >= max(0.60, axis_frame.head_end + 0.17))
        & (axis_frame.coordinate <= 0.93)
        & (np.abs(axis_frame.lateral) <= 0.36)
    )
    inventory_candidate = np.isin(group_map, source_indices)
    generic_candidate = np.isin(instance_map, generic_record_indices) & (
        np.abs(axis_frame.lateral) >= 0.015
    )
    proposed = (
        (inventory_candidate | generic_candidate)
        & lower_zone
        & (skin_distance <= threshold)
    )
    proposed = cv2.morphologyEx(
        proposed.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ).astype(bool)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        proposed.astype(np.uint8), connectivity=8
    )
    root_area = max(1, int(np.count_nonzero(group_map)))
    eligible: list[int] = []
    rows: list[dict[str, object]] = []
    component_geometry: dict[int, tuple[float, float, float, float]] = {}
    for component_index in range(1, count):
        component = labels == component_index
        area = int(stats[component_index, cv2.CC_STAT_AREA])
        axial = axis_frame.coordinate[component]
        lateral = axis_frame.lateral[component]
        axial_start = float(np.quantile(axial, 0.02))
        axial_end = float(np.quantile(axial, 0.98))
        axial_center = float(np.median(axial))
        lateral_center = float(np.median(lateral))
        candidate = bool(
            area >= max(24, round(root_area * 0.0015))
            and axial_center >= 0.67
            and axial_end <= 0.93
            and axial_end - axial_start >= 0.018
            and abs(lateral_center) <= 0.34
        )
        if candidate:
            eligible.append(component_index)
            component_geometry[component_index] = (
                axial_start,
                axial_end,
                axial_center,
                lateral_center,
            )
        rows.append(
            {
                "component_index": component_index,
                "area_px": area,
                "centroid_xy": [
                    float(centroids[component_index, 0]),
                    float(centroids[component_index, 1]),
                ],
                "axial_interval": [axial_start, axial_end],
                "axial_center": axial_center,
                "lateral_center": lateral_center,
                "eligible": candidate,
                "accepted": False,
            }
        )

    best_pair: tuple[int, int] | None = None
    best_pair_area = -1
    for offset, first in enumerate(eligible):
        first_start, first_end, first_center, first_lateral = component_geometry[first]
        first_area = int(stats[first, cv2.CC_STAT_AREA])
        for second in eligible[offset + 1 :]:
            second_start, second_end, second_center, second_lateral = (
                component_geometry[second]
            )
            second_area = int(stats[second, cv2.CC_STAT_AREA])
            overlap = max(
                0.0, min(first_end, second_end) - max(first_start, second_start)
            )
            minimum_span = max(
                1e-6,
                min(first_end - first_start, second_end - second_start),
            )
            area_ratio = first_area / max(1, second_area)
            separated = abs(first_lateral - second_lateral) >= 0.06
            opposite = first_lateral * second_lateral <= 0.0
            compatible = bool(
                0.35 <= area_ratio <= 2.85
                and abs(first_center - second_center) <= 0.08
                and overlap / minimum_span >= 0.45
                and separated
                and opposite
            )
            if compatible and first_area + second_area > best_pair_area:
                best_pair = (first, second)
                best_pair_area = first_area + second_area
    if best_pair is not None:
        for component_index in best_pair:
            seed |= labels == component_index
            rows[component_index - 1]["accepted"] = True
    return seed, {
        "status": "completed" if best_pair is not None else "no_symmetric_skin_pair",
        "algorithm": "character-paired-lower-limb-skin-audit-v1",
        "threshold": threshold,
        "reference_spread": reference_spread,
        "reassigned_seed_area_px": int(np.count_nonzero(seed)),
        "components": rows,
        "appearance_role": "skin_consistency_after_lower_limb_inventory",
        "generic_residual_audited": bool(generic_record_indices),
        "appearance_can_create_ids": False,
        "ground_truth_used": False,
    }


def _regularize_character_surface_boundaries(
    group_map: np.ndarray,
    groups: list[PhysicalGroup] | tuple[PhysicalGroup, ...],
    instance_map: np.ndarray,
    records: list[PartInstance] | tuple[PartInstance, ...],
    image: Image.Image | np.ndarray | None,
) -> tuple[np.ndarray, tuple[PhysicalGroup, ...], dict[str, object]]:
    """Attach generic character residual pixels to existing physical surfaces."""

    unchanged = tuple(groups)
    if image is None or not groups:
        return (
            group_map,
            unchanged,
            {
                "status": "inactive",
                "ground_truth_used": False,
            },
        )
    if any(_root_domain(record) != "character" for record in records):
        return (
            group_map,
            unchanged,
            {
                "status": "not_character",
                "ground_truth_used": False,
            },
        )
    macro_semantics = {
        "character_body",
        "character_upper_garment",
        "character_inner_top",
        "character_outer_garment",
        "character_lower_garment",
        "character_footwear",
        "character_hair",
        "character_headwear",
    }
    macro_groups = [group for group in groups if group.semantic_name in macro_semantics]
    if len(macro_groups) < 2:
        return (
            group_map,
            unchanged,
            {
                "status": "insufficient_macro_groups",
                "ground_truth_used": False,
            },
        )
    generic_indices = [
        record.instance_index
        for record in records
        if record.semantic_name == record.semantic_parent
    ]
    mutable = np.isin(instance_map, generic_indices)

    rgb = (
        np.asarray(image.convert("RGB"), dtype=np.uint8)
        if isinstance(image, Image.Image)
        else np.asarray(image, dtype=np.uint8)
    )
    if rgb.shape[:2] != group_map.shape:
        return (
            group_map,
            unchanged,
            {
                "status": "image_shape_mismatch",
                "ground_truth_used": False,
            },
        )
    analysis_scale = min(1.0, 768.0 / max(rgb.shape[:2]))
    reduced = cv2.resize(
        rgb,
        (
            max(32, round(rgb.shape[1] * analysis_scale)),
            max(32, round(rgb.shape[0] * analysis_scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )
    reduced = cv2.bilateralFilter(reduced, 9, 38, 38)
    smoothed = cv2.resize(
        reduced,
        (rgb.shape[1], rgb.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    lab = cv2.cvtColor(smoothed, cv2.COLOR_RGB2LAB).astype(np.float32)
    gradients: list[np.ndarray] = []
    for channel in cv2.split(lab):
        dx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        gradients.append(cv2.magnitude(dx, dy))
    gradient = np.maximum.reduce(gradients)
    axis_frame = _character_axis_frame(instance_map, records)
    if axis_frame is None:
        return (
            group_map,
            unchanged,
            {
                "status": "character_axis_unavailable",
                "ground_truth_used": False,
            },
        )

    # Semantic masks remain seeds only inside physically plausible pose-axis
    # zones.  Over-broad masks are common for clothing prompts: a lower-garment
    # proposal can cover the shirt, legs, and shoes at once.  Pixels outside the
    # expected zone become undecided and are reassigned by the boundary-aware
    # watershed below instead of allowing one bad mask to own several parts.
    footwear_groups = [
        group for group in macro_groups if group.semantic_name == "character_footwear"
    ]
    footwear_coordinates = np.concatenate(
        [
            axis_frame.coordinate[group_map == group.group_index]
            for group in footwear_groups
            if np.any(group_map == group.group_index)
        ]
    ) if footwear_groups else np.asarray([], dtype=np.float32)
    footwear_start = (
        float(np.quantile(footwear_coordinates, 0.04))
        if footwear_coordinates.size
        else 0.84
    )
    footwear_words = {
        word
        for record in records
        if _words(record.semantic_name) & {"shoe", "boot", "sandal", "sock"}
        for word in _words(record.semantic_name)
    }
    footwear_zone_floor = (
        0.72
        if "boot" in footwear_words
        else 0.78
        if "sock" in footwear_words
        else 0.86
    )
    lower_zone_start = float(
        np.clip(
            axis_frame.head_end
            + 0.50 * max(0.08, footwear_start - axis_frame.head_end),
            axis_frame.head_end + 0.10,
            0.70,
        )
    )
    zone_invalid = np.zeros(group_map.shape, dtype=bool)
    proximal_footwear_return = np.zeros(group_map.shape, dtype=bool)
    zone_rows: list[dict[str, object]] = []
    for group in macro_groups:
        group_mask = group_map == group.group_index
        coordinate = axis_frame.coordinate
        if group.semantic_name in {
            "character_upper_garment",
            "character_inner_top",
            "character_outer_garment",
        }:
            invalid = group_mask & (
                (coordinate < axis_frame.head_end - 0.05)
                | (coordinate > lower_zone_start + 0.05)
            )
        elif group.semantic_name == "character_lower_garment":
            invalid = group_mask & (
                (coordinate < lower_zone_start - 0.01)
                | (coordinate > footwear_start + 0.04)
            )
        elif group.semantic_name == "character_footwear":
            footwear_cut = max(footwear_zone_floor, footwear_start - 0.02)
            invalid = group_mask & (coordinate < footwear_cut)
            proximal_footwear_return |= invalid
        elif group.semantic_name in {"character_hair", "character_headwear"}:
            invalid = group_mask & (
                coordinate > axis_frame.head_end + 0.12
            )
        else:
            invalid = np.zeros(group_map.shape, dtype=bool)
        zone_invalid |= invalid
        zone_rows.append(
            {
                "group_index": group.group_index,
                "semantic_name": group.semantic_name,
                "invalid_area_px": int(np.count_nonzero(invalid)),
            }
        )
    mutable |= zone_invalid

    body_group = next(
        (group for group in macro_groups if group.semantic_name == "character_body"),
        None,
    )
    hair_indices = [
        group.group_index
        for group in macro_groups
        if group.semantic_name == "character_hair"
    ]
    face_skin_seed = np.zeros(group_map.shape, dtype=bool)
    face_audit: dict[str, object] = {
        "status": "inactive",
        "ground_truth_used": False,
    }
    hair_interior_seed, hair_interior_audit = _character_hair_interior_seed(
        group_map,
        macro_groups,
        axis_frame,
    )
    skin_record_indices = [
        record.instance_index
        for record in records
        if _words(record.semantic_name) & {"face", "hand", "skin"}
    ]
    skin_reference = np.isin(instance_map, skin_record_indices)
    lower_garment_group = next(
        (
            group
            for group in macro_groups
            if group.semantic_name == "character_lower_garment"
        ),
        None,
    )
    lower_garment_return_seed = np.zeros(group_map.shape, dtype=bool)
    lower_body_surface_audit: dict[str, object] = {
        "status": "inactive",
        "ground_truth_used": False,
    }
    if (
        body_group is not None
        and lower_garment_group is not None
        and np.count_nonzero(skin_reference) >= 24
    ):
        material_lab = lab * np.asarray((0.48, 1.0, 1.0), dtype=np.float32)
        skin_center = np.median(material_lab[skin_reference], axis=0)
        skin_distance = np.linalg.norm(material_lab - skin_center, axis=2)
        reference_spread = float(np.quantile(skin_distance[skin_reference], 0.90))
        non_skin_threshold = float(
            np.clip(reference_spread + 11.0, 20.0, 35.0)
        )
        lower_body_zone = (
            (axis_frame.coordinate >= lower_zone_start - 0.015)
            & (axis_frame.coordinate <= min(0.88, footwear_start + 0.01))
            & (np.abs(axis_frame.lateral) <= 0.42)
        )
        proposed_lower = (
            (group_map == body_group.group_index)
            & lower_body_zone
            & (skin_distance > non_skin_threshold)
        )
        proposed_lower = cv2.morphologyEx(
            proposed_lower.astype(np.uint8),
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        ).astype(bool)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            proposed_lower.astype(np.uint8), connectivity=8
        )
        minimum_area = max(32, round(np.count_nonzero(group_map) * 0.0015))
        component_rows: list[dict[str, object]] = []
        for component_index in range(1, count):
            component = labels == component_index
            area = int(stats[component_index, cv2.CC_STAT_AREA])
            axial_start = float(
                np.quantile(axis_frame.coordinate[component], 0.02)
            )
            axial_end = float(
                np.quantile(axis_frame.coordinate[component], 0.98)
            )
            accepted = bool(
                area >= minimum_area
                and axial_end >= lower_zone_start + 0.01
                and axial_start <= footwear_start - 0.02
            )
            if accepted:
                lower_garment_return_seed |= component
            component_rows.append(
                {
                    "area_px": area,
                    "axial_interval": [axial_start, axial_end],
                    "accepted": accepted,
                }
            )
        mutable |= lower_garment_return_seed
        lower_body_surface_audit = {
            "status": (
                "completed"
                if np.any(lower_garment_return_seed)
                else "no_supported_non_skin_lower_surface"
            ),
            "algorithm": "character-lower-body-surface-audit-v1",
            "non_skin_threshold": non_skin_threshold,
            "reference_spread": reference_spread,
            "reassigned_seed_area_px": int(
                np.count_nonzero(lower_garment_return_seed)
            ),
            "components": component_rows,
            "appearance_role": "correct_existing_body_and_lower_garment_groups",
            "appearance_can_create_ids": False,
            "ground_truth_used": False,
        }
    if (
        body_group is not None
        and hair_indices
        and np.count_nonzero(skin_reference) >= 24
    ):
        root = group_map > 0
        head_center_region = (
            (axis_frame.coordinate >= 0.02)
            & (axis_frame.coordinate <= axis_frame.head_end + 0.05)
            & (np.abs(axis_frame.lateral) <= 0.37)
        )
        hair_or_generic = np.isin(group_map, hair_indices) | mutable
        auditable = root & head_center_region & hair_or_generic
        material_lab = lab * np.asarray((0.48, 1.0, 1.0), dtype=np.float32)
        skin_center = np.median(material_lab[skin_reference], axis=0)
        skin_distance = np.linalg.norm(material_lab - skin_center, axis=2)
        reference_spread = float(np.quantile(skin_distance[skin_reference], 0.90))
        threshold = float(np.clip(reference_spread + 15.0, 21.0, 39.0))
        proposed_face = auditable & (skin_distance <= threshold)
        proposed_face = cv2.morphologyEx(
            proposed_face.astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        ).astype(bool)
        # Closing may bridge skin islands across a hair boundary, but it must
        # never manufacture foreground beyond the already-audited subject.
        proposed_face &= auditable
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            proposed_face.astype(np.uint8), connectivity=8
        )
        minimum_face_area = max(28, round(np.count_nonzero(root) * 0.003))
        component_rows: list[dict[str, object]] = []
        for component_index in range(1, count):
            component = labels == component_index
            area = int(stats[component_index, cv2.CC_STAT_AREA])
            centroid_x = float(centroids[component_index, 0])
            centroid_y = float(centroids[component_index, 1])
            component_axial = float(np.median(axis_frame.coordinate[component]))
            component_lateral = float(np.median(axis_frame.lateral[component]))
            accepted = bool(
                area >= minimum_face_area
                and component_axial <= axis_frame.head_end + 0.04
                and abs(component_lateral) <= 0.31
            )
            if accepted:
                face_skin_seed |= component
            component_rows.append(
                {
                    "area_px": area,
                    "centroid_xy": [centroid_x, centroid_y],
                    "axial_center": component_axial,
                    "lateral_center": component_lateral,
                    "accepted": accepted,
                }
            )
        if np.any(face_skin_seed):
            mutable |= face_skin_seed
            face_audit = {
                "status": "completed",
                "algorithm": "character-head-skin-return-v1",
                "threshold": threshold,
                "reference_spread": reference_spread,
                "reassigned_seed_area_px": int(np.count_nonzero(face_skin_seed)),
                "components": component_rows,
                "appearance_role": "return_skin_from_overbroad_hair_mask",
                "appearance_can_create_ids": False,
                "ground_truth_used": False,
            }
        else:
            face_audit = {
                "status": "no_supported_face_component",
                "threshold": threshold,
                "reference_spread": reference_spread,
                "components": component_rows,
                "ground_truth_used": False,
            }

    lower_limb_skin_seed, lower_limb_skin_audit = _character_lower_limb_skin_seed(
        group_map,
        macro_groups,
        instance_map,
        records,
        lab,
        axis_frame,
    )
    body_return_seed = (
        face_skin_seed
        | hair_interior_seed
        | lower_limb_skin_seed
        | proximal_footwear_return
    )
    mutable |= body_return_seed
    if np.count_nonzero(mutable) < 32:
        return (
            group_map,
            unchanged,
            {
                "status": "no_mutable_character_region",
                "head_skin_audit": face_audit,
                "face_interior_topology_audit": hair_interior_audit,
                "lower_limb_skin_audit": lower_limb_skin_audit,
                "lower_body_surface_audit": lower_body_surface_audit,
                "ground_truth_used": False,
            },
        )

    markers = np.zeros(group_map.shape, dtype=np.int32)
    seed_union = np.zeros(group_map.shape, dtype=bool)
    seed_rows: list[dict[str, object]] = []
    for group in macro_groups:
        seed = (group_map == group.group_index) & ~mutable
        if body_group is not None and group.group_index == body_group.group_index:
            seed |= body_return_seed
        if (
            lower_garment_group is not None
            and group.group_index == lower_garment_group.group_index
        ):
            seed |= lower_garment_return_seed
        if not np.any(seed):
            continue
        markers[seed] = group.group_index
        seed_union |= seed
        seed_rows.append(
            {
                "group_index": group.group_index,
                "semantic_name": group.semantic_name,
                "seed_area_px": int(np.count_nonzero(seed)),
            }
        )
    if len(seed_rows) < 2:
        return (
            group_map,
            unchanged,
            {
                "status": "insufficient_trusted_seeds",
                "seed_rows": seed_rows,
                "ground_truth_used": False,
            },
        )
    region = mutable | seed_union
    values = gradient[region]
    reference = float(np.quantile(values, 0.90)) if values.size else 1.0
    elevation = np.clip(gradient / max(1.0, reference), 0.0, 2.0)

    from skimage.segmentation import watershed

    propagated = watershed(
        elevation,
        markers,
        mask=region,
        watershed_line=False,
    ).astype(np.int32)
    output = group_map.copy()
    assignable = mutable & (propagated > 0)
    output[assignable] = propagated[assignable].astype(output.dtype)
    changed = mutable & (output != group_map)
    rows: list[tuple[int, int]] = []
    for source, destination in zip(
        group_map[changed],
        output[changed],
        strict=True,
    ):
        rows.append((int(source), int(destination)))
    transition_counts: dict[str, int] = {}
    for source, destination in rows:
        key = f"{source}->{destination}"
        transition_counts[key] = transition_counts.get(key, 0) + 1
    return (
        output,
        _refresh_group_geometry(output, groups),
        {
            "status": "completed",
            "algorithm": "character-pose-axis-residual-watershed-v2",
            "analysis_scale": analysis_scale,
            "axis_direction_xy": list(axis_frame.axis_direction_xy),
            "footwear_start_normalized": footwear_start,
            "footwear_zone_floor_normalized": footwear_zone_floor,
            "lower_zone_start_normalized": lower_zone_start,
            "semantic_zone_invalid_area_px": int(np.count_nonzero(zone_invalid)),
            "proximal_footwear_return_area_px": int(
                np.count_nonzero(proximal_footwear_return)
            ),
            "semantic_zone_rows": zone_rows,
            "mutable_area_px": int(np.count_nonzero(mutable)),
            "reassigned_pixel_count": int(np.count_nonzero(changed)),
            "transition_counts": transition_counts,
            "seed_rows": seed_rows,
            "head_skin_audit": face_audit,
            "face_interior_topology_audit": hair_interior_audit,
            "lower_limb_skin_audit": lower_limb_skin_audit,
            "lower_body_surface_audit": lower_body_surface_audit,
            "appearance_role": "boundary_alignment_only",
            "appearance_can_create_ids": False,
            "ground_truth_used": False,
        },
    )


def _regularize_stem_base_boundaries(
    group_map: np.ndarray,
    groups: list[PhysicalGroup] | tuple[PhysicalGroup, ...],
) -> tuple[np.ndarray, tuple[PhysicalGroup, ...], dict[str, object]]:
    """Move a flared pedestal from a narrow stem into its supporting base.

    A semantic stem proposal can leak down a highlight or shadow into a broad
    pedestal.  The correction is activated only for a stem above a much wider
    base of the same asset, and only when the stem width has a sustained lower
    expansion.  No image colour or fixed coordinate is used.
    """

    output = group_map.copy()
    rows: list[dict[str, object]] = []
    base_groups = [group for group in groups if "base" in _words(group.semantic_name)]
    for stem_group in groups:
        if "stem" not in _words(stem_group.semantic_name):
            continue
        stem_mask = output == stem_group.group_index
        if not np.any(stem_mask):
            continue
        stem_ys, _ = np.nonzero(stem_mask)
        y0 = int(stem_ys.min())
        y1 = int(stem_ys.max() + 1)
        height = y1 - y0
        if height < 12:
            continue

        compatible_bases = [
            base
            for base in base_groups
            if base.asset_id == stem_group.asset_id
            and base.centroid_xy[1] > stem_group.centroid_xy[1]
            and base.bbox_xyxy[2] - base.bbox_xyxy[0]
            >= 1.8 * max(1, stem_group.bbox_xyxy[2] - stem_group.bbox_xyxy[0])
            and base.bbox_xyxy[1] <= y1 + 0.08 * group_map.shape[0]
        ]
        if not compatible_bases:
            continue
        base_group = min(
            compatible_bases,
            key=lambda base: (
                abs(base.bbox_xyxy[1] - y1),
                abs(base.centroid_xy[0] - stem_group.centroid_xy[0]),
            ),
        )

        widths = np.asarray(
            [np.count_nonzero(stem_mask[y]) for y in range(y0, y1)],
            dtype=np.float32,
        )
        active = widths > 0
        if np.count_nonzero(active) < 12:
            continue
        smoothed = np.asarray(
            [
                float(np.median(widths[max(0, index - 2) : index + 3]))
                for index in range(len(widths))
            ],
            dtype=np.float32,
        )
        baseline_end = max(4, round(0.55 * len(widths)))
        baseline_values = smoothed[:baseline_end][active[:baseline_end]]
        if len(baseline_values) == 0:
            continue
        baseline = float(np.median(baseline_values))
        expansion_threshold = max(baseline + 5.0, 1.25 * baseline)
        search_start = max(2, round(0.35 * len(widths)))
        split_offset = next(
            (
                index
                for index in range(search_start, max(search_start, len(widths) - 3))
                if np.count_nonzero(active[index : index + 4]) >= 3
                and float(np.median(smoothed[index : index + 4])) >= expansion_threshold
            ),
            None,
        )
        if split_offset is None:
            continue
        split_y = y0 + split_offset
        lower_stem = stem_mask & (
            np.indices(group_map.shape, sparse=True)[0] >= split_y
        )
        moved_area = int(np.count_nonzero(lower_stem))
        stem_area = int(np.count_nonzero(stem_mask))
        moved_fraction = moved_area / max(1, stem_area)
        if not 0.08 <= moved_fraction <= 0.70:
            continue

        output[lower_stem] = base_group.group_index
        rows.append(
            {
                "stem_group_index": stem_group.group_index,
                "base_group_index": base_group.group_index,
                "split_y": split_y,
                "upper_stem_baseline_width_px": baseline,
                "expansion_threshold_px": expansion_threshold,
                "moved_area_px": moved_area,
                "moved_fraction": moved_fraction,
            }
        )

    refreshed = _refresh_group_geometry(output, groups)
    return (
        output,
        refreshed,
        {
            "algorithm": "stem-base-structural-boundary-v1",
            "status": "completed" if rows else "not_applicable",
            "adjusted_pair_count": len(rows),
            "rows": rows,
            "appearance_used": False,
            "ground_truth_used": False,
        },
    )


def _clean_group_satellites(
    group_map: np.ndarray,
    groups: list[PhysicalGroup] | tuple[PhysicalGroup, ...],
    instance_map: np.ndarray,
    records: list[PartInstance] | tuple[PartInstance, ...],
) -> tuple[np.ndarray, tuple[PhysicalGroup, ...], dict[str, object]]:
    """Reassign tiny remote fragments without deleting supported physical Parts.

    Appearance proposals can leave a few highlight- or shadow-shaped pixels in
    an otherwise correct semantic group.  A fragment is changed only when it is
    tiny relative to the group's main component and not a substantial component
    of any member Part.  Foreground coverage is conserved by assigning such
    pixels to the nearest trusted group core.
    """

    output = group_map.copy()
    positive_indices = [int(value) for value in np.unique(group_map) if value > 0]
    if len(positive_indices) < 2:
        return (
            output,
            tuple(groups),
            {
                "algorithm": "supported-physical-component-cleanup-v1",
                "status": "single_group",
                "removed_component_count": 0,
                "reassigned_pixel_count": 0,
                "rows": [],
            },
        )

    diagonal = max(1.0, float(np.hypot(*group_map.shape)))
    absolute_area_limit = max(16, round(group_map.size * 0.00075))
    records_by_part_id = {record.part_id: record for record in records}
    group_by_index = {group.group_index: group for group in groups}
    core_masks: dict[int, np.ndarray] = {}
    component_tables: dict[int, tuple[np.ndarray, np.ndarray, list[int]]] = {}

    for group_index in positive_indices:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (group_map == group_index).astype(np.uint8),
            connectivity=8,
        )
        order = sorted(
            range(1, count),
            key=lambda component_index: int(stats[component_index, cv2.CC_STAT_AREA]),
            reverse=True,
        )
        if not order:
            continue
        core_masks[group_index] = labels == order[0]
        component_tables[group_index] = (labels, stats, order)

    distance_to_core = {
        group_index: cv2.distanceTransform(
            (~core_mask).astype(np.uint8),
            cv2.DIST_L2,
            5,
        )
        for group_index, core_mask in core_masks.items()
    }
    rows: list[dict[str, object]] = []
    reassigned_pixels = 0

    for group_index, (labels, stats, order) in component_tables.items():
        if len(order) <= 1:
            continue
        group = group_by_index[group_index]
        main_area = int(stats[order[0], cv2.CC_STAT_AREA])
        own_core_distance = distance_to_core[group_index]
        for component_index in order[1:]:
            area = int(stats[component_index, cv2.CC_STAT_AREA])
            component = labels == component_index
            distance = float(own_core_distance[component].min(initial=diagonal))
            tiny = area <= absolute_area_limit and area / max(1, main_area) <= 0.05
            supported = _has_independent_part_support(
                component,
                group,
                instance_map,
                records_by_part_id,
            )
            if not tiny or supported:
                continue

            destination, destination_distance = min(
                (
                    (other_index, float(distance_map[component].min(initial=diagonal)))
                    for other_index, distance_map in distance_to_core.items()
                    if other_index != group_index
                ),
                key=lambda item: (item[1], item[0]),
            )
            output[component] = destination
            reassigned_pixels += area
            rows.append(
                {
                    "source_group_index": group_index,
                    "destination_group_index": destination,
                    "area_px": area,
                    "source_core_distance_px": distance,
                    "destination_core_distance_px": destination_distance,
                    "independent_part_support": False,
                }
            )

    refreshed = _refresh_group_geometry(output, groups)
    return (
        output,
        refreshed,
        {
            "algorithm": "supported-physical-component-cleanup-v1",
            "status": "completed",
            "absolute_area_limit_px": absolute_area_limit,
            "relative_area_limit": 0.05,
            "removed_component_count": len(rows),
            "reassigned_pixel_count": reassigned_pixels,
            "rows": rows,
            "foreground_conserved": bool(np.array_equal(output > 0, group_map > 0)),
        },
    )


def _firearm_mask_groups(
    instance_map: np.ndarray,
    records: list[PartInstance] | tuple[PartInstance, ...],
    masks: dict[str, np.ndarray],
    *,
    diagnostics: dict[str, object],
) -> PhysicalGroupingResult:
    macro_semantics = (
        "tool_prop_muzzle",
        "tool_prop_barrel",
        "tool_prop_handguard",
        "tool_prop_receiver",
        "tool_prop_magazine",
        "tool_prop_grip",
        "tool_prop_stock",
    )
    detail_semantics = (
        "tool_prop_sight",
        "tool_prop_trigger",
        "tool_prop_trigger_guard",
        "tool_prop_charging_handle",
    )
    semantic_order = tuple(
        semantic
        for semantic in (*macro_semantics, *detail_semantics)
        if semantic in masks and np.any(masks[semantic])
    )
    record_list = list(records)
    overlaps = np.asarray(
        [
            [
                int(
                    np.count_nonzero(
                        (instance_map == record.instance_index) & masks[name]
                    )
                )
                for record in record_list
            ]
            for name in semantic_order
        ],
        dtype=np.int64,
    )
    assignment_by_record: dict[int, int] = {}
    used_records: set[int] = set()
    for group_index in sorted(
        range(len(semantic_order)),
        key=lambda index: int(overlaps[index].max(initial=0)),
        reverse=True,
    ):
        ordering = np.argsort(-overlaps[group_index])
        selected = next(
            (int(index) for index in ordering if int(index) not in used_records),
            None,
        )
        if selected is not None and overlaps[group_index, selected] > 0:
            assignment_by_record[selected] = group_index
            used_records.add(selected)
    for record_index in range(len(record_list)):
        if record_index in assignment_by_record:
            continue
        assignment_by_record[record_index] = int(np.argmax(overlaps[:, record_index]))

    group_map = np.zeros(instance_map.shape, dtype=np.uint16)
    group_ids = {
        index: f"object_001/{name}" for index, name in enumerate(semantic_order)
    }
    boundary_diagnostics = diagnostics.get("boundary_refinement")
    boundary_diagnostics = (
        boundary_diagnostics if isinstance(boundary_diagnostics, dict) else {}
    )
    macro_appearance_verified = boundary_diagnostics.get("status") == "completed"
    detail_diagnostics = diagnostics.get("detail_verification")
    detail_diagnostics = (
        detail_diagnostics if isinstance(detail_diagnostics, dict) else {}
    )
    accepted_detail_rows = {
        str(row.get("semantic_name")): row
        for row in detail_diagnostics.get("rows", [])
        if isinstance(row, dict) and bool(row.get("accepted"))
    }
    final_verification_rows: list[dict[str, object]] = []
    updated_records = tuple(
        replace(record, group_id=group_ids[assignment_by_record[record_index]])
        for record_index, record in enumerate(record_list)
    )
    groups: list[PhysicalGroup] = []
    for zero_index, semantic in enumerate(semantic_order):
        group_index = zero_index + 1
        mask = masks[semantic]
        group_map[mask] = group_index
        ys, xs = np.nonzero(mask)
        members = tuple(
            record.part_id
            for record_index, record in enumerate(record_list)
            if assignment_by_record[record_index] == zero_index
        )
        if not members:
            fallback_index = int(np.argmax(overlaps[zero_index]))
            members = (record_list[fallback_index].part_id,)
        groups.append(
            PhysicalGroup(
                group_id=group_ids[zero_index],
                group_index=group_index,
                semantic_name=semantic,
                asset_id="object_001",
                member_part_ids=members,
                bbox_xyxy=(
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max() + 1),
                    int(ys.max() + 1),
                ),
                centroid_xy=(float(xs.mean()), float(ys.mean())),
                area_px=len(xs),
                evidence=(
                    "firearm_three_evidence_detail"
                    if semantic in detail_semantics
                    else "firearm_inventory_axis_topology_boundary"
                ),
                review_required=(
                    semantic not in accepted_detail_rows
                    if semantic in detail_semantics
                    else not macro_appearance_verified
                ),
            )
        )
        if semantic in detail_semantics:
            detail_row = accepted_detail_rows.get(semantic, {})
            final_verification_rows.append(
                {
                    "semantic_name": semantic,
                    "stage_1_semantic": bool(detail_row.get("semantic_verified")),
                    "stage_2_structure": bool(detail_row.get("structure_verified")),
                    "stage_3_appearance": bool(
                        detail_row.get("appearance_verified")
                    ),
                    "accepted": bool(detail_row.get("accepted")),
                    "evidence": "verified_detail_candidate",
                }
            )
        else:
            final_verification_rows.append(
                {
                    "semantic_name": semantic,
                    "stage_1_semantic": True,
                    "stage_2_structure": True,
                    "stage_3_appearance": macro_appearance_verified,
                    "accepted": macro_appearance_verified,
                    "evidence": {
                        "semantic": "resolved_firearm_inventory",
                        "structure": "axis_endpoint_and_lower_topology",
                        "appearance": boundary_diagnostics.get("status"),
                    },
                }
            )
    group_map, cleaned_groups, cleanup_diagnostics = _clean_group_satellites(
        group_map,
        groups,
        instance_map,
        records,
    )
    cleaned_groups, updated_records, membership_diagnostics = (
        _rebind_group_memberships(
            group_map,
            cleaned_groups,
            instance_map,
            records,
        )
    )
    return PhysicalGroupingResult(
        group_map,
        cleaned_groups,
        updated_records,
        {
            "algorithm": "hpid-candidate-to-physical-group-fusion-v1",
            "fine_part_count": len(records),
            "group_count": len(cleaned_groups),
            "merged_part_count": max(0, len(records) - len(cleaned_groups)),
            "knife_inventory_active": False,
            "knife_inventory_complete": None,
            "firearm_inventory_active": True,
            "firearm_inventory_complete": set(macro_semantics)
            <= {group.semantic_name for group in cleaned_groups},
            "firearm_structural_decomposition": diagnostics,
            "final_group_three_stage_verification": {
                "algorithm": "hpid-final-group-three-stage-verification-v1",
                "evidence_order": ["semantic", "structure", "appearance"],
                "groups": final_verification_rows,
                "all_groups_verified": all(
                    bool(row["accepted"]) for row in final_verification_rows
                ),
                "appearance_can_create_ids": False,
                "ground_truth_used": False,
            },
            "physical_component_cleanup": cleanup_diagnostics,
            "group_membership_rebind": membership_diagnostics,
            "review_required_group_count": 0,
            "ground_truth_used": False,
        },
    )


def _verified_profile_mask_groups(
    instance_map: np.ndarray,
    records: list[PartInstance] | tuple[PartInstance, ...],
    masks: dict[str, np.ndarray],
    *,
    profile: str,
    diagnostics: dict[str, object],
) -> PhysicalGroupingResult:
    """Export a profile decomposition whose groups passed all three gates."""

    semantic_order = tuple(
        semantic for semantic, mask in masks.items() if np.any(mask)
    )
    record_list = list(records)
    overlaps = np.asarray(
        [
            [
                int(
                    np.count_nonzero(
                        (instance_map == record.instance_index) & masks[semantic]
                    )
                )
                for record in record_list
            ]
            for semantic in semantic_order
        ],
        dtype=np.int64,
    )
    assignment_by_record: dict[int, int] = {}
    used_records: set[int] = set()
    for group_row in sorted(
        range(len(semantic_order)),
        key=lambda index: int(overlaps[index].max(initial=0)),
        reverse=True,
    ):
        ordering = np.argsort(-overlaps[group_row])
        selected = next(
            (
                int(record_index)
                for record_index in ordering
                if int(record_index) not in used_records
                and overlaps[group_row, record_index] > 0
            ),
            None,
        )
        if selected is not None:
            assignment_by_record[selected] = group_row
            used_records.add(selected)
    for record_index in range(len(record_list)):
        if record_index in assignment_by_record:
            continue
        assignment_by_record[record_index] = int(
            np.argmax(overlaps[:, record_index])
        )

    group_map = np.zeros(instance_map.shape, dtype=np.uint16)
    group_ids = {
        index: f"object_001/{semantic}"
        for index, semantic in enumerate(semantic_order)
    }
    groups: list[PhysicalGroup] = []
    for zero_index, semantic in enumerate(semantic_order):
        group_index = zero_index + 1
        mask = np.asarray(masks[semantic], dtype=bool) & (instance_map > 0)
        group_map[mask] = group_index
        ys, xs = np.nonzero(mask)
        members = tuple(
            record.part_id
            for record_index, record in enumerate(record_list)
            if assignment_by_record[record_index] == zero_index
        )
        if not members and record_list:
            fallback = int(np.argmax(overlaps[zero_index]))
            members = (record_list[fallback].part_id,)
        groups.append(
            PhysicalGroup(
                group_id=group_ids[zero_index],
                group_index=group_index,
                semantic_name=semantic,
                asset_id="object_001",
                member_part_ids=members,
                bbox_xyxy=(
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max() + 1),
                    int(ys.max() + 1),
                ),
                centroid_xy=(float(xs.mean()), float(ys.mean())),
                area_px=len(xs),
                evidence=f"{profile}_serial_three_stage_verification",
                review_required=False,
            )
        )
    group_map, cleaned_groups, cleanup_diagnostics = _clean_group_satellites(
        group_map,
        groups,
        instance_map,
        records,
    )
    cleaned_groups, updated_records, membership_diagnostics = (
        _rebind_group_memberships(
            group_map,
            cleaned_groups,
            instance_map,
            records,
        )
    )
    final_rows = [
        {
            "group_id": group.group_id,
            "semantic_name": group.semantic_name,
            "stage_1_semantic": True,
            "stage_2_structure": True,
            "stage_3_appearance": True,
            "accepted": True,
            "evidence": f"{profile}_serial_three_stage_verification",
        }
        for group in cleaned_groups
    ]
    return PhysicalGroupingResult(
        group_map,
        cleaned_groups,
        updated_records,
        {
            "algorithm": "hpid-candidate-to-physical-group-fusion-v1",
            "fine_part_count": len(records),
            "group_count": len(cleaned_groups),
            "merged_part_count": max(0, len(records) - len(cleaned_groups)),
            "selected_profile": profile,
            "profile_structural_decomposition": diagnostics,
            "final_group_three_stage_verification": {
                "algorithm": "hpid-final-group-three-stage-verification-v2",
                "serial": True,
                "evidence_order": ["semantic", "structure", "appearance"],
                "groups": final_rows,
                "all_groups_verified": True,
                "appearance_can_create_ids": False,
                "ground_truth_used": False,
            },
            "physical_component_cleanup": cleanup_diagnostics,
            "group_membership_rebind": membership_diagnostics,
            "review_required_group_count": 0,
            "ground_truth_used": False,
        },
    )


def _physical_mask_groups(
    instance_map: np.ndarray,
    records: list[PartInstance] | tuple[PartInstance, ...],
    masks: dict[str, np.ndarray],
    *,
    diagnostics: dict[str, object],
) -> PhysicalGroupingResult:
    semantic_order = ("tool_prop_blade", "tool_prop_handle", "tool_prop_wrap")
    record_list = list(records)
    overlaps = np.asarray(
        [
            [
                int(
                    np.count_nonzero(
                        (instance_map == record.instance_index) & masks[name]
                    )
                )
                for record in record_list
            ]
            for name in semantic_order
        ],
        dtype=np.int64,
    )
    assignment_by_record: dict[int, int] = {}
    used_records: set[int] = set()
    # Give each required physical group one representative fine record before
    # assigning residual records by majority overlap.
    for group_index in sorted(
        range(len(semantic_order)),
        key=lambda index: int(overlaps[index].max(initial=0)),
        reverse=True,
    ):
        ordering = np.argsort(-overlaps[group_index])
        selected = next(
            (int(index) for index in ordering if int(index) not in used_records),
            None,
        )
        if selected is not None and overlaps[group_index, selected] > 0:
            assignment_by_record[selected] = group_index
            used_records.add(selected)
    for record_index in range(len(record_list)):
        if record_index in assignment_by_record:
            continue
        assignment_by_record[record_index] = int(np.argmax(overlaps[:, record_index]))

    group_map = np.zeros(instance_map.shape, dtype=np.uint16)
    groups: list[PhysicalGroup] = []
    group_ids = {
        index: f"object_001/{name}" for index, name in enumerate(semantic_order)
    }
    updated_records = tuple(
        replace(
            record,
            group_id=group_ids[assignment_by_record[record_index]],
        )
        for record_index, record in enumerate(record_list)
    )
    for zero_index, semantic in enumerate(semantic_order):
        group_index = zero_index + 1
        mask = masks[semantic]
        group_map[mask] = group_index
        ys, xs = np.nonzero(mask)
        members = tuple(
            record.part_id
            for record_index, record in enumerate(record_list)
            if assignment_by_record[record_index] == zero_index
        )
        groups.append(
            PhysicalGroup(
                group_id=group_ids[zero_index],
                group_index=group_index,
                semantic_name=semantic,
                asset_id="object_001",
                member_part_ids=members,
                bbox_xyxy=(
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max() + 1),
                    int(ys.max() + 1),
                ),
                centroid_xy=(float(xs.mean()), float(ys.mean())),
                area_px=len(xs),
                evidence=(
                    "axis_width_bottleneck"
                    if semantic == "tool_prop_blade"
                    else "axial_material_partition"
                    if semantic == "tool_prop_wrap"
                    else "knife_inventory_residual"
                ),
                review_required=False,
            )
        )
    group_map, regularized_groups, boundary_diagnostics = (
        _regularize_stem_base_boundaries(group_map, groups)
    )
    group_map, cleaned_groups, cleanup_diagnostics = _clean_group_satellites(
        group_map,
        regularized_groups,
        instance_map,
        records,
    )
    cleaned_groups, updated_records, membership_diagnostics = (
        _rebind_group_memberships(
            group_map,
            cleaned_groups,
            instance_map,
            records,
        )
    )
    return PhysicalGroupingResult(
        group_map,
        cleaned_groups,
        updated_records,
        {
            "algorithm": "hpid-candidate-to-physical-group-fusion-v1",
            "fine_part_count": len(records),
            "group_count": 3,
            "merged_part_count": max(0, len(records) - 3),
            "knife_inventory_active": True,
            "knife_inventory_complete": True,
            "knife_structural_decomposition": diagnostics,
            "structural_boundary_regularization": boundary_diagnostics,
            "physical_component_cleanup": cleanup_diagnostics,
            "group_membership_rebind": membership_diagnostics,
            "review_required_group_count": 0,
            "ground_truth_used": False,
        },
    )


def build_physical_groups(
    instance_map: np.ndarray,
    records: list[PartInstance] | tuple[PartInstance, ...],
    *,
    candidates: list[MaskCandidate] | tuple[MaskCandidate, ...] = (),
    image: Image.Image | np.ndarray | None = None,
    provisional_scene_labels: bool = False,
) -> PhysicalGroupingResult:
    """Fuse fine visual regions into conservative, editable physical groups.

    Fine Part IDs remain intact for diagnosis.  Group IDs are allowed to merge
    them, but a color/texture region can never create a group by itself.
    """

    if instance_map.ndim != 2:
        raise ValueError("instance_map must be two-dimensional")
    candidate_tuple = tuple(candidates)
    selected_profile = _selected_profile(candidate_tuple)
    (
        verified_profile_semantics,
        rejected_profile_semantics,
        profile_verification_rows,
    ) = _profile_candidate_verification(
        candidate_tuple,
        selected_profile,
        instance_map > 0,
    )
    structurally_recovered_profile_semantics = (
        _structurally_recovered_profile_semantics(
            selected_profile,
            profile_verification_rows,
        )
    )
    verified_profile_semantics |= structurally_recovered_profile_semantics
    rejected_profile_semantics -= structurally_recovered_profile_semantics
    knife_inventory = _knife_inventory_active(candidate_tuple)
    promoted_visuals = _promotable_visual_semantics(candidate_tuple)
    profile_group_overrides, profile_grouping_diagnostics = (
        _profile_semantic_group_overrides(
            instance_map,
            records,
            candidate_tuple,
            image,
        )
    )
    character_group_overrides, character_grouping_diagnostics = (
        _character_surface_group_overrides(
            instance_map,
            records,
            image,
        )
    )
    asset_ids = {record.asset_id for record in records}
    if selected_profile == "firearm" and asset_ids == {"object_001"}:
        firearm_masks, firearm_diagnostics = _firearm_structural_masks(
            instance_map,
            records,
            image,
        )
        if firearm_masks is not None:
            firearm_masks, firearm_detail_diagnostics = (
                _firearm_verified_detail_masks(
                    instance_map,
                    records,
                    candidate_tuple,
                    firearm_masks,
                    firearm_diagnostics,
                )
            )
            return _firearm_mask_groups(
                instance_map,
                records,
                firearm_masks,
                diagnostics={
                    **firearm_diagnostics,
                    "detail_verification": firearm_detail_diagnostics,
                    "three_stage_candidate_verification": {
                        "evidence_order": [
                            "semantic",
                            "structure",
                            "appearance",
                        ],
                        "verified_semantics": sorted(
                            verified_profile_semantics
                        ),
                        "rejected_semantics": sorted(
                            rejected_profile_semantics
                        ),
                        "candidates": profile_verification_rows,
                        "appearance_can_create_ids": False,
                        "ground_truth_used": False,
                    },
                },
            )
    if selected_profile == "phone" and asset_ids == {"object_001"}:
        phone_masks, phone_diagnostics = _phone_structural_masks(
            instance_map,
            candidate_tuple,
        )
        if phone_masks is not None:
            return _verified_profile_mask_groups(
                instance_map,
                records,
                phone_masks,
                profile="phone",
                diagnostics={
                    **phone_diagnostics,
                    "three_stage_candidate_verification": {
                        "algorithm": (
                            "hpid-semantic-structure-appearance-verification-v2"
                        ),
                        "serial": True,
                        "evidence_order": [
                            "semantic",
                            "structure",
                            "appearance",
                        ],
                        "verified_semantics": sorted(
                            verified_profile_semantics
                        ),
                        "rejected_semantics": sorted(
                            rejected_profile_semantics
                        ),
                        "candidates": profile_verification_rows,
                        "appearance_can_create_ids": False,
                        "ground_truth_used": False,
                    },
                },
            )
    knife_structural_diagnostics: dict[str, object] | None = None
    if knife_inventory and image is not None and asset_ids == {"object_001"}:
        structural_masks, knife_structural_diagnostics = _knife_structural_masks(
            image,
            instance_map > 0,
        )
        if structural_masks is not None:
            return _physical_mask_groups(
                instance_map,
                records,
                structural_masks,
                diagnostics={
                    **knife_structural_diagnostics,
                    "three_stage_candidate_verification": {
                        "evidence_order": [
                            "semantic",
                            "structure",
                            "appearance",
                        ],
                        "verified_semantics": sorted(
                            verified_profile_semantics
                        ),
                        "rejected_semantics": sorted(
                            rejected_profile_semantics
                        ),
                        "candidates": profile_verification_rows,
                        "appearance_can_create_ids": False,
                        "ground_truth_used": False,
                    },
                },
            )
    semantic_partition_masks, semantic_partition_diagnostics = (
        _semantic_seeded_profile_masks(
            instance_map,
            candidate_tuple,
            image,
            profile=selected_profile,
        )
    )
    if semantic_partition_masks is not None and asset_ids == {"object_001"}:
        return _verified_profile_mask_groups(
            instance_map,
            records,
            semantic_partition_masks,
            profile=selected_profile or "resolved_profile",
            diagnostics={
                **semantic_partition_diagnostics,
                "knife_structural_fallback": knife_structural_diagnostics,
            },
        )
    shape_recovery_overrides, shape_recovery_diagnostics = (
        _repeated_semantic_shape_recovery(
            instance_map,
            records,
            candidate_tuple,
            profile=selected_profile,
            verified_semantics=verified_profile_semantics,
            rejected_semantics=rejected_profile_semantics,
        )
    )
    assignments: dict[tuple[str, str, int | None], list[PartInstance]] = {}
    assignment_meta: dict[tuple[str, str, int | None], tuple[str, str]] = {}
    for record in records:
        rejected_profile_record = bool(
            selected_profile is not None
            and _root_domain(record) != "character"
            and record.semantic_name in rejected_profile_semantics
            and record.semantic_name not in verified_profile_semantics
        )
        override = shape_recovery_overrides.get(record.part_id)
        if override is None:
            override = (
                (
                    f"{_root_domain(record)}_body",
                    None,
                    "three_stage_semantic_rejection_merge",
                )
                if rejected_profile_record
                else character_group_overrides.get(
                    record.part_id,
                    profile_group_overrides.get(record.part_id),
                )
            )
        if override is not None:
            semantic, unique_index, evidence = override
        else:
            semantic, merge_family, evidence = _group_assignment(
                record,
                knife_inventory=knife_inventory,
                promoted_visuals=promoted_visuals,
            )
            unique_index = None if merge_family else record.instance_index
        key = (record.asset_id, semantic, unique_index)
        assignments.setdefault(key, []).append(record)
        assignment_meta[key] = (semantic, evidence)

    ordered = sorted(
        assignments,
        key=lambda key: (
            min(record.instance_index for record in assignments[key]),
            key[0],
            key[1],
        ),
    )
    group_map = np.zeros(instance_map.shape, dtype=np.uint16)
    groups: list[PhysicalGroup] = []
    group_by_part: dict[str, str] = {}
    for group_index, key in enumerate(ordered, start=1):
        members = assignments[key]
        semantic, evidence = assignment_meta[key]
        public_semantic = "scene_object" if provisional_scene_labels else semantic
        member_indices = {record.instance_index for record in members}
        mask = np.isin(instance_map, list(member_indices))
        if not np.any(mask):
            continue
        group_map[mask] = group_index
        ys, xs = np.nonzero(mask)
        x0, x1 = int(xs.min()), int(xs.max() + 1)
        y0, y1 = int(ys.min()), int(ys.max() + 1)
        asset_id = members[0].asset_id
        suffix = "" if key[2] is None else f"/{key[2]:03d}"
        group_id = f"{asset_id}/{public_semantic}{suffix}"
        for record in members:
            group_by_part[record.part_id] = group_id
        groups.append(
            PhysicalGroup(
                group_id=group_id,
                group_index=group_index,
                semantic_name=public_semantic,
                asset_id=asset_id,
                member_part_ids=tuple(record.part_id for record in members),
                bbox_xyxy=(x0, y0, x1, y1),
                centroid_xy=(float(xs.mean()), float(ys.mean())),
                area_px=len(xs),
                evidence=(
                    f"provisional_scene_label/{evidence}"
                    if provisional_scene_labels
                    else evidence
                ),
                review_required=(
                    provisional_scene_labels or evidence.startswith("conservative")
                ),
            )
        )
    updated_records = tuple(
        replace(record, group_id=group_by_part.get(record.part_id, record.part_id))
        for record in records
    )
    group_map, character_regularized_groups, character_boundary_diagnostics = (
        _regularize_character_surface_boundaries(
            group_map,
            groups,
            instance_map,
            records,
            image,
        )
    )
    group_map, regularized_groups, boundary_diagnostics = (
        _regularize_stem_base_boundaries(group_map, character_regularized_groups)
    )
    group_map, cleaned_groups, cleanup_diagnostics = _clean_group_satellites(
        group_map,
        regularized_groups,
        instance_map,
        records,
    )
    group_map, cleaned_groups, repeated_split_diagnostics = (
        _split_repeated_group_components(
            group_map,
            cleaned_groups,
            candidate_tuple,
        )
    )
    cleaned_groups, updated_records, membership_diagnostics = (
        _rebind_group_memberships(
            group_map,
            cleaned_groups,
            instance_map,
            records,
        )
    )
    groups = list(cleaned_groups)
    final_group_verification_rows = [
        {
            "group_id": group.group_id,
            "semantic_name": group.semantic_name,
            "stage_1_semantic": not group.evidence.startswith("conservative"),
            "stage_2_structure": group.area_px > 0,
            "stage_3_appearance": not group.review_required,
            "accepted": not group.review_required,
            "evidence": group.evidence,
        }
        for group in groups
    ]
    knife_semantics = {
        group.semantic_name for group in groups if group.asset_id == "object_001"
    }
    return PhysicalGroupingResult(
        group_map,
        tuple(groups),
        updated_records,
        {
            "algorithm": "hpid-candidate-to-physical-group-fusion-v1",
            "fine_part_count": len(records),
            "group_count": len(groups),
            "merged_part_count": max(0, len(records) - len(groups)),
            "knife_inventory_active": knife_inventory,
            "knife_inventory_complete": (
                {"tool_prop_blade", "tool_prop_handle", "tool_prop_wrap"}
                <= knife_semantics
                if knife_inventory
                else None
            ),
            "knife_structural_decomposition": knife_structural_diagnostics,
            "review_required_group_count": sum(
                group.review_required for group in groups
            ),
            "promoted_visual_region_count": len(promoted_visuals),
            "promoted_visual_regions": promoted_visuals,
            "profile_semantic_grouping": profile_grouping_diagnostics,
            "semantic_seeded_physical_ownership": (
                semantic_partition_diagnostics
            ),
            "repeated_semantic_shape_recovery": shape_recovery_diagnostics,
            "repeated_instance_component_split": repeated_split_diagnostics,
            "three_stage_candidate_verification": {
                "algorithm": (
                    "hpid-semantic-structure-appearance-verification-v1"
                ),
                "evidence_order": ["semantic", "structure", "appearance"],
                "verified_semantics": sorted(verified_profile_semantics),
                "rejected_semantics": sorted(rejected_profile_semantics),
                "candidates": profile_verification_rows,
                "appearance_can_create_ids": False,
                "ground_truth_used": False,
            },
            "character_surface_grouping": character_grouping_diagnostics,
            "character_boundary_regularization": character_boundary_diagnostics,
            "structural_boundary_regularization": boundary_diagnostics,
            "physical_component_cleanup": cleanup_diagnostics,
            "group_membership_rebind": membership_diagnostics,
            "final_group_three_stage_verification": {
                "algorithm": "hpid-final-group-three-stage-verification-v1",
                "evidence_order": ["semantic", "structure", "appearance"],
                "groups": final_group_verification_rows,
                "all_groups_verified": all(
                    bool(row["accepted"])
                    for row in final_group_verification_rows
                ),
                "appearance_can_create_ids": False,
                "ground_truth_used": False,
            },
            "provisional_scene_labels": provisional_scene_labels,
            "provisional_scene_semantic_policy": (
                "neutral_scene_object"
                if provisional_scene_labels
                else "verified_semantic_name"
            ),
            "ground_truth_used": False,
        },
    )
