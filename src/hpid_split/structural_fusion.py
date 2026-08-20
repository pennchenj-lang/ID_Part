from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from .fusion import MaskCandidate, mask_iou
from .prompt_bank import DomainPrompt, PartProfile, PartPrompt


@dataclass(frozen=True)
class StructuralFusionConfig:
    minimum_strip_elongation: float = 8.0
    minimum_strip_root_fraction: float = 0.018
    maximum_strip_root_fraction: float = 0.72
    minimum_residual_root_fraction: float = 0.06
    maximum_residual_root_fraction: float = 0.90
    minimum_structural_coverage: float = 0.70
    minimum_axial_rank: float = 0.52
    axial_profile_bin_count: int = 32
    minimum_root_axial_elongation: float = 12.0
    minimum_generated_strip_elongation: float = 12.0
    minimum_axial_width_ratio: float = 1.38
    minimum_axial_partition_score: float = 0.40
    minimum_axial_terminal_fraction: float = 0.06
    maximum_axial_terminal_fraction: float = 0.88
    minimum_axial_residual_fraction: float = 0.12
    minimum_incumbent_effective_score: float = 0.42
    minimum_planar_tile_root_fraction: float = 0.08
    minimum_planar_union_root_fraction: float = 0.55
    minimum_planar_pair_containment: float = 0.12
    source_reliability: float = 0.74


@dataclass(frozen=True)
class StructuralFusionResult:
    candidates: tuple[MaskCandidate, ...]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class _MaskShape:
    area: int
    root_fraction: float
    elongation: float
    bbox_aspect: float
    direction_xy: tuple[float, float]


@dataclass(frozen=True)
class _AxialPartition:
    wide_mask: np.ndarray
    narrow_mask: np.ndarray
    score: float
    split_fraction: float
    wide_at_low_axis_end: bool
    width_ratio: float
    neck_ratio: float
    local_width_contrast: float
    root_elongation: float
    wide_fraction: float
    narrow_fraction: float
    wide_elongation: float
    narrow_elongation: float


_STRIP_TOKENS = {
    "arm",
    "band",
    "barrel",
    "blade",
    "cable",
    "chisel",
    "handle",
    "leg",
    "mast",
    "neck",
    "rail",
    "rod",
    "shaft",
    "stem",
    "stock",
    "tube",
}

_MASS_SUFFIXES = (
    "_bowl",
    "_brush",
    "_head",
    "_pan_body",
)

_MASS_PRIORITY = {
    "_pan_body": 4,
    "_bowl": 3,
    "_head": 2,
    "_brush": 1,
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


def _shape(mask: np.ndarray, root_area: int) -> _MaskShape:
    ys, xs = np.nonzero(mask)
    area = len(xs)
    if not area:
        return _MaskShape(0, 0.0, 1.0, 1.0, (1.0, 0.0))
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    bbox_aspect = max(width / max(1, height), height / max(1, width))
    elongation = 1.0
    direction = np.asarray([1.0, 0.0], dtype=np.float64)
    if area >= 8:
        points = np.column_stack((xs, ys)).astype(np.float64)
        eigenvalues, eigenvectors = np.linalg.eigh(np.cov(points.T))
        elongation = float(eigenvalues[-1] / max(1e-6, eigenvalues[0]))
        direction = eigenvectors[:, -1]
    return _MaskShape(
        area=area,
        root_fraction=area / max(1, root_area),
        elongation=elongation,
        bbox_aspect=float(bbox_aspect),
        direction_xy=(float(direction[0]), float(direction[1])),
    )


def _semantic_tokens(part: PartPrompt) -> set[str]:
    return set(part.semantic_name.casefold().split("_"))


def _is_strip_part(part: PartPrompt) -> bool:
    return bool(_semantic_tokens(part) & _STRIP_TOKENS)


def _is_mass_part(part: PartPrompt) -> bool:
    return part.semantic_name.casefold().endswith(_MASS_SUFFIXES)


def _mass_priority(part: PartPrompt) -> int:
    name = part.semantic_name.casefold()
    return max(
        (priority for suffix, priority in _MASS_PRIORITY.items() if name.endswith(suffix)),
        default=0,
    )


def _selected_parts(
    root: MaskCandidate, domain: DomainPrompt
) -> tuple[tuple[PartPrompt, ...], str | None]:
    label = str(
        root.metadata.get("resolved_object_label")
        or root.metadata.get("root_model_label")
        or root.prompt
        or domain.name
    )
    profile_hint = root.metadata.get("selected_part_profile")
    parts, profile, _ = domain.select_parts(
        label,
        profile_hint=(str(profile_hint) if profile_hint else None),
        profile_hint_source=("resolved_root_profile" if profile_hint else None),
    )
    return parts, profile


def _structural_pair(
    root: MaskCandidate, domain: DomainPrompt
) -> tuple[PartPrompt, PartPrompt, str | None] | None:
    parts, profile = _selected_parts(root, domain)
    structural = tuple(
        part
        for part in parts
        if not part.detail
        and part.maximum_instances <= 2
        and part.semantic_name != f"{root.semantic_name}_body"
        and part.appearance_relation is None
    )
    strips = tuple(part for part in structural if _is_strip_part(part))
    masses = tuple(part for part in structural if _is_mass_part(part))
    if len(strips) != 1 or not masses:
        return None
    ranked_masses = sorted(
        masses,
        key=lambda part: (_mass_priority(part), part.priority),
        reverse=True,
    )
    if len(ranked_masses) > 1 and _mass_priority(
        ranked_masses[0]
    ) == _mass_priority(ranked_masses[1]):
        return None
    return strips[0], ranked_masses[0], profile


def _wide_narrow_pair(
    root: MaskCandidate, domain: DomainPrompt
) -> tuple[PartPrompt, PartPrompt, str | None] | None:
    pair = _structural_pair(root, domain)
    if pair is not None:
        narrow_part, wide_part, profile = pair
        return wide_part, narrow_part, profile

    parts, profile = _selected_parts(root, domain)
    if profile != "screwdriver":
        return None
    handles = [
        part
        for part in parts
        if not part.detail and part.semantic_name.casefold().endswith("_handle")
    ]
    shafts = [
        part
        for part in parts
        if not part.detail and part.semantic_name.casefold().endswith("_shaft")
    ]
    if len(handles) != 1 or len(shafts) != 1:
        return None
    return handles[0], shafts[0], profile


def _axial_width_profile(
    mask: np.ndarray,
    bin_count: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) < max(24, bin_count):
        return None
    points = np.column_stack((xs, ys)).astype(np.float64)
    center = points.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov((points - center).T))
    root_elongation = float(eigenvalues[-1] / max(1e-6, eigenvalues[0]))
    direction = eigenvectors[:, -1]
    perpendicular = np.asarray([-direction[1], direction[0]], dtype=np.float64)
    axial = (points - center) @ direction
    transverse = (points - center) @ perpendicular
    axial_span = float(axial.max() - axial.min())
    if axial_span < 8.0:
        return None
    normalized = (axial - axial.min()) / axial_span
    counts = np.zeros(bin_count, dtype=np.int64)
    widths = np.zeros(bin_count, dtype=np.float64)
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        selected = (normalized >= lower) & (
            normalized < upper if index < bin_count - 1 else normalized <= upper
        )
        counts[index] = int(np.count_nonzero(selected))
        if counts[index] >= 2:
            segment = transverse[selected]
            widths[index] = float(
                np.quantile(segment, 0.95) - np.quantile(segment, 0.05)
            )
    populated = np.flatnonzero(widths > 0.0)
    if len(populated) < max(4, bin_count // 4):
        return None
    widths = np.interp(np.arange(bin_count), populated, widths[populated])
    widths = np.convolve(
        np.pad(widths, (1, 1), mode="edge"),
        np.ones(3, dtype=np.float64) / 3.0,
        mode="valid",
    )
    return ys, xs, normalized, counts, widths, root_elongation


def _silhouette_axial_partition(
    root_mask: np.ndarray,
    wide_part: PartPrompt,
    narrow_part: PartPrompt,
    config: StructuralFusionConfig,
) -> _AxialPartition | None:
    profile = _axial_width_profile(root_mask.astype(bool), config.axial_profile_bin_count)
    if profile is None:
        return None
    ys, xs, normalized, counts, widths, root_elongation = profile
    if root_elongation < config.minimum_root_axial_elongation:
        return None

    bin_count = len(widths)
    total_area = max(1, int(counts.sum()))
    best: _AxialPartition | None = None
    for split_index in range(4, bin_count - 3):
        for wide_at_low_end in (True, False):
            wide_indices = (
                np.arange(0, split_index)
                if wide_at_low_end
                else np.arange(split_index, bin_count)
            )
            narrow_indices = (
                np.arange(split_index, bin_count)
                if wide_at_low_end
                else np.arange(0, split_index)
            )
            wide_fraction = float(counts[wide_indices].sum() / total_area)
            narrow_fraction = 1.0 - wide_fraction
            wide_lower = max(
                config.minimum_axial_terminal_fraction,
                wide_part.minimum_parent_fraction * 0.6,
            )
            wide_upper = min(
                config.maximum_axial_terminal_fraction,
                wide_part.maximum_parent_fraction * 1.20,
            )
            narrow_lower = max(
                config.minimum_axial_residual_fraction,
                narrow_part.minimum_parent_fraction * 0.6,
            )
            narrow_upper = min(0.94, narrow_part.maximum_parent_fraction * 1.20)
            if not (
                wide_lower <= wide_fraction <= wide_upper
                and narrow_lower <= narrow_fraction <= narrow_upper
            ):
                continue

            wide_width = float(np.quantile(widths[wide_indices], 0.65))
            narrow_width = float(np.quantile(widths[narrow_indices], 0.65))
            width_ratio = wide_width / max(1e-6, narrow_width)
            if width_ratio < config.minimum_axial_width_ratio:
                continue

            split_fraction = split_index / bin_count
            wide_pixels = (
                normalized < split_fraction
                if wide_at_low_end
                else normalized >= split_fraction
            )
            wide_mask = np.zeros(root_mask.shape, dtype=bool)
            narrow_mask = np.zeros(root_mask.shape, dtype=bool)
            wide_mask[ys[wide_pixels], xs[wide_pixels]] = True
            narrow_mask[ys[~wide_pixels], xs[~wide_pixels]] = True
            wide_shape = _shape(wide_mask, total_area)
            narrow_shape = _shape(narrow_mask, total_area)
            if (
                narrow_shape.elongation
                < config.minimum_generated_strip_elongation
                or narrow_shape.elongation < wide_shape.elongation * 1.5
            ):
                continue

            boundary_width = min(
                float(widths[split_index - 1]),
                float(widths[split_index]),
            )
            neck_ratio = boundary_width / max(1e-6, wide_width)
            left_width = float(
                np.mean(widths[max(0, split_index - 3) : split_index])
            )
            right_width = float(
                np.mean(widths[split_index : min(bin_count, split_index + 3)])
            )
            local_contrast = abs(left_width - right_width) / max(
                1e-6, wide_width, narrow_width
            )
            score = float(
                0.45 * min(width_ratio / 4.0, 1.0)
                + 0.25 * max(0.0, 1.0 - neck_ratio)
                + 0.20 * min(local_contrast, 1.0)
                + 0.10 * min(narrow_fraction / 0.35, 1.0)
            )
            if score < config.minimum_axial_partition_score:
                continue
            candidate = _AxialPartition(
                wide_mask=wide_mask,
                narrow_mask=narrow_mask,
                score=score,
                split_fraction=split_fraction,
                wide_at_low_axis_end=wide_at_low_end,
                width_ratio=width_ratio,
                neck_ratio=neck_ratio,
                local_width_contrast=local_contrast,
                root_elongation=root_elongation,
                wide_fraction=wide_fraction,
                narrow_fraction=narrow_fraction,
                wide_elongation=wide_shape.elongation,
                narrow_elongation=narrow_shape.elongation,
            )
            if best is None or candidate.score > best.score:
                best = candidate
    return best


def _generated_axial_candidate(
    root: MaskCandidate,
    part: PartPrompt,
    mask: np.ndarray,
    *,
    profile: str | None,
    role: str,
    peer_semantic: str,
    partition: _AxialPartition,
    source_reliability: float,
) -> MaskCandidate:
    semantic_parent = part.semantic_parent or root.semantic_name
    assembly_parent = part.assembly_parent or semantic_parent
    candidate_key = (
        f"{_candidate_key(root)}/silhouette-axial:{part.semantic_name}:01"
    )
    root_area = max(1, int(np.count_nonzero(root.mask)))
    return MaskCandidate(
        semantic_name=part.semantic_name,
        semantic_parent=semantic_parent,
        mask=mask,
        score=float(np.clip(0.78 + 0.16 * partition.score, 0.0, 0.95)),
        source="hpid-structural-fusion/silhouette-axial-partition",
        prompt=part.prompts[0],
        source_reliability=source_reliability,
        metadata={
            "source_family": "hpid-structural-fusion-v2",
            "root_origin": root.metadata.get("root_origin"),
            "root_index": root.metadata.get("root_index"),
            "candidate_key": candidate_key,
            "parent_candidate_key": _candidate_key(root),
            "assembly_parent_semantic": assembly_parent,
            "assembly_parent_candidate_key": _candidate_key(root),
            "hierarchy_depth": 1,
            "structural_fusion": True,
            "structural_root_evidence": True,
            "structural_fusion_algorithm": (
                "profile-constrained-silhouette-axial-partition-v1"
            ),
            "structural_role": role,
            "structural_profile": profile,
            "structural_peer_semantic": peer_semantic,
            "structural_partition_score": partition.score,
            "structural_split_fraction": partition.split_fraction,
            "structural_width_ratio": partition.width_ratio,
            "structural_neck_ratio": partition.neck_ratio,
            "structural_local_width_contrast": (
                partition.local_width_contrast
            ),
            "structural_root_elongation": partition.root_elongation,
            "root_area_fraction": float(np.count_nonzero(mask) / root_area),
            "maximum_instances": part.maximum_instances,
            "detail": part.detail,
            "ground_truth_used": False,
        },
    )


def _rail_step_pair(
    root: MaskCandidate, domain: DomainPrompt
) -> tuple[PartPrompt, PartPrompt, str | None] | None:
    parts, profile = _selected_parts(root, domain)
    rails = tuple(
        part
        for part in parts
        if not part.detail and part.semantic_name.casefold().endswith("_rail")
    )
    steps = tuple(
        part
        for part in parts
        if not part.detail
        and part.semantic_name.casefold().endswith(("_step", "_rung"))
    )
    if len(rails) != 1 or len(steps) != 1:
        return None
    return rails[0], steps[0], profile


def _hinged_parts(
    root: MaskCandidate, domain: DomainPrompt
) -> tuple[PartPrompt, PartPrompt, PartPrompt, str | None] | None:
    parts, profile = _selected_parts(root, domain)

    def one(suffixes: tuple[str, ...]) -> PartPrompt | None:
        matches = [
            part
            for part in parts
            if part.semantic_name.casefold().endswith(suffixes)
        ]
        return matches[0] if len(matches) == 1 else None

    pivot = one(("_pivot",))
    handle = one(("_handle",))
    terminal = one(("_blade", "_jaw"))
    if pivot is None or handle is None or terminal is None:
        return None
    return pivot, handle, terminal, profile


def _planar_surface_part(
    root: MaskCandidate, domain: DomainPrompt
) -> tuple[PartPrompt, str | None] | None:
    """Return a cover-like surface only for an inventory that actually defines it."""

    parts, profile = _selected_parts(root, domain)
    covers = tuple(
        part
        for part in parts
        if not part.detail and part.semantic_name.casefold().endswith("_cover")
    )
    if profile != "book" or len(covers) != 1:
        return None
    return covers[0], profile


def _overlap_components(
    candidates: list[MaskCandidate], minimum_containment: float
) -> list[list[MaskCandidate]]:
    remaining = set(range(len(candidates)))
    components: list[list[MaskCandidate]] = []
    while remaining:
        seed = remaining.pop()
        pending = [seed]
        component = {seed}
        while pending:
            index = pending.pop()
            first = candidates[index].mask.astype(bool)
            first_area = max(1, int(np.count_nonzero(first)))
            neighbours: list[int] = []
            for other_index in remaining:
                second = candidates[other_index].mask.astype(bool)
                second_area = max(1, int(np.count_nonzero(second)))
                intersection = int(np.count_nonzero(first & second))
                containment = intersection / min(first_area, second_area)
                if containment >= minimum_containment:
                    neighbours.append(other_index)
            for neighbour in neighbours:
                remaining.remove(neighbour)
                component.add(neighbour)
                pending.append(neighbour)
        components.append([candidates[index] for index in sorted(component)])
    return components


def _aggregate_planar_surface(
    root: MaskCandidate,
    candidates: list[MaskCandidate],
    part: PartPrompt,
    *,
    profile: str | None,
    config: StructuralFusionConfig,
) -> tuple[MaskCandidate | None, tuple[str, ...], dict[str, object]]:
    root_mask = root.mask.astype(bool)
    root_area = max(1, int(np.count_nonzero(root_mask)))
    tiles = []
    for candidate in candidates:
        if (
            _root_key(candidate) != _root_key(root)
            or candidate.semantic_name != part.semantic_name
            or not candidate.metadata.get("visual_region")
        ):
            continue
        constrained = candidate.mask.astype(bool) & root_mask
        fraction = int(np.count_nonzero(constrained)) / root_area
        containment = int(np.count_nonzero(constrained)) / max(
            1, int(np.count_nonzero(candidate.mask))
        )
        if (
            fraction >= config.minimum_planar_tile_root_fraction
            and containment >= 0.90
        ):
            tiles.append(replace(candidate, mask=constrained))
    groups = [
        group
        for group in _overlap_components(
            tiles, config.minimum_planar_pair_containment
        )
        if len(group) >= 2
    ]
    if not groups:
        return None, (), {
            "status": "no_overlapping_surface_tiles",
            "tile_count": len(tiles),
        }
    ranked: list[tuple[float, list[MaskCandidate], np.ndarray]] = []
    for group in groups:
        union = np.logical_or.reduce([candidate.mask for candidate in group]) & root_mask
        coverage = int(np.count_nonzero(union)) / root_area
        ranked.append((coverage, group, union))
    coverage, group, union = max(ranked, key=lambda row: row[0])
    if coverage < config.minimum_planar_union_root_fraction:
        return None, (), {
            "status": "insufficient_surface_coverage",
            "tile_count": len(tiles),
            "best_group_size": len(group),
            "best_union_root_fraction": coverage,
        }
    candidate_key = (
        f"{_candidate_key(root)}/structural-planar-surface:{part.semantic_name}:01"
    )
    semantic_parent = part.semantic_parent or root.semantic_name
    assembly_parent = part.assembly_parent or semantic_parent
    aggregate = MaskCandidate(
        semantic_name=part.semantic_name,
        semantic_parent=semantic_parent,
        mask=union,
        score=float(
            np.clip(
                0.50 * max(candidate.score for candidate in group)
                + 0.30 * coverage
                + 0.15,
                0.0,
                1.0,
            )
        ),
        source="hpid-structural-fusion/planar-tile-union",
        prompt=part.prompts[0],
        source_reliability=max(
            config.source_reliability,
            max(candidate.source_reliability for candidate in group),
        ),
        metadata={
            "source_family": "hpid-structural-fusion-v1",
            "root_origin": root.metadata.get("root_origin"),
            "root_index": root.metadata.get("root_index"),
            "candidate_key": candidate_key,
            "parent_candidate_key": _candidate_key(root),
            "assembly_parent_semantic": assembly_parent,
            "assembly_parent_candidate_key": _candidate_key(root),
            "hierarchy_depth": 1,
            "visual_region": True,
            "generic_visual_region": False,
            "visual_region_kind": "panel",
            "structural_fusion": True,
            "structural_fusion_algorithm": "profile-planar-tile-union-v1",
            "structural_role": "overlapping_surface_tile_union",
            "structural_profile": profile,
            "structural_source_candidate_keys": [
                _candidate_key(candidate) for candidate in group
            ],
            "root_containment": 1.0,
            "root_area_fraction": coverage,
            "maximum_instances": part.maximum_instances,
            "detail": part.detail,
            "ground_truth_used": False,
        },
    )
    return aggregate, tuple(_candidate_key(candidate) for candidate in group), {
        "status": "aggregated",
        "tile_count": len(tiles),
        "group_size": len(group),
        "union_root_fraction": coverage,
        "candidate_key": candidate_key,
    }


def _centroid(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return np.zeros(2, dtype=np.float64)
    return np.asarray([float(xs.mean()), float(ys.mean())], dtype=np.float64)


def _relabel_structural_visual(
    candidate: MaskCandidate,
    part: PartPrompt,
    root: MaskCandidate,
    *,
    profile: str | None,
    role: str,
    evidence: dict[str, object],
    source_reliability: float,
    parent_candidate_key: str | None = None,
) -> MaskCandidate:
    semantic_parent = part.semantic_parent or root.semantic_name
    assembly_parent = part.assembly_parent or semantic_parent
    return replace(
        candidate,
        semantic_name=part.semantic_name,
        semantic_parent=semantic_parent,
        score=float(np.clip(0.68 * candidate.score + 0.24, 0.0, 1.0)),
        source=f"{candidate.source}/structural-fusion",
        prompt=part.prompts[0],
        source_reliability=max(candidate.source_reliability, source_reliability),
        metadata={
            **candidate.metadata,
            "generic_visual_region": False,
            "structural_fusion": True,
            "structural_fusion_algorithm": (
                "profile-constrained-structural-relations-v2"
            ),
            "structural_role": role,
            "structural_profile": profile,
            **evidence,
            "maximum_instances": part.maximum_instances,
            "detail": part.detail,
            "parent_candidate_key": parent_candidate_key or _candidate_key(root),
            "assembly_parent_semantic": assembly_parent,
            "assembly_parent_candidate_key": (
                parent_candidate_key or _candidate_key(root)
            ),
            "ground_truth_used": False,
        },
    )


def _infer_profile_from_part_consensus(
    root: MaskCandidate,
    domain: DomainPrompt,
    candidates: list[MaskCandidate],
) -> tuple[MaskCandidate, dict[str, object]]:
    """Infer an object profile only from multiple visible, named part cues."""

    existing = root.metadata.get("selected_part_profile")
    if existing and root.metadata.get("profile_resolution_status") == "accepted":
        return root, {
            "status": "kept_existing",
            "selected_profile": str(existing),
        }
    profile_frequency: dict[str, int] = {}
    for profile in domain.part_profiles:
        for semantic in set(profile.part_semantics):
            profile_frequency[semantic] = profile_frequency.get(semantic, 0) + 1
    observed = {
        candidate.semantic_name
        for candidate in candidates
        if _root_key(candidate) == _root_key(root)
        and candidate.semantic_name != root.semantic_name
        and not candidate.metadata.get("generic_visual_region")
        and candidate.score * candidate.source_reliability >= 0.34
    }
    ranked: list[tuple[float, int, PartProfile, tuple[str, ...]]] = []
    for profile in domain.part_profiles:
        support = tuple(sorted(observed & set(profile.part_semantics)))
        score = sum(1.0 / profile_frequency[name] for name in support)
        ranked.append((score, len(support), profile, support))
    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    if not ranked:
        return root, {"status": "no_profiles", "selected_profile": None}
    top_score, top_count, top_profile, top_support = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else 0.0
    accepted = bool(
        top_count >= 2
        and top_score >= 1.5
        and (top_score - second_score >= 0.75 or top_count >= 3)
    )
    diagnostics = {
        "status": "accepted" if accepted else "unresolved",
        "selected_profile": top_profile.name if accepted else None,
        "support_semantics": list(top_support),
        "support_count": top_count,
        "support_score": top_score,
        "runner_up_score": second_score,
        "ground_truth_used": False,
    }
    if not accepted:
        return root, diagnostics
    metadata = {
        **root.metadata,
        "selected_part_profile": top_profile.name,
        "profile_resolution_status": "accepted",
        "profile_hint_source": "observed_part_consensus",
        "resolved_object_label": top_profile.root_hints[0],
    }
    return replace(root, metadata=metadata), diagnostics


def _root_axis_positions(
    root: MaskCandidate,
    candidates: list[MaskCandidate],
    parts: tuple[PartPrompt, ...],
) -> tuple[dict[str, float], float] | None:
    ys, xs = np.nonzero(root.mask)
    if len(xs) < 16:
        return None
    points = np.column_stack((xs, ys)).astype(np.float64)
    center = points.mean(axis=0)
    _, eigenvectors = np.linalg.eigh(np.cov((points - center).T))
    axis = eigenvectors[:, -1]
    root_projection = (points - center) @ axis
    span = float(root_projection.max() - root_projection.min())
    if span < 4.0:
        return None

    def position(mask: np.ndarray) -> float:
        centroid = _centroid(mask)
        projected = float((centroid - center) @ axis)
        return float(
            np.clip(
                2.0 * (projected - float(root_projection.min())) / span - 1.0,
                -1.0,
                1.0,
            )
        )

    part_by_name = {part.semantic_name: part for part in parts}
    anchors: list[tuple[float, float]] = []
    for candidate in candidates:
        if _root_key(candidate) != _root_key(root):
            continue
        part = part_by_name.get(candidate.semantic_name)
        if part is None or part.axis_position is None:
            continue
        anchors.append((position(candidate.mask), part.axis_position))
    if len(anchors) < 2:
        return None
    positive_error = float(np.mean([abs(value - target) for value, target in anchors]))
    negative_error = float(np.mean([abs(-value - target) for value, target in anchors]))
    sign = -1.0 if negative_error < positive_error else 1.0
    return {
        _candidate_key(candidate): sign * position(candidate.mask)
        for candidate in candidates
        if _root_key(candidate) == _root_key(root)
    }, min(positive_error, negative_error)


def _relabel_generic_profile_regions(
    root: MaskCandidate,
    domain: DomainPrompt,
    output_by_key: dict[str, MaskCandidate],
    evidence_candidates: list[MaskCandidate],
    *,
    source_reliability: float,
) -> tuple[int, list[dict[str, object]]]:
    profile_name = root.metadata.get("selected_part_profile")
    if not profile_name:
        return 0, []
    parts, selected_profile, _ = domain.select_parts(
        str(root.metadata.get("resolved_object_label") or profile_name),
        profile_hint=str(profile_name),
        profile_hint_source="observed_part_consensus",
    )
    if selected_profile is None:
        return 0, []
    axis_result = _root_axis_positions(root, evidence_candidates, parts)
    if axis_result is None:
        return 0, []
    positions, calibration_error = axis_result
    if calibration_error > 0.48:
        return 0, []
    present = {
        candidate.semantic_name
        for candidate in evidence_candidates
        if _root_key(candidate) == _root_key(root)
        and not candidate.metadata.get("generic_visual_region")
    }
    missing = [
        part
        for part in parts
        if part.semantic_name not in present
        and not part.detail
        and part.axis_position is not None
    ]
    root_area = max(1, int(np.count_nonzero(root.mask)))
    assignments: list[dict[str, object]] = []
    for key, candidate in sorted(
        output_by_key.items(),
        key=lambda item: int(np.count_nonzero(item[1].mask)),
        reverse=True,
    ):
        if (
            _root_key(candidate) != _root_key(root)
            or not candidate.metadata.get("generic_visual_region")
            or key not in positions
        ):
            continue
        area_fraction = int(np.count_nonzero(candidate.mask & root.mask)) / root_area
        position = positions[key]
        compatible: list[tuple[float, PartPrompt]] = []
        for part in missing:
            if not (
                part.minimum_parent_fraction * 0.6
                <= area_fraction
                <= min(1.0, part.maximum_parent_fraction * 1.15)
            ):
                continue
            residual = abs(position - float(part.axis_position)) / part.axis_tolerance
            if residual <= 1.0:
                compatible.append((residual, part))
        compatible.sort(key=lambda row: row[0])
        if not compatible:
            continue
        best_residual, best_part = compatible[0]
        runner_up = compatible[1][0] if len(compatible) > 1 else None
        if best_residual > 0.72 or (
            runner_up is not None and runner_up - best_residual < 0.22
        ):
            continue
        output_by_key[key] = _relabel_structural_visual(
            candidate,
            best_part,
            root,
            profile=selected_profile,
            role="profile_axis_missing_role",
            evidence={
                "structural_axis_position": position,
                "structural_axis_target": best_part.axis_position,
                "structural_axis_residual": best_residual,
                "structural_axis_calibration_error": calibration_error,
                "root_area_fraction": area_fraction,
            },
            source_reliability=source_reliability,
        )
        missing.remove(best_part)
        present.add(best_part.semantic_name)
        assignments.append(
            {
                "candidate_key": key,
                "semantic_name": best_part.semantic_name,
                "axis_position": position,
                "axis_residual": best_residual,
                "root_area_fraction": area_fraction,
            }
        )
    return len(assignments), assignments


def _strip_rank(
    candidate: MaskCandidate,
    shape: _MaskShape,
    strip_part: PartPrompt,
) -> float:
    shape_score = float(
        np.clip(np.log2(max(1.0, shape.elongation)) / 7.0, 0.0, 1.0)
    )
    effective = float(candidate.score * candidate.source_reliability)
    semantic_bonus = 0.12 if candidate.semantic_name == strip_part.semantic_name else 0.0
    visual_bonus = 0.05 if candidate.metadata.get("visual_region") else 0.0
    return 0.48 * shape_score + 0.35 * effective + semantic_bonus + visual_bonus


def _largest_residual_component(
    root_mask: np.ndarray,
    strip_mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    residual = root_mask.astype(bool) & ~strip_mask.astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        residual.astype(np.uint8), 8
    )
    if count <= 1:
        return residual, int(max(0, count - 1))
    root_area = int(np.count_nonzero(root_mask))
    minimum_area = max(12, round(root_area * 0.012))
    components = [
        component_id
        for component_id in range(1, count)
        if int(stats[component_id, cv2.CC_STAT_AREA]) >= minimum_area
    ]
    if not components:
        return np.zeros(root_mask.shape, dtype=bool), int(count - 1)
    selected_id = max(
        components,
        key=lambda component_id: int(stats[component_id, cv2.CC_STAT_AREA]),
    )
    return labels == selected_id, int(count - 1)


def _existing_mass_is_credible(
    candidates: list[MaskCandidate],
    root: MaskCandidate,
    mass_part: PartPrompt,
    config: StructuralFusionConfig,
) -> bool:
    root_area = max(1, int(np.count_nonzero(root.mask)))
    for candidate in candidates:
        if (
            _root_key(candidate) != _root_key(root)
            or candidate.semantic_name != mass_part.semantic_name
        ):
            continue
        shape = _shape(candidate.mask & root.mask, root_area)
        if (
            candidate.score * candidate.source_reliability
            >= config.minimum_incumbent_effective_score
            and mass_part.minimum_parent_fraction * 0.6
            <= shape.root_fraction
            <= min(1.0, mass_part.maximum_parent_fraction * 1.15)
        ):
            return True
    return False


def refine_profile_structure(
    visual_candidates: list[MaskCandidate],
    roots: list[MaskCandidate],
    all_candidates: list[MaskCandidate],
    domains: dict[str, DomainPrompt],
    *,
    config: StructuralFusionConfig | None = None,
) -> StructuralFusionResult:
    """Resolve simple axial assemblies from model proposals and root ownership.

    The operation is profile constrained but image independent: no coordinates,
    masks, or thresholds are learned from a benchmark annotation. It applies
    only when the selected inventory contains one elongated major part and one
    compact major part, such as handle+bowl or handle+head.
    """

    config = config or StructuralFusionConfig()
    output_by_key = {
        _candidate_key(candidate): candidate for candidate in visual_candidates
    }
    rows: list[dict[str, object]] = []
    generated_count = 0
    reassigned_count = 0
    aggregated_surface_count = 0
    silhouette_partition_count = 0
    pool_by_key = {
        _candidate_key(candidate): candidate
        for candidate in (*all_candidates, *visual_candidates)
    }
    profile_consensus_rows: list[dict[str, object]] = []
    profile_axis_assignments: list[dict[str, object]] = []
    for root in roots:
        domain = domains.get(root.semantic_name)
        if domain is not None:
            root, profile_row = _infer_profile_from_part_consensus(
                root,
                domain,
                list(pool_by_key.values()),
            )
            profile_consensus_rows.append(
                {"root_key": _root_key(root), **profile_row}
            )
            assignment_count, assignment_rows = _relabel_generic_profile_regions(
                root,
                domain,
                output_by_key,
                list(pool_by_key.values()),
                source_reliability=config.source_reliability,
            )
            reassigned_count += assignment_count
            profile_axis_assignments.extend(
                {"root_key": _root_key(root), **row} for row in assignment_rows
            )
        planar_surface = (
            _planar_surface_part(root, domain) if domain is not None else None
        )
        if planar_surface is not None:
            surface_part, surface_profile = planar_surface
            aggregate, consumed_keys, aggregate_diagnostics = (
                _aggregate_planar_surface(
                    root,
                    list(output_by_key.values()),
                    surface_part,
                    profile=surface_profile,
                    config=config,
                )
            )
            if aggregate is not None:
                for key in consumed_keys:
                    output_by_key.pop(key, None)
                output_by_key[_candidate_key(aggregate)] = aggregate
                aggregated_surface_count += 1
            rows.append(
                {
                    "root_key": _root_key(root),
                    "profile": surface_profile,
                    "strategy": "overlapping_planar_surface_union",
                    "semantic_name": surface_part.semantic_name,
                    **aggregate_diagnostics,
                }
            )
        rail_step = _rail_step_pair(root, domain) if domain is not None else None
        if rail_step is not None:
            rail_part, step_part, rail_profile = rail_step
            root_area = max(1, int(np.count_nonzero(root.mask)))
            root_shape = _shape(root.mask, root_area)
            root_direction = np.asarray(root_shape.direction_xy, dtype=np.float64)
            proposals: list[
                tuple[int, float, MaskCandidate, _MaskShape, PartPrompt, float]
            ] = []
            for candidate in list(output_by_key.values()):
                if (
                    _root_key(candidate) != _root_key(root)
                    or not candidate.metadata.get("generic_visual_region")
                ):
                    continue
                shape = _shape(candidate.mask & root.mask, root_area)
                if shape.elongation < 4.0 or not 0.002 <= shape.root_fraction <= 0.65:
                    continue
                direction = np.asarray(shape.direction_xy, dtype=np.float64)
                alignment = float(abs(direction @ root_direction))
                if alignment >= 0.72:
                    part = rail_part
                    role_order = 0
                    relation_score = alignment
                elif alignment <= 0.48:
                    part = step_part
                    role_order = 1
                    relation_score = 1.0 - alignment
                else:
                    continue
                score = (
                    0.48 * relation_score
                    + 0.28 * float(candidate.score * candidate.source_reliability)
                    + 0.24
                    * float(
                        np.clip(
                            np.log2(max(1.0, shape.elongation)) / 7.0,
                            0.0,
                            1.0,
                        )
                    )
                )
                proposals.append(
                    (role_order, score, candidate, shape, part, alignment)
                )
            proposals.sort(key=lambda item: (item[0], -item[1]))
            accepted_by_semantic: dict[str, int] = {}
            rail_parent_key: str | None = None
            relation_rows: list[dict[str, object]] = []
            for _, score, candidate, shape, part, alignment in proposals:
                accepted = accepted_by_semantic.get(part.semantic_name, 0)
                if accepted >= part.maximum_instances:
                    continue
                key = _candidate_key(candidate)
                parent_key = (
                    rail_parent_key
                    if part.semantic_name == step_part.semantic_name
                    else _candidate_key(root)
                )
                output_by_key[key] = _relabel_structural_visual(
                    candidate,
                    part,
                    root,
                    profile=rail_profile,
                    role=(
                        "parallel_rail"
                        if part.semantic_name == rail_part.semantic_name
                        else "perpendicular_crossbar"
                    ),
                    evidence={
                        "structural_axis_alignment": alignment,
                        "structural_elongation": shape.elongation,
                        "structural_rank": score,
                    },
                    source_reliability=config.source_reliability,
                    parent_candidate_key=parent_key,
                )
                if part.semantic_name == rail_part.semantic_name and rail_parent_key is None:
                    rail_parent_key = key
                accepted_by_semantic[part.semantic_name] = accepted + 1
                reassigned_count += 1
                relation_rows.append(
                    {
                        "candidate_key": key,
                        "semantic_name": part.semantic_name,
                        "axis_alignment": alignment,
                        "elongation": shape.elongation,
                        "rank": score,
                    }
                )
            rows.append(
                {
                    "root_key": _root_key(root),
                    "profile": rail_profile,
                    "status": "completed",
                    "strategy": "parallel_rail_perpendicular_crossbar",
                    "rail_semantic": rail_part.semantic_name,
                    "step_semantic": step_part.semantic_name,
                    "accepted_by_semantic": accepted_by_semantic,
                    "assignments": relation_rows,
                }
            )
        hinged = _hinged_parts(root, domain) if domain is not None else None
        if hinged is not None:
            pivot_part, handle_part, terminal_part, hinged_profile = hinged
            root_area = max(1, int(np.count_nonzero(root.mask)))
            pivot_candidates = [
                candidate
                for candidate in pool_by_key.values()
                if _root_key(candidate) == _root_key(root)
                and candidate.semantic_name == pivot_part.semantic_name
            ]
            terminal_candidates = [
                candidate
                for candidate in pool_by_key.values()
                if _root_key(candidate) == _root_key(root)
                and candidate.semantic_name == terminal_part.semantic_name
                and _shape(candidate.mask & root.mask, root_area).elongation >= 3.5
            ]
            hinge_rows: list[dict[str, object]] = []
            if pivot_candidates and terminal_candidates:
                pivot = max(
                    pivot_candidates,
                    key=lambda candidate: (
                        candidate.score * candidate.source_reliability
                    ),
                )
                pivot_center = _centroid(pivot.mask)
                terminal_vectors: list[np.ndarray] = []
                for terminal in terminal_candidates:
                    vector = _centroid(terminal.mask) - pivot_center
                    norm = float(np.linalg.norm(vector))
                    if norm > 1.0:
                        terminal_vectors.append(vector / norm)
                if terminal_vectors:
                    terminal_direction = np.sum(terminal_vectors, axis=0)
                    direction_norm = float(np.linalg.norm(terminal_direction))
                    if direction_norm > 1e-6:
                        terminal_direction /= direction_norm
                    handle_proposals: list[
                        tuple[float, MaskCandidate, _MaskShape, float]
                    ] = []
                    for candidate in list(output_by_key.values()):
                        if (
                            _root_key(candidate) != _root_key(root)
                            or not candidate.metadata.get("generic_visual_region")
                        ):
                            continue
                        shape = _shape(candidate.mask & root.mask, root_area)
                        if (
                            shape.elongation < 8.0
                            or not 0.004 <= shape.root_fraction <= 0.42
                        ):
                            continue
                        vector = _centroid(candidate.mask) - pivot_center
                        distance = float(np.linalg.norm(vector))
                        if distance <= 2.0:
                            continue
                        direction = vector / distance
                        directional_cosine = float(direction @ terminal_direction)
                        if directional_cosine > 0.05:
                            continue
                        if any(
                            mask_iou(candidate.mask, terminal.mask) >= 0.20
                            for terminal in terminal_candidates
                        ):
                            continue
                        opposite_score = float(
                            np.clip((0.05 - directional_cosine) / 1.05, 0.0, 1.0)
                        )
                        score = (
                            0.52 * opposite_score
                            + 0.28
                            * float(candidate.score * candidate.source_reliability)
                            + 0.20
                            * float(
                                np.clip(
                                    np.log2(max(1.0, shape.elongation)) / 7.0,
                                    0.0,
                                    1.0,
                                )
                            )
                        )
                        handle_proposals.append(
                            (score, candidate, shape, directional_cosine)
                        )
                    handle_proposals.sort(key=lambda item: item[0], reverse=True)
                    for score, candidate, shape, cosine in handle_proposals[
                        : min(2, handle_part.maximum_instances)
                    ]:
                        key = _candidate_key(candidate)
                        output_by_key[key] = _relabel_structural_visual(
                            candidate,
                            handle_part,
                            root,
                            profile=hinged_profile,
                            role="hinged_opposite_branch",
                            evidence={
                                "structural_pivot_candidate_key": (
                                    _candidate_key(pivot)
                                ),
                                "structural_terminal_semantic": (
                                    terminal_part.semantic_name
                                ),
                                "structural_directional_cosine": cosine,
                                "structural_elongation": shape.elongation,
                                "structural_rank": score,
                            },
                            source_reliability=config.source_reliability,
                        )
                        reassigned_count += 1
                        hinge_rows.append(
                            {
                                "candidate_key": key,
                                "semantic_name": handle_part.semantic_name,
                                "directional_cosine": cosine,
                                "elongation": shape.elongation,
                                "rank": score,
                            }
                        )
            rows.append(
                {
                    "root_key": _root_key(root),
                    "profile": hinged_profile,
                    "status": "completed",
                    "strategy": "pivot_terminal_opposite_branch",
                    "pivot_semantic": pivot_part.semantic_name,
                    "terminal_semantic": terminal_part.semantic_name,
                    "handle_semantic": handle_part.semantic_name,
                    "assignment_count": len(hinge_rows),
                    "assignments": hinge_rows,
                }
            )
        pair = _wide_narrow_pair(root, domain) if domain is not None else None
        if pair is None:
            continue
        mass_part, strip_part, profile = pair
        root_area = max(1, int(np.count_nonzero(root.mask)))
        eligible: list[tuple[float, MaskCandidate, _MaskShape]] = []
        for candidate in pool_by_key.values():
            if (
                _root_key(candidate) != _root_key(root)
                or candidate.semantic_name == root.semantic_name
                or (
                    candidate.semantic_name != strip_part.semantic_name
                    and not candidate.metadata.get("generic_visual_region")
                )
            ):
                continue
            constrained = candidate.mask.astype(bool) & root.mask.astype(bool)
            containment = int(np.count_nonzero(constrained)) / max(
                1, int(np.count_nonzero(candidate.mask))
            )
            shape = _shape(constrained, root_area)
            if (
                containment < 0.80
                or shape.elongation < config.minimum_strip_elongation
                or shape.root_fraction < config.minimum_strip_root_fraction
                or shape.root_fraction > config.maximum_strip_root_fraction
                or shape.root_fraction
                > min(1.0, strip_part.maximum_parent_fraction * 1.15)
            ):
                continue
            eligible.append(
                (_strip_rank(candidate, shape, strip_part), candidate, shape)
            )
        if not eligible:
            partition = _silhouette_axial_partition(
                root.mask,
                mass_part,
                strip_part,
                config,
            )
            if partition is not None:
                wide = _generated_axial_candidate(
                    root,
                    mass_part,
                    partition.wide_mask,
                    profile=profile,
                    role="wide_terminal_segment",
                    peer_semantic=strip_part.semantic_name,
                    partition=partition,
                    source_reliability=config.source_reliability,
                )
                narrow = _generated_axial_candidate(
                    root,
                    strip_part,
                    partition.narrow_mask,
                    profile=profile,
                    role="elongated_residual_segment",
                    peer_semantic=mass_part.semantic_name,
                    partition=partition,
                    source_reliability=config.source_reliability,
                )
                output_by_key[_candidate_key(wide)] = wide
                output_by_key[_candidate_key(narrow)] = narrow
                generated_count += 2
                silhouette_partition_count += 1
                rows.append(
                    {
                        "root_key": _root_key(root),
                        "profile": profile,
                        "status": "completed_silhouette_axial_partition",
                        "strategy": (
                            "profile_constrained_silhouette_width_change"
                        ),
                        "strip_semantic": strip_part.semantic_name,
                        "mass_semantic": mass_part.semantic_name,
                        "partition_score": partition.score,
                        "split_fraction": partition.split_fraction,
                        "wide_at_low_axis_end": (
                            partition.wide_at_low_axis_end
                        ),
                        "width_ratio": partition.width_ratio,
                        "neck_ratio": partition.neck_ratio,
                        "local_width_contrast": (
                            partition.local_width_contrast
                        ),
                        "root_elongation": partition.root_elongation,
                        "wide_fraction": partition.wide_fraction,
                        "narrow_fraction": partition.narrow_fraction,
                        "wide_elongation": partition.wide_elongation,
                        "narrow_elongation": partition.narrow_elongation,
                    }
                )
                continue
            rows.append(
                {
                    "root_key": _root_key(root),
                    "profile": profile,
                    "status": "no_elongated_anchor",
                    "strip_semantic": strip_part.semantic_name,
                    "mass_semantic": mass_part.semantic_name,
                }
            )
            continue
        strip_rank, strip, strip_shape = max(eligible, key=lambda item: item[0])
        if strip_rank < config.minimum_axial_rank:
            rows.append(
                {
                    "root_key": _root_key(root),
                    "profile": profile,
                    "status": "low_confidence_elongated_anchor",
                    "strip_semantic": strip_part.semantic_name,
                    "mass_semantic": mass_part.semantic_name,
                    "strip_rank": strip_rank,
                    "minimum_axial_rank": config.minimum_axial_rank,
                }
            )
            continue
        strip_key = _candidate_key(strip)
        selected_strip = strip
        if strip_key in output_by_key and strip.semantic_name != strip_part.semantic_name:
            selected_strip = _relabel_structural_visual(
                strip,
                strip_part,
                root,
                profile=profile,
                role="elongated_anchor",
                evidence={
                    "structural_elongation": strip_shape.elongation,
                    "structural_rank": strip_rank,
                },
                source_reliability=config.source_reliability,
            )
            output_by_key[strip_key] = selected_strip
            reassigned_count += 1

        generated = False
        residual_fraction = 0.0
        residual_components = 0
        if not _existing_mass_is_credible(
            all_candidates, root, mass_part, config
        ):
            residual, residual_components = _largest_residual_component(
                root.mask, selected_strip.mask
            )
            residual_area = int(np.count_nonzero(residual))
            residual_fraction = residual_area / root_area
            coverage = int(
                np.count_nonzero(residual | (selected_strip.mask & root.mask))
            ) / root_area
            lower = max(
                config.minimum_residual_root_fraction,
                mass_part.minimum_parent_fraction * 0.6,
            )
            upper = min(
                config.maximum_residual_root_fraction,
                mass_part.maximum_parent_fraction * 1.10,
            )
            if (
                residual_area >= 12
                and lower <= residual_fraction <= upper
                and coverage >= config.minimum_structural_coverage
            ):
                semantic_parent = mass_part.semantic_parent or root.semantic_name
                assembly_parent = mass_part.assembly_parent or semantic_parent
                candidate_key = (
                    f"{_candidate_key(root)}/structural-residual:"
                    f"{mass_part.semantic_name}:01"
                )
                output_by_key[candidate_key] = MaskCandidate(
                    semantic_name=mass_part.semantic_name,
                    semantic_parent=semantic_parent,
                    mask=residual,
                    score=float(np.clip(0.62 + 0.28 * coverage, 0.0, 1.0)),
                    source="hpid-structural-fusion/axial-residual",
                    prompt=mass_part.prompts[0],
                    source_reliability=config.source_reliability,
                    metadata={
                        "source_family": "hpid-structural-fusion-v1",
                        "root_origin": root.metadata.get("root_origin"),
                        "root_index": root.metadata.get("root_index"),
                        "candidate_key": candidate_key,
                        "parent_candidate_key": _candidate_key(root),
                        "assembly_parent_semantic": assembly_parent,
                        "assembly_parent_candidate_key": _candidate_key(root),
                        "hierarchy_depth": 1,
                        "structural_fusion": True,
                        "structural_root_evidence": True,
                        "structural_fusion_algorithm": (
                            "profile-constrained-axial-residual-v1"
                        ),
                        "structural_role": "owned_root_residual",
                        "structural_profile": profile,
                        "structural_anchor_semantic": strip_part.semantic_name,
                        "structural_anchor_candidate_key": strip_key,
                        "structural_component_count": residual_components,
                        "structural_coverage": coverage,
                        "root_area_fraction": residual_fraction,
                        "maximum_instances": mass_part.maximum_instances,
                        "detail": mass_part.detail,
                        "ground_truth_used": False,
                    },
                )
                generated = True
                generated_count += 1
        rows.append(
            {
                "root_key": _root_key(root),
                "profile": profile,
                "status": "completed",
                "strip_semantic": strip_part.semantic_name,
                "mass_semantic": mass_part.semantic_name,
                "selected_strip_candidate_key": strip_key,
                "selected_strip_was_generic": bool(
                    strip.metadata.get("generic_visual_region")
                ),
                "strip_elongation": strip_shape.elongation,
                "strip_rank": strip_rank,
                "mass_generated": generated,
                "residual_fraction": residual_fraction,
                "residual_component_count": residual_components,
            }
        )
    return StructuralFusionResult(
        tuple(output_by_key.values()),
        {
            "algorithm": "hpid-profile-constrained-structural-fusion-v1",
            "root_count": len(roots),
            "eligible_root_count": len(rows),
            "semantic_reassignment_count": reassigned_count,
            "generated_residual_count": generated_count,
            "aggregated_surface_count": aggregated_surface_count,
            "silhouette_partition_count": silhouette_partition_count,
            "observed_part_profile_consensus": profile_consensus_rows,
            "profile_axis_assignments": profile_axis_assignments,
            "roots": rows,
            "ground_truth_used": False,
        },
    )
