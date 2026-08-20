from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from .fusion import MaskCandidate, mask_iou
from .prompt_bank import DomainPrompt, PartPrompt


@dataclass(frozen=True)
class RootGeometryConfig:
    minimum_terminal_elongation: float = 3.0
    maximum_terminal_pivot_distance_ratio: float = 0.15
    maximum_handle_pivot_distance_ratio: float = 0.30
    maximum_panel_pivot_distance_ratio: float = 0.07
    minimum_refined_root_fraction: float = 0.12
    maximum_refined_root_fraction: float = 0.82
    handle_band_diagonal_ratio: float = 0.014
    source_reliability: float = 0.82


@dataclass(frozen=True)
class RootGeometryResult:
    candidates: tuple[MaskCandidate, ...]
    diagnostics: dict[str, object]


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


def _is_root(candidate: MaskCandidate) -> bool:
    return bool(
        candidate.semantic_name == candidate.semantic_parent
        and candidate.metadata.get("parent_candidate_key") is None
    )


def _centroid(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return np.zeros(2, dtype=np.float64)
    return np.asarray([float(xs.mean()), float(ys.mean())], dtype=np.float64)


def _elongation(mask: np.ndarray) -> float:
    ys, xs = np.nonzero(mask)
    if len(xs) < 8:
        return 1.0
    points = np.column_stack((xs, ys)).astype(np.float64)
    eigenvalues = np.linalg.eigvalsh(np.cov(points.T))
    return float(eigenvalues[-1] / max(1e-6, eigenvalues[0]))


def _minimum_mask_distance(first: np.ndarray, second: np.ndarray) -> float:
    first = first.astype(bool)
    second = second.astype(bool)
    if np.any(first & second):
        return 0.0
    if not np.any(first) or not np.any(second):
        return float("inf")
    distance = cv2.distanceTransform((~second).astype(np.uint8), cv2.DIST_L2, 5)
    return float(distance[first].min())


def _deduplicate(
    candidates: list[MaskCandidate],
    *,
    maximum_count: int,
) -> list[MaskCandidate]:
    ranked = sorted(
        candidates,
        key=lambda candidate: (
            candidate.score * candidate.source_reliability,
            int(np.count_nonzero(candidate.mask)),
        ),
        reverse=True,
    )
    kept: list[MaskCandidate] = []
    for candidate in ranked:
        area = max(1, int(np.count_nonzero(candidate.mask)))
        duplicate = False
        for existing in kept:
            existing_area = max(1, int(np.count_nonzero(existing.mask)))
            intersection = int(np.count_nonzero(candidate.mask & existing.mask))
            containment = intersection / min(area, existing_area)
            if mask_iou(candidate.mask, existing.mask) >= 0.52 or containment >= 0.88:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(candidate)
        if len(kept) >= maximum_count:
            break
    return kept


def _bbox(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return [0, 0, 0, 0]
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max() + 1),
        int(ys.max() + 1),
    ]


def _selected_hinged_parts(
    root: MaskCandidate,
    domain: DomainPrompt,
) -> tuple[PartPrompt, PartPrompt, PartPrompt, PartPrompt | None, str] | None:
    profile_value = root.metadata.get("selected_part_profile")
    profile = str(profile_value) if profile_value else None
    if profile is None:
        return None
    if profile not in {item.name for item in domain.part_profiles}:
        return None
    parts, selected_profile, _ = domain.select_parts(
        str(root.metadata.get("root_model_label") or root.prompt),
        profile_hint=profile,
        profile_hint_source=str(
            root.metadata.get("profile_hint_source") or "specific_root_label"
        ),
    )
    if selected_profile != profile:
        return None

    def one(suffixes: tuple[str, ...]) -> PartPrompt | None:
        matches = [
            part for part in parts if part.semantic_name.casefold().endswith(suffixes)
        ]
        return matches[0] if len(matches) == 1 else None

    pivot = one(("_pivot",))
    handle = one(("_handle",))
    terminal = one(("_blade", "_jaw"))
    finger_hole = one(("_finger_hole",))
    if pivot is None or handle is None or terminal is None:
        return None
    return pivot, handle, terminal, finger_hole, profile


def _refine_handle_ring(
    mask: np.ndarray,
    joint_support: np.ndarray,
    *,
    diagonal: float,
    config: RootGeometryConfig,
) -> tuple[np.ndarray, bool, int]:
    mask = mask.astype(bool)
    area = int(np.count_nonzero(mask))
    if area < 80:
        return mask, False, 0
    width = int(
        np.clip(round(diagonal * config.handle_band_diagonal_ratio), 2, 14)
    )
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    band = mask & (distance <= float(width))
    joint_radius = max(2, width * 2)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * joint_radius + 1, 2 * joint_radius + 1),
    )
    near_joint = mask & cv2.dilate(joint_support.astype(np.uint8), kernel).astype(bool)
    refined = band | near_joint
    refined_area = int(np.count_nonzero(refined))
    retained = refined_area / max(1, area)
    if not 0.15 <= retained <= 0.78:
        return mask, False, width
    return refined, True, width


def _relabel_terminal(
    candidate: MaskCandidate,
    part: PartPrompt,
    root: MaskCandidate,
) -> MaskCandidate:
    semantic_parent = part.semantic_parent or root.semantic_name
    metadata = {
        **candidate.metadata,
        "generic_visual_region": False,
        "structural_fusion": True,
        "structural_fusion_algorithm": "hinged-part-graph-v1",
        "structural_role": "terminal_branch",
        "structural_profile": root.metadata.get("selected_part_profile"),
        "parent_candidate_key": _candidate_key(root),
        "assembly_parent_semantic": semantic_parent,
        "assembly_parent_candidate_key": _candidate_key(root),
        "maximum_instances": part.maximum_instances,
        "detail": part.detail,
        "ground_truth_used": False,
    }
    return replace(
        candidate,
        semantic_name=part.semantic_name,
        semantic_parent=semantic_parent,
        prompt=part.prompts[0],
        source_reliability=max(candidate.source_reliability, 0.82),
        metadata=metadata,
    )


def refine_root_geometry_from_parts(
    candidates: list[MaskCandidate],
    roots: list[MaskCandidate],
    domains: dict[str, DomainPrompt],
    *,
    config: RootGeometryConfig | None = None,
) -> RootGeometryResult:
    """Tighten a root only when a profile-constrained physical graph is complete.

    The current implementation handles pivoted tools.  It requires a compact
    pivot, distinct elongated terminal branches, and opposite handle evidence.
    No benchmark annotation or target mask is used.
    """

    config = config or RootGeometryConfig()
    replacements: dict[str, MaskCandidate] = {}
    candidate_replacements: dict[str, MaskCandidate] = {}
    rows: list[dict[str, object]] = []
    by_root: dict[str, list[MaskCandidate]] = {}
    for candidate in candidates:
        by_root.setdefault(_root_key(candidate), []).append(candidate)

    for root in roots:
        domain = domains.get(root.semantic_name)
        hinged = _selected_hinged_parts(root, domain) if domain is not None else None
        if hinged is None:
            continue
        pivot_part, handle_part, terminal_part, finger_hole_part, profile = hinged
        root_key = _root_key(root)
        root_mask = root.mask.astype(bool)
        root_area = max(1, int(np.count_nonzero(root_mask)))
        height, width = root_mask.shape
        diagonal = max(1.0, float(np.hypot(height, width)))
        owned = [
            candidate
            for candidate in by_root.get(root_key, [])
            if not _is_root(candidate) and np.any(candidate.mask & root_mask)
        ]
        pivots = [
            candidate
            for candidate in owned
            if candidate.semantic_name == pivot_part.semantic_name
            and int(np.count_nonzero(candidate.mask & root_mask)) / root_area <= 0.10
        ]
        if not pivots:
            rows.append(
                {
                    "root_key": root_key,
                    "profile": profile,
                    "status": "missing_pivot",
                }
            )
            continue
        pivot = max(
            pivots,
            key=lambda candidate: candidate.score * candidate.source_reliability,
        )
        pivot_mask = pivot.mask.astype(bool) & root_mask
        pivot_center = _centroid(pivot_mask)

        named_terminals = [
            candidate
            for candidate in owned
            if candidate.semantic_name == terminal_part.semantic_name
        ]
        generic_terminals = [
            candidate
            for candidate in owned
            if candidate.metadata.get("generic_visual_region")
            and candidate.metadata.get("visual_region")
            and _elongation(candidate.mask & root_mask)
            >= config.minimum_terminal_elongation
        ]
        terminal_pool = []
        for candidate in [*named_terminals, *generic_terminals]:
            constrained = candidate.mask.astype(bool) & root_mask
            if (
                _elongation(constrained) < config.minimum_terminal_elongation
                or _minimum_mask_distance(pivot_mask, constrained)
                > config.maximum_terminal_pivot_distance_ratio * diagonal
            ):
                continue
            terminal_pool.append(replace(candidate, mask=constrained))
        terminals = _deduplicate(
            terminal_pool,
            maximum_count=max(2, terminal_part.maximum_instances),
        )
        if len(terminals) < 2:
            rows.append(
                {
                    "root_key": root_key,
                    "profile": profile,
                    "status": "insufficient_terminal_branches",
                    "terminal_count": len(terminals),
                }
            )
            continue

        terminal_vectors = []
        for terminal in terminals:
            vector = _centroid(terminal.mask) - pivot_center
            norm = float(np.linalg.norm(vector))
            if norm > 1.0:
                terminal_vectors.append(vector / norm)
        if len(terminal_vectors) < 2:
            continue
        terminal_direction = np.sum(terminal_vectors, axis=0)
        terminal_norm = float(np.linalg.norm(terminal_direction))
        if terminal_norm <= 1e-6:
            continue
        terminal_direction /= terminal_norm

        handle_pool: list[MaskCandidate] = []
        for candidate in owned:
            if candidate.semantic_name != handle_part.semantic_name:
                continue
            constrained = candidate.mask.astype(bool) & root_mask
            distance = _minimum_mask_distance(pivot_mask, constrained)
            if distance > config.maximum_handle_pivot_distance_ratio * diagonal:
                continue
            vector = _centroid(constrained) - pivot_center
            norm = float(np.linalg.norm(vector))
            directional_cosine = (
                float((vector / norm) @ terminal_direction) if norm > 1.0 else -1.0
            )
            if directional_cosine > 0.40:
                continue
            handle_pool.append(replace(candidate, mask=constrained))
        handles = _deduplicate(
            handle_pool,
            maximum_count=max(2, handle_part.maximum_instances),
        )
        if not handles:
            rows.append(
                {
                    "root_key": root_key,
                    "profile": profile,
                    "status": "missing_opposite_handle",
                }
            )
            continue

        terminal_masks = [candidate.mask.astype(bool) for candidate in terminals]
        joint_support = pivot_mask | np.logical_or.reduce(terminal_masks)
        refined_handles: list[MaskCandidate] = []
        ring_refinement_count = 0
        ring_widths: list[int] = []
        for handle in handles:
            refined_mask = handle.mask.astype(bool)
            refined_as_ring = False
            ring_width = 0
            if finger_hole_part is not None:
                refined_mask, refined_as_ring, ring_width = _refine_handle_ring(
                    refined_mask,
                    joint_support,
                    diagonal=diagonal,
                    config=config,
                )
            metadata = {
                **handle.metadata,
                "root_geometry_support": True,
                "handle_ring_refined": refined_as_ring,
                "handle_ring_width_px": ring_width,
                "ground_truth_used": False,
            }
            refined = replace(
                handle,
                mask=refined_mask,
                source_reliability=max(
                    handle.source_reliability, config.source_reliability
                ),
                metadata=metadata,
            )
            refined_handles.append(refined)
            candidate_replacements[_candidate_key(handle)] = refined
            if refined_as_ring:
                ring_refinement_count += 1
                ring_widths.append(ring_width)

        refined_terminals: list[MaskCandidate] = []
        for terminal in terminals:
            refined = (
                _relabel_terminal(terminal, terminal_part, root)
                if terminal.semantic_name != terminal_part.semantic_name
                else terminal
            )
            refined = replace(
                refined,
                metadata={
                    **refined.metadata,
                    "root_geometry_support": True,
                    "ground_truth_used": False,
                },
            )
            refined_terminals.append(refined)
            candidate_replacements[_candidate_key(terminal)] = refined

        support_masks = [
            pivot_mask,
            *(candidate.mask.astype(bool) for candidate in refined_terminals),
            *(candidate.mask.astype(bool) for candidate in refined_handles),
        ]
        for candidate in owned:
            if not (
                candidate.metadata.get("generic_visual_region")
                and candidate.metadata.get("visual_region_kind") == "panel"
            ):
                continue
            constrained = candidate.mask.astype(bool) & root_mask
            fraction = int(np.count_nonzero(constrained)) / root_area
            if (
                fraction <= 0.18
                and _minimum_mask_distance(pivot_mask, constrained)
                <= config.maximum_panel_pivot_distance_ratio * diagonal
            ):
                support_masks.append(constrained)

        support = np.logical_or.reduce(support_masks) & root_mask
        radius = max(1, round(diagonal * 0.004))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        support = cv2.morphologyEx(
            support.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)
        support |= (
            cv2.dilate(support.astype(np.uint8), kernel).astype(bool) & root_mask
        )
        refined_area = int(np.count_nonzero(support))
        refined_fraction = refined_area / root_area
        if not (
            config.minimum_refined_root_fraction
            <= refined_fraction
            <= config.maximum_refined_root_fraction
        ):
            rows.append(
                {
                    "root_key": root_key,
                    "profile": profile,
                    "status": "unsafe_refined_area",
                    "refined_root_fraction": refined_fraction,
                }
            )
            continue

        replacements[root_key] = replace(
            root,
            mask=support,
            metadata={
                **root.metadata,
                "box_xyxy": _bbox(support),
                "root_geometry_refined": True,
                "root_geometry_algorithm": "hinged-part-graph-v1",
                "root_geometry_original_area_px": root_area,
                "root_geometry_refined_area_px": refined_area,
                "root_geometry_refined_fraction": refined_fraction,
                "root_geometry_pivot_candidate_key": _candidate_key(pivot),
                "ground_truth_used": False,
            },
        )
        rows.append(
            {
                "root_key": root_key,
                "profile": profile,
                "status": "refined",
                "pivot_candidate_key": _candidate_key(pivot),
                "terminal_candidate_keys": [
                    _candidate_key(candidate) for candidate in refined_terminals
                ],
                "handle_candidate_keys": [
                    _candidate_key(candidate) for candidate in refined_handles
                ],
                "ring_refinement_count": ring_refinement_count,
                "ring_widths_px": ring_widths,
                "original_area_px": root_area,
                "refined_area_px": refined_area,
                "refined_root_fraction": refined_fraction,
            }
        )

    output: list[MaskCandidate] = []
    for candidate in candidates:
        key = _candidate_key(candidate)
        root_key = _root_key(candidate)
        if _is_root(candidate) and root_key in replacements:
            output.append(replacements[root_key])
        elif key in candidate_replacements:
            output.append(candidate_replacements[key])
        else:
            output.append(candidate)
    return RootGeometryResult(
        tuple(output),
        {
            "algorithm": "hpid-profile-root-geometry-v1",
            "root_count": len(roots),
            "eligible_root_count": len(rows),
            "refined_root_count": len(replacements),
            "candidate_replacement_count": len(candidate_replacements),
            "roots": rows,
            "ground_truth_used": False,
        },
    )
