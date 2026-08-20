from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot

import cv2
import numpy as np

from .fusion import MaskCandidate


@dataclass(frozen=True)
class RootCleanupConfig:
    minimum_detached_area_ratio: float = 0.005
    maximum_detached_distance_ratio: float = 0.03
    minimum_support_overlap: float = 0.28


@dataclass(frozen=True)
class RootCleanupResult:
    candidates: tuple[MaskCandidate, ...]
    diagnostics: dict[str, object]


def _root_key(candidate: MaskCandidate) -> str:
    return (
        f"{candidate.metadata.get('root_origin', 'legacy')}::"
        f"{candidate.metadata.get('root_index', 'unknown')}"
    )


def _candidate_key(candidate: MaskCandidate) -> str:
    return str(candidate.metadata.get("candidate_key", ""))


def _is_root(candidate: MaskCandidate) -> bool:
    return bool(
        candidate.semantic_name == candidate.semantic_parent
        and candidate.metadata.get("root_index") is not None
        and candidate.metadata.get("parent_candidate_key") is None
    )


def _touches_frame(mask: np.ndarray) -> bool:
    return bool(
        np.any(mask[0])
        or np.any(mask[-1])
        or np.any(mask[:, 0])
        or np.any(mask[:, -1])
    )


def _source_family(candidate: MaskCandidate) -> str:
    explicit = str(candidate.metadata.get("source_family", "")).strip()
    if explicit:
        return explicit
    return candidate.source.rsplit("/", maxsplit=1)[0]


def _is_semantic_support(candidate: MaskCandidate, root: MaskCandidate) -> bool:
    if candidate.semantic_name == root.semantic_name:
        return False
    if bool(candidate.metadata.get("guided_prompt")):
        return True
    if bool(candidate.metadata.get("generic_visual_region")):
        return False
    return "_visual_" not in candidate.semantic_name


def _component_support(
    component: np.ndarray,
    root: MaskCandidate,
    scoped_candidates: list[MaskCandidate],
    minimum_overlap: float,
) -> tuple[bool, list[str]]:
    area = max(1, int(np.count_nonzero(component)))
    families: set[str] = set()
    semantic = False
    evidence: list[str] = []
    for candidate in scoped_candidates:
        if _candidate_key(candidate) == _candidate_key(root):
            continue
        overlap = int(np.count_nonzero(candidate.mask & component)) / area
        if overlap < minimum_overlap:
            continue
        families.add(_source_family(candidate))
        semantic |= _is_semantic_support(candidate, root)
        evidence.append(candidate.semantic_name)
    return bool(semantic or len(families) >= 2), sorted(set(evidence))


def _distance_to_main(component: np.ndarray, main: np.ndarray) -> float:
    # OpenCV's distance transform reports the distance to zero pixels.  The
    # inverse main mask therefore gives every secondary pixel its distance to
    # the retained subject component.
    distance = cv2.distanceTransform((~main).astype(np.uint8), cv2.DIST_L2, 5)
    values = distance[component]
    return float(values.min()) if len(values) else float("inf")


def _clean_root(
    root: MaskCandidate,
    scoped_candidates: list[MaskCandidate],
    target_point_xy: tuple[float, float] | None,
    config: RootCleanupConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    mask = root.mask.astype(bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    components = [
        (int(stats[index, cv2.CC_STAT_AREA]), index)
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) > 0
    ]
    if not components:
        return mask, {
            "root_key": _root_key(root),
            "status": "empty_root_kept",
            "removed_component_count": 0,
        }

    target_component = None
    if target_point_xy is not None:
        x = round(target_point_xy[0])
        y = round(target_point_xy[1])
        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
            label = int(labels[y, x])
            if label > 0:
                target_component = label
    main_index = target_component or max(components)[1]
    main = labels == main_index
    main_area = max(1, int(np.count_nonzero(main)))
    diagonal = max(1.0, hypot(mask.shape[0], mask.shape[1]))
    clean = main.copy()
    component_rows: list[dict[str, object]] = []
    for area, index in sorted(components, reverse=True):
        if index == main_index:
            continue
        component = labels == index
        distance = _distance_to_main(component, main)
        supported, evidence = _component_support(
            component,
            root,
            scoped_candidates,
            config.minimum_support_overlap,
        )
        frame_contact = _touches_frame(component)
        keep = bool(
            not frame_contact
            and area / main_area >= config.minimum_detached_area_ratio
            and distance / diagonal <= config.maximum_detached_distance_ratio
            and supported
        )
        if keep:
            clean |= component
        component_rows.append(
            {
                "area_px": area,
                "area_to_main": float(area / main_area),
                "distance_px": distance,
                "distance_to_diagonal": float(distance / diagonal),
                "touches_frame": frame_contact,
                "independent_support": supported,
                "support_semantics": evidence,
                "kept": keep,
            }
        )
    return clean, {
        "root_key": _root_key(root),
        "status": "cleaned" if not np.array_equal(clean, mask) else "unchanged",
        "original_area_px": int(np.count_nonzero(mask)),
        "clean_area_px": int(np.count_nonzero(clean)),
        "main_component_area_px": main_area,
        "component_count": len(components),
        "removed_component_count": sum(not row["kept"] for row in component_rows),
        "components": component_rows,
        "target_component_used": target_component is not None,
    }


def clean_primary_roots(
    candidates: list[MaskCandidate] | tuple[MaskCandidate, ...],
    roots: list[MaskCandidate] | tuple[MaskCandidate, ...],
    *,
    target_point_xy: tuple[float, float] | None = None,
    config: RootCleanupConfig | None = None,
) -> RootCleanupResult:
    """Remove disconnected frame/UI debris and clip children to clean roots."""

    config = config or RootCleanupConfig()
    roots_by_key = {_root_key(root): root for root in roots}
    clean_masks: dict[str, np.ndarray] = {}
    root_rows: list[dict[str, object]] = []
    for root_key, root in roots_by_key.items():
        scoped = [candidate for candidate in candidates if _root_key(candidate) == root_key]
        clean, row = _clean_root(
            root,
            scoped,
            target_point_xy,
            config,
        )
        clean_masks[root_key] = clean
        root_rows.append(row)

    output: list[MaskCandidate] = []
    clipped_count = 0
    dropped_count = 0
    for candidate in candidates:
        root_key = _root_key(candidate)
        clean = clean_masks.get(root_key)
        if clean is None:
            output.append(candidate)
            continue
        clipped = clean if _is_root(candidate) else candidate.mask & clean
        if not np.any(clipped):
            dropped_count += 1
            continue
        if not np.array_equal(clipped, candidate.mask):
            clipped_count += 1
            metadata = {
                **candidate.metadata,
                "root_cleanup_clipped": True,
                "root_cleanup_original_area_px": int(np.count_nonzero(candidate.mask)),
                "root_cleanup_area_px": int(np.count_nonzero(clipped)),
            }
            output.append(replace(candidate, mask=clipped, metadata=metadata))
        else:
            output.append(candidate)
    return RootCleanupResult(
        tuple(output),
        {
            "algorithm": "hpid-connected-root-cleanup-v1",
            "root_count": len(root_rows),
            "clipped_candidate_count": clipped_count,
            "dropped_candidate_count": dropped_count,
            "roots": root_rows,
            "ground_truth_used": False,
        },
    )
