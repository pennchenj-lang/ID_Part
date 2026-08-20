from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace

import cv2
import numpy as np
from PIL import Image

from .fusion import MaskCandidate, mask_iou


@dataclass(frozen=True)
class MaskRefinementConfig:
    grabcut_iterations: int = 3
    maximum_grabcut_candidates: int | None = None
    trimap_radius_factor: float = 0.018
    maximum_detail_radius_px: int = 2
    maximum_part_radius_px: int = 5
    maximum_root_radius_px: int = 7
    minimum_area_ratio: float = 0.70
    maximum_area_ratio: float = 1.30
    minimum_overlap_iou: float = 0.70
    minimum_edge_score_ratio: float = 0.90
    standard_hole_fraction: float = 0.00025
    detail_hole_fraction: float = 0.00003
    standard_component_fraction: float = 0.00004
    detail_component_fraction: float = 0.000008


@dataclass(frozen=True)
class MaskRefinementResult:
    candidates: tuple[MaskCandidate, ...]
    diagnostics: dict[str, object]


_DETAIL_TOKENS = (
    "eye",
    "eyebrow",
    "eyelash",
    "button",
    "lace",
    "cuff",
    "sole",
    "heel",
    "label",
    "trigger",
    "port",
)


def _is_detail(candidate: MaskCandidate) -> bool:
    if bool(candidate.metadata.get("dense_semantic_fallback")):
        return True
    if candidate.metadata.get("visual_region_kind") in {"detail", "strip"}:
        return True
    if int(candidate.metadata.get("hierarchy_depth", 0)) >= 3:
        return True
    return any(token in candidate.semantic_name for token in _DETAIL_TOKENS)


def _edge_strength(image_bgr: np.ndarray) -> np.ndarray:
    smoothed = cv2.GaussianBlur(image_bgr, (0, 0), 0.65)
    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB).astype(np.float32)
    magnitudes: list[np.ndarray] = []
    for channel in cv2.split(lab):
        gradient_x = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        magnitudes.append(cv2.magnitude(gradient_x, gradient_y))
    return np.maximum.reduce(magnitudes)


def _edge_alignment_score(mask: np.ndarray, edge_strength: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    kernel = np.ones((3, 3), dtype=np.uint8)
    binary = mask.astype(np.uint8)
    boundary = cv2.dilate(binary, kernel).astype(bool) ^ cv2.erode(
        binary, kernel
    ).astype(bool)
    return float(edge_strength[boundary].mean()) if np.any(boundary) else 0.0


def _mask_box(mask: np.ndarray, padding: int) -> tuple[int, int, int, int]:
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0, 0, 0, 0
    return (
        max(0, int(xs.min()) - padding),
        max(0, int(ys.min()) - padding),
        min(width, int(xs.max() + 1) + padding),
        min(height, int(ys.max() + 1) + padding),
    )


def _fill_small_holes(mask: np.ndarray, maximum_area: int) -> np.ndarray:
    output = mask.copy()
    height, width = output.shape
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        (~output).astype(np.uint8), 8
    )
    for component_id in range(1, count):
        x, y, component_width, component_height, area = stats[component_id]
        touches_border = (
            x == 0
            or y == 0
            or x + component_width == width
            or y + component_height == height
        )
        if not touches_border and int(area) <= maximum_area:
            output[components == component_id] = True
    return output


def _remove_small_components(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    if count <= 2:
        return mask
    output = mask.copy()
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < minimum_area and area < largest:
            output[components == component_id] = False
    return output


def _candidate_root_key(candidate: MaskCandidate) -> tuple[str, str] | None:
    origin = candidate.metadata.get("root_origin")
    index = candidate.metadata.get("root_index")
    if origin is None or index is None:
        return None
    return str(origin), str(index)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    component_id = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == component_id


def _reconcile_axial_partitions(
    candidates: list[MaskCandidate],
) -> tuple[list[MaskCandidate], dict[str, object]]:
    algorithm = "profile-constrained-silhouette-axial-partition-v1"
    roots: dict[tuple[str, str], int] = {}
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, candidate in enumerate(candidates):
        key = _candidate_root_key(candidate)
        if key is None:
            continue
        if candidate.semantic_name == candidate.semantic_parent:
            roots[key] = index
        if candidate.metadata.get("structural_fusion_algorithm") == algorithm:
            groups[key].append(index)

    output = list(candidates)
    rows: list[dict[str, object]] = []
    for key, child_indices in groups.items():
        root_index = roots.get(key)
        if root_index is None or len(child_indices) != 2:
            continue
        root = output[root_index]
        children = [output[index] for index in child_indices]
        seeds = [_largest_component(child.mask.astype(bool)) for child in children]
        if any(not np.any(seed) for seed in seeds):
            continue

        supported = seeds[0] | seeds[1]
        component_count, root_labels, _, _ = cv2.connectedComponentsWithStats(
            root.mask.astype(np.uint8), 8
        )
        reconciled_root = np.zeros_like(root.mask, dtype=bool)
        for component_id in range(1, component_count):
            component = root_labels == component_id
            if np.any(component & supported):
                reconciled_root |= component
        if not np.any(reconciled_root):
            continue

        seeds = [seed & reconciled_root for seed in seeds]
        if any(not np.any(seed) for seed in seeds):
            continue
        exclusive_first = seeds[0] & ~seeds[1]
        exclusive_second = seeds[1] & ~seeds[0]
        undecided = reconciled_root & ~(exclusive_first | exclusive_second)
        distance_first = cv2.distanceTransform(
            (~seeds[0]).astype(np.uint8), cv2.DIST_L2, 3
        )
        distance_second = cv2.distanceTransform(
            (~seeds[1]).astype(np.uint8), cv2.DIST_L2, 3
        )
        first_mask = exclusive_first | (
            undecided & (distance_first <= distance_second)
        )
        second_mask = reconciled_root & ~first_mask
        if not np.any(first_mask) or not np.any(second_mask):
            continue

        removed_root_pixels = int(
            np.count_nonzero(root.mask) - np.count_nonzero(reconciled_root)
        )
        root_evidence = {
            "algorithm": "hpid-axial-partition-reconciliation-v1",
            "sibling_count": 2,
            "removed_unsupported_root_pixels": removed_root_pixels,
            "reconciled_root_area_px": int(np.count_nonzero(reconciled_root)),
            "ground_truth_used": False,
        }
        output[root_index] = replace(
            root,
            mask=reconciled_root,
            metadata={
                **root.metadata,
                "structural_partition_reconciliation": root_evidence,
            },
        )
        for child_index, child_mask in zip(
            child_indices, (first_mask, second_mask), strict=True
        ):
            child = output[child_index]
            child_evidence = {
                **root_evidence,
                "assigned_area_px": int(np.count_nonzero(child_mask)),
                "peer_semantic": child.metadata.get("structural_peer_semantic"),
            }
            output[child_index] = replace(
                child,
                mask=child_mask,
                metadata={
                    **child.metadata,
                    "structural_partition_reconciliation": child_evidence,
                },
            )
        rows.append(
            {
                "root_origin": key[0],
                "root_index": key[1],
                **root_evidence,
            }
        )
    return output, {
        "algorithm": "hpid-axial-partition-reconciliation-v1",
        "reconciled_partition_count": len(rows),
        "removed_unsupported_root_pixels": sum(
            int(row["removed_unsupported_root_pixels"]) for row in rows
        ),
        "partitions": rows,
        "ground_truth_used": False,
    }


def _refine_one(
    image_bgr: np.ndarray,
    candidate: MaskCandidate,
    config: MaskRefinementConfig,
    *,
    allow_grabcut: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    original = candidate.mask.astype(bool)
    area = int(np.count_nonzero(original))
    image_area = int(original.size)
    detail = _is_detail(candidate)
    root = candidate.semantic_name == candidate.semantic_parent
    maximum_radius = (
        config.maximum_detail_radius_px
        if detail
        else (
            config.maximum_root_radius_px if root else config.maximum_part_radius_px
        )
    )
    radius = max(
        1,
        min(maximum_radius, round(np.sqrt(area) * config.trimap_radius_factor)),
    )
    padding = radius * 4 + 6
    x0, y0, x1, y1 = _mask_box(original, padding)
    local_mask = original[y0:y1, x0:x1]
    local_image = image_bgr[y0:y1, x0:x1]
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    eroded = cv2.erode(local_mask.astype(np.uint8), kernel).astype(bool)
    dilated = cv2.dilate(local_mask.astype(np.uint8), kernel).astype(bool)
    grabcut_used = False
    accepted = False
    refined_local = local_mask.copy()
    proposed_area_ratio = 1.0
    proposed_iou = 1.0
    local_edges = _edge_strength(local_image)
    input_edge_score = _edge_alignment_score(local_mask, local_edges)
    proposed_edge_score = input_edge_score
    proposed_edge_ratio = 1.0
    edge_guard_passed = True
    if allow_grabcut and eroded.any() and np.count_nonzero(dilated & ~eroded) >= 8:
        trimap = np.full(local_mask.shape, cv2.GC_BGD, dtype=np.uint8)
        trimap[dilated] = cv2.GC_PR_BGD
        trimap[local_mask] = cv2.GC_PR_FGD
        trimap[eroded] = cv2.GC_FGD
        background_model = np.zeros((1, 65), dtype=np.float64)
        foreground_model = np.zeros((1, 65), dtype=np.float64)
        try:
            cv2.grabCut(
                local_image,
                trimap,
                None,
                background_model,
                foreground_model,
                config.grabcut_iterations,
                cv2.GC_INIT_WITH_MASK,
            )
            grabcut_used = True
            proposal = (trimap == cv2.GC_FGD) | (trimap == cv2.GC_PR_FGD)
            proposed_area_ratio = np.count_nonzero(proposal) / max(1, area)
            proposed_iou = mask_iou(proposal, local_mask)
            proposed_edge_score = _edge_alignment_score(proposal, local_edges)
            proposed_edge_ratio = (proposed_edge_score + 1.0) / (
                input_edge_score + 1.0
            )
            edge_guard_passed = (
                proposed_edge_ratio >= config.minimum_edge_score_ratio
            )
            accepted = (
                config.minimum_area_ratio
                <= proposed_area_ratio
                <= config.maximum_area_ratio
                and proposed_iou >= config.minimum_overlap_iou
                and edge_guard_passed
            )
            if accepted:
                refined_local = proposal
        except cv2.error:
            grabcut_used = False

    if not detail:
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        refined_local = cv2.morphologyEx(
            refined_local.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel
        ).astype(bool)
    hole_fraction = (
        config.detail_hole_fraction if detail else config.standard_hole_fraction
    )
    refined_local = _fill_small_holes(
        refined_local, max(8, round(image_area * hole_fraction))
    )
    component_fraction = (
        config.detail_component_fraction
        if detail
        else config.standard_component_fraction
    )
    refined_local = _remove_small_components(
        refined_local, max(6, round(image_area * component_fraction))
    )
    output = np.zeros_like(original)
    output[y0:y1, x0:x1] = refined_local
    return output, {
        "detail_mode": bool(detail),
        "radius_px": radius,
        "grabcut_used": bool(grabcut_used),
        "grabcut_scheduled": bool(allow_grabcut),
        "grabcut_accepted": bool(accepted),
        "proposed_area_ratio": float(proposed_area_ratio),
        "proposed_iou": float(proposed_iou),
        "input_edge_score": float(input_edge_score),
        "proposed_edge_score": float(proposed_edge_score),
        "proposed_edge_score_ratio": float(proposed_edge_ratio),
        "edge_guard_passed": bool(edge_guard_passed),
        "input_area_px": area,
        "output_area_px": int(np.count_nonzero(output)),
    }


def refine_candidate_masks(
    image: Image.Image,
    candidates: list[MaskCandidate],
    *,
    config: MaskRefinementConfig | None = None,
) -> MaskRefinementResult:
    """Snap candidate boundaries to the image and remove mask artifacts.

    Refinement is constrained to a narrow band around each proposal. Area and
    overlap guards reject unstable GrabCut updates, so this stage cannot invent
    a distant part or silently replace the detector's semantic hypothesis.
    """

    config = config or MaskRefinementConfig()
    if config.grabcut_iterations < 1:
        raise ValueError("grabcut_iterations must be positive")
    if (
        config.maximum_grabcut_candidates is not None
        and config.maximum_grabcut_candidates < 0
    ):
        raise ValueError("maximum_grabcut_candidates must be non-negative")
    image_bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    if config.maximum_grabcut_candidates is None:
        scheduled_indices = set(range(len(candidates)))
    else:
        ranked_indices = sorted(
            range(len(candidates)),
            key=lambda index: (
                not bool(candidates[index].metadata.get("generic_visual_region")),
                candidates[index].semantic_name
                != candidates[index].semantic_parent,
                candidates[index].score * candidates[index].source_reliability,
                int(np.count_nonzero(candidates[index].mask)),
            ),
            reverse=True,
        )
        scheduled_indices = set(
            ranked_indices[: config.maximum_grabcut_candidates]
        )
    output: list[MaskCandidate] = []
    rows: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates):
        mask, row = _refine_one(
            image_bgr,
            candidate,
            config,
            allow_grabcut=index in scheduled_indices,
        )
        if not np.any(mask):
            mask = candidate.mask.copy()
            row["empty_result_reverted"] = True
        metadata = dict(candidate.metadata)
        metadata["mask_refinement"] = row
        output.append(
            MaskCandidate(
                semantic_name=candidate.semantic_name,
                semantic_parent=candidate.semantic_parent,
                mask=mask,
                score=candidate.score,
                source=candidate.source,
                prompt=candidate.prompt,
                source_reliability=candidate.source_reliability,
                metadata=metadata,
            )
        )
        rows.append(
            {
                "semantic_name": candidate.semantic_name,
                "source": candidate.source,
                **row,
            }
        )
    output, partition_reconciliation = _reconcile_axial_partitions(output)
    return MaskRefinementResult(
        tuple(output),
        {
            "algorithm": "hpid-image-guided-mask-refinement-v1",
            "candidate_count": len(candidates),
            "grabcut_schedule_limit": config.maximum_grabcut_candidates,
            "grabcut_scheduled_count": len(scheduled_indices),
            "grabcut_attempt_count": sum(bool(row["grabcut_used"]) for row in rows),
            "grabcut_accept_count": sum(
                bool(row["grabcut_accepted"]) for row in rows
            ),
            "edge_guard_rejection_count": sum(
                bool(row["grabcut_used"]) and not bool(row["edge_guard_passed"])
                for row in rows
            ),
            "changed_candidate_count": sum(
                int(row["input_area_px"]) != int(row["output_area_px"])
                for row in rows
            ),
            "candidates": rows,
            "structural_partition_reconciliation": partition_reconciliation,
            "ground_truth_used": False,
        },
    )
