from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .visual_regions import VisualMaskProposal


@dataclass(frozen=True)
class SceneInstanceConfig:
    """Fast, category-independent instance partitioning for layered scenes."""

    analysis_maximum_dimension: int = 640
    color_cluster_count: int = 8
    minimum_surface_cluster_fraction: float = 0.045
    minimum_surface_component_fraction: float = 0.022
    minimum_surface_span_fraction: float = 0.42
    object_color_distance: float = 20.0
    minimum_object_fraction: float = 0.00022
    minimum_seed_containment: float = 0.72
    minimum_seed_area_fraction: float = 0.00018
    maximum_seed_area_fraction: float = 0.24
    minimum_peak_distance: int = 8
    peak_distance_fraction: float = 0.014
    maximum_markers: int = 56
    maximum_partitions: int = 48
    gradient_weight: float = 0.78
    distance_weight: float = 0.22
    minimum_surface_merge_contact: float = 0.40
    maximum_surface_merge_lab_distance: float = 76.0
    strong_surface_merge_contact: float = 0.58
    strong_surface_merge_lab_distance: float = 96.0
    maximum_merged_object_fraction: float = 0.10
    minimum_structural_face_contact: float = 0.06
    minimum_structural_horizontal_overlap: float = 0.82
    minimum_structural_vertical_overlap: float = 0.55
    minimum_structural_face_aspect_ratio: float = 1.30
    maximum_structural_face_area_ratio: float = 0.68
    maximum_structural_face_lab_distance: float = 100.0
    strong_projected_face_contact: float = 0.28
    strong_projected_horizontal_overlap: float = 0.80
    strong_projected_vertical_overlap: float = 0.55
    strong_projected_maximum_area_ratio: float = 0.50
    strong_projected_maximum_lab_distance: float = 150.0
    strong_projected_maximum_merged_fraction: float = 0.24
    minimum_shared_envelope_support: float = 0.72
    minimum_shared_envelope_precision: float = 0.52
    maximum_structural_merged_fraction: float = 0.13


@dataclass(frozen=True)
class SceneInstancePartition:
    proposal: VisualMaskProposal
    seed_index: int | None


@dataclass(frozen=True)
class SceneInstanceResult:
    partitions: tuple[SceneInstancePartition, ...]
    replaced_seed_indices: frozenset[int]
    diagnostics: dict[str, object]


def _area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def _box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)


def _largest_component_geometry(mask: np.ndarray) -> tuple[int, float]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return 0, 0.0
    largest_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    largest_area = int(stats[largest_index, cv2.CC_STAT_AREA])
    width = int(stats[largest_index, cv2.CC_STAT_WIDTH])
    height = int(stats[largest_index, cv2.CC_STAT_HEIGHT])
    return largest_area, float(max(width, height))


def _surface_model(
    lab: np.ndarray,
    envelope: np.ndarray,
    config: SceneInstanceConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]] | None:
    pixels = lab[envelope]
    if len(pixels) < 96:
        return None
    cluster_count = min(config.color_cluster_count, max(2, len(pixels) // 48))
    cv2.setRNGSeed(7341)
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        24,
        0.35,
    )
    _, sampled_labels, centers = cv2.kmeans(
        pixels.astype(np.float32),
        cluster_count,
        None,
        criteria,
        1,
        cv2.KMEANS_PP_CENTERS,
    )
    distances = np.linalg.norm(
        lab[:, :, None, :] - centers[None, None, :, :], axis=3
    )
    labels = distances.argmin(axis=2).astype(np.int16)
    labels[~envelope] = -1
    envelope_area = max(1, _area(envelope))
    ex0, ey0, ex1, ey1 = _box(envelope)
    envelope_span = max(1, ex1 - ex0, ey1 - ey0)
    ranked: list[tuple[float, int, int, int, float]] = []
    for cluster_index in range(cluster_count):
        cluster = labels == cluster_index
        total = _area(cluster)
        largest, span = _largest_component_geometry(cluster)
        score = largest + 0.25 * total
        ranked.append((score, cluster_index, largest, total, span))
    _, surface_index, largest, total, span = max(ranked)
    total_fraction = total / envelope_area
    largest_fraction = largest / envelope_area
    span_fraction = span / envelope_span
    if (
        total_fraction < config.minimum_surface_cluster_fraction
        or largest_fraction < config.minimum_surface_component_fraction
        or span_fraction < config.minimum_surface_span_fraction
    ):
        return None
    surface_center = centers[surface_index]
    surface_distance = np.linalg.norm(lab - surface_center, axis=2)
    within_surface = surface_distance[labels == surface_index]
    robust_spread = (
        float(np.quantile(within_surface, 0.92)) if within_surface.size else 0.0
    )
    threshold = max(config.object_color_distance, 1.55 * robust_spread)
    return surface_distance, labels == surface_index, {
        "surface_cluster_index": int(surface_index),
        "surface_cluster_fraction": float(total_fraction),
        "surface_component_fraction": float(largest_fraction),
        "surface_span_fraction": float(span_fraction),
        "surface_lab": [float(value) for value in surface_center],
        "surface_spread": robust_spread,
        "object_distance_threshold": float(threshold),
        "sampled_label_count": len(sampled_labels),
    }


def _lab_gradient(lab: np.ndarray, envelope: np.ndarray) -> np.ndarray:
    gradients: list[np.ndarray] = []
    for channel in cv2.split(lab):
        dx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        gradients.append(cv2.magnitude(dx, dy))
    gradient = np.maximum.reduce(gradients)
    values = gradient[envelope]
    reference = float(np.quantile(values, 0.90)) if values.size else 1.0
    return np.clip(gradient / max(1.0, reference), 0.0, 2.0)


def _marker_point(mask: np.ndarray) -> tuple[int, int, float] | None:
    if not np.any(mask):
        return None
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    flat_index = int(np.argmax(distance))
    y, x = np.unravel_index(flat_index, distance.shape)
    return int(y), int(x), float(distance[y, x])


def _merge_adjacent_object_surfaces(
    labels: np.ndarray,
    lab: np.ndarray,
    object_mask: np.ndarray,
    config: SceneInstanceConfig,
    *,
    seed_masks: tuple[np.ndarray, ...] = (),
) -> tuple[np.ndarray, dict[str, object]]:
    active = [int(value) for value in np.unique(labels) if value > 0]
    if len(active) < 2:
        return labels, {"merge_count": 0, "merge_rows": []}

    masks = {label: labels == label for label in active}
    areas = {label: _area(mask) for label, mask in masks.items()}
    perimeters = {
        label: _area(
            mask
            & ~cv2.erode(
                mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
            ).astype(bool)
        )
        for label, mask in masks.items()
    }
    medians = {label: np.median(lab[mask], axis=0) for label, mask in masks.items()}
    boxes = {label: _box(mask) for label, mask in masks.items()}
    contacts: dict[tuple[int, int], int] = {}
    for first, second in (
        (labels[:, :-1], labels[:, 1:]),
        (labels[:-1], labels[1:]),
    ):
        changed = (first != second) & (first > 0) & (second > 0)
        for left, right in zip(first[changed], second[changed], strict=True):
            pair = tuple(sorted((int(left), int(right))))
            contacts[pair] = contacts.get(pair, 0) + 1

    parent = {label: label for label in active}

    def find(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    rows: list[dict[str, object]] = []
    object_area = max(1, _area(object_mask))
    for (left, right), contact in sorted(
        contacts.items(), key=lambda item: item[1], reverse=True
    ):
        contact_ratio = contact / max(1, min(perimeters[left], perimeters[right]))
        lab_distance = float(np.linalg.norm(medians[left] - medians[right]))
        merged_fraction = (areas[left] + areas[right]) / object_area
        smaller, larger = (
            (left, right) if areas[left] <= areas[right] else (right, left)
        )
        sx0, sy0, sx1, sy1 = boxes[smaller]
        lx0, ly0, lx1, ly1 = boxes[larger]
        horizontal_overlap = max(0, min(sx1, lx1) - max(sx0, lx0))
        horizontal_overlap_ratio = horizontal_overlap / max(
            1, min(sx1 - sx0, lx1 - lx0)
        )
        area_ratio = areas[smaller] / max(1, areas[larger])
        larger_height = max(1, ly1 - ly0)
        smaller_center_y = 0.5 * (sy0 + sy1)
        upper_structure = bool(
            smaller_center_y <= ly0 + 0.68 * larger_height
            and ly1 >= sy1 + 0.15 * larger_height
        )
        pair_union = masks[left] | masks[right]
        shared_envelope_support = 0.0
        shared_envelope_precision = 0.0
        for seed in seed_masks:
            seed = seed.astype(bool)
            left_support = _area(seed & masks[left]) / max(1, areas[left])
            right_support = _area(seed & masks[right]) / max(1, areas[right])
            support = min(left_support, right_support)
            precision = _area(seed & pair_union) / max(1, _area(seed))
            if (support, precision) > (
                shared_envelope_support,
                shared_envelope_precision,
            ):
                shared_envelope_support = float(support)
                shared_envelope_precision = float(precision)
        appearance_merge = bool(
            merged_fraction <= config.maximum_merged_object_fraction
            and (
                contact_ratio >= config.minimum_surface_merge_contact
                and lab_distance <= config.maximum_surface_merge_lab_distance
                or contact_ratio >= config.strong_surface_merge_contact
                and lab_distance <= config.strong_surface_merge_lab_distance
            )
        )
        structural_face_merge = bool(
            merged_fraction <= config.maximum_structural_merged_fraction
            and contact_ratio >= config.minimum_structural_face_contact
            and horizontal_overlap_ratio
            >= config.minimum_structural_horizontal_overlap
            and area_ratio <= config.maximum_structural_face_area_ratio
            and upper_structure
            and lab_distance <= config.maximum_structural_face_lab_distance
            and shared_envelope_support >= config.minimum_shared_envelope_support
            and shared_envelope_precision >= config.minimum_shared_envelope_precision
        )
        accepted = appearance_merge or structural_face_merge
        rows.append(
            {
                "labels": [left, right],
                "contact_ratio": float(contact_ratio),
                "lab_distance": lab_distance,
                "merged_fraction": float(merged_fraction),
                "horizontal_overlap_ratio": float(horizontal_overlap_ratio),
                "area_ratio": float(area_ratio),
                "upper_structure": upper_structure,
                "shared_envelope_support": shared_envelope_support,
                "shared_envelope_precision": shared_envelope_precision,
                "merge_reason": (
                    "appearance_continuity"
                    if appearance_merge
                    else "shared_envelope_structural_face"
                    if structural_face_merge
                    else None
                ),
                "accepted": accepted,
            }
        )
        if accepted:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

    groups: dict[int, list[int]] = {}
    for label in active:
        groups.setdefault(find(label), []).append(label)
    if len(groups) == len(active):
        return labels, {"merge_count": 0, "merge_rows": rows}
    merged = np.zeros_like(labels)
    for output_label, members in enumerate(groups.values(), start=1):
        merged[np.isin(labels, members)] = output_label
    return merged, {
        "merge_count": len(active) - len(groups),
        "pre_merge_partition_count": len(active),
        "post_merge_partition_count": len(groups),
        "merge_rows": rows,
    }


def _merge_projected_face_fragments(
    labels: np.ndarray,
    lab: np.ndarray,
    object_mask: np.ndarray,
    config: SceneInstanceConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    """Merge projected top/side faces without collapsing adjacent instances."""

    active = [int(value) for value in np.unique(labels) if value > 0]
    if len(active) < 2:
        return labels, {"merge_count": 0, "merge_rows": []}
    masks = {label: labels == label for label in active}
    areas = {label: _area(mask) for label, mask in masks.items()}
    boxes = {label: _box(mask) for label, mask in masks.items()}
    perimeters = {
        label: _area(
            mask
            & ~cv2.erode(
                mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
            ).astype(bool)
        )
        for label, mask in masks.items()
    }
    medians = {label: np.median(lab[mask], axis=0) for label, mask in masks.items()}
    contacts: dict[tuple[int, int], int] = {}
    for first, second in (
        (labels[:, :-1], labels[:, 1:]),
        (labels[:-1], labels[1:]),
    ):
        changed = (first != second) & (first > 0) & (second > 0)
        for left, right in zip(first[changed], second[changed], strict=True):
            pair = tuple(sorted((int(left), int(right))))
            contacts[pair] = contacts.get(pair, 0) + 1

    parent = {label: label for label in active}

    def find(label: int) -> int:
        while parent[label] != label:
            parent[label] = parent[parent[label]]
            label = parent[label]
        return label

    rows: list[dict[str, object]] = []
    provisional_edges: list[tuple[int, int, int, int, int, tuple[float, ...]]] = []
    object_area = max(1, _area(object_mask))
    for (left, right), contact in sorted(
        contacts.items(), key=lambda item: item[1], reverse=True
    ):
        smaller, larger = (
            (left, right) if areas[left] <= areas[right] else (right, left)
        )
        sx0, sy0, sx1, sy1 = boxes[smaller]
        lx0, ly0, lx1, ly1 = boxes[larger]
        smaller_width = max(1, sx1 - sx0)
        smaller_height = max(1, sy1 - sy0)
        larger_width = max(1, lx1 - lx0)
        larger_height = max(1, ly1 - ly0)
        horizontal_overlap = max(0, min(sx1, lx1) - max(sx0, lx0))
        vertical_overlap = max(0, min(sy1, ly1) - max(sy0, ly0))
        horizontal_overlap_ratio = horizontal_overlap / max(
            1, min(smaller_width, larger_width)
        )
        vertical_overlap_ratio = vertical_overlap / smaller_height
        contact_ratio = contact / max(
            1, min(perimeters[left], perimeters[right])
        )
        area_ratio = areas[smaller] / max(1, areas[larger])
        aspect_ratio = smaller_width / smaller_height
        lab_distance = float(np.linalg.norm(medians[left] - medians[right]))
        merged_fraction = (areas[left] + areas[right]) / object_area
        smaller_center_y = 0.5 * (sy0 + sy1)
        upper_structure = bool(
            smaller_center_y <= ly0 + 0.68 * larger_height
            and ly1 >= sy1 + 0.15 * larger_height
        )
        standard_face = bool(
            merged_fraction <= config.maximum_structural_merged_fraction
            and contact_ratio >= config.minimum_structural_face_contact
            and horizontal_overlap_ratio
            >= config.minimum_structural_horizontal_overlap
            and vertical_overlap_ratio >= config.minimum_structural_vertical_overlap
            and area_ratio <= config.maximum_structural_face_area_ratio
            and aspect_ratio >= config.minimum_structural_face_aspect_ratio
            and upper_structure
            and lab_distance <= config.maximum_structural_face_lab_distance
        )
        strong_projected_face = bool(
            merged_fraction <= config.strong_projected_maximum_merged_fraction
            and contact_ratio >= config.strong_projected_face_contact
            and horizontal_overlap_ratio
            >= config.strong_projected_horizontal_overlap
            and vertical_overlap_ratio >= config.strong_projected_vertical_overlap
            and area_ratio <= config.strong_projected_maximum_area_ratio
            and aspect_ratio >= config.minimum_structural_face_aspect_ratio
            and upper_structure
            and lab_distance <= config.strong_projected_maximum_lab_distance
        )
        provisional_accepted = standard_face or strong_projected_face
        rows.append(
            {
                "labels": [left, right],
                "contact_ratio": float(contact_ratio),
                "horizontal_overlap_ratio": float(horizontal_overlap_ratio),
                "vertical_overlap_ratio": float(vertical_overlap_ratio),
                "area_ratio": float(area_ratio),
                "face_aspect_ratio": float(aspect_ratio),
                "upper_structure": upper_structure,
                "lab_distance": lab_distance,
                "merged_fraction": float(merged_fraction),
                "merge_reason": (
                    "projected_face"
                    if standard_face
                    else "strong_projected_face"
                    if strong_projected_face
                    else None
                ),
                "provisional_accepted": provisional_accepted,
                "accepted": False,
            }
        )
        if provisional_accepted:
            provisional_edges.append(
                (
                    len(rows) - 1,
                    left,
                    right,
                    smaller,
                    larger,
                    (
                        float(contact_ratio),
                        float(vertical_overlap_ratio),
                        float(horizontal_overlap_ratio),
                        -float(lab_distance),
                    ),
                )
            )

    smaller_labels = {edge[3] for edge in provisional_edges}
    best_edge_by_smaller: dict[int, int] = {}
    for edge_index, edge in enumerate(provisional_edges):
        smaller = edge[3]
        current_index = best_edge_by_smaller.get(smaller)
        if current_index is None or edge[5] > provisional_edges[current_index][5]:
            best_edge_by_smaller[smaller] = edge_index
    for edge_index, (row_index, left, right, smaller, larger, _) in enumerate(
        provisional_edges
    ):
        if larger in smaller_labels:
            rows[row_index]["rejection_reason"] = "transitive_face_chain"
            continue
        if best_edge_by_smaller.get(smaller) != edge_index:
            rows[row_index]["rejection_reason"] = "weaker_competing_body"
            continue
        rows[row_index]["accepted"] = True
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    groups: dict[int, list[int]] = {}
    for label in active:
        groups.setdefault(find(label), []).append(label)
    if len(groups) == len(active):
        return labels, {"merge_count": 0, "merge_rows": rows}
    merged = np.zeros_like(labels)
    for output_label, members in enumerate(groups.values(), start=1):
        merged[np.isin(labels, members)] = output_label
    return merged, {
        "merge_count": len(active) - len(groups),
        "pre_merge_partition_count": len(active),
        "post_merge_partition_count": len(groups),
        "merge_rows": rows,
    }


def partition_scene_instances(
    image: Image.Image,
    envelope_proposal: VisualMaskProposal,
    seed_proposals: list[VisualMaskProposal],
    *,
    config: SceneInstanceConfig | None = None,
) -> SceneInstanceResult:
    """Partition repeated scene objects without category-specific rules.

    A dominant connected appearance cluster estimates the supporting scene
    surface. Existing segmentation proposals become trusted instance markers;
    distance peaks add markers only in uncovered object-like regions. A
    Lab-gradient watershed then separates touching objects while preserving the
    original scene envelope as the fallback layer.
    """

    config = config or SceneInstanceConfig()
    image = image.convert("RGB")
    full_envelope = envelope_proposal.mask.astype(bool)
    image_area = max(1, image.width * image.height)
    envelope_fraction = _area(full_envelope) / image_area
    if envelope_fraction < 0.24:
        return SceneInstanceResult(
            (),
            frozenset(),
            {
                "algorithm": "hpid-scene-instance-partition-v1",
                "status": "skipped_small_scene_envelope",
                "envelope_fraction": envelope_fraction,
                "ground_truth_used": False,
            },
        )

    scale = min(
        1.0,
        config.analysis_maximum_dimension / max(image.width, image.height),
    )
    width = max(32, round(image.width * scale))
    height = max(32, round(image.height * scale))
    rgb = cv2.resize(
        np.asarray(image), (width, height), interpolation=cv2.INTER_AREA
    )
    envelope = _resize_mask(full_envelope, width, height)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    surface = _surface_model(lab, envelope, config)
    if surface is None:
        return SceneInstanceResult(
            (),
            frozenset(),
            {
                "algorithm": "hpid-scene-instance-partition-v1",
                "status": "skipped_no_dominant_support_surface",
                "analysis_size": [width, height],
                "envelope_fraction": envelope_fraction,
                "ground_truth_used": False,
            },
        )
    surface_distance, _, surface_diagnostics = surface
    threshold = float(surface_diagnostics["object_distance_threshold"])
    appearance_foreground = envelope & (surface_distance >= threshold)
    appearance_foreground = cv2.morphologyEx(
        appearance_foreground.astype(np.uint8),
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    ).astype(bool)
    appearance_foreground = cv2.morphologyEx(
        appearance_foreground.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8),
    ).astype(bool)

    analysis_area = max(1, width * height)
    minimum_object_area = max(
        18, round(analysis_area * config.minimum_object_fraction)
    )
    seed_rows: list[tuple[int, VisualMaskProposal, np.ndarray]] = []
    seed_union = np.zeros(envelope.shape, dtype=bool)
    for seed_index, proposal in enumerate(seed_proposals):
        seed = _resize_mask(proposal.mask.astype(bool), width, height) & envelope
        seed_area = _area(seed)
        original_area = max(
            1, _area(_resize_mask(proposal.mask.astype(bool), width, height))
        )
        containment = seed_area / original_area
        fraction = seed_area / max(1, _area(envelope))
        if (
            seed_area < minimum_object_area
            or containment < config.minimum_seed_containment
            or not config.minimum_seed_area_fraction
            <= fraction
            <= config.maximum_seed_area_fraction
        ):
            continue
        seed_rows.append((seed_index, proposal, seed))
        seed_union |= seed

    object_mask = appearance_foreground | seed_union
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        object_mask.astype(np.uint8), connectivity=8
    )
    retained = np.zeros_like(object_mask)
    for component_index in range(1, count):
        component = components == component_index
        component_area = int(stats[component_index, cv2.CC_STAT_AREA])
        if component_area >= minimum_object_area or np.any(component & seed_union):
            retained |= component
    object_mask = retained
    if not np.any(object_mask):
        return SceneInstanceResult(
            (),
            frozenset(),
            {
                "algorithm": "hpid-scene-instance-partition-v1",
                "status": "skipped_empty_object_foreground",
                "analysis_size": [width, height],
                **surface_diagnostics,
                "ground_truth_used": False,
            },
        )

    marker_map = np.zeros(object_mask.shape, dtype=np.int32)
    marker_to_seed: dict[int, int] = {}
    marker_points: list[tuple[int, int, float]] = []
    marker_seed_masks: list[np.ndarray] = []
    replaced_seed_indices: set[int] = set()
    suppressed_seed_markers: list[dict[str, object]] = []
    marker_index = 0
    minimum_peak_distance = max(
        config.minimum_peak_distance,
        round(min(width, height) * config.peak_distance_fraction),
    )
    for seed_index, _, seed in sorted(
        seed_rows,
        key=lambda item: (item[1].score, _area(item[2])),
        reverse=True,
    ):
        point = _marker_point(seed & object_mask)
        if point is None:
            continue
        y, x, radius = point
        if marker_map[y, x] > 0:
            continue
        seed_area = max(1, _area(seed))
        duplicate_structure = None
        for selected_seed in marker_seed_masks:
            selected_area = max(1, _area(selected_seed))
            intersection = _area(seed & selected_seed)
            overlap_of_smaller = intersection / min(seed_area, selected_area)
            if overlap_of_smaller >= 0.08:
                duplicate_structure = overlap_of_smaller
                break
        if duplicate_structure is not None:
            suppressed_seed_markers.append(
                {
                    "seed_index": seed_index,
                    "overlap_of_smaller": float(duplicate_structure),
                    "reason": "overlapping_object_faces",
                }
            )
            continue
        nearest = min(
            (
                (
                    float(np.hypot(x - old_x, y - old_y)),
                    old_y,
                    old_x,
                    old_radius,
                )
                for old_y, old_x, old_radius in marker_points
            ),
            default=None,
        )
        if nearest is not None and nearest[0] < max(
            minimum_peak_distance,
            0.72 * (radius + nearest[3]),
        ):
            suppressed_seed_markers.append(
                {
                    "seed_index": seed_index,
                    "distance_px": nearest[0],
                    "radius_px": radius,
                    "nearest_radius_px": nearest[3],
                    "reason": "same_local_structure",
                }
            )
            continue
        marker_index += 1
        marker_map[y, x] = marker_index
        marker_to_seed[marker_index] = seed_index
        marker_points.append(point)
        marker_seed_masks.append(seed)
        replaced_seed_indices.add(seed_index)
        if marker_index >= config.maximum_markers:
            break

    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    distance = ndi.distance_transform_edt(object_mask)
    coordinates = peak_local_max(
        distance,
        min_distance=minimum_peak_distance,
        threshold_abs=2.5,
        labels=object_mask,
        num_peaks=config.maximum_markers,
        exclude_border=False,
    )
    protected_seed_area = cv2.dilate(
        seed_union.astype(np.uint8), np.ones((5, 5), dtype=np.uint8)
    ).astype(bool)
    for y_raw, x_raw in coordinates:
        if marker_index >= config.maximum_markers:
            break
        y = int(y_raw)
        x = int(x_raw)
        if protected_seed_area[y, x]:
            continue
        radius = float(distance[y, x])
        too_close = any(
            np.hypot(x - old_x, y - old_y)
            < max(minimum_peak_distance, 0.72 * (radius + old_radius))
            for old_y, old_x, old_radius in marker_points
        )
        if too_close:
            continue
        marker_index += 1
        marker_map[y, x] = marker_index
        marker_points.append((y, x, radius))

    if marker_index < 1:
        return SceneInstanceResult(
            (),
            frozenset(),
            {
                "algorithm": "hpid-scene-instance-partition-v1",
                "status": "skipped_no_instance_markers",
                "analysis_size": [width, height],
                **surface_diagnostics,
                "ground_truth_used": False,
            },
        )

    gradient = _lab_gradient(lab, envelope)
    normalized_distance = distance / max(1.0, float(distance.max()))
    elevation = (
        config.gradient_weight * gradient
        - config.distance_weight * normalized_distance
    )
    labels = watershed(
        elevation,
        marker_map,
        mask=object_mask,
        watershed_line=False,
    )
    labels, surface_merge_diagnostics = _merge_adjacent_object_surfaces(
        labels,
        lab,
        object_mask,
        config,
        seed_masks=tuple(seed for _, _, seed in seed_rows),
    )
    labels, projected_face_diagnostics = _merge_projected_face_fragments(
        labels,
        lab,
        object_mask,
        config,
    )
    surface_merge_diagnostics["projected_face_consolidation"] = (
        projected_face_diagnostics
    )

    partitions: list[SceneInstancePartition] = []
    rows: list[dict[str, object]] = []
    for label in (int(value) for value in np.unique(labels) if value > 0):
        mask = labels == label
        area = _area(mask)
        if area < minimum_object_area:
            continue
        overlapping_markers = marker_map[mask]
        seed_candidates = [
            marker_to_seed[int(marker)]
            for marker in np.unique(overlapping_markers)
            if int(marker) in marker_to_seed
        ]
        seed_index = seed_candidates[0] if seed_candidates else None
        mean_distance = float(surface_distance[mask].mean())
        if seed_index is None and mean_distance < 0.86 * threshold:
            continue
        full_mask = _resize_mask(mask, image.width, image.height) & full_envelope
        if _area(full_mask) < max(20, round(image_area * 0.00004)):
            continue
        boundary = mask & ~cv2.erode(
            mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
        ).astype(bool)
        boundary_alignment = float(
            np.clip(gradient[boundary].mean() / 1.15, 0.0, 1.0)
            if np.any(boundary)
            else 0.0
        )
        seed_score = (
            float(seed_proposals[seed_index].score)
            if seed_index is not None
            else 0.70
        )
        score = float(
            np.clip(0.62 * seed_score + 0.25 * boundary_alignment + 0.10, 0.0, 0.96)
        )
        proposal = VisualMaskProposal(
            mask=full_mask,
            score=score,
            bbox_xyxy=_box(full_mask),
            scale_level=0,
            view_id="scene-instance-partition",
            support_views=("scene-instance-partition",),
            support_levels=(0,),
            best_view_iou=0.0,
            boundary_alignment=boundary_alignment,
            source="hpid-scene-instance-partition/appearance-watershed",
            geometric_support=float(
                np.clip(mean_distance / max(1.0, 1.6 * threshold), 0.0, 1.0)
            ),
        )
        partitions.append(SceneInstancePartition(proposal, seed_index))
        rows.append(
            {
                "partition_index": len(partitions),
                "seed_index": seed_index,
                "area_fraction": _area(full_mask) / image_area,
                "mean_surface_distance": mean_distance,
                "boundary_alignment": boundary_alignment,
                "score": score,
            }
        )

    partitions.sort(
        key=lambda item: (
            _box(item.proposal.mask)[1],
            _box(item.proposal.mask)[0],
            -_area(item.proposal.mask),
        )
    )
    if len(partitions) > config.maximum_partitions:
        partitions = partitions[: config.maximum_partitions]
        retained_seed_indices = {
            item.seed_index for item in partitions if item.seed_index is not None
        }
        replaced_seed_indices &= {int(index) for index in retained_seed_indices}

    return SceneInstanceResult(
        tuple(partitions),
        frozenset(replaced_seed_indices),
        {
            "algorithm": "hpid-scene-instance-partition-v1",
            "status": "applied" if partitions else "no_accepted_partitions",
            "analysis_size": [width, height],
            "envelope_fraction": envelope_fraction,
            "input_seed_count": len(seed_proposals),
            "accepted_seed_count": len(seed_rows),
            "replaced_seed_count": len(replaced_seed_indices),
            "marker_count": marker_index,
            "suppressed_seed_marker_count": len(suppressed_seed_markers),
            "suppressed_seed_markers": suppressed_seed_markers,
            "partition_count": len(partitions),
            "appearance_foreground_fraction": _area(appearance_foreground)
            / analysis_area,
            "object_foreground_fraction": _area(object_mask) / analysis_area,
            "minimum_peak_distance": minimum_peak_distance,
            "surface_merge": surface_merge_diagnostics,
            "candidate_rows": rows,
            **surface_diagnostics,
            "ground_truth_used": False,
        },
    )
