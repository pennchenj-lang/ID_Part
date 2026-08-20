from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .fusion import MaskCandidate, mask_iou
from .visual_regions import VisualMaskProposal


@dataclass(frozen=True)
class ShapeProposalConfig:
    """Category-independent silhouette decomposition for attached structures."""

    minimum_area_fraction: float = 0.008
    maximum_area_fraction: float = 0.62
    minimum_outer_boundary_contact: float = 0.48
    minimum_peak_distance_fraction: float = 0.040
    maximum_markers: int = 14
    maximum_regions_per_root: int = 12
    duplicate_iou: float = 0.82
    merge_shared_boundary_ratio: float = 0.26


@dataclass(frozen=True)
class ShapeProposalResult:
    proposals: tuple[VisualMaskProposal, ...]
    diagnostics: dict[str, object]


def _area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def _box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _root_key(root: MaskCandidate) -> str:
    return (
        f"{root.metadata.get('root_origin', 'legacy')}::"
        f"{root.metadata.get('root_index', 'unknown')}"
    )


def _outer_boundary_contact(mask: np.ndarray, root: np.ndarray) -> float:
    kernel = np.ones((3, 3), dtype=np.uint8)
    boundary = mask & ~cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
    root_boundary = root & ~cv2.erode(root.astype(np.uint8), kernel).astype(bool)
    if not np.any(boundary):
        return 0.0
    near_outer = cv2.dilate(root_boundary.astype(np.uint8), kernel).astype(bool)
    return float(np.count_nonzero(boundary & near_outer) / np.count_nonzero(boundary))


def _merge_watershed_regions(
    labels: np.ndarray,
    *,
    minimum_shared_boundary_ratio: float,
) -> list[np.ndarray]:
    maximum = int(labels.max())
    if maximum < 1:
        return []
    perimeters = np.zeros(maximum + 1, dtype=np.int64)
    kernel = np.ones((3, 3), dtype=np.uint8)
    for label in range(1, maximum + 1):
        mask = labels == label
        boundary = mask & ~cv2.erode(mask.astype(np.uint8), kernel).astype(bool)
        perimeters[label] = _area(boundary)
    shared: dict[tuple[int, int], int] = {}
    for first, second in (
        (labels[:, :-1], labels[:, 1:]),
        (labels[:-1, :], labels[1:, :]),
    ):
        active = (first > 0) & (second > 0) & (first != second)
        for left, right in zip(first[active], second[active], strict=True):
            key = tuple(sorted((int(left), int(right))))
            shared[key] = shared.get(key, 0) + 1
    groups: list[set[int]] = [{label} for label in range(1, maximum + 1)]

    def group_perimeter(group: set[int]) -> int:
        internal = sum(
            length
            for (first, second), length in shared.items()
            if first in group and second in group
        )
        return max(1, int(sum(perimeters[label] for label in group) - 2 * internal))

    while len(groups) > 1:
        best: tuple[float, int, int] | None = None
        group_perimeters = [group_perimeter(group) for group in groups]
        for first_index, first_group in enumerate(groups):
            for second_index in range(first_index + 1, len(groups)):
                second_group = groups[second_index]
                length = sum(
                    value
                    for (first, second), value in shared.items()
                    if (first in first_group and second in second_group)
                    or (second in first_group and first in second_group)
                )
                ratio = length / max(
                    1,
                    min(
                        group_perimeters[first_index],
                        group_perimeters[second_index],
                    ),
                )
                if best is None or ratio > best[0]:
                    best = (ratio, first_index, second_index)
        if best is None or best[0] < minimum_shared_boundary_ratio:
            break
        _, first_index, second_index = best
        groups[first_index] |= groups[second_index]
        del groups[second_index]
    return [np.isin(labels, sorted(group)) for group in groups]


def propose_shape_regions(
    roots: list[MaskCandidate],
    *,
    existing_visual_proposals: list[VisualMaskProposal] | None = None,
    config: ShapeProposalConfig | None = None,
) -> ShapeProposalResult:
    """Propose appendages and lobes from silhouette bottlenecks."""

    config = config or ShapeProposalConfig()
    try:
        from scipy import ndimage as ndi
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed
    except ImportError as error:
        raise RuntimeError(
            "shape-bottleneck proposals require scipy and scikit-image"
        ) from error

    proposals: list[VisualMaskProposal] = []
    rows: list[dict[str, object]] = []
    suppressed_by_visual_count = 0
    for root in roots:
        root_mask = root.mask.astype(bool)
        root_area = _area(root_mask)
        if root_area < 64:
            continue
        distance = ndi.distance_transform_edt(root_mask)
        minimum_distance = max(
            4,
            round(np.sqrt(root_area) * config.minimum_peak_distance_fraction),
        )
        coordinates = peak_local_max(
            distance,
            min_distance=minimum_distance,
            threshold_abs=max(2.0, 0.035 * float(distance.max())),
            labels=root_mask,
            num_peaks=config.maximum_markers,
            exclude_border=False,
        )
        if len(coordinates) < 2:
            continue
        markers = np.zeros(root_mask.shape, dtype=np.int32)
        for index, (y, x) in enumerate(coordinates, start=1):
            markers[int(y), int(x)] = index
        labels = watershed(-distance, markers, mask=root_mask, watershed_line=False)
        regions = _merge_watershed_regions(
            labels,
            minimum_shared_boundary_ratio=config.merge_shared_boundary_ratio,
        )
        root_proposals: list[VisualMaskProposal] = []
        for mask in regions:
            area = _area(mask)
            fraction = area / root_area
            if not config.minimum_area_fraction <= fraction <= config.maximum_area_fraction:
                continue
            outer_contact = _outer_boundary_contact(mask, root_mask)
            if outer_contact < config.minimum_outer_boundary_contact:
                continue
            score = float(
                np.clip(
                    0.48
                    + 0.38 * outer_contact
                    + 0.08 * min(1.0, fraction / 0.12),
                    0.0,
                    0.94,
                )
            )
            proposal = VisualMaskProposal(
                mask=mask,
                score=score,
                bbox_xyxy=_box(mask),
                scale_level=0,
                view_id="shape-bottleneck",
                support_views=("shape-bottleneck",),
                support_levels=(0,),
                best_view_iou=0.0,
                boundary_alignment=outer_contact,
                target_root_key=_root_key(root),
                source="hpid-shape-bottleneck/watershed",
                geometric_support=outer_contact,
            )
            already_covered = False
            for visual in existing_visual_proposals or ():
                visual_mask = visual.mask.astype(bool)
                visual_area = _area(visual_mask)
                if visual_area < 16:
                    continue
                root_containment = _area(visual_mask & root_mask) / visual_area
                visual_fraction = visual_area / root_area
                if root_containment < 0.80 or not 0.002 <= visual_fraction <= 0.55:
                    continue
                intersection = _area(mask & visual_mask)
                if (
                    intersection / max(1, area) >= 0.08
                    and intersection / visual_area >= 0.75
                ):
                    already_covered = True
                    break
            if already_covered:
                suppressed_by_visual_count += 1
                continue
            if any(
                mask_iou(proposal.mask, incumbent.mask) >= config.duplicate_iou
                for incumbent in root_proposals
            ):
                continue
            root_proposals.append(proposal)
            rows.append(
                {
                    "root_key": _root_key(root),
                    "area_fraction": fraction,
                    "outer_boundary_contact": outer_contact,
                    "score": score,
                }
            )
        root_proposals.sort(
            key=lambda proposal: (proposal.score, _area(proposal.mask)), reverse=True
        )
        proposals.extend(root_proposals[: config.maximum_regions_per_root])

    return ShapeProposalResult(
        tuple(proposals),
        {
            "algorithm": "hpid-shape-bottleneck-watershed-v1",
            "root_count": len(roots),
            "proposal_count": len(proposals),
            "suppressed_by_visual_count": suppressed_by_visual_count,
            "merge_shared_boundary_ratio": config.merge_shared_boundary_ratio,
            "candidate_rows": rows,
            "ground_truth_used": False,
        },
    )
