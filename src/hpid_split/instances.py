from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .taxonomy import Taxonomy


@dataclass(frozen=True)
class PartInstance:
    part_id: str
    semantic_name: str
    semantic_parent: str
    instance_index: int
    side: str
    bbox_xyxy: tuple[int, int, int, int]
    centroid_xy: tuple[float, float]
    area_px: int
    asset_id: str = "object_001"
    assembly_parent_id: str | None = None
    group_id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _side_name(centroid_x: float, reference_x: float, dead_zone: float) -> str:
    if centroid_x < reference_x - dead_zone:
        return "left"
    if centroid_x > reference_x + dead_zone:
        return "right"
    return "center"


def semantic_to_part_ids(
    labels: np.ndarray,
    taxonomy: Taxonomy,
    *,
    minimum_area: int = 12,
    asset_id: str = "object_001",
) -> tuple[np.ndarray, list[PartInstance]]:
    """Convert a semantic map into deterministic, source-coordinate Part IDs."""
    if labels.ndim != 2:
        raise ValueError("labels must be a two-dimensional array")
    _, width = labels.shape
    foreground_x = np.nonzero(labels > 0)[1]
    reference_x = float(np.median(foreground_x)) if len(foreground_x) else width / 2.0
    dead_zone = max(2.0, width * 0.025)
    instance_map = np.zeros(labels.shape, dtype=np.uint16)
    records: list[PartInstance] = []
    numeric_id = 1
    for class_id in range(1, taxonomy.num_fine_classes):
        count, components, stats, centroids = cv2.connectedComponentsWithStats(
            (labels == class_id).astype(np.uint8),
            connectivity=8,
        )
        candidates = []
        for component_id in range(1, count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < minimum_area:
                continue
            x = int(stats[component_id, cv2.CC_STAT_LEFT])
            y = int(stats[component_id, cv2.CC_STAT_TOP])
            w = int(stats[component_id, cv2.CC_STAT_WIDTH])
            h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            cx, cy = (float(value) for value in centroids[component_id])
            side = _side_name(cx, reference_x, dead_zone)
            candidates.append((side, cy, cx, component_id, area, (x, y, x + w, y + h)))
        side_order = {"left": 0, "center": 1, "right": 2}
        candidates.sort(key=lambda item: (side_order[item[0]], item[1], item[2]))
        counters = {"left": 0, "center": 0, "right": 0}
        for side, cy, cx, component_id, area, bbox in candidates:
            counters[side] += 1
            semantic_name = taxonomy.fine_names[class_id]
            parent_name = taxonomy.parent_names[taxonomy.fine_to_parent[class_id]]
            part_id = f"{parent_name}/{semantic_name}/{side}/{counters[side]:02d}"
            instance_map[components == component_id] = numeric_id
            records.append(
                PartInstance(
                    part_id=part_id,
                    semantic_name=semantic_name,
                    semantic_parent=parent_name,
                    instance_index=numeric_id,
                    side=side,
                    bbox_xyxy=bbox,
                    centroid_xy=(cx, cy),
                    area_px=area,
                    asset_id=asset_id,
                )
            )
            numeric_id += 1
    return instance_map, records
