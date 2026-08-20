from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import linear_sum_assignment

from .instances import PartInstance


def preserve_part_ids(
    current_map: np.ndarray,
    current_records: list[PartInstance],
    previous_map: np.ndarray,
    previous_records: list[PartInstance],
    *,
    minimum_match_score: float = 0.20,
) -> list[PartInstance]:
    """Reuse previous IDs using semantic-aware mask and centroid matching."""
    if current_map.shape != previous_map.shape:
        raise ValueError(
            "current and previous instance maps must use the same source coordinates"
        )
    output = list(current_records)
    names = sorted({record.semantic_name for record in current_records})
    used_ids = {record.part_id for record in previous_records}
    for semantic_name in names:
        current_indices = [
            index
            for index, record in enumerate(current_records)
            if record.semantic_name == semantic_name
        ]
        previous_group = [
            record
            for record in previous_records
            if record.semantic_name == semantic_name
        ]
        if not current_indices or not previous_group:
            continue
        score = np.zeros((len(current_indices), len(previous_group)), dtype=np.float32)
        diagonal = float(np.hypot(*current_map.shape))
        for row, current_index in enumerate(current_indices):
            current = current_records[current_index]
            current_mask = current_map == current.instance_index
            for column, previous in enumerate(previous_group):
                previous_mask = previous_map == previous.instance_index
                union = np.count_nonzero(current_mask | previous_mask)
                iou = (
                    np.count_nonzero(current_mask & previous_mask) / union
                    if union
                    else 0.0
                )
                distance = np.hypot(
                    current.centroid_xy[0] - previous.centroid_xy[0],
                    current.centroid_xy[1] - previous.centroid_xy[1],
                )
                location = np.exp(-4.0 * distance / max(1.0, diagonal))
                score[row, column] = 0.80 * iou + 0.20 * location
        rows, columns = linear_sum_assignment(1.0 - score)
        for row, column in zip(rows, columns):
            if score[row, column] < minimum_match_score:
                continue
            current_index = current_indices[int(row)]
            output[current_index] = replace(
                output[current_index],
                part_id=previous_group[int(column)].part_id,
            )

    assigned = {record.part_id for record in output if record.part_id in used_ids}
    counts: dict[str, int] = {}
    for index, record in enumerate(output):
        if record.part_id in assigned:
            continue
        base = "/".join(record.part_id.split("/")[:-1])
        counts[base] = counts.get(base, 0) + 1
        candidate = record.part_id
        while candidate in assigned:
            counts[base] += 1
            candidate = f"{base}/{counts[base]:02d}"
        output[index] = replace(record, part_id=candidate)
        assigned.add(candidate)
    return output
