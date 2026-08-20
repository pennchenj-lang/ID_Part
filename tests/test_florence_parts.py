from __future__ import annotations

import numpy as np

from hpid_split.florence_parts import (
    _polygon_groups,
    _rasterize_polygons,
    _split_instances,
)


def test_polygon_groups_accepts_florence_label_nesting() -> None:
    payload = {
        "polygons": [
            [
                [1, 1, 8, 1, 8, 8, 1, 8],
                [12, 2, 16, 2, 16, 6, 12, 6],
            ]
        ],
        "labels": [""],
    }

    groups = _polygon_groups(payload)

    assert len(groups) == 1
    assert len(groups[0]) == 2


def test_polygon_groups_accepts_a_single_flat_polygon() -> None:
    groups = _polygon_groups(
        {"polygons": [[1, 1, 8, 1, 8, 8, 1, 8]]}
    )

    assert groups == [[[1.0, 1.0, 8.0, 1.0, 8.0, 8.0, 1.0, 8.0]]]


def test_rasterization_clips_coordinates_to_image() -> None:
    mask = _rasterize_polygons(
        [[-5, -5, 6, -5, 6, 6, -5, 6]],
        (10, 10),
    )

    assert mask.dtype == bool
    assert mask[0, 0]
    assert mask[6, 6]
    assert not mask[9, 9]


def test_repeated_part_mask_splits_connected_instances() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:7, 2:7] = True
    mask[11:18, 11:18] = True

    components = _split_instances(
        mask,
        maximum_instances=4,
        minimum_area_px=8,
    )

    assert [int(np.count_nonzero(item)) for item in components] == [49, 25]


def test_single_part_keeps_disconnected_visible_fragments_together() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[2:7, 2:7] = True
    mask[11:18, 11:18] = True

    components = _split_instances(
        mask,
        maximum_instances=1,
        minimum_area_px=8,
    )

    assert len(components) == 1
    assert np.array_equal(components[0], mask)
