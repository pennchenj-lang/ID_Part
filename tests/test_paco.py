import numpy as np

from hpid_split.paco import decode_coco_segmentation


def test_decode_uncompressed_coco_rle_uses_column_major_order() -> None:
    segmentation = {"size": [3, 4], "counts": [1, 2, 3, 2, 4]}

    mask = decode_coco_segmentation(segmentation, 3, 4)

    expected_flat = np.asarray(
        [0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=bool
    )
    assert np.array_equal(mask, expected_flat.reshape((3, 4), order="F"))


def test_decode_coco_polygon_fills_region() -> None:
    segmentation = [[1, 1, 4, 1, 4, 4, 1, 4]]

    mask = decode_coco_segmentation(segmentation, 6, 6)

    assert mask[2, 2]
    assert not mask[0, 0]
    assert int(np.count_nonzero(mask)) == 16


def test_decode_compressed_paco_rle_matches_known_area() -> None:
    segmentation = {
        "size": [3, 4],
        "counts": "12301",
    }

    mask = decode_coco_segmentation(segmentation, 3, 4)

    expected = decode_coco_segmentation(
        {"size": [3, 4], "counts": [1, 2, 3, 2, 4]}, 3, 4
    )
    assert np.array_equal(mask, expected)
