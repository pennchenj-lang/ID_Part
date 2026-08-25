import cv2
import numpy as np
from PIL import Image

from hpid_split.scene_instances import (
    SceneInstanceConfig,
    _merge_adjacent_object_surfaces,
    _merge_projected_face_fragments,
    partition_scene_instances,
)
from hpid_split.visual_regions import VisualMaskProposal


def _proposal(mask: np.ndarray, score: float = 0.92) -> VisualMaskProposal:
    return VisualMaskProposal(mask.astype(bool), score, boundary_alignment=0.88)


def test_scene_partition_separates_touching_repeated_objects() -> None:
    image = np.full((180, 240, 3), 248, dtype=np.uint8)
    envelope = np.zeros((180, 240), dtype=np.uint8)
    cv2.rectangle(envelope, (8, 30), (232, 174), 1, -1)
    image[envelope.astype(bool)] = (64, 166, 58)

    objects: list[np.ndarray] = []
    for center, color in zip(
        ((75, 104), (112, 100), (154, 107)),
        ((135, 130, 112), (155, 151, 132), (112, 111, 99)),
        strict=True,
    ):
        mask = np.zeros_like(envelope)
        cv2.ellipse(mask, center, (27, 23), 0, 0, 360, 1, -1)
        image[mask.astype(bool)] = color
        objects.append(mask)

    result = partition_scene_instances(
        Image.fromarray(image),
        _proposal(envelope),
        [_proposal(objects[0]), _proposal(objects[2])],
    )

    assert result.diagnostics["status"] == "applied"
    assert len(result.partitions) >= 3
    assert result.replaced_seed_indices == frozenset({0, 1})
    accumulated = np.zeros_like(envelope, dtype=bool)
    for partition in result.partitions:
        assert not np.any(accumulated & partition.proposal.mask)
        accumulated |= partition.proposal.mask
    assert result.diagnostics["ground_truth_used"] is False


def test_scene_partition_accepts_a_multitone_support_surface() -> None:
    image = np.full((180, 260, 3), 248, dtype=np.uint8)
    envelope = np.zeros((180, 260), dtype=np.uint8)
    cv2.rectangle(envelope, (8, 34), (252, 174), 1, -1)
    image[envelope.astype(bool)] = (58, 158, 54)
    image[34:175, 130:253] = (76, 178, 66)
    image[42:175:18, 8:253] = (69, 169, 61)

    objects: list[np.ndarray] = []
    for center, color in zip(
        ((72, 105), (128, 101), (192, 108)),
        ((170, 150, 126), (128, 119, 103), (198, 183, 151)),
        strict=True,
    ):
        mask = np.zeros_like(envelope)
        cv2.ellipse(mask, center, (24, 27), 0, 0, 360, 1, -1)
        image[mask.astype(bool)] = color
        objects.append(mask)

    result = partition_scene_instances(
        Image.fromarray(image),
        _proposal(envelope),
        [_proposal(objects[0]), _proposal(objects[2])],
    )

    assert result.diagnostics["status"] == "applied"
    assert len(result.diagnostics["surface_cluster_indices"]) >= 2
    assert len(result.partitions) >= 3


def test_scene_partition_keeps_peaks_in_a_large_compound_cluster() -> None:
    image = np.full((180, 280, 3), 248, dtype=np.uint8)
    envelope = np.zeros((180, 280), dtype=np.uint8)
    cv2.rectangle(envelope, (8, 34), (272, 174), 1, -1)
    image[envelope.astype(bool)] = (62, 166, 58)
    for center, color in zip(
        ((65, 108), (115, 108), (165, 108), (215, 108)),
        ((178, 126, 112), (112, 126, 176), (190, 170, 112), (105, 108, 102)),
        strict=True,
    ):
        mask = np.zeros_like(envelope)
        cv2.circle(mask, center, 34, 1, -1)
        image[mask.astype(bool)] = color

    result = partition_scene_instances(
        Image.fromarray(image),
        _proposal(envelope),
        [],
    )

    assert result.diagnostics["status"] == "applied"
    assert result.diagnostics["marker_count"] >= 4
    assert result.diagnostics["color_component_marker_count"] >= 4
    assert len(result.partitions) >= 4


def test_scene_partition_consolidates_multiple_face_seeds_of_one_object() -> None:
    image = np.full((180, 260, 3), 246, dtype=np.uint8)
    envelope = np.zeros((180, 260), dtype=np.uint8)
    cv2.rectangle(envelope, (8, 34), (252, 174), 1, -1)
    image[envelope.astype(bool)] = (62, 166, 58)
    left_object = np.zeros_like(envelope)
    cv2.rectangle(left_object, (36, 72), (112, 154), 1, -1)
    left_top = np.zeros_like(envelope)
    cv2.rectangle(left_top, (42, 64), (106, 96), 1, -1)
    left_side = np.zeros_like(envelope)
    cv2.rectangle(left_side, (42, 92), (106, 150), 1, -1)
    right_object = np.zeros_like(envelope)
    cv2.rectangle(right_object, (158, 76), (226, 152), 1, -1)
    image[left_object.astype(bool)] = (146, 132, 112)
    image[left_top.astype(bool)] = (184, 160, 132)
    image[right_object.astype(bool)] = (110, 112, 102)

    result = partition_scene_instances(
        Image.fromarray(image),
        _proposal(envelope),
        [
            _proposal(left_top, 0.96),
            _proposal(left_side, 0.94),
            _proposal(right_object, 0.95),
        ],
    )

    assert result.diagnostics["status"] == "applied"
    assert result.diagnostics["suppressed_seed_marker_count"] >= 1
    assert len(result.partitions) == 2


def test_scene_partition_skips_texture_without_dominant_surface() -> None:
    generator = np.random.default_rng(117)
    image = generator.integers(0, 255, size=(120, 160, 3), dtype=np.uint8)
    envelope = np.ones((120, 160), dtype=bool)

    result = partition_scene_instances(
        Image.fromarray(image), _proposal(envelope), []
    )

    assert not result.partitions
    assert result.diagnostics["status"].startswith("skipped_")
    assert result.diagnostics["ground_truth_used"] is False


def test_scene_surface_merge_uses_shared_object_envelope() -> None:
    labels = np.zeros((80, 100), dtype=np.int32)
    labels[12:30, 26:68] = 1
    labels[30:70, 22:74] = 2
    object_mask = labels > 0
    lab = np.zeros((80, 100, 3), dtype=np.float32)
    lab[labels == 1] = (190, 140, 135)
    lab[labels == 2] = (105, 125, 120)
    shared_envelope = object_mask.copy()

    merged, diagnostics = _merge_adjacent_object_surfaces(
        labels,
        lab,
        object_mask,
        SceneInstanceConfig(maximum_structural_merged_fraction=1.0),
        seed_masks=(shared_envelope,),
    )

    assert set(np.unique(merged)) == {0, 1}
    row = diagnostics["merge_rows"][0]
    assert row["accepted"] is True
    assert row["merge_reason"] == "shared_envelope_structural_face"


def test_scene_surface_merge_keeps_adjacent_objects_without_shared_envelope() -> None:
    labels = np.zeros((80, 100), dtype=np.int32)
    labels[12:30, 26:68] = 1
    labels[30:70, 22:74] = 2
    object_mask = labels > 0
    lab = np.zeros((80, 100, 3), dtype=np.float32)
    lab[labels == 1] = (190, 140, 135)
    lab[labels == 2] = (105, 125, 120)

    merged, diagnostics = _merge_adjacent_object_surfaces(
        labels,
        lab,
        object_mask,
        SceneInstanceConfig(maximum_structural_merged_fraction=1.0),
        seed_masks=(labels == 1, labels == 2),
    )

    assert set(np.unique(merged)) == {0, 1, 2}
    assert diagnostics["merge_rows"][0]["accepted"] is False


def test_scene_projected_face_merge_joins_wide_top_to_body() -> None:
    labels = np.zeros((100, 120), dtype=np.int32)
    labels[20:88, 26:96] = 2
    labels[12:36, 30:82] = 1
    object_mask = labels > 0
    lab = np.zeros((100, 120, 3), dtype=np.float32)
    lab[labels == 1] = (185, 140, 135)
    lab[labels == 2] = (110, 125, 120)

    merged, diagnostics = _merge_projected_face_fragments(
        labels,
        lab,
        object_mask,
        SceneInstanceConfig(maximum_structural_merged_fraction=1.0),
        seed_masks=(object_mask,),
    )

    assert set(np.unique(merged)) == {0, 1}
    assert diagnostics["merge_rows"][0]["accepted"] is True


def test_scene_projected_face_merge_keeps_separately_seeded_objects() -> None:
    labels = np.zeros((100, 120), dtype=np.int32)
    labels[20:88, 26:96] = 2
    labels[12:36, 30:82] = 1
    object_mask = labels > 0
    lab = np.zeros((100, 120, 3), dtype=np.float32)
    lab[labels == 1] = (185, 140, 135)
    lab[labels == 2] = (110, 125, 120)

    merged, diagnostics = _merge_projected_face_fragments(
        labels,
        lab,
        object_mask,
        SceneInstanceConfig(maximum_structural_merged_fraction=1.0),
        seed_masks=(labels == 1, labels == 2),
    )

    assert set(np.unique(merged)) == {0, 1, 2}
    assert diagnostics["merge_rows"][0]["projected_face_evidence"] is False
    assert diagnostics["merge_rows"][0]["accepted"] is False


def test_scene_projected_face_merge_joins_a_compact_offset_top() -> None:
    labels = np.zeros((120, 150), dtype=np.int32)
    labels[42:105, 38:118] = 2
    labels[24:64, 48:102] = 1
    object_mask = labels > 0
    lab = np.zeros((120, 150, 3), dtype=np.float32)
    lab[labels == 1] = (200, 145, 136)
    lab[labels == 2] = (86, 124, 118)
    unrelated = np.zeros_like(labels, dtype=bool)
    unrelated[8:24, 8:24] = True
    object_mask |= unrelated

    merged, diagnostics = _merge_projected_face_fragments(
        labels,
        lab,
        object_mask,
        SceneInstanceConfig(
            compact_projected_maximum_merged_fraction=1.0,
            maximum_unseeded_projected_face_fraction=1.0,
        ),
    )

    assert set(np.unique(merged)) == {0, 1}
    assert diagnostics["merge_rows"][0]["merge_reason"] in {
        "compact_projected_face",
        "strong_projected_face",
    }
    assert diagnostics["merge_rows"][0]["accepted"] is True


def test_scene_projected_face_merge_keeps_side_by_side_instances() -> None:
    labels = np.zeros((100, 140), dtype=np.int32)
    labels[25:82, 15:65] = 1
    labels[25:82, 65:115] = 2
    object_mask = labels > 0
    lab = np.zeros((100, 140, 3), dtype=np.float32)
    lab[labels == 1] = (125, 130, 130)
    lab[labels == 2] = (120, 132, 128)

    merged, diagnostics = _merge_projected_face_fragments(
        labels,
        lab,
        object_mask,
        SceneInstanceConfig(maximum_structural_merged_fraction=1.0),
    )

    assert set(np.unique(merged)) == {0, 1, 2}
    assert diagnostics["merge_rows"][0]["accepted"] is False
