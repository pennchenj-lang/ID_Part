import json

import numpy as np
from PIL import Image

from hpid_split.fusion import MaskCandidate
from hpid_split.mask_refinement import (
    MaskRefinementConfig,
    _reconcile_axial_partitions,
    refine_candidate_masks,
)


def test_refinement_fills_small_hole_and_removes_isolated_noise() -> None:
    image = np.zeros((80, 80, 3), dtype=np.uint8)
    image[20:60, 20:60] = (220, 70, 40)
    mask = np.zeros((80, 80), dtype=bool)
    mask[19:61, 19:61] = True
    mask[35:37, 35:37] = False
    mask[5, 5] = True
    candidate = MaskCandidate(
        "asset",
        "asset",
        mask,
        0.8,
        "model/root",
        metadata={"root_index": 1, "sam_quality": 0.8},
    )

    result = refine_candidate_masks(Image.fromarray(image), [candidate])
    refined = result.candidates[0]

    assert refined.mask[35, 35]
    assert not refined.mask[5, 5]
    assert "mask_refinement" in refined.metadata
    assert result.diagnostics["ground_truth_used"] is False
    json.dumps(result.diagnostics)


def test_refinement_preserves_candidate_contract() -> None:
    image = Image.new("RGB", (32, 32), "white")
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 10:22] = True
    candidate = MaskCandidate(
        "asset_button",
        "asset",
        mask,
        0.61,
        "model/details",
        prompt="button",
        source_reliability=0.73,
        metadata={"hierarchy_depth": 3},
    )

    refined = refine_candidate_masks(image, [candidate]).candidates[0]

    assert refined.semantic_name == candidate.semantic_name
    assert refined.semantic_parent == candidate.semantic_parent
    assert refined.score == candidate.score
    assert refined.source == candidate.source
    assert refined.prompt == candidate.prompt
    assert refined.source_reliability == candidate.source_reliability
    assert refined.mask.shape == candidate.mask.shape


def test_refinement_expands_trimap_and_keeps_better_image_boundary() -> None:
    image = np.full((220, 220, 3), 20, dtype=np.uint8)
    image[60:160, 60:160] = (225, 90, 35)
    mask = np.zeros((220, 220), dtype=bool)
    mask[56:164, 56:164] = True
    truth = np.zeros_like(mask)
    truth[60:160, 60:160] = True
    candidate = MaskCandidate("asset", "asset", mask, 0.9, "model/root")

    result = refine_candidate_masks(Image.fromarray(image), [candidate])
    refined = result.candidates[0]
    row = refined.metadata["mask_refinement"]

    assert row["radius_px"] >= 2
    assert row["edge_guard_passed"] is True
    assert row["proposed_edge_score_ratio"] >= 0.90
    assert np.count_nonzero(refined.mask ^ truth) <= np.count_nonzero(mask ^ truth)


def test_visual_detail_region_uses_detail_preserving_cleanup() -> None:
    image = Image.new("RGB", (48, 48), "white")
    mask = np.zeros((48, 48), dtype=bool)
    mask[20:24, 8:40] = True
    candidate = MaskCandidate(
        "asset_visual_detail_01",
        "asset",
        mask,
        0.7,
        "visual/model",
        metadata={"visual_region_kind": "detail"},
    )

    refined = refine_candidate_masks(image, [candidate]).candidates[0]

    assert refined.metadata["mask_refinement"]["detail_mode"] is True


def test_fast_schedule_limits_grabcut_but_cleans_every_candidate() -> None:
    image = Image.new("RGB", (64, 64), "white")
    candidates = []
    for index in range(5):
        mask = np.zeros((64, 64), dtype=bool)
        mask[8 + index : 42 + index, 9:48] = True
        mask[2, 2 + index] = True
        candidates.append(
            MaskCandidate(
                f"asset_part_{index}",
                "asset",
                mask,
                0.9 - index * 0.05,
                "test/source",
            )
        )

    result = refine_candidate_masks(
        image,
        candidates,
        config=MaskRefinementConfig(
            grabcut_iterations=1,
            maximum_grabcut_candidates=2,
        ),
    )

    assert result.diagnostics["grabcut_scheduled_count"] == 2
    assert sum(
        candidate.metadata["mask_refinement"]["grabcut_scheduled"]
        for candidate in result.candidates
    ) == 2
    assert all(
        not candidate.mask[2, 2 + index]
        for index, candidate in enumerate(result.candidates)
    )


def test_axial_siblings_remove_unsupported_root_fragments_and_remain_disjoint() -> None:
    shape = (80, 140)
    root_mask = np.zeros(shape, dtype=bool)
    root_mask[30:50, 10:130] = True
    root_mask[5:10, 5:10] = True
    left = np.zeros(shape, dtype=bool)
    left[30:50, 10:72] = True
    left[5:10, 5:10] = True
    right = np.zeros(shape, dtype=bool)
    right[30:50, 68:130] = True
    shared = {
        "root_origin": "test",
        "root_index": 1,
        "structural_fusion_algorithm": (
            "profile-constrained-silhouette-axial-partition-v1"
        ),
    }
    candidates = [
        MaskCandidate(
            "tool",
            "tool",
            root_mask,
            0.9,
            "root",
            metadata={"root_origin": "test", "root_index": 1},
        ),
        MaskCandidate(
            "tool_head",
            "tool_body",
            left,
            0.85,
            "structural",
            metadata={**shared, "structural_peer_semantic": "tool_handle"},
        ),
        MaskCandidate(
            "tool_handle",
            "tool_body",
            right,
            0.85,
            "structural",
            metadata={**shared, "structural_peer_semantic": "tool_head"},
        ),
    ]

    reconciled, diagnostics = _reconcile_axial_partitions(candidates)
    clean_root = reconciled[0].mask
    clean_left = reconciled[1].mask
    clean_right = reconciled[2].mask

    assert not np.any(clean_root[5:10, 5:10])
    assert np.array_equal(clean_left | clean_right, clean_root)
    assert not np.any(clean_left & clean_right)
    assert diagnostics["reconciled_partition_count"] == 1
    assert diagnostics["removed_unsupported_root_pixels"] == 25
    assert diagnostics["ground_truth_used"] is False
