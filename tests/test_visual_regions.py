import json

import numpy as np
from PIL import Image

from hpid_split.fusion import MaskCandidate
from hpid_split.visual_regions import (
    VisualMaskProposal,
    VisualRegionConfig,
    _consolidate_multiview_proposals,
    _region_kind,
    visual_region_candidates_from_masks,
)


def _mask(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((100, 100), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _candidate(name: str, parent: str, mask: np.ndarray) -> MaskCandidate:
    return MaskCandidate(
        semantic_name=name,
        semantic_parent=parent,
        mask=mask,
        score=0.9,
        source="test/source",
        metadata={
            "root_origin": "test-origin",
            "root_index": 1,
            "candidate_key": "root:1" if name == parent else f"root:1/{name}:01",
            "parent_candidate_key": None if name == parent else "root:1",
            "assembly_parent_semantic": parent,
        },
    )


def test_diagonal_thin_region_is_classified_by_occupied_shape() -> None:
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv_points = np.asarray([[10, 72], [75, 12]], dtype=np.int32)
    import cv2

    cv2.polylines(mask, [cv_points], False, 1, thickness=4)

    assert _region_kind(mask.astype(bool), root_area=5_000) == "strip"


def test_visual_regions_create_nested_generic_parts_and_semantic_support() -> None:
    root = _candidate("tool_prop", "tool_prop", _mask(5, 95, 5, 95))
    semantic = _candidate("tool_prop_handle", "tool_prop", _mask(60, 88, 65, 90))
    large = _mask(20, 70, 20, 70)
    small = _mask(30, 35, 30, 35)
    outside = _mask(0, 4, 0, 4)
    result = visual_region_candidates_from_masks(
        [
            VisualMaskProposal(large, 0.95),
            VisualMaskProposal(small, 0.93),
            VisualMaskProposal(semantic.mask, 0.92),
            VisualMaskProposal(outside, 0.99),
        ],
        [root],
        [root, semantic],
        config=VisualRegionConfig(minimum_root_area_fraction=0.0002),
    )

    generic = [
        candidate
        for candidate in result.candidates
        if candidate.metadata["generic_visual_region"]
    ]
    supported = [
        candidate
        for candidate in result.candidates
        if not candidate.metadata["generic_visual_region"]
    ]
    assert len(generic) == 2
    assert len(supported) == 1
    assert supported[0].semantic_name == "tool_prop_handle"
    assert (
        supported[0].metadata["semantic_support_candidate_key"]
        == semantic.metadata["candidate_key"]
    )
    small_candidate = min(generic, key=lambda candidate: np.count_nonzero(candidate.mask))
    large_candidate = max(generic, key=lambda candidate: np.count_nonzero(candidate.mask))
    assert small_candidate.semantic_parent == large_candidate.semantic_name
    assert small_candidate.metadata["hierarchy_depth"] == 2
    assert result.diagnostics["rejected_outside_root_count"] == 1
    assert result.diagnostics["ground_truth_used"] is False
    json.dumps(result.diagnostics)


def test_visual_regions_namespace_repeated_same_domain_roots() -> None:
    left_root = _candidate("structure", "structure", _mask(5, 45, 5, 45))
    right_root = MaskCandidate(
        semantic_name="structure",
        semantic_parent="structure",
        mask=_mask(55, 95, 55, 95),
        score=0.88,
        source="test/source",
        metadata={
            "root_origin": "test-origin",
            "root_index": 2,
            "candidate_key": "root:2",
            "parent_candidate_key": None,
            "assembly_parent_semantic": "structure",
        },
    )
    result = visual_region_candidates_from_masks(
        [
            VisualMaskProposal(_mask(10, 30, 10, 30), 0.95),
            VisualMaskProposal(_mask(60, 80, 60, 80), 0.94),
        ],
        [left_root, right_root],
        [left_root, right_root],
        config=VisualRegionConfig(minimum_root_area_fraction=0.0002),
    )

    names = {candidate.semantic_name for candidate in result.candidates}
    assert names == {
        "structure_asset_01_visual_panel_01",
        "structure_asset_02_visual_panel_01",
    }


def test_targeted_visual_proposal_cannot_be_claimed_by_scene_layer() -> None:
    terrain = MaskCandidate(
        semantic_name="terrain",
        semantic_parent="terrain",
        mask=_mask(0, 100, 0, 100),
        score=0.99,
        source="test/source",
        metadata={
            "root_origin": "test-origin",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
            "scene_role": "scene_layer",
        },
    )
    prop = MaskCandidate(
        semantic_name="natural_object",
        semantic_parent="natural_object",
        mask=_mask(20, 80, 20, 80),
        score=0.80,
        source="test/source",
        metadata={
            "root_origin": "test-origin",
            "root_index": 2,
            "candidate_key": "root:2",
            "parent_candidate_key": None,
            "scene_role": "object",
        },
    )
    proposal = VisualMaskProposal(
        _mask(30, 45, 30, 45),
        0.94,
        target_root_key="test-origin::2",
    )

    result = visual_region_candidates_from_masks(
        [proposal],
        [terrain, prop],
        [terrain, prop],
        config=VisualRegionConfig(minimum_root_area_fraction=0.0002),
    )

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.metadata["root_index"] == 2
    assert candidate.metadata["proposal_target_root_key"] == "test-origin::2"


def test_selected_scene_root_is_not_reintroduced_as_layer_child() -> None:
    terrain = MaskCandidate(
        "terrain",
        "terrain",
        _mask(0, 100, 0, 100),
        0.99,
        "test/source",
        metadata={
            "root_origin": "test-origin",
            "root_index": 1,
            "candidate_key": "root:1",
            "scene_role": "scene_layer",
        },
    )
    prop = MaskCandidate(
        "natural_object",
        "natural_object",
        _mask(20, 80, 20, 80),
        0.90,
        "test/source",
        metadata={
            "root_origin": "test-origin",
            "root_index": 2,
            "candidate_key": "root:2",
            "scene_role": "object",
        },
    )

    result = visual_region_candidates_from_masks(
        [VisualMaskProposal(prop.mask, 0.96)],
        [terrain, prop],
        [terrain, prop],
        config=VisualRegionConfig(minimum_root_area_fraction=0.0002),
    )

    assert result.candidates == ()
    assert result.diagnostics["rejected_selected_root_count"] == 1


def test_scene_part_is_routed_to_smallest_compatible_object_root() -> None:
    terrain = MaskCandidate(
        "terrain",
        "terrain",
        _mask(0, 100, 0, 100),
        0.99,
        "test/source",
        metadata={
            "root_origin": "test-origin",
            "root_index": 1,
            "candidate_key": "root:1",
            "scene_role": "scene_layer",
        },
    )
    prop = MaskCandidate(
        "natural_object",
        "natural_object",
        _mask(20, 80, 20, 80),
        0.88,
        "test/source",
        metadata={
            "root_origin": "test-origin",
            "root_index": 2,
            "candidate_key": "root:2",
            "scene_role": "object",
        },
    )

    result = visual_region_candidates_from_masks(
        [VisualMaskProposal(_mask(35, 45, 35, 45), 0.94)],
        [terrain, prop],
        [terrain, prop],
        config=VisualRegionConfig(minimum_root_area_fraction=0.0002),
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].metadata["root_index"] == 2


def test_multiview_consensus_keeps_identical_masks_for_different_roots_separate() -> None:
    image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    mask = _mask(20, 40, 20, 40)

    consolidated, diagnostics = _consolidate_multiview_proposals(
        image,
        [
            VisualMaskProposal(
                mask,
                0.90,
                view_id="root-a",
                target_root_key="origin::1",
            ),
            VisualMaskProposal(
                mask,
                0.91,
                view_id="root-b",
                target_root_key="origin::2",
            ),
        ],
        VisualRegionConfig(),
    )

    assert len(consolidated) == 2
    assert diagnostics["duplicate_view_proposals_removed"] == 0


def test_untrusted_dense_semantic_does_not_infect_visual_region() -> None:
    root = _candidate("character", "character", _mask(5, 95, 5, 95))
    false_eyewear = MaskCandidate(
        semantic_name="character_eyewear",
        semantic_parent="character_head",
        mask=_mask(35, 55, 20, 40),
        score=0.95,
        source="clipseg/dense",
        metadata={
            "root_origin": "test-origin",
            "root_index": 1,
            "candidate_key": "root:1/character_eyewear:01",
            "parent_candidate_key": "root:1",
            "dense_semantic_fallback": True,
        },
    )

    result = visual_region_candidates_from_masks(
        [VisualMaskProposal(false_eyewear.mask, 0.96)],
        [root],
        [root, false_eyewear],
        config=VisualRegionConfig(minimum_root_area_fraction=0.0002),
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].metadata["generic_visual_region"] is True
    assert result.candidates[0].semantic_name.startswith("character_visual_")


def test_user_guided_semantic_can_relabel_visual_region_below_auto_threshold() -> None:
    root = _candidate("asset", "asset", _mask(5, 95, 5, 95))
    guided = MaskCandidate(
        semantic_name="requested_switch",
        semantic_parent="asset",
        mask=_mask(35, 55, 20, 40),
        score=0.20,
        source="sam3/guided",
        metadata={
            "root_origin": "test-origin",
            "root_index": 1,
            "candidate_key": "root:1/requested_switch:01",
            "parent_candidate_key": "root:1",
            "guided_prompt": True,
        },
    )

    result = visual_region_candidates_from_masks(
        [VisualMaskProposal(guided.mask, 0.96)],
        [root],
        [root, guided],
        config=VisualRegionConfig(minimum_root_area_fraction=0.0002),
    )

    assert result.candidates[0].metadata["generic_visual_region"] is False
    assert result.candidates[0].semantic_name == "requested_switch"


def test_repeated_named_parts_relabel_their_shared_visual_region() -> None:
    root = _candidate("character", "character", _mask(5, 95, 5, 95))
    left = _candidate("character_hair", "character", _mask(20, 40, 10, 30))
    right = MaskCandidate(
        semantic_name="character_hair",
        semantic_parent="character",
        mask=_mask(20, 40, 70, 90),
        score=0.88,
        source="test/source",
        metadata={
            "root_origin": "test-origin",
            "root_index": 1,
            "candidate_key": "root:1/character_hair:02",
            "parent_candidate_key": "root:1",
        },
    )
    shared = _mask(15, 45, 8, 92)

    result = visual_region_candidates_from_masks(
        [VisualMaskProposal(shared, 0.94)],
        [root],
        [root, left, right],
        config=VisualRegionConfig(minimum_root_area_fraction=0.0002),
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].semantic_name == "character_hair"
    assert result.candidates[0].metadata["generic_visual_region"] is False


def test_visual_candidate_namespace_prevents_cross_backend_key_collisions() -> None:
    root = _candidate("character", "character", _mask(5, 95, 5, 95))
    proposal = VisualMaskProposal(_mask(25, 50, 25, 55), 0.92)

    result = visual_region_candidates_from_masks(
        [proposal],
        [root],
        [root],
        config=VisualRegionConfig(minimum_root_area_fraction=0.0002),
        candidate_namespace="contour",
    )

    candidate = result.candidates[0]
    assert candidate.metadata["candidate_key"] == "root:1/contour-visual-region:01"
    assert candidate.semantic_name == "character_visual_contour_panel_01"


def test_multiview_consensus_merges_duplicate_crop_masks() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[20:70, 25:75] = 230
    global_mask = _mask(20, 70, 25, 75)
    crop_mask = _mask(21, 70, 25, 74)

    consolidated, diagnostics = _consolidate_multiview_proposals(
        Image.fromarray(image),
        [
            VisualMaskProposal(
                global_mask,
                0.90,
                scale_level=0,
                view_id="global",
            ),
            VisualMaskProposal(
                crop_mask,
                0.94,
                scale_level=1,
                view_id="layer-1-crop-0",
            ),
        ],
        VisualRegionConfig(),
    )

    assert len(consolidated) == 1
    assert consolidated[0].support_views == ("global", "layer-1-crop-0")
    assert consolidated[0].best_view_iou > 0.90
    assert diagnostics["duplicate_view_proposals_removed"] == 1
    assert diagnostics["multi_view_cluster_count"] == 1


def test_isolated_crop_proposal_is_retained_but_downweighted() -> None:
    image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    proposal = VisualMaskProposal(
        _mask(30, 40, 30, 40),
        0.90,
        scale_level=1,
        view_id="layer-1-crop-1",
    )

    consolidated, diagnostics = _consolidate_multiview_proposals(
        image,
        [proposal],
        VisualRegionConfig(isolated_crop_score_scale=0.90),
    )

    assert len(consolidated) == 1
    assert consolidated[0].score == 0.81
    assert diagnostics["consolidated_proposal_count"] == 1
