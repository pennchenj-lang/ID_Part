import numpy as np

from hpid_split.instances import PartInstance
from hpid_split.occlusion import (
    rank_occluder_hypotheses,
    structural_non_occluder_indices,
    validate_amodal_proposal,
)


def test_only_adjacent_ids_become_occluder_hypotheses() -> None:
    instance_map = np.zeros((48, 64), dtype=np.uint16)
    instance_map[14:34, 12:28] = 1
    instance_map[14:34, 28:42] = 2
    instance_map[4:10, 50:58] = 3
    visible = instance_map == 1
    hypotheses = rank_occluder_hypotheses(visible, instance_map, 1)
    assert [item.instance_index for item in hypotheses] == [2]
    assert np.all(hypotheses[0].search_mask <= (instance_map == 2))


def test_model_supported_connected_continuation_is_accepted() -> None:
    visible = np.zeros((40, 56), dtype=bool)
    visible[12:28, 10:26] = True
    search = np.zeros_like(visible)
    search[12:28, 26:38] = True
    proposed = visible.copy()
    proposed[14:26, 26:34] = True
    evidence = validate_amodal_proposal(visible, proposed, search, 0.82)
    assert evidence.accepted
    assert evidence.added_area_px == 12 * 8
    assert np.all(evidence.full_mask[visible])


def test_proposal_leaking_outside_tested_occluder_is_rejected() -> None:
    visible = np.zeros((40, 56), dtype=bool)
    visible[12:28, 10:26] = True
    search = np.zeros_like(visible)
    search[12:28, 26:32] = True
    proposed = visible.copy()
    proposed[10:30, 26:50] = True
    evidence = validate_amodal_proposal(visible, proposed, search, 0.90)
    assert not evidence.accepted
    assert evidence.reason == "proposal_leaks_outside_occluder_search"
    assert evidence.added_area_px == 0


def test_no_hidden_continuation_does_not_expand_visible_mask() -> None:
    visible = np.zeros((32, 40), dtype=bool)
    visible[9:23, 8:20] = True
    search = np.zeros_like(visible)
    search[9:23, 20:30] = True
    evidence = validate_amodal_proposal(visible, visible.copy(), search, 0.95)
    assert not evidence.accepted
    assert evidence.reason == "no_supported_hidden_continuation"
    assert np.array_equal(evidence.full_mask, visible)


def test_semantic_ancestors_and_assembly_parent_are_not_occluders() -> None:
    root = PartInstance(
        "character/character/center/01",
        "character",
        "character",
        1,
        "center",
        (0, 0, 30, 30),
        (15.0, 15.0),
        500,
    )
    head = PartInstance(
        "character/character_head/center/01",
        "character_head",
        "character",
        2,
        "center",
        (5, 5, 25, 25),
        (15.0, 15.0),
        250,
        assembly_parent_id=root.part_id,
    )
    eye = PartInstance(
        "character_head/character_eye/left/01",
        "character_eye",
        "character_head",
        3,
        "left",
        (9, 10, 14, 15),
        (11.5, 12.5),
        20,
        assembly_parent_id=head.part_id,
    )
    excluded = structural_non_occluder_indices(eye, (root, head, eye))
    assert excluded == frozenset({1, 2, 3})


def test_low_combined_evidence_is_rejected_even_when_shape_checks_pass() -> None:
    visible = np.zeros((40, 56), dtype=bool)
    visible[12:28, 10:26] = True
    search = np.zeros_like(visible)
    search[12:28, 26:38] = True
    proposed = visible.copy()
    proposed[14:26, 26:34] = True
    evidence = validate_amodal_proposal(visible, proposed, search, 0.40)
    assert not evidence.accepted
    assert evidence.reason == "weak_combined_evidence"


def test_sideways_spread_is_not_accepted_as_hidden_continuation() -> None:
    visible = np.zeros((64, 96), dtype=bool)
    visible[28:48, 28:68] = True
    search = np.zeros_like(visible)
    search[20:28, 0:96] = True
    proposed = visible | search
    evidence = validate_amodal_proposal(visible, proposed, search, 0.95)
    assert not evidence.accepted
    assert evidence.reason in {
        "implausible_lateral_spread",
        "implausible_orthogonal_span",
    }


def test_directional_continuation_passes_orthogonal_geometry_gate() -> None:
    visible = np.zeros((64, 96), dtype=bool)
    visible[28:48, 28:68] = True
    search = np.zeros_like(visible)
    search[18:28, 25:71] = True
    proposed = visible | search
    evidence = validate_amodal_proposal(visible, proposed, search, 0.95)
    assert evidence.accepted
    assert evidence.orthogonal_precision == 1.0
    assert evidence.orthogonal_span_ratio < 1.30
