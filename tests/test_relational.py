import numpy as np
from PIL import Image

from hpid_split.fusion import MaskCandidate
from hpid_split.prompt_bank import DomainPrompt, PartPrompt, PromptBank
from hpid_split.relational import propose_relational_candidates
from hpid_split.taxonomy import Taxonomy


def _fixture() -> tuple[Image.Image, np.ndarray, Taxonomy, PromptBank]:
    rgb = np.full((96, 128, 3), 245, dtype=np.uint8)
    rgb[10:90, 20:108] = (226, 178, 142)
    rgb[55:68, 34:51] = (42, 31, 25)
    rgb[55:68, 77:94] = (42, 31, 25)
    rgb[44:48, 32:53] = (64, 43, 31)
    rgb[44:48, 75:96] = (64, 43, 31)
    rgb[52:56, 33:52] = (35, 26, 22)
    rgb[52:56, 76:95] = (35, 26, 22)

    labels = np.zeros((96, 128), dtype=np.uint8)
    labels[10:90, 20:108] = 1
    labels[55:68, 34:51] = 2
    labels[55:68, 77:94] = 2
    taxonomy = Taxonomy(
        fine_names=("background", "asset_head", "asset_eye"),
        parent_names=("background", "asset", "asset_head"),
        fine_to_parent=(0, 1, 2),
        detail_names=("asset_eye",),
    )
    prompt_bank = PromptBank(
        (
            DomainPrompt(
                name="asset",
                root_prompts=("object",),
                parts=(
                    PartPrompt(
                        semantic_name="asset_head",
                        prompts=("head",),
                        maximum_instances=1,
                    ),
                    PartPrompt(
                        semantic_name="asset_eye",
                        semantic_parent="asset_head",
                        prompts=("eye",),
                        maximum_instances=2,
                    ),
                    PartPrompt(
                        semantic_name="asset_brow",
                        semantic_parent="asset_head",
                        prompts=("brow",),
                        maximum_parent_fraction=0.10,
                        maximum_instances=2,
                        appearance_anchor="asset_eye",
                        appearance_relation="above",
                        appearance_minimum_contrast=0.04,
                    ),
                    PartPrompt(
                        semantic_name="asset_lash",
                        semantic_parent="asset_head",
                        prompts=("lash",),
                        maximum_parent_fraction=0.10,
                        maximum_instances=2,
                        appearance_anchor="asset_eye",
                        appearance_relation="upper_boundary",
                        appearance_minimum_contrast=0.03,
                    ),
                ),
            ),
        )
    )
    return Image.fromarray(rgb), labels, taxonomy, prompt_bank


def test_relational_appearance_recovers_dark_details_from_eye_anchors() -> None:
    image, labels, taxonomy, prompt_bank = _fixture()

    generated = propose_relational_candidates(
        image, labels, taxonomy, prompt_bank
    )

    brows = [
        item.mask for item in generated.candidates if item.semantic_name == "asset_brow"
    ]
    lashes = [
        item.mask for item in generated.candidates if item.semantic_name == "asset_lash"
    ]
    expected_brows = np.zeros(labels.shape, dtype=bool)
    expected_brows[44:48, 32:53] = True
    expected_brows[44:48, 75:96] = True
    expected_lashes = np.zeros(labels.shape, dtype=bool)
    expected_lashes[52:56, 33:52] = True
    expected_lashes[52:56, 76:95] = True

    assert len(brows) == 2
    assert len(lashes) == 2
    assert np.count_nonzero(np.logical_or.reduce(brows) & expected_brows) > 100
    assert np.count_nonzero(np.logical_or.reduce(lashes) & expected_lashes) > 40
    assert generated.diagnostics["ground_truth_used"] is False
    assert all(
        candidate.metadata["ground_truth_used"] is False
        for candidate in generated.candidates
    )


def test_relational_appearance_emits_no_candidate_without_anchor_prediction() -> None:
    image, labels, taxonomy, prompt_bank = _fixture()
    labels[labels == 2] = 1

    generated = propose_relational_candidates(
        image, labels, taxonomy, prompt_bank
    )

    assert generated.candidates == ()
    assert generated.diagnostics["candidate_count"] == 0


def test_repetitive_control_regions_receive_independent_part_candidates() -> None:
    rgb = np.full((180, 100, 3), 210, dtype=np.uint8)
    labels = np.zeros((180, 100), dtype=np.uint8)
    labels[10:170, 20:80] = 1
    for row in range(5):
        for column in range(3):
            y = 35 + row * 22
            x = 30 + column * 17
            rgb[y : y + 9, x : x + 12] = 35
    taxonomy = Taxonomy(
        fine_names=("background", "device"),
        parent_names=("background", "device"),
        fine_to_parent=(0, 1),
        detail_names=(),
    )
    prompt_bank = PromptBank(
        (DomainPrompt(name="device", root_prompts=("device",), parts=()),)
    )
    root_mask = labels == 1
    root = MaskCandidate(
        semantic_name="device",
        semantic_parent="device",
        mask=root_mask,
        score=0.9,
        source="fixture",
        metadata={
            "candidate_key": "fixture:device:1",
            "selected_part_profile": "controls",
        },
    )

    generated = propose_relational_candidates(
        Image.fromarray(rgb),
        labels,
        taxonomy,
        prompt_bank,
        roots=(root,),
    )

    buttons = [
        candidate
        for candidate in generated.candidates
        if candidate.semantic_name == "device_button"
    ]
    assert len(buttons) == 15
    assert all(candidate.semantic_parent == "device_body" for candidate in buttons)
    assert all(candidate.metadata["ground_truth_used"] is False for candidate in buttons)
    rows = generated.diagnostics["repetitive_physical_details"]
    assert rows[0]["status"] == "accepted"
    assert rows[0]["accepted_count"] == 15


def test_repetitive_detail_rule_is_profile_gated() -> None:
    rgb = np.full((120, 100, 3), 210, dtype=np.uint8)
    labels = np.zeros((120, 100), dtype=np.uint8)
    labels[10:110, 20:80] = 1
    for row in range(4):
        rgb[25 + row * 18 : 33 + row * 18, 35:47] = 35
    taxonomy = Taxonomy(
        fine_names=("background", "device"),
        parent_names=("background", "device"),
        fine_to_parent=(0, 1),
        detail_names=(),
    )
    prompt_bank = PromptBank(
        (DomainPrompt(name="device", root_prompts=("device",), parts=()),)
    )
    root = MaskCandidate(
        semantic_name="device",
        semantic_parent="device",
        mask=labels == 1,
        score=0.9,
        source="fixture",
        metadata={
            "candidate_key": "fixture:device:1",
            "selected_part_profile": "phone",
        },
    )

    generated = propose_relational_candidates(
        Image.fromarray(rgb), labels, taxonomy, prompt_bank, roots=(root,)
    )

    assert generated.candidates == ()
