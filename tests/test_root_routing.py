import numpy as np

from hpid_split.fusion import MaskCandidate
from hpid_split.root_routing import (
    RootRoutingConfig,
    _select_salient_group,
    propagate_scene_object_identity,
    route_asset_roots,
)


def _candidate(
    semantic_name: str,
    semantic_parent: str,
    mask: np.ndarray,
    score: float,
    origin: str,
    root_index: int,
    *,
    root: bool = False,
    domain_evidence: float | None = None,
    domain_contrast: float | None = None,
    root_label_specificity: float = 0.0,
    part_profile_specificity: float = 0.0,
    selected_part_profile: str | None = None,
    profile_hint_source: str | None = None,
    semantic_mask_probability: float | None = None,
    root_query_mode: str | None = None,
    global_proposal_rank: int | None = None,
    global_proposal_score: float = 0.0,
    global_proposal_accepted: bool = False,
) -> MaskCandidate:
    root_key = f"root:{root_index}"
    return MaskCandidate(
        semantic_name,
        semantic_parent,
        mask,
        score,
        f"{origin}/{'root' if root else 'hierarchy-1'}",
        metadata={
            "root_origin": origin,
            "root_index": root_index,
            "candidate_key": root_key if root else f"{root_key}/{semantic_name}:01",
            "parent_candidate_key": None if root else root_key,
            "sam_quality": 0.9,
            "root_label_specificity": root_label_specificity,
            "part_profile_specificity": part_profile_specificity,
            **(
                {
                    "sam_multimask_selection": {
                        "selected_index": 0,
                        "target_rows": [
                            {"probability": semantic_mask_probability, "rank": 1}
                        ],
                    }
                }
                if semantic_mask_probability is not None
                else {}
            ),
            **(
                {
                    "selected_part_profile": selected_part_profile,
                    "profile_hint_source": profile_hint_source,
                }
                if selected_part_profile is not None
                else {}
            ),
            **(
                {"root_query_mode": root_query_mode}
                if root_query_mode is not None
                else {}
            ),
            **(
                {
                    "global_asset_proposal_rank": global_proposal_rank,
                    "global_asset_proposal_score": global_proposal_score,
                    "global_asset_proposal_accepted": global_proposal_accepted,
                }
                if global_proposal_rank is not None
                else {}
            ),
            **(
                {
                    "domain_evidence_score": domain_evidence,
                    "domain_evidence_contrast": domain_contrast or 0.0,
                }
                if domain_evidence is not None
                else {}
            ),
        },
    )


def test_scene_identity_is_propagated_to_descendants_by_root_key() -> None:
    shape = (40, 60)
    root_mask = np.zeros(shape, dtype=bool)
    root_mask[4:36, 6:54] = True
    child_mask = np.zeros(shape, dtype=bool)
    child_mask[12:28, 18:42] = True
    root = _candidate("asset", "asset", root_mask, 0.95, "detector", 7, root=True)
    root = MaskCandidate(
        root.semantic_name,
        root.semantic_parent,
        root.mask,
        root.score,
        root.source,
        metadata={
            **root.metadata,
            "scene_object_id": "object_004",
            "physical_group_id": "physical:04",
            "scene_role": "primary",
        },
    )
    child = _candidate("asset_panel", "asset", child_mask, 0.82, "detector", 7)

    propagated, diagnostics = propagate_scene_object_identity([root, child], [root])

    assert propagated[1].metadata["scene_object_id"] == "object_004"
    assert propagated[1].metadata["physical_group_id"] == "physical:04"
    assert propagated[1].metadata["scene_identity_propagated"] is True
    assert diagnostics["propagated_candidate_count"] == 1


def test_direct_user_prompt_root_outranks_unrelated_support_geometry() -> None:
    shape = (120, 180)
    belt = np.zeros(shape, dtype=bool)
    belt[48:68, 12:168] = True
    centered_support = np.zeros(shape, dtype=bool)
    centered_support[8:115, 58:122] = True
    candidates = [
        _candidate(
            "daily_object",
            "daily_object",
            belt,
            0.58,
            "prompt-model",
            1,
            root=True,
            semantic_mask_probability=0.64,
            root_query_mode="user_asset_prompt",
        ),
        _candidate(
            "daily_object",
            "daily_object",
            centered_support,
            0.96,
            "support-model",
            1,
            root=True,
            semantic_mask_probability=0.95,
            root_query_mode="user_asset_prompt_support",
        ),
    ]

    result = route_asset_roots(
        candidates,
        image_shape=shape,
        config=RootRoutingConfig(include_attached_roots=False),
    )

    selected = next(iter(result.candidates))
    assert np.array_equal(selected.mask, belt)
    assert result.diagnostics["prompt_owned_group_count"] == 1
    assert result.diagnostics["prompt_ownership_enforced"] is True


def test_direct_prompt_consensus_recovers_disconnected_visible_fragments() -> None:
    shape = (120, 220)
    left = np.zeros(shape, dtype=bool)
    left[46:66, 10:55] = True
    right = np.zeros(shape, dtype=bool)
    right[44:64, 165:210] = True
    composite = left | right
    composite[48:62, 92:128] = True
    contextual_host = np.zeros(shape, dtype=bool)
    contextual_host[1:119, 1:219] = True
    candidates = [
        _candidate(
            "daily_object",
            "daily_object",
            left,
            0.90,
            "prompt-left",
            1,
            root=True,
            semantic_mask_probability=0.94,
            root_query_mode="user_asset_prompt",
        ),
        _candidate(
            "daily_object",
            "daily_object",
            right,
            0.86,
            "prompt-right",
            1,
            root=True,
            semantic_mask_probability=0.88,
            root_query_mode="user_asset_prompt",
        ),
        _candidate(
            "daily_object",
            "daily_object",
            composite,
            0.48,
            "prompt-composite",
            1,
            root=True,
            semantic_mask_probability=0.67,
            root_query_mode="user_asset_prompt",
        ),
        _candidate(
            "daily_object",
            "daily_object",
            contextual_host,
            0.99,
            "prompt-context",
            1,
            root=True,
            semantic_mask_probability=0.98,
            root_query_mode="user_asset_prompt",
        ),
    ]

    result = route_asset_roots(
        candidates,
        image_shape=shape,
        config=RootRoutingConfig(include_attached_roots=False),
    )

    selected = next(iter(result.candidates))
    assert np.array_equal(selected.mask, composite)
    consensus = result.diagnostics["prompt_fragment_consensus"]
    assert consensus["selected_physical_group_id"] is not None
    selected_row = next(
        row
        for row in consensus["rows"]
        if row["physical_group_id"] == consensus["selected_physical_group_id"]
    )
    assert selected_row["support_count"] == 2
    context_row = next(
        row
        for row in consensus["rows"]
        if row["representative_root_key"] == "prompt-context::1"
    )
    assert context_row["eligible"] is False


def test_primary_router_prefers_text_supported_spanning_frame_over_solid_occluder(
) -> None:
    shape = (100, 100)
    framed_target = np.zeros(shape, dtype=bool)
    framed_target[8:95, 8:13] = True
    framed_target[8:13, 8:45] = True
    framed_target[45:50, 8:45] = True
    framed_target[90:95, 8:45] = True
    solid_occluder = np.zeros(shape, dtype=bool)
    solid_occluder[30:88, 52:98] = True
    candidates = [
        _candidate(
            "furniture",
            "furniture",
            framed_target,
            0.82,
            "model",
            1,
            root=True,
            semantic_mask_probability=0.94,
        ),
        _candidate(
            "furniture",
            "furniture",
            solid_occluder,
            0.86,
            "model",
            2,
            root=True,
            semantic_mask_probability=0.22,
        ),
    ]

    result = route_asset_roots(
        candidates,
        image_shape=shape,
        config=RootRoutingConfig(mode="primary", include_attached_roots=False),
    )

    selected = next(
        candidate
        for candidate in result.candidates
        if candidate.semantic_name == "furniture"
    )
    assert selected.metadata["root_index"] == 1
    score_rows = {
        row["root_key"]: row for row in result.diagnostics["root_scores"]
    }
    assert score_rows["model::1"]["frame_extent"] > score_rows["model::2"][
        "frame_extent"
    ]
    assert score_rows["model::1"]["semantic_mask_probability"] == 0.94


def test_primary_router_uses_extent_for_cross_semantic_salience_tie() -> None:
    shape = (100, 120)
    spanning_furniture = np.zeros(shape, dtype=bool)
    spanning_furniture[8:92, 5:115] = True
    compact_high_confidence_object = np.zeros(shape, dtype=bool)
    compact_high_confidence_object[36:68, 48:76] = True
    candidates = [
        _candidate(
            "furniture",
            "furniture",
            spanning_furniture,
            0.50,
            "model",
            1,
            root=True,
        ),
        _candidate(
            "character",
            "character",
            compact_high_confidence_object,
            0.65,
            "model",
            2,
            root=True,
        ),
    ]

    result = route_asset_roots(
        candidates,
        image_shape=shape,
        config=RootRoutingConfig(mode="primary", include_attached_roots=False),
    )

    assert result.diagnostics["selected_semantic"] == "furniture"


def test_salience_tie_break_does_not_double_count_detector_confidence() -> None:
    spanning = {
        "group_score": 0.700,
        "winner": {
            "routing_score": 0.45,
            "metrics": {
                "frame_extent": 0.99,
                "bbox_salience": 1.0,
                "area_salience": 1.0,
                "physical_salience_score": 0.72,
                "sam_score": 0.95,
                "detector_score": 0.65,
            },
        },
    }
    compact = {
        "group_score": 0.704,
        "winner": {
            "routing_score": 0.54,
            "metrics": {
                "frame_extent": 0.31,
                "bbox_salience": 0.60,
                "area_salience": 0.46,
                "physical_salience_score": 0.73,
                "sam_score": 0.97,
                "detector_score": 0.86,
            },
        },
    }

    selected, competitive_count = _select_salient_group([compact, spanning])

    assert selected is spanning
    assert competitive_count == 2


def test_rank_one_global_proposal_can_recover_a_sparse_occluded_root() -> None:
    occluded_root = {
        "group_score": 0.627,
        "physical_score": 0.601,
        "winner": {
            "routing_score": 0.29,
            "semantic_category_score": 0.879,
            "metrics": {
                "frame_extent": 0.99,
                "bbox_salience": 1.0,
                "area_salience": 0.70,
                "physical_salience_score": 0.601,
                "sam_score": 0.53,
                "detector_score": 0.84,
                "global_asset_proposal_rank": 1,
                "global_asset_proposal_priority": 0.72,
                "global_asset_proposal_accepted": False,
            },
        },
    }
    visible_subobject = {
        "group_score": 0.772,
        "physical_score": 0.838,
        "winner": {
            "routing_score": 0.54,
            "semantic_category_score": 0.778,
            "metrics": {
                "frame_extent": 0.53,
                "bbox_salience": 1.0,
                "area_salience": 1.0,
                "physical_salience_score": 0.838,
                "sam_score": 0.89,
                "detector_score": 0.69,
                "global_asset_proposal_rank": 3,
                "global_asset_proposal_priority": 0.27,
                "global_asset_proposal_accepted": False,
            },
        },
    }

    selected, _ = _select_salient_group([visible_subobject, occluded_root])

    assert selected is occluded_root


def test_weak_rank_one_proposal_cannot_replace_clear_physical_evidence() -> None:
    weak_proposal = {
        "group_score": 0.40,
        "physical_score": 0.43,
        "winner": {
            "routing_score": 0.25,
            "metrics": {
                "frame_extent": 0.95,
                "bbox_salience": 0.90,
                "area_salience": 0.30,
                "physical_salience_score": 0.43,
                "sam_score": 0.40,
                "detector_score": 0.52,
                "global_asset_proposal_rank": 1,
                "global_asset_proposal_priority": 0.70,
                "global_asset_proposal_accepted": False,
            },
        },
    }
    clear_object = {
        "group_score": 0.75,
        "physical_score": 0.82,
        "winner": {
            "routing_score": 0.60,
            "metrics": {
                "frame_extent": 0.68,
                "bbox_salience": 0.90,
                "area_salience": 0.90,
                "physical_salience_score": 0.82,
                "sam_score": 0.91,
                "detector_score": 0.88,
                "global_asset_proposal_rank": None,
                "global_asset_proposal_priority": 0.0,
                "global_asset_proposal_accepted": False,
            },
        },
    }

    selected, _ = _select_salient_group([weak_proposal, clear_object])

    assert selected is clear_object


def test_compact_ambiguous_rank_one_cannot_replace_complete_asset_root() -> None:
    compact_candidate = {
        "group_score": 0.712,
        "physical_score": 0.696,
        "winner": {
            "routing_score": 0.45,
            "semantic_category_score": 0.81,
            "metrics": {
                "frame_extent": 0.53,
                "bbox_salience": 0.83,
                "area_salience": 0.53,
                "physical_salience_score": 0.696,
                "sam_score": 0.96,
                "detector_score": 0.52,
                "global_asset_proposal_rank": 1,
                "global_asset_proposal_priority": 0.81,
                "global_asset_proposal_accepted": False,
            },
        },
    }
    complete_asset = {
        "group_score": 0.830,
        "physical_score": 0.820,
        "winner": {
            "routing_score": 0.49,
            "semantic_category_score": 0.87,
            "metrics": {
                "frame_extent": 0.91,
                "bbox_salience": 1.0,
                "area_salience": 1.0,
                "physical_salience_score": 0.820,
                "sam_score": 0.96,
                "detector_score": 1.0,
                "global_asset_proposal_rank": 3,
                "global_asset_proposal_priority": 0.37,
                "global_asset_proposal_accepted": False,
            },
        },
    }

    selected, _ = _select_salient_group([compact_candidate, complete_asset])

    assert selected is complete_asset


def test_unaccepted_rank_one_internal_accessory_cannot_replace_full_subject() -> None:
    full_subject = {
        "group_score": 0.830,
        "physical_score": 0.898,
        "winner": {
            "routing_score": 0.585,
            "semantic_category_score": 0.88,
            "metrics": {
                "frame_extent": 0.88,
                "bbox_salience": 1.0,
                "area_salience": 1.0,
                "physical_salience_score": 0.898,
                "sam_score": 0.95,
                "detector_score": 0.71,
                "global_asset_proposal_rank": None,
                "global_asset_proposal_priority": 0.0,
                "global_asset_proposal_accepted": False,
            },
        },
    }
    internal_accessory = {
        "group_score": 0.792,
        "physical_score": 0.807,
        "winner": {
            "routing_score": 0.457,
            "semantic_category_score": 0.92,
            "metrics": {
                "frame_extent": 0.47,
                "bbox_salience": 0.83,
                "area_salience": 0.66,
                "physical_salience_score": 0.807,
                "sam_score": 0.97,
                "detector_score": 1.0,
                "global_asset_proposal_rank": 1,
                "global_asset_proposal_priority": 0.71,
                "global_asset_proposal_accepted": False,
            },
        },
    }

    selected, _ = _select_salient_group([internal_accessory, full_subject])

    assert selected is full_subject


def test_accepted_global_proposal_wins_over_broader_same_asset_hypothesis() -> None:
    shape = (100, 100)
    broad = np.zeros(shape, dtype=bool)
    broad[10:90, 10:90] = True
    routed = np.zeros(shape, dtype=bool)
    routed[15:85, 15:85] = True
    candidates = [
        _candidate("tool_prop", "tool_prop", broad, 0.65, "broad", 1, root=True),
        _candidate(
            "tool_prop",
            "tool_prop",
            routed,
            0.55,
            "global",
            2,
            root=True,
            root_query_mode="global_asset_proposal",
            global_proposal_rank=1,
            global_proposal_score=0.28,
            global_proposal_accepted=True,
        ),
    ]

    result = route_asset_roots(
        candidates,
        image_shape=shape,
        config=RootRoutingConfig(mode="primary", include_attached_roots=False),
    )

    root = next(
        candidate
        for candidate in result.candidates
        if candidate.semantic_name == candidate.semantic_parent
    )
    assert root.metadata["root_origin"] == "global"
    selected_row = next(
        row for row in result.diagnostics["root_scores"] if row["selected"]
    )
    assert selected_row["global_asset_proposal_priority"] == 1.0


def test_low_rank_global_proposal_cannot_override_clear_geometry() -> None:
    shape = (100, 100)
    broad = np.zeros(shape, dtype=bool)
    broad[8:92, 8:92] = True
    weak = np.zeros(shape, dtype=bool)
    weak[35:65, 35:65] = True
    candidates = [
        _candidate("container", "container", broad, 0.62, "broad", 1, root=True),
        _candidate(
            "tool_prop",
            "tool_prop",
            weak,
            0.40,
            "global",
            2,
            root=True,
            root_query_mode="global_asset_proposal",
            global_proposal_rank=5,
            global_proposal_score=0.18,
        ),
    ]

    result = route_asset_roots(
        candidates,
        image_shape=shape,
        config=RootRoutingConfig(mode="primary", include_attached_roots=False),
    )

    assert result.diagnostics["selected_semantic"] == "container"


def test_primary_router_keeps_cross_source_subject_and_rejects_background() -> None:
    shape = (100, 120)
    primary_a = np.zeros(shape, dtype=bool)
    primary_a[18:94, 28:78] = True
    primary_b = np.zeros(shape, dtype=bool)
    primary_b[20:94, 30:80] = True
    head = np.zeros(shape, dtype=bool)
    head[20:42, 42:66] = True
    body = np.zeros(shape, dtype=bool)
    body[43:88, 34:74] = True
    background = np.zeros(shape, dtype=bool)
    background[:28, :] = True
    unrelated = np.zeros(shape, dtype=bool)
    unrelated[44:76, 92:116] = True
    candidates = [
        _candidate("character", "character", primary_a, 0.58, "model-a", 1, root=True),
        _candidate("character_head", "character", head, 0.51, "model-a", 1),
        _candidate("character_body", "character", body, 0.49, "model-a", 1),
        _candidate("structure", "structure", background, 0.92, "model-a", 2, root=True),
        _candidate("character", "character", primary_b, 0.62, "model-b", 4, root=True),
        _candidate("container", "container", unrelated, 0.55, "model-b", 5, root=True),
    ]

    result = route_asset_roots(
        candidates,
        image_shape=shape,
        config=RootRoutingConfig(mode="primary", include_attached_roots=False),
    )

    assert result.diagnostics["selected_semantic"] == "character"
    assert result.diagnostics["selected_root_count"] == 1
    assert result.diagnostics["selected_evidence_root_count"] == 2
    assert {item.semantic_name for item in result.candidates} == {
        "character",
        "character_head",
        "character_body",
    }
    assert sum(item.semantic_name == "character" for item in result.candidates) == 1
    canonical = next(
        item for item in result.candidates if item.semantic_name == "character"
    )
    for child in (
        item for item in result.candidates if item.semantic_name != "character"
    ):
        assert child.metadata["root_origin"] == canonical.metadata["root_origin"]
        assert child.metadata["root_index"] == canonical.metadata["root_index"]
    assert result.diagnostics["ground_truth_used"] is False


def test_primary_router_keeps_coherent_attached_prop_not_fragmented_false_root() -> (
    None
):
    shape = (80, 100)
    character = np.zeros(shape, dtype=bool)
    character[10:76, 25:62] = True
    staff = np.zeros(shape, dtype=bool)
    staff[8:74, 63:66] = True
    fragmented = np.zeros(shape, dtype=bool)
    fragmented[20:22, 30:32] = True
    fragmented[35:37, 45:47] = True
    fragmented[50:52, 58:60] = True
    candidates = [
        _candidate("character", "character", character, 0.61, "model-a", 1, root=True),
        _candidate("tool_prop", "tool_prop", staff, 0.42, "model-a", 2, root=True),
        _candidate("device", "device", fragmented, 0.80, "model-a", 3, root=True),
    ]

    result = route_asset_roots(candidates, image_shape=shape)

    selected = {item.semantic_name for item in result.candidates}
    assert selected == {"character", "tool_prop"}
    roles = {
        row["semantic_name"]: row["selection_role"]
        for row in result.diagnostics["root_scores"]
        if row["selected"]
    }
    assert roles == {"character": "primary", "tool_prop": "attached_root"}


def test_primary_router_does_not_promote_contained_alternative_domain_root() -> None:
    shape = (80, 100)
    tool = np.zeros(shape, dtype=bool)
    tool[12:70, 8:92] = True
    contained_container = np.zeros(shape, dtype=bool)
    contained_container[30:62, 45:82] = True
    candidates = [
        _candidate("tool_prop", "tool_prop", tool, 0.70, "model-a", 1, root=True),
        _candidate(
            "container",
            "container",
            contained_container,
            0.50,
            "model-a",
            2,
            root=True,
        ),
    ]

    result = route_asset_roots(candidates, image_shape=shape)

    assert {item.semantic_name for item in result.candidates} == {"tool_prop"}


def test_primary_router_rejects_border_surface_with_weak_domain_evidence() -> None:
    shape = (100, 120)
    character = np.zeros(shape, dtype=bool)
    character[10:92, 35:82] = True
    floor = np.zeros(shape, dtype=bool)
    floor[88:100, :] = True
    false_seat = np.zeros(shape, dtype=bool)
    false_seat[90:100, 12:108] = True
    candidates = [
        _candidate("character", "character", character, 0.75, "model", 1, root=True),
        _candidate(
            "furniture",
            "furniture",
            floor,
            0.70,
            "model",
            2,
            root=True,
            domain_evidence=0.0002,
            domain_contrast=0.0001,
        ),
        _candidate("furniture_seat", "furniture", false_seat, 0.62, "model", 2),
    ]

    result = route_asset_roots(candidates, image_shape=shape)

    assert {item.semantic_name for item in result.candidates} == {"character"}
    floor_row = next(
        row
        for row in result.diagnostics["root_scores"]
        if row["semantic_name"] == "furniture"
    )
    assert floor_row["attachment_rejection_reason"] == "weak_border_domain"


def test_all_root_mode_is_lossless() -> None:
    shape = (30, 30)
    first = np.zeros(shape, dtype=bool)
    first[2:12, 2:12] = True
    second = np.zeros(shape, dtype=bool)
    second[18:28, 18:28] = True
    candidates = [
        _candidate("asset", "asset", first, 0.7, "model", 1, root=True),
        _candidate("container", "container", second, 0.6, "model", 2, root=True),
    ]

    result = route_asset_roots(
        candidates,
        image_shape=shape,
        config=RootRoutingConfig(mode="all"),
    )

    assert result.candidates == tuple(candidates)
    assert result.diagnostics["rejected_candidate_count"] == 0


def test_scene_mode_canonicalizes_each_physical_object_once() -> None:
    shape = (100, 160)
    character_a = np.zeros(shape, dtype=bool)
    character_a[8:94, 8:58] = True
    character_b = np.zeros(shape, dtype=bool)
    character_b[10:94, 10:60] = True
    head_a = np.zeros(shape, dtype=bool)
    head_a[10:34, 20:46] = True
    head_b = np.zeros(shape, dtype=bool)
    head_b[11:35, 21:47] = True
    weapon = np.zeros(shape, dtype=bool)
    weapon[34:68, 80:152] = True
    stock = np.zeros(shape, dtype=bool)
    stock[42:63, 121:151] = True
    candidates = [
        _candidate(
            "character", "character", character_a, 0.72, "model-a", 1, root=True
        ),
        _candidate("character_head", "character", head_a, 0.61, "model-a", 1),
        _candidate(
            "character", "character", character_b, 0.69, "model-b", 3, root=True
        ),
        _candidate("character_head", "character", head_b, 0.58, "model-b", 3),
        _candidate("tool_prop", "tool_prop", weapon, 0.68, "model-a", 2, root=True),
        _candidate("tool_prop_stock", "tool_prop", stock, 0.57, "model-a", 2),
    ]

    result = route_asset_roots(
        candidates,
        image_shape=shape,
        config=RootRoutingConfig(mode="scene"),
    )

    roots = [
        candidate
        for candidate in result.candidates
        if candidate.semantic_name == candidate.semantic_parent
    ]
    assert len(roots) == 2
    assert {candidate.semantic_name for candidate in roots} == {
        "character",
        "tool_prop",
    }
    assert result.diagnostics["selected_root_count"] == 2
    assert result.diagnostics["physical_group_count"] == 2
    assert {candidate.metadata["scene_object_id"] for candidate in roots} == {
        "object_001",
        "object_002",
    }
    character_root = next(
        candidate for candidate in roots if candidate.semantic_name == "character"
    )
    character_parts = [
        candidate
        for candidate in result.candidates
        if candidate.semantic_name == "character_head"
    ]
    assert len(character_parts) == 2
    assert all(
        candidate.metadata["root_origin"] == character_root.metadata["root_origin"]
        for candidate in character_parts
    )
    assert all(
        candidate.metadata["scene_object_id"]
        == character_root.metadata["scene_object_id"]
        for candidate in character_parts
    )


def test_scene_mode_keeps_two_instances_of_the_same_category_separate() -> None:
    shape = (80, 140)
    left = np.zeros(shape, dtype=bool)
    left[10:70, 6:55] = True
    right = np.zeros(shape, dtype=bool)
    right[12:72, 84:134] = True
    candidates = [
        _candidate("container", "container", left, 0.70, "model", 1, root=True),
        _candidate("container", "container", right, 0.69, "model", 2, root=True),
    ]

    result = route_asset_roots(
        candidates,
        image_shape=shape,
        config=RootRoutingConfig(mode="scene"),
    )

    assert len(result.candidates) == 2
    assert (
        len({candidate.metadata["scene_object_id"] for candidate in result.candidates})
        == 2
    )


def test_scene_mode_merges_same_box_fragmented_root_hypotheses() -> None:
    shape = (80, 100)
    first = np.zeros(shape, dtype=bool)
    second = np.zeros(shape, dtype=bool)
    first[10:70:2, 20:80] = True
    first[69, 20:80] = True
    second[11:70:2, 20:80] = True
    second[10, 20:80] = True
    candidates = [
        _candidate("furniture", "furniture", first, 0.61, "model-a", 1, root=True),
        _candidate(
            "natural_object",
            "natural_object",
            second,
            0.66,
            "model-b",
            2,
            root=True,
        ),
    ]

    result = route_asset_roots(
        candidates,
        image_shape=shape,
        config=RootRoutingConfig(mode="scene"),
    )

    assert result.diagnostics["physical_group_count"] == 1
    assert result.diagnostics["selected_root_count"] == 1


def test_scene_mode_marks_terrain_as_a_scene_layer() -> None:
    shape = (80, 120)
    terrain = np.zeros(shape, dtype=bool)
    terrain[52:, :] = True
    result = route_asset_roots(
        [_candidate("terrain", "terrain", terrain, 0.75, "model", 1, root=True)],
        image_shape=shape,
        config=RootRoutingConfig(mode="scene"),
    )

    assert result.candidates[0].metadata["scene_role"] == "scene_layer"


def test_scene_mode_keeps_small_terrain_chunk_as_an_object() -> None:
    shape = (80, 120)
    terrain_chunk = np.zeros(shape, dtype=bool)
    terrain_chunk[20:30, 40:50] = True
    result = route_asset_roots(
        [
            _candidate(
                "terrain",
                "terrain",
                terrain_chunk,
                0.75,
                "model",
                1,
                root=True,
            )
        ],
        image_shape=shape,
        config=RootRoutingConfig(mode="scene"),
    )

    assert result.candidates[0].metadata["scene_role"] == "object"


def test_single_root_primary_mode_reports_selected_semantic() -> None:
    shape = (30, 30)
    mask = np.zeros(shape, dtype=bool)
    mask[4:26, 5:25] = True
    candidate = _candidate("device", "device", mask, 0.7, "model", 1, root=True)

    result = route_asset_roots([candidate], image_shape=shape)

    assert result.diagnostics["selected_semantic"] == "device"
    assert result.diagnostics["selected_root_count"] == 1


def test_target_point_selects_one_of_multiple_same_category_instances() -> None:
    shape = (80, 120)
    left = np.zeros(shape, dtype=bool)
    left[12:70, 6:48] = True
    right = np.zeros(shape, dtype=bool)
    right[18:68, 72:114] = True
    candidates = [
        _candidate("furniture", "furniture", left, 0.91, "model", 1, root=True),
        _candidate("furniture", "furniture", right, 0.62, "model", 2, root=True),
    ]

    result = route_asset_roots(
        candidates,
        image_shape=shape,
        config=RootRoutingConfig(
            include_attached_roots=False,
            target_point_xy=(92.0, 42.0),
        ),
    )

    selected = next(iter(result.candidates))
    assert np.array_equal(selected.mask, right)
    assert result.diagnostics["target_point_routing"]["status"] == (
        "selected_containing_root"
    )
    assert result.diagnostics["ground_truth_used"] is False


def test_target_point_outside_image_is_rejected() -> None:
    shape = (40, 60)
    mask = np.zeros(shape, dtype=bool)
    mask[5:35, 8:52] = True
    candidate = _candidate("asset", "asset", mask, 0.8, "model", 1, root=True)

    with np.testing.assert_raises(ValueError):
        route_asset_roots(
            [candidate],
            image_shape=shape,
            config=RootRoutingConfig(target_point_xy=(100.0, 20.0)),
        )


def test_physical_first_router_resolves_same_phone_mask_as_device() -> None:
    shape = (120, 100)
    phone = np.zeros(shape, dtype=bool)
    phone[12:112, 18:82] = True
    candidates = [
        _candidate(
            "character",
            "character",
            phone,
            0.44,
            "model",
            1,
            root=True,
            domain_evidence=0.58,
            domain_contrast=0.10,
            root_label_specificity=0.88,
        ),
        _candidate(
            "device",
            "device",
            phone,
            0.40,
            "model",
            2,
            root=True,
            domain_evidence=0.66,
            domain_contrast=0.18,
            root_label_specificity=0.88,
            part_profile_specificity=0.88,
        ),
    ]

    result = route_asset_roots(
        candidates,
        image_shape=shape,
        config=RootRoutingConfig(include_attached_roots=False),
    )

    assert result.diagnostics["selected_semantic"] == "device"
    assert len(result.diagnostics["physical_groups"]) == 1
    assert result.diagnostics["physical_groups"][0]["semantic_names"] == [
        "character",
        "device",
    ]


def test_category_evidence_can_use_screen_but_geometry_keeps_full_device() -> None:
    shape = (120, 100)
    full_phone = np.zeros(shape, dtype=bool)
    full_phone[8:114, 20:80] = True
    screen = np.zeros(shape, dtype=bool)
    screen[20:86, 28:72] = True
    candidates = [
        _candidate(
            "device",
            "device",
            screen,
            0.51,
            "model",
            1,
            root=True,
            domain_evidence=0.82,
            domain_contrast=0.16,
            root_label_specificity=1.0,
            part_profile_specificity=0.88,
        ),
        _candidate(
            "device",
            "device",
            full_phone,
            0.47,
            "model",
            2,
            root=True,
            domain_evidence=0.62,
            domain_contrast=0.12,
            root_label_specificity=0.88,
            part_profile_specificity=0.88,
        ),
        _candidate(
            "daily_object",
            "daily_object",
            full_phone,
            0.54,
            "model",
            3,
            root=True,
            domain_evidence=0.45,
            domain_contrast=0.09,
            root_label_specificity=0.35,
        ),
    ]

    result = route_asset_roots(
        candidates,
        image_shape=shape,
        config=RootRoutingConfig(include_attached_roots=False),
    )

    assert result.diagnostics["selected_semantic"] == "device"
    selected_root = next(
        candidate
        for candidate in result.candidates
        if candidate.semantic_name == "device"
    )
    assert np.array_equal(selected_root.mask, full_phone)


def test_physical_first_router_prefers_full_bottle_over_small_false_root() -> None:
    shape = (120, 100)
    bottle = np.zeros(shape, dtype=bool)
    bottle[8:114, 22:78] = True
    label_patch = np.zeros(shape, dtype=bool)
    label_patch[50:70, 38:62] = True
    candidates = [
        _candidate(
            "natural_object",
            "natural_object",
            label_patch,
            0.61,
            "model",
            1,
            root=True,
            domain_evidence=0.82,
            domain_contrast=0.20,
        ),
        _candidate(
            "natural_object",
            "natural_object",
            bottle,
            0.46,
            "model",
            2,
            root=True,
            domain_evidence=0.70,
            domain_contrast=0.14,
        ),
        _candidate(
            "container",
            "container",
            bottle,
            0.43,
            "model",
            3,
            root=True,
            domain_evidence=0.74,
            domain_contrast=0.18,
            root_label_specificity=1.0,
            part_profile_specificity=1.0,
        ),
    ]

    result = route_asset_roots(
        candidates,
        image_shape=shape,
        config=RootRoutingConfig(include_attached_roots=False),
    )

    assert result.diagnostics["selected_semantic"] == "container"
    assert len(result.diagnostics["physical_groups"]) == 2
    selected_group = next(
        item
        for item in result.diagnostics["physical_groups"]
        if item["selected_as_primary"]
    )
    assert selected_group["semantic_names"] == ["container", "natural_object"]


def test_specific_detector_margin_overrides_context_biased_domain_rank() -> None:
    shape = (160, 120)
    lamp = np.zeros(shape, dtype=bool)
    lamp[28:144, 44:78] = True
    candidates = [
        _candidate(
            "device",
            "device",
            lamp,
            0.45,
            "model",
            1,
            root=True,
            domain_evidence=0.05,
            root_label_specificity=1.0,
            part_profile_specificity=1.0,
            selected_part_profile="lamp",
        ),
        _candidate(
            "tool_prop",
            "tool_prop",
            lamp,
            0.24,
            "model",
            2,
            root=True,
            domain_evidence=0.85,
            root_label_specificity=1.0,
            part_profile_specificity=1.0,
            selected_part_profile="drill",
        ),
    ]

    result = route_asset_roots(
        candidates,
        image_shape=shape,
        config=RootRoutingConfig(include_attached_roots=False),
    )

    assert result.diagnostics["selected_semantic"] == "device"
    selected = next(iter(result.candidates))
    assert selected.metadata["semantic_arbitration_override"] == (
        "specific_detector_margin"
    )


def test_specific_profile_detector_overrides_unspecific_context_label() -> None:
    shape = (160, 120)
    lamp = np.zeros(shape, dtype=bool)
    lamp[12:148, 31:83] = True
    candidates = [
        _candidate(
            "device",
            "device",
            lamp,
            0.41,
            "model",
            1,
            root=True,
            domain_evidence=0.08,
            root_label_specificity=1.0,
            part_profile_specificity=1.0,
            selected_part_profile="lamp",
        ),
        _candidate(
            "tool_prop",
            "tool_prop",
            lamp,
            0.24,
            "model",
            2,
            root=True,
            domain_evidence=0.82,
            root_label_specificity=1.0,
        ),
    ]

    result = route_asset_roots(
        candidates,
        image_shape=shape,
        config=RootRoutingConfig(include_attached_roots=False),
    )

    assert result.diagnostics["selected_semantic"] == "device"


def test_global_proposal_rejects_full_frame_degenerate_duplicate() -> None:
    shape = (100, 100)
    full_frame = np.ones(shape, dtype=bool)
    tight_object = np.zeros(shape, dtype=bool)
    tight_object[5:76, 14:86] = True
    candidates = [
        _candidate(
            "device",
            "device",
            full_frame,
            0.50,
            "model",
            1,
            root=True,
            root_label_specificity=1.0,
            part_profile_specificity=1.0,
            selected_part_profile="clock_watch",
            global_proposal_rank=1,
            global_proposal_score=0.30,
            global_proposal_accepted=True,
        ),
        _candidate(
            "device",
            "device",
            tight_object,
            0.28,
            "model",
            2,
            root=True,
            root_label_specificity=1.0,
            part_profile_specificity=1.0,
            selected_part_profile="clock_watch",
            global_proposal_rank=1,
            global_proposal_score=0.30,
            global_proposal_accepted=True,
        ),
    ]

    result = route_asset_roots(
        candidates,
        image_shape=shape,
        config=RootRoutingConfig(include_attached_roots=False),
    )

    selected = next(iter(result.candidates))
    assert np.array_equal(selected.mask, tight_object)


def test_physical_first_router_keeps_distinct_person_and_vehicle_groups() -> None:
    shape = (120, 160)
    person = np.zeros(shape, dtype=bool)
    person[12:112, 10:52] = True
    vehicle = np.zeros(shape, dtype=bool)
    vehicle[42:106, 52:154] = True
    candidates = [
        _candidate(
            "character",
            "character",
            person,
            0.72,
            "model",
            1,
            root=True,
            domain_evidence=0.80,
        ),
        _candidate(
            "vehicle",
            "vehicle",
            vehicle,
            0.68,
            "model",
            2,
            root=True,
            domain_evidence=0.79,
            part_profile_specificity=1.0,
        ),
    ]

    result = route_asset_roots(candidates, image_shape=shape)

    assert len(result.diagnostics["physical_groups"]) == 2
    assert {
        tuple(item["semantic_names"]) for item in result.diagnostics["physical_groups"]
    } == {("character",), ("vehicle",)}


def test_isolated_profile_evidence_survives_full_geometry_selection() -> None:
    shape = (120, 100)
    full_headset = np.zeros(shape, dtype=bool)
    full_headset[10:110, 16:84] = True
    earpad = np.zeros(shape, dtype=bool)
    earpad[34:86, 22:49] = True
    candidates = [
        _candidate(
            "device",
            "device",
            full_headset,
            0.46,
            "broad-model",
            1,
            root=True,
            domain_evidence=0.58,
            root_label_specificity=0.35,
        ),
        _candidate(
            "device",
            "device",
            earpad,
            0.72,
            "profile-model",
            2,
            root=True,
            domain_evidence=0.82,
            domain_contrast=0.18,
            root_label_specificity=1.0,
            part_profile_specificity=1.0,
            selected_part_profile="earphone",
            profile_hint_source="isolated_profile_query",
        ),
    ]

    result = route_asset_roots(
        candidates,
        image_shape=shape,
        config=RootRoutingConfig(include_attached_roots=False),
    )

    selected = next(iter(result.candidates))
    assert np.array_equal(selected.mask, full_headset)
    assert selected.metadata["selected_part_profile"] == "earphone"
    assert selected.metadata["profile_hint_source"] == "isolated_profile_query"


def test_profile_geometry_cannot_jump_to_contextual_host_group() -> None:
    shape = (400, 100)
    headset = np.zeros(shape, dtype=bool)
    headset[30:95, 10:90] = True
    contextual_person = np.zeros(shape, dtype=bool)
    contextual_person[30:385, 4:96] = True
    candidates = [
        _candidate(
            "device",
            "device",
            headset,
            1.0,
            "tiny-model",
            1,
            root=True,
            root_label_specificity=1.0,
            part_profile_specificity=1.0,
            selected_part_profile="earphone",
            profile_hint_source="user_asset_prompt",
            semantic_mask_probability=0.97,
        ),
        _candidate(
            "device",
            "device",
            headset,
            0.94,
            "base-model",
            1,
            root=True,
            root_label_specificity=1.0,
            part_profile_specificity=1.0,
            selected_part_profile="earphone",
            profile_hint_source="user_asset_prompt",
            semantic_mask_probability=0.92,
        ),
        _candidate(
            "device",
            "device",
            contextual_person,
            0.28,
            "base-model",
            2,
            root=True,
            root_label_specificity=1.0,
            part_profile_specificity=1.0,
            selected_part_profile="earphone",
            profile_hint_source="user_asset_prompt",
            semantic_mask_probability=0.28,
        ),
    ]

    result = route_asset_roots(
        candidates,
        image_shape=shape,
        config=RootRoutingConfig(include_attached_roots=False),
    )

    selected = next(iter(result.candidates))
    assert np.array_equal(selected.mask, headset)
    assert result.diagnostics["selected_physical_group_id"] == "physical:01"
    assert result.diagnostics["canonical_root_key"] in {
        "base-model::1",
        "tiny-model::1",
    }


def test_profile_geometry_can_expand_to_audited_cross_source_object() -> None:
    shape = (140, 100)
    pump_crop = np.zeros(shape, dtype=bool)
    pump_crop[10:40, 30:70] = True
    full_dispenser = np.zeros(shape, dtype=bool)
    full_dispenser[15:132, 20:80] = True
    contextual_host = np.zeros(shape, dtype=bool)
    contextual_host[15:139, 2:98] = True
    pump_detail = np.zeros(shape, dtype=bool)
    pump_detail[12:28, 40:60] = True
    candidates = [
        _candidate(
            "container",
            "container",
            pump_crop,
            0.96,
            "tiny-model",
            1,
            root=True,
            root_label_specificity=1.0,
            part_profile_specificity=1.0,
            selected_part_profile="soap_dispenser",
            profile_hint_source="user_asset_prompt",
            semantic_mask_probability=0.98,
        ),
        _candidate(
            "pump",
            "container",
            pump_detail,
            0.88,
            "tiny-model",
            1,
        ),
        _candidate(
            "container",
            "container",
            full_dispenser,
            0.90,
            "base-model",
            1,
            root=True,
            root_label_specificity=1.0,
            part_profile_specificity=1.0,
            selected_part_profile="soap_dispenser",
            profile_hint_source="user_asset_prompt",
            semantic_mask_probability=0.92,
        ),
        _candidate(
            "container",
            "container",
            contextual_host,
            0.05,
            "base-model",
            2,
            root=True,
            root_label_specificity=1.0,
            part_profile_specificity=1.0,
            selected_part_profile="soap_dispenser",
            profile_hint_source="user_asset_prompt",
            semantic_mask_probability=0.05,
        ),
    ]

    result = route_asset_roots(
        candidates,
        image_shape=shape,
        config=RootRoutingConfig(
            include_attached_roots=False,
            target_point_xy=(50.0, 12.0),
        ),
    )

    roots = [
        candidate
        for candidate in result.candidates
        if candidate.semantic_name == candidate.semantic_parent
    ]
    assert len(roots) == 1
    assert np.array_equal(roots[0].mask, full_dispenser)
    assert not np.array_equal(roots[0].mask, contextual_host)
    assert result.diagnostics["canonical_root_key"] == "base-model::1"
    roles = {
        row["root_key"]: row["selection_role"]
        for row in result.diagnostics["root_scores"]
    }
    assert roles["tiny-model::1"] == "profile_geometry_evidence"
    assert roles["base-model::1"] == "primary"


def test_full_geometry_keeps_its_specific_profile_during_same_domain_fusion() -> None:
    shape = (120, 100)
    full_tree = np.zeros(shape, dtype=bool)
    full_tree[8:114, 18:82] = True
    competing_crop = np.zeros(shape, dtype=bool)
    competing_crop[14:108, 22:78] = True
    candidates = [
        _candidate(
            "natural_object",
            "natural_object",
            full_tree,
            0.48,
            "tree-model",
            1,
            root=True,
            domain_evidence=0.62,
            root_label_specificity=1.0,
            part_profile_specificity=1.0,
            selected_part_profile="tree_or_log",
            profile_hint_source="specific_root_label",
        ),
        _candidate(
            "natural_object",
            "natural_object",
            competing_crop,
            0.69,
            "crystal-model",
            2,
            root=True,
            domain_evidence=0.68,
            root_label_specificity=1.0,
            part_profile_specificity=1.0,
            selected_part_profile="crystal",
            profile_hint_source="specific_root_label",
        ),
    ]

    result = route_asset_roots(
        candidates,
        image_shape=shape,
        config=RootRoutingConfig(include_attached_roots=False),
    )

    selected = next(iter(result.candidates))
    assert np.array_equal(selected.mask, full_tree)
    assert selected.metadata["selected_part_profile"] == "tree_or_log"
    assert selected.metadata["profile_hint_source"] == "specific_root_label"
