import numpy as np

from hpid_split.instances import PartInstance
from hpid_split.quality import assess_product_quality


def _record(
    index: int,
    semantic_name: str,
    *,
    semantic_parent: str = "asset",
    asset_id: str = "object_001",
    area_px: int = 100,
) -> PartInstance:
    return PartInstance(
        part_id=f"{semantic_parent}/{semantic_name}/center/{index:02d}",
        semantic_name=semantic_name,
        semantic_parent=semantic_parent,
        instance_index=index,
        side="center",
        bbox_xyxy=(2, 2, 12, 12),
        centroid_xy=(7.0, 7.0),
        area_px=area_px,
        asset_id=asset_id,
    )


def _ambiguous_diagnostics(*, point_requested: bool) -> dict[str, object]:
    return {
        "root_routing": {
            "mode": "primary",
            "selected_physical_group_id": "physical:01",
            "salience_tie_candidate_count": 2,
            "prompt_owned_group_count": 2,
            "target_point_routing": {"requested": point_requested},
            "prompt_fragment_consensus": {
                "selected_physical_group_id": None,
            },
            "root_scores": [
                {
                    "root_key": "root:1",
                    "semantic_name": "asset",
                    "bbox_xyxy": [1, 2, 20, 30],
                },
                {
                    "root_key": "root:2",
                    "semantic_name": "asset",
                    "bbox_xyxy": [22, 2, 42, 30],
                },
            ],
            "physical_groups": [
                {
                    "physical_group_id": "physical:01",
                    "selected_root_key": "root:1",
                    "group_score": 0.72,
                },
                {
                    "physical_group_id": "physical:02",
                    "selected_root_key": "root:2",
                    "group_score": 0.71,
                },
            ],
        }
    }


def test_quality_report_requires_target_selection_for_close_primary_roots() -> None:
    report = assess_product_quality(
        np.ones((32, 48), dtype=np.uint16),
        [_record(1, "asset", semantic_parent="asset")],
        _ambiguous_diagnostics(point_requested=False),
    )

    assert report["status"] == "target_selection_required"
    assert report["evidence_grade"] == "C"
    assert report["review_reasons"] == ["ambiguous_primary_asset"]
    target = report["target_selection"]
    assert target["ambiguous"] is True
    assert len(target["candidates"]) == 2
    assert target["candidates"][0]["selected"] is True
    assert report["ground_truth_used"] is False


def test_explicit_target_point_resolves_root_ambiguity_audit() -> None:
    report = assess_product_quality(
        np.ones((32, 48), dtype=np.uint16),
        [_record(1, "asset", semantic_parent="asset")],
        _ambiguous_diagnostics(point_requested=True),
    )

    assert report["status"] == "ready"
    assert report["evidence_grade"] == "A"
    assert report["target_selection"]["ambiguous"] is False


def test_initial_cross_category_choice_survives_later_route_pruning() -> None:
    diagnostics = {
        "initial_root_routing": {
            "mode": "primary",
            "selected_physical_group_id": "physical:01",
            "root_scores": [
                {
                    "root_key": "root:1",
                    "semantic_name": "device",
                    "bbox_xyxy": [2, 2, 24, 28],
                },
                {
                    "root_key": "root:2",
                    "semantic_name": "character",
                    "bbox_xyxy": [26, 1, 47, 31],
                },
            ],
            "physical_groups": [
                {
                    "physical_group_id": "physical:01",
                    "selected_root_key": "root:1",
                    "selected_semantic": "device",
                    "group_score": 0.83,
                },
                {
                    "physical_group_id": "physical:02",
                    "selected_root_key": "root:2",
                    "selected_semantic": "character",
                    "group_score": 0.68,
                },
            ],
        },
        "root_routing": {
            "mode": "primary",
            "selected_physical_group_id": "physical:01",
            "physical_groups": [
                {
                    "physical_group_id": "physical:01",
                    "selected_root_key": "root:1",
                    "selected_semantic": "device",
                    "group_score": 0.83,
                }
            ],
        },
    }

    report = assess_product_quality(
        np.ones((32, 48), dtype=np.uint16),
        [_record(1, "device", semantic_parent="device")],
        diagnostics,
    )

    assert report["status"] == "target_selection_required"
    assert report["target_selection"]["audit_stage"] == "initial_root_routing"


def test_dominant_root_residual_requires_review_for_confirmed_profile() -> None:
    records = [
        _record(1, "tool_prop", semantic_parent="tool_prop", area_px=3000),
        _record(2, "tool_prop_head", semantic_parent="tool_prop", area_px=300),
        _record(3, "tool_prop_handle", semantic_parent="tool_prop", area_px=300),
        _record(4, "tool_prop_grip", semantic_parent="tool_prop", area_px=300),
    ]
    diagnostics = {
        "root_routing": {"mode": "primary", "physical_groups": []},
        "profile_root_resolution": {"selected_profiles": ["hammer"]},
    }

    report = assess_product_quality(
        np.ones((64, 64), dtype=np.uint16), records, diagnostics
    )

    assert report["status"] == "review_recommended"
    assert "dominant_unassigned_root_residual" in report["review_reasons"]
    assert report["part_structure"]["dominant_root_residual"] is True


def test_many_generic_regions_are_exported_with_review_status() -> None:
    records = [_record(1, "asset", semantic_parent="asset")]
    records.extend(
        _record(index, f"asset_visual_detail_{index:02d}")
        for index in range(2, 9)
    )

    report = assess_product_quality(
        np.ones((64, 64), dtype=np.uint16),
        records,
        {"root_routing": {"mode": "primary", "physical_groups": []}},
    )

    assert report["status"] == "review_recommended"
    assert report["evidence_grade"] == "B"
    assert "many_generic_part_names" in report["review_reasons"]
    assert "add_part_prompts_for_semantic_ids" in report["recommended_actions"]


def test_large_generic_area_is_not_reported_as_grade_a() -> None:
    records = [
        _record(1, "asset", semantic_parent="asset", area_px=2500),
        _record(2, "asset_visual_region", area_px=1200),
        _record(3, "asset_panel", area_px=1300),
    ]

    report = assess_product_quality(
        np.ones((64, 64), dtype=np.uint16),
        records,
        {"root_routing": {"mode": "primary", "physical_groups": []}},
    )

    assert report["status"] == "review_recommended"
    assert "large_unresolved_generic_area" in report["review_reasons"]
    assert report["part_structure"]["generic_area_ratio"] > 0.20


def test_severe_repeated_semantic_area_imbalance_requires_review() -> None:
    records = [
        _record(1, "character", semantic_parent="character", area_px=3000),
        _record(2, "character_eye", area_px=23),
        _record(3, "character_eye", area_px=850),
    ]
    records.extend(
        _record(index, f"character_part_{index}", area_px=120)
        for index in range(4, 13)
    )

    report = assess_product_quality(
        np.ones((512, 512), dtype=np.uint16),
        records,
        {"root_routing": {"mode": "primary", "physical_groups": []}},
    )

    assert report["status"] == "review_recommended"
    assert "severe_repeated_part_area_imbalance" in report["review_reasons"]
    rows = report["part_structure"]["repeated_part_area_imbalances"]
    assert rows[0]["semantic_name"] == "character_eye"


def test_cross_asset_parent_is_never_reported_as_ready() -> None:
    parent = _record(1, "asset", semantic_parent="asset", asset_id="object_001")
    child = PartInstance(
        part_id="asset/asset_panel/center/01",
        semantic_name="asset_panel",
        semantic_parent="asset",
        instance_index=2,
        side="center",
        bbox_xyxy=(14, 2, 24, 12),
        centroid_xy=(19.0, 7.0),
        area_px=100,
        asset_id="object_002",
        assembly_parent_id=parent.part_id,
    )

    report = assess_product_quality(
        np.ones((32, 48), dtype=np.uint16),
        [parent, child],
        {"root_routing": {"mode": "scene", "physical_groups": []}},
    )

    assert report["status"] == "invalid_hierarchy"
    assert report["evidence_grade"] == "C"
    assert "cross_asset_assembly_parent" in report["review_reasons"]


def test_unresolved_cross_domain_router_disagreement_requires_review() -> None:
    report = assess_product_quality(
        np.ones((32, 48), dtype=np.uint16),
        [_record(1, "character", semantic_parent="character")],
        {
            "root_routing": {"mode": "primary", "physical_groups": []},
            "asset_domain_routing": {
                "rows": [
                    {
                        "root_key": "root:1",
                        "original_domain": "character",
                        "domain_accepted": False,
                        "asset_route": {
                            "accepted": False,
                            "reason": "ambiguous_candidate_set",
                            "candidate_domains": ["daily_object", "container"]
                        },
                    }
                ]
            },
        },
    )

    assert report["status"] == "review_recommended"
    assert report["evidence_grade"] == "B"
    assert "asset_domain_uncertain" in report["review_reasons"]


def test_confirmed_vlm_root_audit_clears_router_uncertainty() -> None:
    report = assess_product_quality(
        np.ones((32, 48), dtype=np.uint16),
        [_record(1, "daily_object", semantic_parent="daily_object")],
        {
            "root_routing": {"mode": "primary", "physical_groups": []},
            "asset_domain_routing": {
                "rows": [
                    {
                        "root_key": "root:1",
                        "original_domain": "container",
                        "domain_accepted": True,
                        "asset_route": {
                            "accepted": False,
                            "reason": "ambiguous_candidate_set",
                            "candidate_domains": ["container"],
                        },
                    }
                ]
            },
            "vlm_root_audit": {
                "rows": [{"root_key": "root:1", "status": "corrected"}]
            },
        },
    )

    assert report["status"] == "ready"
    assert "asset_domain_uncertain" not in report["review_reasons"]


def test_independent_profile_consensus_clears_exact_label_uncertainty() -> None:
    report = assess_product_quality(
        np.ones((32, 48), dtype=np.uint16),
        [
            _record(1, "tool_prop", semantic_parent="tool_prop"),
            _record(2, "tool_prop_receiver", semantic_parent="tool_prop"),
        ],
        {
            "root_routing": {"mode": "primary", "physical_groups": []},
            "asset_domain_routing": {
                "rows": [
                    {
                        "root_key": "root:1",
                        "original_domain": "tool_prop",
                        "domain_accepted": True,
                        "asset_route": {
                            "accepted": False,
                            "reason": "ambiguous_candidate_set",
                            "candidate_domains": ["tool_prop"],
                            "alternatives": [
                                {
                                    "asset_domain": "tool_prop",
                                    "asset_profile": "firearm",
                                }
                            ],
                        },
                    }
                ]
            },
            "profile_root_resolution": {
                "profile_consensus": {
                    "roots": [
                        {
                            "root_key": "profile-root:1",
                            "status": "accepted",
                            "selected_profile": "firearm",
                        }
                    ]
                }
            },
        },
    )

    assert report["status"] == "ready"
    assert "asset_domain_uncertain" not in report["review_reasons"]


def test_mismatched_profile_does_not_clear_ambiguous_asset_route() -> None:
    report = assess_product_quality(
        np.ones((32, 48), dtype=np.uint16),
        [_record(1, "container", semantic_parent="container")],
        {
            "root_routing": {"mode": "primary", "physical_groups": []},
            "asset_domain_routing": {
                "rows": [
                    {
                        "root_key": "root:1",
                        "original_domain": "container",
                        "domain_accepted": True,
                        "routing_applicable": True,
                        "asset_route": {
                            "accepted": False,
                            "reason": "ambiguous_candidate_set",
                            "candidate_domains": ["container", "device"],
                            "alternatives": [
                                {
                                    "asset_domain": "device",
                                    "asset_profile": "kettle",
                                }
                            ],
                        },
                    }
                ]
            },
            "profile_root_resolution": {
                "profile_consensus": {
                    "roots": [
                        {
                            "status": "accepted",
                            "selected_profile": "game_loot_container",
                        }
                    ]
                }
            },
        },
    )

    assert report["status"] == "review_recommended"
    assert "asset_domain_uncertain" in report["review_reasons"]


def test_low_evidence_out_of_inventory_domain_requires_review() -> None:
    diagnostics = {
        "root_routing": {
            "mode": "primary",
            "physical_groups": [],
            "root_scores": [
                {"root_key": "root:1", "domain_evidence_score": 0.38}
            ],
        },
        "asset_domain_routing": {
            "rows": [
                {
                    "root_key": "root:1",
                    "original_domain": "character",
                    "routing_applicable": False,
                    "domain_accepted": False,
                    "asset_route": {
                        "accepted": False,
                        "reason": "ambiguous_candidate_set",
                        "candidate_domains": ["daily_object", "container"],
                    },
                }
            ]
        },
    }

    report = assess_product_quality(
        np.ones((32, 48), dtype=np.uint16),
        [_record(1, "character", semantic_parent="character")],
        diagnostics,
    )

    assert report["status"] == "review_recommended"
    assert "asset_domain_uncertain" in report["review_reasons"]


def test_strong_out_of_inventory_domain_evidence_clears_router_noise() -> None:
    diagnostics = {
        "root_routing": {
            "mode": "primary",
            "physical_groups": [],
            "root_scores": [
                {"root_key": "root:1", "domain_evidence_score": 0.67}
            ],
        },
        "asset_domain_routing": {
            "rows": [
                {
                    "root_key": "root:1",
                    "original_domain": "character",
                    "routing_applicable": False,
                    "domain_accepted": False,
                    "asset_route": {
                        "accepted": False,
                        "reason": "ambiguous_candidate_set",
                        "candidate_domains": ["daily_object", "device"],
                    },
                }
            ]
        },
    }

    report = assess_product_quality(
        np.ones((32, 48), dtype=np.uint16),
        [_record(1, "character", semantic_parent="character")],
        diagnostics,
    )

    assert report["status"] == "ready"
    assert "asset_domain_uncertain" not in report["review_reasons"]


def _cross_view_diagnostics(
    *,
    global_accepted: bool,
    crop_accepted: bool,
    semantic_probability: float,
    frame_extent: float,
    touched_sides: int,
) -> dict[str, object]:
    return {
        "global_asset_proposal": {
            "route": {"accepted": global_accepted},
        },
        "root_routing": {
            "mode": "primary",
            "physical_groups": [],
            "root_scores": [
                {
                    "root_key": "root:1",
                    "selected": True,
                    "semantic_mask_probability": semantic_probability,
                    "frame_extent": frame_extent,
                    "touched_sides": touched_sides,
                }
            ],
        },
        "asset_domain_routing": {
            "rows": [
                {
                    "root_key": "root:1",
                    "original_domain": "daily_object",
                    "domain_accepted": True,
                    "asset_route": {
                        "accepted": True,
                        "asset_label": "pillow",
                        "candidate_domains": ["daily_object"],
                        "reason": "accepted_cross_view_asset_consensus",
                    },
                    "root_crop_asset_route": {
                        "accepted": crop_accepted,
                    },
                    "cross_view_consensus": {
                        "accepted": True,
                        "status": "accepted_top_label_agreement",
                    },
                }
            ]
        },
    }


def test_two_ambiguous_views_do_not_create_grade_a_evidence() -> None:
    report = assess_product_quality(
        np.ones((32, 48), dtype=np.uint16),
        [_record(1, "daily_object", semantic_parent="daily_object")],
        _cross_view_diagnostics(
            global_accepted=False,
            crop_accepted=False,
            semantic_probability=0.55,
            frame_extent=0.60,
            touched_sides=0,
        ),
    )

    assert report["status"] == "review_recommended"
    assert report["evidence_grade"] == "B"
    assert "weak_cross_view_asset_consensus" in report["review_reasons"]


def test_independently_accepted_view_supports_cross_view_consensus() -> None:
    report = assess_product_quality(
        np.ones((32, 48), dtype=np.uint16),
        [_record(1, "daily_object", semantic_parent="daily_object")],
        _cross_view_diagnostics(
            global_accepted=True,
            crop_accepted=False,
            semantic_probability=0.55,
            frame_extent=0.60,
            touched_sides=0,
        ),
    )

    assert report["status"] == "ready"
    assert "weak_cross_view_asset_consensus" not in report["review_reasons"]


def test_frame_filling_low_evidence_singleton_requires_target_selection() -> None:
    report = assess_product_quality(
        np.ones((32, 48), dtype=np.uint16),
        [_record(1, "daily_object", semantic_parent="daily_object")],
        _cross_view_diagnostics(
            global_accepted=False,
            crop_accepted=False,
            semantic_probability=0.14,
            frame_extent=1.0,
            touched_sides=4,
        ),
    )

    assert report["status"] == "target_selection_required"
    assert report["evidence_grade"] == "C"
    assert "coherent_wrong_target_risk" in report["review_reasons"]
    assert report["target_selection"]["ambiguous"] is True


def test_heterogeneous_high_scoring_roots_require_target_selection() -> None:
    diagnostics = _ambiguous_diagnostics(point_requested=False)
    root_routing = diagnostics["root_routing"]
    root_routing["salience_tie_candidate_count"] = 1
    root_routing["prompt_owned_group_count"] = 0
    root_routing["physical_groups"][0]["selected_semantic"] = "character"
    root_routing["physical_groups"][0]["group_score"] = 0.76
    root_routing["physical_groups"][1]["selected_semantic"] = "vehicle"
    root_routing["physical_groups"][1]["group_score"] = 0.63

    report = assess_product_quality(
        np.ones((64, 64), dtype=np.uint16),
        [_record(1, "character", semantic_parent="character")],
        diagnostics,
    )

    assert report["status"] == "target_selection_required"
    assert report["target_selection"]["heterogeneous_target_ambiguity"] is True
    assert "ambiguous_primary_asset" in report["review_reasons"]


def test_close_same_category_physical_roots_still_require_selection() -> None:
    diagnostics = _ambiguous_diagnostics(point_requested=False)
    root_routing = diagnostics["root_routing"]
    root_routing["salience_tie_candidate_count"] = 1
    root_routing["prompt_owned_group_count"] = 0
    root_routing["physical_groups"][0]["selected_semantic"] = "character"
    root_routing["physical_groups"][0]["group_score"] = 0.69
    root_routing["physical_groups"][1]["selected_semantic"] = "character"
    root_routing["physical_groups"][1]["group_score"] = 0.67

    report = assess_product_quality(
        np.ones((64, 64), dtype=np.uint16),
        [_record(1, "character", semantic_parent="character")],
        diagnostics,
    )

    assert report["status"] == "target_selection_required"
    assert report["target_selection"]["close_physical_target_ambiguity"] is True


def test_exact_profile_route_without_children_is_not_ready() -> None:
    diagnostics = {
        "root_routing": {"mode": "primary", "physical_groups": []},
        "global_asset_proposal": {
            "route": {
                "accepted": True,
                "reason": "accepted_exact_label",
                "asset_profile": "controls",
            }
        },
    }

    report = assess_product_quality(
        np.ones((64, 64), dtype=np.uint16),
        [_record(1, "device", semantic_parent="device")],
        diagnostics,
    )

    assert report["status"] == "review_recommended"
    assert "no_part_decomposition" in report["review_reasons"]
    assert report["part_structure"]["decomposition_missing"] is True


def test_large_scene_part_explosion_is_never_ready() -> None:
    records = [
        _record(
            index,
            f"scene_part_{index}",
            asset_id=f"object_{((index - 1) // 5) + 1:03d}",
            area_px=120,
        )
        for index in range(1, 131)
    ]

    report = assess_product_quality(
        np.ones((512, 512), dtype=np.uint16),
        records,
        {"root_routing": {"mode": "scene", "physical_groups": []}},
    )

    assert report["status"] == "review_recommended"
    assert "unusually_many_scene_parts" in report["review_reasons"]
