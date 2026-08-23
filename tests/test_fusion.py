import inspect

import numpy as np

from hpid_split.fusion import (
    FusionConfig,
    MaskCandidate,
    _hierarchical_part_ids,
    _is_identity_visibility_sliver,
    fuse_candidates,
    suppress_correlated_semantic_hypotheses,
    taxonomy_from_candidates,
)


def _mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    output = np.zeros(shape, dtype=bool)
    x0, y0, x1, y1 = box
    output[y0:y1, x0:x1] = True
    return output


def _conditional_candidate(
    semantic_name: str,
    semantic_parent: str,
    mask: np.ndarray,
    score: float,
    *,
    geometry: float,
    confirmed: bool,
    corroboration: float = 0.0,
) -> MaskCandidate:
    return MaskCandidate(
        semantic_name,
        semantic_parent,
        mask,
        score,
        "conditional-part/model",
        source_reliability=0.86,
        metadata={
            "root_origin": "root-model",
            "root_index": 0,
            "retrieval_geometry_compatibility": geometry,
            "cross_source_confirmed": confirmed,
            "cross_source_best_iou": corroboration,
        },
    )


def test_correlated_semantic_suppression_keeps_stronger_physical_label() -> None:
    shape = (64, 64)
    broad = _mask(shape, (12, 18, 48, 42))
    alternate = _mask(shape, (14, 18, 49, 43))
    candidates = [
        _conditional_candidate(
            "vehicle_head_tube",
            "vehicle_frame",
            broad,
            0.65,
            geometry=0.38,
            confirmed=True,
            corroboration=0.69,
        ),
        _conditional_candidate(
            "vehicle_handlebar",
            "vehicle_frame",
            alternate,
            0.80,
            geometry=0.62,
            confirmed=True,
            corroboration=0.42,
        ),
    ]

    kept, rows = suppress_correlated_semantic_hypotheses(candidates)

    assert [candidate.semantic_name for candidate in kept] == ["vehicle_handlebar"]
    assert rows[0]["dropped_semantic"] == "vehicle_head_tube"


def test_correlated_semantic_suppression_preserves_true_nested_part() -> None:
    shape = (80, 80)
    housing = _mask(shape, (8, 8, 72, 72))
    screen = _mask(shape, (24, 24, 56, 50))
    candidates = [
        _conditional_candidate(
            "device_housing",
            "device_body",
            housing,
            0.82,
            geometry=0.80,
            confirmed=True,
            corroboration=0.55,
        ),
        _conditional_candidate(
            "device_screen",
            "device_housing",
            screen,
            0.78,
            geometry=0.84,
            confirmed=True,
            corroboration=0.48,
        ),
    ]

    kept, rows = suppress_correlated_semantic_hypotheses(candidates)

    assert {candidate.semantic_name for candidate in kept} == {
        "device_housing",
        "device_screen",
    }
    assert rows == ()


def test_correlated_semantic_suppression_rejects_unlocalized_child() -> None:
    shape = (72, 72)
    parent = _mask(shape, (12, 18, 58, 56))
    child = _mask(shape, (10, 16, 60, 58))
    candidates = [
        _conditional_candidate(
            "tool_prop_brush",
            "tool_prop_shaft",
            parent,
            0.91,
            geometry=0.81,
            confirmed=True,
            corroboration=0.76,
        ),
        _conditional_candidate(
            "tool_prop_lower_bristles",
            "tool_prop_brush",
            child,
            0.74,
            geometry=0.63,
            confirmed=True,
            corroboration=0.66,
        ),
    ]

    kept, rows = suppress_correlated_semantic_hypotheses(candidates)

    assert [candidate.semantic_name for candidate in kept] == ["tool_prop_brush"]
    assert rows[0]["dropped_semantic"] == "tool_prop_lower_bristles"


def test_fusion_has_no_ground_truth_argument() -> None:
    parameters = inspect.signature(fuse_candidates).parameters
    assert "target" not in parameters
    assert "truth" not in parameters
    assert "ground_truth" not in parameters


def test_release_config_disables_negative_boundary_ablation() -> None:
    assert FusionConfig().use_boundary_ownership is False


def test_tiny_visible_remainder_of_large_identity_is_a_conflict_sliver() -> None:
    config = FusionConfig()

    assert _is_identity_visibility_sliver(41, 10_900, 62_370, 20, config)
    assert not _is_identity_visibility_sliver(239, 239, 62_370, 20, config)
    assert not _is_identity_visibility_sliver(400, 10_900, 62_370, 20, config)


def test_part_decomposition_conserves_the_accepted_root_union() -> None:
    shape = (80, 100)
    root = _mask(shape, (5, 4, 95, 76))
    broad_child = _mask(shape, (8, 10, 82, 68))
    detail = _mask(shape, (52, 24, 66, 38))
    result = fuse_candidates(
        [
            MaskCandidate("asset", "asset", root, 0.48, "root"),
            MaskCandidate("asset_panel", "asset", broad_child, 0.82, "parts"),
            MaskCandidate("asset_button", "asset", detail, 0.88, "details"),
        ],
        image_shape=shape,
    )

    assert np.all(result.semantic_map[root] > 0)
    assert result.diagnostics["lost_root_pixels_after_conservation"] == 0
    assert result.diagnostics["root_union_area_px"] == int(np.count_nonzero(root))


def test_specific_part_owns_pixels_inside_a_broad_host_panel() -> None:
    shape = (100, 100)
    root = _mask(shape, (5, 5, 95, 95))
    base_panel = _mask(shape, (10, 20, 90, 90))
    keyboard = _mask(shape, (25, 35, 75, 65))
    candidates = [
        MaskCandidate("device", "device", root, 0.98, "root"),
        MaskCandidate(
            "device_base_panel",
            "device_body",
            base_panel,
            0.95,
            "visual",
        ),
        MaskCandidate(
            "device_keyboard",
            "device_body",
            keyboard,
            0.32,
            "detector",
        ),
    ]
    baseline = fuse_candidates(
        candidates,
        image_shape=shape,
        config=FusionConfig(use_specificity_ownership=False),
    )
    specific = fuse_candidates(candidates, image_shape=shape)
    baseline_base = baseline.taxonomy.fine_names.index("device_base_panel")
    specific_keyboard = specific.taxonomy.fine_names.index("device_keyboard")

    assert baseline.semantic_map[50, 50] == baseline_base
    assert specific.semantic_map[50, 50] == specific_keyboard
    assert specific.diagnostics["specificity_host_suppressions"] >= 1


def test_nested_specific_part_owns_pixels_inside_root_envelope() -> None:
    shape = (100, 100)
    root = _mask(shape, (5, 5, 95, 95))
    cover = _mask(shape, (15, 15, 85, 85))
    result = fuse_candidates(
        [
            MaskCandidate(
                "asset",
                "asset",
                root,
                0.98,
                "root",
                metadata={"root_origin": "test", "root_index": 1},
            ),
            MaskCandidate(
                "asset_cover",
                "asset_body",
                cover,
                0.72,
                "parts",
                metadata={"root_origin": "test", "root_index": 1},
            ),
        ],
        image_shape=shape,
    )
    cover_id = result.taxonomy.fine_names.index("asset_cover")

    assert result.semantic_map[50, 50] == cover_id
    assert result.diagnostics["specificity_host_suppressions"] >= 1


def test_weak_nested_part_does_not_force_root_to_yield() -> None:
    shape = (100, 100)
    root = _mask(shape, (5, 5, 95, 95))
    weak_part = _mask(shape, (15, 15, 85, 85))
    result = fuse_candidates(
        [
            MaskCandidate(
                "asset",
                "asset",
                root,
                0.98,
                "root",
                metadata={"root_origin": "test", "root_index": 1},
            ),
            MaskCandidate(
                "asset_handle",
                "asset_body",
                weak_part,
                0.24,
                "parts",
                metadata={"root_origin": "test", "root_index": 1},
            ),
        ],
        image_shape=shape,
    )
    root_id = result.taxonomy.fine_names.index("asset")

    assert result.semantic_map[50, 50] == root_id
    assert result.diagnostics["specificity_host_suppressions"] == 0


def test_same_source_visual_relabel_does_not_force_root_to_yield() -> None:
    shape = (100, 100)
    root = _mask(shape, (5, 5, 95, 95))
    visual_panel = _mask(shape, (15, 15, 85, 85))
    result = fuse_candidates(
        [
            MaskCandidate(
                "asset",
                "asset",
                root,
                0.98,
                "root",
                metadata={"root_origin": "test", "root_index": 1},
            ),
            MaskCandidate(
                "asset_lid",
                "asset_body",
                visual_panel,
                0.90,
                "sam-visual",
                source_reliability=0.40,
                metadata={
                    "root_origin": "test",
                    "root_index": 1,
                    "visual_region": True,
                    "generic_visual_region": False,
                },
            ),
        ],
        image_shape=shape,
    )
    root_id = result.taxonomy.fine_names.index("asset")

    assert result.semantic_map[50, 50] == root_id
    assert result.diagnostics["specificity_host_suppressions"] == 0


def test_vetted_structural_partition_can_reassign_root_ownership() -> None:
    shape = (80, 120)
    root = _mask(shape, (5, 20, 115, 60))
    structural_part = _mask(shape, (8, 24, 100, 56))
    result = fuse_candidates(
        [
            MaskCandidate(
                "tool",
                "tool",
                root,
                0.98,
                "root",
                metadata={"root_origin": "test", "root_index": 1},
            ),
            MaskCandidate(
                "tool_handle",
                "tool_body",
                structural_part,
                0.84,
                "structural",
                source_reliability=0.74,
                metadata={
                    "root_origin": "test",
                    "root_index": 1,
                    "structural_fusion": True,
                    "structural_root_evidence": True,
                },
            ),
        ],
        image_shape=shape,
    )
    handle_id = result.taxonomy.fine_names.index("tool_handle")

    assert result.semantic_map[40, 50] == handle_id
    assert result.diagnostics["specificity_host_suppressions"] >= 1


def test_stronger_child_suppresses_near_identical_parent_proposal() -> None:
    shape = (80, 80)
    root = _mask(shape, (5, 5, 75, 75))
    ring = _mask(shape, (20, 20, 60, 60))
    result = fuse_candidates(
        [
            MaskCandidate(
                "vehicle",
                "vehicle",
                root,
                0.95,
                "root",
                metadata={"root_origin": "test", "root_index": 1},
            ),
            MaskCandidate(
                "vehicle_wheel",
                "vehicle_body",
                ring,
                0.55,
                "detector",
                source_reliability=0.80,
                metadata={"root_origin": "test", "root_index": 1},
            ),
            MaskCandidate(
                "vehicle_tire",
                "vehicle_wheel",
                ring,
                0.72,
                "detector",
                source_reliability=0.82,
                metadata={"root_origin": "test", "root_index": 1},
            ),
        ],
        image_shape=shape,
    )

    accepted_names = [candidate.semantic_name for candidate in result.accepted_candidates]
    assert "vehicle_wheel" not in accepted_names
    assert "vehicle_tire" in accepted_names
    assert result.diagnostics["hierarchical_duplicates_suppressed"] == 1


def test_root_coverage_conservation_can_be_disabled_for_ablation() -> None:
    shape = (48, 48)
    root = _mask(shape, (5, 5, 43, 43))
    result = fuse_candidates(
        [MaskCandidate("asset", "asset", root, 0.05, "root", source_reliability=0.1)],
        image_shape=shape,
        config=FusionConfig(use_root_coverage_conservation=False),
    )

    assert result.diagnostics["ablation"]["root_coverage_conservation"] is False


def test_cleanup_removes_tiny_parent_fallback_islands_after_child_reassignment() -> None:
    shape = (512, 512)
    root = _mask(shape, (40, 40, 472, 472))
    body = _mask(shape, (80, 80, 432, 432))
    noise = _mask(shape, (10, 10, 13, 13))
    result = fuse_candidates(
        [
            MaskCandidate("asset", "asset", root, 0.98, "root"),
            MaskCandidate("asset_body", "asset", body | noise, 0.85, "parts"),
        ],
        image_shape=shape,
    )

    assert result.semantic_map[11, 11] == 0


def test_scale_adaptive_cleanup_keeps_supported_detail_component() -> None:
    shape = (512, 512)
    root = _mask(shape, (40, 40, 472, 472))
    detail = _mask(shape, (120, 120, 125, 125))
    result = fuse_candidates(
        [
            MaskCandidate("asset", "asset", root, 0.98, "root"),
            MaskCandidate("asset_button", "asset", detail, 0.86, "details"),
        ],
        image_shape=shape,
    )
    detail_id = result.taxonomy.fine_names.index("asset_button")

    assert result.semantic_map[122, 122] == detail_id


def test_hierarchy_keeps_supported_detail_and_rejects_orphan() -> None:
    shape = (64, 64)
    candidates = [
        MaskCandidate("object", "object", _mask(shape, (8, 8, 56, 56)), 0.95, "root"),
        MaskCandidate(
            "object_handle",
            "object",
            _mask(shape, (18, 24, 34, 38)),
            0.76,
            "parts",
        ),
        MaskCandidate(
            "object_handle",
            "object",
            _mask(shape, (0, 0, 6, 6)),
            0.92,
            "parts",
        ),
    ]
    result = fuse_candidates(candidates, image_shape=shape)
    handle_id = result.taxonomy.fine_names.index("object_handle")
    assert result.semantic_map[30, 24] == handle_id
    assert result.semantic_map[2, 2] == 0
    assert result.diagnostics["ground_truth_used"] is False


def test_hard_parent_support_preserves_inside_evidence_and_penalizes_orphan() -> None:
    shape = (64, 64)
    root = _mask(shape, (12, 12, 52, 52))
    child = _mask(shape, (20, 20, 34, 34)) | _mask(shape, (0, 0, 8, 8))
    candidates = [
        MaskCandidate("asset", "asset", root, 0.95, "root"),
        MaskCandidate("asset_handle", "asset", child, 0.82, "parts"),
    ]
    common = {
        "use_consensus": False,
        "use_parent_residual": False,
        "use_direct_gate": False,
        "use_boundary_ownership": False,
        "detail_bonus": 0.0,
    }
    unconstrained = fuse_candidates(
        candidates,
        image_shape=shape,
        config=FusionConfig(use_parent_support=False, **common),
    )
    constrained = fuse_candidates(
        candidates,
        image_shape=shape,
        config=FusionConfig(use_parent_support=True, **common),
    )
    child_id = constrained.taxonomy.fine_names.index("asset_handle")
    assert (
        constrained.evidence[child_id, 26, 26]
        == unconstrained.evidence[child_id, 26, 26]
    )
    assert constrained.evidence[child_id, 3, 3] < unconstrained.evidence[child_id, 3, 3]


def test_explicit_root_envelope_supports_an_unmaterialized_semantic_parent() -> None:
    shape = (64, 64)
    root = _mask(shape, (8, 8, 56, 56))
    trigger = _mask(shape, (28, 36, 34, 43))
    candidates = [
        MaskCandidate(
            "tool_prop",
            "tool_prop",
            root,
            0.96,
            "test/root",
            metadata={
                "root_origin": "test",
                "root_index": 1,
                "candidate_key": "root:1",
                "parent_candidate_key": None,
            },
        ),
        MaskCandidate(
            "tool_prop_trigger",
            "tool_prop_body",
            trigger,
            0.55,
            "sam2-amg/test/semantic-rerank",
            source_reliability=0.66,
            metadata={
                "root_origin": "test",
                "root_index": 1,
                "candidate_key": "root:1/trigger",
                "parent_candidate_key": "root:1",
                "visual_region": True,
                "generic_visual_region": False,
                "semantic_reranked": True,
                "semantic_rerank_route": "semantic_inventory_evidence_rescue",
                "detail": True,
            },
        ),
    ]

    result = fuse_candidates(candidates, image_shape=shape)
    trigger_id = result.taxonomy.fine_names.index("tool_prop_trigger")

    assert np.count_nonzero(result.semantic_map == trigger_id) > 0


def test_parent_envelope_supports_child_inside_visible_parent_hole() -> None:
    shape = (64, 64)
    root = _mask(shape, (4, 4, 60, 60))
    head = _mask(shape, (12, 12, 52, 52))
    head[24:40, 24:40] = False
    eye = _mask(shape, (27, 28, 37, 36))
    candidates = [
        MaskCandidate("asset", "asset", root, 0.98, "root"),
        MaskCandidate("asset_head", "asset", head, 0.90, "parts"),
        MaskCandidate(
            "asset_eye",
            "asset_head",
            eye,
            0.86,
            "dense",
            metadata={"dense_semantic_fallback": True},
        ),
    ]
    visible_only = fuse_candidates(
        candidates,
        image_shape=shape,
        config=FusionConfig(use_parent_envelope=False),
    )
    envelope = fuse_candidates(
        candidates,
        image_shape=shape,
        config=FusionConfig(use_parent_envelope=True),
    )

    eye_id = envelope.taxonomy.fine_names.index("asset_eye")
    assert envelope.evidence[eye_id, 31, 31] > visible_only.evidence[eye_id, 31, 31]
    assert envelope.semantic_map[31, 31] == eye_id
    assert envelope.diagnostics["ablation"]["parent_envelope"] is True


def test_transitive_residual_allows_grandchild_to_replace_root_fallback() -> None:
    shape = (64, 64)
    root = _mask(shape, (4, 4, 60, 60))
    head = _mask(shape, (12, 12, 52, 52))
    head[22:42, 22:42] = False
    eye = _mask(shape, (26, 27, 38, 36))
    candidates = [
        MaskCandidate("asset", "asset", root, 0.99, "root"),
        MaskCandidate("asset_head", "asset", head, 0.86, "parts"),
        MaskCandidate(
            "asset_eye",
            "asset_head",
            eye,
            0.72,
            "dense",
            metadata={"dense_semantic_fallback": True},
        ),
    ]
    shallow = fuse_candidates(
        candidates,
        image_shape=shape,
        config=FusionConfig(
            use_parent_envelope=True,
            parent_dilation_ratio=0.06,
            use_transitive_residual=False,
            use_direct_gate=False,
            detail_bonus=0.0,
        ),
    )
    transitive = fuse_candidates(
        candidates,
        image_shape=shape,
        config=FusionConfig(
            use_parent_envelope=True,
            parent_dilation_ratio=0.06,
            use_transitive_residual=True,
            use_direct_gate=False,
            detail_bonus=0.0,
        ),
    )

    eye_id = transitive.taxonomy.fine_names.index("asset_eye")
    shallow_pixels = np.count_nonzero(shallow.semantic_map == eye_id)
    transitive_pixels = np.count_nonzero(transitive.semantic_map == eye_id)
    assert transitive_pixels > shallow_pixels
    assert transitive.semantic_map[31, 31] == eye_id
    assert transitive.diagnostics["ablation"]["transitive_residual"] is True


def test_dense_hierarchy_relaxation_does_not_change_standard_candidates() -> None:
    shape = (48, 48)
    root = _mask(shape, (3, 3, 45, 45))
    parent = _mask(shape, (10, 8, 38, 34))
    child = _mask(shape, (18, 16, 30, 24))
    candidates = [
        MaskCandidate("asset", "asset", root, 0.97, "root"),
        MaskCandidate("asset_parent", "asset", parent, 0.84, "parts"),
        MaskCandidate("asset_child", "asset_parent", child, 0.74, "parts"),
    ]
    baseline = fuse_candidates(candidates, image_shape=shape)
    dense_relaxation = fuse_candidates(
        candidates,
        image_shape=shape,
        config=FusionConfig(
            use_parent_envelope=True,
            use_transitive_residual=True,
            transitive_residual_dense_only=True,
        ),
    )

    assert np.array_equal(dense_relaxation.semantic_map, baseline.semantic_map)
    assert np.array_equal(dense_relaxation.instance_map, baseline.instance_map)


def test_cross_source_consensus_can_resolve_a_conflict() -> None:
    shape = (48, 48)
    root = _mask(shape, (4, 4, 44, 44))
    part = _mask(shape, (15, 15, 33, 33))
    candidates = [
        MaskCandidate("asset", "asset", root, 0.98, "root"),
        MaskCandidate("asset_handle", "asset", part, 0.58, "model_a"),
        MaskCandidate("asset_handle", "asset", part, 0.58, "model_b"),
        MaskCandidate("asset_button", "asset", part, 0.70, "model_c"),
    ]
    consensus = fuse_candidates(
        candidates,
        image_shape=shape,
        config=FusionConfig(
            use_consensus=True,
            use_direct_gate=False,
            use_boundary_ownership=False,
            detail_bonus=0.0,
        ),
    )
    maximum = fuse_candidates(
        candidates,
        image_shape=shape,
        config=FusionConfig(
            use_consensus=False,
            use_direct_gate=False,
            use_boundary_ownership=False,
            detail_bonus=0.0,
        ),
    )
    handle_id = consensus.taxonomy.fine_names.index("asset_handle")
    button_id = maximum.taxonomy.fine_names.index("asset_button")
    assert consensus.semantic_map[24, 24] == handle_id
    assert maximum.semantic_map[24, 24] == button_id
    assert consensus.diagnostics["classes_with_multiple_sources"] == ["asset_handle"]


def test_cross_source_identity_hypotheses_merge_into_one_part_id() -> None:
    shape = (48, 48)
    root = _mask(shape, (4, 4, 44, 44))
    first = _mask(shape, (14, 15, 32, 33))
    second = _mask(shape, (15, 14, 33, 32))
    result = fuse_candidates(
        [
            MaskCandidate("asset", "asset", root, 0.98, "root"),
            MaskCandidate("asset_handle", "asset", first, 0.80, "model-a"),
            MaskCandidate("asset_handle", "asset", second, 0.76, "model-b"),
        ],
        image_shape=shape,
    )

    handles = [
        record for record in result.instances if record.semantic_name == "asset_handle"
    ]
    assert len(handles) == 1
    assert result.diagnostics["identity_hypotheses_merged"] == 1


def test_strongly_contained_support_merges_with_one_connected_identity() -> None:
    shape = (80, 80)
    root = _mask(shape, (4, 4, 76, 76))
    whole_garment = _mask(shape, (22, 28, 58, 70))
    local_support = _mask(shape, (43, 51, 57, 69))
    result = fuse_candidates(
        [
            MaskCandidate("asset", "asset", root, 0.98, "root"),
            MaskCandidate(
                "asset_lower_clothing",
                "asset",
                whole_garment,
                0.74,
                "grounded/model",
            ),
            MaskCandidate(
                "asset_lower_clothing",
                "asset",
                local_support,
                0.86,
                "visual/model",
                metadata={"visual_region": True},
            ),
        ],
        image_shape=shape,
    )

    garments = [
        record
        for record in result.instances
        if record.semantic_name == "asset_lower_clothing"
    ]
    assert len(garments) == 1
    assert result.diagnostics["identity_hypotheses_merged"] == 1


def test_disjoint_local_supports_share_one_connected_whole_anchor() -> None:
    shape = (100, 100)
    root = _mask(shape, (3, 3, 97, 97))
    whole = _mask(shape, (20, 20, 80, 82))
    upper_left = _mask(shape, (22, 22, 38, 38))
    lower_right = _mask(shape, (62, 64, 78, 80))
    result = fuse_candidates(
        [
            MaskCandidate("asset", "asset", root, 0.98, "root"),
            MaskCandidate("asset_panel", "asset", whole, 0.60, "grounded/model"),
            MaskCandidate(
                "asset_panel", "asset", upper_left, 0.94, "visual-a/model"
            ),
            MaskCandidate(
                "asset_panel", "asset", lower_right, 0.93, "visual-b/model"
            ),
        ],
        image_shape=shape,
    )

    panels = [
        record for record in result.instances if record.semantic_name == "asset_panel"
    ]
    assert len(panels) == 1
    assert result.diagnostics["identity_hypotheses_merged"] == 2


def test_disconnected_pair_does_not_collapse_repeated_instances() -> None:
    shape = (80, 80)
    root = _mask(shape, (4, 4, 76, 76))
    left = _mask(shape, (14, 52, 32, 66))
    right = _mask(shape, (48, 52, 66, 66))
    result = fuse_candidates(
        [
            MaskCandidate("asset", "asset", root, 0.98, "root"),
            MaskCandidate(
                "asset_shoe", "asset", left | right, 0.72, "grounded/model"
            ),
            MaskCandidate("asset_shoe", "asset", left, 0.88, "visual-a/model"),
            MaskCandidate("asset_shoe", "asset", right, 0.87, "visual-b/model"),
        ],
        image_shape=shape,
    )

    shoes = [
        record for record in result.instances if record.semantic_name == "asset_shoe"
    ]
    assert len(shoes) == 2


def test_instance_cap_drops_excess_support_without_stitching_one_part_id() -> None:
    shape = (80, 80)
    root = _mask(shape, (4, 4, 76, 76))
    upper = _mask(shape, (12, 12, 64, 36))
    lower = _mask(shape, (14, 40, 66, 64))
    result = fuse_candidates(
        [
            MaskCandidate("asset", "asset", root, 0.98, "root"),
            MaskCandidate(
                "asset_screen",
                "asset",
                upper,
                0.82,
                "model-a",
                metadata={"maximum_instances": 1},
            ),
            MaskCandidate(
                "asset_screen",
                "asset",
                lower,
                0.80,
                "model-b",
                metadata={"maximum_instances": 1},
            ),
        ],
        image_shape=shape,
    )

    screens = [
        record for record in result.instances if record.semantic_name == "asset_screen"
    ]
    assert len(screens) == 1
    screen_pixels = result.instance_map == screens[0].instance_index
    assert not np.any(screen_pixels & lower)
    assert result.diagnostics["identity_groups_dropped_by_instance_cap"] == 1
    assert result.diagnostics["identity_groups_merged_by_instance_cap"] == 0


def test_instance_cap_is_applied_per_physical_root_in_scene() -> None:
    shape = (80, 120)
    left_root = _mask(shape, (4, 4, 56, 76))
    right_root = _mask(shape, (64, 4, 116, 76))
    left_screen = _mask(shape, (14, 20, 46, 54))
    right_screen = _mask(shape, (74, 20, 106, 54))
    result = fuse_candidates(
        [
            MaskCandidate(
                "device",
                "device",
                left_root,
                0.98,
                "root",
                metadata={"root_origin": "scene", "root_index": 1},
            ),
            MaskCandidate(
                "device",
                "device",
                right_root,
                0.97,
                "root",
                metadata={"root_origin": "scene", "root_index": 2},
            ),
            MaskCandidate(
                "device_screen",
                "device",
                left_screen,
                0.88,
                "parts",
                metadata={
                    "root_origin": "scene",
                    "root_index": 1,
                    "maximum_instances": 1,
                },
            ),
            MaskCandidate(
                "device_screen",
                "device",
                right_screen,
                0.87,
                "parts",
                metadata={
                    "root_origin": "scene",
                    "root_index": 2,
                    "maximum_instances": 1,
                },
            ),
        ],
        image_shape=shape,
    )

    screens = [
        record for record in result.instances if record.semantic_name == "device_screen"
    ]
    assert len(screens) == 2
    assert result.diagnostics["identity_groups_merged_by_instance_cap"] == 0
    assert result.diagnostics["identity_groups_dropped_by_instance_cap"] == 0
    assert result.diagnostics["identity_instance_cap_scope_count"] == 2


def test_nearby_uncovered_remainder_attaches_to_existing_identity() -> None:
    shape = (64, 64)
    root = _mask(shape, (3, 3, 61, 61))
    handle = _mask(shape, (10, 20, 20, 32))
    nearby_remainder = _mask(shape, (22, 22, 27, 27))
    candidates = [
        MaskCandidate("asset", "asset", root, 0.98, "root"),
        MaskCandidate("asset_handle", "asset", handle, 0.82, "parts"),
    ]
    taxonomy = taxonomy_from_candidates(candidates)
    labels = np.zeros(shape, dtype=np.int32)
    labels[root] = taxonomy.fine_names.index("asset")
    labels[handle | nearby_remainder] = taxonomy.fine_names.index("asset_handle")
    instance_map, records, diagnostics = _hierarchical_part_ids(
        labels,
        taxonomy,
        candidates,
        minimum_area=4,
        config=FusionConfig(remainder_merge_distance_ratio=0.10),
    )
    handles = [record for record in records if record.semantic_name == "asset_handle"]

    assert len(handles) == 1
    assert instance_map[24, 24] == handles[0].instance_index
    assert diagnostics["remainder_components_attached"] == 1


def test_same_source_near_duplicates_are_not_double_counted() -> None:
    shape = (40, 40)
    duplicate = _mask(shape, (8, 8, 32, 32))
    candidates = [
        MaskCandidate("asset", "asset", duplicate, 0.9, "detector"),
        MaskCandidate("asset", "asset", duplicate.copy(), 0.8, "detector"),
    ]
    result = fuse_candidates(candidates, image_shape=shape)
    assert result.diagnostics["same_source_duplicates_removed"] == 1
    assert result.diagnostics["accepted_candidate_count"] == 1


def test_correlated_stages_from_one_source_family_are_not_consensus_votes() -> None:
    shape = (40, 40)
    root = _mask(shape, (2, 2, 38, 38))
    part = _mask(shape, (10, 10, 30, 30))
    single = [
        MaskCandidate("asset", "asset", root, 0.95, "root"),
        MaskCandidate(
            "asset_handle",
            "asset",
            part,
            0.58,
            "grounded-sam2[detector|segmenter]/hierarchy-1",
            metadata={"parent_candidate_key": "parent-a"},
        ),
    ]
    repeated = [
        *single,
        MaskCandidate(
            "asset_handle",
            "asset",
            part,
            0.58,
            "grounded-sam2[detector|segmenter]/hierarchy-2",
            metadata={"parent_candidate_key": "parent-b"},
        ),
    ]
    single_result = fuse_candidates(single, image_shape=shape)
    repeated_result = fuse_candidates(repeated, image_shape=shape)
    handle_id = repeated_result.taxonomy.fine_names.index("asset_handle")

    assert np.array_equal(
        repeated_result.evidence[handle_id], single_result.evidence[handle_id]
    )
    assert repeated_result.diagnostics["classes_with_multiple_sources"] == []


def test_model_repository_slash_does_not_collapse_independent_sources() -> None:
    shape = (40, 40)
    root = _mask(shape, (2, 2, 38, 38))
    part = _mask(shape, (10, 10, 30, 30))
    result = fuse_candidates(
        [
            MaskCandidate("asset", "asset", root, 0.95, "root"),
            MaskCandidate(
                "asset_handle",
                "asset",
                part,
                0.58,
                "grounded-sam2[org/model-tiny|org/sam]/hierarchy-1",
            ),
            MaskCandidate(
                "asset_handle",
                "asset",
                part,
                0.58,
                "grounded-sam2[org/model-base|org/sam]/hierarchy-1",
            ),
        ],
        image_shape=shape,
    )

    assert result.diagnostics["same_source_duplicates_removed"] == 0
    assert result.diagnostics["classes_with_multiple_sources"] == ["asset_handle"]


def test_tiny_and_base_checkpoints_do_not_create_false_consensus() -> None:
    shape = (40, 40)
    root = _mask(shape, (2, 2, 38, 38))
    part = _mask(shape, (10, 10, 30, 30))
    result = fuse_candidates(
        [
            MaskCandidate("asset", "asset", root, 0.95, "root"),
            MaskCandidate(
                "asset_handle",
                "asset",
                part,
                0.58,
                "grounded-sam2[IDEA-Research/grounding-dino-tiny|org/sam]/profile-refine",
            ),
            MaskCandidate(
                "asset_handle",
                "asset",
                part,
                0.62,
                "grounded-sam2[IDEA-Research/grounding-dino-base|org/sam]/profile-refine",
            ),
        ],
        image_shape=shape,
    )

    assert result.diagnostics["classes_with_multiple_sources"] == []


def test_correlated_checkpoint_mask_arbitration_uses_independent_semantic_gate() -> None:
    shape = (48, 48)
    root = _mask(shape, (2, 2, 46, 46))
    cleaner = _mask(shape, (10, 12, 38, 30))
    overshrunk = _mask(shape, (13, 12, 38, 30))
    result = fuse_candidates(
        [
            MaskCandidate("asset", "asset", root, 0.95, "root"),
            MaskCandidate(
                "asset_band",
                "asset",
                cleaner,
                0.55,
                "grounded-sam2[IDEA-Research/grounding-dino-tiny|org/sam]/profile-refine",
                source_reliability=0.84,
                metadata={
                    "sam_quality": 0.89,
                    "profile_dense_top_mean": 0.82,
                    "profile_dense_contrast": 0.14,
                },
            ),
            MaskCandidate(
                "asset_band",
                "asset",
                overshrunk,
                0.64,
                "grounded-sam2[IDEA-Research/grounding-dino-base|org/sam]/profile-refine",
                source_reliability=0.84,
                metadata={
                    "sam_quality": 0.90,
                    "profile_dense_top_mean": 0.82,
                    "profile_dense_contrast": 0.07,
                },
            ),
        ],
        image_shape=shape,
    )

    accepted_bands = [
        candidate
        for candidate in result.accepted_candidates
        if candidate.semantic_name == "asset_band"
    ]
    assert len(accepted_bands) == 1
    assert "grounding-dino-tiny" in accepted_bands[0].source


def test_disjoint_cross_source_hypotheses_are_softly_downweighted() -> None:
    shape = (64, 64)
    root = _mask(shape, (2, 2, 62, 62))
    left = _mask(shape, (8, 20, 24, 36))
    right = _mask(shape, (40, 20, 56, 36))
    candidates = [
        MaskCandidate("asset", "asset", root, 0.95, "root"),
        MaskCandidate("asset_handle", "asset", left, 0.75, "model-a"),
        MaskCandidate("asset_handle", "asset", right, 0.75, "model-b"),
    ]
    calibrated = fuse_candidates(
        candidates,
        image_shape=shape,
        config=FusionConfig(
            use_parent_residual=False,
            use_direct_gate=False,
            detail_bonus=0.0,
            uncorroborated_source_penalty=0.78,
        ),
    )
    uncalibrated = fuse_candidates(
        candidates,
        image_shape=shape,
        config=FusionConfig(
            use_parent_residual=False,
            use_direct_gate=False,
            detail_bonus=0.0,
            uncorroborated_source_penalty=1.0,
        ),
    )
    handle_id = calibrated.taxonomy.fine_names.index("asset_handle")

    assert calibrated.evidence[handle_id, 24, 16] < uncalibrated.evidence[
        handle_id, 24, 16
    ]
    assert calibrated.diagnostics["uncorroborated_cross_source_candidates"] == 2


def test_disconnected_visible_regions_keep_one_candidate_identity() -> None:
    shape = (64, 64)
    root = _mask(shape, (5, 5, 59, 59))
    disconnected = _mask(shape, (12, 20, 22, 30)) | _mask(shape, (42, 20, 52, 30))
    result = fuse_candidates(
        [
            MaskCandidate(
                "asset",
                "asset",
                root,
                0.95,
                "root",
                metadata={"root_index": 7},
            ),
            MaskCandidate(
                "asset_handle",
                "asset",
                disconnected,
                0.83,
                "parts",
                metadata={"root_index": 7},
            ),
        ],
        image_shape=shape,
    )
    handles = [
        record for record in result.instances if record.semantic_name == "asset_handle"
    ]
    assert len(handles) == 1
    assert handles[0].assembly_parent_id is not None
    assert np.count_nonzero(result.instance_map == handles[0].instance_index) == 200


def test_two_disjoint_candidates_receive_two_part_ids() -> None:
    shape = (64, 64)
    root = _mask(shape, (5, 5, 59, 59))
    result = fuse_candidates(
        [
            MaskCandidate("asset", "asset", root, 0.95, "root"),
            MaskCandidate(
                "asset_wheel", "asset", _mask(shape, (10, 36, 22, 50)), 0.8, "parts"
            ),
            MaskCandidate(
                "asset_wheel", "asset", _mask(shape, (42, 36, 54, 50)), 0.8, "parts"
            ),
        ],
        image_shape=shape,
    )
    wheels = [
        record for record in result.instances if record.semantic_name == "asset_wheel"
    ]
    assert len(wheels) == 2
    assert {record.side for record in wheels} == {"left", "right"}


def test_nested_semantic_hierarchy_resolves_actual_parent_part_ids() -> None:
    shape = (72, 72)
    root = _mask(shape, (4, 4, 68, 68))
    head = _mask(shape, (15, 10, 57, 48))
    eye = _mask(shape, (26, 24, 38, 32))
    result = fuse_candidates(
        [
            MaskCandidate(
                "asset", "asset", root, 0.98, "root", metadata={"root_index": 3}
            ),
            MaskCandidate(
                "asset_head",
                "asset",
                head,
                0.90,
                "parts",
                metadata={"root_index": 3},
            ),
            MaskCandidate(
                "asset_eye",
                "asset_head",
                eye,
                0.86,
                "details",
                metadata={"root_index": 3},
            ),
        ],
        image_shape=shape,
    )
    records = {record.semantic_name: record for record in result.instances}
    assert records["asset_head"].assembly_parent_id == records["asset"].part_id
    assert records["asset_eye"].assembly_parent_id == records["asset_head"].part_id
    eye_id = result.taxonomy.fine_names.index("asset_eye")
    assert result.semantic_map[27, 30] == eye_id


def test_side_is_measured_relative_to_actual_parent_instance() -> None:
    shape = (80, 100)
    root = _mask(shape, (2, 2, 98, 78))
    head = _mask(shape, (58, 10, 94, 50))
    left_eye = _mask(shape, (64, 24, 70, 30))
    right_eye = _mask(shape, (82, 24, 88, 30))
    result = fuse_candidates(
        [
            MaskCandidate("asset", "asset", root, 0.98, "root"),
            MaskCandidate("asset_head", "asset", head, 0.90, "parts"),
            MaskCandidate("asset_eye", "asset_head", left_eye, 0.86, "details"),
            MaskCandidate("asset_eye", "asset_head", right_eye, 0.84, "details"),
        ],
        image_shape=shape,
    )
    eyes = [record for record in result.instances if record.semantic_name == "asset_eye"]

    assert {record.side for record in eyes} == {"left", "right"}
    assert len({record.assembly_parent_id for record in eyes}) == 1


def test_part_spanning_parent_center_is_labeled_center() -> None:
    shape = (64, 80)
    root = _mask(shape, (4, 4, 76, 60))
    garment = _mask(shape, (18, 24, 64, 48))
    result = fuse_candidates(
        [
            MaskCandidate("asset", "asset", root, 0.98, "root"),
            MaskCandidate("asset_garment", "asset", garment, 0.88, "parts"),
        ],
        image_shape=shape,
    )
    record = next(
        item for item in result.instances if item.semantic_name == "asset_garment"
    )

    assert record.side == "center"


def test_query_parent_can_differ_from_semantic_and_assembly_parent() -> None:
    shape = (64, 64)
    root = _mask(shape, (4, 4, 60, 60))
    body = _mask(shape, (12, 12, 52, 52))
    overlay = _mask(shape, (8, 8, 28, 20))
    result = fuse_candidates(
        [
            MaskCandidate(
                "asset",
                "asset",
                root,
                0.98,
                "root",
                metadata={
                    "root_index": 1,
                    "candidate_key": "root:1",
                },
            ),
            MaskCandidate(
                "asset_body",
                "asset",
                body,
                0.91,
                "parts",
                metadata={
                    "root_index": 1,
                    "candidate_key": "root:1/asset_body:01",
                    "parent_candidate_key": "root:1",
                    "assembly_parent_semantic": "asset",
                    "assembly_parent_candidate_key": "root:1",
                },
            ),
            MaskCandidate(
                "asset_overlay",
                "asset",
                overlay,
                0.85,
                "details",
                metadata={
                    "root_index": 1,
                    "candidate_key": "root:1/asset_body:01/asset_overlay:01",
                    "parent_candidate_key": "root:1/asset_body:01",
                    "query_parent_semantic": "asset_body",
                    "assembly_parent_semantic": "asset_body",
                    "assembly_parent_candidate_key": "root:1/asset_body:01",
                },
            ),
        ],
        image_shape=shape,
    )
    records = {record.semantic_name: record for record in result.instances}
    assert records["asset_overlay"].semantic_parent == "asset"
    assert records["asset_overlay"].assembly_parent_id == records["asset_body"].part_id


def test_assembly_dependency_controls_instance_creation_order() -> None:
    shape = (48, 48)
    root = _mask(shape, (2, 2, 46, 46))
    parent = _mask(shape, (10, 10, 38, 38))
    child = _mask(shape, (12, 12, 24, 24))
    result = fuse_candidates(
        [
            MaskCandidate(
                "asset",
                "asset",
                root,
                0.98,
                "root",
                metadata={"root_index": 1, "candidate_key": "root:1"},
            ),
            MaskCandidate(
                "asset_z_parent",
                "asset",
                parent,
                0.90,
                "parts",
                metadata={
                    "root_index": 1,
                    "candidate_key": "root:1/asset_z_parent:01",
                    "assembly_parent_semantic": "asset",
                    "assembly_parent_candidate_key": "root:1",
                },
            ),
            MaskCandidate(
                "asset_a_child",
                "asset",
                child,
                0.96,
                "details",
                metadata={
                    "root_index": 1,
                    "candidate_key": "root:1/asset_z_parent:01/asset_a_child:01",
                    "assembly_parent_semantic": "asset_z_parent",
                    "assembly_parent_candidate_key": "root:1/asset_z_parent:01",
                },
            ),
        ],
        image_shape=shape,
    )
    records = {record.semantic_name: record for record in result.instances}
    assert (
        records["asset_z_parent"].instance_index
        < records["asset_a_child"].instance_index
    )
    assert (
        records["asset_a_child"].assembly_parent_id == records["asset_z_parent"].part_id
    )


def test_scene_object_identity_is_preserved_in_part_records() -> None:
    shape = (48, 64)
    left = _mask(shape, (2, 4, 28, 44))
    right = _mask(shape, (36, 4, 62, 44))
    result = fuse_candidates(
        [
            MaskCandidate(
                "asset",
                "asset",
                left,
                0.95,
                "scene",
                metadata={
                    "root_index": 1,
                    "scene_object_id": "object_001",
                    "candidate_key": "root:1",
                },
            ),
            MaskCandidate(
                "asset",
                "asset",
                right,
                0.95,
                "scene",
                metadata={
                    "root_index": 2,
                    "scene_object_id": "object_002",
                    "candidate_key": "root:2",
                },
            ),
        ],
        image_shape=shape,
    )

    assert {record.asset_id for record in result.instances} == {
        "object_001",
        "object_002",
    }
    assert len({record.part_id for record in result.instances}) == 2


def test_null_scene_identity_falls_back_to_primary_asset_id() -> None:
    shape = (24, 32)
    root = _mask(shape, (2, 2, 30, 22))
    result = fuse_candidates(
        [
            MaskCandidate(
                "asset",
                "asset",
                root,
                0.95,
                "automatic",
                metadata={"root_index": 1, "scene_object_id": None},
            )
        ],
        image_shape=shape,
    )

    assert result.instances[0].asset_id == "object_001"


def test_scene_children_only_attach_to_parents_in_the_same_asset() -> None:
    shape = (48, 80)
    left_root = _mask(shape, (2, 4, 36, 44))
    right_root = _mask(shape, (44, 4, 78, 44))
    left_child = _mask(shape, (8, 12, 28, 34))
    right_child = _mask(shape, (52, 12, 72, 34))
    result = fuse_candidates(
        [
            MaskCandidate(
                "asset",
                "asset",
                left_root,
                0.95,
                "scene",
                metadata={"root_index": 1, "scene_object_id": "object_001"},
            ),
            MaskCandidate(
                "asset",
                "asset",
                right_root,
                0.94,
                "scene",
                metadata={"root_index": 2, "scene_object_id": "object_002"},
            ),
            MaskCandidate(
                "asset_panel",
                "asset",
                left_child,
                0.88,
                "scene",
                metadata={"root_index": 1, "scene_object_id": "object_001"},
            ),
            MaskCandidate(
                "asset_panel",
                "asset",
                right_child,
                0.87,
                "scene",
                metadata={"root_index": 2, "scene_object_id": "object_002"},
            ),
        ],
        image_shape=shape,
    )

    by_part_id = {record.part_id: record for record in result.instances}
    children = [
        record for record in result.instances if record.semantic_name == "asset_panel"
    ]
    assert len(children) == 2
    for child in children:
        parent = by_part_id[child.assembly_parent_id]
        assert parent.asset_id == child.asset_id


def test_nested_scene_object_owns_pixels_over_same_semantic_scene_layer() -> None:
    shape = (64, 80)
    scene_layer = _mask(shape, (2, 2, 78, 62))
    nested_object = _mask(shape, (28, 20, 48, 44))
    nested_part = _mask(shape, (34, 26, 42, 36))
    result = fuse_candidates(
        [
            MaskCandidate(
                "terrain",
                "terrain",
                scene_layer,
                0.96,
                "scene",
                metadata={
                    "root_index": 1,
                    "scene_object_id": "scene_001",
                    "scene_role": "scene_layer",
                    "candidate_key": "root:1",
                    "proposal_first_evidence": {
                        "derived_from_background_complement": True,
                        "area_fraction": 0.89,
                    },
                },
            ),
            MaskCandidate(
                "terrain",
                "terrain",
                nested_object,
                0.82,
                "scene",
                metadata={
                    "root_index": 2,
                    "scene_object_id": "object_002",
                    "scene_role": "object",
                    "candidate_key": "root:2",
                },
            ),
            MaskCandidate(
                "terrain_rock",
                "terrain",
                nested_part,
                0.94,
                "scene-part",
                metadata={
                    "root_index": 2,
                    "scene_object_id": "object_002",
                    "candidate_key": "root:2/visual-region:01",
                    "parent_candidate_key": "root:2",
                    "assembly_parent_semantic": "terrain",
                    "assembly_parent_candidate_key": "root:2",
                },
            ),
        ],
        image_shape=shape,
    )

    rock = next(
        record for record in result.instances if record.semantic_name == "terrain_rock"
    )
    by_part_id = {record.part_id: record for record in result.instances}
    terrain_assets = {
        record.asset_id
        for record in result.instances
        if record.semantic_name == "terrain"
    }
    assert terrain_assets == {"scene_001", "object_002"}
    assert rock.assembly_parent_id is not None
    assert by_part_id[rock.assembly_parent_id].asset_id == "object_002"
    assert result.diagnostics["unresolved_assembly_parent_ids"] == []


def test_scene_layer_is_fallback_behind_overlapping_object_root() -> None:
    shape = (64, 80)
    scene_layer = _mask(shape, (2, 2, 78, 62))
    object_mask = _mask(shape, (26, 18, 54, 46))
    result = fuse_candidates(
        [
            MaskCandidate(
                "terrain",
                "terrain",
                scene_layer,
                0.99,
                "scene",
                metadata={
                    "root_index": 1,
                    "scene_object_id": "scene_001",
                    "scene_role": "scene_layer",
                    "candidate_key": "root:1",
                    "proposal_first_evidence": {
                        "derived_from_background_complement": True,
                        "area_fraction": 0.89,
                    },
                },
            ),
            MaskCandidate(
                "furniture",
                "furniture",
                object_mask,
                0.75,
                "scene",
                metadata={
                    "root_index": 2,
                    "scene_object_id": "object_002",
                    "scene_role": "object",
                    "candidate_key": "root:2",
                },
            ),
        ],
        image_shape=shape,
    )

    by_semantic = {record.semantic_name: record for record in result.instances}
    assert {"terrain", "furniture"} <= set(by_semantic)
    assert by_semantic["furniture"].asset_id == "object_002"
    furniture_id = result.taxonomy.fine_names.index("furniture")
    assert result.semantic_map[30, 30] == furniture_id
