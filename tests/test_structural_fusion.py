from dataclasses import replace

import cv2
import numpy as np

from hpid_split.fusion import MaskCandidate
from hpid_split.prompt_bank import DomainPrompt, PartProfile, PartPrompt
from hpid_split.structural_fusion import refine_profile_structure


def _candidate(
    semantic_name: str,
    semantic_parent: str,
    mask: np.ndarray,
    *,
    key: str,
    score: float = 0.8,
    reliability: float = 0.8,
    generic: bool = False,
) -> MaskCandidate:
    return MaskCandidate(
        semantic_name=semantic_name,
        semantic_parent=semantic_parent,
        mask=mask,
        score=score,
        source="test/source",
        prompt=semantic_name,
        source_reliability=reliability,
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": key,
            "parent_candidate_key": None if key == "root:1" else "root:1",
            "visual_region": generic,
            "generic_visual_region": generic,
            "root_area_fraction": float(mask.mean()),
            "ground_truth_used": False,
        },
    )


def test_axial_structure_replaces_bad_incumbent_and_recovers_residual_mass() -> None:
    shape = (96, 180)
    strip = np.zeros(shape, dtype=bool)
    strip[44:51, 8:126] = True
    bowl = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(bowl, (145, 47), (30, 23), 0, 0, 360, 1, -1)
    bowl = bowl.astype(bool)
    root_mask = strip | bowl
    root = _candidate("tool_prop", "tool_prop", root_mask, key="root:1")
    root = replace(
        root,
        metadata={**root.metadata, "root_model_label": "spoon"},
    )
    wrong_incumbent = _candidate(
        "tool_prop_handle",
        "tool_prop_body",
        bowl,
        key="root:1/wrong-handle",
        score=0.25,
        reliability=0.75,
    )
    visual_strip = _candidate(
        "tool_prop_visual_panel_01",
        "tool_prop",
        strip,
        key="root:1/visual:01",
        score=0.90,
        reliability=0.72,
        generic=True,
    )
    domain = DomainPrompt(
        name="tool_prop",
        root_prompts=("tool",),
        parts=(
            PartPrompt("tool_prop_body", ("tool body",)),
            PartPrompt(
                "tool_prop_handle",
                ("handle",),
                semantic_parent="tool_prop_body",
                maximum_instances=1,
            ),
            PartPrompt(
                "tool_prop_bowl",
                ("spoon bowl",),
                semantic_parent="tool_prop_body",
                maximum_instances=1,
            ),
            PartPrompt(
                "tool_prop_rim",
                ("rim",),
                semantic_parent="tool_prop_bowl",
                maximum_instances=1,
                detail=True,
            ),
        ),
    )

    result = refine_profile_structure(
        [visual_strip],
        [root],
        [root, wrong_incumbent],
        {"tool_prop": domain},
    )

    by_name = {candidate.semantic_name: candidate for candidate in result.candidates}
    assert np.array_equal(by_name["tool_prop_handle"].mask, strip)
    assert np.array_equal(by_name["tool_prop_bowl"].mask, root_mask & ~strip)
    assert not np.any(
        by_name["tool_prop_handle"].mask & by_name["tool_prop_bowl"].mask
    )
    assert np.array_equal(
        by_name["tool_prop_handle"].mask | by_name["tool_prop_bowl"].mask,
        root_mask,
    )
    assert by_name["tool_prop_handle"].metadata["ground_truth_used"] is False
    assert by_name["tool_prop_bowl"].metadata["ground_truth_used"] is False
    assert result.diagnostics["semantic_reassignment_count"] == 1
    assert result.diagnostics["generated_residual_count"] == 1


def test_named_part_consensus_recovers_profile_and_relabels_generic_stock() -> None:
    shape = (90, 220)
    root_mask = np.zeros(shape, dtype=bool)
    root_mask[10:80, 5:215] = True
    root = _candidate("tool_prop", "tool_prop", root_mask, key="root:1")

    def region(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
        mask = np.zeros(shape, dtype=bool)
        mask[y0:y1, x0:x1] = True
        return mask

    barrel = _candidate(
        "tool_prop_barrel",
        "tool_prop",
        region(8, 25, 58, 38),
        key="root:1/barrel",
    )
    receiver = _candidate(
        "tool_prop_receiver",
        "tool_prop",
        region(82, 24, 130, 52),
        key="root:1/receiver",
    )
    magazine = _candidate(
        "tool_prop_magazine",
        "tool_prop",
        region(88, 50, 112, 78),
        key="root:1/magazine",
    )
    generic_stock = _candidate(
        "tool_prop_visual_panel_01",
        "tool_prop",
        region(145, 15, 212, 75),
        key="root:1/visual-stock",
        generic=True,
    )
    parts = (
        PartPrompt(
            "tool_prop_stock",
            ("stock",),
            minimum_parent_fraction=0.03,
            maximum_parent_fraction=0.50,
            maximum_instances=1,
            axis_position=0.85,
            axis_tolerance=0.40,
        ),
        PartPrompt(
            "tool_prop_receiver",
            ("receiver",),
            minimum_parent_fraction=0.03,
            maximum_parent_fraction=0.42,
            maximum_instances=1,
            axis_position=0.0,
            axis_tolerance=0.50,
        ),
        PartPrompt(
            "tool_prop_barrel",
            ("barrel",),
            minimum_parent_fraction=0.008,
            maximum_parent_fraction=0.22,
            maximum_instances=2,
            axis_position=-0.65,
            axis_tolerance=0.42,
        ),
        PartPrompt(
            "tool_prop_magazine",
            ("magazine",),
            minimum_parent_fraction=0.015,
            maximum_parent_fraction=0.22,
            maximum_instances=1,
            axis_position=-0.15,
            axis_tolerance=0.32,
        ),
        PartPrompt("tool_prop_handle", ("handle",), maximum_instances=1),
        PartPrompt("tool_prop_shaft", ("shaft",), maximum_instances=1),
    )
    domain = DomainPrompt(
        name="tool_prop",
        root_prompts=("tool",),
        parts=parts,
        part_profiles=(
            PartProfile(
                "firearm",
                ("firearm", "rifle"),
                (
                    "tool_prop_stock",
                    "tool_prop_receiver",
                    "tool_prop_barrel",
                    "tool_prop_magazine",
                ),
            ),
            PartProfile(
                "screwdriver",
                ("screwdriver",),
                ("tool_prop_handle", "tool_prop_shaft"),
            ),
        ),
    )

    result = refine_profile_structure(
        [generic_stock],
        [root],
        [root, barrel, receiver, magazine],
        {"tool_prop": domain},
    )

    assert result.candidates[0].semantic_name == "tool_prop_stock"
    assert result.candidates[0].metadata["structural_profile"] == "firearm"
    consensus = result.diagnostics["observed_part_profile_consensus"][0]
    assert consensus["status"] == "accepted"
    assert consensus["selected_profile"] == "firearm"
    assert result.diagnostics["profile_axis_assignments"][0]["semantic_name"] == (
        "tool_prop_stock"
    )


def test_multi_strip_inventory_is_not_forced_into_binary_structure() -> None:
    root_mask = np.zeros((64, 64), dtype=bool)
    root_mask[8:56, 8:56] = True
    root = _candidate("tool_prop", "tool_prop", root_mask, key="root:1")
    visual = _candidate(
        "tool_prop_visual_strip_01",
        "tool_prop",
        root_mask,
        key="root:1/visual:01",
        generic=True,
    )
    domain = DomainPrompt(
        name="tool_prop",
        root_prompts=("tool",),
        parts=(
            PartPrompt(
                "tool_prop_handle",
                ("handle",),
                maximum_instances=1,
            ),
            PartPrompt(
                "tool_prop_blade",
                ("blade",),
                maximum_instances=1,
            ),
            PartPrompt(
                "tool_prop_head",
                ("head",),
                maximum_instances=1,
            ),
        ),
    )

    result = refine_profile_structure(
        [visual], [root], [root], {"tool_prop": domain}
    )

    assert result.candidates == (visual,)
    assert result.diagnostics["eligible_root_count"] == 0


def test_parallel_rails_and_perpendicular_crossbars_receive_distinct_ids() -> None:
    shape = (120, 90)
    left_rail = np.zeros(shape, dtype=bool)
    left_rail[8:112, 15:21] = True
    right_rail = np.zeros(shape, dtype=bool)
    right_rail[8:112, 68:74] = True
    upper_step = np.zeros(shape, dtype=bool)
    upper_step[34:39, 18:71] = True
    lower_step = np.zeros(shape, dtype=bool)
    lower_step[78:83, 18:71] = True
    root_mask = left_rail | right_rail | upper_step | lower_step
    root = _candidate("tool_prop", "tool_prop", root_mask, key="root:1")
    visual = [
        _candidate(
            f"tool_prop_visual_strip_{index:02d}",
            "tool_prop",
            mask,
            key=f"root:1/visual:{index:02d}",
            generic=True,
        )
        for index, mask in enumerate(
            (left_rail, right_rail, upper_step, lower_step), start=1
        )
    ]
    domain = DomainPrompt(
        name="tool_prop",
        root_prompts=("ladder",),
        parts=(
            PartPrompt(
                "tool_prop_rail",
                ("rail",),
                maximum_instances=2,
            ),
            PartPrompt(
                "tool_prop_step",
                ("step",),
                semantic_parent="tool_prop_rail",
                maximum_instances=12,
            ),
            PartPrompt(
                "tool_prop_top_cap",
                ("top cap",),
                maximum_instances=2,
            ),
        ),
    )

    result = refine_profile_structure(
        visual, [root], [root], {"tool_prop": domain}
    )

    semantics = [candidate.semantic_name for candidate in result.candidates]
    assert semantics.count("tool_prop_rail") == 2
    assert semantics.count("tool_prop_step") == 2
    assert result.diagnostics["semantic_reassignment_count"] == 4
    assert all(
        candidate.metadata["ground_truth_used"] is False
        for candidate in result.candidates
    )


def test_hinged_tool_assigns_opposite_branches_as_handles() -> None:
    shape = (120, 120)

    def branch(start: tuple[int, int], end: tuple[int, int], thickness: int) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        cv2.line(mask, start, end, 1, thickness=thickness)
        return mask.astype(bool)

    pivot_mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(pivot_mask, (60, 58), 5, 1, -1)
    pivot_mask = pivot_mask.astype(bool)
    blades = (
        branch((58, 56), (32, 10), 7),
        branch((62, 56), (92, 14), 7),
    )
    handles = (
        branch((57, 62), (30, 108), 9),
        branch((63, 62), (92, 108), 9),
    )
    root_mask = pivot_mask.copy()
    for mask in (*blades, *handles):
        root_mask |= mask
    root = _candidate("tool_prop", "tool_prop", root_mask, key="root:1")
    pivot = _candidate(
        "tool_prop_pivot",
        "tool_prop_body",
        pivot_mask,
        key="root:1/pivot",
    )
    blade_candidates = [
        _candidate(
            "tool_prop_blade",
            "tool_prop_body",
            mask,
            key=f"root:1/blade:{index}",
        )
        for index, mask in enumerate(blades, start=1)
    ]
    visual_handles = [
        _candidate(
            f"tool_prop_visual_strip_{index:02d}",
            "tool_prop",
            mask,
            key=f"root:1/visual:{index}",
            generic=True,
        )
        for index, mask in enumerate(handles, start=1)
    ]
    domain = DomainPrompt(
        name="tool_prop",
        root_prompts=("scissors",),
        parts=(
            PartPrompt(
                "tool_prop_handle",
                ("handle",),
                maximum_instances=2,
            ),
            PartPrompt(
                "tool_prop_blade",
                ("blade",),
                maximum_instances=2,
            ),
            PartPrompt(
                "tool_prop_pivot",
                ("pivot",),
                maximum_instances=1,
                detail=True,
            ),
        ),
    )

    result = refine_profile_structure(
        visual_handles,
        [root],
        [root, pivot, *blade_candidates],
        {"tool_prop": domain},
    )

    assert [candidate.semantic_name for candidate in result.candidates] == [
        "tool_prop_handle",
        "tool_prop_handle",
    ]
    assert all(
        candidate.metadata["structural_role"] == "hinged_opposite_branch"
        for candidate in result.candidates
    )


def test_overlapping_book_surface_tiles_become_one_cover_hypothesis() -> None:
    shape = (100, 100)
    root_mask = np.zeros(shape, dtype=bool)
    root_mask[5:95, 5:95] = True
    upper = np.zeros(shape, dtype=bool)
    upper[8:62, 8:92] = True
    lower = np.zeros(shape, dtype=bool)
    lower[48:92, 8:92] = True
    base_root = _candidate(
        "daily_object", "daily_object", root_mask, key="root:1"
    )
    root = replace(
        base_root,
        metadata={
            **base_root.metadata,
            "root_model_label": "book",
            "selected_part_profile": "book",
        },
    )
    tiles = []
    for index, mask in enumerate((upper, lower), start=1):
        tile = _candidate(
            "daily_object_cover",
            "daily_object_body",
            mask,
            key=f"root:1/cover:{index}",
        )
        tiles.append(
            replace(tile, metadata={**tile.metadata, "visual_region": True})
        )
    domain = DomainPrompt(
        name="daily_object",
        root_prompts=("book",),
        parts=(
            PartPrompt("daily_object_body", ("book body",)),
            PartPrompt(
                "daily_object_cover",
                ("book cover",),
                semantic_parent="daily_object_body",
                maximum_instances=2,
            ),
            PartPrompt(
                "daily_object_page",
                ("book page",),
                semantic_parent="daily_object_body",
                maximum_instances=6,
            ),
        ),
        part_profiles=(
            PartProfile(
                name="book",
                root_hints=("book",),
                part_semantics=("daily_object_cover", "daily_object_page"),
            ),
        ),
    )

    result = refine_profile_structure(
        tiles, [root], [root, *tiles], {"daily_object": domain}
    )
    covers = [
        candidate
        for candidate in result.candidates
        if candidate.semantic_name == "daily_object_cover"
    ]

    assert len(covers) == 1
    assert np.array_equal(covers[0].mask, (upper | lower) & root_mask)
    assert covers[0].metadata["structural_fusion_algorithm"] == (
        "profile-planar-tile-union-v1"
    )
    assert result.diagnostics["aggregated_surface_count"] == 1


def test_silhouette_width_change_recovers_wrench_head_and_handle() -> None:
    shape = (100, 220)
    handle = np.zeros(shape, dtype=np.uint8)
    cv2.line(handle, (18, 52), (150, 52), 1, thickness=9)
    head = np.zeros(shape, dtype=np.uint8)
    cv2.circle(head, (178, 52), 28, 1, -1)
    cv2.circle(head, (184, 52), 12, 0, -1)
    root_mask = (handle | head).astype(bool)
    root = replace(
        _candidate("tool_prop", "tool_prop", root_mask, key="root:1"),
        metadata={
            **_candidate(
                "tool_prop", "tool_prop", root_mask, key="root:1"
            ).metadata,
            "root_model_label": "wrench",
            "selected_part_profile": "wrench",
        },
    )
    domain = DomainPrompt(
        name="tool_prop",
        root_prompts=("tool",),
        parts=(
            PartPrompt("tool_prop_body", ("tool body",)),
            PartPrompt(
                "tool_prop_handle",
                ("handle",),
                semantic_parent="tool_prop_body",
                maximum_parent_fraction=0.82,
                maximum_instances=1,
            ),
            PartPrompt(
                "tool_prop_head",
                ("wrench head",),
                semantic_parent="tool_prop_body",
                maximum_instances=1,
            ),
            PartPrompt(
                "tool_prop_jaw",
                ("wrench jaw",),
                semantic_parent="tool_prop_head",
                maximum_instances=2,
            ),
        ),
    )

    result = refine_profile_structure([], [root], [root], {"tool_prop": domain})
    by_name = {candidate.semantic_name: candidate for candidate in result.candidates}

    assert set(by_name) == {"tool_prop_handle", "tool_prop_head"}
    assert not np.any(by_name["tool_prop_handle"].mask & by_name["tool_prop_head"].mask)
    assert np.array_equal(
        by_name["tool_prop_handle"].mask | by_name["tool_prop_head"].mask,
        root_mask,
    )
    assert all(
        candidate.metadata["structural_root_evidence"] is True
        for candidate in by_name.values()
    )
    assert result.diagnostics["silhouette_partition_count"] == 1
    assert result.diagnostics["ground_truth_used"] is False


def test_uniform_elongated_bar_is_not_invented_as_two_parts() -> None:
    root_mask = np.zeros((80, 220), dtype=bool)
    root_mask[34:46, 10:210] = True
    root = replace(
        _candidate("tool_prop", "tool_prop", root_mask, key="root:1"),
        metadata={
            **_candidate(
                "tool_prop", "tool_prop", root_mask, key="root:1"
            ).metadata,
            "root_model_label": "wrench",
            "selected_part_profile": "wrench",
        },
    )
    domain = DomainPrompt(
        name="tool_prop",
        root_prompts=("tool",),
        parts=(
            PartPrompt(
                "tool_prop_handle", ("handle",), maximum_instances=1
            ),
            PartPrompt("tool_prop_head", ("head",), maximum_instances=1),
        ),
    )

    result = refine_profile_structure([], [root], [root], {"tool_prop": domain})

    assert result.candidates == ()
    assert result.diagnostics["silhouette_partition_count"] == 0
    assert result.diagnostics["roots"][0]["status"] == "no_elongated_anchor"


def test_screwdriver_profile_maps_wide_grip_and_narrow_shaft() -> None:
    shape = (90, 240)
    grip = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(grip, (58, 45), (46, 21), 0, 0, 360, 1, -1)
    shaft = np.zeros(shape, dtype=np.uint8)
    cv2.line(shaft, (96, 45), (226, 45), 1, thickness=5)
    root_mask = (grip | shaft).astype(bool)
    base = _candidate("tool_prop", "tool_prop", root_mask, key="root:1")
    root = replace(
        base,
        metadata={
            **base.metadata,
            "root_model_label": "screwdriver",
            "selected_part_profile": "screwdriver",
        },
    )
    domain = DomainPrompt(
        name="tool_prop",
        root_prompts=("tool",),
        parts=(
            PartPrompt(
                "tool_prop_handle",
                ("handle",),
                maximum_parent_fraction=0.82,
                maximum_instances=1,
            ),
            PartPrompt(
                "tool_prop_shaft",
                ("shaft",),
                maximum_parent_fraction=0.82,
                maximum_instances=1,
            ),
            PartPrompt(
                "tool_prop_tip", ("tip",), maximum_instances=1, detail=True
            ),
        ),
        part_profiles=(
            PartProfile(
                name="screwdriver",
                root_hints=("screwdriver",),
                part_semantics=(
                    "tool_prop_handle",
                    "tool_prop_shaft",
                    "tool_prop_tip",
                ),
            ),
        ),
    )

    result = refine_profile_structure([], [root], [root], {"tool_prop": domain})
    by_name = {candidate.semantic_name: candidate for candidate in result.candidates}

    assert set(by_name) == {"tool_prop_handle", "tool_prop_shaft"}
    assert int(np.count_nonzero(by_name["tool_prop_handle"].mask & grip)) > int(
        np.count_nonzero(by_name["tool_prop_handle"].mask & shaft)
    )
    assert int(np.count_nonzero(by_name["tool_prop_shaft"].mask & shaft)) > int(
        np.count_nonzero(by_name["tool_prop_shaft"].mask & grip)
    )
