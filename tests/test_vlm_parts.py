from __future__ import annotations

import json

import numpy as np
from PIL import Image

from hpid_split.foundation import SegmentProposal
from hpid_split.fusion import MaskCandidate
from hpid_split.prompt_bank import DomainPrompt, PartProfile, PartPrompt
from hpid_split.vlm_parts import (
    PartRegion,
    PlannedPart,
    VlmPartConfig,
    VlmPartGenerator,
    apply_dynamic_object_profile_corrections,
    assign_plans_to_regions,
    build_dynamic_object_identity_prompt,
    build_dynamic_part_inventory_prompt,
    build_region_batch_label_prompt,
    build_region_label_prompt,
    build_region_ownership_prompt,
    make_region_batch_query_image,
    make_region_query_image,
    make_root_query_image,
    parse_dynamic_object_identity,
    parse_dynamic_part_inventory,
    parse_part_plan,
    parse_region_batch_label_plan,
    parse_region_label_plan,
    parse_region_ownership_plan,
)


def _part(
    name: str,
    *,
    maximum_instances: int = 4,
    semantic_parent: str = "furniture",
) -> PartPrompt:
    return PartPrompt(
        semantic_name=name,
        prompts=(name.replace("_", " "),),
        semantic_parent=semantic_parent,
        minimum_parent_fraction=0.001,
        maximum_parent_fraction=0.90,
        maximum_instances=maximum_instances,
    )


def test_parse_plan_is_bounded_by_inventory_and_declared_coordinates() -> None:
    seat = _part("furniture_seat", maximum_instances=1)
    response = """Result:\n```json
    {"coordinate_system":"normalized_1000","parts":[
      {"semantic_name":"furniture seat","bbox_2d":[100,200,900,600],
       "confidence":0.91,"visible":true},
      {"semantic_name":"imagined_motor","bbox_2d":[0,0,1000,1000],
       "confidence":0.99,"visible":true}
    ]}
    ```"""

    parsed = parse_part_plan(
        response,
        image_size=(200, 100),
        allowed_parts={seat.semantic_name: seat},
    )

    assert len(parsed.parts) == 1
    assert parsed.parts[0].semantic_name == "furniture_seat"
    assert parsed.parts[0].box_xyxy == (20, 20, 180, 60)
    assert parsed.diagnostics["rejection_counts"]["unknown_semantic"] == 1
    assert parsed.diagnostics["ground_truth_used"] is False


def test_parse_plan_rejects_ambiguous_coordinate_system() -> None:
    seat = _part("furniture_seat")
    parsed = parse_part_plan(
        '{"parts":[{"semantic_name":"furniture_seat",'
        '"bbox_2d":[1,2,3,4],"confidence":0.9}]}',
        image_size=(100, 100),
        allowed_parts={seat.semantic_name: seat},
    )

    assert parsed.parts == ()
    assert parsed.diagnostics["status"] == "invalid_coordinate_system"


def test_parse_plan_can_apply_the_prompt_declared_coordinate_system() -> None:
    seat = _part("furniture_seat", maximum_instances=1)
    parsed = parse_part_plan(
        '{"parts":[{"semantic_name":"furniture_seat",'
        '"bbox_2d":[100,200,900,600],"confidence":0.9}]}',
        image_size=(200, 100),
        allowed_parts={seat.semantic_name: seat},
        expected_coordinate_system="normalized_1000",
    )

    assert len(parsed.parts) == 1
    assert parsed.parts[0].box_xyxy == (20, 20, 180, 60)
    assert parsed.diagnostics["coordinate_system_recovered"] is True


def test_dynamic_inventory_rejects_synonyms_and_incidental_surface_labels() -> None:
    existing = _part("device_button", semantic_parent="device")
    response = json.dumps(
        {
            "parts": [
                {
                    "name": "button",
                    "description": "existing control button",
                    "confidence": 0.99,
                    "visible": True,
                    "physical": True,
                    "maximum_instances": 2,
                },
                {
                    "name": "blue reflection",
                    "description": "gloss on the casing",
                    "confidence": 0.99,
                    "visible": True,
                    "physical": True,
                },
                {
                    "name": "charging coil",
                    "description": "separate circular charging coil",
                    "confidence": 0.93,
                    "visible": True,
                    "physical": True,
                    "detail": False,
                    "maximum_instances": 1,
                },
            ]
        }
    )

    parsed = parse_dynamic_part_inventory(
        response,
        domain_name="device",
        existing_parts=(existing,),
    )

    assert [part.semantic_name for part in parsed.parts] == [
        "device_dynamic_charging_coil"
    ]
    assert parsed.diagnostics["rejection_counts"]["existing_synonym"] == 1
    assert parsed.diagnostics["rejection_counts"]["incidental_surface"] == 1
    assert parsed.diagnostics["ground_truth_used"] is False
    prompt = build_dynamic_part_inventory_prompt(
        object_label="wireless charger",
        domain_name="device",
        existing_parts=(existing,),
        maximum_parts=8,
    )
    assert "independently editable Part ID" in prompt
    assert "device_button" in prompt
    assert "candidate is untrusted" in prompt
    assert "object_confidence" in prompt


def test_dynamic_object_identity_requires_specific_high_confidence_name() -> None:
    prompt = build_dynamic_object_identity_prompt(
        candidate_object_label="screwdriver",
        domain_name="tool_prop",
    )
    assert "candidate is untrusted" in prompt
    assert "Do not list parts" in prompt
    assert '"object_confidence"' in prompt

    accepted = parse_dynamic_object_identity(
        json.dumps(
            {
                "object": "AK-47 assault rifle",
                "object_confidence": 0.96,
                "candidate_label_matches": False,
                "parts": [],
            }
        )
    )
    low_confidence = parse_dynamic_object_identity(
        json.dumps(
            {
                "object": "rifle",
                "object_confidence": 0.55,
                "parts": [],
            }
        )
    )
    generic = parse_dynamic_object_identity(
        json.dumps(
            {
                "object": "unknown object",
                "object_confidence": 0.99,
                "parts": [],
            }
        )
    )

    assert accepted.label == "ak-47 assault rifle"
    assert accepted.diagnostics["candidate_label_matches"] is False
    assert low_confidence.label is None
    assert generic.label is None


def test_open_set_profile_correction_drops_candidates_from_wrong_profile() -> None:
    mask = np.ones((20, 20), dtype=bool)
    root_metadata = {
        "root_origin": "test",
        "root_index": 1,
        "candidate_key": "root:1",
        "parent_candidate_key": None,
    }
    root = MaskCandidate(
        "tool_prop", "tool_prop", mask, 0.95, "root", metadata=root_metadata
    )
    wrong_handle = MaskCandidate(
        "tool_prop_handle",
        "tool_prop_body",
        mask,
        0.8,
        "guided",
        metadata={**root_metadata, "parent_candidate_key": "root:1"},
    )
    body = MaskCandidate(
        "tool_prop_body",
        "tool_prop",
        mask,
        0.8,
        "guided",
        metadata={**root_metadata, "parent_candidate_key": "root:1"},
    )
    visual = MaskCandidate(
        "tool_prop_visual_panel_01",
        "tool_prop",
        mask,
        0.8,
        "visual",
        metadata={
            **root_metadata,
            "parent_candidate_key": "root:1",
            "visual_region": True,
            "generic_visual_region": True,
        },
    )
    wrongly_named_visual = MaskCandidate(
        "tool_prop_handle",
        "tool_prop_body",
        mask,
        0.81,
        "visual-rerank",
        metadata={
            **root_metadata,
            "candidate_key": "root:1/visual:2",
            "parent_candidate_key": "root:1",
            "visual_region": True,
            "generic_visual_region": False,
        },
    )
    firearm_parts = (
        _part("tool_prop_body", semantic_parent="tool_prop"),
        _part("tool_prop_stock", semantic_parent="tool_prop"),
    )
    screwdriver_handle = _part(
        "tool_prop_handle", semantic_parent="tool_prop"
    )
    domain = DomainPrompt(
        "tool_prop",
        ("rifle", "screwdriver"),
        (*firearm_parts, screwdriver_handle),
        part_profiles=(
            PartProfile(
                "firearm",
                ("rifle", "firearm"),
                ("tool_prop_body", "tool_prop_stock"),
            ),
            PartProfile(
                "screwdriver",
                ("screwdriver",),
                ("tool_prop_body", "tool_prop_handle"),
            ),
        ),
    )
    rows = (
        {
            "root_key": "test::1",
            "domain": "tool_prop",
            "initial_selected_profile": "screwdriver",
            "selected_profile": "firearm",
            "dynamic_inventory": {
                "resolved_object_label": "assault rifle",
                "object_identity": {"status": "accepted"},
            },
        },
    )

    applied = apply_dynamic_object_profile_corrections(
        (root, wrong_handle, body, visual, wrongly_named_visual),
        rows,
        {"tool_prop": domain},
    )

    names = [candidate.semantic_name for candidate in applied.candidates]
    assert "tool_prop_handle" not in names
    assert "tool_prop_body" in names
    assert "tool_prop_visual_panel_01" in names
    assert applied.candidates[0].metadata["selected_part_profile"] == "firearm"
    assert applied.diagnostics["dropped_candidate_count"] == 2


def test_parse_plan_applies_duplicate_and_instance_caps() -> None:
    leg = _part("furniture_leg", maximum_instances=2)
    rows = [
        {
            "semantic_name": "furniture_leg",
            "bbox_2d": box,
            "confidence": confidence,
            "visible": True,
        }
        for box, confidence in (
            ([50, 100, 220, 900], 0.91),
            ([55, 105, 218, 895], 0.89),
            ([400, 100, 560, 900], 0.88),
            ([720, 100, 900, 900], 0.87),
        )
    ]
    parsed = parse_part_plan(
        json.dumps({"coordinate_system": "normalized_1000", "parts": rows}),
        image_size=(100, 100),
        allowed_parts={leg.semantic_name: leg},
    )

    assert len(parsed.parts) == 2
    assert parsed.diagnostics["rejection_counts"]["duplicate_box"] == 1
    assert parsed.diagnostics["rejection_counts"]["instance_cap"] == 1


def test_region_label_parser_is_closed_world_and_confidence_gated() -> None:
    seat = _part("furniture_seat")
    accepted = parse_region_label_plan(
        '{"is_semantic_part":true,"region_kind":"physical_component",'
        '"semantic_name":"furniture seat","confidence":0.91,'
        '"matches_target_region":true}',
        allowed_parts={seat.semantic_name: seat},
    )
    invented = parse_region_label_plan(
        '{"is_semantic_part":true,"region_kind":"physical_component",'
        '"semantic_name":"imagined_motor","confidence":0.99,'
        '"matches_target_region":true}',
        allowed_parts={seat.semantic_name: seat},
    )

    assert accepted.semantic_name == "furniture_seat"
    assert invented.semantic_name is None
    assert invented.diagnostics["status"] == "label_outside_inventory"


def test_region_label_parser_rejects_surface_marks_even_with_high_confidence() -> None:
    parsed = parse_region_ownership_plan(
        '{"is_semantic_part":false,'
        '"region_kind":"incidental_surface_pattern",'
        '"confidence":0.99}',
    )

    assert parsed.is_semantic_part is False
    assert parsed.region_kind == "incidental_surface_pattern"
    assert parsed.diagnostics["status"] == "nonsemantic_region"


def test_region_query_image_preserves_target_and_dims_context() -> None:
    image = Image.new("RGB", (100, 100), (200, 100, 50))
    root = np.zeros((100, 100), dtype=bool)
    root[10:90, 10:90] = True
    region = np.zeros((100, 100), dtype=bool)
    region[45:55, 45:55] = True

    query = np.asarray(
        make_region_query_image(image, root_mask=root, region_mask=region)
    )

    assert query.shape == (192, 388, 3)
    assert np.any(np.all(query == np.asarray([200, 100, 50]), axis=2))
    assert np.any(np.all(query == np.asarray([127, 127, 127]), axis=2))
    assert np.any(np.all(query == np.asarray([255, 40, 40]), axis=2))
    left_target_pixels = np.count_nonzero(
        np.all(query[:, :192] == np.asarray([200, 100, 50]), axis=2)
    )
    right_target_pixels = np.count_nonzero(
        np.all(query[:, 196:] == np.asarray([200, 100, 50]), axis=2)
    )
    assert right_target_pixels > left_target_pixels


def test_root_query_image_keeps_scene_context_and_enlarges_target() -> None:
    pixels = np.full((100, 160, 3), (30, 80, 150), dtype=np.uint8)
    pixels[35:65, 70:90] = np.asarray((210, 120, 40), dtype=np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    root = np.zeros((100, 160), dtype=bool)
    root[35:65, 70:90] = True

    query = np.asarray(make_root_query_image(image, root_mask=root))

    assert query.shape == (256, 516, 3)
    assert np.any(np.all(query == np.asarray([255, 40, 40]), axis=2))
    left_target_pixels = np.count_nonzero(
        np.all(query[:, :256] == np.asarray([210, 120, 40]), axis=2)
    )
    right_target_pixels = np.count_nonzero(
        np.all(query[:, 260:] == np.asarray([210, 120, 40]), axis=2)
    )
    assert left_target_pixels > 0
    assert right_target_pixels > left_target_pixels


def test_region_label_prompt_includes_physical_definition() -> None:
    seat = PartPrompt(
        "furniture_seat",
        ("seat",),
        planner_description="the horizontal load-bearing sitting surface",
        planner_exclusions=("furniture_backrest",),
    )

    prompt = build_region_label_prompt(
        object_label="chair",
        domain_name="furniture",
        parts=(seat,),
        region_kind="panel",
    )

    assert "horizontal load-bearing sitting surface" in prompt
    assert "furniture_backrest" in prompt

    ownership_prompt = build_region_ownership_prompt(
        object_label="chair",
        domain_name="furniture",
        proposal_kind="panel",
    )
    assert "background_through_opening" in ownership_prompt
    assert "background seen between chair" in ownership_prompt


def test_region_batch_prompt_image_and_parser_are_closed_world() -> None:
    stock = _part("tool_stock", maximum_instances=1, semantic_parent="tool")
    grip = _part("tool_grip", maximum_instances=1, semantic_parent="tool")
    root = np.ones((80, 100), dtype=bool)
    first = np.zeros_like(root)
    first[10:45, 8:42] = True
    second = np.zeros_like(root)
    second[35:72, 60:88] = True

    query = make_region_batch_query_image(
        Image.new("RGB", (100, 80), "white"),
        root_mask=root,
        regions=(("R1", first), ("R2", second)),
    )
    assert query.size == (646, 176)
    assert np.asarray(query).std() > 0

    prompt = build_region_batch_label_prompt(
        object_label="rifle",
        domain_name="tool",
        parts=(stock, grip),
        region_specs=(("R1", "panel"), ("R2", "panel")),
    )
    assert "R1" in prompt and "R2" in prompt
    assert "must not invent, resize, merge, or split" in prompt

    response = json.dumps(
        {
            "regions": [
                {
                    "region_id": "R1",
                    "semantic_name": "tool_stock",
                    "region_kind": "physical_component",
                    "confidence": 0.96,
                    "matches_target_region": True,
                },
                {
                    "region_id": "R2",
                    "semantic_name": "tool_grip",
                    "region_kind": "incidental_surface_pattern",
                    "confidence": 0.99,
                    "matches_target_region": True,
                },
                {
                    "region_id": "R9",
                    "semantic_name": "invented_part",
                    "region_kind": "physical_component",
                    "confidence": 0.99,
                    "matches_target_region": True,
                },
            ]
        }
    )
    parsed, diagnostics = parse_region_batch_label_plan(
        response,
        region_ids=("R1", "R2"),
        allowed_parts={stock.semantic_name: stock, grip.semantic_name: grip},
        minimum_confidence=0.65,
    )
    assert parsed["R1"].semantic_name == "tool_stock"
    assert parsed["R2"].semantic_name is None
    assert parsed["R2"].diagnostics["status"] == "nonphysical_region_kind"
    assert diagnostics["accepted_region_count"] == 1
    assert diagnostics["unknown_region_id_count"] == 1


def test_region_batch_parser_recovers_semantic_name_in_kind_field() -> None:
    wheel = _part("vehicle_wheel", semantic_parent="vehicle")
    response = json.dumps(
        {
            "regions": [
                {
                    "region_id": "R1",
                    "semantic_name": "vehicle_wheel",
                    "region_kind": "vehicle_wheel",
                    "confidence": 0.95,
                    "matches_target_region": True,
                }
            ]
        }
    )

    parsed, diagnostics = parse_region_batch_label_plan(
        response,
        region_ids=("R1",),
        allowed_parts={wheel.semantic_name: wheel},
        minimum_confidence=0.65,
    )

    assert parsed["R1"].semantic_name == "vehicle_wheel"
    assert parsed["R1"].entity_kind == "physical_component"
    assert parsed["R1"].diagnostics["region_kind_recovered"] is True
    assert diagnostics["accepted_region_count"] == 1


def test_global_assignment_prevents_one_region_from_receiving_two_labels() -> None:
    shape = (100, 100)
    root = np.ones(shape, dtype=bool)
    shared = np.zeros(shape, dtype=bool)
    shared[20:70, 20:70] = True
    second = np.zeros(shape, dtype=bool)
    second[20:70, 72:92] = True
    seat = _part("furniture_seat", maximum_instances=1)
    back = _part("furniture_backrest", maximum_instances=1)
    plans = (
        PlannedPart("furniture_seat", (15, 15, 75, 75), 0.9),
        PlannedPart("furniture_backrest", (15, 15, 75, 75), 0.9),
    )
    regions = (
        PartRegion(shared, "visual", 0.9, "shared"),
        PartRegion(second, "visual", 0.9, "second"),
    )

    assignments, diagnostics = assign_plans_to_regions(
        plans,
        regions,
        allowed_parts={
            seat.semantic_name: seat,
            back.semantic_name: back,
        },
        root_mask=root,
    )

    assert len(assignments) == 1
    assert assignments[0].region_index == 0
    assert diagnostics["unmatched_plan_count"] == 1


def test_global_assignment_does_not_relabel_established_semantics() -> None:
    shape = (80, 80)
    root = np.ones(shape, dtype=bool)
    stock_mask = np.zeros(shape, dtype=bool)
    stock_mask[20:60, 10:35] = True
    stock = _part("tool_stock", maximum_instances=1, semantic_parent="tool")
    barrel = _part("tool_barrel", maximum_instances=1, semantic_parent="tool")
    plans = (PlannedPart("tool_barrel", (5, 15, 40, 65), 0.9),)
    regions = (
        PartRegion(
            stock_mask,
            "trusted",
            0.9,
            "stock",
            semantic_name="tool_stock",
            generic=False,
        ),
    )

    assignments, diagnostics = assign_plans_to_regions(
        plans,
        regions,
        allowed_parts={
            stock.semantic_name: stock,
            barrel.semantic_name: barrel,
        },
        root_mask=root,
    )

    assert assignments == ()
    assert diagnostics["feasible_pair_count"] == 0


def test_semantic_support_can_confirm_a_coarse_vlm_box() -> None:
    shape = (100, 100)
    root = np.ones(shape, dtype=bool)
    mask = np.zeros(shape, dtype=bool)
    mask[60:90, 60:90] = True
    stock = _part("tool_stock", maximum_instances=1, semantic_parent="tool")
    plans = (PlannedPart("tool_stock", (35, 35, 95, 95), 0.9),)
    regions = (
        PartRegion(
            mask,
            "trusted",
            0.9,
            "stock",
            semantic_name="tool_stock",
            generic=False,
        ),
    )

    assignments, _ = assign_plans_to_regions(
        plans,
        regions,
        allowed_parts={stock.semantic_name: stock},
        root_mask=root,
    )

    assert len(assignments) == 1
    assert assignments[0].semantic_support is True


class _FakePlanner:
    backend_id = "fake-vlm"

    def generate_response(self, image: Image.Image, prompt: str) -> str:
        if "Audit one red-outlined candidate" in prompt:
            return json.dumps(
                {
                    "is_semantic_part": True,
                    "region_kind": "physical_component",
                    "confidence": 0.93,
                }
            )
        if "Classify one red-outlined candidate region" in prompt:
            return json.dumps(
                {
                    "semantic_name": "furniture_seat",
                    "confidence": 0.92,
                    "matches_target_region": True,
                }
            )
        assert "furniture_" in prompt
        return json.dumps(
            {
                "coordinate_system": "normalized_1000",
                "parts": [
                    {
                        "semantic_name": "furniture_seat",
                        "bbox_2d": [100, 100, 900, 500],
                        "confidence": 0.9,
                        "visible": True,
                    },
                    {
                        "semantic_name": "furniture_leg",
                        "bbox_2d": [150, 450, 300, 950],
                        "confidence": 0.86,
                        "visible": True,
                    },
                ],
            }
        )


class _CountingBoxPlanner:
    backend_id = "counting-vlm"

    def __init__(self) -> None:
        self.box_query_prompts: list[str] = []

    def generate_response(self, image: Image.Image, prompt: str) -> str:
        if "Return JSON only, using exactly this schema" in prompt:
            self.box_query_prompts.append(prompt)
            return json.dumps(
                {"coordinate_system": "normalized_1000", "parts": []}
            )
        return json.dumps({"regions": []})


def _box_segmenter(
    image: Image.Image, detections: list[object]
) -> list[SegmentProposal]:
    proposals = []
    for detection in detections:
        mask = np.zeros((image.height, image.width), dtype=bool)
        x0, y0, x1, y1 = detection.box_xyxy
        mask[y0:y1, x0:x1] = True
        proposals.append(SegmentProposal(mask, 0.92))
    return proposals


def test_vlm_boxes_become_root_constrained_nonfinal_candidates() -> None:
    shape = (100, 100)
    root_mask = np.zeros(shape, dtype=bool)
    root_mask[10:90, 10:90] = True
    root = MaskCandidate(
        "furniture",
        "furniture",
        root_mask,
        0.98,
        "root",
        prompt="stool",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
        },
    )
    parts = (
        _part("furniture_seat", maximum_instances=1),
        _part("furniture_leg", maximum_instances=4),
    )
    domain = DomainPrompt(
        name="furniture",
        root_prompts=("stool",),
        parts=parts,
    )
    generator = VlmPartGenerator(
        _FakePlanner(),
        _box_segmenter,
        config=VlmPartConfig(
            minimum_raw_root_containment=0.1,
            allow_direct_sam_regions=True,
            use_box_planner_queries=True,
        ),
    )

    result = generator.generate(
        Image.new("RGB", (100, 100), "white"),
        [root],
        {"furniture": domain},
    )

    assert {item.semantic_name for item in result.candidates} == {
        "furniture_seat",
        "furniture_leg",
    }
    assert all(np.all(item.mask <= root_mask) for item in result.candidates)
    assert all(item.metadata["ground_truth_used"] is False for item in result.candidates)
    assert result.diagnostics["final_part_ids_assigned_by_backend"] is False
    assert result.diagnostics["ground_truth_used"] is False


def test_bulk_box_planning_uses_bounded_semantic_batches() -> None:
    root_mask = np.ones((100, 100), dtype=bool)
    root = MaskCandidate(
        "tool",
        "tool",
        root_mask,
        0.98,
        "root",
        prompt="multi-part tool",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
        },
    )
    parts = tuple(_part(f"tool_part_{index}") for index in range(7))
    domain = DomainPrompt("tool", ("multi-part tool",), parts)
    planner = _CountingBoxPlanner()
    generator = VlmPartGenerator(
        planner,
        _box_segmenter,
        config=VlmPartConfig(
            use_region_label_queries=False,
            use_box_planner_queries=True,
            use_per_semantic_queries=False,
            query_established_semantics=True,
            planner_batch_size=3,
            maximum_planner_queries=3,
            maximum_total_planner_queries=3,
        ),
    )

    result = generator.generate(
        Image.new("RGB", (100, 100), "white"),
        [root],
        {"tool": domain},
    )

    assert len(planner.box_query_prompts) == 3
    assert [
        len(row["requested_semantics"])
        for row in result.diagnostics["roots"][0]["plan_queries"]
    ] == [3, 3, 1]


def test_vlm_can_name_generic_visual_panels_without_overwriting_semantics() -> None:
    shape = (100, 100)
    root_mask = np.zeros(shape, dtype=bool)
    root_mask[5:95, 5:95] = True
    panel_mask = np.zeros(shape, dtype=bool)
    panel_mask[15:50, 15:85] = True
    root = MaskCandidate(
        "furniture",
        "furniture",
        root_mask,
        0.98,
        "root",
        prompt="stool",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
        },
    )
    panel = MaskCandidate(
        "furniture_visual_panel_01",
        "furniture",
        panel_mask,
        0.9,
        "visual",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1/visual:1",
            "parent_candidate_key": "root:1",
            "visual_region": True,
            "generic_visual_region": True,
            "visual_region_kind": "panel",
        },
    )
    seat = _part("furniture_seat", maximum_instances=1)
    domain = DomainPrompt("furniture", ("stool",), (seat,))
    generator = VlmPartGenerator(_FakePlanner(), _box_segmenter)

    result = generator.generate(
        Image.new("RGB", (100, 100), "white"),
        [root],
        {"furniture": domain},
        existing_candidates=[root, panel],
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].semantic_name == "furniture_seat"
    assert np.array_equal(result.candidates[0].mask, panel_mask)


def test_region_label_reuses_prior_physicality_evidence() -> None:
    shape = (80, 80)
    root_mask = np.ones(shape, dtype=bool)
    panel_mask = np.zeros(shape, dtype=bool)
    panel_mask[18:62, 12:68] = True
    root = MaskCandidate(
        "furniture",
        "furniture",
        root_mask,
        0.99,
        "root",
        prompt="stool",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
        },
    )
    panel = MaskCandidate(
        "furniture_visual_panel_01",
        "furniture",
        panel_mask,
        0.92,
        "visual",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1/visual:1",
            "parent_candidate_key": "root:1",
            "visual_region": True,
            "generic_visual_region": True,
            "visual_region_kind": "panel",
            "vlm_physicality_audit": {
                "decision": "physical_supported",
                "label": "physical_component",
                "confidence": 0.95,
                "planner_backend": "earlier-audit",
            },
        },
    )
    seat = _part("furniture_seat", maximum_instances=1)
    domain = DomainPrompt("furniture", ("stool",), (seat,))
    generator = VlmPartGenerator(
        _FakePlanner(),
        _box_segmenter,
        config=VlmPartConfig(maximum_planner_queries=1),
    )

    result = generator.generate(
        Image.new("RGB", shape, "white"),
        [root],
        {"furniture": domain},
        existing_candidates=[root, panel],
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].semantic_name == "furniture_seat"
    row = result.diagnostics["roots"][0]["region_label_queries"][0]
    assert row["ownership_evidence_reused"] is True
    assert row["ownership_parse"]["status"] == "reused_physicality_audit"
    assert result.diagnostics["total_planner_query_count"] == 1


def test_region_label_prioritizes_uncovered_semantics() -> None:
    shape = (80, 80)
    root_mask = np.ones(shape, dtype=bool)
    strong_mask = np.zeros(shape, dtype=bool)
    strong_mask[8:22, 5:70] = True
    weak_mask = np.zeros(shape, dtype=bool)
    weak_mask[25:38, 5:70] = True
    generic_mask = np.zeros(shape, dtype=bool)
    generic_mask[42:72, 12:68] = True
    root = MaskCandidate(
        "tool",
        "tool",
        root_mask,
        0.99,
        "root",
        prompt="rifle",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
        },
    )
    strong = MaskCandidate(
        "tool_barrel",
        "tool",
        strong_mask,
        0.9,
        "grounded",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1/barrel:1",
            "parent_candidate_key": "root:1",
        },
    )
    weak = MaskCandidate(
        "tool_barrel",
        "tool",
        weak_mask,
        0.99,
        "visual",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1/visual:weak",
            "parent_candidate_key": "root:1",
            "visual_region": True,
            "generic_visual_region": False,
            "visual_region_kind": "panel",
            "semantic_reranked": True,
            "semantic_rerank_probability": 0.2,
            "semantic_rerank_margin": 0.01,
            "sam_quality": 0.99,
        },
    )
    generic = MaskCandidate(
        "tool_visual_panel_01",
        "tool",
        generic_mask,
        0.75,
        "visual",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1/visual:generic",
            "parent_candidate_key": "root:1",
            "visual_region": True,
            "generic_visual_region": True,
            "visual_region_kind": "panel",
            "sam_quality": 0.75,
        },
    )
    barrel = _part("tool_barrel", maximum_instances=1, semantic_parent="tool")
    handguard = _part(
        "tool_handguard", maximum_instances=1, semantic_parent="tool"
    )
    domain = DomainPrompt("tool", ("rifle",), (barrel, handguard))
    generator = VlmPartGenerator(
        _RegionLabelPlanner(),
        _box_segmenter,
        config=VlmPartConfig(maximum_planner_queries=2),
    )

    result = generator.generate(
        Image.new("RGB", shape, "white"),
        [root],
        {"tool": domain},
        existing_candidates=[root, strong, weak, generic],
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].semantic_name == "tool_handguard"
    assert np.array_equal(result.candidates[0].mask, generic_mask)


class _BrokenPlanner:
    backend_id = "broken-vlm"

    def generate_response(self, image: Image.Image, prompt: str) -> str:
        raise RuntimeError("synthetic failure")


class _RegionLabelPlanner:
    backend_id = "fake-region-label-vlm"

    def generate_response(self, image: Image.Image, prompt: str) -> str:
        if "Audit one red-outlined candidate" in prompt:
            return json.dumps(
                {
                    "is_semantic_part": True,
                    "region_kind": "physical_component",
                    "confidence": 0.94,
                }
            )
        if "Classify one red-outlined candidate region" in prompt:
            return json.dumps(
                {
                    "semantic_name": "tool_handguard",
                    "confidence": 0.92,
                    "matches_target_region": True,
                }
            )
        return json.dumps(
            {"coordinate_system": "normalized_1000", "parts": []}
        )


class _DynamicInventoryPlanner:
    backend_id = "fake-dynamic-inventory-vlm"

    def generate_response(self, image: Image.Image, prompt: str) -> str:
        if "Identify the single visible physical object" in prompt:
            return json.dumps(
                {
                    "object": "wireless charger",
                    "object_confidence": 0.97,
                    "candidate_label_matches": True,
                }
            )
        if "missing from the existing inventory" in prompt:
            return json.dumps(
                {
                    "parts": [
                        {
                            "name": "charging coil",
                            "description": "separate circular charging coil",
                            "confidence": 0.96,
                            "visible": True,
                            "physical": True,
                            "detail": False,
                            "maximum_instances": 1,
                        }
                    ]
                }
            )
        if "Audit one red-outlined candidate" in prompt:
            return json.dumps(
                {
                    "is_semantic_part": True,
                    "region_kind": "physical_component",
                    "confidence": 0.97,
                }
            )
        if "Classify one red-outlined candidate region" in prompt:
            return json.dumps(
                {
                    "semantic_name": "device_dynamic_charging_coil",
                    "confidence": 0.95,
                    "matches_target_region": True,
                }
            )
        return json.dumps({"coordinate_system": "normalized_1000", "parts": []})


class _BatchRegionLabelPlanner:
    backend_id = "fake-batch-region-label-vlm"

    def generate_response(self, image: Image.Image, prompt: str) -> str:
        assert "Classify several numbered candidate regions" in prompt
        return json.dumps(
            {
                "regions": [
                    {
                        "region_id": "R1",
                        "semantic_name": "tool_stock",
                        "region_kind": "physical_component",
                        "confidence": 0.97,
                        "matches_target_region": True,
                    },
                    {
                        "region_id": "R2",
                        "semantic_name": "tool_grip",
                        "region_kind": "physical_component",
                        "confidence": 0.95,
                        "matches_target_region": True,
                    },
                ]
            }
        )


def test_batch_region_labels_use_one_query_for_multiple_independent_masks() -> None:
    root_mask = np.ones((100, 100), dtype=bool)
    stock_mask = np.zeros_like(root_mask)
    stock_mask[12:46, 8:42] = True
    grip_mask = np.zeros_like(root_mask)
    grip_mask[50:88, 62:88] = True
    root = MaskCandidate(
        "tool",
        "tool",
        root_mask,
        0.99,
        "root",
        prompt="rifle",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
        },
    )

    def visual(key: str, mask: np.ndarray) -> MaskCandidate:
        return MaskCandidate(
            f"tool_visual_{key}",
            "tool",
            mask,
            0.94,
            "visual",
            metadata={
                "root_origin": "test",
                "root_index": 1,
                "candidate_key": f"root:1/{key}",
                "parent_candidate_key": "root:1",
                "visual_region": True,
                "generic_visual_region": True,
                "visual_region_kind": "panel",
                "sam_quality": 0.94,
            },
        )

    stock = _part("tool_stock", maximum_instances=1, semantic_parent="tool")
    grip = _part("tool_grip", maximum_instances=1, semantic_parent="tool")
    domain = DomainPrompt("tool", ("rifle",), (stock, grip))
    generator = VlmPartGenerator(
        _BatchRegionLabelPlanner(),
        _box_segmenter,
        config=VlmPartConfig(
            use_batched_region_label_queries=True,
            region_label_batch_size=8,
            maximum_region_label_queries=8,
            maximum_planner_queries=1,
            maximum_total_planner_queries=1,
            maximum_roots=1,
        ),
    )
    result = generator.generate(
        Image.new("RGB", (100, 100), "white"),
        [root],
        {"tool": domain},
        existing_candidates=[
            root,
            visual("stock", stock_mask),
            visual("grip", grip_mask),
        ],
    )

    assert {candidate.semantic_name for candidate in result.candidates} == {
        "tool_stock",
        "tool_grip",
    }
    assert result.diagnostics["total_planner_query_count"] == 1
    assert result.diagnostics["roots"][0]["region_label_query_count"] == 1


def test_dynamic_inventory_requires_an_audited_independent_region() -> None:
    root_mask = np.ones((80, 80), dtype=bool)
    coil_mask = np.zeros((80, 80), dtype=bool)
    coil_mask[20:55, 22:58] = True
    root = MaskCandidate(
        "device",
        "device",
        root_mask,
        0.99,
        "root",
        prompt="wireless charger",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
        },
    )
    region = MaskCandidate(
        "device_visual_panel_01",
        "device",
        coil_mask,
        0.95,
        "visual",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1/visual:1",
            "parent_candidate_key": "root:1",
            "visual_region": True,
            "generic_visual_region": True,
            "visual_region_kind": "panel",
            "sam_quality": 0.95,
        },
    )
    domain = DomainPrompt("device", ("device",), ())
    generator = VlmPartGenerator(
        _DynamicInventoryPlanner(),
        _box_segmenter,
        config=VlmPartConfig(
            use_dynamic_inventory=True,
            maximum_planner_queries=4,
            maximum_total_planner_queries=4,
        ),
    )

    result = generator.generate(
        Image.new("RGB", (80, 80), "white"),
        [root],
        {"device": domain},
        existing_candidates=[root, region],
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].semantic_name == "device_dynamic_charging_coil"
    assert np.array_equal(result.candidates[0].mask, coil_mask)
    assert result.diagnostics["dynamic_inventory_query_count"] == 1
    assert result.diagnostics["total_planner_query_count"] == 3
    assert result.diagnostics["ground_truth_used"] is False


def test_region_labels_obey_instance_cap_and_keep_better_boundary() -> None:
    root_mask = np.ones((80, 80), dtype=bool)
    first_mask = np.zeros((80, 80), dtype=bool)
    first_mask[10:35, 10:40] = True
    second_mask = np.zeros((80, 80), dtype=bool)
    second_mask[42:70, 10:40] = True
    root = MaskCandidate(
        "tool",
        "tool",
        root_mask,
        0.99,
        "root",
        prompt="rifle",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
        },
    )
    regions = [
        MaskCandidate(
            f"tool_visual_panel_{index:02d}",
            "tool",
            mask,
            quality,
            "visual",
            metadata={
                "root_origin": "test",
                "root_index": 1,
                "candidate_key": f"root:1/visual:{index}",
                "parent_candidate_key": "root:1",
                "visual_region": True,
                "generic_visual_region": True,
                "visual_region_kind": "panel",
                "sam_quality": quality,
            },
        )
        for index, mask, quality in (
            (1, first_mask, 0.96),
            (2, second_mask, 0.72),
        )
    ]
    handguard = _part(
        "tool_handguard",
        maximum_instances=1,
        semantic_parent="tool",
    )
    domain = DomainPrompt("tool", ("rifle",), (handguard,))
    generator = VlmPartGenerator(
        _RegionLabelPlanner(),
        _box_segmenter,
        config=VlmPartConfig(maximum_planner_queries=2),
    )

    result = generator.generate(
        Image.new("RGB", (80, 80), "white"),
        [root],
        {"tool": domain},
        existing_candidates=[root, *regions],
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].semantic_name == "tool_handguard"
    assert np.array_equal(result.candidates[0].mask, first_mask)
    assert result.candidates[0].metadata["vlm_region_label"] is True


def test_region_label_can_corroborate_an_existing_instance_at_the_cap() -> None:
    root_mask = np.ones((80, 80), dtype=bool)
    existing_mask = np.zeros((80, 80), dtype=bool)
    existing_mask[18:52, 12:48] = True
    audited_mask = np.zeros((80, 80), dtype=bool)
    audited_mask[20:50, 10:50] = True
    root = MaskCandidate(
        "tool",
        "tool",
        root_mask,
        0.99,
        "root",
        prompt="rifle",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
        },
    )
    existing = MaskCandidate(
        "tool_handguard",
        "tool",
        existing_mask,
        0.72,
        "grounded",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1/handguard:1",
            "parent_candidate_key": "root:1",
        },
    )
    audited_region = MaskCandidate(
        "tool_visual_panel_01",
        "tool",
        audited_mask,
        0.94,
        "visual",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1/visual:1",
            "parent_candidate_key": "root:1",
            "visual_region": True,
            "generic_visual_region": True,
            "visual_region_kind": "panel",
            "sam_quality": 0.94,
        },
    )
    handguard = _part(
        "tool_handguard",
        maximum_instances=1,
        semantic_parent="tool",
    )
    domain = DomainPrompt("tool", ("rifle",), (handguard,))
    generator = VlmPartGenerator(
        _RegionLabelPlanner(),
        _box_segmenter,
        config=VlmPartConfig(maximum_planner_queries=2),
    )

    result = generator.generate(
        Image.new("RGB", (80, 80), "white"),
        [root],
        {"tool": domain},
        existing_candidates=[root, existing, audited_region],
    )

    assert len(result.candidates) == 1
    assert np.array_equal(result.candidates[0].mask, audited_mask)
    assert result.candidates[0].metadata["vlm_region_corroborates_existing"] is True
    assert (
        result.candidates[0].metadata["vlm_region_corroborated_candidate_key"]
        == "root:1/handguard:1"
    )


def test_region_label_gate_rejects_thin_exterior_fragments() -> None:
    root_mask = np.zeros((100, 100), dtype=bool)
    root_mask[10:90, 10:90] = True
    fragment_mask = np.zeros((100, 100), dtype=bool)
    fragment_mask[12:88, 10:12] = True
    root = MaskCandidate(
        "furniture",
        "furniture",
        root_mask,
        0.99,
        "root",
        prompt="stool",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
        },
    )
    fragment = MaskCandidate(
        "furniture_visual_panel_01",
        "furniture",
        fragment_mask,
        0.95,
        "visual",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1/visual:1",
            "parent_candidate_key": "root:1",
            "visual_region": True,
            "generic_visual_region": True,
            "visual_region_kind": "panel",
        },
    )
    seat = _part("furniture_seat", maximum_instances=1)
    domain = DomainPrompt("furniture", ("stool",), (seat,))
    generator = VlmPartGenerator(_RegionLabelPlanner(), _box_segmenter)

    result = generator.generate(
        Image.new("RGB", (100, 100), "white"),
        [root],
        {"furniture": domain},
        existing_candidates=[root, fragment],
    )

    assert result.candidates == ()
    assert result.diagnostics["rejection_counts"][
        "ownership_geometry_rejected"
    ] == 1


def test_axis_structure_consensus_is_not_treated_as_a_weak_label() -> None:
    mask = np.ones((20, 20), dtype=bool)
    candidate = MaskCandidate(
        "tool_prop_grip",
        "tool_prop",
        mask,
        0.7,
        "visual",
        metadata={
            "visual_region": True,
            "semantic_reranked": True,
            "semantic_axis_structure_rescue": True,
            "semantic_rerank_probability": 0.15,
            "semantic_rerank_margin": 0.03,
        },
    )
    generator = VlmPartGenerator(_RegionLabelPlanner(), _box_segmenter)

    assert generator._is_weak_semantic_region(candidate) is False


def test_global_query_budget_bounds_multi_object_scenes() -> None:
    roots = []
    regions = []
    for index, x0 in enumerate((5, 55), start=1):
        root_mask = np.zeros((60, 110), dtype=bool)
        root_mask[5:55, x0 : x0 + 45] = True
        panel_mask = np.zeros((60, 110), dtype=bool)
        panel_mask[15:45, x0 + 8 : x0 + 35] = True
        roots.append(
            MaskCandidate(
                "tool",
                "tool",
                root_mask,
                0.98,
                "root",
                prompt="game prop",
                metadata={
                    "root_origin": "scene",
                    "root_index": index,
                    "candidate_key": f"root:{index}",
                },
            )
        )
        regions.append(
            MaskCandidate(
                f"tool_visual_panel_{index:02d}",
                "tool",
                panel_mask,
                0.9,
                "visual",
                metadata={
                    "root_origin": "scene",
                    "root_index": index,
                    "candidate_key": f"root:{index}/visual:1",
                    "parent_candidate_key": f"root:{index}",
                    "visual_region": True,
                    "generic_visual_region": True,
                    "visual_region_kind": "panel",
                },
            )
        )
    handguard = _part("tool_handguard", maximum_instances=1, semantic_parent="tool")
    domain = DomainPrompt("tool", ("game prop",), (handguard,))
    generator = VlmPartGenerator(
        _RegionLabelPlanner(),
        _box_segmenter,
        config=VlmPartConfig(
            maximum_planner_queries=2,
            maximum_total_planner_queries=2,
            maximum_roots=2,
        ),
    )

    result = generator.generate(
        Image.new("RGB", (110, 60), "white"),
        roots,
        {"tool": domain},
        existing_candidates=[*roots, *regions],
    )

    assert len(result.candidates) == 2
    assert result.diagnostics["total_planner_query_count"] == 2
    assert result.diagnostics["total_query_budget_exhausted"] is True
    assert result.diagnostics["root_count"] == 2


def test_planner_failure_does_not_destroy_base_pipeline() -> None:
    root_mask = np.ones((40, 40), dtype=bool)
    root = MaskCandidate(
        "furniture",
        "furniture",
        root_mask,
        0.9,
        "root",
        metadata={"root_origin": "test", "root_index": 1},
    )
    part = _part("furniture_seat")
    domain = DomainPrompt("furniture", ("chair",), (part,))
    generator = VlmPartGenerator(
        _BrokenPlanner(),
        _box_segmenter,
        config=VlmPartConfig(use_box_planner_queries=True),
    )

    result = generator.generate(
        Image.new("RGB", (40, 40), "white"),
        [root],
        {"furniture": domain},
    )

    assert result.candidates == ()
    assert result.diagnostics["rejection_counts"]["planner_error"] == 1
