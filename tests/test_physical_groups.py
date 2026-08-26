import cv2
import numpy as np
from PIL import Image

from hpid_split.fusion import MaskCandidate
from hpid_split.instances import PartInstance
from hpid_split.physical_groups import (
    _candidate_three_stage_verification,
    _firearm_handguard_from_structure_and_material,
    _refine_inventory_boundaries,
    _semantic_seeded_profile_masks,
    build_physical_groups,
)


def _record(index: int, semantic: str, parent: str) -> PartInstance:
    return PartInstance(
        part_id=f"{parent}/{semantic}/center/{index:02d}",
        semantic_name=semantic,
        semantic_parent=parent,
        instance_index=index,
        side="center",
        bbox_xyxy=(index * 2, 1, index * 2 + 2, 5),
        centroid_xy=(index * 2 + 0.5, 2.5),
        area_px=8,
    )


def _strip_map(count: int) -> np.ndarray:
    output = np.zeros((6, count * 3 + 1), dtype=np.uint16)
    for index in range(1, count + 1):
        output[1:5, index * 3 - 2 : index * 3] = index
    return output


def _knife_root(shape: tuple[int, int]) -> MaskCandidate:
    return MaskCandidate(
        semantic_name="tool_prop",
        semantic_parent="tool_prop",
        mask=np.ones(shape, dtype=bool),
        score=0.8,
        source="test/root",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
            "asset_router_candidate_labels": ["knife"],
            "asset_router_candidate_domains": ["tool_prop"],
        },
    )


def _firearm_root(shape: tuple[int, int]) -> MaskCandidate:
    return MaskCandidate(
        semantic_name="tool_prop",
        semantic_parent="tool_prop",
        mask=np.ones(shape, dtype=bool),
        score=0.8,
        source="test/root",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
            "selected_part_profile": "firearm",
        },
    )


def _phone_root(shape: tuple[int, int]) -> MaskCandidate:
    return MaskCandidate(
        semantic_name="device",
        semantic_parent="device",
        mask=np.ones(shape, dtype=bool),
        score=0.9,
        source="test/root",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
            "selected_part_profile": "phone",
        },
    )


def test_knife_inventory_exports_exactly_three_editable_groups() -> None:
    instance_map = _strip_map(5)
    records = [
        _record(1, "tool_prop", "tool_prop"),
        _record(2, "tool_prop_blade", "tool_prop_body"),
        _record(3, "tool_prop_guard", "tool_prop_body"),
        _record(4, "tool_prop_wrap", "tool_prop_handle"),
        _record(5, "tool_prop_visual_panel_01", "tool_prop"),
    ]

    result = build_physical_groups(
        instance_map,
        records,
        candidates=[_knife_root(instance_map.shape)],
    )

    assert {group.semantic_name for group in result.groups} == {
        "tool_prop_blade",
        "tool_prop_handle",
        "tool_prop_wrap",
    }
    assert result.diagnostics["knife_inventory_complete"] is True
    handle = next(
        group for group in result.groups if group.semantic_name == "tool_prop_handle"
    )
    assert len(handle.member_part_ids) == 3
    assert all(record.group_id for record in result.records)


def test_character_body_and_garment_use_hierarchical_groups() -> None:
    instance_map = _strip_map(6)
    records = [
        _record(1, "character", "character"),
        _record(2, "character_arm", "character_body"),
        _record(3, "character_hand", "character_body"),
        _record(4, "character_shirt", "character_clothing"),
        _record(5, "character_sleeve", "character_clothing"),
        _record(6, "character_hair", "character_head"),
    ]

    result = build_physical_groups(instance_map, records)

    assert {group.semantic_name for group in result.groups} == {
        "character_body",
        "character_upper_garment",
        "character_hair",
    }
    body = next(
        group for group in result.groups if group.semantic_name == "character_body"
    )
    garment = next(
        group
        for group in result.groups
        if group.semantic_name == "character_upper_garment"
    )
    assert len(body.member_part_ids) == 3
    assert len(garment.member_part_ids) == 2


def test_character_layered_clothing_keeps_inner_outer_and_lower_groups() -> None:
    shape = (220, 140)
    instance_map = np.zeros(shape, dtype=np.uint16)
    instance_map[18:196, 35:105] = 1
    instance_map[12:62, 30:110] = 2
    instance_map[28:70, 43:97] = 3
    instance_map[72:122, 52:88] = 4
    instance_map[74:137, 92:116] = 5
    instance_map[122:158, 43:98] = 6
    instance_map[158:198, 45:62] = 7
    instance_map[158:198, 78:95] = 8
    instance_map[196:214, 39:101] = 9
    image = np.full((*shape, 3), 244, dtype=np.uint8)
    image[instance_map == 1] = (222, 168, 145)
    image[instance_map == 2] = (55, 67, 73)
    image[instance_map == 3] = (224, 170, 146)
    image[instance_map == 4] = (235, 234, 218)
    image[instance_map == 5] = (151, 190, 160)
    image[instance_map == 6] = (83, 91, 102)
    image[np.isin(instance_map, [7, 8])] = (225, 166, 143)
    image[instance_map == 9] = (91, 138, 122)

    def record(index: int, semantic: str, parent: str = "character") -> PartInstance:
        mask = instance_map == index
        ys, xs = np.nonzero(mask)
        return PartInstance(
            part_id=f"{parent}/{semantic}/center/{index:02d}",
            semantic_name=semantic,
            semantic_parent=parent,
            instance_index=index,
            side="center",
            bbox_xyxy=(int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)),
            centroid_xy=(float(xs.mean()), float(ys.mean())),
            area_px=int(mask.sum()),
        )

    records = [
        record(1, "character"),
        record(2, "character_hair"),
        record(3, "character_head"),
        record(4, "character_upper_clothing"),
        record(5, "character_sleeve", "character_upper_clothing"),
        record(6, "character_visual_panel_03"),
        record(7, "character_leg"),
        record(8, "character_leg"),
        record(9, "character_shoe"),
    ]

    result = build_physical_groups(
        instance_map,
        records,
        image=Image.fromarray(image),
    )
    by_name = {group.semantic_name: group for group in result.groups}

    assert result.group_map[90, 70] == by_name["character_inner_top"].group_index
    assert result.group_map[100, 104] == by_name["character_outer_garment"].group_index
    assert result.group_map[140, 70] == by_name["character_lower_garment"].group_index
    assert result.group_map[45, 70] == by_name["character_body"].group_index
    assert "character_headwear" not in by_name
    assert result.diagnostics["character_surface_grouping"][
        "layered_upper_garment"
    ]["detected"] is True
    assert np.array_equal(result.group_map > 0, instance_map > 0)


def test_character_same_garment_torso_and_sleeve_are_not_split_by_shading() -> None:
    shape = (180, 120)
    instance_map = np.zeros(shape, dtype=np.uint16)
    instance_map[12:166, 30:90] = 1
    instance_map[16:52, 34:86] = 2
    instance_map[55:112, 42:78] = 3
    instance_map[58:116, 78:104] = 4
    instance_map[142:174, 38:82] = 5
    image = np.full((*shape, 3), 244, dtype=np.uint8)
    image[instance_map == 1] = (222, 168, 145)
    image[instance_map == 2] = (48, 55, 61)
    image[instance_map == 3] = (151, 190, 160)
    image[instance_map == 4] = (146, 184, 156)
    image[instance_map == 5] = (82, 133, 118)

    def record(index: int, semantic: str, parent: str = "character") -> PartInstance:
        mask = instance_map == index
        ys, xs = np.nonzero(mask)
        return PartInstance(
            part_id=f"{parent}/{semantic}/center/{index:02d}",
            semantic_name=semantic,
            semantic_parent=parent,
            instance_index=index,
            side="center",
            bbox_xyxy=(int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)),
            centroid_xy=(float(xs.mean()), float(ys.mean())),
            area_px=int(mask.sum()),
        )

    result = build_physical_groups(
        instance_map,
        [
            record(1, "character"),
            record(2, "character_head"),
            record(3, "character_upper_clothing"),
            record(4, "character_sleeve", "character_upper_clothing"),
            record(5, "character_shoe"),
        ],
        image=Image.fromarray(image),
    )

    semantics = {group.semantic_name for group in result.groups}
    assert "character_upper_garment" in semantics
    assert "character_inner_top" not in semantics
    assert "character_outer_garment" not in semantics
    assert result.diagnostics["character_surface_grouping"][
        "layered_upper_garment"
    ]["detected"] is False


def test_character_leg_remains_in_body_group_with_image_audit_enabled() -> None:
    shape = (180, 120)
    instance_map = np.zeros(shape, dtype=np.uint16)
    instance_map[12:58, 38:82] = 1
    instance_map[58:112, 34:86] = 2
    instance_map[112:164, 38:55] = 3
    instance_map[112:164, 65:82] = 4
    image = np.full((*shape, 3), 242, dtype=np.uint8)
    image[np.isin(instance_map, [1, 3, 4])] = (224, 170, 145)
    image[instance_map == 2] = (65, 125, 198)

    def record(index: int, semantic: str) -> PartInstance:
        mask = instance_map == index
        ys, xs = np.nonzero(mask)
        return PartInstance(
            part_id=f"character/{semantic}/center/{index:02d}",
            semantic_name=semantic,
            semantic_parent="character",
            instance_index=index,
            side="center",
            bbox_xyxy=(
                int(xs.min()),
                int(ys.min()),
                int(xs.max() + 1),
                int(ys.max() + 1),
            ),
            centroid_xy=(float(xs.mean()), float(ys.mean())),
            area_px=int(mask.sum()),
        )

    result = build_physical_groups(
        instance_map,
        [
            record(1, "character_head"),
            record(2, "character_upper_clothing"),
            record(3, "character_leg"),
            record(4, "character_leg"),
        ],
        image=Image.fromarray(image),
    )
    body = next(
        group for group in result.groups if group.semantic_name == "character_body"
    )

    assert result.group_map[140, 46] == body.group_index
    assert result.group_map[140, 72] == body.group_index
    leg_records = [
        record for record in result.records if record.semantic_name == "character_leg"
    ]
    assert {record.group_id for record in leg_records} == {body.group_id}
    members_from_records = {
        group.group_id: {
            record.part_id
            for record in result.records
            if record.group_id == group.group_id
        }
        for group in result.groups
    }
    assert all(
        set(group.member_part_ids) == members_from_records[group.group_id]
        for group in result.groups
    )


def test_character_landmarks_recover_shirt_shorts_and_footwear_groups() -> None:
    shape = (180, 120)
    instance_map = np.zeros(shape, dtype=np.uint16)
    root = np.zeros(shape, dtype=bool)
    root[12:58, 39:81] = True
    root[58:122, 32:88] = True
    root[65:82, 10:110] = True
    root[122:166, 38:55] = True
    root[122:166, 65:82] = True
    instance_map[root] = 1
    instance_map[14:57, 40:80] = 2
    instance_map[70:79, 10:20] = 3
    instance_map[70:79, 100:110] = 4
    instance_map[65:81, 20:38] = 5
    instance_map[58:110, 33:87] = 6
    instance_map[110:130, 34:86] = 7
    instance_map[146:166, 36:57] = 8
    instance_map[146:166, 63:84] = 9
    instance_map[10:27, 37:83] = 10

    image = np.full((*shape, 3), 25, dtype=np.uint8)
    image[root] = (205, 160, 145)
    image[instance_map == 2] = (208, 163, 148)
    image[np.isin(instance_map, [3, 4])] = (210, 164, 149)
    image[np.isin(instance_map, [5, 6])] = (75, 130, 205)
    image[instance_map == 7] = (45, 52, 78)
    image[np.isin(instance_map, [8, 9])] = (90, 58, 42)
    image[instance_map == 10] = (52, 34, 28)

    def record(index: int, semantic: str, parent: str = "character") -> PartInstance:
        mask = instance_map == index
        ys, xs = np.nonzero(mask)
        return PartInstance(
            part_id=f"{parent}/{semantic}/center/{index:02d}",
            semantic_name=semantic,
            semantic_parent=parent,
            instance_index=index,
            side="center",
            bbox_xyxy=(
                int(xs.min()),
                int(ys.min()),
                int(xs.max() + 1),
                int(ys.max() + 1),
            ),
            centroid_xy=(float(xs.mean()), float(ys.mean())),
            area_px=int(mask.sum()),
        )

    records = [
        record(1, "character"),
        record(2, "character_head"),
        record(3, "character_hand"),
        record(4, "character_hand"),
        record(5, "character_arm"),
        record(6, "character_visual_panel_01"),
        record(7, "character_visual_panel_02"),
        record(8, "character_lower_clothing"),
        record(9, "character_visual_panel_03"),
        record(10, "character_hair"),
    ]

    result = build_physical_groups(
        instance_map,
        records,
        image=Image.fromarray(image),
    )
    by_name = {group.semantic_name: group for group in result.groups}

    assert result.group_map[80, 60] == by_name["character_upper_garment"].group_index
    assert result.group_map[72, 28] == by_name["character_upper_garment"].group_index
    assert result.group_map[118, 60] == by_name["character_lower_garment"].group_index
    assert result.group_map[155, 46] == by_name["character_footwear"].group_index
    assert result.group_map[35, 60] == by_name["character_body"].group_index
    assert np.array_equal(result.group_map > 0, instance_map > 0)
    grouping = result.diagnostics["character_surface_grouping"]
    regularization = result.diagnostics["character_boundary_regularization"]
    assert grouping["appearance_can_create_ids"] is False
    assert regularization["reassigned_pixel_count"] > 0


def test_character_headwear_and_lower_zone_labels_are_audited() -> None:
    shape = (180, 120)
    instance_map = np.zeros(shape, dtype=np.uint16)
    instance_map[8:55, 26:94] = 1
    instance_map[55:82, 35:85] = 2
    instance_map[76:104, 22:98] = 3
    instance_map[104:158, 34:86] = 4
    instance_map[146:166, 40:80] = 5
    instance_map[88:132, 18:30] = 6
    instance_map[88:132, 90:102] = 7
    image = np.full((*shape, 3), 245, dtype=np.uint8)
    image[instance_map == 1] = (88, 70, 155)
    image[instance_map == 2] = (232, 172, 142)
    image[instance_map == 3] = (70, 150, 205)
    image[instance_map == 4] = (65, 70, 80)
    image[instance_map == 5] = (65, 70, 80)
    image[np.isin(instance_map, [6, 7])] = (230, 170, 140)

    def record(index: int, semantic: str) -> PartInstance:
        mask = instance_map == index
        ys, xs = np.nonzero(mask)
        return PartInstance(
            part_id=f"character/{semantic}/center/{index:02d}",
            semantic_name=semantic,
            semantic_parent="character",
            instance_index=index,
            side="center",
            bbox_xyxy=(
                int(xs.min()),
                int(ys.min()),
                int(xs.max() + 1),
                int(ys.max() + 1),
            ),
            centroid_xy=(float(xs.mean()), float(ys.mean())),
            area_px=int(mask.sum()),
        )

    records = [
        record(1, "character_headwear"),
        record(2, "character_face"),
        record(3, "character_upper_clothing"),
        record(4, "character_lower_clothing"),
        record(5, "character_torso"),
        record(6, "character_hand"),
        record(7, "character_hand"),
    ]

    result = build_physical_groups(
        instance_map,
        records,
        image=Image.fromarray(image),
    )
    by_semantic = {group.semantic_name: group for group in result.groups}

    assert result.group_map[20, 60] == by_semantic["character_headwear"].group_index
    assert (
        result.group_map[153, 60] == by_semantic["character_lower_garment"].group_index
    )


def test_character_skin_is_returned_from_an_overbroad_hair_mask() -> None:
    shape = (180, 120)
    instance_map = np.zeros(shape, dtype=np.uint16)
    root = np.zeros(shape, dtype=bool)
    root[12:72, 24:96] = True
    root[70:145, 34:86] = True
    root[78:118, 14:106] = True
    instance_map[root] = 1
    instance_map[12:72, 24:96] = 2
    instance_map[86:112, 14:26] = 3
    instance_map[72:122, 32:88] = 4

    image = np.full((*shape, 3), 245, dtype=np.uint8)
    image[root] = (224, 168, 144)
    image[instance_map == 2] = (45, 35, 34)
    image[30:66, 38:82] = (225, 170, 145)
    image[instance_map == 3] = (226, 171, 146)
    image[instance_map == 4] = (75, 135, 205)

    def record(index: int, semantic: str) -> PartInstance:
        mask = instance_map == index
        ys, xs = np.nonzero(mask)
        return PartInstance(
            part_id=f"character/{semantic}/center/{index:02d}",
            semantic_name=semantic,
            semantic_parent="character",
            instance_index=index,
            side="center",
            bbox_xyxy=(
                int(xs.min()),
                int(ys.min()),
                int(xs.max() + 1),
                int(ys.max() + 1),
            ),
            centroid_xy=(float(xs.mean()), float(ys.mean())),
            area_px=int(mask.sum()),
        )

    result = build_physical_groups(
        instance_map,
        [
            record(1, "character"),
            record(2, "character_hair"),
            record(3, "character_hand"),
            record(4, "character_upper_clothing"),
        ],
        image=Image.fromarray(image),
    )
    by_name = {group.semantic_name: group for group in result.groups}

    assert result.group_map[45, 60] == by_name["character_body"].group_index
    assert result.group_map[20, 60] == by_name["character_hair"].group_index
    assert np.array_equal(result.group_map > 0, instance_map > 0)
    audit = result.diagnostics["character_boundary_regularization"]["head_skin_audit"]
    assert audit["status"] == "completed"
    assert audit["appearance_can_create_ids"] is False


def test_character_face_interior_islands_are_not_kept_as_hair() -> None:
    shape = (190, 140)
    instance_map = np.zeros(shape, dtype=np.uint16)
    instance_map[68:154, 42:98] = 1
    instance_map[22:82, 34:106] = 2
    instance_map[18:45, 30:110] = 3
    instance_map[49:65, 48:62] = 3
    instance_map[49:65, 78:92] = 3
    instance_map[72:118, 38:102] = 4
    instance_map[116:145, 42:98] = 5
    instance_map[84:102, 20:38] = 6
    instance_map[84:102, 102:120] = 7

    image = np.full((*shape, 3), 245, dtype=np.uint8)
    image[instance_map > 0] = (225, 170, 145)
    image[instance_map == 3] = (42, 34, 30)
    image[instance_map == 4] = (70, 120, 190)
    image[instance_map == 5] = (55, 65, 80)

    def record(index: int, semantic: str) -> PartInstance:
        mask = instance_map == index
        ys, xs = np.nonzero(mask)
        return PartInstance(
            part_id=f"character/{semantic}/center/{index:02d}",
            semantic_name=semantic,
            semantic_parent="character",
            instance_index=index,
            side="center",
            bbox_xyxy=(
                int(xs.min()),
                int(ys.min()),
                int(xs.max() + 1),
                int(ys.max() + 1),
            ),
            centroid_xy=(float(xs.mean()), float(ys.mean())),
            area_px=int(mask.sum()),
        )

    result = build_physical_groups(
        instance_map,
        [
            record(1, "character"),
            record(2, "character_head"),
            record(3, "character_hair"),
            record(4, "character_upper_clothing"),
            record(5, "character_lower_clothing"),
            record(6, "character_hand"),
            record(7, "character_hand"),
        ],
        image=Image.fromarray(image),
    )
    by_name = {group.semantic_name: group for group in result.groups}

    assert result.group_map[28, 70] == by_name["character_hair"].group_index
    assert result.group_map[56, 54] == by_name["character_body"].group_index
    assert result.group_map[56, 84] == by_name["character_body"].group_index
    assert np.array_equal(result.group_map > 0, instance_map > 0)
    audit = result.diagnostics["character_boundary_regularization"][
        "face_interior_topology_audit"
    ]
    assert audit["status"] == "completed"
    assert audit["appearance_used"] is False


def test_character_bare_lower_legs_return_from_overbroad_footwear() -> None:
    shape = (230, 150)
    instance_map = np.zeros(shape, dtype=np.uint16)
    instance_map[72:180, 48:102] = 1
    instance_map[18:76, 38:112] = 2
    instance_map[16:42, 34:116] = 3
    instance_map[82:128, 42:108] = 4
    instance_map[126:158, 48:102] = 5
    instance_map[158:215, 50:72] = 6
    instance_map[158:215, 78:100] = 6
    instance_map[92:112, 22:42] = 7
    instance_map[92:112, 108:128] = 8

    image = np.full((*shape, 3), 245, dtype=np.uint8)
    image[instance_map > 0] = (222, 168, 143)
    image[instance_map == 3] = (40, 32, 30)
    image[instance_map == 4] = (66, 120, 190)
    image[instance_map == 5] = (45, 50, 60)
    image[198:215, 50:72] = (35, 38, 45)
    image[198:215, 78:100] = (35, 38, 45)

    def record(index: int, semantic: str) -> PartInstance:
        mask = instance_map == index
        ys, xs = np.nonzero(mask)
        return PartInstance(
            part_id=f"character/{semantic}/center/{index:02d}",
            semantic_name=semantic,
            semantic_parent="character",
            instance_index=index,
            side="center",
            bbox_xyxy=(
                int(xs.min()),
                int(ys.min()),
                int(xs.max() + 1),
                int(ys.max() + 1),
            ),
            centroid_xy=(float(xs.mean()), float(ys.mean())),
            area_px=int(mask.sum()),
        )

    result = build_physical_groups(
        instance_map,
        [
            record(1, "character"),
            record(2, "character_head"),
            record(3, "character_hair"),
            record(4, "character_upper_clothing"),
            record(5, "character_lower_clothing"),
            record(6, "character_shoe"),
            record(7, "character_hand"),
            record(8, "character_hand"),
        ],
        image=Image.fromarray(image),
    )
    by_name = {group.semantic_name: group for group in result.groups}

    assert result.group_map[174, 60] == by_name["character_body"].group_index
    assert result.group_map[174, 88] == by_name["character_body"].group_index
    assert result.group_map[206, 60] == by_name["character_footwear"].group_index
    assert result.group_map[206, 88] == by_name["character_footwear"].group_index
    assert result.group_map[140, 75] == by_name["character_lower_garment"].group_index
    assert np.array_equal(result.group_map > 0, instance_map > 0)
    audit = result.diagnostics["character_boundary_regularization"][
        "lower_limb_skin_audit"
    ]
    assert audit["status"] == "completed"
    assert audit["appearance_can_create_ids"] is False


def test_character_visual_garments_follow_pose_axis_after_rotation() -> None:
    shape = (240, 240)
    upright = np.zeros(shape, dtype=np.uint16)
    upright[72:194, 91:149] = 1
    upright[24:82, 82:158] = 2
    upright[18:48, 78:162] = 3
    upright[84:128, 86:154] = 4
    upright[130:169, 88:152] = 5
    upright[178:218, 88:110] = 6
    upright[178:218, 130:152] = 6
    upright[92:112, 62:86] = 7
    upright[92:112, 154:178] = 8

    upright_image = np.full((*shape, 3), 245, dtype=np.uint8)
    upright_image[upright > 0] = (222, 168, 143)
    upright_image[upright == 3] = (40, 32, 30)
    upright_image[upright == 4] = (60, 125, 205)
    upright_image[upright == 5] = (55, 65, 80)
    upright_image[upright == 6] = (35, 38, 45)

    transform = cv2.getRotationMatrix2D((120.0, 120.0), 34.0, 1.0)
    instance_map = cv2.warpAffine(
        upright,
        transform,
        (shape[1], shape[0]),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    image = cv2.warpAffine(
        upright_image,
        transform,
        (shape[1], shape[0]),
        flags=cv2.INTER_LINEAR,
        borderValue=(245, 245, 245),
    )

    semantics = {
        1: "character",
        2: "character_head",
        3: "character_hair",
        4: "character_visual_panel_01",
        5: "character_visual_panel_02",
        6: "character_shoe",
        7: "character_hand",
        8: "character_hand",
    }

    def record(index: int) -> PartInstance:
        mask = instance_map == index
        ys, xs = np.nonzero(mask)
        semantic = semantics[index]
        return PartInstance(
            part_id=f"character/{semantic}/center/{index:02d}",
            semantic_name=semantic,
            semantic_parent="character",
            instance_index=index,
            side="center",
            bbox_xyxy=(
                int(xs.min()),
                int(ys.min()),
                int(xs.max() + 1),
                int(ys.max() + 1),
            ),
            centroid_xy=(float(xs.mean()), float(ys.mean())),
            area_px=int(mask.sum()),
        )

    result = build_physical_groups(
        instance_map,
        [record(index) for index in semantics],
        image=Image.fromarray(image),
    )
    groups_by_part = {
        part_id: group.semantic_name
        for group in result.groups
        for part_id in group.member_part_ids
    }

    assert groups_by_part["character/character_visual_panel_01/center/04"] == (
        "character_upper_garment"
    )
    assert groups_by_part["character/character_visual_panel_02/center/05"] == (
        "character_lower_garment"
    )
    assert np.array_equal(result.group_map > 0, instance_map > 0)
    diagnostics = result.diagnostics["character_surface_grouping"]
    assert diagnostics["algorithm"] == "character-pose-axis-surface-grouping-v2"


def test_open_set_visual_regions_merge_instead_of_becoming_fake_groups() -> None:
    instance_map = _strip_map(3)
    records = [
        _record(1, "device", "device"),
        _record(2, "device_visual_panel_01", "device"),
        _record(3, "device_visual_strip_01", "device"),
    ]

    result = build_physical_groups(instance_map, records)

    assert len(result.groups) == 1
    assert result.groups[0].semantic_name == "device_body"
    assert result.groups[0].review_required is True


def test_profile_semantics_own_visual_regions_before_appearance_evidence() -> None:
    shape = (24, 24)
    instance_map = np.zeros(shape, dtype=np.uint16)
    instance_map[2:16, 2:20] = 2
    instance_map[6:9, 14:19] = 5
    instance_map[16:19, 10:13] = 4
    instance_map[19:23, 6:18] = 3
    instance_map[3:5, 3:7] = 1

    def record(index: int, semantic: str, parent: str) -> PartInstance:
        mask = instance_map == index
        ys, xs = np.nonzero(mask)
        return PartInstance(
            part_id=f"object_001/{semantic}/{index:02d}",
            semantic_name=semantic,
            semantic_parent=parent,
            instance_index=index,
            side="center",
            bbox_xyxy=(
                int(xs.min()),
                int(ys.min()),
                int(xs.max() + 1),
                int(ys.max() + 1),
            ),
            centroid_xy=(float(xs.mean()), float(ys.mean())),
            area_px=int(mask.sum()),
        )

    records = [
        record(1, "device", "device"),
        record(2, "device_globe_sphere", "device_body"),
        record(3, "device_base", "device_body"),
        record(4, "device_globe_stem", "device_base"),
        record(5, "device_visual_panel_01", "device"),
    ]
    candidates = [
        MaskCandidate(
            "device",
            "device",
            instance_map > 0,
            0.95,
            "test/root",
            metadata={
                "root_origin": "test",
                "root_index": 1,
                "candidate_key": "root:1",
                "parent_candidate_key": None,
                "selected_part_profile": "globe",
            },
        )
    ]
    for record_row in records[1:4]:
        candidates.append(
            MaskCandidate(
                record_row.semantic_name,
                record_row.semantic_parent,
                instance_map == record_row.instance_index,
                0.88,
                "test/profile-refinement",
                metadata={
                    "root_origin": "test",
                    "root_index": 1,
                    "candidate_key": f"root:1/{record_row.semantic_name}",
                    "parent_candidate_key": "root:1",
                    "selected_part_profile": "globe",
                    "profile_refinement": True,
                },
            )
        )

    result = build_physical_groups(
        instance_map,
        records,
        candidates=candidates,
        image=Image.new("RGB", shape[::-1], "white"),
    )

    assert {group.semantic_name for group in result.groups} == {
        "device_globe_sphere",
        "device_base",
        "device_globe_stem",
    }
    sphere = next(
        group for group in result.groups if group.semantic_name == "device_globe_sphere"
    )
    assert records[0].part_id in sphere.member_part_ids
    assert records[4].part_id in sphere.member_part_ids
    assert result.diagnostics["profile_semantic_grouping"]["evidence_order"] == [
        "semantic_inventory",
        "geometry",
        "appearance",
    ]


def test_globe_ring_continuation_survives_shading_review_as_one_group() -> None:
    shape = (120, 120)
    instance_map = np.zeros(shape, dtype=np.uint16)
    instance_map[10:82, 14:88] = 1
    instance_map[18:74, 20:82] = 2
    instance_map[14:72, 84:92] = 3
    instance_map[70:91, 77:92] = 4
    instance_map[84:104, 54:65] = 5
    instance_map[102:117, 35:84] = 6

    def record(index: int, semantic: str, parent: str) -> PartInstance:
        mask = instance_map == index
        ys, xs = np.nonzero(mask)
        return PartInstance(
            part_id=f"object_001/{semantic}/{index:02d}",
            semantic_name=semantic,
            semantic_parent=parent,
            instance_index=index,
            side="center",
            bbox_xyxy=(int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)),
            centroid_xy=(float(xs.mean()), float(ys.mean())),
            area_px=int(mask.sum()),
        )

    records = [
        record(1, "device", "device"),
        record(2, "device_globe_sphere", "device_body"),
        record(3, "device_globe_meridian_ring", "device_body"),
        record(4, "device_visual_strip_04", "device"),
        record(5, "device_globe_stem", "device_base"),
        record(6, "device_base", "device_body"),
    ]
    candidates = [
        MaskCandidate(
            "device",
            "device",
            instance_map > 0,
            0.95,
            "test/root",
            metadata={
                "candidate_key": "root:1",
                "selected_part_profile": "globe",
            },
        )
    ]
    for record_row in (records[1], records[4], records[5]):
        candidates.append(
            MaskCandidate(
                record_row.semantic_name,
                record_row.semantic_parent,
                instance_map == record_row.instance_index,
                0.9,
                "test/profile-refine",
                metadata={
                    "candidate_key": f"root:1/{record_row.semantic_name}",
                    "selected_part_profile": "globe",
                    "profile_refinement": True,
                },
            )
        )
    candidates.append(
        MaskCandidate(
            "device_globe_meridian_ring",
            "device_body",
            instance_map == 3,
            0.56,
            "test/proposal-first/semantic-rerank",
            metadata={
                "candidate_key": "root:1/visual-region:ring",
                "semantic_reranked": True,
                "semantic_rerank_route": "base",
                "semantic_rerank_probability": 0.26,
                "semantic_rerank_margin": 0.008,
                "semantic_rerank_profile": "globe",
                "root_area_fraction": 0.06,
                "proposal_boundary_alignment": 1.0,
                "cross_source_confirmed": True,
                "appearance_graph_evidence": {
                    "boundary_alignment": 1.0,
                    "boundary_closure": 0.89,
                    "independent_cue_count": 2,
                    "shading_only_penalty": 0.76,
                },
            },
        )
    )

    result = build_physical_groups(
        instance_map,
        records,
        candidates=candidates,
        image=Image.new("RGB", shape[::-1], "white"),
    )
    by_name = {group.semantic_name: group for group in result.groups}

    assert result.group_map[30, 88] == by_name[
        "device_globe_meridian_ring"
    ].group_index
    assert result.group_map[80, 84] == by_name[
        "device_globe_meridian_ring"
    ].group_index
    assert result.group_map[94, 59] == by_name["device_globe_stem"].group_index
    verification_rows = result.diagnostics["profile_structural_decomposition"][
        "candidates"
    ]
    assert any(
        row["semantic_name"] == "device_globe_meridian_ring"
        and row["structurally_recovered"] is True
        for row in verification_rows
    )


def test_tiny_remote_appearance_satellite_is_reassigned_to_nearest_group() -> None:
    instance_map = np.zeros((80, 100), dtype=np.uint16)
    instance_map[10:50, 10:50] = 1
    instance_map[60:76, 30:70] = 2
    instance_map[49:51, 20:22] = 2
    records = [
        _record(1, "device_globe_sphere", "device_body"),
        _record(2, "device_base", "device_body"),
    ]

    result = build_physical_groups(instance_map, records)

    assert np.all(result.group_map[49:51, 20:22] == 1)
    assert np.array_equal(result.group_map > 0, instance_map > 0)
    cleanup = result.diagnostics["physical_component_cleanup"]
    assert cleanup["removed_component_count"] == 1
    assert cleanup["reassigned_pixel_count"] == 4
    sphere = next(
        group for group in result.groups if group.semantic_name == "device_globe_sphere"
    )
    assert sphere.area_px == 1602


def test_tiny_disconnected_region_with_its_own_part_evidence_is_preserved() -> None:
    instance_map = np.zeros((80, 100), dtype=np.uint16)
    instance_map[8:48, 8:48] = 1
    instance_map[65:67, 14:16] = 2
    instance_map[60:76, 70:90] = 3
    records = [
        _record(1, "character_arm", "character_body"),
        _record(2, "character_hand", "character_body"),
        _record(3, "character_hair", "character_head"),
    ]

    result = build_physical_groups(instance_map, records)

    body = next(
        group for group in result.groups if group.semantic_name == "character_body"
    )
    assert np.all(result.group_map[65:67, 14:16] == body.group_index)
    cleanup = result.diagnostics["physical_component_cleanup"]
    assert cleanup["removed_component_count"] == 0


def test_flared_lower_stem_is_reassigned_to_the_structural_base() -> None:
    instance_map = np.zeros((100, 120), dtype=np.uint16)
    instance_map[60:90, 20:100] = 2
    instance_map[10:52, 55:65] = 1
    for y in range(52, 70):
        half_width = 6 + (y - 52) // 2
        instance_map[y, 60 - half_width : 60 + half_width] = 1
    records = [
        _record(1, "device_globe_stem", "device_base"),
        _record(2, "device_base", "device_body"),
    ]

    result = build_physical_groups(instance_map, records)

    stem = next(
        group for group in result.groups if group.semantic_name == "device_globe_stem"
    )
    base = next(
        group for group in result.groups if group.semantic_name == "device_base"
    )
    assert result.group_map[20, 60] == stem.group_index
    assert result.group_map[65, 60] == base.group_index
    boundary = result.diagnostics["structural_boundary_regularization"]
    assert boundary["adjusted_pair_count"] == 1
    assert boundary["appearance_used"] is False


def test_firearm_axis_topology_rejects_highlight_strips_as_public_parts() -> None:
    instance_map = np.zeros((160, 360), dtype=np.uint16)
    root = np.zeros(instance_map.shape, dtype=bool)
    root[52:96, 20:330] = True
    root[42:112, 270:350] = True
    root[92:148, 120:170] = True
    root[92:146, 200:228] = True
    instance_map[root] = 1
    instance_map[52:96, 20:52] = 2
    instance_map[42:112, 270:350] = 3
    instance_map[55:94, 65:135] = 4
    instance_map[66:80, 52:122] = 5
    instance_map[57:61, 175:255] = 6
    instance_map[62:68, 175:215] = 7
    instance_map[58:92, 145:260] = 8
    records = [
        _record(1, "tool_prop", "tool_prop"),
        _record(2, "tool_prop_muzzle", "tool_prop"),
        _record(3, "tool_prop_stock", "tool_prop"),
        _record(4, "tool_prop_handguard", "tool_prop"),
        _record(5, "tool_prop_barrel", "tool_prop"),
        _record(6, "tool_prop_visual_strip_01", "tool_prop"),
        _record(7, "tool_prop_charging_handle", "tool_prop"),
        _record(8, "tool_prop_receiver", "tool_prop"),
    ]

    result = build_physical_groups(
        instance_map,
        records,
        candidates=[_firearm_root(instance_map.shape)],
    )

    by_name = {group.semantic_name: group for group in result.groups}
    assert set(by_name) == {
        "tool_prop_muzzle",
        "tool_prop_barrel",
        "tool_prop_handguard",
        "tool_prop_receiver",
        "tool_prop_magazine",
        "tool_prop_grip",
        "tool_prop_stock",
    }
    assert result.group_map[130, 145] == by_name["tool_prop_magazine"].group_index
    assert result.group_map[130, 214] == by_name["tool_prop_grip"].group_index
    assert result.group_map[59, 200] == by_name["tool_prop_receiver"].group_index
    assert result.group_map[70, 310] == by_name["tool_prop_stock"].group_index
    assert np.array_equal(result.group_map > 0, instance_map > 0)
    assert result.diagnostics["firearm_inventory_complete"] is True
    decomposition = result.diagnostics["firearm_structural_decomposition"]
    assert decomposition["appearance_used"] is False


def test_three_stage_gate_rejects_weak_semantics_despite_closed_colour_region() -> None:
    mask = np.zeros((80, 160), dtype=bool)
    mask[28:56, 35:82] = True
    candidate = MaskCandidate(
        semantic_name="tool_prop_magazine",
        semantic_parent="tool_prop",
        mask=mask,
        score=0.82,
        source="hpid-appearance-contour/closed-edge/semantic-rerank",
        metadata={
            "selected_part_profile": "firearm",
            "visual_region": True,
            "semantic_reranked": True,
            "semantic_rerank_route": "contextual_axis_structure_rescue",
            "semantic_rerank_probability": 0.083,
            "semantic_rerank_margin": 0.0045,
            "proposal_boundary_alignment": 1.0,
            "geometric_support": 0.86,
            "axis_consistency_gate": {"accepted": True},
            "appearance_graph_evidence": {
                "boundary_closure": 0.96,
                "independent_cue_count": 3,
                "shading_only_penalty": 0.0,
            },
        },
    )

    verification = _candidate_three_stage_verification(candidate)

    assert verification["stage_1_semantic"]["verified"] is False
    assert verification["stage_2_structure"]["evaluated"] is False
    assert verification["stage_2_structure"]["raw_verified"] is True
    assert verification["stage_2_structure"]["verified"] is False
    assert verification["stage_3_appearance"]["evaluated"] is False
    assert verification["stage_3_appearance"]["raw_verified"] is True
    assert verification["stage_3_appearance"]["verified"] is False
    assert verification["accepted"] is False


def test_three_stage_gate_rejects_shading_after_semantic_and_structure_pass() -> None:
    mask = np.zeros((80, 160), dtype=bool)
    mask[22:58, 45:118] = True
    candidate = MaskCandidate(
        semantic_name="tool_prop_receiver",
        semantic_parent="tool_prop",
        mask=mask,
        score=0.88,
        source="sam2-amg/test/semantic-rerank",
        metadata={
            "selected_part_profile": "firearm",
            "visual_region": True,
            "semantic_reranked": True,
            "semantic_rerank_route": "base",
            "semantic_rerank_probability": 0.24,
            "semantic_rerank_margin": 0.03,
            "proposal_boundary_alignment": 0.91,
            "geometric_support": 0.76,
            "axis_consistency_gate": {"accepted": True},
            "appearance_graph_evidence": {
                "boundary_closure": 0.68,
                "independent_cue_count": 2,
                "shading_only_penalty": 0.84,
            },
        },
    )

    verification = _candidate_three_stage_verification(candidate)

    assert verification["stage_1_semantic"]["verified"] is True
    assert verification["stage_2_structure"]["verified"] is True
    assert verification["stage_3_appearance"]["verified"] is False
    assert verification["accepted"] is False


def test_three_stage_gate_rejects_single_model_named_shadow_without_structure() -> None:
    mask = np.zeros((100, 160), dtype=bool)
    mask[35:65, 44:116] = True
    candidate = MaskCandidate(
        semantic_name="vehicle_seat",
        semantic_parent="vehicle_body",
        mask=mask,
        score=0.82,
        source="conditional-part[model]/direct-calibrated-mask",
        metadata={
            "selected_part_profile": "road_vehicle",
            "direct_conditional_mask": True,
            "root_area_fraction": 0.08,
            "geometric_support": 0.46,
            "cross_source_confirmed": False,
            "appearance_graph_evidence": {
                "boundary_alignment": 0.42,
                "boundary_closure": 0.18,
                "independent_cue_count": 1,
                "shading_only_penalty": 0.88,
            },
        },
    )

    verification = _candidate_three_stage_verification(candidate)

    assert verification["stage_1_semantic"]["verified"] is True
    assert verification["stage_2_structure"]["verified"] is False
    assert verification["stage_3_appearance"]["evaluated"] is False
    assert verification["accepted"] is False


def test_three_stage_gate_keeps_bounded_window_despite_reflection() -> None:
    mask = np.zeros((100, 160), dtype=bool)
    mask[22:72, 36:126] = True
    candidate = MaskCandidate(
        semantic_name="vehicle_windshield",
        semantic_parent="vehicle_body",
        mask=mask,
        score=0.84,
        source="conditional-part[model]/direct-calibrated-mask",
        metadata={
            "selected_part_profile": "road_vehicle",
            "direct_conditional_mask": True,
            "root_area_fraction": 0.24,
            "geometric_support": 0.59,
            "cross_source_confirmed": False,
            "appearance_graph_evidence": {
                "boundary_alignment": 0.92,
                "boundary_closure": 0.74,
                "independent_cue_count": 2,
                "shading_only_penalty": 0.82,
            },
        },
    )

    verification = _candidate_three_stage_verification(candidate)

    assert verification["stage_2_structure"]["bounded_photometric_surface"] is True
    assert verification["stage_2_structure"]["verified"] is True
    assert verification["stage_3_appearance"]["verified"] is True
    assert verification["accepted"] is True


def test_three_stage_gate_accepts_inventory_label_with_closed_multi_cue_boundary() -> None:
    mask = np.zeros((100, 160), dtype=bool)
    mask[22:78, 35:125] = True
    candidate = MaskCandidate(
        semantic_name="vehicle_wheel",
        semantic_parent="vehicle",
        mask=mask,
        score=0.78,
        source="sam2-amg/test/semantic-rerank",
        metadata={
            "selected_part_profile": "two_wheeler",
            "visual_region": True,
            "semantic_reranked": True,
            "semantic_rerank_route": "semantic_inventory_evidence_rescue",
            "semantic_rerank_probability": 0.16,
            "semantic_rerank_margin": 0.02,
            "proposal_boundary_alignment": 0.92,
            "geometric_support": 0.0,
            "root_area_fraction": 0.28,
            "axis_consistency_gate": {"accepted": True},
            "appearance_graph_evidence": {
                "boundary_closure": 0.82,
                "independent_cue_count": 3,
                "shading_only_penalty": 0.03,
            },
        },
    )

    verification = _candidate_three_stage_verification(candidate)

    assert verification["stage_1_semantic"]["verified"] is True
    assert verification["stage_2_structure"]["verified"] is True
    assert verification["stage_2_structure"]["reason"] == "closed_boundary_structure"
    assert verification["stage_3_appearance"]["verified"] is True
    assert verification["accepted"] is True


def test_three_stage_gate_accepts_semantic_part_with_strong_closed_boundary() -> None:
    mask = np.zeros((100, 160), dtype=bool)
    mask[22:48, 35:105] = True
    candidate = MaskCandidate(
        semantic_name="container_lid",
        semantic_parent="container_body",
        mask=mask,
        score=0.55,
        source="sam2-amg/test/semantic-rerank",
        metadata={
            "selected_part_profile": "box",
            "visual_region": True,
            "semantic_reranked": True,
            "semantic_rerank_route": "base",
            "semantic_rerank_probability": 0.17,
            "semantic_rerank_margin": 0.007,
            "proposal_boundary_alignment": 0.70,
            "geometric_support": 0.0,
            "root_area_fraction": 0.08,
            "axis_consistency_gate": {"accepted": True},
            "appearance_graph_evidence": {
                "boundary_closure": 0.88,
                "independent_cue_count": 1,
                "shading_only_penalty": 0.0,
            },
        },
    )

    verification = _candidate_three_stage_verification(candidate)

    assert verification["stage_1_semantic"]["verified"] is True
    assert verification["stage_2_structure"]["reason"] == "closed_boundary_structure"
    assert verification["stage_2_structure"]["verified"] is True
    assert verification["stage_3_appearance"]["verified"] is True
    assert verification["accepted"] is True


def test_three_stage_gate_accepts_inventory_topology_complement() -> None:
    mask = np.zeros((80, 180), dtype=bool)
    mask[28:54, 112:168] = True
    candidate = MaskCandidate(
        semantic_name="tool_prop_handle",
        semantic_parent="tool_prop_body",
        mask=mask,
        score=0.31,
        source="hpid-topology-v2/terminal_complement",
        metadata={
            "selected_part_profile": "knife",
            "topology_refinement": True,
            "topology_dense_gate": True,
            "topology_diagnostics": {"selected_component": 1},
            "root_area_fraction": 0.24,
            "axis_consistency_gate": {"accepted": True},
        },
    )

    verification = _candidate_three_stage_verification(candidate)

    assert verification["stage_1_semantic"]["verified"] is True
    assert verification["stage_2_structure"]["reason"] == (
        "inventory_topology_complement"
    )
    assert verification["stage_2_structure"]["verified"] is True
    assert verification["stage_3_appearance"]["verified"] is True
    assert verification["accepted"] is True


def _semantic_partition_candidates(
    root: np.ndarray,
    blade: np.ndarray,
    handle: np.ndarray,
) -> tuple[MaskCandidate, ...]:
    return (
        MaskCandidate(
            "tool_prop",
            "tool_prop",
            root,
            0.94,
            "test/root",
            metadata={
                "root_origin": "test",
                "root_index": 1,
                "selected_part_profile": "knife",
            },
        ),
        MaskCandidate(
            "tool_prop_blade",
            "tool_prop_body",
            blade,
            0.58,
            "grounded-sam2/test/profile-refine",
            metadata={
                "root_origin": "test",
                "root_index": 1,
                "selected_part_profile": "knife",
                "maximum_instances": 1,
                "root_area_fraction": float(blade.sum() / root.sum()),
                "axis_consistency_gate": {"accepted": True},
            },
        ),
        MaskCandidate(
            "tool_prop_handle",
            "tool_prop_body",
            handle,
            0.52,
            "hpid-topology-v2/terminal_complement",
            metadata={
                "root_origin": "test",
                "root_index": 1,
                "selected_part_profile": "knife",
                "maximum_instances": 1,
                "topology_refinement": True,
                "topology_dense_gate": True,
                "topology_diagnostics": {"selected_component": 1},
                "root_area_fraction": float(handle.sum() / root.sum()),
                "axis_consistency_gate": {"accepted": True},
            },
        ),
    )


def test_semantic_seeded_partition_completes_root_without_visual_ids() -> None:
    shape = (90, 240)
    root = np.zeros(shape, dtype=bool)
    root[24:66, 12:226] = True
    blade = np.zeros(shape, dtype=bool)
    blade[28:62, 16:92] = True
    handle = np.zeros(shape, dtype=bool)
    handle[30:60, 154:220] = True
    instance_map = np.zeros(shape, dtype=np.uint16)
    instance_map[root] = 1
    image = np.full((*shape, 3), 24, dtype=np.uint8)
    image[root] = (130, 145, 162)
    image[root & (np.indices(shape)[1] >= 128)] = (128, 78, 42)
    candidates = _semantic_partition_candidates(root, blade, handle)

    masks, diagnostics = _semantic_seeded_profile_masks(
        instance_map,
        candidates,
        Image.fromarray(image),
        profile="knife",
    )

    assert masks is not None
    assert set(masks) == {"tool_prop_blade", "tool_prop_handle"}
    assert np.array_equal(np.logical_or.reduce(list(masks.values())), root)
    assert not np.any(masks["tool_prop_blade"] & masks["tool_prop_handle"])
    assert np.all(masks["tool_prop_blade"][blade])
    assert np.all(masks["tool_prop_handle"][handle])
    assert diagnostics["evidence_order"] == [
        "semantic",
        "structure",
        "appearance",
    ]
    assert diagnostics["appearance_can_create_ids"] is False
    assert diagnostics["ground_truth_used"] is False


def test_semantic_seeded_partition_is_candidate_order_invariant() -> None:
    shape = (72, 180)
    root = np.zeros(shape, dtype=bool)
    root[18:54, 8:172] = True
    blade = np.zeros(shape, dtype=bool)
    blade[21:51, 12:70] = True
    handle = np.zeros(shape, dtype=bool)
    handle[22:50, 116:168] = True
    instance_map = np.zeros(shape, dtype=np.uint16)
    instance_map[root] = 1
    candidates = _semantic_partition_candidates(root, blade, handle)

    first, _ = _semantic_seeded_profile_masks(
        instance_map,
        candidates,
        None,
        profile="knife",
    )
    second, _ = _semantic_seeded_profile_masks(
        instance_map,
        tuple(reversed(candidates)),
        None,
        profile="knife",
    )

    assert first is not None and second is not None
    assert all(np.array_equal(first[name], second[name]) for name in first)


def test_semantic_seeded_partition_refuses_one_weak_seed() -> None:
    shape = (70, 170)
    root = np.zeros(shape, dtype=bool)
    root[16:54, 8:162] = True
    blade = np.zeros(shape, dtype=bool)
    blade[20:50, 12:58] = True
    instance_map = np.zeros(shape, dtype=np.uint16)
    instance_map[root] = 1
    candidates = _semantic_partition_candidates(
        root,
        blade,
        np.zeros(shape, dtype=bool),
    )[:2]

    masks, diagnostics = _semantic_seeded_profile_masks(
        instance_map,
        candidates,
        None,
        profile="knife",
    )

    assert masks is None
    assert diagnostics["status"] == "insufficient_verified_macro_seeds"


def test_firearm_handguard_uses_material_only_after_structural_seeding() -> None:
    shape = (100, 180)
    fore_end = np.zeros(shape, dtype=bool)
    fore_end[25:75, 20:150] = True
    normal = np.indices(shape)[0].astype(np.float64)
    semantic_seed = np.zeros(shape, dtype=bool)
    semantic_seed[52:73, 42:112] = True

    image = np.full((*shape, 3), 18, dtype=np.uint8)
    image[fore_end] = (82, 86, 92)
    image[52:73, 42:112] = (145, 78, 34)
    image[31:45, 48:105] = (151, 82, 38)
    image[35:38, 118:145] = (215, 218, 220)

    handguard, diagnostics = _firearm_handguard_from_structure_and_material(
        Image.fromarray(image),
        fore_end,
        normal,
        semantic_seed,
    )

    assert handguard[62, 72]
    assert handguard[37, 72]
    assert not handguard[32, 130]
    assert diagnostics["appearance_used"] is True
    assert diagnostics["appearance_can_create_ids"] is False


def test_inventory_boundary_refinement_moves_a_coarse_cut_to_the_visible_seam() -> None:
    shape = (80, 180)
    root = np.zeros(shape, dtype=bool)
    root[10:70, 10:170] = True
    coarse_left = root.copy()
    coarse_left[:, 90:] = False
    coarse_right = root & ~coarse_left
    image = np.full((*shape, 3), 12, dtype=np.uint8)
    image[root & (np.indices(shape)[1] < 98)] = (70, 72, 76)
    image[root & (np.indices(shape)[1] >= 98)] = (190, 194, 200)

    refined, diagnostics = _refine_inventory_boundaries(
        Image.fromarray(image),
        root,
        {"left_part": coarse_left, "right_part": coarse_right},
    )

    first_right_x = [
        int(np.flatnonzero(refined["right_part"][y])[0]) for y in range(20, 60)
    ]
    assert 96 <= float(np.median(first_right_x)) <= 100
    assert np.array_equal(refined["left_part"] | refined["right_part"], root)
    assert not np.any(refined["left_part"] & refined["right_part"])
    assert diagnostics["status"] == "completed"
    assert diagnostics["elevation_quantization_levels"] == 4096
    assert diagnostics["reassigned_pixel_count"] > 0
    assert diagnostics["appearance_can_create_ids"] is False


def test_strong_closed_panel_can_become_an_open_set_physical_group() -> None:
    instance_map = _strip_map(2)
    records = [
        _record(1, "device", "device"),
        _record(2, "device_visual_panel_01", "device"),
    ]
    panel_mask = instance_map == 2
    candidate = MaskCandidate(
        semantic_name="device_visual_panel_01",
        semantic_parent="device",
        mask=panel_mask,
        score=0.86,
        source="hpid-appearance-contour/closed-edge",
        metadata={
            "root_area_fraction": 0.18,
            "proposal_boundary_alignment": 0.94,
            "cross_source_confirmed": True,
            "appearance_graph_evidence": {
                "boundary_closure": 0.74,
                "independent_cue_count": 3,
            },
            "physical_region_gate": {
                "nested_surface_texture": False,
                "laminar_surface_strip": False,
            },
        },
    )

    result = build_physical_groups(
        instance_map,
        records,
        candidates=[candidate],
    )

    assert {group.semantic_name for group in result.groups} == {
        "device_body",
        "device_physical_panel_01",
    }
    assert result.diagnostics["promoted_visual_region_count"] == 1


def test_closed_appearance_region_alone_cannot_create_an_editable_group() -> None:
    instance_map = _strip_map(2)
    records = [
        _record(1, "device", "device"),
        _record(2, "device_visual_panel_01", "device"),
    ]
    panel_mask = instance_map == 2
    candidate = MaskCandidate(
        semantic_name="device_visual_panel_01",
        semantic_parent="device",
        mask=panel_mask,
        score=0.93,
        source="hpid-appearance-contour/closed-edge",
        metadata={
            "root_area_fraction": 0.20,
            "proposal_boundary_alignment": 0.98,
            "appearance_graph_evidence": {
                "boundary_closure": 0.91,
                "independent_cue_count": 3,
            },
            "physical_region_gate": {
                "nested_surface_texture": False,
                "laminar_surface_strip": False,
                "cross_source_structure": False,
            },
        },
    )

    result = build_physical_groups(
        instance_map,
        records,
        candidates=[candidate],
    )

    assert len(result.groups) == 1
    assert result.groups[0].semantic_name == "device_body"
    assert result.diagnostics["promoted_visual_region_count"] == 0


def test_phone_screen_requires_serial_semantic_structure_appearance_bridge() -> None:
    shape = (180, 110)
    root = np.zeros(shape, dtype=bool)
    root[8:172, 18:92] = True
    screen = np.zeros(shape, dtype=bool)
    screen[34:132, 29:81] = True
    instance_map = np.zeros(shape, dtype=np.uint16)
    instance_map[root] = 1
    instance_map[screen] = 2

    def record(index: int, semantic: str) -> PartInstance:
        mask = instance_map == index
        ys, xs = np.nonzero(mask)
        return PartInstance(
            part_id=f"device/{semantic}/center/{index:02d}",
            semantic_name=semantic,
            semantic_parent="device",
            instance_index=index,
            side="center",
            bbox_xyxy=(
                int(xs.min()),
                int(ys.min()),
                int(xs.max() + 1),
                int(ys.max() + 1),
            ),
            centroid_xy=(float(xs.mean()), float(ys.mean())),
            area_px=int(mask.sum()),
        )

    semantic_frame = root & ~screen
    candidates = [
        _phone_root(shape),
        MaskCandidate(
            semantic_name="device_screen",
            semantic_parent="device",
            mask=semantic_frame,
            score=0.72,
            source="hpid-appearance-graph/test/semantic-rerank",
            metadata={
                "candidate_key": "root:1/semantic-screen",
                "semantic_rerank_profile": "phone",
                "visual_region": True,
                "visual_region_kind": "panel",
                "semantic_reranked": True,
                "semantic_rerank_route": "base",
                "semantic_rerank_probability": 0.25,
                "semantic_rerank_margin": 0.02,
                "proposal_boundary_alignment": 0.70,
                "appearance_graph_evidence": {
                    "boundary_closure": 0.55,
                    "independent_cue_count": 2,
                    "shading_only_penalty": 0.0,
                },
            },
        ),
        MaskCandidate(
            semantic_name="device_visual_panel_01",
            semantic_parent="device",
            mask=screen,
            score=0.91,
            source="hpid-appearance-contour/closed-edge",
            metadata={
                "candidate_key": "root:1/visual-screen-surface",
                "visual_region": True,
                "visual_region_kind": "panel",
                "geometric_support": 0.84,
                "proposal_boundary_alignment": 0.96,
                "physical_region_gate": {
                    "profile": "phone",
                    "shape_structure": True,
                    "outer_boundary_contact": 0.0,
                },
                "appearance_graph_evidence": {
                    "boundary_closure": 0.86,
                    "independent_cue_count": 3,
                    "shading_only_penalty": 0.0,
                },
            },
        ),
    ]

    result = build_physical_groups(
        instance_map,
        [record(1, "device"), record(2, "device_visual_panel_01")],
        candidates=candidates,
    )
    by_name = {group.semantic_name: group for group in result.groups}

    assert set(by_name) == {"device_body", "device_screen"}
    assert result.group_map[60, 50] == by_name["device_screen"].group_index
    assert result.group_map[20, 50] == by_name["device_body"].group_index
    final_gate = result.diagnostics["final_group_three_stage_verification"]
    assert final_gate["serial"] is True
    assert final_gate["all_groups_verified"] is True
    bridge = result.diagnostics["profile_structural_decomposition"]
    assert bridge["evidence_order"] == ["semantic", "structure", "appearance"]
    assert bridge["selected_candidate_key"] == "root:1/visual-screen-surface"


def test_phone_highlight_panel_fails_third_stage_and_stays_in_body() -> None:
    shape = (180, 110)
    root = np.zeros(shape, dtype=bool)
    root[8:172, 18:92] = True
    highlight = np.zeros(shape, dtype=bool)
    highlight[34:132, 29:81] = True
    instance_map = np.zeros(shape, dtype=np.uint16)
    instance_map[root] = 1
    instance_map[highlight] = 2
    records = [
        PartInstance(
            part_id="device/device/center/01",
            semantic_name="device",
            semantic_parent="device",
            instance_index=1,
            side="center",
            bbox_xyxy=(18, 8, 92, 172),
            centroid_xy=(55.0, 90.0),
            area_px=int(np.count_nonzero(instance_map == 1)),
        ),
        PartInstance(
            part_id="device/device_visual_panel_01/center/01",
            semantic_name="device_visual_panel_01",
            semantic_parent="device",
            instance_index=2,
            side="center",
            bbox_xyxy=(29, 34, 81, 132),
            centroid_xy=(55.0, 83.0),
            area_px=int(np.count_nonzero(instance_map == 2)),
        ),
    ]
    candidates = [
        _phone_root(shape),
        MaskCandidate(
            "device_screen",
            "device",
            root & ~highlight,
            0.72,
            "hpid-appearance-graph/test/semantic-rerank",
            metadata={
                "candidate_key": "root:1/semantic-screen",
                "semantic_rerank_profile": "phone",
                "visual_region": True,
                "visual_region_kind": "panel",
                "semantic_rerank_route": "base",
                "semantic_rerank_probability": 0.25,
                "semantic_rerank_margin": 0.02,
            },
        ),
        MaskCandidate(
            "device_visual_panel_01",
            "device",
            highlight,
            0.90,
            "hpid-appearance-contour/closed-edge",
            metadata={
                "candidate_key": "root:1/highlight",
                "visual_region": True,
                "visual_region_kind": "panel",
                "geometric_support": 0.86,
                "proposal_boundary_alignment": 0.96,
                "physical_region_gate": {
                    "profile": "phone",
                    "shape_structure": True,
                    "outer_boundary_contact": 0.0,
                },
                "appearance_graph_evidence": {
                    "boundary_closure": 0.86,
                    "independent_cue_count": 3,
                    "shading_only_penalty": 0.88,
                },
            },
        ),
    ]

    result = build_physical_groups(
        instance_map,
        records,
        candidates=candidates,
    )

    assert len(result.groups) == 1
    assert result.groups[0].semantic_name == "device_body"


def test_repeated_named_scene_objects_keep_distinct_groups() -> None:
    instance_map = _strip_map(2)
    records = [
        _record(1, "natural_object_rock", "natural_object"),
        _record(2, "natural_object_rock", "natural_object"),
    ]

    result = build_physical_groups(instance_map, records)

    assert len(result.groups) == 2
    assert result.groups[0].group_id != result.groups[1].group_id


def test_fast_scene_marks_every_semantic_label_for_review() -> None:
    instance_map = _strip_map(2)
    records = [
        _record(1, "character", "character"),
        _record(2, "natural_object_rock", "natural_object"),
    ]

    result = build_physical_groups(
        instance_map,
        records,
        provisional_scene_labels=True,
    )

    assert all(group.review_required for group in result.groups)
    assert all(
        group.evidence.startswith("provisional_scene_label/") for group in result.groups
    )
    assert {group.semantic_name for group in result.groups} == {"scene_object"}
    assert len({group.group_id for group in result.groups}) == 2
    assert result.diagnostics["provisional_scene_labels"] is True


def test_knife_structural_fusion_uses_shape_and_relative_material_not_red() -> None:
    height, width = 120, 320
    root = np.zeros((height, width), dtype=bool)
    root[22:98, 18:170] = True
    root[46:76, 170:302] = True
    image = np.full((height, width, 3), 18, dtype=np.uint8)
    image[root] = (118, 145, 170)
    image[46:76, 170:207] = (155, 104, 58)
    image[46:76, 207:264] = (72, 196, 116)
    image[46:76, 264:302] = (150, 98, 55)
    instance_map = np.zeros_like(root, dtype=np.uint16)
    instance_map[root & (np.indices(root.shape)[1] < 120)] = 1
    instance_map[
        root & (np.indices(root.shape)[1] >= 120) & (np.indices(root.shape)[1] < 200)
    ] = 2
    instance_map[
        root & (np.indices(root.shape)[1] >= 200) & (np.indices(root.shape)[1] < 250)
    ] = 3
    instance_map[root & (np.indices(root.shape)[1] >= 250)] = 4
    records = [
        _record(1, "tool_prop_visual_panel_01", "tool_prop"),
        _record(2, "tool_prop_visual_panel_02", "tool_prop"),
        _record(3, "tool_prop_visual_panel_03", "tool_prop"),
        _record(4, "tool_prop_visual_panel_04", "tool_prop"),
    ]

    result = build_physical_groups(
        instance_map,
        records,
        candidates=[_knife_root(root.shape)],
        image=Image.fromarray(image),
    )

    assert result.diagnostics["knife_inventory_complete"] is True
    assert len(result.groups) == 3
    wrap = next(
        group for group in result.groups if group.semantic_name == "tool_prop_wrap"
    )
    blade = next(
        group for group in result.groups if group.semantic_name == "tool_prop_blade"
    )
    assert 215 <= wrap.centroid_xy[0] <= 255
    assert blade.centroid_xy[0] < wrap.centroid_xy[0]
    assert np.array_equal(result.group_map > 0, root)


def _geometry_record(
    instance_map: np.ndarray,
    index: int,
    semantic: str,
    parent: str,
) -> PartInstance:
    mask = instance_map == index
    ys, xs = np.nonzero(mask)
    return PartInstance(
        part_id=f"{parent}/{semantic}/center/{index:02d}",
        semantic_name=semantic,
        semantic_parent=parent,
        instance_index=index,
        side="center",
        bbox_xyxy=(
            int(xs.min()),
            int(ys.min()),
            int(xs.max() + 1),
            int(ys.max() + 1),
        ),
        centroid_xy=(float(xs.mean()), float(ys.mean())),
        area_px=len(xs),
    )


def _profile_candidate(
    semantic: str,
    parent: str,
    mask: np.ndarray,
    *,
    profile: str,
    maximum_instances: int,
) -> MaskCandidate:
    return MaskCandidate(
        semantic,
        parent,
        mask,
        0.72,
        "grounded-sam2/test/profile-refine",
        metadata={
            "candidate_key": f"root:1/{semantic}",
            "selected_part_profile": profile,
            "maximum_instances": maximum_instances,
            "root_area_fraction": float(mask.mean()),
        },
    )


def test_large_outer_surface_cannot_be_exported_as_container_inner() -> None:
    shape = (100, 100)
    root = np.zeros(shape, dtype=bool)
    root[10:90, 10:90] = True
    false_inner = np.zeros(shape, dtype=bool)
    false_inner[10:88, 18:82] = True
    instance_map = np.zeros(shape, dtype=np.uint16)
    instance_map[root] = 1
    instance_map[false_inner] = 2
    records = [
        _geometry_record(instance_map, 1, "container", "container"),
        _geometry_record(instance_map, 2, "container_inner", "container_body"),
    ]
    candidates = [
        MaskCandidate(
            "container",
            "container",
            root,
            0.9,
            "test/root",
            metadata={"selected_part_profile": "bottle_jar"},
        ),
        _profile_candidate(
            "container_inner",
            "container_body",
            false_inner,
            profile="bottle_jar",
            maximum_instances=2,
        ),
    ]

    result = build_physical_groups(instance_map, records, candidates=candidates)

    assert {group.semantic_name for group in result.groups} == {"container_body"}
    rows = result.diagnostics["three_stage_candidate_verification"]["candidates"]
    inner = next(row for row in rows if row["semantic_name"] == "container_inner")
    assert inner["stage_2_structure"]["verified"] is False
    assert inner["stage_2_structure"]["reason"] == "internal_part_consumes_root"


def test_open_container_can_keep_a_large_visible_inner_surface() -> None:
    shape = (120, 160)
    root = np.zeros(shape, dtype=bool)
    root[10:110, 12:148] = True
    inner = np.zeros(shape, dtype=bool)
    inner[18:102, 24:136] = True
    instance_map = np.zeros(shape, dtype=np.uint16)
    instance_map[root] = 1
    instance_map[inner] = 2
    records = [
        _geometry_record(instance_map, 1, "container", "container"),
        _geometry_record(instance_map, 2, "container_inner", "container_body"),
    ]
    candidates = [
        MaskCandidate(
            "container",
            "container",
            root,
            0.9,
            "test/root",
            metadata={"selected_part_profile": "open_container"},
        ),
        _profile_candidate(
            "container_inner",
            "container_body",
            inner,
            profile="open_container",
            maximum_instances=1,
        ),
    ]

    result = build_physical_groups(instance_map, records, candidates=candidates)

    assert any(group.semantic_name == "container_inner" for group in result.groups)
    rows = result.diagnostics["three_stage_candidate_verification"]["candidates"]
    verified = next(row for row in rows if row["semantic_name"] == "container_inner")
    assert verified["stage_2_structure"]["semantic_shape_consistency"][
        "accepted"
    ] is True


def test_round_region_mislabeled_as_fork_recovers_as_second_wheel() -> None:
    shape = (120, 260)
    yy, xx = np.indices(shape)
    left = (xx - 62) ** 2 + (yy - 72) ** 2 <= 34**2
    right = (xx - 198) ** 2 + (yy - 72) ** 2 <= 34**2
    bridge = (yy >= 48) & (yy < 63) & (xx >= 62) & (xx <= 198)
    root = left | right | bridge
    instance_map = np.zeros(shape, dtype=np.uint16)
    instance_map[root] = 1
    instance_map[left] = 2
    instance_map[right] = 3
    records = [
        _geometry_record(instance_map, 1, "vehicle", "vehicle"),
        _geometry_record(instance_map, 2, "vehicle_wheel", "vehicle_body"),
        _geometry_record(instance_map, 3, "vehicle_fork", "vehicle_frame"),
    ]
    candidates = [
        MaskCandidate(
            "vehicle",
            "vehicle",
            root,
            0.9,
            "test/root",
            metadata={"selected_part_profile": "two_wheeler"},
        ),
        _profile_candidate(
            "vehicle_wheel",
            "vehicle_body",
            left,
            profile="two_wheeler",
            maximum_instances=8,
        ),
        _profile_candidate(
            "vehicle_fork",
            "vehicle_frame",
            right,
            profile="two_wheeler",
            maximum_instances=1,
        ),
    ]

    result = build_physical_groups(instance_map, records, candidates=candidates)

    wheel_groups = [
        group for group in result.groups if group.semantic_name == "vehicle_wheel"
    ]
    assert len(wheel_groups) == 2
    assert all(group.group_id != wheel_groups[0].group_id for group in wheel_groups[1:])
    assert not any(group.semantic_name == "vehicle_fork" for group in result.groups)
    recovery = result.diagnostics["repeated_semantic_shape_recovery"]
    assert recovery["recovered_count"] == 1


def test_disconnected_repeated_part_mask_exports_independent_group_ids() -> None:
    shape = (120, 260)
    yy, xx = np.indices(shape)
    left = (xx - 62) ** 2 + (yy - 72) ** 2 <= 34**2
    right = (xx - 198) ** 2 + (yy - 72) ** 2 <= 34**2
    bridge = (yy >= 48) & (yy < 63) & (xx >= 62) & (xx <= 198)
    root = left | right | bridge
    instance_map = np.zeros(shape, dtype=np.uint16)
    instance_map[root] = 1
    instance_map[left | right] = 2
    records = [
        _geometry_record(instance_map, 1, "vehicle", "vehicle"),
        _geometry_record(instance_map, 2, "vehicle_wheel", "vehicle_body"),
    ]
    candidates = [
        MaskCandidate(
            "vehicle",
            "vehicle",
            root,
            0.9,
            "test/root",
            metadata={"selected_part_profile": "two_wheeler"},
        ),
        _profile_candidate(
            "vehicle_wheel",
            "vehicle_body",
            left,
            profile="two_wheeler",
            maximum_instances=8,
        ),
    ]

    result = build_physical_groups(instance_map, records, candidates=candidates)

    wheel_groups = [
        group for group in result.groups if group.semantic_name == "vehicle_wheel"
    ]
    assert len(wheel_groups) == 2
    assert len({group.group_id for group in wheel_groups}) == 2
    split = result.diagnostics["repeated_instance_component_split"]
    assert split["split_group_count"] == 1


def test_fragmented_label_does_not_create_repeated_editable_groups() -> None:
    shape = (100, 180)
    root = np.zeros(shape, dtype=bool)
    root[12:88, 12:168] = True
    left = np.zeros(shape, dtype=bool)
    left[34:66, 42:70] = True
    right = np.zeros(shape, dtype=bool)
    right[34:66, 110:138] = True
    label = left | right
    instance_map = np.zeros(shape, dtype=np.uint16)
    instance_map[root] = 1
    instance_map[label] = 2
    records = [
        _geometry_record(instance_map, 1, "container", "container"),
        _geometry_record(instance_map, 2, "container_label", "container_body"),
    ]
    candidates = [
        MaskCandidate(
            "container",
            "container",
            root,
            0.9,
            "test/root",
            metadata={"selected_part_profile": "bottle_jar"},
        ),
        _profile_candidate(
            "container_label",
            "container_body",
            label,
            profile="bottle_jar",
            maximum_instances=4,
        ),
    ]

    result = build_physical_groups(instance_map, records, candidates=candidates)

    label_groups = [
        group for group in result.groups if group.semantic_name == "container_label"
    ]
    assert len(label_groups) == 1
    split = result.diagnostics["repeated_instance_component_split"]
    assert split["split_group_count"] == 0


def _residual_root_candidate(profile: str, domain: str, root: np.ndarray) -> MaskCandidate:
    return MaskCandidate(
        domain,
        domain,
        root,
        0.95,
        "test/root",
        metadata={
            "candidate_key": "root:1",
            "selected_part_profile": profile,
        },
    )


def _residual_part_candidate(
    profile: str,
    semantic: str,
    parent: str,
    mask: np.ndarray,
    *,
    key: str,
    score: float = 0.72,
) -> MaskCandidate:
    return MaskCandidate(
        semantic,
        parent,
        mask,
        score,
        "grounded-sam2/test/profile-refine",
        metadata={
            "candidate_key": key,
            "selected_part_profile": profile,
            "maximum_instances": 1,
            "root_area_fraction": float(mask.mean()),
        },
    )


def test_bottle_profile_keeps_cap_and_label_but_merges_surface_transitions() -> None:
    shape = (160, 80)
    root = np.zeros(shape, dtype=bool)
    root[10:150, 16:64] = True
    lid = np.zeros(shape, dtype=bool)
    lid[10:28, 26:54] = True
    label = np.zeros(shape, dtype=bool)
    label[72:118, 22:58] = True
    shoulder = np.zeros(shape, dtype=bool)
    shoulder[28:45, 20:60] = True
    instance_map = np.zeros(shape, dtype=np.uint16)
    instance_map[root] = 1
    candidates = (
        _residual_root_candidate("bottle_jar", "container", root),
        _residual_part_candidate(
            "bottle_jar", "container_lid", "container_body", lid, key="lid"
        ),
        _residual_part_candidate(
            "bottle_jar", "container_label", "container_body", label, key="label"
        ),
        _residual_part_candidate(
            "bottle_jar",
            "container_shoulder",
            "container_body",
            shoulder,
            key="shoulder",
        ),
    )

    masks, diagnostics = _semantic_seeded_profile_masks(
        instance_map, candidates, None, profile="bottle_jar"
    )

    assert masks is not None
    assert set(masks) == {"container_body", "container_label", "container_lid"}
    assert np.all(masks["container_body"][shoulder])
    assert np.array_equal(np.logical_or.reduce(list(masks.values())), root)
    assert diagnostics["photometric_regions_can_create_ids"] is False


def test_simple_object_rejects_flooded_neck_and_recovers_cap_adjacent_neck() -> None:
    shape = (150, 90)
    root = np.zeros(shape, dtype=bool)
    root[8:142, 15:75] = True
    cap = np.zeros(shape, dtype=bool)
    cap[8:30, 34:58] = True
    collar = np.zeros(shape, dtype=bool)
    collar[20:48, 28:64] = True
    flooded_neck = np.zeros(shape, dtype=bool)
    flooded_neck[45:132, 18:72] = True
    label = np.zeros(shape, dtype=bool)
    label[78:118, 24:66] = True
    instance_map = np.zeros(shape, dtype=np.uint16)
    instance_map[root] = 1
    visual = MaskCandidate(
        "daily_object_visual_panel_01",
        "daily_object",
        collar,
        0.80,
        "sam2-amg/test/point-grid",
        metadata={"candidate_key": "visual-collar", "visual_region": True},
    )
    candidates = (
        _residual_root_candidate("simple_object", "daily_object", root),
        _residual_part_candidate(
            "simple_object", "daily_object_cap", "daily_object_body", cap, key="cap"
        ),
        _residual_part_candidate(
            "simple_object",
            "daily_object_neck",
            "daily_object_body",
            flooded_neck,
            key="bad-neck",
        ),
        _residual_part_candidate(
            "simple_object",
            "daily_object_label",
            "daily_object_body",
            label,
            key="label",
        ),
        visual,
    )

    masks, diagnostics = _semantic_seeded_profile_masks(
        instance_map, candidates, None, profile="simple_object"
    )

    assert masks is not None
    assert "daily_object_neck" in masks
    assert np.count_nonzero(masks["daily_object_neck"]) < np.count_nonzero(
        flooded_neck
    )
    assert diagnostics["cap_neck_structure"]["status"] == "completed"
    assert np.array_equal(np.logical_or.reduce(list(masks.values())), root)


def test_chair_mid_axis_wide_stretcher_is_resolved_as_seat() -> None:
    shape = (160, 110)
    root = np.zeros(shape, dtype=bool)
    root[8:152, 12:98] = True
    backrest = np.zeros(shape, dtype=bool)
    backrest[12:62, 20:90] = True
    mislabeled_seat = np.zeros(shape, dtype=bool)
    mislabeled_seat[78:98, 24:88] = True
    instance_map = np.zeros(shape, dtype=np.uint16)
    instance_map[root] = 1
    candidates = (
        _residual_root_candidate("chair", "furniture", root),
        _residual_part_candidate(
            "chair",
            "furniture_backrest",
            "furniture_body",
            backrest,
            key="backrest",
        ),
        _residual_part_candidate(
            "chair",
            "furniture_stretcher",
            "furniture_leg",
            mislabeled_seat,
            key="stretcher",
            score=0.36,
        ),
    )

    masks, diagnostics = _semantic_seeded_profile_masks(
        instance_map, candidates, None, profile="chair"
    )

    assert masks is not None
    assert set(masks) == {
        "furniture_backrest",
        "furniture_frame",
        "furniture_seat",
    }
    assert np.all(masks["furniture_seat"][mislabeled_seat])
    row = next(
        row
        for row in diagnostics["candidates"]
        if row["semantic_name"] == "furniture_stretcher"
    )
    assert row["structural_reason"] == "mid_axis_wide_surface_rescue"


def test_scissors_consensus_rejects_whole_object_blade_proposal() -> None:
    shape = (100, 240)
    root = np.zeros(shape, dtype=bool)
    root[18:82, 10:230] = True
    correct_blade = np.zeros(shape, dtype=bool)
    correct_blade[24:58, 150:224] = True
    flooded_blade = root.copy()
    flooded_blade[:, :135] = False
    flooded_blade[18:82, 85:230] = True
    pivot = np.zeros(shape, dtype=bool)
    pivot[45:53, 140:148] = True
    instance_map = np.zeros(shape, dtype=np.uint16)
    instance_map[root] = 1
    candidates: list[MaskCandidate] = [
        _residual_root_candidate("scissors_pliers", "tool_prop", root),
        _residual_part_candidate(
            "scissors_pliers",
            "tool_prop_blade",
            "tool_prop_body",
            flooded_blade,
            key="blade-flooded",
        ),
    ]
    for index in range(3):
        candidates.append(
            _residual_part_candidate(
                "scissors_pliers",
                "tool_prop_blade",
                "tool_prop_body",
                correct_blade,
                key=f"blade-correct-{index}",
            )
        )
    candidates.append(
        _residual_part_candidate(
            "scissors_pliers",
            "tool_prop_pivot",
            "tool_prop_body",
            pivot,
            key="pivot",
        )
    )

    masks, _ = _semantic_seeded_profile_masks(
        instance_map, tuple(candidates), None, profile="scissors_pliers"
    )

    assert masks is not None
    assert np.array_equal(masks["tool_prop_blade"], correct_blade)
    assert np.all(masks["tool_prop_handle"][root & ~correct_blade & ~pivot])


def test_vehicle_windshield_is_refined_only_inside_roof_hood_slot() -> None:
    shape = (150, 200)
    root = np.zeros(shape, dtype=bool)
    root[12:140, 20:180] = True
    roof = np.zeros(shape, dtype=bool)
    roof[12:28, 45:155] = True
    hood = np.zeros(shape, dtype=bool)
    hood[88:112, 38:162] = True
    headlight_left = np.zeros(shape, dtype=bool)
    headlight_left[102:116, 30:48] = True
    headlight_right = np.zeros(shape, dtype=bool)
    headlight_right[102:116, 152:170] = True
    image = np.full((*shape, 3), 225, dtype=np.uint8)
    image[root] = (205, 205, 205)
    image[28:88, 48:152] = (42, 69, 84)
    instance_map = np.zeros(shape, dtype=np.uint16)
    instance_map[root] = 1
    candidates = (
        _residual_root_candidate("road_vehicle", "vehicle", root),
        _residual_part_candidate(
            "road_vehicle", "vehicle_roof", "vehicle_body", roof, key="roof"
        ),
        _residual_part_candidate(
            "road_vehicle", "vehicle_hood", "vehicle_body", hood, key="hood"
        ),
        _residual_part_candidate(
            "road_vehicle",
            "vehicle_headlight",
            "vehicle_body",
            headlight_left,
            key="headlight-left",
        ),
        _residual_part_candidate(
            "road_vehicle",
            "vehicle_headlight",
            "vehicle_body",
            headlight_right,
            key="headlight-right",
        ),
    )

    masks, diagnostics = _semantic_seeded_profile_masks(
        instance_map,
        candidates,
        Image.fromarray(image),
        profile="road_vehicle",
    )

    assert masks is not None
    assert "vehicle_windshield" in masks
    windshield_y, _ = np.nonzero(masks["vehicle_windshield"])
    assert windshield_y.min() >= 25
    assert windshield_y.max() < 92
    assert diagnostics["windshield_structure"]["status"] == "completed"
    assert diagnostics["appearance_can_create_ids"] is False


def test_verified_label_seed_recovers_wrapped_band_from_boundary_support() -> None:
    shape = (150, 90)
    root = np.zeros(shape, dtype=bool)
    root[8:142, 15:75] = True
    label_seed = np.zeros(shape, dtype=bool)
    label_seed[86:106, 50:66] = True
    broad_support = np.zeros(shape, dtype=bool)
    broad_support[70:120, 18:55] = True
    instance_map = np.zeros(shape, dtype=np.uint16)
    instance_map[root] = 1
    image = np.full((*shape, 3), 232, dtype=np.uint8)
    image[root] = (166, 112, 70)
    image[70:120, 15:75] = (48, 72, 104)
    support = MaskCandidate(
        "daily_object_neck",
        "daily_object_body",
        broad_support,
        0.72,
        "sam2-amg/test/point-grid",
        metadata={
            "candidate_key": "broad-boundary-support",
            "selected_part_profile": "simple_object",
            "root_area_fraction": float(broad_support.mean()),
            "proposal_boundary_alignment": 0.88,
            "appearance_graph_evidence": {
                "boundary_closure": 0.62,
                "shading_only_penalty": 0.05,
            },
        },
    )
    candidates = (
        _residual_root_candidate("simple_object", "daily_object", root),
        _residual_part_candidate(
            "simple_object",
            "daily_object_label",
            "daily_object_body",
            label_seed,
            key="label-seed",
        ),
        support,
    )

    masks, diagnostics = _semantic_seeded_profile_masks(
        instance_map,
        candidates,
        Image.fromarray(image),
        profile="simple_object",
    )

    assert masks is not None
    assert np.all(masks["daily_object_label"][95, 15:75])
    extension = diagnostics["candidate_consensus"]["daily_object_label"][
        "visual_boundary_extension"
    ]
    assert extension["wrapped_band_completed"] is True
    assert extension["appearance_can_create_ids"] is False


def test_chair_verified_seat_expands_to_complete_cushion_surface() -> None:
    shape = (160, 110)
    root = np.zeros(shape, dtype=bool)
    root[8:152, 12:98] = True
    backrest = np.zeros(shape, dtype=bool)
    backrest[12:62, 20:90] = True
    seat_seed = np.zeros(shape, dtype=bool)
    seat_seed[86:99, 26:82] = True
    instance_map = np.zeros(shape, dtype=np.uint16)
    instance_map[root] = 1
    image = np.full((*shape, 3), 230, dtype=np.uint8)
    image[root] = (120, 78, 42)
    image[72:103, 20:90] = (134, 142, 150)
    candidates = (
        _residual_root_candidate("chair", "furniture", root),
        _residual_part_candidate(
            "chair",
            "furniture_backrest",
            "furniture_body",
            backrest,
            key="backrest",
        ),
        _residual_part_candidate(
            "chair",
            "furniture_stretcher",
            "furniture_leg",
            seat_seed,
            key="seat-seed",
            score=0.36,
        ),
    )

    masks, diagnostics = _semantic_seeded_profile_masks(
        instance_map,
        candidates,
        Image.fromarray(image),
        profile="chair",
    )

    assert masks is not None
    assert masks["furniture_seat"][76, 50]
    assert diagnostics["chair_seat_structure"]["status"] == "completed"
    assert diagnostics["chair_seat_structure"]["appearance_can_create_ids"] is False


def test_road_vehicle_recovers_paired_wheels_and_keeps_grille_separate() -> None:
    shape = (160, 200)
    root = np.zeros(shape, dtype=bool)
    root[15:120, 20:180] = True
    root[108:145, 25:55] = True
    root[108:145, 145:175] = True
    roof = np.zeros(shape, dtype=bool)
    roof[15:31, 45:155] = True
    hood = np.zeros(shape, dtype=bool)
    hood[84:105, 38:162] = True
    grille = np.zeros(shape, dtype=bool)
    grille[96:106, 72:128] = True
    headlight_left = np.zeros(shape, dtype=bool)
    headlight_left[96:108, 30:50] = True
    headlight_right = np.zeros(shape, dtype=bool)
    headlight_right[96:108, 150:170] = True
    instance_map = np.zeros(shape, dtype=np.uint16)
    instance_map[root] = 1
    image = np.full((*shape, 3), 235, dtype=np.uint8)
    image[root] = (205, 208, 212)
    image[31:84, 48:152] = (44, 65, 82)
    image[108:145, 25:55] = (24, 25, 28)
    image[108:145, 145:175] = (24, 25, 28)
    candidates = (
        _residual_root_candidate("road_vehicle", "vehicle", root),
        _residual_part_candidate(
            "road_vehicle", "vehicle_roof", "vehicle_body", roof, key="roof"
        ),
        _residual_part_candidate(
            "road_vehicle", "vehicle_hood", "vehicle_body", hood, key="hood"
        ),
        _residual_part_candidate(
            "road_vehicle",
            "vehicle_grille",
            "vehicle_body",
            grille,
            key="grille",
        ),
        _residual_part_candidate(
            "road_vehicle",
            "vehicle_headlight",
            "vehicle_body",
            headlight_left,
            key="headlight-left",
        ),
        _residual_part_candidate(
            "road_vehicle",
            "vehicle_headlight",
            "vehicle_body",
            headlight_right,
            key="headlight-right",
        ),
    )

    masks, diagnostics = _semantic_seeded_profile_masks(
        instance_map,
        candidates,
        Image.fromarray(image),
        profile="road_vehicle",
    )

    assert masks is not None
    assert "vehicle_grille" in masks
    assert "vehicle_wheel" in masks
    count, _labels = cv2.connectedComponents(
        masks["vehicle_wheel"].astype(np.uint8)
    )
    assert count == 3
    assert diagnostics["wheel_pair_structure"]["status"] == "completed"
    assert diagnostics["wheel_pair_structure"]["appearance_can_create_ids"] is False
