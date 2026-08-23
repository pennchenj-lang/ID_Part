import json
from pathlib import Path

import pytest

from hpid_split.prompt_bank import (
    DomainPrompt,
    PartProfile,
    PartProfileOverride,
    PartPrompt,
    PartSubtype,
    PromptBank,
)

REPOSITORY_PROMPT_BANK = (
    Path(__file__).resolve().parents[1] / "configs" / "general_asset_prompts.json"
)


def test_domain_match_rejects_concatenated_ambiguous_part_label() -> None:
    domain = DomainPrompt(
        name="tool_prop",
        root_prompts=("tool",),
        parts=(
            PartPrompt("tool_prop_barrel", ("barrel",)),
            PartPrompt("tool_prop_guard", ("guard",)),
        ),
    )

    assert domain.match_part("tool prop barrel prop guard") is None
    assert domain.match_part("barrel") == domain.parts[0]


def test_firearm_inventory_rejects_implausibly_large_barrel_and_grip_masks() -> None:
    bank = PromptBank.from_json(REPOSITORY_PROMPT_BANK)
    domain = next(item for item in bank.domains if item.name == "tool_prop")
    parts, profile, _ = domain.select_parts("assault rifle")
    by_name = {part.semantic_name: part for part in parts}

    assert profile == "firearm"
    assert by_name["tool_prop_barrel"].maximum_parent_fraction == 0.22
    assert by_name["tool_prop_grip"].maximum_parent_fraction == 0.18
    assert by_name["tool_prop_stock"].maximum_parent_fraction == 0.5
    assert by_name["tool_prop_stock"].axis_position == 0.85
    assert by_name["tool_prop_magazine"].axis_position == -0.15
    assert by_name["tool_prop_grip"].axis_position == 0.28


def test_globe_inventory_requires_targeted_semantic_mask_discovery() -> None:
    bank = PromptBank.from_json(REPOSITORY_PROMPT_BANK)
    domain = next(item for item in bank.domains if item.name == "device")
    profile = next(item for item in domain.part_profiles if item.name == "globe")

    assert profile.requires_grounded_refinement is True
    assert set(profile.part_semantics) == {
        "device_globe_sphere",
        "device_globe_meridian_ring",
        "device_globe_axis",
        "device_globe_stem",
        "device_base",
    }
    microwave = next(item for item in domain.part_profiles if item.name == "microwave")
    assert microwave.requires_grounded_refinement is True
    character = next(item for item in bank.domains if item.name == "character")
    humanoid = next(item for item in character.part_profiles if item.name == "humanoid")
    assert humanoid.requires_grounded_refinement is True
    assert "character_upper_clothing" in humanoid.part_semantics
    assert "character_bag" not in humanoid.part_semantics
    character_bag = next(
        part for part in character.parts if part.semantic_name == "character_bag"
    )
    assert character_bag.maximum_parent_fraction == 0.16


def test_power_drill_and_jackhammer_use_distinct_part_inventories() -> None:
    bank = PromptBank.from_json(REPOSITORY_PROMPT_BANK)
    domain = next(item for item in bank.domains if item.name == "tool_prop")

    drill_parts, drill_profile, _ = domain.select_parts("electric power drill")
    drill_by_name = {part.semantic_name: part for part in drill_parts}
    jackhammer_parts, jackhammer_profile, _ = domain.select_parts(
        "pneumatic jackhammer demolition hammer"
    )
    jackhammer_names = {part.semantic_name for part in jackhammer_parts}

    assert drill_profile == "drill"
    assert drill_by_name["tool_prop_handle"].maximum_parent_fraction == 0.42
    assert "tool_prop_power_trigger" in drill_by_name
    assert "tool_prop_trigger" not in drill_by_name
    assert jackhammer_profile == "jackhammer"
    assert jackhammer_names == {
        "tool_prop_body",
        "tool_prop_auxiliary_handle",
        "tool_prop_power_trigger",
        "tool_prop_tool_holder",
        "tool_prop_chisel",
    }
    assert "tool_prop_handle" not in jackhammer_names


def test_knife_profile_exposes_one_ordered_blade_and_handle() -> None:
    bank = PromptBank.from_json(REPOSITORY_PROMPT_BANK)
    domain = next(item for item in bank.domains if item.name == "tool_prop")

    parts, profile, _ = domain.select_parts("knife")
    by_name = {part.semantic_name: part for part in parts}

    assert profile == "knife"
    assert by_name["tool_prop_blade"].maximum_instances == 1
    assert by_name["tool_prop_handle"].maximum_instances == 1
    assert by_name["tool_prop_blade"].maximum_parent_fraction == 0.94
    assert by_name["tool_prop_blade"].axis_position == -0.55
    assert by_name["tool_prop_handle"].axis_position == 0.85
    assert by_name["tool_prop_handle"].topology_anchor == "tool_prop_blade"
    assert by_name["tool_prop_handle"].topology_relation == "terminal_complement"
    assert by_name["tool_prop_guard"].detail is True


def test_box_profile_caps_repeated_structural_parts() -> None:
    bank = PromptBank.from_json(REPOSITORY_PROMPT_BANK)
    domain = next(item for item in bank.domains if item.name == "container")

    parts, profile, _ = domain.select_parts("crate")
    by_name = {part.semantic_name: part for part in parts}

    assert profile == "box"
    assert by_name["container_side"].maximum_instances == 4
    assert by_name["container_lid"].maximum_instances == 1
    assert by_name["container_handle"].maximum_instances == 2


def test_laptop_keyboard_explicitly_opts_into_dense_region_recovery() -> None:
    bank = PromptBank.from_json(REPOSITORY_PROMPT_BANK)
    domain = next(item for item in bank.domains if item.name == "device")
    parts, profile, _ = domain.select_parts("laptop computer")
    keyboard = next(part for part in parts if part.semantic_name == "device_keyboard")

    assert profile == "laptop"
    assert keyboard.dense_fallback is True
    assert keyboard.detail is False
    assert keyboard.minimum_parent_fraction == 0.04
    assert keyboard.maximum_parent_fraction == 0.45


def test_microwave_profile_applies_category_conditioned_panel_geometry() -> None:
    bank = PromptBank.from_json(REPOSITORY_PROMPT_BANK)
    domain = next(item for item in bank.domains if item.name == "device")

    parts, profile_name, diagnostics = domain.select_parts("microwave oven")
    by_name = {part.semantic_name: part for part in parts}

    assert profile_name == "microwave"
    assert by_name["device_door"].minimum_parent_fraction == 0.18
    assert by_name["device_door"].maximum_parent_fraction == 0.82
    assert by_name["device_screen"].maximum_parent_fraction == 0.12
    assert by_name["device_control_panel"].maximum_parent_fraction == 0.28
    assert set(diagnostics["profile_part_overrides"]) == {
        "device_door",
        "device_control_panel",
        "device_screen",
        "device_handle",
    }


def test_profile_override_rejects_an_invalid_effective_interval() -> None:
    domain = DomainPrompt(
        name="device",
        root_prompts=("microwave",),
        parts=(
            PartPrompt(
                "device_screen",
                ("screen",),
                minimum_parent_fraction=0.10,
            ),
        ),
        part_profiles=(
            PartProfile(
                "microwave",
                ("microwave",),
                ("device_screen",),
                part_overrides=(
                    PartProfileOverride("device_screen", maximum_parent_fraction=0.05),
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="invalid parent-area interval"):
        PromptBank((domain,))


def test_prompt_bank_rejects_unknown_semantic_parent() -> None:
    with pytest.raises(ValueError, match="unknown parent"):
        PromptBank.from_dict(
            {
                "domains": [
                    {
                        "name": "asset",
                        "root_prompts": ["object"],
                        "parts": [
                            {
                                "semantic_name": "asset_button",
                                "semantic_parent": "missing_body",
                                "prompts": ["button"],
                            }
                        ],
                    }
                ]
            }
        )


def test_prompt_bank_rejects_parent_cycles() -> None:
    with pytest.raises(ValueError, match="cycle"):
        PromptBank.from_dict(
            {
                "domains": [
                    {
                        "name": "asset",
                        "root_prompts": ["object"],
                        "parts": [
                            {
                                "semantic_name": "asset_a",
                                "semantic_parent": "asset_b",
                                "prompts": ["part a"],
                            },
                            {
                                "semantic_name": "asset_b",
                                "semantic_parent": "asset_a",
                                "prompts": ["part b"],
                            },
                        ],
                    }
                ]
            }
        )


def test_prompt_bank_rejects_unknown_appearance_anchor() -> None:
    with pytest.raises(ValueError, match="unknown appearance anchor"):
        PromptBank.from_dict(
            {
                "domains": [
                    {
                        "name": "asset",
                        "root_prompts": ["object"],
                        "parts": [
                            {
                                "semantic_name": "asset_detail",
                                "prompts": ["detail"],
                                "appearance_anchor": "missing_anchor",
                                "appearance_relation": "above",
                            }
                        ],
                    }
                ]
            }
        )


def test_domain_profile_selects_category_specific_parts_and_defaults() -> None:
    domain = DomainPrompt(
        name="device",
        root_prompts=("electronic device",),
        parts=(
            PartPrompt("device_body", ("body",)),
            PartPrompt("device_button", ("button",)),
            PartPrompt("device_screen", ("screen",)),
            PartPrompt("device_keyboard", ("keyboard",)),
        ),
        default_part_semantics=("device_body", "device_button"),
        part_profiles=(
            PartProfile(
                "phone",
                ("phone", "smartphone"),
                ("device_screen", "device_button"),
            ),
            PartProfile(
                "laptop",
                ("laptop", "notebook computer"),
                ("device_screen", "device_keyboard"),
            ),
        ),
    )

    parts, profile, diagnostics = domain.select_parts("smartphone")

    assert profile == "phone"
    assert {part.semantic_name for part in parts} == {
        "device_body",
        "device_button",
        "device_screen",
    }
    assert diagnostics["selection_reason"] == "root_label_match"


def test_domain_profile_uses_conservative_defaults_for_ambiguous_root() -> None:
    domain = DomainPrompt(
        name="device",
        root_prompts=("electronic device",),
        parts=(
            PartPrompt("device_body", ("body",)),
            PartPrompt("device_screen", ("screen",)),
            PartPrompt("device_keyboard", ("keyboard",)),
        ),
        default_part_semantics=("device_body",),
        part_profiles=(
            PartProfile("phone", ("phone",), ("device_screen",)),
            PartProfile("laptop", ("laptop",), ("device_keyboard",)),
        ),
    )

    parts, profile, diagnostics = domain.select_parts("electronic device")

    assert profile is None
    assert [part.semantic_name for part in parts] == ["device_body"]
    assert diagnostics["selection_reason"] == "default_inventory_fallback"


def test_isolated_profile_hint_overrides_ambiguous_detector_label() -> None:
    domain = DomainPrompt(
        name="device",
        root_prompts=("electronic device",),
        parts=(
            PartPrompt("device_body", ("body",)),
            PartPrompt("device_blade", ("fan blade",)),
            PartPrompt("device_headband", ("headphone headband",)),
        ),
        default_part_semantics=("device_body",),
        part_profiles=(
            PartProfile("fan", ("fan",), ("device_blade",)),
            PartProfile("earphone", ("earphone",), ("device_headband",)),
        ),
    )

    parts, profile, diagnostics = domain.select_parts(
        "head",
        profile_hint="earphone",
        profile_hint_source="isolated_profile_query",
    )

    assert profile == "earphone"
    assert {part.semantic_name for part in parts} == {
        "device_body",
        "device_headband",
    }
    assert diagnostics["selection_reason"] == "isolated_profile_query"


def test_prompt_bank_rejects_unknown_profile_part() -> None:
    with pytest.raises(ValueError, match="references unknown parts"):
        PromptBank.from_dict(
            {
                "domains": [
                    {
                        "name": "device",
                        "root_prompts": ["device"],
                        "parts": [
                            {
                                "semantic_name": "device_body",
                                "prompts": ["body"],
                            }
                        ],
                        "part_profiles": [
                            {
                                "name": "phone",
                                "root_hints": ["phone"],
                                "parts": ["device_missing"],
                            }
                        ],
                    }
                ]
            }
        )


def test_prompt_bank_json_include_extends_existing_profile(tmp_path: Path) -> None:
    extension = tmp_path / "extension.json"
    extension.write_text(
        json.dumps(
            {
                "domain_extensions": [
                    {
                        "name": "device",
                        "parts": [
                            {
                                "semantic_name": "device_dial",
                                "prompts": ["dial"],
                                "semantic_parent": "device_body",
                            }
                        ],
                        "profile_parts": {"phone": ["device_dial"]},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    base = tmp_path / "base.json"
    base.write_text(
        json.dumps(
            {
                "include": [extension.name],
                "domains": [
                    {
                        "name": "device",
                        "root_prompts": ["device"],
                        "parts": [
                            {
                                "semantic_name": "device_body",
                                "prompts": ["body"],
                            }
                        ],
                        "part_profiles": [
                            {
                                "name": "phone",
                                "root_hints": ["phone"],
                                "parts": ["device_body"],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    bank = PromptBank.from_json(base)

    domain = bank.domains[0]
    assert {part.semantic_name for part in domain.parts} == {
        "device_body",
        "device_dial",
    }
    assert domain.part_profiles[0].part_semantics == (
        "device_body",
        "device_dial",
    )


def test_profile_extension_does_not_pollute_narrow_subtype_inventory(
    tmp_path: Path,
) -> None:
    extension = tmp_path / "extension.json"
    extension.write_text(
        json.dumps(
            {
                "domain_extensions": [
                    {
                        "name": "device",
                        "parts": [
                            {
                                "semantic_name": "device_dial",
                                "prompts": ["dial"],
                                "semantic_parent": "device_body",
                            }
                        ],
                        "profile_parts": {"phone": ["device_dial"]},
                        "subtype_parts": {},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    base = tmp_path / "base.json"
    base.write_text(
        json.dumps(
            {
                "include": [extension.name],
                "domains": [
                    {
                        "name": "device",
                        "root_prompts": ["device"],
                        "parts": [
                            {
                                "semantic_name": "device_body",
                                "prompts": ["body"],
                            }
                        ],
                        "part_profiles": [
                            {
                                "name": "phone",
                                "root_hints": ["phone", "mobile handset"],
                                "parts": ["device_body"],
                                "subtypes": [
                                    {
                                        "name": "mobile",
                                        "root_hints": ["mobile handset"],
                                        "parts": ["device_body"],
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    domain = PromptBank.from_json(base).domains[0]
    broad_parts, _, _ = domain.select_parts("phone")
    subtype_parts, _, diagnostics = domain.select_parts("mobile handset")

    assert {part.semantic_name for part in broad_parts} == {
        "device_body",
        "device_dial",
    }
    assert {part.semantic_name for part in subtype_parts} == {"device_body"}
    assert diagnostics["selected_subtype"] == "mobile"


def test_crate_subtype_excludes_carton_shoulder_extension() -> None:
    bank = PromptBank.from_json(REPOSITORY_PROMPT_BANK)
    domain = next(item for item in bank.domains if item.name == "container")

    crate_parts, profile, diagnostics = domain.select_parts("crate")
    names = {part.semantic_name for part in crate_parts}

    assert profile == "box"
    assert diagnostics["selected_subtype"] == "crate"
    assert "container_inner_wall" in names
    assert "container_bottom" in names
    assert "container_shoulder" not in names
    assert "container_outer_side" not in names


def test_carton_subtype_keeps_physical_outer_side_without_crate_handle() -> None:
    bank = PromptBank.from_json(REPOSITORY_PROMPT_BANK)
    domain = next(item for item in bank.domains if item.name == "container")

    parts, profile, diagnostics = domain.select_parts("carton")
    names = {part.semantic_name for part in parts}

    assert profile == "box"
    assert diagnostics["selected_subtype"] == "box_or_carton"
    assert "container_outer_side" in names
    assert "container_bottom" in names
    assert "container_handle" not in names


def test_plastic_bag_subtype_excludes_hard_bag_hardware() -> None:
    bank = PromptBank.from_json(REPOSITORY_PROMPT_BANK)
    domain = next(item for item in bank.domains if item.name == "container")

    parts, profile, diagnostics = domain.select_parts("plastic bag")
    names = {part.semantic_name for part in parts}

    assert profile == "bag"
    assert diagnostics["selected_subtype"] == "plastic_bag"
    assert names == {
        "container_body",
        "container_handle",
        "container_opening",
        "container_inner",
    }


def test_wallet_and_belt_subtypes_keep_distinct_part_inventories() -> None:
    bank = PromptBank.from_json(REPOSITORY_PROMPT_BANK)
    domain = next(item for item in bank.domains if item.name == "daily_object")

    belt_parts, belt_profile, belt_diagnostics = domain.select_parts("belt")
    wallet_parts, wallet_profile, wallet_diagnostics = domain.select_parts("wallet")
    belt_names = {part.semantic_name for part in belt_parts}
    wallet_names = {part.semantic_name for part in wallet_parts}

    assert belt_profile == wallet_profile == "wallet_belt"
    assert belt_diagnostics["selected_subtype"] == "belt"
    assert wallet_diagnostics["selected_subtype"] == "wallet"
    assert "daily_object_belt_hole" in belt_names
    assert "daily_object_flap" not in belt_names
    assert "daily_object_flap" in wallet_names
    assert "daily_object_belt_hole" not in wallet_names


def test_prompt_bank_extension_can_add_domain_prompts_profiles_and_domain(
    tmp_path: Path,
) -> None:
    extension = tmp_path / "extension.json"
    extension.write_text(
        json.dumps(
            {
                "new_domains": [
                    {
                        "name": "terrain",
                        "root_prompts": ["terrain"],
                        "default_part_semantics": ["terrain_ground"],
                        "parts": [
                            {
                                "semantic_name": "terrain_ground",
                                "prompts": ["ground"],
                            }
                        ],
                    }
                ],
                "domain_extensions": [
                    {
                        "name": "device",
                        "root_prompts": ["game controller"],
                        "parts": [
                            {
                                "semantic_name": "device_stick",
                                "prompts": ["analog stick"],
                                "semantic_parent": "device_body",
                            }
                        ],
                        "part_profiles": [
                            {
                                "name": "controller",
                                "root_hints": ["game controller"],
                                "scene_root_query_groups": [["game controller"]],
                                "parts": ["device_body", "device_stick"],
                            }
                        ],
                        "profile_parts": {},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    base = tmp_path / "base.json"
    base.write_text(
        json.dumps(
            {
                "include": [extension.name],
                "domains": [
                    {
                        "name": "device",
                        "root_prompts": ["device"],
                        "parts": [
                            {
                                "semantic_name": "device_body",
                                "prompts": ["body"],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    bank = PromptBank.from_json(base)

    device = next(domain for domain in bank.domains if domain.name == "device")
    terrain = next(domain for domain in bank.domains if domain.name == "terrain")
    assert "game controller" in device.root_prompts
    assert device.part_profiles[0].name == "controller"
    assert device.part_profiles[0].scene_root_query_groups == (("game controller",),)
    assert terrain.default_part_semantics == ("terrain_ground",)


def test_generic_root_prompt_has_lower_specificity_than_named_category() -> None:
    domain = DomainPrompt(
        name="device",
        root_prompts=("cell phone", "electronic device"),
        generic_root_prompts=("electronic device",),
        parts=(),
    )

    assert domain.root_label_specificity("cell phone") == 1.0
    assert domain.root_label_specificity("electronic device") == 0.35


def test_profile_matching_does_not_find_car_inside_scarf() -> None:
    road_vehicle = PartProfile("road_vehicle", ("car", "automobile"), ())
    soft_good = PartProfile("soft_good", ("scarf", "towel"), ())

    assert road_vehicle.match_score("scarf") == 0.0
    assert soft_good.match_score("red scarf") == pytest.approx(0.88)


def test_profile_query_groups_separate_objects_sharing_one_part_template() -> None:
    profile = PartProfile(
        "flatware",
        ("plate", "serving plate", "tray", "serving tray"),
        (),
        root_query_groups=(
            ("plate", "serving plate"),
            ("tray", "serving tray"),
        ),
    )

    assert profile.query_hints("tray") == ("tray", "serving tray")
    assert profile.query_hints("serving plate") == ("plate", "serving plate")


def test_profile_subtype_selects_only_parts_valid_for_the_object_kind() -> None:
    domain = DomainPrompt(
        name="device",
        root_prompts=("computer mouse", "computer keyboard"),
        parts=(
            PartPrompt("device_body", ("device body",)),
            PartPrompt("device_button", ("button",)),
            PartPrompt("device_key", ("keyboard key",)),
            PartPrompt("device_wheel", ("mouse wheel",)),
        ),
        default_part_semantics=("device_body",),
        part_profiles=(
            PartProfile(
                "computer_peripheral",
                ("computer mouse", "computer keyboard"),
                ("device_button", "device_key", "device_wheel"),
                part_subtypes=(
                    PartSubtype(
                        "mouse",
                        ("computer mouse",),
                        ("device_button", "device_wheel"),
                    ),
                    PartSubtype(
                        "keyboard",
                        ("computer keyboard",),
                        ("device_button", "device_key"),
                    ),
                ),
            ),
        ),
    )

    parts, profile, diagnostics = domain.select_parts(
        "wireless computer mouse",
        profile_hint="computer_peripheral",
        profile_hint_source="test",
    )

    assert profile == "computer_peripheral"
    assert {part.semantic_name for part in parts} == {
        "device_body",
        "device_button",
        "device_wheel",
    }
    assert diagnostics["selected_subtype"] == "mouse"
    assert diagnostics["subtype_root_hints"] == ["computer mouse"]


def test_prompt_bank_rejects_subtype_part_outside_profile_inventory() -> None:
    with pytest.raises(ValueError, match="outside profile"):
        PromptBank.from_dict(
            {
                "domains": [
                    {
                        "name": "device",
                        "root_prompts": ["mouse"],
                        "parts": [
                            {"semantic_name": "device_button", "prompts": ["button"]},
                            {"semantic_name": "device_key", "prompts": ["key"]},
                        ],
                        "part_profiles": [
                            {
                                "name": "mouse",
                                "root_hints": ["mouse"],
                                "parts": ["device_button"],
                                "subtypes": [
                                    {
                                        "name": "mouse",
                                        "root_hints": ["mouse"],
                                        "parts": ["device_key"],
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }
        )


def test_prompt_bank_rejects_unknown_generic_root_prompt() -> None:
    with pytest.raises(ValueError, match="generic root prompts outside"):
        PromptBank(
            (
                DomainPrompt(
                    name="device",
                    root_prompts=("cell phone",),
                    generic_root_prompts=("machine",),
                    parts=(),
                ),
            )
        )
