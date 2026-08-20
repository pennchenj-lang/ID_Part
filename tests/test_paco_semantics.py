from hpid_split.paco_semantics import (
    canonical_part_token,
    canonical_semantic_name,
    normalize_paco_name,
)


def test_paco_semantics_canonicalize_modern_device_aliases() -> None:
    assert normalize_paco_name("Mouse (computer equipment)") == (
        "mouse_computer_equipment"
    )
    assert canonical_part_token("left button", "device") == "button"
    assert canonical_part_token("scroll-wheel", "device") == "wheel"
    assert canonical_part_token("wire", "device") == "cable"


def test_paco_semantics_use_category_specific_object_part_mapping() -> None:
    assert (
        canonical_semantic_name(
            "bottom",
            "tool_prop",
            object_category="pan_(for_cooking)",
        )
        == "tool_prop_pan_body"
    )
    assert (
        canonical_semantic_name(
            "bottom",
            "container",
            object_category="plate",
        )
        == "container_bottom"
    )


def test_paco_semantics_reject_unknown_part_outside_profile_inventory() -> None:
    assert (
        canonical_semantic_name(
            "decorative unicorn",
            "device",
            allowed_semantics={"device_screen", "device_button"},
        )
        is None
    )


def test_category_alias_overrides_ambiguous_domain_alias() -> None:
    assert (
        canonical_part_token(
            "shoulder",
            "daily_object",
            object_category="soap",
        )
        == "shoulder"
    )
    assert canonical_part_token("shoulder", "daily_object") == "panel"
