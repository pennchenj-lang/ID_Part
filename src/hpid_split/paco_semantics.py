from __future__ import annotations

import re
from collections.abc import Collection

# PACO names the same physical part differently across object categories.  These
# aliases define the evaluation/training bridge to HPID's domain-level ontology;
# they are never exposed to inference as labels or masks.
_DOMAIN_ALIASES: dict[str, dict[str, str]] = {
    "container": {
        "cap": "lid",
        "drawing": "label",
        "foot": "base",
        "inner_body": "inner",
        "inner_side": "inner_wall",
        "loop": "handle",
        "mouth": "opening",
        "sticker": "label",
        "tapering_top": "shoulder",
        "text": "label",
        "zip": "zipper",
    },
    "daily_object": {
        "backstay": "upper",
        "bar": "buckle_bar",
        "fringes": "fringe",
        "hole": "belt_hole",
        "inner_body": "inner",
        "inner_side": "inner_lining",
        "loop": "belt_loop",
        "neckband": "collar",
        "outsole": "sole",
        "prong": "buckle_prong",
        "push_pull_cap": "cap",
        "quarter": "upper",
        "shoulder": "panel",
        "strap": "band",
        "toe_box": "toe",
        "vamp": "upper",
        "yoke": "panel",
    },
    "device": {
        "back": "back_panel",
        "back_cover": "back_panel",
        "bottom": "bottom_panel",
        "case": "body",
        "door_handle": "handle",
        "ear_pads": "earpad",
        "food_cup": "jar",
        "hand": "clock_hand",
        "housing": "body",
        "left_button": "button",
        "light": "bulb",
        "pipe": "rod",
        "seal_ring": "ring",
        "shade_cap": "shade",
        "shade_inner_side": "shade",
        "side_button": "button",
        "string": "chain",
        "switch": "button",
        "right_button": "button",
        "scroll_wheel": "wheel",
        "side": "side_panel",
        "strap": "band",
        "time_display": "screen",
        "top": "top_panel",
        "wire": "cable",
    },
    "furniture": {
        "arm": "armrest",
        "back": "backrest",
        "footrest": "stretcher",
        "step": "stretcher",
    },
    "tool_prop": {
        "hole": "sound_hole",
        "key": "tuning_key",
        "screw": "pivot",
    },
    "vehicle": {
        "logo": "badge",
        "saddle": "seat",
        "splashboard": "dashboard",
        "steeringwheel": "steering_wheel",
        "turnsignal": "turn_signal",
        "wheel": "tire",
        "windowpane": "window",
    },
}

_CATEGORY_ALIASES: dict[str, dict[str, str]] = {
    "basket": {
        "cover": "lid",
        "inner_side": "inner_wall",
        "side": "outer_side",
    },
    "blender": {
        "cover": "lid",
        "cup": "jar",
        "vapour_cover": "lid",
    },
    "clock": {"decoration": "logo"},
    "box": {"inner_side": "side"},
    "carton": {"inner_side": "side"},
    "crate": {"inner_side": "side"},
    "handbag": {
        "inner_body": "inner",
        "rim": "opening",
    },
    "helmet": {"inner_side": "inner_lining"},
    "hat": {"inner_side": "inner_lining"},
    "microwave_oven": {"inner_side": "side_panel"},
    "pan_for_cooking": {
        "base": "pan_body",
        "bottom": "pan_body",
        "inner_side": "pan_body",
    },
    "plastic_bag": {
        "hem": "rim",
        "inner_body": "inner",
        "text": "label",
    },
    "pliers": {
        "blade": "jaw",
        "joint": "pivot",
    },
    "spoon": {
        "neck": "handle",
        "tip": "bowl",
    },
    "soap": {"shoulder": "shoulder"},
    "trash_can": {
        "hole": "opening",
    },
    "watch": {"window": "screen"},
}


def normalize_paco_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def canonical_part_token(
    part_name: str,
    expected_domain: str,
    *,
    object_category: str | None = None,
) -> str:
    """Map a PACO part label to the HPID domain vocabulary.

    The mapping is intentionally deterministic and category-aware.  It does not
    inspect pixels and therefore cannot leak benchmark annotations into model
    inference.
    """

    token = normalize_paco_name(part_name)
    category = normalize_paco_name(object_category or "")
    category_token = _CATEGORY_ALIASES.get(category, {}).get(token)
    if category_token is not None:
        return category_token
    return _DOMAIN_ALIASES.get(expected_domain, {}).get(token, token)


def canonical_semantic_name(
    part_name: str,
    expected_domain: str,
    *,
    object_category: str | None = None,
    allowed_semantics: Collection[str] | None = None,
) -> str | None:
    """Return one canonical HPID semantic, or ``None`` when it is unsupported."""

    token = canonical_part_token(
        part_name,
        expected_domain,
        object_category=object_category,
    )
    semantic_name = f"{expected_domain}_{token}"
    if allowed_semantics is None:
        return semantic_name
    allowed = set(allowed_semantics)
    return semantic_name if semantic_name in allowed else None


def domain_aliases() -> dict[str, dict[str, str]]:
    """Return a defensive copy for audit reports and external adapters."""

    return {domain: dict(values) for domain, values in _DOMAIN_ALIASES.items()}
