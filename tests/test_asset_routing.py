from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image

from hpid_split.asset_routing import (
    AssetRoute,
    AssetRouter,
    AssetRouterConfig,
    ProfileTextRoute,
    ProfileTextRouterConfig,
    masked_asset_view,
    reconcile_asset_routes,
    resolve_asset_domain,
    resolve_explicit_asset_prompt,
    resolve_full_image_domain_prior,
    route_profile_text_inventories,
    route_profile_text_inventory,
)
from hpid_split.cli import _apply_asset_domain_routes
from hpid_split.fusion import MaskCandidate
from hpid_split.prompt_bank import DomainPrompt, PartProfile, PromptBank


def _domain_prior_root(
    semantic_name: str,
    physical_domains: tuple[str, ...],
) -> MaskCandidate:
    return MaskCandidate(
        semantic_name,
        semantic_name,
        np.ones((24, 24), dtype=bool),
        0.56,
        "root",
        prompt=semantic_name,
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
            "physical_group_semantic_names": list(physical_domains),
        },
    )


class _FakeEncoder:
    model_name = "fake-router"

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        rows = []
        for image in images:
            mean = float(np.asarray(image, dtype=np.float32).mean())
            rows.append([1.0, 0.0] if mean > 100 else [0.0, 1.0])
        return np.asarray(rows, dtype=np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return np.asarray([[1.0, 0.0] for _ in texts], dtype=np.float32)


class _ProfileEncoder:
    model_name = "fake-profile-router"

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        return np.asarray([[1.0, 0.0] for _ in images], dtype=np.float32)

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [[0.14, 0.0] if "firearm" in text else [0.06, 0.0] for text in texts],
            dtype=np.float32,
        )


def _ambiguous_route(
    rows: tuple[dict[str, object], ...],
) -> AssetRoute:
    top = rows[0]
    return AssetRoute(
        accepted=False,
        asset_label=None,
        asset_domain=None,
        asset_profile=None,
        score=float(top["score"]),
        margin=float(top["score"]) - float(rows[1]["score"]),
        alternatives=rows,
        candidate_labels=tuple(str(row["asset_label"]) for row in rows),
        candidate_domains=tuple(
            dict.fromkeys(str(row["asset_domain"]) for row in rows)
        ),
        reason="ambiguous_candidate_set",
    )


def test_explicit_prompt_uses_token_boundaries_and_not_word_substrings() -> None:
    index = SimpleNamespace(
        labels=("car", "scarf"),
        label_metadata={
            "car": {
                "asset_domain": "vehicle",
                "asset_profile": "road_vehicle",
                "aliases": ["car", "automobile"],
            },
            "scarf": {
                "asset_domain": "daily_object",
                "asset_profile": "soft_good",
                "aliases": ["scarf"],
            },
        },
    )

    route = resolve_explicit_asset_prompt(index, "red scarf")

    assert route.accepted is True
    assert route.asset_label == "scarf"
    assert route.asset_domain == "daily_object"
    assert route.asset_profile == "soft_good"
    assert all(row["asset_label"] != "car" for row in route.alternatives)


def test_cross_view_top_label_consensus_promotes_an_exact_asset() -> None:
    full = _ambiguous_route(
        (
            {
                "asset_label": "plate",
                "asset_domain": "container",
                "asset_profile": "flatware",
                "score": 0.223,
            },
            {
                "asset_label": "pan",
                "asset_domain": "tool_prop",
                "asset_profile": "pan",
                "score": 0.208,
            },
        )
    )
    crop = _ambiguous_route(
        (
            {
                "asset_label": "plate",
                "asset_domain": "container",
                "asset_profile": "flatware",
                "score": 0.270,
            },
            {
                "asset_label": "belt",
                "asset_domain": "daily_object",
                "asset_profile": "wallet_belt",
                "score": 0.265,
            },
            {
                "asset_label": "broom",
                "asset_domain": "tool_prop",
                "asset_profile": "broom",
                "score": 0.260,
            },
        )
    )

    reconciled, diagnostics = reconcile_asset_routes(full, crop)

    assert reconciled.accepted is True
    assert reconciled.asset_label == "plate"
    assert reconciled.asset_domain == "container"
    assert reconciled.asset_profile == "flatware"
    assert diagnostics["status"] == "accepted_top_label_agreement"


def test_cross_view_rank_one_near_tie_preserves_asset_semantics() -> None:
    full = _ambiguous_route(
        (
            {
                "asset_label": "pan",
                "asset_domain": "tool_prop",
                "asset_profile": "pan",
                "score": 0.181,
            },
            {
                "asset_label": "box",
                "asset_domain": "container",
                "asset_profile": "box",
                "score": 0.178,
            },
        )
    )
    crop = _ambiguous_route(
        (
            {
                "asset_label": "knife",
                "asset_domain": "tool_prop",
                "asset_profile": "knife",
                "score": 0.251,
            },
            {
                "asset_label": "pan",
                "asset_domain": "tool_prop",
                "asset_profile": "pan",
                "score": 0.2495,
            },
        )
    )

    reconciled, diagnostics = reconcile_asset_routes(
        full,
        crop,
        root_global_proposal_rank=1,
    )

    assert reconciled.accepted is True
    assert reconciled.asset_label == "pan"
    assert diagnostics["status"] == "accepted_rank_one_near_tie"


def test_cross_view_conflict_does_not_force_the_full_image_label() -> None:
    full = _ambiguous_route(
        (
            {
                "asset_label": "stool",
                "asset_domain": "furniture",
                "asset_profile": "chair",
                "score": 0.240,
            },
            {
                "asset_label": "chair",
                "asset_domain": "furniture",
                "asset_profile": "chair",
                "score": 0.200,
            },
        )
    )
    crop = _ambiguous_route(
        (
            {
                "asset_label": "dog",
                "asset_domain": "animal",
                "asset_profile": "quadruped",
                "score": 0.310,
            },
            {
                "asset_label": "stool",
                "asset_domain": "furniture",
                "asset_profile": "chair",
                "score": 0.250,
            },
        )
    )

    reconciled, diagnostics = reconcile_asset_routes(
        full,
        crop,
        root_global_proposal_rank=1,
    )

    assert reconciled is crop
    assert reconciled.accepted is False
    assert diagnostics["status"] == "insufficient_cross_view_consensus"


def test_router_cannot_override_a_domain_missing_from_its_inventory() -> None:
    mask = np.ones((20, 20), dtype=bool)
    root = MaskCandidate(
        "character",
        "character",
        mask,
        0.9,
        "root",
        prompt="person",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
        },
    )
    route = AssetRoute(
        accepted=False,
        asset_label=None,
        asset_domain=None,
        asset_profile=None,
        score=0.22,
        margin=0.01,
        alternatives=(
            {
                "asset_label": "sweater",
                "asset_domain": "daily_object",
                "score": 0.22,
            },
            {
                "asset_label": "shoe",
                "asset_domain": "daily_object",
                "score": 0.21,
            },
        ),
        candidate_labels=("sweater", "shoe"),
        candidate_domains=("daily_object",),
        reason="ambiguous_candidate_set",
    )
    prompt_bank = PromptBank(
        (
            DomainPrompt("character", ("person",), ()),
            DomainPrompt("daily_object", ("clothing",), ()),
        )
    )

    candidates, diagnostics = _apply_asset_domain_routes(
        [root],
        [root],
        {"test::1": route},
        prompt_bank,
        config=AssetRouterConfig(),
        supported_domains={"daily_object"},
    )

    assert candidates[0].semantic_name == "character"
    assert candidates[0].metadata["asset_domain_audit_required"] is False
    assert diagnostics["corrected_domain_count"] == 0
    assert diagnostics["rows"][0]["routing_applicable"] is False
    assert (
        diagnostics["rows"][0]["domain_resolution"]["reason"]
        == "current_domain_outside_router_inventory"
    )


def test_exact_route_can_correct_a_domain_missing_from_router_inventory() -> None:
    mask = np.ones((20, 20), dtype=bool)
    root = MaskCandidate(
        "structure",
        "structure",
        mask,
        0.9,
        "root",
        prompt="object",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
        },
    )
    route = AssetRoute(
        accepted=True,
        asset_label="clock",
        asset_domain="device",
        asset_profile="clock",
        score=0.31,
        margin=0.08,
        alternatives=(
            {
                "asset_label": "clock",
                "asset_domain": "device",
                "asset_profile": "clock",
                "score": 0.31,
            },
        ),
        candidate_labels=("clock",),
        candidate_domains=("device",),
        reason="accepted_exact_label",
    )
    prompt_bank = PromptBank(
        (
            DomainPrompt("structure", ("building",), ()),
            DomainPrompt("device", ("clock",), ()),
        )
    )

    candidates, diagnostics = _apply_asset_domain_routes(
        [root],
        [root],
        {"test::1": route},
        prompt_bank,
        config=AssetRouterConfig(),
        supported_domains={"device"},
    )

    assert candidates[0].semantic_name == "device"
    assert candidates[0].metadata["resolved_object_label"] == "clock"
    assert diagnostics["corrected_domain_count"] == 1
    assert diagnostics["rows"][0]["routing_applicable"] is True


def test_ambiguous_router_cannot_replace_a_specific_detected_profile() -> None:
    mask = np.zeros((40, 30), dtype=bool)
    mask[2:38, 10:22] = True
    root = MaskCandidate(
        "device",
        "device",
        mask,
        0.43,
        "root",
        prompt="lamp",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
            "root_model_label": "lamp",
            "selected_part_profile": "lamp",
            "part_profile_specificity": 1.0,
            "profile_hint_source": "specific_root_label",
        },
    )
    route = _ambiguous_route(
        (
            {
                "asset_label": "plate",
                "asset_domain": "container",
                "asset_profile": "flatware",
                "score": 0.28,
            },
            {
                "asset_label": "cup",
                "asset_domain": "container",
                "asset_profile": "drinkware",
                "score": 0.27,
            },
        )
    )
    prompt_bank = PromptBank(
        (
            DomainPrompt(
                "device",
                ("device",),
                (),
                part_profiles=(PartProfile("lamp", ("lamp",), ()),),
            ),
            DomainPrompt(
                "container",
                ("container",),
                (),
                part_profiles=(PartProfile("flatware", ("plate",), ()),),
            ),
        )
    )

    candidates, diagnostics = _apply_asset_domain_routes(
        [root],
        [root],
        {"test::1": route},
        prompt_bank,
        config=AssetRouterConfig(),
        supported_domains={"device", "container"},
    )

    assert candidates[0].semantic_name == "device"
    assert candidates[0].metadata["selected_part_profile"] == "lamp"
    assert diagnostics["corrected_domain_count"] == 0
    assert diagnostics["rows"][0]["specific_root_domain_preserved"] is True


def test_cross_view_exact_label_corrects_a_conflicting_specific_root() -> None:
    mask = np.ones((24, 24), dtype=bool)
    root = MaskCandidate(
        "tool_prop",
        "tool_prop",
        mask,
        0.43,
        "root",
        prompt="pan",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
            "root_model_label": "pan",
            "selected_part_profile": "pan",
            "part_profile_specificity": 1.0,
            "profile_hint_source": "specific_root_label",
        },
    )
    plate_route = AssetRoute(
        accepted=True,
        asset_label="plate",
        asset_domain="container",
        asset_profile="flatware",
        score=0.31,
        margin=0.07,
        alternatives=(
            {
                "asset_label": "plate",
                "asset_domain": "container",
                "asset_profile": "flatware",
                "score": 0.31,
            },
        ),
        candidate_labels=("plate",),
        candidate_domains=("container",),
        reason="accepted_exact_label",
    )
    prompt_bank = PromptBank(
        (
            DomainPrompt(
                "tool_prop",
                ("tool",),
                (),
                part_profiles=(PartProfile("pan", ("pan",), ()),),
            ),
            DomainPrompt(
                "container",
                ("container",),
                (),
                part_profiles=(PartProfile("flatware", ("plate",), ()),),
            ),
        )
    )

    candidates, diagnostics = _apply_asset_domain_routes(
        [root],
        [root],
        {"test::1": plate_route},
        prompt_bank,
        config=AssetRouterConfig(),
        supported_domains={"tool_prop", "container"},
        full_image_route=plate_route,
    )

    routed_root = candidates[0]
    assert routed_root.semantic_name == "container"
    assert routed_root.metadata["selected_part_profile"] == "flatware"
    assert routed_root.metadata["resolved_object_label"] == "plate"
    row = diagnostics["rows"][0]
    assert row["cross_view_exact_domain"] is True
    assert row["specific_root_domain_preserved"] is False
    assert diagnostics["corrected_domain_count"] == 1


def test_cross_view_consensus_cannot_override_domain_missing_from_inventory() -> None:
    mask = np.ones((20, 20), dtype=bool)
    root = MaskCandidate(
        "character",
        "character",
        mask,
        0.9,
        "root",
        prompt="person",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
        },
    )
    route = AssetRoute(
        accepted=True,
        asset_label="shoe",
        asset_domain="daily_object",
        asset_profile="footwear",
        score=0.20,
        margin=0.01,
        alternatives=(
            {
                "asset_label": "shoe",
                "asset_domain": "daily_object",
                "asset_profile": "footwear",
                "score": 0.20,
            },
        ),
        candidate_labels=("shoe",),
        candidate_domains=("daily_object",),
        reason="accepted_cross_view_asset_consensus",
    )
    prompt_bank = PromptBank(
        (
            DomainPrompt("character", ("person",), ()),
            DomainPrompt("daily_object", ("shoe",), ()),
        )
    )

    candidates, diagnostics = _apply_asset_domain_routes(
        [root],
        [root],
        {"test::1": route},
        prompt_bank,
        config=AssetRouterConfig(),
        supported_domains={"daily_object"},
    )

    assert candidates[0].semantic_name == "character"
    assert diagnostics["corrected_domain_count"] == 0
    row = diagnostics["rows"][0]
    assert row["routing_applicable"] is False
    assert row["independently_accepted_exact_route"] is False
    assert (
        row["domain_resolution"]["reason"] == "current_domain_outside_router_inventory"
    )


def test_cross_view_consensus_does_not_lock_same_domain_part_profile() -> None:
    mask = np.ones((20, 20), dtype=bool)
    root = MaskCandidate(
        "tool_prop",
        "tool_prop",
        mask,
        0.9,
        "root",
        prompt="tool",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
        },
    )
    route = AssetRoute(
        accepted=True,
        asset_label="screwdriver",
        asset_domain="tool_prop",
        asset_profile="screwdriver",
        score=0.21,
        margin=0.0,
        alternatives=(
            {
                "asset_label": "screwdriver",
                "asset_domain": "tool_prop",
                "asset_profile": "screwdriver",
                "score": 0.21,
            },
        ),
        candidate_labels=("screwdriver",),
        candidate_domains=("tool_prop",),
        reason="accepted_cross_view_asset_consensus",
    )
    prompt_bank = PromptBank(
        (
            DomainPrompt(
                "tool_prop",
                ("tool",),
                (),
                part_profiles=(
                    PartProfile("screwdriver", ("screwdriver",), ()),
                    PartProfile("rifle", ("rifle",), ()),
                ),
            ),
        )
    )

    candidates, diagnostics = _apply_asset_domain_routes(
        [root],
        [root],
        {"test::1": route},
        prompt_bank,
        config=AssetRouterConfig(),
        supported_domains={"tool_prop"},
    )

    routed_root = candidates[0]
    assert routed_root.semantic_name == "tool_prop"
    assert routed_root.metadata["selected_part_profile"] is None
    assert routed_root.metadata["profile_resolution_status"] is None
    assert routed_root.metadata["asset_router_exact_label_accepted"] is False
    assert routed_root.metadata["resolved_object_label"] == "tool prop"
    assert routed_root.metadata["root_model_label"] == "tool prop"
    assert routed_root.metadata["root_label_specificity"] == 0.0
    row = diagnostics["rows"][0]
    assert row["domain_accepted"] is True
    assert row["selected_profile"] is None
    assert row["independently_accepted_exact_route"] is False


def test_exact_full_and_crop_asset_agreement_can_lock_a_profile() -> None:
    mask = np.ones((20, 20), dtype=bool)
    root = MaskCandidate(
        "device",
        "device",
        mask,
        0.9,
        "root",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
        },
    )
    route = AssetRoute(
        accepted=True,
        asset_label="cellular telephone",
        asset_domain="device",
        asset_profile="phone",
        score=0.31,
        margin=0.08,
        alternatives=(
            {
                "asset_label": "cellular telephone",
                "asset_domain": "device",
                "asset_profile": "phone",
                "score": 0.31,
            },
        ),
        candidate_labels=("cellular telephone",),
        candidate_domains=("device",),
        reason="accepted_exact_label",
    )
    prompt_bank = PromptBank(
        (
            DomainPrompt(
                "device",
                ("device",),
                (),
                part_profiles=(PartProfile("phone", ("phone",), ()),),
            ),
        )
    )

    candidates, diagnostics = _apply_asset_domain_routes(
        [root],
        [root],
        {"test::1": route},
        prompt_bank,
        config=AssetRouterConfig(),
        supported_domains={"device"},
        full_image_route=route,
    )

    routed_root = candidates[0]
    assert routed_root.metadata["selected_part_profile"] == "phone"
    assert routed_root.metadata["profile_hint_source"] == (
        "cross_view_exact_asset_router"
    )
    assert diagnostics["rows"][0]["cross_view_exact_profile"] is True


def test_exact_full_image_profile_locks_its_single_selected_global_root() -> None:
    mask = np.ones((20, 20), dtype=bool)
    root = MaskCandidate(
        "tool_prop",
        "tool_prop",
        mask,
        0.9,
        "root",
        prompt="pan",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
            "root_query_mode": "global_asset_proposal",
            "root_model_label": "pan",
            "selected_part_profile": "pan",
            "part_profile_specificity": 1.0,
            "global_asset_proposal_accepted": True,
            "global_asset_proposal_rank": 1,
        },
    )
    full = AssetRoute(
        accepted=True,
        asset_label="pan (for cooking)",
        asset_domain="tool_prop",
        asset_profile="pan",
        score=0.28,
        margin=0.04,
        alternatives=(),
        candidate_labels=("pan (for cooking)",),
        candidate_domains=("tool_prop",),
        reason="accepted_exact_label",
    )
    local = _ambiguous_route(
        (
            {
                "asset_label": "spoon",
                "asset_domain": "tool_prop",
                "asset_profile": "spoon",
                "score": 0.25,
            },
            {
                "asset_label": "pan (for cooking)",
                "asset_domain": "tool_prop",
                "asset_profile": "pan",
                "score": 0.24,
            },
        )
    )
    prompt_bank = PromptBank(
        (
            DomainPrompt(
                "tool_prop",
                ("tool",),
                (),
                part_profiles=(
                    PartProfile("pan", ("pan",), ()),
                    PartProfile("spoon", ("spoon",), ()),
                ),
            ),
        )
    )

    candidates, diagnostics = _apply_asset_domain_routes(
        [root],
        [root],
        {"test::1": local},
        prompt_bank,
        config=AssetRouterConfig(),
        supported_domains={"tool_prop"},
        full_image_route=full,
    )

    assert candidates[0].metadata["selected_part_profile"] == "pan"
    assert (
        candidates[0].metadata["profile_hint_source"]
        == "accepted_global_asset_profile"
    )
    assert diagnostics["rows"][0]["exact_global_root_profile_lock"] is True


def test_exact_global_profile_lock_moves_profile_and_root_to_same_domain() -> None:
    mask = np.ones((20, 20), dtype=bool)
    root = MaskCandidate(
        "tool_prop",
        "tool_prop",
        mask,
        0.9,
        "root",
        prompt="shoe",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
            "root_query_mode": "global_asset_proposal",
            "root_model_label": "shoe",
            "selected_part_profile": "footwear",
            "part_profile_specificity": 1.0,
            "global_asset_proposal_accepted": True,
            "global_asset_proposal_rank": 1,
        },
    )
    full = AssetRoute(
        accepted=True,
        asset_label="shoe",
        asset_domain="daily_object",
        asset_profile="footwear",
        score=0.29,
        margin=0.04,
        alternatives=(),
        candidate_labels=("shoe",),
        candidate_domains=("daily_object",),
        reason="accepted_exact_label",
    )
    local = _ambiguous_route(
        (
            {
                "asset_label": "knife",
                "asset_domain": "tool_prop",
                "asset_profile": "knife",
                "score": 0.25,
            },
            {
                "asset_label": "screwdriver",
                "asset_domain": "tool_prop",
                "asset_profile": "screwdriver",
                "score": 0.24,
            },
        )
    )
    prompt_bank = PromptBank(
        (
            DomainPrompt("tool_prop", ("tool",), ()),
            DomainPrompt(
                "daily_object",
                ("object",),
                (),
                part_profiles=(PartProfile("footwear", ("shoe",), ()),),
            ),
        )
    )

    candidates, diagnostics = _apply_asset_domain_routes(
        [root],
        [root],
        {"test::1": local},
        prompt_bank,
        config=AssetRouterConfig(),
        supported_domains={"tool_prop", "daily_object"},
        full_image_route=full,
    )

    assert candidates[0].semantic_name == "daily_object"
    assert candidates[0].metadata["selected_part_profile"] == "footwear"
    assert candidates[0].metadata["resolved_object_label"] == "shoe"
    assert diagnostics["rows"][0]["resolved_domain"] == "daily_object"


def test_exact_global_identity_is_not_renamed_by_conflicting_local_crop() -> None:
    root = MaskCandidate(
        "daily_object",
        "daily_object",
        np.ones((20, 20), dtype=bool),
        0.9,
        "root",
        prompt="shoe",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
            "root_query_mode": "global_asset_proposal",
            "root_model_label": "shoe",
            "selected_part_profile": "footwear",
            "part_profile_specificity": 1.0,
            "global_asset_proposal_accepted": True,
            "global_asset_proposal_rank": 1,
        },
    )
    full = AssetRoute(
        accepted=True,
        asset_label="shoe",
        asset_domain="daily_object",
        asset_profile="footwear",
        score=0.30,
        margin=0.05,
        alternatives=(),
        candidate_labels=("shoe",),
        candidate_domains=("daily_object",),
        reason="accepted_exact_label",
    )
    local = AssetRoute(
        accepted=True,
        asset_label="knife",
        asset_domain="tool_prop",
        asset_profile="knife",
        score=0.31,
        margin=0.04,
        alternatives=(),
        candidate_labels=("knife",),
        candidate_domains=("tool_prop",),
        reason="accepted_exact_label",
    )
    prompt_bank = PromptBank(
        (
            DomainPrompt(
                "daily_object",
                ("object",),
                (),
                part_profiles=(PartProfile("footwear", ("shoe",), ()),),
            ),
            DomainPrompt(
                "tool_prop",
                ("tool",),
                (),
                part_profiles=(PartProfile("knife", ("knife",), ()),),
            ),
        )
    )

    candidates, _ = _apply_asset_domain_routes(
        [root],
        [root],
        {"test::1": local},
        prompt_bank,
        config=AssetRouterConfig(),
        supported_domains={"daily_object", "tool_prop"},
        full_image_route=full,
    )

    routed_root = candidates[0]
    assert routed_root.semantic_name == "daily_object"
    assert routed_root.metadata["selected_part_profile"] == "footwear"
    assert routed_root.metadata["resolved_object_label"] == "shoe"


def test_physical_inventory_overrides_conflicting_cross_view_shape_match() -> None:
    mask = np.ones((20, 20), dtype=bool)
    root = MaskCandidate(
        "device",
        "device",
        mask,
        0.9,
        "root",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
        },
    )
    lamp_route = _ambiguous_route(
        (
            {
                "asset_label": "lamp",
                "asset_domain": "device",
                "asset_profile": "lamp",
                "score": 0.24,
            },
            {
                "asset_label": "clock",
                "asset_domain": "device",
                "asset_profile": "clock_watch",
                "score": 0.238,
            },
        )
    )
    globe_inventory = ProfileTextRoute(
        accepted=True,
        profile="globe",
        score=0.12,
        margin=0.048,
        alternatives=(),
        reason="accepted_profile_inventory",
    )
    prompt_bank = PromptBank(
        (
            DomainPrompt(
                "device",
                ("device",),
                (),
                part_profiles=(
                    PartProfile("lamp", ("lamp",), ()),
                    PartProfile("globe", ("globe",), ()),
                ),
            ),
        )
    )

    candidates, diagnostics = _apply_asset_domain_routes(
        [root],
        [root],
        {"test::1": lamp_route},
        prompt_bank,
        config=AssetRouterConfig(),
        supported_domains={"device"},
        full_image_route=lamp_route,
        profile_text_routes_by_root={"test::1": {"device": globe_inventory}},
    )

    routed_root = candidates[0]
    assert routed_root.metadata["selected_part_profile"] == "globe"
    assert routed_root.metadata["profile_hint_source"] == (
        "lightweight_profile_text_router"
    )
    assert diagnostics["rows"][0]["cross_view_exact_profile"] is True


def test_full_image_exact_domain_can_relabel_a_structurally_supported_root() -> None:
    root = _domain_prior_root("character", ("character", "device"))
    local = _ambiguous_route(
        (
            {"asset_label": "hat", "asset_domain": "daily_object", "score": 0.24},
            {"asset_label": "belt", "asset_domain": "daily_object", "score": 0.23},
        )
    )
    full = AssetRoute(
        accepted=True,
        asset_label="calculator",
        asset_domain="device",
        asset_profile="controls",
        score=0.29,
        margin=0.05,
        alternatives=(
            {"asset_label": "calculator", "asset_domain": "device", "score": 0.29},
            {"asset_label": "telephone", "asset_domain": "device", "score": 0.23},
            {"asset_label": "hat", "asset_domain": "daily_object", "score": 0.20},
        ),
        candidate_labels=("calculator",),
        candidate_domains=("device",),
        reason="accepted_exact_label",
    )
    prompt_bank = PromptBank(
        (
            DomainPrompt("character", ("person",), ()),
            DomainPrompt("device", ("device",), ()),
            DomainPrompt("daily_object", ("object",), ()),
        )
    )

    candidates, diagnostics = _apply_asset_domain_routes(
        [root],
        [root],
        {"test::1": local},
        prompt_bank,
        config=AssetRouterConfig(),
        supported_domains={"device", "daily_object"},
        full_image_route=full,
    )

    assert candidates[0].semantic_name == "device"
    row = diagnostics["rows"][0]
    assert row["full_image_domain_prior_applied"] is True
    assert row["full_image_domain_structural_support"] is True
    assert row["domain_resolution"]["reason"] == (
        "accepted_full_image_exact_domain_with_cross_source_support"
    )


def test_full_image_exact_identity_locks_inventory_when_geometry_is_generic() -> None:
    root = _domain_prior_root("terrain", ("daily_object",))
    local = _ambiguous_route(
        (
            {
                "asset_label": "newspaper",
                "asset_domain": "daily_object",
                "asset_profile": "book",
                "score": 0.25,
            },
            {
                "asset_label": "hat",
                "asset_domain": "daily_object",
                "asset_profile": "hat",
                "score": 0.24,
            },
        )
    )
    full = AssetRoute(
        accepted=True,
        asset_label="belt",
        asset_domain="daily_object",
        asset_profile="wallet_belt",
        score=0.30,
        margin=0.05,
        alternatives=(
            {
                "asset_label": "belt",
                "asset_domain": "daily_object",
                "asset_profile": "wallet_belt",
                "score": 0.30,
            },
            {
                "asset_label": "hat",
                "asset_domain": "daily_object",
                "asset_profile": "hat",
                "score": 0.22,
            },
        ),
        candidate_labels=("belt",),
        candidate_domains=("daily_object",),
        reason="accepted_exact_label",
    )
    misleading_inventory = ProfileTextRoute(
        accepted=True,
        profile="book",
        score=0.16,
        margin=0.05,
        alternatives=(),
        reason="accepted_profile_inventory",
    )
    prompt_bank = PromptBank(
        (
            DomainPrompt("terrain", ("terrain",), ()),
            DomainPrompt(
                "daily_object",
                ("object",),
                (),
                part_profiles=(
                    PartProfile("wallet_belt", ("belt",), ()),
                    PartProfile("book", ("book",), ()),
                    PartProfile("hat", ("hat",), ()),
                ),
            ),
        )
    )

    candidates, diagnostics = _apply_asset_domain_routes(
        [root],
        [root],
        {"test::1": local},
        prompt_bank,
        config=AssetRouterConfig(),
        supported_domains={"daily_object"},
        full_image_route=full,
        profile_text_routes_by_root={
            "test::1": {"daily_object": misleading_inventory}
        },
    )

    routed_root = candidates[0]
    assert routed_root.semantic_name == "daily_object"
    assert routed_root.metadata["selected_part_profile"] == "wallet_belt"
    assert routed_root.metadata["resolved_object_label"] == "belt"
    assert routed_root.metadata["profile_hint_source"] == (
        "accepted_full_image_asset_profile"
    )
    row = diagnostics["rows"][0]
    assert row["full_image_exact_profile_lock"] is True
    assert row["profile_text_route"]["profile"] == "book"


def test_local_exact_asset_route_has_priority_over_full_image_prior() -> None:
    root = _domain_prior_root("character", ("character", "container"))
    local = AssetRoute(
        accepted=True,
        asset_label="sweater",
        asset_domain="daily_object",
        asset_profile="garment",
        score=0.31,
        margin=0.06,
        alternatives=(
            {"asset_label": "sweater", "asset_domain": "daily_object", "score": 0.31},
        ),
        candidate_labels=("sweater",),
        candidate_domains=("daily_object",),
        reason="accepted_exact_label",
    )
    full = AssetRoute(
        accepted=True,
        asset_label="cup",
        asset_domain="container",
        asset_profile="drinkware",
        score=0.28,
        margin=0.04,
        alternatives=(
            {"asset_label": "cup", "asset_domain": "container", "score": 0.28},
        ),
        candidate_labels=("cup",),
        candidate_domains=("container",),
        reason="accepted_exact_label",
    )
    prompt_bank = PromptBank(
        (
            DomainPrompt("character", ("person",), ()),
            DomainPrompt("container", ("container",), ()),
            DomainPrompt("daily_object", ("object",), ()),
        )
    )

    candidates, diagnostics = _apply_asset_domain_routes(
        [root],
        [root],
        {"test::1": local},
        prompt_bank,
        config=AssetRouterConfig(),
        supported_domains={"container", "daily_object"},
        full_image_route=full,
    )

    assert candidates[0].semantic_name == "daily_object"
    assert diagnostics["rows"][0]["full_image_domain_prior_applied"] is False


def test_explicit_asset_prompt_cannot_be_relabelled_by_visual_router() -> None:
    root = MaskCandidate(
        "daily_object",
        "daily_object",
        np.ones((24, 24), dtype=bool),
        0.61,
        "root",
        prompt="scarf",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
            "root_query_mode": "user_asset_prompt",
            "root_model_label": "scarf",
            "selected_part_profile": "scarf",
            "part_profile_specificity": 1.0,
            "profile_hint_source": "user_asset_prompt",
            "profile_resolution_status": "accepted",
        },
    )
    visual_route = AssetRoute(
        accepted=True,
        asset_label="bicycle",
        asset_domain="vehicle",
        asset_profile="bicycle",
        score=0.31,
        margin=0.06,
        alternatives=(
            {"asset_label": "bicycle", "asset_domain": "vehicle", "score": 0.31},
        ),
        candidate_labels=("bicycle",),
        candidate_domains=("vehicle",),
        reason="accepted_exact_label",
    )
    prompt_bank = PromptBank(
        (
            DomainPrompt(
                "daily_object",
                ("object",),
                (),
                part_profiles=(PartProfile("scarf", ("scarf",), ()),),
            ),
            DomainPrompt(
                "vehicle",
                ("vehicle",),
                (),
                part_profiles=(PartProfile("bicycle", ("bicycle",), ()),),
            ),
        )
    )

    candidates, diagnostics = _apply_asset_domain_routes(
        [root],
        [root],
        {"test::1": visual_route},
        prompt_bank,
        config=AssetRouterConfig(),
        supported_domains={"daily_object", "vehicle"},
        full_image_route=visual_route,
    )

    routed_root = candidates[0]
    assert routed_root.semantic_name == "daily_object"
    assert routed_root.metadata["selected_part_profile"] == "scarf"
    assert routed_root.metadata["profile_hint_source"] == "user_asset_prompt"
    row = diagnostics["rows"][0]
    assert row["explicit_user_prompt_lock"] is True
    assert row["domain_resolution"]["reason"] == (
        "explicit_user_prompt_domain_profile_lock"
    )


def test_near_threshold_full_image_domain_needs_structural_consensus() -> None:
    root = _domain_prior_root("character", ("character", "device"))
    local = _ambiguous_route(
        (
            {"asset_label": "hat", "asset_domain": "daily_object", "score": 0.24},
            {"asset_label": "belt", "asset_domain": "daily_object", "score": 0.23},
        )
    )
    alternatives = (
        {"asset_label": "calculator", "asset_domain": "device", "score": 0.242},
        {"asset_label": "mouse", "asset_domain": "device", "score": 0.231},
        {"asset_label": "telephone", "asset_domain": "device", "score": 0.228},
        {"asset_label": "laptop", "asset_domain": "device", "score": 0.224},
        {"asset_label": "pen", "asset_domain": "daily_object", "score": 0.220},
    )
    full = AssetRoute(
        accepted=False,
        asset_label=None,
        asset_domain=None,
        asset_profile=None,
        score=0.242,
        margin=0.016,
        alternatives=alternatives,
        candidate_labels=tuple(str(row["asset_label"]) for row in alternatives),
        candidate_domains=("device", "daily_object"),
        reason="ambiguous_candidate_set",
    )

    prior = resolve_full_image_domain_prior(full)
    assert prior.accepted is True
    assert prior.domain == "device"
    assert prior.exact_label_accepted is False
    assert prior.support_ratio == 0.8

    prompt_bank = PromptBank(
        (
            DomainPrompt("character", ("person",), ()),
            DomainPrompt("device", ("device",), ()),
            DomainPrompt("daily_object", ("object",), ()),
        )
    )
    candidates, diagnostics = _apply_asset_domain_routes(
        [root],
        [root],
        {"test::1": local},
        prompt_bank,
        config=AssetRouterConfig(),
        supported_domains={"device", "daily_object"},
        full_image_route=full,
    )

    assert candidates[0].semantic_name == "device"
    assert diagnostics["rows"][0]["full_image_domain_prior_applied"] is True


def test_full_image_exact_label_cannot_relabel_without_independent_support() -> None:
    root = _domain_prior_root("character", ("character",))
    local = _ambiguous_route(
        (
            {"asset_label": "hat", "asset_domain": "daily_object", "score": 0.24},
            {"asset_label": "belt", "asset_domain": "daily_object", "score": 0.23},
        )
    )
    full = AssetRoute(
        accepted=True,
        asset_label="hammer",
        asset_domain="tool_prop",
        asset_profile="hammer",
        score=0.25,
        margin=0.03,
        alternatives=(
            {"asset_label": "hammer", "asset_domain": "tool_prop", "score": 0.25},
            {"asset_label": "box", "asset_domain": "container", "score": 0.22},
            {"asset_label": "can", "asset_domain": "container", "score": 0.21},
            {"asset_label": "plate", "asset_domain": "container", "score": 0.20},
            {"asset_label": "book", "asset_domain": "daily_object", "score": 0.19},
        ),
        candidate_labels=("hammer",),
        candidate_domains=("tool_prop",),
        reason="accepted_exact_label",
    )
    prompt_bank = PromptBank(
        (
            DomainPrompt("character", ("person",), ()),
            DomainPrompt("daily_object", ("object",), ()),
            DomainPrompt("tool_prop", ("tool",), ()),
            DomainPrompt("container", ("container",), ()),
        )
    )

    candidates, diagnostics = _apply_asset_domain_routes(
        [root],
        [root],
        {"test::1": local},
        prompt_bank,
        config=AssetRouterConfig(),
        supported_domains={"daily_object", "tool_prop", "container"},
        full_image_route=full,
    )

    assert candidates[0].semantic_name == "character"
    assert diagnostics["rows"][0]["full_image_domain_prior_applied"] is False


def _index() -> SimpleNamespace:
    return SimpleNamespace(
        manifest={"encoder_model_name": "fake-router"},
        labels=("lamp", "watch"),
        text_embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        label_prototypes=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        image_embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        assets=({"asset_id": "lamp-1"}, {"asset_id": "watch-1"}),
        label_metadata={
            "lamp": {
                "asset_domain": "device",
                "asset_profile": "lamp",
                "asset_indices": [0],
            },
            "watch": {
                "asset_domain": "device",
                "asset_profile": "watch",
                "asset_indices": [1],
            },
        },
    )


def test_masked_asset_view_removes_unrelated_background() -> None:
    rgb = np.full((20, 30, 3), 240, dtype=np.uint8)
    rgb[5:15, 10:20] = 30
    root = np.zeros((20, 30), dtype=bool)
    root[5:15, 10:20] = True

    view = np.asarray(masked_asset_view(Image.fromarray(rgb), root))

    assert (view == 30).any()
    assert (view == 127).any()
    assert not (view == 240).any()


def test_profile_text_router_locks_only_a_clear_physical_inventory() -> None:
    domain = DomainPrompt(
        name="tool_prop",
        root_prompts=("tool",),
        parts=(),
        part_profiles=(
            PartProfile(
                name="firearm",
                root_hints=("rifle",),
                part_semantics=("receiver", "magazine"),
                classifier_prompt="a complete firearm rifle",
            ),
            PartProfile(
                name="hammer",
                root_hints=("hammer",),
                part_semantics=("head", "handle"),
                classifier_prompt="a hand hammer",
            ),
        ),
    )

    accepted = route_profile_text_inventory(
        np.asarray([1.0, 0.0], dtype=np.float32),
        domain,
        _ProfileEncoder(),
    )
    rejected = route_profile_text_inventory(
        np.asarray([1.0, 0.0], dtype=np.float32),
        domain,
        _ProfileEncoder(),
        config=ProfileTextRouterConfig(minimum_score=0.10, minimum_margin=0.10),
    )

    assert accepted.accepted is True
    assert accepted.profile == "firearm"
    assert accepted.reason == "accepted_profile_inventory"
    assert rejected.accepted is False
    assert rejected.profile is None
    assert rejected.reason == "ambiguous_profile_inventory"


def test_profile_text_router_scores_all_domain_inventories_in_one_batch() -> None:
    class CountingEncoder(_ProfileEncoder):
        def __init__(self) -> None:
            self.text_batch_count = 0

        def encode_texts(self, texts: list[str]) -> np.ndarray:
            self.text_batch_count += 1
            return super().encode_texts(texts)

    encoder = CountingEncoder()
    domains = (
        DomainPrompt(
            "tool_prop",
            ("tool",),
            (),
            part_profiles=(
                PartProfile(
                    "firearm",
                    ("rifle",),
                    (),
                    classifier_prompt="a firearm",
                ),
            ),
        ),
        DomainPrompt(
            "device",
            ("device",),
            (),
            part_profiles=(
                PartProfile(
                    "globe",
                    ("world globe",),
                    (),
                    classifier_prompt="a world globe",
                ),
            ),
        ),
    )

    routes = route_profile_text_inventories(
        np.asarray([1.0, 0.0], dtype=np.float32),
        domains,
        encoder,
    )

    assert encoder.text_batch_count == 1
    assert set(routes) == {"tool_prop", "device"}


def test_asset_router_combines_text_and_train_prototypes() -> None:
    router = AssetRouter(
        _index(),
        _FakeEncoder(),
        config=AssetRouterConfig(minimum_score=0.0, minimum_margin=0.0),
    )
    image = Image.fromarray(np.full((16, 16, 3), 220, dtype=np.uint8))
    root = np.ones((16, 16), dtype=bool)

    route = router.route(image, root)

    assert route.accepted
    assert route.asset_label == "lamp"
    assert route.asset_profile == "lamp"
    assert route.candidate_labels == ("lamp",)
    assert route.reason == "accepted_exact_label"
    assert route.alternatives[0]["nearest_asset_ids"] == ["lamp-1"]


def test_asset_router_returns_bounded_cross_domain_candidates_when_ambiguous() -> None:
    router = AssetRouter(
        _index(),
        _FakeEncoder(),
        config=AssetRouterConfig(
            prototype_weight=0.5,
            text_weight=0.5,
            nearest_asset_weight=0.0,
            minimum_score=0.0,
            minimum_margin=1.10,
            maximum_candidate_labels=2,
            maximum_candidate_score_drop=1.0,
        ),
    )
    image = Image.fromarray(np.full((16, 16, 3), 220, dtype=np.uint8))
    root = np.ones((16, 16), dtype=bool)

    route = router.route(image, root)

    assert not route.accepted
    assert route.asset_label is None
    assert route.reason == "ambiguous_candidate_set"
    assert route.candidate_labels == ("lamp", "watch")
    assert route.candidate_domains == ("device",)


def test_ambiguous_labels_can_still_resolve_a_majority_domain() -> None:
    route = AssetRoute(
        accepted=False,
        asset_label=None,
        asset_domain=None,
        asset_profile=None,
        score=0.24,
        margin=0.004,
        alternatives=tuple(
            {
                "asset_label": label,
                "asset_domain": domain,
                "score": score,
            }
            for label, domain, score in (
                ("drill", "tool_prop", 0.240),
                ("pen", "daily_object", 0.236),
                ("screwdriver", "tool_prop", 0.234),
                ("sponge", "daily_object", 0.232),
                ("knife", "tool_prop", 0.230),
            )
        ),
        candidate_labels=("drill", "pen", "screwdriver", "sponge", "knife"),
        candidate_domains=("tool_prop", "daily_object"),
        reason="ambiguous_candidate_set",
    )

    resolution = resolve_asset_domain(route, "character")

    assert resolution.accepted is True
    assert resolution.resolved_domain == "tool_prop"
    assert resolution.resolved_profile is None
    assert resolution.support_count == 3
    assert resolution.support_ratio == 0.6


def test_tied_cross_domain_candidates_do_not_force_a_domain() -> None:
    route = AssetRoute(
        accepted=False,
        asset_label=None,
        asset_domain=None,
        asset_profile=None,
        score=0.24,
        margin=0.002,
        alternatives=(
            {"asset_label": "drill", "asset_domain": "tool_prop", "score": 0.24},
            {"asset_label": "pen", "asset_domain": "daily_object", "score": 0.238},
        ),
        candidate_labels=("drill", "pen"),
        candidate_domains=("tool_prop", "daily_object"),
        reason="ambiguous_candidate_set",
    )

    resolution = resolve_asset_domain(route, "character")

    assert resolution.accepted is False
    assert resolution.resolved_domain is None
    assert resolution.reason == "ambiguous_cross_domain_candidates"


def test_ambiguous_router_retains_supported_current_domain_over_label_count() -> None:
    route = AssetRoute(
        accepted=False,
        asset_label=None,
        asset_domain=None,
        asset_profile=None,
        score=0.287,
        margin=0.002,
        alternatives=(
            {
                "asset_label": "stool",
                "asset_domain": "furniture",
                "asset_profile": "chair",
                "score": 0.287,
            },
            {"asset_label": "ladder", "asset_domain": "tool_prop", "score": 0.285},
            {"asset_label": "basket", "asset_domain": "container", "score": 0.272},
            {"asset_label": "crate", "asset_domain": "container", "score": 0.269},
            {"asset_label": "box", "asset_domain": "container", "score": 0.268},
        ),
        candidate_labels=("stool", "ladder", "basket", "crate", "box"),
        candidate_domains=("furniture", "tool_prop", "container"),
        reason="ambiguous_candidate_set",
    )

    resolution = resolve_asset_domain(route, "furniture")

    assert resolution.accepted is True
    assert resolution.resolved_domain == "furniture"
    assert resolution.resolved_profile is None
    assert resolution.resolved_asset_label is None
    assert (
        resolution.asset_label_reason == "ambiguous_route_not_promoted_to_exact_label"
    )
    assert resolution.reason == "retained_current_domain_under_ambiguity"


def test_retained_domain_does_not_reuse_cross_domain_top_label() -> None:
    route = AssetRoute(
        accepted=False,
        asset_label=None,
        asset_domain=None,
        asset_profile=None,
        score=0.289,
        margin=0.004,
        alternatives=(
            {
                "asset_label": "ladder",
                "asset_domain": "tool_prop",
                "asset_profile": "ladder",
                "score": 0.289,
            },
            {
                "asset_label": "stool",
                "asset_domain": "furniture",
                "asset_profile": "chair",
                "score": 0.285,
            },
        ),
        candidate_labels=("ladder", "stool"),
        candidate_domains=("tool_prop", "furniture"),
        reason="ambiguous_candidate_set",
    )

    resolution = resolve_asset_domain(route, "furniture")

    assert resolution.accepted is True
    assert resolution.resolved_domain == "furniture"
    assert resolution.resolved_profile is None
    assert resolution.resolved_asset_label is None
    assert (
        resolution.asset_label_reason == "ambiguous_route_not_promoted_to_exact_label"
    )
