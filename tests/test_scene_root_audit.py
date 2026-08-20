from __future__ import annotations

import numpy as np
from PIL import Image

from hpid_split.fusion import MaskCandidate
from hpid_split.prompt_bank import DomainPrompt, PartProfile, PartPrompt
from hpid_split.scene_root_audit import (
    SceneRootAuditConfig,
    SceneRootAuditor,
    apply_scene_root_audit,
    parse_scene_root_audit_response,
    parse_scene_root_batch_audit_response,
    select_scene_root_audit_candidates,
)


def _mask(x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((40, 100), dtype=bool)
    mask[5:35, x0:x1] = True
    return mask


def _root(
    domain: str,
    index: int,
    x0: int,
    *,
    profile: str | None = None,
    global_agreement: bool = True,
    global_domain: str | None = None,
) -> MaskCandidate:
    metadata: dict[str, object] = {
        "root_origin": "scene",
        "root_index": index,
        "candidate_key": f"root:{index}",
        "parent_candidate_key": None,
        "ontology_routing_algorithm": "test-ontology",
        "ontology_global_visual_agreement": global_agreement,
        "ontology_global_profile_domain": global_domain or domain,
        "ontology_visual_domain_winner": global_domain or domain,
        "ontology_profile_decision": "visual_profile_confirmed",
        "ontology_consensus_label": None,
    }
    if profile is not None:
        metadata["selected_part_profile"] = profile
    return MaskCandidate(
        domain,
        domain,
        _mask(x0, x0 + 10),
        0.8,
        "test/root",
        prompt=profile or domain,
        metadata=metadata,
    )


def _domains() -> dict[str, DomainPrompt]:
    rock_face = PartPrompt(
        "terrain_rock",
        ("rock face",),
        semantic_parent="natural_object",
    )
    seat = PartPrompt(
        "furniture_seat",
        ("seat",),
        semantic_parent="furniture",
    )
    natural = DomainPrompt(
        "natural_object",
        ("natural object",),
        (rock_face,),
        part_profiles=(
            PartProfile("rock", ("rock",), ("terrain_rock",)),
            PartProfile("tree", ("tree",), ("terrain_rock",)),
        ),
    )
    furniture = DomainPrompt(
        "furniture",
        ("furniture",),
        (seat,),
        part_profiles=(PartProfile("chair", ("chair",), ("furniture_seat",)),),
    )
    terrain = DomainPrompt("terrain", ("terrain",), ())
    return {domain.name: domain for domain in (natural, furniture, terrain)}


def test_root_audit_parser_is_json_and_inventory_bounded() -> None:
    accepted = parse_scene_root_audit_response(
        'prefix {"label":"natural_rock","certainty":"high"}'
    )
    rejected = parse_scene_root_audit_response(
        '{"label":"imagined_category","confidence":0.99}'
    )

    assert accepted.label == "natural_rock"
    assert accepted.confidence == 0.95
    assert accepted.diagnostics["ground_truth_used"] is False
    assert rejected.label is None
    assert rejected.diagnostics["status"] == "unknown_label"


def test_root_audit_parser_keeps_numeric_confidence_compatibility() -> None:
    parsed = parse_scene_root_audit_response(
        '{"label":"natural_tree","confidence":0.91}'
    )

    assert parsed.label == "natural_tree"
    assert parsed.confidence == 0.91


def test_batch_root_audit_parser_is_id_and_inventory_bounded() -> None:
    parsed, diagnostics = parse_scene_root_batch_audit_response(
        '{"objects":['
        '{"id":1,"label":"natural_rock","certainty":"high"},'
        '{"id":2,"label":"invented","certainty":"high"},'
        '{"id":7,"label":"natural_tree","certainty":"high"}'
        "]}",
        expected_count=2,
    )

    assert parsed[1].label == "natural_rock"
    assert parsed[2].label is None
    assert diagnostics["invalid_id_count"] == 1


def test_natural_scene_prioritizes_an_uncertain_manmade_outlier() -> None:
    roots = [
        _root("natural_object", index, index * 11, profile="rock")
        for index in range(1, 6)
    ]
    roots.append(
        _root(
            "furniture",
            6,
            70,
            profile="chair",
            global_agreement=False,
            global_domain="natural_object",
        )
    )

    selected = select_scene_root_audit_candidates(roots)

    assert len(selected) == 1
    assert selected[0][0].semantic_name == "furniture"
    assert "manmade_label_in_natural_scene" in selected[0][2]


def test_natural_scene_audits_manmade_outliers_without_optional_ontology() -> None:
    roots = [
        _root("natural_object", index, index * 11, profile="rock")
        for index in range(1, 4)
    ]
    roots.append(_root("terrain", 4, 44, profile="outdoor_level"))
    furniture = _root("furniture", 5, 66, profile="chair")
    plain_metadata = {
        key: value
        for key, value in furniture.metadata.items()
        if not key.startswith("ontology_")
    }
    roots.append(
        MaskCandidate(
            furniture.semantic_name,
            furniture.semantic_parent,
            furniture.mask,
            furniture.score,
            furniture.source,
            prompt=furniture.prompt,
            metadata=plain_metadata,
        )
    )

    selected = select_scene_root_audit_candidates(roots)

    assert len(selected) == 1
    assert selected[0][0].semantic_name == "furniture"
    assert "no_ontology_router_evidence" in selected[0][2]
    assert "manmade_label_in_natural_scene" in selected[0][2]


def test_small_character_is_not_audited_on_weak_cross_domain_noise() -> None:
    roots = [
        _root("natural_object", index, index * 11, profile="rock")
        for index in range(1, 6)
    ]
    roots.append(
        _root(
            "character",
            6,
            70,
            global_agreement=False,
            global_domain="natural_object",
        )
    )

    selected = select_scene_root_audit_candidates(roots)

    assert selected == ()


def test_primary_root_with_rejected_exact_asset_route_is_audited() -> None:
    root = _root("character", 1, 20)
    metadata = {
        key: value
        for key, value in root.metadata.items()
        if not key.startswith("ontology_")
    }
    metadata.update(
        {
            "root_origin": "primary",
            "asset_domain_audit_required": True,
            "asset_router_exact_label_accepted": False,
            "asset_router_candidate_domains": ["daily_object", "container"],
        }
    )
    uncertain = MaskCandidate(
        root.semantic_name,
        root.semantic_parent,
        root.mask,
        root.score,
        root.source,
        prompt=root.prompt,
        metadata=metadata,
    )

    selected = select_scene_root_audit_candidates(
        [uncertain],
        config=SceneRootAuditConfig(maximum_candidates=1),
    )

    assert len(selected) == 1
    assert "asset_router_exact_label_uncertain" in selected[0][2]


def test_primary_root_with_accepted_exact_asset_route_is_not_audited() -> None:
    root = _root("daily_object", 1, 20)
    metadata = {
        key: value
        for key, value in root.metadata.items()
        if not key.startswith("ontology_")
    }
    metadata.update(
        {
            "root_origin": "primary",
            "asset_domain_audit_required": False,
            "asset_router_exact_label_accepted": True,
        }
    )
    certain = MaskCandidate(
        root.semantic_name,
        root.semantic_parent,
        root.mask,
        root.score,
        root.source,
        prompt=root.prompt,
        metadata=metadata,
    )

    selected = select_scene_root_audit_candidates([certain])

    assert selected == ()


def test_scene_root_auditor_corrects_only_high_confidence_labels() -> None:
    class Planner:
        backend_id = "test-planner"

        def generate_response(self, image: Image.Image, prompt: str) -> str:
            assert image.width > image.height
            assert "natural_rock" in prompt
            return '{"label":"natural_rock","confidence":0.98}'

    root = _root(
        "furniture",
        1,
        20,
        profile="chair",
        global_agreement=False,
        global_domain="natural_object",
    )
    auditor = SceneRootAuditor(
        Planner(),
        config=SceneRootAuditConfig(natural_scene_minimum_ratio=0.0),
    )

    result = auditor.audit(Image.new("RGB", (100, 40)), [root], _domains())

    assert result.roots[0].semantic_name == "natural_object"
    assert result.roots[0].metadata["selected_part_profile"] == "rock"
    assert result.roots[0].metadata["vlm_root_audit_applied"] is True
    assert result.diagnostics["correction_count"] == 1
    assert result.diagnostics["ground_truth_used"] is False


def test_scene_root_auditor_batches_multiple_targets_into_one_query() -> None:
    class Planner:
        backend_id = "test-batch-planner"

        def __init__(self) -> None:
            self.query_count = 0

        def generate_response(self, image: Image.Image, prompt: str) -> str:
            self.query_count += 1
            assert "contact sheet" in prompt
            assert image.width > image.height
            return (
                '{"objects":['
                '{"id":1,"label":"natural_rock","certainty":"high"},'
                '{"id":2,"label":"natural_rock","certainty":"high"}'
                "]}"
            )

    roots = [
        _root("furniture", 1, 20, profile="chair"),
        _root("furniture", 2, 50, profile="chair"),
    ]
    for root in roots:
        for key in tuple(root.metadata):
            if key.startswith("ontology_"):
                root.metadata.pop(key)
    planner = Planner()
    auditor = SceneRootAuditor(
        planner,
        config=SceneRootAuditConfig(
            maximum_queries=1,
            batch_size=6,
            natural_scene_minimum_ratio=0.0,
        ),
    )

    result = auditor.audit(Image.new("RGB", (100, 40)), roots, _domains())

    assert planner.query_count == 1
    assert result.diagnostics["query_count"] == 1
    assert result.diagnostics["audited_root_count"] == 2
    assert result.diagnostics["correction_count"] == 2
    assert all(root.semantic_name == "natural_object" for root in result.roots)


def test_applying_root_audit_drops_old_parts_and_resets_visual_regions() -> None:
    original = _root(
        "furniture",
        1,
        20,
        profile="chair",
        global_agreement=False,
        global_domain="natural_object",
    )
    corrected = MaskCandidate(
        "natural_object",
        "natural_object",
        original.mask,
        original.score,
        original.source,
        prompt="rock",
        metadata={
            **original.metadata,
            "selected_part_profile": "rock",
            "vlm_root_audit_applied": True,
        },
    )
    seat = MaskCandidate(
        "furniture_seat",
        "furniture",
        _mask(22, 28),
        0.7,
        "test/part",
        metadata={
            "root_origin": "scene",
            "root_index": 1,
            "candidate_key": "root:1/seat",
            "parent_candidate_key": "root:1",
        },
    )
    visual = MaskCandidate(
        "furniture_visual_panel_01",
        "furniture",
        _mask(28, 34),
        0.6,
        "test/visual",
        metadata={
            "root_origin": "scene",
            "root_index": 1,
            "candidate_key": "root:1/visual:1",
            "parent_candidate_key": "root:1",
            "visual_region": True,
            "generic_visual_region": False,
            "visual_region_kind": "panel",
            "semantic_reranked": True,
        },
    )

    applied = apply_scene_root_audit(
        [original, seat, visual], [corrected], _domains()
    )

    assert [candidate.semantic_name for candidate in applied.candidates[:1]] == [
        "natural_object"
    ]
    assert len(applied.candidates) == 2
    reset = applied.candidates[1]
    assert reset.semantic_parent == "natural_object"
    assert reset.metadata["generic_visual_region"] is True
    assert "semantic_reranked" not in reset.metadata
    assert applied.diagnostics["dropped_incompatible_candidate_count"] == 1
    assert applied.diagnostics["reset_visual_candidate_count"] == 1
