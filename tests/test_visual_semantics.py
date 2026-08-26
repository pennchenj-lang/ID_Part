from __future__ import annotations

from dataclasses import replace

import numpy as np
from PIL import Image

from hpid_split.fusion import MaskCandidate
from hpid_split.prompt_bank import DomainPrompt, PartProfile, PartPrompt
from hpid_split.visual_semantics import (
    PhysicalRegionGateConfig,
    VisualSemanticConfig,
    enforce_axis_consistency,
    filter_unresolved_visual_regions,
    rerank_visual_candidates,
)


def _mask(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((100, 100), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _root(profile: str | None = "phone") -> MaskCandidate:
    metadata: dict[str, object] = {
        "root_origin": "test",
        "root_index": 1,
        "candidate_key": "root:1",
        "parent_candidate_key": None,
    }
    if profile is not None:
        metadata["selected_part_profile"] = profile
    return MaskCandidate(
        semantic_name="device",
        semantic_parent="device",
        mask=_mask(5, 95, 5, 95),
        score=0.95,
        source="test/root",
        prompt="phone",
        metadata=metadata,
    )


def _visual(
    key: str,
    mask: np.ndarray,
    *,
    kind: str,
    fraction: float,
) -> MaskCandidate:
    return MaskCandidate(
        semantic_name=f"device_visual_{kind}_01",
        semantic_parent="device",
        mask=mask,
        score=0.90,
        source="sam2-amg/test",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": key,
            "parent_candidate_key": "root:1",
            "assembly_parent_semantic": "device",
            "assembly_parent_candidate_key": "root:1",
            "generic_visual_region": True,
            "visual_region": True,
            "visual_region_kind": kind,
            "root_area_fraction": fraction,
            "sam_quality": 0.9,
        },
    )


def _domain() -> DomainPrompt:
    return DomainPrompt(
        name="device",
        root_prompts=("phone", "device"),
        generic_root_prompts=("device",),
        default_part_semantics=("device_body", "device_button"),
        parts=(
            PartPrompt("device_body", ("device body",), maximum_instances=1),
            PartPrompt(
                "device_screen",
                ("phone screen",),
                semantic_parent="device_body",
                maximum_instances=1,
            ),
            PartPrompt(
                "device_button",
                ("phone button",),
                semantic_parent="device_body",
                maximum_parent_fraction=0.08,
                maximum_instances=2,
                detail=True,
            ),
        ),
        part_profiles=(
            PartProfile(
                "phone",
                ("phone",),
                ("device_screen", "device_button"),
            ),
        ),
    )


class _Ranker:
    def __init__(self, rows: dict[str, dict[str, tuple[float, float]]]) -> None:
        self.rows = rows

    def rank_regions_labels(
        self,
        image: Image.Image,
        regions: list[tuple[str, np.ndarray]],
        labels: list[tuple[str, str]],
        **_: object,
    ) -> dict[str, dict[str, dict[str, float | str | int]]]:
        output: dict[str, dict[str, dict[str, float | str | int]]] = {}
        for key, _ in regions:
            ordered = sorted(
                self.rows[key].items(), key=lambda item: item[1][0], reverse=True
            )
            ranks = {name: rank for rank, (name, _) in enumerate(ordered, start=1)}
            output[key] = {
                name: {
                    "prompt": prompt,
                    "combined_similarity": self.rows[key][name][0],
                    "probability": self.rows[key][name][1],
                    "rank": ranks[name],
                    "full_similarity": self.rows[key][name][0],
                    "masked_similarity": self.rows[key][name][0],
                }
                for name, prompt in labels
            }
        return output


def test_visual_semantic_reranker_accepts_profile_consistent_region() -> None:
    root = _root()
    visual = _visual(
        "root:1/visual-region:01",
        _mask(20, 65, 20, 80),
        kind="panel",
        fraction=0.33,
    )
    result = rerank_visual_candidates(
        Image.new("RGB", (100, 100), "white"),
        [visual],
        [root],
        [root],
        {"device": _domain()},
        _Ranker(
            {
                "root:1/visual-region:01": {
                    "device_screen": (0.42, 0.78),
                    "device_button": (0.31, 0.22),
                }
            }
        ),
    )

    candidate = result.candidates[0]
    assert candidate.semantic_name == "device_screen"
    assert candidate.semantic_parent == "device_body"
    assert candidate.metadata["semantic_reranked"] is True
    assert candidate.metadata["generic_visual_region"] is False
    assert result.diagnostics["accepted_semantic_count"] == 1


def test_single_router_candidate_constrains_inventory_without_exact_acceptance() -> None:
    root = _root(profile=None)
    root = replace(
        root,
        metadata={
            **root.metadata,
            "asset_router_candidate_labels": ["phone"],
            "asset_router_candidate_domains": ["device"],
        },
    )
    visual = _visual(
        "root:1/visual-region:01",
        _mask(20, 65, 20, 80),
        kind="panel",
        fraction=0.33,
    )

    result = rerank_visual_candidates(
        Image.new("RGB", (100, 100), "white"),
        [visual],
        [root],
        [root],
        {"device": _domain()},
        _Ranker(
            {
                "root:1/visual-region:01": {
                    "device_screen": (0.42, 0.78),
                    "device_button": (0.31, 0.22),
                }
            }
        ),
    )

    assert result.candidates[0].semantic_name == "device_screen"
    root_row = result.diagnostics["roots"][0]
    assert root_row["inventory_reason"] == "router_candidate_inventory_review"


def test_capacity_aware_assignment_uses_supported_second_choice() -> None:
    root = replace(
        _root(),
        semantic_name="tool_prop",
        semantic_parent="tool_prop",
        prompt="knife",
        metadata={**_root().metadata, "selected_part_profile": "knife"},
    )
    blade = replace(
        _visual(
            "root:1/visual-region:blade",
            _mask(20, 70, 20, 80),
            kind="strip",
            fraction=0.38,
        ),
        semantic_name="tool_prop_visual_strip_01",
        semantic_parent="tool_prop",
    )
    handle = replace(
        _visual(
            "root:1/visual-region:handle",
            _mask(72, 86, 35, 60),
            kind="strip",
            fraction=0.05,
        ),
        semantic_name="tool_prop_visual_strip_02",
        semantic_parent="tool_prop",
    )
    domain = DomainPrompt(
        name="tool_prop",
        root_prompts=("knife",),
        parts=(
            PartPrompt(
                "tool_prop_blade",
                ("knife blade",),
                semantic_parent="tool_prop",
                maximum_instances=1,
                axis_position=-0.55,
            ),
            PartPrompt(
                "tool_prop_handle",
                ("knife handle",),
                semantic_parent="tool_prop",
                maximum_parent_fraction=0.58,
                maximum_instances=1,
                axis_position=0.85,
            ),
        ),
        part_profiles=(
            PartProfile(
                "knife",
                ("knife",),
                ("tool_prop_blade", "tool_prop_handle"),
            ),
        ),
    )
    ranker = _Ranker(
        {
            "root:1/visual-region:blade": {
                "tool_prop_blade": (0.52, 0.86),
                "tool_prop_handle": (0.31, 0.14),
            },
            "root:1/visual-region:handle": {
                "tool_prop_blade": (0.51, 0.70),
                "tool_prop_handle": (0.47, 0.42),
            },
        }
    )

    result = rerank_visual_candidates(
        Image.new("RGB", (100, 100), "white"),
        [blade, handle],
        [root],
        [root],
        {"tool_prop": domain},
        ranker,
    )

    by_key = {
        candidate.metadata["candidate_key"]: candidate
        for candidate in result.candidates
    }
    assert (
        by_key["root:1/visual-region:blade"].semantic_name
        == "tool_prop_blade"
    )
    assigned = by_key["root:1/visual-region:handle"]
    assert assigned.semantic_name == "tool_prop_handle"
    assert assigned.metadata["semantic_rerank_route"] == "capacity_fallback_base"
    assert result.diagnostics["capacity_fallback_accepted_count"] == 1


def test_intrinsic_axis_prior_disambiguates_ordered_tool_parts_when_mirrored() -> None:
    root = MaskCandidate(
        semantic_name="tool_prop",
        semantic_parent="tool_prop",
        mask=_mask(25, 75, 5, 95),
        score=0.95,
        source="test/root",
        prompt="rifle",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
            "selected_part_profile": "firearm",
        },
    )
    stock = MaskCandidate(
        semantic_name="tool_prop_stock",
        semantic_parent="tool_prop",
        mask=_mask(30, 70, 75, 92),
        score=0.90,
        source="test/detector",
        prompt="stock",
        source_reliability=0.9,
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1/stock",
            "parent_candidate_key": "root:1",
            "root_area_fraction": 0.15,
        },
    )
    magazine = replace(
        _visual(
            "root:1/visual-region:magazine",
            _mask(45, 70, 30, 48),
            kind="panel",
            fraction=0.10,
        ),
        semantic_parent="tool_prop",
    )
    grip = replace(
        _visual(
            "root:1/visual-region:grip",
            _mask(45, 72, 60, 70),
            kind="panel",
            fraction=0.06,
        ),
        semantic_parent="tool_prop",
    )
    trigger = replace(
        _visual(
            "root:1/visual-region:trigger",
            _mask(40, 44, 55, 59),
            kind="detail",
            fraction=0.004,
        ),
        semantic_parent="tool_prop",
    )
    domain = DomainPrompt(
        name="tool_prop",
        root_prompts=("tool",),
        parts=(
            PartPrompt(
                "tool_prop_stock",
                ("stock",),
                maximum_instances=1,
                axis_position=0.85,
                axis_tolerance=0.38,
            ),
            PartPrompt(
                "tool_prop_magazine",
                ("magazine",),
                maximum_instances=1,
                axis_position=-0.15,
                axis_tolerance=0.32,
            ),
            PartPrompt(
                "tool_prop_grip",
                ("grip",),
                maximum_instances=1,
                axis_position=0.28,
                axis_tolerance=0.32,
            ),
            PartPrompt(
                "tool_prop_trigger",
                ("trigger",),
                maximum_parent_fraction=0.1,
                maximum_instances=1,
                detail=True,
                axis_position=0.12,
                axis_tolerance=0.32,
            ),
        ),
        part_profiles=(
            PartProfile(
                "firearm",
                ("rifle",),
                (
                    "tool_prop_stock",
                    "tool_prop_magazine",
                    "tool_prop_grip",
                    "tool_prop_trigger",
                ),
            ),
        ),
    )
    ranker = _Ranker(
        {
            "root:1/visual-region:magazine": {
                "tool_prop_stock": (0.32, 0.20),
                "tool_prop_magazine": (0.405, 0.30),
                "tool_prop_grip": (0.400, 0.29),
                "tool_prop_trigger": (0.25, 0.10),
            },
            "root:1/visual-region:grip": {
                "tool_prop_stock": (0.34, 0.20),
                "tool_prop_magazine": (0.410, 0.31),
                "tool_prop_grip": (0.402, 0.30),
                "tool_prop_trigger": (0.26, 0.10),
            },
            "root:1/visual-region:trigger": {
                "tool_prop_stock": (0.20, 0.10),
                "tool_prop_magazine": (0.30, 0.20),
                "tool_prop_grip": (0.43, 0.40),
                "tool_prop_trigger": (0.45, 0.30),
            },
        }
    )

    result = rerank_visual_candidates(
        Image.new("RGB", (100, 100), "white"),
        [magazine, grip, trigger],
        [root],
        [root, stock],
        {"tool_prop": domain},
        ranker,
    )

    by_key = {
        str(candidate.metadata["candidate_key"]): candidate
        for candidate in result.candidates
    }
    assert (
        by_key["root:1/visual-region:magazine"].semantic_name
        == "tool_prop_magazine"
    )
    assert by_key["root:1/visual-region:grip"].semantic_name == "tool_prop_grip"
    assert by_key["root:1/visual-region:trigger"].metadata[
        "generic_visual_region"
    ]
    assert (
        by_key["root:1/visual-region:magazine"].metadata[
            "semantic_axis_structure_rescue"
        ]
        is True
    )
    assert (
        by_key["root:1/visual-region:grip"].metadata[
            "semantic_axis_structure_rescue"
        ]
        is True
    )
    axis = result.diagnostics["roots"][0]["axis_context"]
    assert axis["anchor_count"] == 1
    assert axis["orientation_margin"] > 0.05


def test_high_confidence_visual_endpoint_can_orient_without_becoming_truth() -> None:
    root = MaskCandidate(
        semantic_name="tool_prop",
        semantic_parent="tool_prop",
        mask=_mask(25, 75, 5, 95),
        score=0.95,
        source="test/root",
        prompt="rifle",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
            "selected_part_profile": "firearm",
        },
    )
    stock = replace(
        _visual(
            "root:1/visual-region:stock",
            _mask(30, 70, 75, 92),
            kind="panel",
            fraction=0.15,
        ),
        semantic_parent="tool_prop",
    )
    magazine = replace(
        _visual(
            "root:1/visual-region:magazine",
            _mask(45, 70, 30, 48),
            kind="panel",
            fraction=0.10,
        ),
        semantic_parent="tool_prop",
    )
    domain = DomainPrompt(
        name="tool_prop",
        root_prompts=("tool",),
        parts=(
            PartPrompt(
                "tool_prop_stock",
                ("stock",),
                maximum_instances=1,
                axis_position=0.85,
                axis_tolerance=0.38,
            ),
            PartPrompt(
                "tool_prop_magazine",
                ("magazine",),
                maximum_instances=1,
                axis_position=-0.15,
                axis_tolerance=0.32,
            ),
        ),
        part_profiles=(
            PartProfile(
                "firearm",
                ("rifle",),
                ("tool_prop_stock", "tool_prop_magazine"),
            ),
        ),
    )
    ranker = _Ranker(
        {
            "root:1/visual-region:stock": {
                "tool_prop_stock": (0.45, 0.40),
                "tool_prop_magazine": (0.20, 0.10),
            },
            "root:1/visual-region:magazine": {
                "tool_prop_stock": (0.32, 0.20),
                "tool_prop_magazine": (0.405, 0.30),
            },
        }
    )

    result = rerank_visual_candidates(
        Image.new("RGB", (100, 100), "white"),
        [stock, magazine],
        [root],
        [root],
        {"tool_prop": domain},
        ranker,
    )

    axis = result.diagnostics["roots"][0]["axis_context"]
    assert axis is not None
    assert axis["provisional_anchor_count"] >= 1
    assert axis["orientation_margin"] > 0.05


def test_final_axis_gate_rejects_a_central_part_at_the_terminal_end() -> None:
    root = MaskCandidate(
        semantic_name="tool_prop",
        semantic_parent="tool_prop",
        mask=_mask(25, 75, 5, 95),
        score=0.95,
        source="test/root",
        prompt="rifle",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
            "selected_part_profile": "firearm",
        },
    )
    barrel = MaskCandidate(
        semantic_name="tool_prop_barrel",
        semantic_parent="tool_prop",
        mask=_mask(35, 55, 8, 42),
        score=0.92,
        source="test/detector-a",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1/barrel",
            "parent_candidate_key": "root:1",
            "root_area_fraction": 0.15,
        },
    )
    false_handle = MaskCandidate(
        semantic_name="tool_prop_charging_handle",
        semantic_parent="tool_prop",
        mask=_mask(34, 56, 82, 93),
        score=0.74,
        source="test/detector-b",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1/false-handle",
            "parent_candidate_key": "root:1",
            "root_area_fraction": 0.03,
        },
    )
    domain = DomainPrompt(
        name="tool_prop",
        root_prompts=("rifle",),
        parts=(
            PartPrompt(
                "tool_prop_barrel",
                ("barrel",),
                maximum_instances=1,
                axis_position=-0.65,
                axis_tolerance=0.42,
            ),
            PartPrompt(
                "tool_prop_charging_handle",
                ("charging handle",),
                maximum_parent_fraction=0.08,
                maximum_instances=1,
                axis_position=0.0,
                axis_tolerance=0.45,
                detail=True,
            ),
        ),
        part_profiles=(
            PartProfile(
                "firearm",
                ("rifle",),
                ("tool_prop_barrel", "tool_prop_charging_handle"),
            ),
        ),
    )

    result = enforce_axis_consistency(
        [root, barrel, false_handle],
        [root],
        {"tool_prop": domain},
    )

    assert {candidate.semantic_name for candidate in result.candidates} == {
        "tool_prop",
        "tool_prop_barrel",
    }
    assert result.diagnostics["rejected_candidate_count"] == 1
    rejected = result.diagnostics["roots"][0]["evaluated"][1]
    assert rejected["semantic_name"] == "tool_prop_charging_handle"
    assert rejected["axis_distance"] > rejected["axis_hard_limit"]


def test_visual_semantic_reranker_includes_object_context_in_part_labels() -> None:
    original = _root()
    root = replace(
        original,
        prompt="smartphone",
        metadata={**original.metadata, "root_model_label": "smartphone"},
    )
    visual = _visual(
        "root:1/visual-region:01",
        _mask(20, 65, 20, 80),
        kind="panel",
        fraction=0.33,
    )

    class CapturingRanker(_Ranker):
        def __init__(self, rows: dict[str, dict[str, tuple[float, float]]]) -> None:
            super().__init__(rows)
            self.labels: list[tuple[str, str]] = []

        def rank_regions_labels(
            self,
            image: Image.Image,
            regions: list[tuple[str, np.ndarray]],
            labels: list[tuple[str, str]],
            **kwargs: object,
        ) -> dict[str, dict[str, dict[str, float | str | int]]]:
            self.labels = labels
            return super().rank_regions_labels(image, regions, labels, **kwargs)

    ranker = CapturingRanker(
        {
            "root:1/visual-region:01": {
                "device_screen": (0.42, 0.78),
                "device_button": (0.31, 0.22),
            }
        }
    )
    rerank_visual_candidates(
        Image.new("RGB", (100, 100), "white"),
        [visual],
        [root],
        [root],
        {"device": _domain()},
        ranker,
    )

    prompts = dict(ranker.labels)
    assert prompts["device_screen"] == "phone screen of a smartphone"
    assert prompts["device_button"] == "phone button of a smartphone"


def test_resolved_profile_replaces_an_incompatible_router_context() -> None:
    original = _root()
    root = replace(
        original,
        prompt="screwdriver",
        metadata={
            **original.metadata,
            "root_model_label": "screwdriver",
            "resolved_object_label": "screwdriver",
        },
    )
    visual = _visual(
        "root:1/visual-region:01",
        _mask(20, 65, 20, 80),
        kind="panel",
        fraction=0.33,
    )

    class CapturingRanker(_Ranker):
        def __init__(self, rows: dict[str, dict[str, tuple[float, float]]]) -> None:
            super().__init__(rows)
            self.labels: list[tuple[str, str]] = []

        def rank_regions_labels(
            self,
            image: Image.Image,
            regions: list[tuple[str, np.ndarray]],
            labels: list[tuple[str, str]],
            **kwargs: object,
        ) -> dict[str, dict[str, dict[str, float | str | int]]]:
            self.labels = labels
            return super().rank_regions_labels(image, regions, labels, **kwargs)

    ranker = CapturingRanker(
        {
            "root:1/visual-region:01": {
                "device_screen": (0.42, 0.78),
                "device_button": (0.31, 0.22),
            }
        }
    )
    rerank_visual_candidates(
        Image.new("RGB", (100, 100), "white"),
        [visual],
        [root],
        [root],
        {"device": _domain()},
        ranker,
    )

    prompts = dict(ranker.labels)
    assert prompts["device_screen"] == "phone screen"
    assert "screwdriver" not in " ".join(prompts.values())


def test_contextual_route_cannot_override_confident_base_semantics() -> None:
    original = _root()
    root = replace(
        original,
        prompt="smartphone",
        metadata={**original.metadata, "root_model_label": "smartphone"},
    )
    visual = _visual(
        "root:1/visual-region:01",
        _mask(20, 65, 20, 80),
        kind="panel",
        fraction=0.33,
    )

    class RouteRanker(_Ranker):
        def rank_regions_labels(
            self,
            image: Image.Image,
            regions: list[tuple[str, np.ndarray]],
            labels: list[tuple[str, str]],
            **kwargs: object,
        ) -> dict[str, dict[str, dict[str, float | str | int]]]:
            contextual = any(" of a " in prompt for _, prompt in labels)
            rows = {
                "root:1/visual-region:01": (
                    {
                        "device_screen": (0.31, 0.20),
                        "device_button": (0.62, 0.80),
                    }
                    if contextual
                    else {
                        "device_screen": (0.62, 0.80),
                        "device_button": (0.31, 0.20),
                    }
                )
            }
            return _Ranker(rows).rank_regions_labels(image, regions, labels, **kwargs)

    result = rerank_visual_candidates(
        Image.new("RGB", (100, 100), "white"),
        [visual],
        [root],
        [root],
        {"device": _domain()},
        RouteRanker({}),
    )

    assert result.candidates[0].semantic_name == "device_screen"
    row = result.diagnostics["roots"][0]["candidates"][0]
    assert row["selected_route"] == "base"


def test_contextual_route_can_rescue_an_uncertain_base_label() -> None:
    original = _root()
    root = replace(
        original,
        prompt="smartphone",
        metadata={**original.metadata, "root_model_label": "smartphone"},
    )
    visual = _visual(
        "root:1/visual-region:01",
        _mask(20, 65, 20, 80),
        kind="panel",
        fraction=0.33,
    )

    class RescueRanker(_Ranker):
        def rank_regions_labels(
            self,
            image: Image.Image,
            regions: list[tuple[str, np.ndarray]],
            labels: list[tuple[str, str]],
            **kwargs: object,
        ) -> dict[str, dict[str, dict[str, float | str | int]]]:
            contextual = any(" of a " in prompt for _, prompt in labels)
            rows = {
                "root:1/visual-region:01": (
                    {
                        "device_screen": (0.62, 0.82),
                        "device_button": (0.31, 0.18),
                    }
                    if contextual
                    else {
                        "device_screen": (0.402, 0.51),
                        "device_button": (0.400, 0.49),
                    }
                )
            }
            return _Ranker(rows).rank_regions_labels(image, regions, labels, **kwargs)

    result = rerank_visual_candidates(
        Image.new("RGB", (100, 100), "white"),
        [visual],
        [root],
        [root],
        {"device": _domain()},
        RescueRanker({}),
    )

    assert result.candidates[0].semantic_name == "device_screen"
    row = result.diagnostics["roots"][0]["candidates"][0]
    assert row["selected_route"] == "contextual_rescue"


def test_visual_semantic_reranker_preserves_generic_id_when_ambiguous() -> None:
    root = _root()
    visual = _visual(
        "root:1/visual-region:01",
        _mask(20, 65, 20, 80),
        kind="panel",
        fraction=0.33,
    )
    result = rerank_visual_candidates(
        Image.new("RGB", (100, 100), "white"),
        [visual],
        [root],
        [root],
        {"device": _domain()},
        _Ranker(
            {
                "root:1/visual-region:01": {
                    "device_screen": (0.402, 0.51),
                    "device_button": (0.400, 0.49),
                }
            }
        ),
    )

    assert result.candidates[0] == visual
    assert result.diagnostics["accepted_semantic_count"] == 0
    assert result.diagnostics["unresolved_generic_count"] == 1


def test_visual_semantic_reranker_preserves_ids_for_unresolved_profile() -> None:
    root = _root(profile=None)
    visual = _visual(
        "root:1/visual-region:01",
        _mask(30, 45, 30, 45),
        kind="detail",
        fraction=0.03,
    )
    result = rerank_visual_candidates(
        Image.new("RGB", (100, 100), "white"),
        [visual],
        [root],
        [root],
        {"device": _domain()},
        _Ranker({}),
    )

    assert result.candidates[0] == visual
    assert result.diagnostics["accepted_semantic_count"] == 0
    root_row = result.diagnostics["roots"][0]
    assert root_row["status"] == "insufficient_label_contrast"


def test_visual_semantic_reranker_rejects_impossible_detail_scale() -> None:
    root = _root()
    visual = _visual(
        "root:1/visual-region:01",
        _mask(12, 88, 12, 88),
        kind="panel",
        fraction=0.72,
    )
    result = rerank_visual_candidates(
        Image.new("RGB", (100, 100), "white"),
        [visual],
        [root],
        [root],
        {"device": _domain()},
        _Ranker(
            {
                "root:1/visual-region:01": {
                    "device_screen": (0.30, 0.10),
                    "device_button": (0.65, 0.90),
                }
            }
        ),
    )

    assert result.candidates[0].metadata["generic_visual_region"] is True
    assert result.diagnostics["accepted_semantic_count"] == 0


def test_visual_semantic_reranker_honours_maximum_instances() -> None:
    root = _root()
    visuals = [
        _visual(
            f"root:1/visual-region:{index:02d}",
            _mask(10 + index * 8, 15 + index * 8, 20, 25),
            kind="detail",
            fraction=0.003,
        )
        for index in range(1, 4)
    ]
    rows = {
        f"root:1/visual-region:{index:02d}": {
            "device_screen": (0.20, 0.05),
            "device_button": (0.55 - index * 0.01, 0.95),
        }
        for index in range(1, 4)
    }
    result = rerank_visual_candidates(
        Image.new("RGB", (100, 100), "white"),
        visuals,
        [root],
        [root],
        {"device": _domain()},
        _Ranker(rows),
    )

    assert (
        sum(
            candidate.semantic_name == "device_button"
            for candidate in result.candidates
        )
        == 2
    )
    assert (
        sum(
            bool(candidate.metadata["generic_visual_region"])
            for candidate in result.candidates
        )
        == 1
    )


def test_within_root_repetition_consensus_rescues_near_threshold_details() -> None:
    root = replace(
        _root(),
        prompt="remote control",
        metadata={
            **_root().metadata,
            "selected_part_profile": "controls",
        },
    )
    names = (
        "device_button",
        "device_logo",
        "device_port",
        "device_switch",
        "device_screen",
    )
    parts = tuple(
        PartPrompt(
            name,
            (name.replace("_", " "),),
            semantic_parent="device",
            maximum_parent_fraction=0.08,
            maximum_instances=8 if name == "device_button" else 2,
            detail=name != "device_screen",
        )
        for name in names
    )
    domain = DomainPrompt(
        name="device",
        root_prompts=("remote control",),
        parts=parts,
        part_profiles=(
            PartProfile("controls", ("remote control",), names),
        ),
    )
    visuals = [
        _visual("root:1/visual-region:01", _mask(20, 28, 20, 28), kind="detail", fraction=0.008),
        _visual("root:1/visual-region:02", _mask(35, 43, 20, 28), kind="detail", fraction=0.008),
        _visual("root:1/visual-region:03", _mask(50, 58, 20, 28), kind="detail", fraction=0.008),
    ]
    rows = {
        "root:1/visual-region:01": {
            "device_button": (0.60, 0.70),
            "device_logo": (0.20, 0.10),
            "device_port": (0.18, 0.08),
            "device_switch": (0.16, 0.07),
            "device_screen": (0.14, 0.05),
        },
        "root:1/visual-region:02": {
            "device_button": (0.410, 0.22),
            "device_logo": (0.405, 0.21),
            "device_port": (0.20, 0.19),
            "device_switch": (0.18, 0.18),
            "device_screen": (0.16, 0.20),
        },
        "root:1/visual-region:03": {
            "device_button": (0.410, 0.10),
            "device_logo": (0.405, 0.25),
            "device_port": (0.20, 0.22),
            "device_switch": (0.18, 0.21),
            "device_screen": (0.16, 0.22),
        },
    }

    result = rerank_visual_candidates(
        Image.new("RGB", (100, 100), "white"),
        visuals,
        [root],
        [root],
        {"device": domain},
        _Ranker(rows),
        config=VisualSemanticConfig(),
    )

    assert [candidate.semantic_name for candidate in result.candidates[:2]] == [
        "device_button",
        "device_button",
    ]
    assert result.candidates[1].metadata[
        "semantic_within_root_repetition_consensus"
    ] is True
    assert result.candidates[2].metadata["generic_visual_region"] is True
    assert result.diagnostics["within_root_repetition_consensus_count"] == 1


def test_within_root_consensus_reassigns_capacity_exhausted_confusion() -> None:
    root = replace(
        _root(),
        prompt="remote control",
        metadata={**_root().metadata, "selected_part_profile": "controls"},
    )
    parts = (
        PartPrompt(
            "device_button",
            ("button",),
            semantic_parent="device",
            maximum_parent_fraction=0.08,
            maximum_instances=8,
            detail=True,
        ),
        PartPrompt(
            "device_logo",
            ("logo",),
            semantic_parent="device",
            maximum_parent_fraction=0.08,
            maximum_instances=1,
            detail=True,
        ),
        PartPrompt(
            "device_port",
            ("port",),
            semantic_parent="device",
            maximum_parent_fraction=0.08,
            detail=True,
        ),
        PartPrompt(
            "device_switch",
            ("switch",),
            semantic_parent="device",
            maximum_parent_fraction=0.08,
            detail=True,
        ),
        PartPrompt(
            "device_screen",
            ("screen",),
            semantic_parent="device",
            maximum_parent_fraction=0.20,
        ),
    )
    domain = DomainPrompt(
        name="device",
        root_prompts=("remote control",),
        parts=parts,
        part_profiles=(
            PartProfile(
                "controls",
                ("remote control",),
                tuple(part.semantic_name for part in parts),
                confusion_groups=(("device_button", "device_logo"),),
            ),
        ),
    )
    visual_anchor = _visual(
        "root:1/visual-region:01",
        _mask(20, 28, 20, 28),
        kind="detail",
        fraction=0.008,
    )
    visual_ambiguous = _visual(
        "root:1/visual-region:02",
        _mask(35, 43, 20, 28),
        kind="detail",
        fraction=0.008,
    )
    existing_logo = replace(
        _visual(
            "root:1/existing-logo",
            _mask(65, 70, 35, 55),
            kind="detail",
            fraction=0.01,
        ),
        semantic_name="device_logo",
        metadata={
            **_visual(
                "root:1/existing-logo",
                _mask(65, 70, 35, 55),
                kind="detail",
                fraction=0.01,
            ).metadata,
            "generic_visual_region": False,
        },
    )
    rows = {
        "root:1/visual-region:01": {
            "device_button": (0.60, 0.70),
            "device_logo": (0.20, 0.08),
            "device_port": (0.18, 0.07),
            "device_switch": (0.16, 0.06),
            "device_screen": (0.14, 0.05),
        },
        "root:1/visual-region:02": {
            "device_logo": (0.43, 0.30),
            "device_button": (0.41, 0.27),
            "device_port": (0.20, 0.16),
            "device_switch": (0.18, 0.15),
            "device_screen": (0.16, 0.12),
        },
    }

    result = rerank_visual_candidates(
        Image.new("RGB", (100, 100), "white"),
        [visual_anchor, visual_ambiguous],
        [root],
        [root, existing_logo],
        {"device": domain},
        _Ranker(rows),
    )

    assert [candidate.semantic_name for candidate in result.candidates] == [
        "device_button",
        "device_button",
    ]
    assert (
        result.candidates[1].metadata["semantic_confusion_reassigned_from"]
        == "device_logo"
    )


def test_within_root_consensus_propagates_verified_repeated_macro_part() -> None:
    original_root = _root()
    root = replace(
        original_root,
        prompt="wheeled device",
        metadata={
            **original_root.metadata,
            "selected_part_profile": "rolling_device",
        },
    )
    parts = (
        PartPrompt(
            "device_wheel",
            ("wheel",),
            semantic_parent="device",
            maximum_parent_fraction=0.35,
            maximum_instances=8,
        ),
        PartPrompt(
            "device_tire",
            ("tire",),
            semantic_parent="device_wheel",
            maximum_parent_fraction=0.35,
            maximum_instances=8,
        ),
    )
    domain = DomainPrompt(
        name="device",
        root_prompts=("wheeled device",),
        parts=parts,
        part_profiles=(
            PartProfile(
                "rolling_device",
                ("wheeled device",),
                tuple(part.semantic_name for part in parts),
            ),
        ),
    )
    first = _visual(
        "root:1/visual-region:01",
        _mask(20, 50, 10, 40),
        kind="panel",
        fraction=0.11,
    )
    second = _visual(
        "root:1/visual-region:02",
        _mask(55, 85, 60, 90),
        kind="panel",
        fraction=0.11,
    )
    physical_evidence = {
        "independent_cue_count": 3,
        "boundary_closure": 0.82,
        "shading_only_penalty": 0.08,
        "boundary_alignment": 0.84,
        "multi_view_confirmed": True,
    }
    first = replace(
        first,
        metadata={**first.metadata, "appearance_graph_evidence": physical_evidence},
    )
    second = replace(
        second,
        metadata={**second.metadata, "appearance_graph_evidence": physical_evidence},
    )
    result = rerank_visual_candidates(
        Image.new("RGB", (100, 100), "white"),
        [first, second],
        [root],
        [root],
        {"device": domain},
        _Ranker(
            {
                "root:1/visual-region:01": {
                    "device_wheel": (0.62, 0.80),
                    "device_tire": (0.20, 0.20),
                },
                "root:1/visual-region:02": {
                    "device_tire": (0.411, 0.13),
                    "device_wheel": (0.410, 0.12),
                },
            }
        ),
        config=VisualSemanticConfig(uniform_probability_multiplier=0.32),
    )

    assert [candidate.semantic_name for candidate in result.candidates] == [
        "device_wheel",
        "device_wheel",
    ]
    assert result.candidates[1].metadata["semantic_repetition_kind"] == (
        "macro_component"
    )
    assert result.diagnostics["within_root_repetition_consensus_count"] == 1


def test_repeated_instance_consensus_rescues_near_threshold_part() -> None:
    roots = []
    visuals = []
    rows: dict[str, dict[str, tuple[float, float]]] = {}
    for index in range(1, 5):
        original_root = _root()
        root = replace(
            original_root,
            metadata={
                **original_root.metadata,
                "root_index": index,
                "candidate_key": f"root:{index}",
            },
        )
        original_visual = _visual(
            f"root:{index}/visual-region:01",
            _mask(20, 65, 20, 80),
            kind="panel",
            fraction=0.33,
        )
        visual = replace(
            original_visual,
            metadata={
                **original_visual.metadata,
                "root_index": index,
                "candidate_key": f"root:{index}/visual-region:01",
                "parent_candidate_key": f"root:{index}",
                "assembly_parent_candidate_key": f"root:{index}",
            },
        )
        roots.append(root)
        visuals.append(visual)
        rows[f"root:{index}/visual-region:01"] = (
            {
                "device_screen": (0.60, 0.80),
                "device_button": (0.20, 0.20),
            }
            if index < 4
            else {
                "device_screen": (0.402, 0.53),
                "device_button": (0.400, 0.47),
            }
        )

    result = rerank_visual_candidates(
        Image.new("RGB", (100, 100), "white"),
        visuals,
        roots,
        roots,
        {"device": _domain()},
        _Ranker(rows),
    )

    rescued = result.candidates[-1]
    assert rescued.semantic_name == "device_screen"
    assert rescued.metadata["semantic_repeated_instance_consensus"] is True
    assert result.diagnostics["direct_accepted_semantic_count"] == 3
    assert result.diagnostics["repeated_instance_consensus_count"] == 1


def test_existing_semantic_part_consumes_visual_assignment_capacity() -> None:
    root = _root()
    screen_mask = _mask(20, 65, 20, 80)
    existing = MaskCandidate(
        semantic_name="device_screen",
        semantic_parent="device_body",
        mask=screen_mask,
        score=0.92,
        source="test/existing-screen",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1/existing-screen",
        },
    )
    visual = _visual(
        "root:1/visual-region:01",
        screen_mask.copy(),
        kind="panel",
        fraction=0.33,
    )
    result = rerank_visual_candidates(
        Image.new("RGB", (100, 100), "white"),
        [visual],
        [root],
        [root, existing],
        {"device": _domain()},
        _Ranker(
            {
                "root:1/visual-region:01": {
                    "device_screen": (0.60, 0.92),
                    "device_button": (0.20, 0.08),
                }
            }
        ),
    )

    assert result.candidates[0].semantic_name == "device_visual_panel_01"
    assert result.candidates[0].metadata["generic_visual_region"] is True
    row = result.diagnostics["roots"][0]
    assert row["existing_by_semantic"]["device_screen"] == 1


def test_sparse_prototype_inventory_cannot_delete_resolved_profile_parts() -> None:
    root = _root()
    visual = _visual(
        "root:1/visual-region:01",
        _mask(20, 65, 20, 80),
        kind="panel",
        fraction=0.33,
    )
    result = rerank_visual_candidates(
        Image.new("RGB", (100, 100), "white"),
        [visual],
        [root],
        [root],
        {"device": _domain()},
        _Ranker(
            {
                "root:1/visual-region:01": {
                    "device_screen": (0.60, 0.92),
                    "device_button": (0.20, 0.08),
                }
            }
        ),
        semantic_constraints={"test::1": {"device_button": 1}},
    )

    assert result.candidates[0].semantic_name == "device_screen"
    row = result.diagnostics["roots"][0]
    assert row["prototype_inventory_available"] is True
    assert row["prototype_inventory_constrained"] is False
    assert row["prototype_inventory_advisory"] is True
    assert row["eligible_part_count"] == 2


def _closed_profile_domain() -> DomainPrompt:
    return DomainPrompt(
        name="device",
        root_prompts=("phone",),
        parts=(
            PartPrompt(
                "device_screen",
                ("screen",),
                semantic_parent="device",
                maximum_instances=1,
            ),
        ),
        part_profiles=(
            PartProfile("phone", ("phone",), ("device_screen",)),
        ),
    )


def test_physical_region_gate_rejects_uncorroborated_interior_texture() -> None:
    root = _root()
    texture = _visual(
        "texture",
        _mask(35, 55, 35, 55),
        kind="detail",
        fraction=0.05,
    )

    result = filter_unresolved_visual_regions(
        [root, texture],
        [root],
        {"device": _closed_profile_domain()},
    )

    assert result.candidates == (root,)
    assert result.diagnostics["rejected_texture_fragment_count"] == 1


def test_physical_region_gate_keeps_limited_interior_structure() -> None:
    root = _root()
    panels = [
        _visual(
            f"panel-{index}",
            _mask(20 + index * 10, 28 + index * 10, 30, 70),
            kind="panel",
            fraction=0.04,
        )
        for index in range(3)
    ]

    result = filter_unresolved_visual_regions(
        [root, *panels],
        [root],
        {"device": _closed_profile_domain()},
    )

    assert len(result.candidates) == 3
    assert result.diagnostics["profile_structural_fallback_count"] == 2
    assert result.diagnostics["rejected_texture_fragment_count"] == 1


def test_physical_region_gate_keeps_outer_silhouette_structure() -> None:
    root = _root()
    structural = _visual(
        "structural",
        _mask(30, 70, 5, 30),
        kind="panel",
        fraction=0.12,
    )

    result = filter_unresolved_visual_regions(
        [root, structural],
        [root],
        {"device": _closed_profile_domain()},
    )

    keys = [
        candidate.metadata.get("candidate_key") for candidate in result.candidates
    ]
    assert keys == ["root:1", "structural"]
    assert result.diagnostics["silhouette_structure_count"] == 1


def test_physical_region_gate_rejects_nested_luminance_only_edge_strip() -> None:
    root = _root(profile=None)
    strip = _visual(
        "root:1/visual-region:01",
        _mask(5, 10, 20, 70),
        kind="strip",
        fraction=0.025,
    )
    strip = replace(
        strip,
        metadata={
            **strip.metadata,
            "parent_candidate_key": "root:1/visual-region:02",
            "appearance_graph_evidence": {
                "boundary_closure": 0.54,
                "chroma_contrast": 0.07,
                "luminance_contrast": 0.19,
                "independent_cue_count": 2,
            },
        },
    )

    result = filter_unresolved_visual_regions(
        [root, strip],
        [root],
        {"device": _domain()},
    )

    assert result.candidates == (root,)
    assert result.diagnostics["laminar_surface_strip_rejected_count"] == 1


def test_physical_region_gate_keeps_multiview_edge_strip() -> None:
    root = _root(profile=None)
    strip = _visual(
        "root:1/visual-region:01",
        _mask(5, 10, 20, 70),
        kind="strip",
        fraction=0.025,
    )
    strip = replace(
        strip,
        metadata={
            **strip.metadata,
            "parent_candidate_key": "root:1/visual-region:02",
            "multi_view_confirmed": True,
            "appearance_graph_evidence": {
                "boundary_closure": 0.54,
                "chroma_contrast": 0.07,
                "luminance_contrast": 0.19,
                "independent_cue_count": 2,
            },
        },
    )

    result = filter_unresolved_visual_regions(
        [root, strip],
        [root],
        {"device": _domain()},
    )

    keys = [candidate.metadata.get("candidate_key") for candidate in result.candidates]
    assert keys == ["root:1", "root:1/visual-region:01"]


def test_physical_region_gate_keeps_small_detail_on_outer_silhouette() -> None:
    root = _root()
    structural = _visual(
        "outer-detail",
        _mask(20, 40, 5, 7),
        kind="detail",
        fraction=0.004,
    )
    structural = replace(
        structural,
        metadata={**structural.metadata, "multi_view_confirmed": True},
    )

    result = filter_unresolved_visual_regions(
        [root, structural],
        [root],
        {"device": _closed_profile_domain()},
    )

    assert [candidate.metadata.get("candidate_key") for candidate in result.candidates] == [
        "root:1",
        "outer-detail",
    ]
    assert result.diagnostics["silhouette_structure_count"] == 1


def test_physical_region_gate_rejects_single_view_detail_on_outer_silhouette() -> None:
    root = _root()
    detail = _visual(
        "single-view-detail",
        _mask(20, 40, 5, 7),
        kind="detail",
        fraction=0.004,
    )

    result = filter_unresolved_visual_regions(
        [root, detail],
        [root],
        {"device": _closed_profile_domain()},
    )

    assert result.candidates == (root,)


def test_physical_region_gate_keeps_strong_single_view_silhouette_detail() -> None:
    root = _root()
    detail = _visual(
        "strong-single-view-detail",
        _mask(20, 60, 5, 7),
        kind="detail",
        fraction=0.01,
    )

    result = filter_unresolved_visual_regions(
        [root, detail],
        [root],
        {"device": _closed_profile_domain()},
    )

    assert [candidate.metadata.get("candidate_key") for candidate in result.candidates] == [
        "root:1",
        "strong-single-view-detail",
    ]
    evidence = result.candidates[1].metadata["physical_region_gate"]
    assert evidence["strong_single_view_detail_structure"] is True


def test_physical_region_gate_suppresses_generic_duplicate_of_named_part() -> None:
    root = _root()
    visual = _visual(
        "screen-visual",
        _mask(25, 60, 25, 70),
        kind="panel",
        fraction=0.18,
    )
    named = MaskCandidate(
        semantic_name="device_screen",
        semantic_parent="device",
        mask=_mask(27, 58, 27, 68),
        score=0.8,
        source="grounded/test",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "named-screen",
        },
    )

    result = filter_unresolved_visual_regions(
        [root, visual, named],
        [root],
        {"device": _closed_profile_domain()},
    )

    keys = [
        candidate.metadata.get("candidate_key") for candidate in result.candidates
    ]
    assert keys == ["root:1", "named-screen"]
    assert result.diagnostics["nested_named_region_rejected_count"] == 1


def test_physical_region_gate_rejects_unresolved_open_world_texture() -> None:
    root = _root(profile=None)
    detail = _visual(
        "unresolved-detail",
        _mask(35, 45, 35, 45),
        kind="detail",
        fraction=0.01,
    )

    result = filter_unresolved_visual_regions(
        [root, detail],
        [root],
        {"device": _domain()},
    )

    assert result.candidates == (root,)
    assert result.diagnostics["rejected_texture_fragment_count"] == 1


def test_physical_region_gate_keeps_cross_source_confirmed_structure() -> None:
    root = _root(profile=None)
    panel = _visual(
        "appearance-panel",
        _mask(30, 65, 28, 72),
        kind="panel",
        fraction=0.16,
    )
    panel = replace(
        panel,
        source="hpid-appearance-graph/felzenszwalb",
        metadata={
            **panel.metadata,
            "appearance_graph_evidence": {
                "utility": 0.64,
                "boundary_closure": 0.52,
                "chroma_contrast": 0.24,
                "texture_contrast": 0.14,
                "independent_cue_count": 3,
            },
            "cross_source_confirmed": True,
            "supporting_source_families": ["hpid-appearance-graph", "sam2-amg"],
        },
    )

    result = filter_unresolved_visual_regions(
        [root, panel],
        [root],
        {"device": _domain()},
    )

    assert [candidate.metadata.get("candidate_key") for candidate in result.candidates] == [
        "root:1",
        "appearance-panel",
    ]
    assert result.diagnostics["cross_source_structure_count"] == 1


def test_physical_region_gate_keeps_large_closed_geometric_panel() -> None:
    root = _root(profile=None)
    panel = _visual(
        "closed-panel",
        _mask(22, 78, 24, 76),
        kind="panel",
        fraction=0.30,
    )
    panel = replace(
        panel,
        source="hpid-appearance-contour/closed-edge",
        metadata={**panel.metadata, "geometric_support": 0.84},
    )

    result = filter_unresolved_visual_regions(
        [root, panel],
        [root],
        {"device": _domain()},
        config=PhysicalRegionGateConfig(maximum_profile_structural_fallbacks=0),
    )

    assert [candidate.metadata.get("candidate_key") for candidate in result.candidates] == [
        "root:1",
        "closed-panel",
    ]
    assert result.diagnostics["shape_structure_count"] == 1


def test_physical_region_gate_rejects_tiny_closed_texture_patch() -> None:
    root = _root(profile=None)
    patch = _visual(
        "closed-texture",
        _mask(35, 45, 35, 45),
        kind="detail",
        fraction=0.01,
    )
    patch = replace(
        patch,
        source="hpid-appearance-contour/closed-edge",
        metadata={**patch.metadata, "geometric_support": 0.95},
    )

    result = filter_unresolved_visual_regions(
        [root, patch],
        [root],
        {"device": _domain()},
        config=PhysicalRegionGateConfig(maximum_profile_structural_fallbacks=0),
    )

    assert result.candidates == (root,)


def test_physical_region_gate_suppresses_texture_nested_in_closed_panel() -> None:
    root = _root(profile=None)
    host = _visual(
        "display-host",
        _mask(20, 80, 20, 80),
        kind="panel",
        fraction=0.36,
    )
    host = replace(
        host,
        source="hpid-appearance-contour/closed-edge",
        mask=host.mask & ~_mask(34, 52, 36, 52),
        metadata={**host.metadata, "geometric_support": 0.86},
    )
    icon = _visual(
        "display-icon",
        _mask(36, 48, 38, 50),
        kind="panel",
        fraction=0.014,
    )
    icon = replace(
        icon,
        metadata={**icon.metadata, "cross_source_confirmed": True},
    )

    result = filter_unresolved_visual_regions(
        [root, host, icon],
        [root],
        {"device": _domain()},
    )

    assert [candidate.metadata.get("candidate_key") for candidate in result.candidates] == [
        "root:1",
        "display-host",
    ]
    assert result.diagnostics["nested_surface_texture_rejected_count"] == 1


def test_physical_region_gate_suppresses_generic_duplicate_inside_named_part() -> None:
    root = _root(profile=None)
    named_eye = MaskCandidate(
        semantic_name="device_button",
        semantic_parent="device",
        mask=_mask(35, 50, 35, 50),
        score=0.90,
        source="grounded/test",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "named-button",
        },
    )
    duplicate = _visual(
        "generic-duplicate",
        _mask(37, 48, 37, 48),
        kind="detail",
        fraction=0.012,
    )
    duplicate = replace(
        duplicate,
        metadata={**duplicate.metadata, "cross_source_confirmed": True},
    )

    result = filter_unresolved_visual_regions(
        [root, named_eye, duplicate],
        [root],
        {"device": _domain()},
    )

    assert [candidate.metadata.get("candidate_key") for candidate in result.candidates] == [
        "root:1",
        "named-button",
    ]
    assert result.diagnostics["nested_named_region_rejected_count"] == 1


def test_physical_region_gate_rejects_brightness_only_appearance_region() -> None:
    root = _root(profile=None)
    patch = _visual(
        "brightness-patch",
        _mask(30, 65, 28, 72),
        kind="panel",
        fraction=0.16,
    )
    patch = replace(
        patch,
        source="hpid-appearance-graph/felzenszwalb",
        metadata={
            **patch.metadata,
            "appearance_graph_evidence": {
                "utility": 0.64,
                "boundary_closure": 0.12,
                "chroma_contrast": 0.01,
                "texture_contrast": 0.02,
                "independent_cue_count": 1,
            },
        },
    )

    result = filter_unresolved_visual_regions(
        [root, patch],
        [root],
        {"device": _domain()},
        config=PhysicalRegionGateConfig(maximum_profile_structural_fallbacks=0),
    )

    assert result.candidates == (root,)
    assert result.diagnostics["cross_source_structure_count"] == 0


def test_physical_region_gate_accepts_vlm_supported_physical_detail() -> None:
    root = _root()
    detail = _visual(
        "physical-detail",
        _mask(35, 45, 35, 45),
        kind="detail",
        fraction=0.01,
    )
    detail = replace(
        detail,
        metadata={
            **detail.metadata,
            "vlm_physicality_audit": {"decision": "physical_supported"},
        },
    )

    result = filter_unresolved_visual_regions(
        [root, detail],
        [root],
        {"device": _closed_profile_domain()},
    )

    assert [candidate.metadata.get("candidate_key") for candidate in result.candidates] == [
        "root:1",
        "physical-detail",
    ]
    assert result.diagnostics["vlm_physical_supported_count"] == 1


def test_physical_region_gate_rejects_vlm_supported_surface_panel() -> None:
    root = _root()
    panel = _visual(
        "surface-panel",
        _mask(25, 55, 25, 55),
        kind="panel",
        fraction=0.12,
    )
    panel = replace(
        panel,
        metadata={
            **panel.metadata,
            "vlm_physicality_audit": {"decision": "nonphysical_supported"},
        },
    )

    result = filter_unresolved_visual_regions(
        [root, panel],
        [root],
        {"device": _closed_profile_domain()},
    )

    assert result.candidates == (root,)
    assert result.diagnostics["vlm_nonphysical_rejected_count"] == 1


def test_physical_region_gate_rejects_named_visual_surface_detail() -> None:
    root = _root()
    panel = _visual(
        "named-surface-panel",
        _mask(25, 55, 25, 55),
        kind="panel",
        fraction=0.12,
    )
    panel = replace(
        panel,
        semantic_name="device_screen",
        semantic_parent="device",
        metadata={
            **panel.metadata,
            "generic_visual_region": False,
            "vlm_physicality_audit": {"decision": "nonphysical_supported"},
        },
    )

    result = filter_unresolved_visual_regions(
        [root, panel],
        [root],
        {"device": _closed_profile_domain()},
    )

    assert result.candidates == (root,)
    assert result.diagnostics["vlm_nonphysical_rejected_count"] == 1


def test_physical_region_gate_rejects_named_highlight_or_shadow_only_region() -> None:
    root = _root()
    panel = _visual(
        "named-lighting-patch",
        _mask(25, 55, 25, 55),
        kind="panel",
        fraction=0.12,
    )
    panel = replace(
        panel,
        semantic_name="device_panel",
        semantic_parent="device",
        source="conditional-part[model]/direct-calibrated-mask",
        metadata={
            **panel.metadata,
            "generic_visual_region": False,
            "appearance_graph_evidence": {
                "shading_only_penalty": 0.86,
                "boundary_alignment": 0.33,
                "boundary_closure": 0.14,
                "chroma_contrast": 0.01,
                "texture_contrast": 0.02,
                "luminance_contrast": 0.48,
            },
        },
    )

    result = filter_unresolved_visual_regions(
        [root, panel],
        [root],
        {"device": _closed_profile_domain()},
    )

    assert result.candidates == (root,)
    assert result.diagnostics["photometric_only_named_rejected_count"] == 1


def test_physical_region_gate_blocks_generic_override_for_flat_media_profile() -> None:
    root = _root()
    detail = _visual(
        "printed-detail",
        _mask(35, 45, 35, 45),
        kind="detail",
        fraction=0.01,
    )
    detail = replace(
        detail,
        metadata={
            **detail.metadata,
            "vlm_physicality_audit": {"decision": "physical_supported"},
        },
    )

    result = filter_unresolved_visual_regions(
        [root, detail],
        [root],
        {"device": _closed_profile_domain()},
        config=PhysicalRegionGateConfig(
            generic_region_blocked_profiles=("phone",)
        ),
    )

    assert result.candidates == (root,)
