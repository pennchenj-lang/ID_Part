from __future__ import annotations

import numpy as np
from PIL import Image

from hpid_split.fusion import MaskCandidate
from hpid_split.prompt_bank import DomainPrompt, PartProfile, PartPrompt
from hpid_split.semantic_candidate_audit import (
    SemanticCandidateAuditConfig,
    SemanticCandidateAuditor,
    parse_semantic_candidate_audit,
)


class FakePlanner:
    backend_id = "fake-vlm"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    def generate_response(self, image: Image.Image, prompt: str) -> str:
        self.calls += 1
        assert "Candidate semantic ID" in prompt
        return self.response


def _candidate(
    semantic_name: str,
    mask: np.ndarray,
    *,
    key: str,
    visual: bool = False,
    support_key: str | None = None,
) -> MaskCandidate:
    return MaskCandidate(
        semantic_name,
        "tool_prop_body" if semantic_name != "tool_prop" else "tool_prop",
        mask,
        0.31,
        "test/source",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": key,
            "parent_candidate_key": None if semantic_name == "tool_prop" else "root:1",
            "profile_refinement": semantic_name != "tool_prop" and not visual,
            "visual_region": visual,
            "generic_visual_region": False if visual else None,
            "visual_region_kind": "panel" if visual else None,
            "semantic_support_candidate_key": support_key,
            "root_area_fraction": 0.35,
        },
    )


def test_parser_requires_explicit_json_and_maps_discrete_confidence() -> None:
    verdict, confidence, diagnostics = parse_semantic_candidate_audit(
        'result: {"verdict":"wrong","confidence":"high",'
        '"reason_code":"whole_object"}'
    )
    assert verdict == "wrong"
    assert confidence == 0.95
    assert diagnostics["reason_code"] == "whole_object"


def test_high_confidence_wrong_label_is_removed_but_visual_mask_is_preserved() -> None:
    root_mask = np.zeros((64, 64), dtype=bool)
    root_mask[4:60, 4:60] = True
    part_mask = np.zeros_like(root_mask)
    part_mask[10:52, 12:42] = True
    root = _candidate("tool_prop", root_mask, key="root:1")
    semantic = _candidate("tool_prop_handle", part_mask, key="root:1/handle:01")
    visual = _candidate(
        "tool_prop_handle",
        part_mask,
        key="root:1/visual:01",
        visual=True,
        support_key="root:1/handle:01",
    )
    domain = DomainPrompt(
        "tool_prop",
        ("tool",),
        (
            PartPrompt("tool_prop_body", ("tool body",)),
            PartPrompt(
                "tool_prop_handle",
                ("tool handle",),
                semantic_parent="tool_prop_body",
                maximum_parent_fraction=0.45,
            ),
        ),
        part_profiles=(
            PartProfile("tool", ("tool",), ("tool_prop_handle",)),
        ),
    )
    planner = FakePlanner(
        '{"verdict":"wrong","confidence":"high",'
        '"reason_code":"whole_object"}'
    )
    result = SemanticCandidateAuditor(
        planner,
        config=SemanticCandidateAuditConfig(maximum_queries=1),
    ).audit(
        Image.new("RGB", (64, 64)),
        [root],
        [root, semantic, visual],
        {"tool_prop": domain},
    )

    assert planner.calls == 1
    assert all(_item.metadata.get("candidate_key") != "root:1/handle:01" for _item in result.candidates)
    preserved = next(
        item
        for item in result.candidates
        if item.metadata.get("candidate_key") == "root:1/visual:01"
    )
    assert preserved.metadata["generic_visual_region"] is True
    assert preserved.semantic_name.startswith("tool_prop_visual_panel_audit_")
    assert result.diagnostics["invalidated_candidate_count"] == 1
    assert result.diagnostics["genericized_visual_support_count"] == 1


def test_uncertain_audit_never_removes_candidate() -> None:
    root_mask = np.ones((32, 32), dtype=bool)
    part_mask = np.zeros_like(root_mask)
    part_mask[4:20, 4:20] = True
    root = _candidate("tool_prop", root_mask, key="root:1")
    semantic = _candidate("tool_prop_handle", part_mask, key="root:1/handle:01")
    domain = DomainPrompt(
        "tool_prop",
        ("tool",),
        (
            PartPrompt("tool_prop_body", ("tool body",)),
            PartPrompt("tool_prop_handle", ("tool handle",)),
        ),
        part_profiles=(PartProfile("tool", ("tool",), ("tool_prop_handle",)),),
    )
    result = SemanticCandidateAuditor(
        FakePlanner(
            '{"verdict":"uncertain","confidence":"high",'
            '"reason_code":"insufficient_detail"}'
        ),
        config=SemanticCandidateAuditConfig(maximum_queries=1),
    ).audit(
        Image.new("RGB", (32, 32)),
        [root],
        [root, semantic],
        {"tool_prop": domain},
    )

    assert len(result.candidates) == 2
    assert result.diagnostics["invalidated_candidate_count"] == 0
