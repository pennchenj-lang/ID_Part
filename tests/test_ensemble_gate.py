from __future__ import annotations

import numpy as np

from hpid_split.ensemble_gate import filter_unresolved_ensemble_regions
from hpid_split.fusion import MaskCandidate


def _mask(x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((40, 80), dtype=bool)
    mask[5:35, x0:x1] = True
    return mask


def _root(*, profile: str | None = "firearm") -> MaskCandidate:
    metadata: dict[str, object] = {
        "root_origin": "primary",
        "root_index": 1,
        "candidate_key": "root:1",
        "parent_candidate_key": None,
    }
    if profile is not None:
        metadata["selected_part_profile"] = profile
    return MaskCandidate(
        semantic_name="tool_prop",
        semantic_parent="tool_prop",
        mask=_mask(2, 78),
        score=0.9,
        source="test/root",
        metadata=metadata,
    )


def _part(name: str, index: int) -> MaskCandidate:
    return MaskCandidate(
        semantic_name=name,
        semantic_parent="tool_prop",
        mask=_mask(5 + index * 8, 12 + index * 8),
        score=0.8,
        source="test/named",
        metadata={
            "root_origin": "primary",
            "root_index": 1,
            "candidate_key": f"part:{index}",
            "parent_candidate_key": "root:1",
        },
    )


def _generic(*, vlm_confirmed: bool = False) -> MaskCandidate:
    return MaskCandidate(
        semantic_name="tool_prop_visual_panel_01",
        semantic_parent="tool_prop",
        mask=_mask(45, 75),
        score=0.9,
        source="sam2-amg/test",
        metadata={
            "root_origin": "primary",
            "root_index": 1,
            "candidate_key": "visual:1",
            "parent_candidate_key": "root:1",
            "visual_region": True,
            "generic_visual_region": True,
            "physical_region_gate": {
                "vlm_physical_supported": vlm_confirmed,
            },
        },
    )


def test_gate_removes_unresolved_tiles_after_named_consensus() -> None:
    root = _root()
    candidates = [
        root,
        _part("tool_prop_receiver", 0),
        _part("tool_prop_barrel", 1),
        _part("tool_prop_grip", 2),
        _generic(),
    ]

    result = filter_unresolved_ensemble_regions(candidates, [root])

    assert len(result.candidates) == 4
    assert result.diagnostics["removed_generic_region_count"] == 1


def test_gate_preserves_open_set_and_weakly_named_objects() -> None:
    open_root = _root(profile=None)
    weak_root = _root()

    open_result = filter_unresolved_ensemble_regions(
        [open_root, _generic()], [open_root]
    )
    weak_result = filter_unresolved_ensemble_regions(
        [weak_root, _part("tool_prop_receiver", 0), _generic()], [weak_root]
    )

    assert len(open_result.candidates) == 2
    assert len(weak_result.candidates) == 3


def test_gate_preserves_vlm_confirmed_physical_region() -> None:
    root = _root()
    candidates = [
        root,
        _part("tool_prop_receiver", 0),
        _part("tool_prop_barrel", 1),
        _part("tool_prop_grip", 2),
        _generic(vlm_confirmed=True),
    ]

    result = filter_unresolved_ensemble_regions(candidates, [root])

    assert len(result.candidates) == 5
    assert result.diagnostics["removed_generic_region_count"] == 0
