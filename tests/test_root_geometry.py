from __future__ import annotations

import cv2
import numpy as np

from hpid_split.fusion import MaskCandidate
from hpid_split.prompt_bank import DomainPrompt, PartProfile, PartPrompt
from hpid_split.root_geometry import refine_root_geometry_from_parts


def _candidate(
    semantic_name: str,
    mask: np.ndarray,
    *,
    key: str,
    root: bool = False,
    profile: str | None = None,
) -> MaskCandidate:
    metadata: dict[str, object] = {
        "root_origin": "test",
        "root_index": 1,
        "candidate_key": key,
        "parent_candidate_key": None if root else "root:1",
        "root_model_label": "scissors",
        "profile_hint_source": "specific_root_label",
        "ground_truth_used": False,
    }
    if profile is not None:
        metadata["selected_part_profile"] = profile
    return MaskCandidate(
        semantic_name=semantic_name,
        semantic_parent="tool_prop" if not root else semantic_name,
        mask=mask,
        score=0.9,
        source="test/source",
        prompt=semantic_name,
        source_reliability=0.85,
        metadata=metadata,
    )


def _scissors_domain() -> DomainPrompt:
    parts = (
        PartPrompt(
            "tool_prop_handle",
            ("scissors handle",),
            maximum_instances=2,
        ),
        PartPrompt(
            "tool_prop_blade",
            ("scissors blade",),
            maximum_instances=2,
        ),
        PartPrompt("tool_prop_pivot", ("scissors pivot",), maximum_instances=1),
        PartPrompt(
            "tool_prop_finger_hole",
            ("scissors finger hole",),
            semantic_parent="tool_prop_handle",
            maximum_instances=2,
            detail=True,
        ),
    )
    return DomainPrompt(
        name="tool_prop",
        root_prompts=("tool", "scissors"),
        parts=parts,
        part_profiles=(
            PartProfile(
                "scissors_pliers",
                ("scissors", "pliers"),
                tuple(part.semantic_name for part in parts),
            ),
        ),
    )


def _scissors_candidates() -> list[MaskCandidate]:
    shape = (140, 220)
    pivot = np.zeros(shape, dtype=np.uint8)
    cv2.circle(pivot, (105, 70), 8, 1, -1)

    upper_blade = np.zeros(shape, dtype=np.uint8)
    lower_blade = np.zeros(shape, dtype=np.uint8)
    cv2.line(upper_blade, (103, 66), (196, 28), 1, 10)
    cv2.line(lower_blade, (103, 74), (196, 104), 1, 10)

    upper_handle = np.zeros(shape, dtype=np.uint8)
    lower_handle = np.zeros(shape, dtype=np.uint8)
    cv2.ellipse(upper_handle, (56, 43), (42, 22), -18, 0, 360, 1, -1)
    cv2.ellipse(lower_handle, (56, 98), (42, 23), 18, 0, 360, 1, -1)
    cv2.line(upper_handle, (83, 53), (105, 68), 1, 12)
    cv2.line(lower_handle, (83, 88), (105, 72), 1, 12)

    physical = (
        pivot.astype(bool)
        | upper_blade.astype(bool)
        | lower_blade.astype(bool)
        | upper_handle.astype(bool)
        | lower_handle.astype(bool)
    )
    polluted_root = cv2.dilate(
        physical.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)),
    ).astype(bool)
    polluted_root[15:45, 80:150] = True

    return [
        _candidate(
            "tool_prop",
            polluted_root,
            key="root:1",
            root=True,
            profile="scissors_pliers",
        ),
        _candidate("tool_prop_pivot", pivot.astype(bool), key="pivot:1"),
        _candidate("tool_prop_blade", upper_blade.astype(bool), key="blade:1"),
        _candidate("tool_prop_blade", lower_blade.astype(bool), key="blade:2"),
        _candidate("tool_prop_handle", upper_handle.astype(bool), key="handle:1"),
        _candidate("tool_prop_handle", lower_handle.astype(bool), key="handle:2"),
    ]


def test_hinged_graph_tightens_polluted_root_without_annotations() -> None:
    candidates = _scissors_candidates()
    root = candidates[0]

    result = refine_root_geometry_from_parts(
        candidates,
        [root],
        {"tool_prop": _scissors_domain()},
    )

    refined_root = result.candidates[0]
    assert refined_root.metadata["root_geometry_refined"] is True
    assert np.count_nonzero(refined_root.mask) < np.count_nonzero(root.mask)
    assert result.diagnostics["refined_root_count"] == 1
    assert result.diagnostics["ground_truth_used"] is False
    assert all(
        candidate.metadata.get("ground_truth_used") is False
        for candidate in result.candidates
    )


def test_incomplete_hinged_graph_keeps_original_root() -> None:
    candidates = [
        candidate
        for candidate in _scissors_candidates()
        if candidate.semantic_name != "tool_prop_pivot"
    ]
    root = candidates[0]

    result = refine_root_geometry_from_parts(
        candidates,
        [root],
        {"tool_prop": _scissors_domain()},
    )

    assert np.array_equal(result.candidates[0].mask, root.mask)
    assert result.diagnostics["refined_root_count"] == 0
    assert result.diagnostics["roots"][0]["status"] == "missing_pivot"


def test_non_hinged_profile_is_not_modified() -> None:
    candidates = _scissors_candidates()
    root = candidates[0]
    non_hinged_root = MaskCandidate(
        semantic_name=root.semantic_name,
        semantic_parent=root.semantic_parent,
        mask=root.mask,
        score=root.score,
        source=root.source,
        prompt=root.prompt,
        source_reliability=root.source_reliability,
        metadata={**root.metadata, "selected_part_profile": "hammer"},
    )
    candidates[0] = non_hinged_root

    result = refine_root_geometry_from_parts(
        candidates,
        [non_hinged_root],
        {"tool_prop": _scissors_domain()},
    )

    assert np.array_equal(result.candidates[0].mask, non_hinged_root.mask)
    assert result.diagnostics["eligible_root_count"] == 0
