import numpy as np

from hpid_split.fusion import MaskCandidate
from hpid_split.root_cleanup import clean_primary_roots


def _candidate(
    name: str,
    mask: np.ndarray,
    *,
    key: str,
    parent_key: str | None,
    generic: bool = False,
    source: str = "test/source",
) -> MaskCandidate:
    return MaskCandidate(
        semantic_name=name,
        semantic_parent="tool_prop" if parent_key is not None else name,
        mask=mask,
        score=0.9,
        source=source,
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": key,
            "parent_candidate_key": parent_key,
            "generic_visual_region": generic,
        },
    )


def test_root_cleanup_removes_frame_fragments_and_clips_children() -> None:
    root_mask = np.zeros((120, 160), dtype=bool)
    root_mask[20:95, 35:125] = True
    root_mask[-2:, 15:145] = True
    root_mask[98:108, 145:155] = True
    child_mask = np.zeros_like(root_mask)
    child_mask[45:75, 55:105] = True
    child_mask[-2:, 20:60] = True
    root = _candidate("tool_prop", root_mask, key="root:1", parent_key=None)
    child = _candidate(
        "tool_prop_handle",
        child_mask,
        key="root:1/handle",
        parent_key="root:1",
    )

    result = clean_primary_roots([root, child], [root])

    cleaned_root, cleaned_child = result.candidates
    assert not np.any(cleaned_root.mask[-2:])
    assert not np.any(cleaned_root.mask[:, 145:])
    assert np.all(cleaned_child.mask <= cleaned_root.mask)
    assert not np.any(cleaned_child.mask[-2:])
    assert result.diagnostics["clipped_candidate_count"] == 2


def test_root_cleanup_keeps_near_supported_detached_component() -> None:
    root_mask = np.zeros((120, 160), dtype=bool)
    root_mask[30:90, 35:110] = True
    root_mask[45:65, 112:120] = True
    root = _candidate("tool_prop", root_mask, key="root:1", parent_key=None)
    support_mask = np.zeros_like(root_mask)
    support_mask[45:65, 112:120] = True
    support = _candidate(
        "tool_prop_guard",
        support_mask,
        key="root:1/guard",
        parent_key="root:1",
    )

    result = clean_primary_roots([root, support], [root])

    cleaned_root = result.candidates[0]
    assert np.all(cleaned_root.mask[45:65, 112:120])
    row = result.diagnostics["roots"][0]
    assert row["components"][0]["independent_support"] is True
    assert row["components"][0]["kept"] is True
