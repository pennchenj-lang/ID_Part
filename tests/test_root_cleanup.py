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
    root_origin: str = "test",
    root_index: int = 1,
    profile: str | None = None,
) -> MaskCandidate:
    return MaskCandidate(
        semantic_name=name,
        semantic_parent="tool_prop" if parent_key is not None else name,
        mask=mask,
        score=0.9,
        source=source,
        metadata={
            "root_origin": root_origin,
            "root_index": root_index,
            "candidate_key": key,
            "parent_candidate_key": parent_key,
            "generic_visual_region": generic,
            "selected_part_profile": profile,
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


def test_root_cleanup_subtracts_contained_competing_object() -> None:
    root_mask = np.zeros((120, 160), dtype=bool)
    root_mask[20:100, 15:145] = True
    object_mask = np.zeros_like(root_mask)
    object_mask[35:65, 75:105] = True
    root = _candidate(
        "furniture",
        root_mask,
        key="root:1",
        parent_key=None,
        profile="table",
    )
    competitor = _candidate(
        "container",
        object_mask,
        key="root:2",
        parent_key=None,
        root_origin="other",
        root_index=2,
    )

    result = clean_primary_roots(
        [root],
        [root],
        competing_roots=[root, competitor],
    )

    cleaned_root = result.candidates[0]
    assert not np.any(cleaned_root.mask & object_mask)
    subtraction = result.diagnostics["roots"][0][
        "competing_object_subtraction"
    ]
    assert subtraction["accepted_count"] == 1
    assert subtraction["removed_area_px"] == int(np.count_nonzero(object_mask))


def test_root_cleanup_preserves_competitor_confirmed_as_asset_part() -> None:
    root_mask = np.zeros((120, 160), dtype=bool)
    root_mask[20:100, 15:145] = True
    member_mask = np.zeros_like(root_mask)
    member_mask[35:65, 75:105] = True
    root = _candidate(
        "furniture",
        root_mask,
        key="root:1",
        parent_key=None,
        profile="table",
    )
    member = _candidate(
        "furniture_basket",
        member_mask,
        key="root:1/basket",
        parent_key="root:1",
        source="independent/profile-refine",
    )
    competitor = _candidate(
        "container",
        member_mask,
        key="root:2",
        parent_key=None,
        source="other/root",
        root_origin="other",
        root_index=2,
    )

    result = clean_primary_roots(
        [root, member],
        [root],
        competing_roots=[root, competitor],
    )

    cleaned_root = result.candidates[0]
    assert np.all(cleaned_root.mask[member_mask])
    row = result.diagnostics["roots"][0]["competing_object_subtraction"][
        "rows"
    ][0]
    assert row["protected_as_selected_asset_member"] is True
    assert row["accepted"] is False


def test_root_cleanup_uses_duplicate_competitor_as_corroborating_extent() -> None:
    root_mask = np.zeros((120, 160), dtype=bool)
    root_mask[15:105, 10:150] = True
    first_mask = np.zeros_like(root_mask)
    first_mask[35:65, 65:95] = True
    broader_mask = np.zeros_like(root_mask)
    broader_mask[30:70, 60:100] = True
    root = _candidate(
        "furniture",
        root_mask,
        key="root:1",
        parent_key=None,
        profile="table",
    )
    first = _candidate(
        "container",
        first_mask,
        key="root:2",
        parent_key=None,
        root_origin="other-a",
        root_index=2,
    )
    broader = _candidate(
        "daily_object",
        broader_mask,
        key="root:3",
        parent_key=None,
        root_origin="other-b",
        root_index=3,
    )

    result = clean_primary_roots(
        [root],
        [root],
        competing_roots=[root, first, broader],
    )

    assert not np.any(result.candidates[0].mask & broader_mask)
    rows = result.diagnostics["roots"][0]["competing_object_subtraction"][
        "rows"
    ]
    assert sum(bool(row["accepted"]) for row in rows) == 2
    assert any(bool(row["duplicate_used_as_corroborating_extent"]) for row in rows)


def test_root_cleanup_does_not_subtract_nested_regions_from_non_host_assets() -> None:
    root_mask = np.zeros((120, 160), dtype=bool)
    root_mask[20:100, 15:145] = True
    nested_mask = np.zeros_like(root_mask)
    nested_mask[35:65, 75:105] = True
    root = _candidate(
        "container",
        root_mask,
        key="root:1",
        parent_key=None,
        profile="bottle_jar",
    )
    competitor = _candidate(
        "device",
        nested_mask,
        key="root:2",
        parent_key=None,
        root_origin="other",
        root_index=2,
    )

    result = clean_primary_roots(
        [root],
        [root],
        competing_roots=[root, competitor],
    )

    assert np.all(result.candidates[0].mask[nested_mask])
    subtraction = result.diagnostics["roots"][0][
        "competing_object_subtraction"
    ]
    assert subtraction["status"] == "skipped_non_host_asset"
    assert subtraction["removed_area_px"] == 0
