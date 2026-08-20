from dataclasses import replace

import numpy as np
from PIL import Image

from hpid_split.appearance_graph import optimize_appearance_graph
from hpid_split.fusion import MaskCandidate


def _mask(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((100, 100), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _root() -> MaskCandidate:
    return MaskCandidate(
        semantic_name="asset",
        semantic_parent="asset",
        mask=_mask(10, 90, 10, 90),
        score=0.95,
        source="test/root",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
        },
    )


def _visual(
    key: str,
    mask: np.ndarray,
    *,
    score: float = 0.92,
    generic: bool = True,
    kind: str = "panel",
) -> MaskCandidate:
    return MaskCandidate(
        semantic_name=(f"asset_visual_{kind}_{key}" if generic else key),
        semantic_parent="asset",
        mask=mask,
        score=score,
        source="sam2-amg/test",
        source_reliability=0.68,
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": f"root:1/{key}",
            "parent_candidate_key": "root:1",
            "generic_visual_region": generic,
            "visual_region_kind": kind,
            "sam_quality": score,
            "multi_view_confirmed": False,
            "ground_truth_used": False,
        },
    )


def test_chromatic_boundary_supports_category_independent_part() -> None:
    image = np.full((100, 100, 3), (70, 110, 180), dtype=np.uint8)
    image[20:80, 20:48] = (210, 55, 45)
    candidate = _visual("colored-panel", _mask(20, 80, 20, 48))

    result = optimize_appearance_graph(
        Image.fromarray(image), [candidate], [_root()]
    )

    assert len(result.candidates) == 1
    evidence = result.candidates[0].metadata["appearance_graph_evidence"]
    assert evidence["chroma_contrast"] >= 0.12
    assert result.diagnostics["material_claim"] == "appearance_proxy_only"
    assert result.diagnostics["ground_truth_used"] is False


def test_smooth_luminance_gradient_is_not_promoted_to_part_id() -> None:
    ramp = np.linspace(90, 150, 100, dtype=np.uint8)
    image = np.repeat(ramp[None, :, None], 100, axis=0)
    image = np.repeat(image, 3, axis=2)
    candidate = _visual("illumination-band", _mask(15, 85, 15, 50))

    result = optimize_appearance_graph(
        Image.fromarray(image), [candidate], [_root()]
    )

    assert result.candidates == ()
    assert result.diagnostics["rejections"][0]["reason"] in {
        "low_cross_cue_utility",
        "single_model_without_visible_boundary",
    }


def test_prompt_semantic_can_keep_same_color_part_without_false_material_claim() -> None:
    image = Image.fromarray(np.full((100, 100, 3), 128, dtype=np.uint8))
    candidate = _visual(
        "requested_trigger",
        _mask(45, 58, 45, 56),
        score=0.96,
        generic=False,
        kind="detail",
    )

    result = optimize_appearance_graph(image, [candidate], [_root()])

    assert len(result.candidates) == 1
    assert result.candidates[0].semantic_name == "requested_trigger"
    assert result.candidates[0].metadata["ground_truth_used"] is False


def test_non_laminar_duplicate_regions_do_not_both_survive() -> None:
    image = np.full((100, 100, 3), (45, 80, 160), dtype=np.uint8)
    image[20:82, 18:66] = (210, 50, 40)
    stronger = _visual("stronger", _mask(20, 82, 18, 66), score=0.97)
    weaker = _visual("weaker", _mask(32, 90, 40, 82), score=0.82)
    stronger = replace(
        stronger,
        metadata={**stronger.metadata, "multi_view_confirmed": True},
    )
    weaker = replace(
        weaker,
        metadata={**weaker.metadata, "multi_view_confirmed": True},
    )

    result = optimize_appearance_graph(
        Image.fromarray(image), [stronger, weaker], [_root()]
    )

    keys = {candidate.metadata["candidate_key"] for candidate in result.candidates}
    assert "root:1/stronger" in keys
    assert len(keys) == 1
    assert any(
        row["reason"] == "non_laminar_overlap"
        for row in result.diagnostics["rejections"]
    )


def test_duplicate_regions_from_independent_sources_record_confirmation() -> None:
    image = np.full((100, 100, 3), (45, 80, 160), dtype=np.uint8)
    image[20:82, 18:66] = (210, 50, 40)
    sam = _visual("sam", _mask(20, 82, 18, 66), score=0.96)
    appearance = replace(
        _visual("appearance", _mask(20, 82, 18, 66), score=0.90),
        source="hpid-appearance-graph/felzenszwalb",
        metadata={
            **_visual("appearance", _mask(20, 82, 18, 66)).metadata,
            "source_family": "hpid-appearance-graph",
        },
    )

    result = optimize_appearance_graph(
        Image.fromarray(image), [sam, appearance], [_root()]
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].metadata["cross_source_confirmed"] is True
    assert set(result.candidates[0].metadata["supporting_source_families"]) == {
        "hpid-appearance-graph",
        "sam2-amg",
    }
    assert result.diagnostics["cross_source_confirmation_count"] == 1


def test_large_textured_surface_is_retained_as_structural_panel() -> None:
    image = np.full((100, 100, 3), 118, dtype=np.uint8)
    yy, xx = np.indices((60, 54))
    texture = np.where(((xx + yy) % 2)[..., None], 0, 255)
    image[20:80, 20:74] = np.repeat(texture, 3, axis=2)
    panel_mask = _mask(20, 80, 20, 74)
    panel_mask[44:58, 42:54] = False
    candidate = replace(
        _visual("textured-door", panel_mask, score=0.58),
        source="hpid-appearance-graph/felzenszwalb",
    )

    result = optimize_appearance_graph(
        Image.fromarray(image),
        [candidate],
        [_root()],
        config=None,
    )

    assert len(result.candidates) == 1
    row = result.diagnostics["evidence"][0]
    assert row["decision"] in {
        "large_structural_surface",
        "cross_cue_supported",
    }
    if row["decision"] == "large_structural_surface":
        assert result.candidates[0].mask[50, 48]
