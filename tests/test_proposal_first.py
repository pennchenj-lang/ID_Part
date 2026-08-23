import numpy as np
from PIL import Image

from hpid_split.prompt_bank import DomainPrompt, PromptBank
from hpid_split.proposal_first import (
    ProposalFirstConfig,
    _primary_root_quality,
    _prune_duplicate_scene_object_envelopes,
    generate_proposal_first_roots,
    route_proposal_first_execution,
)
from hpid_split.visual_regions import VisualMaskProposal


def _mask(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((100, 100), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


class _DenseRouter:
    def rank_regions_labels(self, _image, regions, labels, **_kwargs):
        return {
            region_key: {
                label: {
                    "combined_similarity": 0.70 if label == "daily_object" else 0.20,
                    "probability": 0.85 if label == "daily_object" else 0.15,
                }
                for label, _ in labels
            }
            for region_key, _ in regions
        }


def _prompt_bank() -> PromptBank:
    return PromptBank(
        (
            DomainPrompt(
                "daily_object",
                ("object",),
                (),
                classifier_prompt="one handheld or household object",
            ),
            DomainPrompt(
                "terrain",
                ("ground",),
                (),
                classifier_prompt="one terrain surface",
            ),
        )
    )


def _proposals() -> list[VisualMaskProposal]:
    return [
        VisualMaskProposal(
            _mask(18, 84, 8, 58),
            0.92,
            boundary_alignment=0.85,
        ),
        VisualMaskProposal(
            _mask(30, 62, 20, 44),
            0.95,
            boundary_alignment=0.82,
        ),
        VisualMaskProposal(
            _mask(22, 72, 68, 94),
            0.90,
            boundary_alignment=0.80,
        ),
    ]


def _quality_for(
    *,
    fraction: float,
    estimated_foreground: float,
    dominant_component: float = 0.90,
    border_touches: int = 0,
) -> dict[str, object]:
    row = {
        "area_fraction": fraction,
        "saliency_contrast": 0.70,
        "boundary_alignment": 0.85,
        "border_touches": border_touches,
        "rejection": None,
    }
    return _primary_root_quality(
        [(0.8, None, row, "daily_object", 0.8, 0.1)],  # type: ignore[arg-type]
        [row],
        {
            "foreground_fraction": estimated_foreground,
            "dominant_component_fraction": dominant_component,
            "uniform_border": False,
        },
    )


def test_primary_quality_requests_fallback_for_incomplete_tight_crop_root() -> None:
    quality = _quality_for(fraction=0.22, estimated_foreground=0.47)

    assert quality["fallback_recommended"] is True
    assert "selected_root_under_covers_foreground_envelope" in quality["reasons"]


def test_primary_quality_accepts_complete_nonuniform_border_root() -> None:
    quality = _quality_for(fraction=0.27, estimated_foreground=0.29)

    assert quality["fallback_recommended"] is False


def test_primary_quality_requests_fallback_for_background_overcoverage() -> None:
    quality = _quality_for(
        fraction=0.40,
        estimated_foreground=0.14,
        border_touches=2,
    )

    assert quality["fallback_recommended"] is True
    assert "selected_root_over_covers_foreground_envelope" in quality["reasons"]


def test_adaptive_root_backend_uses_proposals_for_isolated_asset() -> None:
    image = np.full((120, 160, 3), 235, dtype=np.uint8)
    image[30:92, 42:118] = (70, 95, 130)

    enabled, diagnostics = route_proposal_first_execution(
        Image.fromarray(image),
        requested=True,
        root_mode="primary",
    )

    assert enabled is True
    assert diagnostics["selected_backend"] == "proposal_first"


def test_adaptive_root_backend_uses_detector_for_tight_crop() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    image[:, :80] = (55, 80, 110)
    image[:, 80:] = (150, 95, 60)

    enabled, diagnostics = route_proposal_first_execution(
        Image.fromarray(image),
        requested=True,
        root_mode="primary",
    )

    assert enabled is False
    assert diagnostics["selected_backend"] == "detector_first"


def test_adaptive_root_backend_uses_detector_for_tiny_isolated_asset() -> None:
    image = np.full((120, 160, 3), 235, dtype=np.uint8)
    image[50:70, 54:106] = (70, 95, 130)

    enabled, diagnostics = route_proposal_first_execution(
        Image.fromarray(image),
        requested=True,
        root_mode="primary",
    )

    assert enabled is False
    assert diagnostics["reason"] == "nonisolated_or_tight_crop"


def test_adaptive_root_backend_keeps_scene_proposal_reuse() -> None:
    image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))

    enabled, diagnostics = route_proposal_first_execution(
        image,
        requested=True,
        root_mode="scene",
    )

    assert enabled is True
    assert diagnostics["reason"] == "scene_reuses_global_proposals"


def test_primary_proposal_first_prefers_complete_root_over_nested_panel() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[18:84, 8:58] = (190, 55, 40)
    image[22:72, 68:94] = (40, 170, 70)

    result = generate_proposal_first_roots(
        Image.fromarray(image),
        _proposals(),
        _prompt_bank(),
        _DenseRouter(),
        config=ProposalFirstConfig(root_mode="primary"),
    )

    assert len(result.roots) == 1
    assert np.array_equal(result.roots[0].mask, _proposals()[0].mask)
    assert result.roots[0].metadata["ground_truth_used"] is False


def test_primary_clear_detail_cannot_replace_complete_articulated_root() -> None:
    image = np.full((140, 100, 3), 238, dtype=np.uint8)
    complete = np.zeros((140, 100), dtype=bool)
    complete[8:136, 18:82] = True
    headwear = np.zeros_like(complete)
    headwear[8:48, 20:80] = True
    torso = np.zeros_like(complete)
    torso[48:96, 28:72] = True
    legs = np.zeros_like(complete)
    legs[94:136, 25:75] = True
    image[complete] = (170, 110, 90)
    proposals = [
        VisualMaskProposal(complete, 0.88, boundary_alignment=0.42),
        VisualMaskProposal(headwear, 0.98, boundary_alignment=1.0),
        VisualMaskProposal(torso, 0.95, boundary_alignment=0.92),
        VisualMaskProposal(legs, 0.94, boundary_alignment=0.90),
    ]

    result = generate_proposal_first_roots(
        Image.fromarray(image),
        proposals,
        _prompt_bank(),
        _DenseRouter(),
        config=ProposalFirstConfig(root_mode="primary"),
    )

    assert len(result.roots) == 1
    assert np.array_equal(result.roots[0].mask, complete)
    evidence = result.roots[0].metadata["proposal_first_evidence"]
    assert evidence["nested_region_count"] >= 3


def test_scene_proposal_first_suppresses_nested_part_but_keeps_other_object() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[18:84, 8:58] = (190, 55, 40)
    image[22:72, 68:94] = (40, 170, 70)

    result = generate_proposal_first_roots(
        Image.fromarray(image),
        _proposals(),
        _prompt_bank(),
        _DenseRouter(),
        config=ProposalFirstConfig(root_mode="scene"),
    )

    assert len(result.roots) == 2
    areas = sorted(np.count_nonzero(root.mask) for root in result.roots)
    assert areas == sorted(
        [np.count_nonzero(_proposals()[0].mask), np.count_nonzero(_proposals()[2].mask)]
    )
    assert result.diagnostics["ground_truth_used"] is False


def test_primary_recovers_complete_foreground_from_background_proposal() -> None:
    image = np.full((100, 100, 3), 30, dtype=np.uint8)
    foreground = _mask(18, 84, 12, 88)
    image[foreground] = (180, 80, 45)
    background = ~foreground
    left_part = _mask(28, 72, 18, 45)
    right_part = _mask(30, 76, 52, 82)
    proposals = [
        VisualMaskProposal(background, 0.98, boundary_alignment=0.82),
        VisualMaskProposal(left_part, 0.96, boundary_alignment=0.96),
        VisualMaskProposal(right_part, 0.95, boundary_alignment=0.95),
    ]

    result = generate_proposal_first_roots(
        Image.fromarray(image),
        proposals,
        _prompt_bank(),
        _DenseRouter(),
        config=ProposalFirstConfig(root_mode="primary"),
    )

    assert len(result.roots) == 1
    assert np.array_equal(result.roots[0].mask, foreground)
    evidence = result.roots[0].metadata["proposal_first_evidence"]
    assert evidence["derived_from_background_complement"] is True
    assert evidence["nested_region_count"] >= 2


def test_scene_replaces_canvas_background_with_foreground_layer() -> None:
    image = np.full((100, 100, 3), 230, dtype=np.uint8)
    scene_layer = _mask(22, 92, 8, 94)
    image[scene_layer] = (70, 150, 60)
    background = ~scene_layer
    left_object = _mask(35, 65, 18, 38)
    right_object = _mask(42, 76, 64, 86)
    proposals = [
        VisualMaskProposal(background, 0.99, boundary_alignment=0.70),
        VisualMaskProposal(left_object, 0.96, boundary_alignment=0.96),
        VisualMaskProposal(right_object, 0.95, boundary_alignment=0.95),
    ]

    result = generate_proposal_first_roots(
        Image.fromarray(image),
        proposals,
        _prompt_bank(),
        _DenseRouter(),
        config=ProposalFirstConfig(root_mode="scene"),
    )

    assert not any(np.array_equal(root.mask, background) for root in result.roots)
    assert any(np.array_equal(root.mask, scene_layer) for root in result.roots)
    assert any(np.array_equal(root.mask, left_object) for root in result.roots)
    assert any(np.array_equal(root.mask, right_object) for root in result.roots)
    canvas_row = result.diagnostics["proposal_rows"][0]
    assert canvas_row["rejection"] == "background_like_scene_canvas"


def test_scene_recovers_two_objects_from_non_frame_panel_backgrounds() -> None:
    image = np.full((120, 200, 3), 245, dtype=np.uint8)
    left_object = np.zeros((120, 200), dtype=bool)
    left_object[20:108, 22:84] = True
    right_object = np.zeros_like(left_object)
    right_object[16:106, 116:184] = True
    image[left_object] = (150, 75, 45)
    image[right_object] = (80, 145, 70)

    left_panel = np.zeros_like(left_object)
    left_panel[4:116, 4:96] = True
    right_panel = np.zeros_like(left_object)
    right_panel[4:116, 104:196] = True
    left_background = left_panel & ~left_object
    right_background = right_panel & ~right_object

    left_roof = np.zeros_like(left_object)
    left_roof[20:48, 22:84] = True
    left_wall = np.zeros_like(left_object)
    left_wall[48:108, 22:84] = True
    right_roof = np.zeros_like(left_object)
    right_roof[16:46, 116:184] = True
    right_wall = np.zeros_like(left_object)
    right_wall[46:106, 116:184] = True
    proposals = [
        VisualMaskProposal(left_background, 0.98, boundary_alignment=0.82),
        VisualMaskProposal(right_background, 0.97, boundary_alignment=0.82),
        VisualMaskProposal(left_roof, 0.95, boundary_alignment=0.94),
        VisualMaskProposal(left_wall, 0.94, boundary_alignment=0.93),
        VisualMaskProposal(right_roof, 0.95, boundary_alignment=0.94),
        VisualMaskProposal(right_wall, 0.94, boundary_alignment=0.93),
    ]

    result = generate_proposal_first_roots(
        Image.fromarray(image),
        proposals,
        _prompt_bank(),
        _DenseRouter(),
        config=ProposalFirstConfig(root_mode="scene"),
    )

    assert len(result.roots) == 2
    assert any(np.array_equal(root.mask, left_object) for root in result.roots)
    assert any(np.array_equal(root.mask, right_object) for root in result.roots)
    assert all(root.metadata["atomic_scene_instance"] for root in result.roots)
    rejected = {
        row.get("rejection") for row in result.diagnostics["proposal_rows"]
    }
    assert "background_surrounding_scene_object" in rejected


def test_scene_envelope_deduplication_rejects_rectangular_panel_fill() -> None:
    panel_fill = np.zeros((100, 100), dtype=bool)
    panel_fill[5:95, 5:95] = True
    house = np.zeros_like(panel_fill)
    house[28:94, 12:88] = True
    for row in range(8, 28):
        inset = 28 - row
        house[row, 12 + inset : 88 - inset] = True
    bad_row = {
        "proposal_index": "panel-fill",
        "derived_scene_object_envelope": True,
        "area_fraction": float(panel_fill.mean()),
        "compactness": 1.0,
        "rootness": 0.90,
        "saliency_contrast": 0.62,
        "rejection": None,
    }
    good_row = {
        "proposal_index": "house",
        "derived_scene_object_envelope": True,
        "area_fraction": float(house.mean()),
        "compactness": float(house.sum() / (86 * 76)),
        "rootness": 0.86,
        "saliency_contrast": 0.78,
        "rejection": None,
    }

    kept, diagnostics = _prune_duplicate_scene_object_envelopes(
        [
            (0.90, VisualMaskProposal(panel_fill, 0.9), bad_row),
            (0.86, VisualMaskProposal(house, 0.9), good_row),
        ]
    )

    assert len(kept) == 1
    assert kept[0][2]["proposal_index"] == "house"
    assert bad_row["rejection"] == "duplicate_scene_object_envelope"
    assert diagnostics[0]["kept_proposal_index"] == "house"
