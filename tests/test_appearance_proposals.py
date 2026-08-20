import cv2
import numpy as np
from PIL import Image

from hpid_split.appearance_proposals import (
    AppearanceProposalConfig,
    propose_appearance_regions,
)
from hpid_split.fusion import MaskCandidate


def _root(mask: np.ndarray) -> MaskCandidate:
    return MaskCandidate(
        semantic_name="asset",
        semantic_parent="asset",
        mask=mask,
        score=0.9,
        source="test",
        prompt="asset",
        metadata={"root_origin": "test", "root_index": 1},
    )


def test_multiscale_appearance_proposals_find_distinct_visible_regions() -> None:
    image = np.full((120, 160, 3), 20, dtype=np.uint8)
    root = np.zeros((120, 160), dtype=bool)
    root[18:104, 20:142] = True
    image[18:104, 20:80] = (195, 55, 45)
    image[18:104, 80:142] = (35, 90, 205)

    result = propose_appearance_regions(Image.fromarray(image), [_root(root)])

    assert len(result.proposals) >= 2
    assert all(proposal.target_root_key == "test::1" for proposal in result.proposals)
    assert all(proposal.source.startswith("hpid-appearance-") for proposal in result.proposals)
    assert result.diagnostics["material_claim"] == "appearance_proxy_only"
    assert result.diagnostics["ground_truth_used"] is False


def test_appearance_proposals_return_empty_without_roots() -> None:
    image = Image.new("RGB", (32, 32), "black")

    result = propose_appearance_regions(image, [])

    assert result.proposals == ()
    assert result.diagnostics["reason"] == "no_roots"


def test_closed_panel_is_retained_alongside_small_details() -> None:
    image = np.full((220, 180, 3), 230, dtype=np.uint8)
    root = np.zeros((220, 180), dtype=bool)
    root[12:208, 24:156] = True
    image[root] = (35, 38, 44)
    image[38:160, 39:141] = (25, 145, 190)
    image[38:42, 39:141] = (245, 245, 245)
    image[156:160, 39:141] = (245, 245, 245)
    image[38:160, 39:43] = (245, 245, 245)
    image[38:160, 137:141] = (245, 245, 245)
    image[176:184, 55:65] = (230, 90, 45)
    image[176:184, 84:94] = (230, 90, 45)
    image[176:184, 113:123] = (230, 90, 45)

    result = propose_appearance_regions(Image.fromarray(image), [_root(root)])

    panel = np.zeros_like(root)
    panel[38:160, 39:141] = True
    assert any(
        proposal.source == "hpid-appearance-contour/closed-edge"
        and np.count_nonzero(proposal.mask & panel) / np.count_nonzero(panel) >= 0.75
        for proposal in result.proposals
    )
    assert result.diagnostics["closed_contour_candidate_count"] >= 1
    assert result.diagnostics["selected_area_strata"]["large"] >= 1


def test_contour_only_mode_does_not_emit_graph_regions() -> None:
    image = np.full((160, 180, 3), 225, dtype=np.uint8)
    root = np.zeros((160, 180), dtype=bool)
    root[15:145, 20:160] = True
    image[root] = (65, 75, 90)
    cv2.ellipse(image, (70, 75), (28, 20), 0, 0, 360, (210, 230, 235), -1)
    cv2.ellipse(image, (125, 75), (28, 20), 0, 0, 360, (210, 230, 235), -1)

    result = propose_appearance_regions(
        Image.fromarray(image),
        [_root(root)],
        config=AppearanceProposalConfig(
            use_graph_regions=False,
            use_enclosed_interiors=False,
        ),
    )

    assert result.proposals
    assert all(
        proposal.source == "hpid-appearance-contour/closed-edge"
        for proposal in result.proposals
    )
    assert result.diagnostics["graph_regions_enabled"] is False
