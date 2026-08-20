import numpy as np

from hpid_split.fusion import MaskCandidate
from hpid_split.shape_proposals import propose_shape_regions
from hpid_split.visual_regions import VisualMaskProposal


def _root(mask: np.ndarray) -> MaskCandidate:
    return MaskCandidate(
        semantic_name="asset",
        semantic_parent="asset",
        mask=mask,
        score=0.9,
        source="test",
        metadata={"root_origin": "test", "root_index": 1},
    )


def test_shape_bottleneck_proposes_attached_lobes_without_class_rules() -> None:
    mask = np.zeros((160, 220), dtype=bool)
    mask[55:105, 70:155] = True
    mask[20:75, 20:78] = True
    mask[85:145, 145:205] = True
    mask[68:92, 64:75] = True
    mask[88:101, 150:160] = True

    result = propose_shape_regions([_root(mask)])

    assert len(result.proposals) >= 2
    assert all(
        proposal.source == "hpid-shape-bottleneck/watershed"
        for proposal in result.proposals
    )
    assert all(proposal.geometric_support >= 0.48 for proposal in result.proposals)
    assert result.diagnostics["ground_truth_used"] is False


def test_shape_bottleneck_supplements_instead_of_replacing_visual_mask() -> None:
    mask = np.zeros((160, 220), dtype=bool)
    mask[55:105, 70:155] = True
    mask[20:75, 20:78] = True
    mask[85:145, 145:205] = True
    mask[68:92, 64:75] = True
    mask[88:101, 150:160] = True
    baseline = propose_shape_regions([_root(mask)])
    assert baseline.proposals
    covered = baseline.proposals[0]

    result = propose_shape_regions(
        [_root(mask)],
        existing_visual_proposals=[
            VisualMaskProposal(covered.mask, 0.95, boundary_alignment=0.9)
        ],
    )

    assert len(result.proposals) < len(baseline.proposals)
    assert result.diagnostics["suppressed_by_visual_count"] >= 1
