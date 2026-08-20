import numpy as np

from experiments.run_foundation_fusion_ablation import (
    _bottleneck_label,
    _candidate_union_for_truth_class,
)
from hpid_split.fusion import MaskCandidate


def test_bottleneck_labels_separate_proposals_from_ownership() -> None:
    assert _bottleneck_label(0.10, 0.10, 0.10) == "proposal_generation"
    assert _bottleneck_label(0.80, 0.50, 0.48) == "candidate_filtering"
    assert _bottleneck_label(0.80, 0.78, 0.40) == "fusion_or_ownership"
    assert _bottleneck_label(0.45, 0.44, 0.42) == "partial_localization"
    assert _bottleneck_label(0.80, 0.78, 0.72) == "retained"


def test_candidate_union_uses_exact_fine_mapping() -> None:
    eye = np.zeros((8, 8), dtype=bool)
    eye[2:4, 1:3] = True
    hair = np.zeros((8, 8), dtype=bool)
    hair[1:5, 1:5] = True
    candidates = [
        MaskCandidate("character_eye", "character_head", eye, 0.8, "model/a"),
        MaskCandidate("character_hair", "character", hair, 0.9, "model/a"),
    ]

    union, count = _candidate_union_for_truth_class(candidates, "eyes", eye.shape)

    assert count == 1
    assert np.array_equal(union, eye)
