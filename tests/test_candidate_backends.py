import numpy as np
import pytest

from hpid_split.candidate_backends import (
    CandidateBackendResult,
    validate_backend_result,
)
from hpid_split.fusion import MaskCandidate


def _candidate(shape: tuple[int, int] = (24, 24)) -> MaskCandidate:
    return MaskCandidate(
        semantic_name="asset_panel",
        semantic_parent="asset",
        mask=np.ones(shape, dtype=bool),
        score=0.8,
        source="test-backend/proposal",
    )


def test_backend_boundary_accepts_auditable_model_evidence() -> None:
    result = CandidateBackendResult(
        (_candidate(),),
        {"algorithm": "test", "ground_truth_used": False},
    )

    assert validate_backend_result(result, image_shape=(24, 24)) is result


def test_backend_boundary_rejects_ground_truth_leakage() -> None:
    result = CandidateBackendResult(
        (_candidate(),),
        {"algorithm": "test", "ground_truth_used": True},
    )

    with pytest.raises(ValueError, match="ground_truth"):
        validate_backend_result(result, image_shape=(24, 24))


def test_backend_boundary_rejects_wrong_mask_shape() -> None:
    result = CandidateBackendResult(
        (_candidate((20, 20)),),
        {"algorithm": "test", "ground_truth_used": False},
    )

    with pytest.raises(ValueError, match="does not match"):
        validate_backend_result(result, image_shape=(24, 24))
