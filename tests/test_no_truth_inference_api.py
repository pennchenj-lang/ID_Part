import inspect

from hpid_split.cli import _combine_profile_refinements, build_parser
from hpid_split.foundation import CandidateGeneration
from hpid_split.model import HPIDSplitNet
from hpid_split.relational import propose_relational_candidates


def test_model_forward_cannot_accept_ground_truth() -> None:
    parameters = set(inspect.signature(HPIDSplitNet.forward).parameters)
    forbidden = {"truth", "target", "annotation", "reference_mask", "ground_truth"}
    assert not parameters.intersection(forbidden)


def test_auto_cli_accepts_multiple_candidate_models_without_truth_input() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "auto",
            "--image",
            "input.png",
            "--output",
            "output",
            "--additional-grounding-model",
            "org/model-a",
            "--additional-grounding-model",
            "org/model-b",
        ]
    )

    assert args.additional_grounding_model == ["org/model-a", "org/model-b"]
    assert args.no_relational_appearance is False
    assert args.target_point is None
    forbidden = {"truth", "target", "ground_truth", "reference_mask"}
    assert not forbidden.intersection(vars(args))


def test_profile_refinement_ensemble_preserves_per_model_provenance() -> None:
    runs = [
        CandidateGeneration(
            (),
            {
                "grounding_model": "org/tiny",
                "roots": [{"root_index": 1, "selected_profile": "phone"}],
                "ground_truth_used": False,
            },
        ),
        CandidateGeneration(
            (),
            {
                "grounding_model": "org/base",
                "roots": [{"root_index": 1, "selected_profile": "phone"}],
                "ground_truth_used": False,
            },
        ),
    ]

    combined = _combine_profile_refinements(runs)

    assert combined.diagnostics["model_run_count"] == 2
    assert combined.diagnostics["grounding_models"] == ["org/tiny", "org/base"]
    assert [row["grounding_model"] for row in combined.diagnostics["roots"]] == [
        "org/tiny",
        "org/base",
    ]
    assert combined.diagnostics["ground_truth_used"] is False


def test_relational_candidate_api_cannot_accept_ground_truth() -> None:
    parameters = set(inspect.signature(propose_relational_candidates).parameters)
    forbidden = {"truth", "target", "annotation", "reference_mask", "ground_truth"}
    assert not parameters.intersection(forbidden)
