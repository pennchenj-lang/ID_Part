import numpy as np

from hpid_split.fusion import MaskCandidate
from hpid_split.profile_resolution import resolve_profile_roots
from hpid_split.prompt_bank import DomainPrompt, PartProfile


def _root(
    mask: np.ndarray,
    *,
    index: int,
    score: float,
    profile: str | None = None,
    consensus: float | None = None,
    classifier_rank: int | None = None,
    classifier_similarity: float = 0.3,
    classifier_probability: float | None = None,
    self_classified: bool = False,
    classifier_probability_ratio: float = 0.0,
    classifier_similarity_margin: float = 0.0,
) -> MaskCandidate:
    metadata: dict[str, object] = {
        "root_origin": "profile" if profile else "broad",
        "root_index": index,
        "candidate_key": f"root:{index}",
        "parent_candidate_key": None,
        "sam_quality": 0.9,
    }
    if profile is not None:
        metadata.update(
            {
                "selected_part_profile": profile,
                "profile_hint_source": "isolated_profile_query",
                "profile_consensus_score": consensus,
                "profile_detector_score": score,
                "profile_classifier": {
                    "rank": classifier_rank,
                    "probability": (
                        classifier_probability
                        if classifier_probability is not None
                        else (0.4 if classifier_rank == 1 else 0.1)
                    ),
                    "combined_similarity": classifier_similarity,
                },
                "profile_classifier_inventory_count": 10,
            }
        )
        if self_classified:
            metadata.update(
                {
                    "profile_self_classifier": metadata["profile_classifier"],
                    "profile_candidate_self_classified": True,
                    "profile_classifier_probability_ratio": (
                        classifier_probability_ratio
                    ),
                    "profile_classifier_similarity_margin": (
                        classifier_similarity_margin
                    ),
                }
            )
    return MaskCandidate(
        "device",
        "device",
        mask,
        score,
        "test/root",
        prompt=profile or "device",
        metadata=metadata,
    )


def test_consensus_uses_classifier_to_correct_higher_wrong_detector_score() -> None:
    mask = np.zeros((100, 80), dtype=bool)
    mask[5:95, 12:68] = True
    broad = _root(mask, index=1, score=0.55)
    phone = _root(
        mask,
        index=2,
        score=0.31,
        profile="phone",
        consensus=0.50,
        classifier_rank=1,
        classifier_similarity=0.31,
    )
    laptop = _root(
        mask,
        index=3,
        score=0.69,
        profile="laptop",
        consensus=0.47,
        classifier_rank=6,
        classifier_similarity=0.24,
    )

    result = resolve_profile_roots([broad], [phone, laptop], image_shape=mask.shape)

    assert result.roots[0].metadata["selected_part_profile"] == "phone"
    assert result.diagnostics["roots"][0]["classifier_support"] is True


def test_winning_profile_can_supply_more_complete_geometry() -> None:
    broad_mask = np.zeros((120, 100), dtype=bool)
    broad_mask[40:80, 30:70] = True
    complete = np.zeros((120, 100), dtype=bool)
    complete[5:115, 15:85] = True
    broad = _root(broad_mask, index=1, score=0.62)
    clock = _root(
        complete,
        index=2,
        score=0.55,
        profile="clock_watch",
        consensus=0.80,
        classifier_rank=1,
    )

    result = resolve_profile_roots([broad], [clock], image_shape=complete.shape)

    assert np.array_equal(result.roots[0].mask, complete)
    assert result.roots[0].metadata["profile_geometry_source"] == "profile::2"


def test_ambiguous_detector_only_profiles_fall_back_to_domain() -> None:
    mask = np.zeros((80, 80), dtype=bool)
    mask[10:70, 10:70] = True
    broad = _root(mask, index=1, score=0.50)
    first = _root(
        mask, index=2, score=0.50, profile="fan", consensus=0.48, classifier_rank=3
    )
    second = _root(
        mask,
        index=3,
        score=0.49,
        profile="kettle",
        consensus=0.47,
        classifier_rank=4,
    )

    result = resolve_profile_roots([broad], [first, second], image_shape=mask.shape)

    assert "selected_part_profile" not in result.roots[0].metadata
    assert result.roots[0].metadata["profile_resolution_status"] == "unresolved"
    assert result.diagnostics["ground_truth_used"] is False


def test_detector_only_profile_requires_classifier_to_rank_it_near_top() -> None:
    mask = np.zeros((80, 80), dtype=bool)
    mask[10:70, 10:70] = True
    broad = _root(mask, index=1, score=0.50)
    fan = _root(
        mask,
        index=2,
        score=0.55,
        profile="fan",
        consensus=0.42,
        classifier_rank=11,
        classifier_similarity=0.22,
    )
    lamp = _root(
        mask,
        index=3,
        score=0.30,
        profile="lamp",
        consensus=0.25,
        classifier_rank=1,
        classifier_similarity=0.24,
    )

    result = resolve_profile_roots([broad], [fan, lamp], image_shape=mask.shape)

    assert "selected_part_profile" not in result.roots[0].metadata
    assert result.diagnostics["roots"][0]["detector_only_support"] is False


def test_nearly_tied_classifier_does_not_override_strong_detector() -> None:
    mask = np.zeros((80, 80), dtype=bool)
    mask[10:70, 10:70] = True
    broad = _root(mask, index=1, score=0.50)
    blender = _root(
        mask,
        index=2,
        score=0.30,
        profile="blender",
        consensus=0.45,
        classifier_rank=1,
        classifier_similarity=0.2773,
    )
    kettle = _root(
        mask,
        index=3,
        score=0.53,
        profile="kettle",
        consensus=0.41,
        classifier_rank=2,
        classifier_similarity=0.2769,
    )

    result = resolve_profile_roots([broad], [blender, kettle], image_shape=mask.shape)

    assert "selected_part_profile" not in result.roots[0].metadata
    diagnostics = result.diagnostics["roots"][0]
    assert diagnostics["classifier_contradicted_by_detector"] is True


def test_low_relative_probability_classifier_winner_falls_back_to_domain() -> None:
    mask = np.zeros((80, 80), dtype=bool)
    mask[10:70, 10:70] = True
    broad = _root(mask, index=1, score=0.50)
    camera = _root(
        mask,
        index=2,
        score=0.52,
        profile="camera",
        consensus=0.62,
        classifier_rank=1,
        classifier_similarity=0.269,
        classifier_probability=0.168,
    )
    watch = _root(
        mask,
        index=3,
        score=0.44,
        profile="clock_watch",
        consensus=0.35,
        classifier_rank=2,
        classifier_similarity=0.258,
        classifier_probability=0.145,
    )

    result = resolve_profile_roots([broad], [camera, watch], image_shape=mask.shape)

    assert "selected_part_profile" not in result.roots[0].metadata
    assert result.diagnostics["roots"][0]["classifier_support"] is False


def test_specific_root_label_can_resolve_classifier_tie() -> None:
    mask = np.zeros((80, 80), dtype=bool)
    mask[10:70, 10:70] = True
    broad = _root(mask, index=1, score=0.50)
    broad = MaskCandidate(
        broad.semantic_name,
        broad.semantic_parent,
        broad.mask,
        broad.score,
        broad.source,
        prompt="device",
        metadata={**broad.metadata, "root_model_label": "computer mouse"},
    )
    mouse = _root(
        mask,
        index=2,
        score=0.40,
        profile="computer_peripheral",
        consensus=0.56,
        classifier_rank=1,
        classifier_similarity=0.270,
        classifier_probability=0.18,
    )
    tablet = _root(
        mask,
        index=3,
        score=0.39,
        profile="tablet",
        consensus=0.47,
        classifier_rank=2,
        classifier_similarity=0.268,
        classifier_probability=0.17,
    )
    domain = DomainPrompt(
        name="device",
        root_prompts=("computer mouse", "tablet"),
        parts=(),
        part_profiles=(
            PartProfile("computer_peripheral", ("computer mouse",), ()),
            PartProfile("tablet", ("tablet",), ()),
        ),
    )

    result = resolve_profile_roots(
        [broad],
        [mouse, tablet],
        image_shape=mask.shape,
        domains={"device": domain},
    )

    assert result.roots[0].metadata["selected_part_profile"] == ("computer_peripheral")
    assert result.diagnostics["roots"][0]["root_label_support"] is True


def test_specific_broad_profile_is_not_overwritten_by_later_weak_type() -> None:
    mask = np.zeros((80, 80), dtype=bool)
    mask[10:70, 10:70] = True
    broad = _root(mask, index=1, score=0.58)
    broad = MaskCandidate(
        broad.semantic_name,
        broad.semantic_parent,
        broad.mask,
        broad.score,
        broad.source,
        prompt="pine tree",
        metadata={
            **broad.metadata,
            "root_model_label": "pine tree",
            "selected_part_profile": "tree_or_log",
            "part_profile_specificity": 1.0,
        },
    )
    wrong = _root(
        mask,
        index=2,
        score=0.62,
        profile="crystal",
        consensus=0.78,
        classifier_rank=1,
        classifier_similarity=0.33,
    )

    result = resolve_profile_roots([broad], [wrong], image_shape=mask.shape)

    assert result.roots[0].metadata["selected_part_profile"] == "tree_or_log"
    assert result.roots[0].metadata["profile_hint_source"] == "specific_root_label"
    assert result.diagnostics["roots"][0]["resolution_source"] == (
        "specific_root_label"
    )


def test_specific_profile_is_recovered_from_preserved_detector_label() -> None:
    mask = np.zeros((80, 80), dtype=bool)
    mask[10:70, 10:70] = True
    broad = _root(mask, index=1, score=0.58)
    broad = MaskCandidate(
        "natural_object",
        "natural_object",
        broad.mask,
        broad.score,
        broad.source,
        prompt="natural object",
        metadata={**broad.metadata, "root_model_label": "pine tree"},
    )
    wrong = MaskCandidate(
        "natural_object",
        "natural_object",
        mask,
        0.62,
        "test/profile",
        prompt="crystal",
        metadata={
            **_root(
                mask,
                index=2,
                score=0.62,
                profile="crystal",
                consensus=0.78,
                classifier_rank=1,
            ).metadata,
        },
    )
    domain = DomainPrompt(
        name="natural_object",
        root_prompts=("pine tree", "crystal"),
        parts=(),
        part_profiles=(
            PartProfile("tree_or_log", ("tree", "pine tree"), ()),
            PartProfile("crystal", ("crystal",), ()),
        ),
    )

    result = resolve_profile_roots(
        [broad],
        [wrong],
        image_shape=mask.shape,
        domains={"natural_object": domain},
    )

    assert result.roots[0].metadata["selected_part_profile"] == "tree_or_log"
    assert result.roots[0].metadata["profile_hint_source"] == "specific_root_label"


def test_profile_candidate_with_conflicting_model_label_is_rejected() -> None:
    mask = np.zeros((80, 80), dtype=bool)
    mask[10:70, 10:70] = True
    broad = _root(mask, index=1, score=0.50)
    wrong = _root(
        mask,
        index=2,
        score=0.60,
        profile="bag",
        consensus=0.70,
        classifier_rank=1,
    )
    wrong = MaskCandidate(
        wrong.semantic_name,
        wrong.semantic_parent,
        wrong.mask,
        wrong.score,
        wrong.source,
        prompt="bag",
        metadata={**wrong.metadata, "root_model_label": "crate"},
    )
    correct = _root(
        mask,
        index=3,
        score=0.40,
        profile="box",
        consensus=0.55,
        classifier_rank=2,
    )
    correct = MaskCandidate(
        correct.semantic_name,
        correct.semantic_parent,
        correct.mask,
        correct.score,
        correct.source,
        prompt="box",
        metadata={**correct.metadata, "root_model_label": "crate"},
    )
    domain = DomainPrompt(
        name="device",
        root_prompts=("bag", "crate"),
        parts=(),
        part_profiles=(
            PartProfile("bag", ("bag",), ()),
            PartProfile("box", ("box", "crate"), ()),
        ),
    )

    result = resolve_profile_roots(
        [broad],
        [wrong, correct],
        image_shape=mask.shape,
        domains={"device": domain},
    )

    assert result.roots[0].metadata["selected_part_profile"] == "box"


def test_self_classified_isolated_candidate_can_replace_wrong_broad_root() -> None:
    broad_mask = np.zeros((100, 100), dtype=bool)
    broad_mask[5:35, 5:35] = True
    object_mask = np.zeros((100, 100), dtype=bool)
    object_mask[25:90, 35:85] = True
    broad = _root(broad_mask, index=1, score=0.52)
    phone = _root(
        object_mask,
        index=2,
        score=0.48,
        profile="phone",
        consensus=0.61,
        classifier_rank=1,
        classifier_similarity=0.31,
        classifier_probability=0.24,
        self_classified=True,
        classifier_probability_ratio=1.8,
        classifier_similarity_margin=0.025,
    )

    result = resolve_profile_roots([broad], [phone], image_shape=broad_mask.shape)

    assert np.array_equal(result.roots[0].mask, object_mask)
    assert result.roots[0].metadata["selected_part_profile"] == "phone"
    assert result.diagnostics["roots"][0]["isolated_root_replacement"] is True


def test_low_affinity_candidate_without_self_support_is_ignored() -> None:
    broad_mask = np.zeros((100, 100), dtype=bool)
    broad_mask[5:35, 5:35] = True
    unrelated_mask = np.zeros((100, 100), dtype=bool)
    unrelated_mask[60:90, 60:90] = True
    broad = _root(broad_mask, index=1, score=0.52)
    unrelated = _root(
        unrelated_mask,
        index=2,
        score=0.60,
        profile="phone",
        consensus=0.70,
        classifier_rank=1,
    )

    result = resolve_profile_roots([broad], [unrelated], image_shape=broad_mask.shape)

    assert np.array_equal(result.roots[0].mask, broad_mask)
    assert result.roots[0].metadata["profile_resolution_status"] == "unresolved"
