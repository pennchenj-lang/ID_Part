from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from PIL import Image

from hpid_split.fusion import MaskCandidate
from hpid_split.ontology_routing import route_scene_ontology
from hpid_split.prompt_bank import DomainPrompt, PartProfile, PromptBank


class _FixedEncoder:
    model_name = "fixed-test-encoder"

    def __init__(
        self,
        image_embeddings: np.ndarray,
        text_embeddings: dict[str, np.ndarray],
    ) -> None:
        self.image_embeddings = image_embeddings.astype(np.float32)
        self.text_embeddings = text_embeddings

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        assert len(images) == len(self.image_embeddings)
        return self.image_embeddings

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        rows = []
        for text in texts:
            key = next(key for key in self.text_embeddings if key in text)
            rows.append(self.text_embeddings[key])
        return np.asarray(rows, dtype=np.float32)


def _bank() -> PromptBank:
    return PromptBank(
        (
            DomainPrompt(
                "natural_object",
                ("natural object",),
                (),
                classifier_prompt="natural-domain",
                part_profiles=(
                    PartProfile(
                        "rock",
                        ("rock",),
                        (),
                        classifier_prompt="rock-profile",
                    ),
                    PartProfile(
                        "tree",
                        ("tree",),
                        (),
                        classifier_prompt="tree-profile",
                    ),
                    PartProfile(
                        "crystal",
                        ("crystal",),
                        (),
                        classifier_prompt="crystal-profile",
                    ),
                ),
            ),
            DomainPrompt(
                "furniture",
                ("furniture",),
                (),
                classifier_prompt="furniture-domain",
                part_profiles=(
                    PartProfile(
                        "chair",
                        ("chair",),
                        (),
                        classifier_prompt="chair-profile",
                    ),
                    PartProfile(
                        "table",
                        ("table",),
                        (),
                        classifier_prompt="table-profile",
                    ),
                ),
            ),
        )
    )


def _root(
    domain: str,
    profile: str,
    index: int,
    x0: int,
) -> MaskCandidate:
    mask = np.zeros((32, 80), dtype=bool)
    mask[4:28, x0 : x0 + 18] = True
    return MaskCandidate(
        domain,
        domain,
        mask,
        0.8,
        "test/root",
        prompt=profile,
        metadata={
            "root_origin": "test",
            "root_index": index,
            "candidate_key": f"root:{index}",
            "parent_candidate_key": None,
            "root_model_label": profile,
            "selected_part_profile": profile,
            "part_profile_specificity": 1.0,
        },
    )


def test_visual_profile_corrects_specific_detector_mistake() -> None:
    root = _root("natural_object", "crystal", 1, 4)
    encoder = _FixedEncoder(
        np.asarray([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        {
            "natural-domain": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "furniture-domain": np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            "rock-profile": np.asarray([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "tree-profile": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "crystal-profile": np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            "chair-profile": np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            "table-profile": np.asarray([0.0, 0.9, 0.0, 0.0, 0.0, 0.0]),
        },
    )

    result = route_scene_ontology(Image.new("RGB", (80, 32)), [root], _bank(), encoder)

    assert result.roots[0].semantic_name == "natural_object"
    assert result.roots[0].metadata["selected_part_profile"] == "tree"
    assert result.roots[0].metadata["ontology_profile_decision"] == (
        "visual_profile_override"
    )


def test_detector_only_agreement_cannot_become_scene_anchor() -> None:
    root = _root("furniture", "chair", 1, 4)
    encoder = _FixedEncoder(
        np.asarray([[0.0, 0.4, 0.21, 0.2, 0.0, 0.0]]),
        {
            "natural-domain": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "furniture-domain": np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            "rock-profile": np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            "tree-profile": np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            "crystal-profile": np.asarray([0.0, 0.0, 0.0, 0.0, 0.1, 0.0]),
            "chair-profile": np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            "table-profile": np.asarray([0.0, 0.0, 0.0, 0.9, 0.1, 0.0]),
        },
    )

    result = route_scene_ontology(Image.new("RGB", (80, 32)), [root], _bank(), encoder)

    row = result.diagnostics["rows"][0]
    assert row["detector_agreement"] is True
    assert row["global_visual_agreement"] is False
    assert row["anchor"] is False
    assert result.diagnostics["anchor_count"] == 0


def test_clear_global_profile_can_correct_a_wrong_broad_domain() -> None:
    root = _root("furniture", "table", 1, 4)
    encoder = _FixedEncoder(
        np.asarray([[0.0, 0.35, 0.62, 0.18, 0.0, 0.0]]),
        {
            "natural-domain": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "furniture-domain": np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            "rock-profile": np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            "tree-profile": np.asarray([0.0, 0.0, 0.1, 0.0, 0.0, 0.0]),
            "crystal-profile": np.asarray([0.0, 0.0, 0.05, 0.0, 0.0, 0.0]),
            "chair-profile": np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            "table-profile": np.asarray([0.0, 0.0, 0.0, 0.9, 0.1, 0.0]),
        },
    )

    result = route_scene_ontology(
        Image.new("RGB", (80, 32)), [root], _bank(), encoder
    )

    corrected = result.roots[0]
    assert corrected.semantic_name == "natural_object"
    assert corrected.metadata["selected_part_profile"] == "rock"
    assert corrected.metadata["ontology_domain_decision"] == (
        "global_profile_domain_override"
    )
    assert corrected.metadata["ontology_profile_decision"] == (
        "global_profile_override"
    )


def test_repeated_scene_instances_correct_one_ambiguous_wrong_domain() -> None:
    roots = [
        _root("natural_object", "rock", 1, 2),
        _root("natural_object", "rock", 2, 30),
        _root("furniture", "table", 3, 58),
    ]
    rock_a = np.asarray([0.17, 0.0, 0.17, 0.0, 0.0, 0.9707])
    rock_b = np.asarray([0.18, 0.0, 0.17, 0.0, 0.0, 0.9694])
    ambiguous = np.asarray([0.0, 0.17, 0.0, 0.13, 0.13, 0.9670])
    image_embeddings = np.stack((rock_a, rock_b, ambiguous), axis=0)
    image_embeddings /= np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    encoder = _FixedEncoder(
        image_embeddings,
        {
            "natural-domain": np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "furniture-domain": np.asarray([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            "rock-profile": np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            "tree-profile": np.asarray([0.0, 0.0, 0.05, 0.0, 0.0, 0.0]),
            "crystal-profile": np.asarray([0.0, 0.0, 0.02, 0.0, 0.0, 0.0]),
            "chair-profile": np.asarray([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            "table-profile": np.asarray([0.0, 0.0, 0.0, 0.99, 0.01, 0.0]),
        },
    )

    result = route_scene_ontology(Image.new("RGB", (80, 32)), roots, _bank(), encoder)

    corrected = result.roots[2]
    assert corrected.semantic_name == "natural_object"
    assert corrected.metadata["selected_part_profile"] == "rock"
    assert corrected.metadata["ontology_profile_decision"] == (
        "scene_repeated_instance_consensus"
    )
    assert result.diagnostics["consensus_resolution_count"] == 1
