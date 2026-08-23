from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from hpid_split.embedding_ranker import (
    ConsensusRegionLabelRanker,
    EmbeddingRegionLabelRanker,
)


class _ColourEncoder:
    model_name = "test-colour-encoder"

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        rows = []
        for image in images:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
            rows.append(rgb.mean(axis=(0, 1)))
        return self._normalize(np.asarray(rows, dtype=np.float32))

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            lowered = text.lower()
            rows.append(
                [
                    float("red" in lowered),
                    float("green" in lowered),
                    float("blue" in lowered),
                ]
            )
        return self._normalize(np.asarray(rows, dtype=np.float32))

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        return values / np.maximum(norms, 1e-8)


def test_embedding_ranker_combines_masked_region_and_context() -> None:
    array = np.zeros((24, 32, 3), dtype=np.uint8)
    array[:, :16, 0] = 255
    array[:, 16:, 2] = 255
    image = Image.fromarray(array)
    red = np.zeros((24, 32), dtype=bool)
    red[:, :16] = True
    blue = ~red
    ranker = EmbeddingRegionLabelRanker(_ColourEncoder())

    rows = ranker.rank_regions_labels(
        image,
        [("left", red), ("right", blue)],
        [("red_part", "red physical part"), ("blue_part", "blue physical part")],
    )

    assert rows["left"]["red_part"]["rank"] == 1
    assert rows["right"]["blue_part"]["rank"] == 1
    assert rows["left"]["red_part"]["probability"] > 0.9


def test_embedding_ranker_rejects_invalid_masks() -> None:
    image = Image.new("RGB", (12, 10), "white")
    ranker = EmbeddingRegionLabelRanker(_ColourEncoder())

    with pytest.raises(ValueError, match="must match"):
        ranker.rank_regions_labels(
            image,
            [("bad", np.ones((2, 2), dtype=bool))],
            [("red", "red")],
        )


class _FixedRanker:
    def __init__(self, rows: dict[str, dict[str, dict[str, float | int | str]]]):
        self.rows = rows

    def rank_regions_labels(self, *args: object, **kwargs: object) -> object:
        return self.rows


def test_consensus_ranker_penalizes_unverified_top_label() -> None:
    labels = [("first", "first"), ("second", "second"), ("third", "third")]
    base = {
        "region": {
            "first": {
                "prompt": "first",
                "full_similarity": 0.3,
                "masked_similarity": 0.4,
                "combined_similarity": 0.4,
                "probability": 0.6,
                "rank": 1,
            },
            "second": {
                "prompt": "second",
                "full_similarity": 0.2,
                "masked_similarity": 0.3,
                "combined_similarity": 0.38,
                "probability": 0.3,
                "rank": 2,
            },
            "third": {
                "prompt": "third",
                "full_similarity": 0.1,
                "masked_similarity": 0.2,
                "combined_similarity": 0.2,
                "probability": 0.1,
                "rank": 3,
            },
        }
    }
    verifier = {
        "region": {
            "first": {**base["region"]["first"], "rank": 3, "probability": 0.1},
            "second": {
                **base["region"]["second"],
                "combined_similarity": 0.43,
                "rank": 1,
                "probability": 0.7,
            },
            "third": {**base["region"]["third"], "rank": 2, "probability": 0.2},
        }
    }
    ranker = ConsensusRegionLabelRanker(
        _FixedRanker(base),
        _FixedRanker(verifier),
        primary_weight=0.5,
        disagreement_rank=2,
    )
    image = Image.new("RGB", (8, 8), "white")
    mask = np.ones((8, 8), dtype=bool)

    rows = ranker.rank_regions_labels(image, [("region", mask)], labels)

    assert rows["region"]["second"]["rank"] == 1
    assert rows["region"]["second"]["primary_rank"] == 2
    assert rows["region"]["second"]["verifier_rank"] == 1
