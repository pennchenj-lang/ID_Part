from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
from PIL import Image

from .asset_routing import ImageTextEncoder, masked_asset_view


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def _highlighted_region_view(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Keep object context visible while making the queried region explicit."""

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    muted = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    muted = np.repeat(muted[..., None], 3, axis=2)
    composited = np.rint(0.55 * muted + 0.45 * 127.0).astype(np.uint8)
    composited[mask] = rgb[mask]
    boundary = cv2.morphologyEx(
        mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
    ).astype(bool)
    composited[boundary] = np.array([255, 48, 48], dtype=np.uint8)
    return Image.fromarray(composited, mode="RGB")


class EmbeddingRegionLabelRanker:
    """Rank SAM regions against a closed part inventory with image-text embeddings.

    The masked crop answers what the region looks like, while the highlighted
    full image preserves where it is attached. The original image contributes
    only a small context term, so unrelated scene content cannot dominate a
    small physical component.
    """

    def __init__(
        self,
        encoder: ImageTextEncoder,
        *,
        crop_weight: float = 0.72,
        highlight_weight: float = 0.22,
    ) -> None:
        if crop_weight < 0.0 or highlight_weight < 0.0:
            raise ValueError("region view weights must be non-negative")
        if crop_weight + highlight_weight > 1.0:
            raise ValueError("region view weights must sum to at most one")
        self.encoder = encoder
        self.crop_weight = crop_weight
        self.highlight_weight = highlight_weight

    @staticmethod
    def _text_embeddings(
        encoder: ImageTextEncoder,
        labels: Sequence[tuple[str, str]],
    ) -> np.ndarray:
        prompts: list[str] = []
        for _, prompt in labels:
            cleaned = " ".join(prompt.split())
            prompts.extend((cleaned, f"a photo of {cleaned}"))
        encoded = encoder.encode_texts(prompts)
        paired = encoded.reshape(len(labels), 2, -1).mean(axis=1)
        return _normalize_rows(paired)

    def rank_regions_labels(
        self,
        image: Image.Image,
        regions: list[tuple[str, np.ndarray]],
        labels: list[tuple[str, str]],
        *,
        masked_weight: float = 0.82,
        temperature: float = 0.035,
        image_batch_size: int = 8,
    ) -> dict[str, dict[str, dict[str, float | str | int]]]:
        if not regions or not labels:
            return {}
        keys = [key for key, _ in regions]
        if len(keys) != len(set(keys)):
            raise ValueError("region classification keys must be unique")
        for _, mask in regions:
            if mask.shape != (image.height, image.width):
                raise ValueError("classification mask must match the image shape")
            if not np.any(mask):
                raise ValueError("classification regions must not be empty")
        if not 0.0 <= masked_weight <= 1.0:
            raise ValueError("masked_weight must be in [0, 1]")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if image_batch_size < 1:
            raise ValueError("image_batch_size must be positive")

        text_embeddings = self._text_embeddings(self.encoder, labels)
        full_embedding = self.encoder.encode_images([image.convert("RGB")])[0]
        crop_views = [
            masked_asset_view(image, mask, padding_ratio=0.14)
            for _, mask in regions
        ]
        highlighted_views = [
            _highlighted_region_view(image, mask) for _, mask in regions
        ]
        crop_embeddings = self.encoder.encode_images(crop_views)
        highlighted_embeddings = self.encoder.encode_images(highlighted_views)

        full_similarities = full_embedding @ text_embeddings.T
        crop_similarities = crop_embeddings @ text_embeddings.T
        highlighted_similarities = highlighted_embeddings @ text_embeddings.T
        region_weight = self.crop_weight + self.highlight_weight
        region_similarities = (
            self.crop_weight * crop_similarities
            + self.highlight_weight * highlighted_similarities
        ) / max(region_weight, 1e-8)
        combined_similarities = (
            (1.0 - masked_weight) * full_similarities[None, :]
            + masked_weight * region_similarities
        )

        output: dict[str, dict[str, dict[str, float | str | int]]] = {}
        for region_index, (region_key, _) in enumerate(regions):
            combined = combined_similarities[region_index]
            shifted = (combined - float(combined.max())) / temperature
            probabilities = np.exp(np.clip(shifted, -40.0, 0.0))
            probabilities /= max(1e-8, float(probabilities.sum()))
            ordering = np.argsort(-combined)
            ranks = {
                int(label_index): rank
                for rank, label_index in enumerate(ordering, start=1)
            }
            output[region_key] = {
                key: {
                    "prompt": prompt,
                    "full_similarity": float(full_similarities[label_index]),
                    "masked_similarity": float(
                        crop_similarities[region_index, label_index]
                    ),
                    "highlighted_similarity": float(
                        highlighted_similarities[region_index, label_index]
                    ),
                    "combined_similarity": float(combined[label_index]),
                    "probability": float(probabilities[label_index]),
                    "rank": ranks[label_index],
                }
                for label_index, (key, prompt) in enumerate(labels)
            }
        return output


class ConsensusRegionLabelRanker:
    """Fuse a proposal ranker and an independent semantic verifier."""

    def __init__(
        self,
        primary: object,
        verifier: object,
        *,
        primary_weight: float = 0.62,
        disagreement_rank: int = 3,
        disagreement_penalty: float = 0.018,
    ) -> None:
        if not 0.0 <= primary_weight <= 1.0:
            raise ValueError("primary weight must be in [0, 1]")
        if disagreement_rank < 1:
            raise ValueError("disagreement rank must be positive")
        self.primary = primary
        self.verifier = verifier
        self.primary_weight = primary_weight
        self.disagreement_rank = disagreement_rank
        self.disagreement_penalty = disagreement_penalty

    def rank_regions_labels(
        self,
        image: Image.Image,
        regions: list[tuple[str, np.ndarray]],
        labels: list[tuple[str, str]],
        *,
        masked_weight: float = 0.82,
        temperature: float = 0.035,
        image_batch_size: int = 8,
    ) -> dict[str, dict[str, dict[str, float | str | int]]]:
        kwargs = {
            "masked_weight": masked_weight,
            "temperature": temperature,
            "image_batch_size": image_batch_size,
        }
        primary_rows = self.primary.rank_regions_labels(
            image, regions, labels, **kwargs
        )
        verifier_rows = self.verifier.rank_regions_labels(
            image, regions, labels, **kwargs
        )
        verifier_weight = 1.0 - self.primary_weight
        output: dict[str, dict[str, dict[str, float | str | int]]] = {}
        for region_key, _ in regions:
            primary = primary_rows[region_key]
            verifier = verifier_rows[region_key]
            fused_scores: dict[str, float] = {}
            for label_key, _ in labels:
                first = primary[label_key]
                second = verifier[label_key]
                score = (
                    self.primary_weight * float(first["combined_similarity"])
                    + verifier_weight * float(second["combined_similarity"])
                )
                first_rank = int(first["rank"])
                second_rank = int(second["rank"])
                if (
                    first_rank == 1
                    and second_rank > self.disagreement_rank
                    or second_rank == 1
                    and first_rank > self.disagreement_rank
                ):
                    score -= self.disagreement_penalty
                fused_scores[label_key] = score

            values = np.asarray(
                [fused_scores[label_key] for label_key, _ in labels],
                dtype=np.float32,
            )
            shifted = (values - float(values.max())) / temperature
            probabilities = np.exp(np.clip(shifted, -40.0, 0.0))
            probabilities /= max(1e-8, float(probabilities.sum()))
            ordering = np.argsort(-values)
            ranks = {
                int(label_index): rank
                for rank, label_index in enumerate(ordering, start=1)
            }
            output[region_key] = {}
            for label_index, (label_key, prompt) in enumerate(labels):
                first = primary[label_key]
                second = verifier[label_key]
                output[region_key][label_key] = {
                    "prompt": prompt,
                    "full_similarity": float(
                        self.primary_weight * float(first["full_similarity"])
                        + verifier_weight * float(second["full_similarity"])
                    ),
                    "masked_similarity": float(
                        self.primary_weight * float(first["masked_similarity"])
                        + verifier_weight * float(second["masked_similarity"])
                    ),
                    "combined_similarity": float(values[label_index]),
                    "probability": float(probabilities[label_index]),
                    "rank": ranks[label_index],
                    "primary_rank": int(first["rank"]),
                    "verifier_rank": int(second["rank"]),
                    "primary_probability": float(first["probability"]),
                    "verifier_probability": float(second["probability"]),
                }
        return output
