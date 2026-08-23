from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_fill_holes

from .prompt_bank import PartPrompt


@dataclass(frozen=True)
class DenseRegion:
    semantic_name: str
    prompt: str
    box_xyxy: tuple[int, int, int, int]
    score: float
    peak_contrast: float


@dataclass(frozen=True)
class DenseMaskQuery:
    """One train-derived semantic query for direct conditional masks."""

    semantic_name: str
    prompts: tuple[str, ...]
    maximum_instances: int
    geometry_mean: tuple[float, float, float, float, float]
    geometry_std: tuple[float, float, float, float, float]
    geometry_samples: tuple[tuple[float, float, float, float, float], ...]


@dataclass(frozen=True)
class DenseMaskRegion:
    semantic_name: str
    prompt: str
    mask: np.ndarray
    score: float
    peak_probability: float
    mean_probability: float
    geometry_compatibility: float
    component_index: int


@dataclass(frozen=True)
class DenseProposalDiagnostics:
    queried_semantics: tuple[str, ...]
    proposed_region_count: int
    rejected_low_contrast_count: int


@dataclass(frozen=True)
class DenseMaskDiagnostics:
    queried_semantics: tuple[str, ...]
    threshold: float
    proposed_region_count: int
    rejected_empty_count: int
    rejected_geometry_count: int


def _mask_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _geometry_descriptor(mask: np.ndarray, root: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = _mask_box(root)
    px0, py0, px1, py1 = _mask_box(mask)
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    return np.asarray(
        [
            ((px0 + px1) / 2 - x0) / width,
            ((py0 + py1) / 2 - y0) / height,
            (px1 - px0) / width,
            (py1 - py0) / height,
            int(np.count_nonzero(mask)) / max(1, int(np.count_nonzero(root))),
        ],
        dtype=np.float32,
    )


def _geometry_compatibility(
    descriptor: np.ndarray,
    query: DenseMaskQuery,
) -> float:
    samples = np.asarray(query.geometry_samples, dtype=np.float32)
    if samples.size == 0:
        samples = np.asarray([query.geometry_mean], dtype=np.float32)
    # Some public part annotations mix a compact component with a union mask.
    # When the area modes are clearly separated, retain the majority mode and
    # choose the smaller mode on a tie. This is train-side robust estimation,
    # not an image/category exception at inference time.
    if len(samples) >= 4:
        areas = np.clip(samples[:, 4], 1e-6, None)
        ordering = np.argsort(areas)
        ordered = areas[ordering]
        log_gaps = np.diff(np.log(ordered))
        split = int(np.argmax(log_gaps)) + 1
        if float(log_gaps[split - 1]) >= np.log(3.0):
            low = ordering[:split]
            high = ordering[split:]
            samples = samples[
                low if len(low) >= len(high) else high
            ]
    scales = np.maximum(
        np.maximum(samples.std(axis=0), np.asarray(query.geometry_std) * 0.50),
        np.asarray((0.10, 0.10, 0.14, 0.14, 0.025), dtype=np.float32),
    )

    def score(value: np.ndarray) -> float:
        squared = ((samples - value[None, :]) / scales) ** 2
        center_distance = np.sqrt(np.mean(squared[:, :2], axis=1))
        eligible = center_distance <= 1.85
        if not np.any(eligible):
            return 0.0
        eligible_squared = squared[eligible]
        robust_distance = 0.65 * eligible_squared.mean(axis=1) + 0.35 * (
            eligible_squared.max(axis=1)
        )
        return float(np.exp(-0.5 * np.min(robust_distance)))

    mirrored = descriptor.copy()
    mirrored[0] = 1.0 - mirrored[0]
    return max(score(descriptor), score(mirrored))


def select_direct_mask_components(
    probability: np.ndarray,
    root: np.ndarray,
    query: DenseMaskQuery,
    *,
    threshold: float,
    minimum_geometry_compatibility: float,
) -> tuple[list[tuple[np.ndarray, float]], int]:
    """Clean a conditional heatmap and retain plausible physical instances."""

    if probability.shape != root.shape:
        raise ValueError("probability and root must have the same shape")
    active = (np.asarray(probability, dtype=np.float32) >= threshold) & root
    if not active.any():
        return [], 0
    active = cv2.morphologyEx(
        active.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), dtype=np.uint8),
    )
    count, components, stats, _ = cv2.connectedComponentsWithStats(active, 8)
    minimum_area = max(6, round(float(np.count_nonzero(root)) * 0.0005))
    ranked = sorted(
        (
            (int(stats[index, cv2.CC_STAT_AREA]), index)
            for index in range(1, count)
            if int(stats[index, cv2.CC_STAT_AREA]) >= minimum_area
        ),
        reverse=True,
    )
    accepted: list[tuple[np.ndarray, float]] = []
    geometry_rejections = 0
    for _, component_id in ranked:
        component = (components == component_id) & root
        compatibility = _geometry_compatibility(
            _geometry_descriptor(component, root), query
        )
        if compatibility < minimum_geometry_compatibility:
            geometry_rejections += 1
            continue
        accepted.append((component, compatibility))
        if len(accepted) >= max(1, query.maximum_instances):
            break
    return accepted, geometry_rejections


def parent_envelope(mask: np.ndarray, dilation_ratio: float) -> np.ndarray:
    """Approximate an amodal parent support region from its visible mask."""
    if mask.ndim != 2:
        raise ValueError("parent masks must be two-dimensional")
    if not mask.any():
        return np.zeros(mask.shape, dtype=bool)
    radius = max(1, round(min(mask.shape) * dilation_ratio))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1),
    )
    expanded = cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
    return binary_fill_holes(expanded).astype(bool)


def select_dense_regions(
    probability: np.ndarray,
    allowed: np.ndarray,
    *,
    maximum: int,
    minimum_peak_probability: float,
    minimum_peak_contrast: float,
    activation_quantile: float,
    peak_ratio: float,
    box_padding_ratio: float,
) -> list[tuple[tuple[int, int, int, int], float, float]]:
    """Convert a relative dense heatmap into separated region hypotheses."""
    if probability.shape != allowed.shape:
        raise ValueError("probability and allowed masks must have the same shape")
    if maximum < 1 or not allowed.any():
        return []
    smoothed = cv2.GaussianBlur(probability.astype(np.float32), (0, 0), 2.0)
    values = smoothed[allowed]
    peak = float(values.max())
    median = float(np.median(values))
    contrast = peak - median
    if peak < minimum_peak_probability or contrast < minimum_peak_contrast:
        return []
    threshold = max(
        float(np.quantile(values, activation_quantile)),
        median + minimum_peak_contrast,
        peak * peak_ratio,
    )
    active = ((smoothed >= threshold) & allowed).astype(np.uint8)
    active = cv2.morphologyEx(active, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, components, stats, _ = cv2.connectedComponentsWithStats(active, 8)
    height, width = probability.shape
    regions: list[tuple[tuple[int, int, int, int], float, float]] = []
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < 8:
            continue
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        w = int(stats[component_id, cv2.CC_STAT_WIDTH])
        h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        pad = max(3, round(max(w, h) * box_padding_ratio))
        component = components == component_id
        box = (
            max(0, x - pad),
            max(0, y - pad),
            min(width, x + w + pad),
            min(height, y + h + pad),
        )
        regions.append((box, float(smoothed[component].max()), contrast))
    regions.sort(key=lambda item: item[1], reverse=True)
    return regions[:maximum]


class DenseSemanticProposer:
    """CLIPSeg heatmaps converted into auditable, class-labelled region boxes."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str,
        local_files_only: bool,
    ) -> None:
        try:
            from transformers import AutoProcessor, CLIPSegForImageSegmentation
        except ImportError as error:
            raise RuntimeError(
                "Install the foundation extra: pip install 'hpid-split[foundation]'"
            ) from error
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        )
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            model_name,
            local_files_only=local_files_only,
        ).to(device)
        self.model.eval()

    def probability_maps(self, image: Image.Image, prompts: list[str]) -> np.ndarray:
        """Return one image-sized CLIPSeg probability map per text prompt."""

        if not prompts:
            return np.zeros((0, image.height, image.width), dtype=np.float32)
        inputs = self.processor(
            text=prompts,
            images=[image] * len(prompts),
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with (
            torch.inference_mode(),
            torch.amp.autocast("cuda", enabled=self.device.startswith("cuda")),
        ):
            logits = self.model(**inputs).logits[:, None]
            logits = torch.nn.functional.interpolate(
                logits,
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            return logits.sigmoid().float().cpu().numpy()

    @property
    def conditional_mask_threshold(self) -> float | None:
        metadata = getattr(self.model.config, "hpid_conditional_parts", None)
        if not isinstance(metadata, dict):
            return None
        threshold = float(metadata.get("calibrated_threshold", 0.0))
        return threshold if 0.0 < threshold < 1.0 else None

    def propose_direct_masks(
        self,
        image: Image.Image,
        root: np.ndarray,
        queries: list[DenseMaskQuery],
        *,
        phrases_per_query: int = 2,
        batch_size: int = 8,
        minimum_geometry_compatibility: float = 0.16,
    ) -> tuple[list[DenseMaskRegion], DenseMaskDiagnostics]:
        """Generate calibrated masks from a train-only conditional checkpoint.

        The learned mask is semantic evidence, not final ownership. Geometry,
        structure, appearance, and the public fusion stage still decide whether
        it receives a persistent Part-ID.
        """

        threshold = self.conditional_mask_threshold
        if threshold is None:
            raise ValueError(
                "direct conditional masks require a checkpoint with a calibrated "
                "hpid_conditional_parts threshold"
            )
        if root.shape != (image.height, image.width):
            raise ValueError("root mask must match the input image")
        if phrases_per_query < 1 or batch_size < 1:
            raise ValueError("phrases_per_query and batch_size must be positive")
        prompt_rows: list[tuple[int, str]] = []
        for query_index, query in enumerate(queries):
            for prompt in tuple(dict.fromkeys(query.prompts))[:phrases_per_query]:
                prompt_rows.append((query_index, prompt))
        probability_rows = [
            self.probability_maps(
                image,
                [prompt for _, prompt in prompt_rows[start : start + batch_size]],
            )
            for start in range(0, len(prompt_rows), batch_size)
        ]
        probabilities = (
            np.concatenate(probability_rows, axis=0)
            if probability_rows
            else np.zeros((0, image.height, image.width), dtype=np.float32)
        )
        grouped: dict[int, list[tuple[str, np.ndarray]]] = {}
        for (query_index, prompt), probability in zip(
            prompt_rows, probabilities, strict=True
        ):
            grouped.setdefault(query_index, []).append((prompt, probability))

        regions: list[DenseMaskRegion] = []
        empty_rejections = 0
        geometry_rejections = 0
        for query_index, query in enumerate(queries):
            rows = grouped.get(query_index, [])
            if not rows:
                empty_rejections += 1
                continue
            probability = np.maximum.reduce([row[1] for row in rows]).astype(
                np.float32
            )
            probability[~root] = 0.0
            components, rejected = select_direct_mask_components(
                probability,
                root,
                query,
                threshold=threshold,
                minimum_geometry_compatibility=minimum_geometry_compatibility,
            )
            geometry_rejections += rejected
            if not components:
                empty_rejections += 1
                continue
            prompt = max(
                rows,
                key=lambda row: float(np.mean(row[1][root])) if root.any() else 0.0,
            )[0]
            for component_index, (component, compatibility) in enumerate(
                components, start=1
            ):
                values = probability[component]
                peak = float(values.max())
                mean = float(values.mean())
                score = float(
                    np.clip(
                        0.55 * mean + 0.25 * peak + 0.20 * compatibility,
                        0.0,
                        1.0,
                    )
                )
                regions.append(
                    DenseMaskRegion(
                        semantic_name=query.semantic_name,
                        prompt=prompt,
                        mask=component,
                        score=score,
                        peak_probability=peak,
                        mean_probability=mean,
                        geometry_compatibility=compatibility,
                        component_index=component_index,
                    )
                )
        return regions, DenseMaskDiagnostics(
            queried_semantics=tuple(query.semantic_name for query in queries),
            threshold=threshold,
            proposed_region_count=len(regions),
            rejected_empty_count=empty_rejections,
            rejected_geometry_count=geometry_rejections,
        )

    def score_regions(
        self,
        image: Image.Image,
        queries: list[tuple[str, str, np.ndarray]],
        *,
        top_fraction: float = 0.05,
    ) -> dict[str, dict[str, float | str]]:
        """Score text evidence inside candidate roots without deriving masks."""

        if not 0.0 < top_fraction <= 1.0:
            raise ValueError("top_fraction must be in (0, 1]")
        if not queries:
            return {}
        for _, _, allowed in queries:
            if allowed.shape != (image.height, image.width):
                raise ValueError("root score masks must match the image shape")
        probabilities = self.probability_maps(
            image, [prompt for _, prompt, _ in queries]
        )
        scores: dict[str, dict[str, float | str]] = {}
        for (key, prompt, allowed), probability in zip(
            queries, probabilities, strict=True
        ):
            values = probability[allowed]
            if not len(values):
                scores[key] = {
                    "prompt": prompt,
                    "top_mean": 0.0,
                    "median": 0.0,
                    "contrast": 0.0,
                }
                continue
            keep = max(1, round(len(values) * top_fraction))
            split = max(0, len(values) - keep)
            top_values = np.partition(values, split)[split:]
            median = float(np.median(values))
            top_mean = float(np.mean(top_values))
            scores[key] = {
                "prompt": prompt,
                "top_mean": top_mean,
                "median": median,
                "contrast": max(0.0, top_mean - median),
            }
        return scores

    def rank_region_labels(
        self,
        image: Image.Image,
        allowed: np.ndarray,
        labels: list[tuple[str, str]],
        *,
        masked_weight: float = 0.35,
        temperature: float = 0.035,
    ) -> dict[str, dict[str, float | str | int]]:
        """Rank category labels with the frozen CLIPSeg image/text encoders."""

        ranked = self.rank_regions_labels(
            image,
            [("region", allowed)],
            labels,
            masked_weight=masked_weight,
            temperature=temperature,
        )
        return ranked.get("region", {})

    @staticmethod
    def _masked_region_view(image: Image.Image, allowed: np.ndarray) -> Image.Image:
        ys, xs = np.nonzero(allowed)
        x0, y0 = int(xs.min()), int(ys.min())
        x1, y1 = int(xs.max() + 1), int(ys.max() + 1)
        pad_x = max(2, round((x1 - x0) * 0.10))
        pad_y = max(2, round((y1 - y0) * 0.10))
        x0, y0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
        x1, y1 = min(image.width, x1 + pad_x), min(image.height, y1 + pad_y)
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)[y0:y1, x0:x1]
        local_mask = allowed[y0:y1, x0:x1]
        composited = np.where(local_mask[..., None], rgb, 127)
        height, width = composited.shape[:2]
        side = max(height, width)
        square = np.full((side, side, 3), 127, dtype=np.uint8)
        offset_y = (side - height) // 2
        offset_x = (side - width) // 2
        square[offset_y : offset_y + height, offset_x : offset_x + width] = composited
        return Image.fromarray(square, mode="RGB")

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
        """Rank one shared label inventory for several isolated image regions."""

        if not regions or not labels:
            return {}
        keys = [key for key, _ in regions]
        if len(keys) != len(set(keys)):
            raise ValueError("region classification keys must be unique")
        for _, allowed in regions:
            if allowed.shape != (image.height, image.width):
                raise ValueError("classification mask must match the image shape")
            if not np.any(allowed):
                raise ValueError("classification regions must not be empty")
        if not 0.0 <= masked_weight <= 1.0:
            raise ValueError("masked_weight must be in [0, 1]")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if image_batch_size < 1:
            raise ValueError("image_batch_size must be positive")

        text_inputs = self.processor(
            text=[f"a photo of {prompt}" for _, prompt in labels],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with (
            torch.inference_mode(),
            torch.amp.autocast("cuda", enabled=self.device.startswith("cuda")),
        ):
            text = self.model.clip.text_model(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs.get("attention_mask"),
            )
            text_features = self.model.clip.text_projection(text.pooler_output)
            text_features = torch.nn.functional.normalize(text_features.float(), dim=1)
            full_pixels = self.processor(
                images=[image.convert("RGB")], return_tensors="pt"
            )["pixel_values"].to(self.device)
            full_vision = self.model.clip.vision_model(pixel_values=full_pixels)
            full_features = self.model.clip.visual_projection(
                full_vision.pooler_output
            )
            full_features = torch.nn.functional.normalize(
                full_features.float(), dim=1
            )
            full_similarities = (full_features @ text_features.T).cpu().numpy()[0]

            masked_views = [
                self._masked_region_view(image, allowed) for _, allowed in regions
            ]
            masked_rows: list[np.ndarray] = []
            for start in range(0, len(masked_views), image_batch_size):
                pixels = self.processor(
                    images=masked_views[start : start + image_batch_size],
                    return_tensors="pt",
                )["pixel_values"].to(self.device)
                vision = self.model.clip.vision_model(pixel_values=pixels)
                features = self.model.clip.visual_projection(vision.pooler_output)
                features = torch.nn.functional.normalize(features.float(), dim=1)
                masked_rows.append((features @ text_features.T).cpu().numpy())
        masked_similarities = np.concatenate(masked_rows, axis=0)
        output: dict[str, dict[str, dict[str, float | str | int]]] = {}
        for region_index, (region_key, _) in enumerate(regions):
            combined = (
                (1.0 - masked_weight) * full_similarities
                + masked_weight * masked_similarities[region_index]
            )
            shifted = (combined - float(combined.max())) / temperature
            probabilities = np.exp(shifted)
            probabilities /= max(1e-8, float(probabilities.sum()))
            ordering = np.argsort(-combined)
            rank_by_index = {
                int(index): rank for rank, index in enumerate(ordering, start=1)
            }
            output[region_key] = {
                key: {
                    "prompt": prompt,
                    "full_similarity": float(full_similarities[label_index]),
                    "masked_similarity": float(
                        masked_similarities[region_index, label_index]
                    ),
                    "combined_similarity": float(combined[label_index]),
                    "probability": float(probabilities[label_index]),
                    "rank": rank_by_index[label_index],
                }
                for label_index, (key, prompt) in enumerate(labels)
            }
        return output

    def propose(
        self,
        image: Image.Image,
        parts: list[PartPrompt],
        allowed: np.ndarray,
        maximum_by_semantic: dict[str, int],
        *,
        minimum_peak_probability: float,
        minimum_peak_contrast: float,
        activation_quantile: float,
        peak_ratio: float,
        box_padding_ratio: float,
    ) -> tuple[list[DenseRegion], DenseProposalDiagnostics]:
        entries: list[tuple[PartPrompt, str]] = []
        for part in parts:
            for prompt in part.dense_phrases:
                entries.append((part, prompt))
        if not entries:
            return [], DenseProposalDiagnostics((), 0, 0)
        probabilities = self.probability_maps(image, [prompt for _, prompt in entries])

        heatmaps: dict[str, np.ndarray] = {}
        prompts_by_semantic: dict[str, str] = {}
        for (part, prompt), probability in zip(entries, probabilities, strict=True):
            previous = heatmaps.get(part.semantic_name)
            if previous is None:
                heatmaps[part.semantic_name] = probability
                prompts_by_semantic[part.semantic_name] = prompt
                continue
            stronger = probability > previous
            if int(np.count_nonzero(stronger & allowed)) > int(
                np.count_nonzero((~stronger) & allowed)
            ):
                prompts_by_semantic[part.semantic_name] = prompt
            heatmaps[part.semantic_name] = np.maximum(previous, probability)

        proposals: list[DenseRegion] = []
        rejected = 0
        for part in parts:
            maximum = maximum_by_semantic.get(part.semantic_name, 0)
            if maximum <= 0:
                continue
            regions = select_dense_regions(
                heatmaps[part.semantic_name],
                allowed,
                maximum=maximum,
                minimum_peak_probability=minimum_peak_probability,
                minimum_peak_contrast=minimum_peak_contrast,
                activation_quantile=activation_quantile,
                peak_ratio=peak_ratio,
                box_padding_ratio=box_padding_ratio,
            )
            if not regions:
                rejected += 1
            proposals.extend(
                DenseRegion(
                    semantic_name=part.semantic_name,
                    prompt=prompts_by_semantic[part.semantic_name],
                    box_xyxy=box,
                    score=score,
                    peak_contrast=contrast,
                )
                for box, score, contrast in regions
            )
        proposals.sort(key=lambda item: item.score, reverse=True)
        return proposals, DenseProposalDiagnostics(
            queried_semantics=tuple(sorted(heatmaps)),
            proposed_region_count=len(proposals),
            rejected_low_contrast_count=rejected,
        )
