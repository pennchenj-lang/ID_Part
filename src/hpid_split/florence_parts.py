from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from .foundation import CandidateGeneration
from .fusion import MaskCandidate, mask_iou
from .guided_prompts import GuidedPromptSpec


@dataclass(frozen=True)
class FlorencePartConfig:
    """Configuration for Florence-2 referring-expression part proposals."""

    model_name: str = "florence-community/Florence-2-base-ft"
    task: str = "<REFERRING_EXPRESSION_SEGMENTATION>"
    crop_padding_ratio: float = 0.08
    phrases_per_part: int = 3
    generation_batch_size: int = 4
    maximum_new_tokens: int = 512
    minimum_area_px: int = 12
    minimum_raw_root_containment: float = 0.18
    maximum_root_area_fraction: float = 0.90
    same_prompt_nms_iou: float = 0.82
    source_reliability: float = 0.79
    local_files_only: bool = False


def _root_key(candidate: MaskCandidate) -> str:
    return (
        f"{candidate.metadata.get('root_origin', 'legacy')}::"
        f"{candidate.metadata.get('root_index', 'unknown')}"
    )


def _mask_box(
    mask: np.ndarray,
    image_size: tuple[int, int],
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0, 0, 0, 0
    width, height = image_size
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
    padding = max(3, round(max(x1 - x0, y1 - y0) * padding_ratio))
    return (
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(width, x1 + padding),
        min(height, y1 + padding),
    )


def _polygon_groups(payload: object) -> list[list[list[float]]]:
    """Normalize Florence polygon nesting into label-level polygon groups."""

    if not isinstance(payload, dict):
        return []
    raw_groups = payload.get("polygons", [])
    if not isinstance(raw_groups, (list, tuple)):
        return []
    groups: list[list[list[float]]] = []
    for raw_group in raw_groups:
        if not isinstance(raw_group, (list, tuple)) or not raw_group:
            continue
        if all(isinstance(value, (int, float)) for value in raw_group):
            groups.append([[float(value) for value in raw_group]])
            continue
        polygons: list[list[float]] = []
        for raw_polygon in raw_group:
            if not isinstance(raw_polygon, (list, tuple)):
                continue
            if len(raw_polygon) < 6 or len(raw_polygon) % 2:
                continue
            if not all(isinstance(value, (int, float)) for value in raw_polygon):
                continue
            polygons.append([float(value) for value in raw_polygon])
        if polygons:
            groups.append(polygons)
    return groups


def _rasterize_polygons(
    polygons: Sequence[Sequence[float]],
    image_shape: tuple[int, int],
) -> np.ndarray:
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    contours: list[np.ndarray] = []
    for polygon in polygons:
        if len(polygon) < 6 or len(polygon) % 2:
            continue
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        points[:, 0] = np.clip(points[:, 0], 0, max(0, width - 1))
        points[:, 1] = np.clip(points[:, 1], 0, max(0, height - 1))
        contour = np.round(points).astype(np.int32)
        if len(np.unique(contour, axis=0)) >= 3:
            contours.append(contour)
    if contours:
        cv2.fillPoly(mask, contours, 1)
    return mask.astype(bool)


def _split_instances(
    mask: np.ndarray,
    *,
    maximum_instances: int,
    minimum_area_px: int,
) -> list[np.ndarray]:
    if maximum_instances <= 1:
        return [mask] if int(np.count_nonzero(mask)) >= minimum_area_px else []
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    components = [
        (int(stats[index, cv2.CC_STAT_AREA]), labels == index)
        for index in range(1, count)
        if int(stats[index, cv2.CC_STAT_AREA]) >= minimum_area_px
    ]
    components.sort(key=lambda item: item[0], reverse=True)
    return [component for _, component in components[:maximum_instances]]


class FlorencePartGenerator:
    """Generate root-constrained part masks from referring expressions.

    Florence supplies semantic polygon proposals. The masks remain hypotheses:
    the prototype and geometry gates downstream decide whether they become HPIDs.
    """

    def __init__(
        self,
        *,
        device: str,
        config: FlorencePartConfig | None = None,
        processor: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self.device = device
        self.config = config or FlorencePartConfig()
        if (processor is None) != (model is None):
            raise ValueError("Florence processor and model must be supplied together")
        if processor is None:
            try:
                from transformers import (
                    AutoProcessor,
                    Florence2ForConditionalGeneration,
                )
            except ImportError as error:
                raise RuntimeError(
                    "Florence part proposals require the foundation extra: "
                    "pip install -e '.[foundation]'"
                ) from error
            processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                local_files_only=self.config.local_files_only,
            )
            dtype = torch.float16 if device.startswith("cuda") else torch.float32
            model = Florence2ForConditionalGeneration.from_pretrained(
                self.config.model_name,
                local_files_only=self.config.local_files_only,
                dtype=dtype,
            ).to(device)
        self.processor = processor
        self.model = model
        self.model.eval()

    def release(self) -> None:
        if self.device.startswith("cuda"):
            self.model.to("cpu")
            torch.cuda.empty_cache()

    def activate(self) -> None:
        self.model.to(self.device)
        self.model.eval()

    def _model_inputs(
        self,
        crop: Image.Image,
        phrases: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        prompts = [f"{self.config.task}{phrase}" for phrase in phrases]
        values = self.processor(
            text=prompts,
            images=[crop] * len(prompts),
            return_tensors="pt",
            padding=True,
        )
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        return {
            key: (
                value.to(self.device, dtype=dtype)
                if key == "pixel_values"
                else value.to(self.device)
            )
            for key, value in values.items()
        }

    def _generate_batch(
        self,
        crop: Image.Image,
        phrases: Sequence[str],
    ) -> list[list[list[list[float]]]]:
        inputs = self._model_inputs(crop, phrases)
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.config.maximum_new_tokens,
                num_beams=1,
                do_sample=False,
            )
        decoded = self.processor.batch_decode(
            generated,
            skip_special_tokens=False,
        )
        outputs: list[list[list[list[float]]]] = []
        for value in decoded:
            parsed = self.processor.post_process_generation(
                value,
                task=self.config.task,
                image_size=crop.size,
            )
            task_payload = (
                parsed.get(self.config.task, {}) if isinstance(parsed, dict) else {}
            )
            outputs.append(_polygon_groups(task_payload))
        return outputs

    def generate(
        self,
        image: Image.Image,
        roots: Sequence[MaskCandidate],
        prompts: tuple[GuidedPromptSpec, ...],
    ) -> CandidateGeneration:
        image = image.convert("RGB")
        candidates: list[MaskCandidate] = []
        prompt_rows: list[dict[str, object]] = []
        rejected = {
            "empty_polygon": 0,
            "outside_root": 0,
            "too_large": 0,
            "too_small": 0,
            "duplicate": 0,
        }
        phrase_query_count = 0
        for root in roots:
            root_mask = root.mask.astype(bool)
            root_area = int(np.count_nonzero(root_mask))
            if root_area < self.config.minimum_area_px:
                continue
            x0, y0, x1, y1 = _mask_box(
                root_mask,
                image.size,
                self.config.crop_padding_ratio,
            )
            if x1 <= x0 or y1 <= y0:
                continue
            crop = image.crop((x0, y0, x1, y1))
            local_root = root_mask[y0:y1, x0:x1]
            crop_array = np.asarray(crop, dtype=np.uint8).copy()
            crop_array[~local_root] = 127
            crop = Image.fromarray(crop_array, mode="RGB")
            root_candidate_key = str(root.metadata.get("candidate_key", "root"))

            queries: list[tuple[GuidedPromptSpec, str]] = []
            for spec in prompts:
                for phrase in spec.phrases[: self.config.phrases_per_part]:
                    queries.append((spec, phrase))
            phrase_query_count += len(queries)
            raw_by_slug: dict[str, list[tuple[np.ndarray, str, float]]] = {}
            for start in range(0, len(queries), self.config.generation_batch_size):
                batch = queries[start : start + self.config.generation_batch_size]
                polygon_outputs = self._generate_batch(
                    crop,
                    [phrase for _, phrase in batch],
                )
                for (spec, phrase), groups in zip(batch, polygon_outputs, strict=True):
                    if not groups:
                        rejected["empty_polygon"] += 1
                        continue
                    for polygons in groups:
                        local_raw = _rasterize_polygons(
                            polygons,
                            (crop.height, crop.width),
                        )
                        raw_area = int(np.count_nonzero(local_raw))
                        if raw_area < self.config.minimum_area_px:
                            rejected["too_small"] += 1
                            continue
                        local_mask = local_raw & local_root
                        area = int(np.count_nonzero(local_mask))
                        containment = area / max(1, raw_area)
                        if (
                            area < self.config.minimum_area_px
                            or containment < self.config.minimum_raw_root_containment
                        ):
                            rejected["outside_root"] += 1
                            continue
                        root_fraction = area / max(1, root_area)
                        if root_fraction > self.config.maximum_root_area_fraction:
                            rejected["too_large"] += 1
                            continue
                        full_mask = np.zeros(root_mask.shape, dtype=bool)
                        full_mask[y0:y1, x0:x1] = local_mask
                        proposal_score = float(
                            np.clip(
                                0.58 + 0.22 * containment + 0.20 * (1 - root_fraction),
                                0.0,
                                1.0,
                            )
                        )
                        raw_by_slug.setdefault(spec.slug, []).append(
                            (full_mask, phrase, proposal_score)
                        )

            for spec in prompts:
                proposed = sorted(
                    raw_by_slug.get(spec.slug, []),
                    key=lambda item: item[2],
                    reverse=True,
                )
                kept: list[tuple[np.ndarray, str, float]] = []
                for raw_mask, phrase, score in proposed:
                    for instance_mask in _split_instances(
                        raw_mask,
                        maximum_instances=spec.maximum_instances,
                        minimum_area_px=self.config.minimum_area_px,
                    ):
                        if any(
                            mask_iou(instance_mask, existing[0])
                            >= self.config.same_prompt_nms_iou
                            for existing in kept
                        ):
                            rejected["duplicate"] += 1
                            continue
                        kept.append((instance_mask, phrase, score))
                        if len(kept) >= spec.maximum_instances:
                            break
                    if len(kept) >= spec.maximum_instances:
                        break
                semantic_name = f"{root.semantic_name}_guided_{spec.slug}"
                for ordinal, (mask, phrase, score) in enumerate(kept, start=1):
                    candidates.append(
                        MaskCandidate(
                            semantic_name=semantic_name,
                            semantic_parent=root.semantic_name,
                            mask=mask,
                            score=score,
                            source=(
                                f"florence2[{self.config.model_name}]/referring-part"
                            ),
                            prompt=phrase,
                            source_reliability=self.config.source_reliability,
                            metadata={
                                "source_family": (
                                    f"florence2[{self.config.model_name}]"
                                ),
                                "root_origin": root.metadata.get("root_origin"),
                                "root_index": root.metadata.get("root_index"),
                                "candidate_key": (
                                    f"{root_candidate_key}/{semantic_name}:"
                                    f"{ordinal:02d}"
                                ),
                                "parent_candidate_key": root_candidate_key,
                                "assembly_parent_semantic": root.semantic_name,
                                "assembly_parent_candidate_key": root_candidate_key,
                                "guided_prompt": True,
                                "guided_prompt_slug": spec.slug,
                                "guided_prompt_label": spec.label,
                                "guided_prompt_phrases": list(spec.phrases),
                                "guided_backend": "florence2-referring-segmentation",
                                "root_clipped": True,
                                "ground_truth_used": False,
                            },
                        )
                    )
                prompt_rows.append(
                    {
                        "root_key": _root_key(root),
                        "label": spec.label,
                        "slug": spec.slug,
                        "phrases": list(spec.phrases),
                        "proposal_count": len(proposed),
                        "accepted_candidate_count": len(kept),
                    }
                )
        return CandidateGeneration(
            tuple(candidates),
            {
                "algorithm": "florence2-root-constrained-part-proposals-v1",
                "model": self.config.model_name,
                "root_count": len(roots),
                "prompt_count": len(prompts),
                "phrase_query_count": phrase_query_count,
                "candidate_count": len(candidates),
                "rejections": rejected,
                "prompts": prompt_rows,
                "root_background_suppressed": True,
                "ground_truth_used": False,
            },
        )
