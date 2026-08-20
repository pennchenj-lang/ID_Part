from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image


@dataclass(frozen=True)
class SemanticRefinementConfig:
    model_name: str = "facebook/sam2.1-hiera-tiny"
    local_files_only: bool = False
    box_padding_ratio: float = 0.10
    ring_ratio: float = 0.08
    positive_point_count: int = 3
    negative_point_count: int = 4
    minimum_component_area: int = 6
    minimum_coarse_recall: float = 0.35
    maximum_area_expansion: float = 5.0
    sam_quality_weight: float = 0.34
    coarse_agreement_weight: float = 0.46
    semantic_support_weight: float = 0.20
    size_penalty_weight: float = 0.08


@dataclass(frozen=True)
class RefinedSemanticMask:
    semantic_name: str
    mask: np.ndarray
    probability: np.ndarray
    used_sam2: bool
    component_count: int
    diagnostics: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class _Prompt:
    semantic_name: str
    component_id: int
    coarse_mask: np.ndarray
    probability: np.ndarray
    box: tuple[float, float, float, float]
    points: tuple[tuple[float, float], ...]
    labels: tuple[int, ...]


def _mask_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _positive_points(mask: np.ndarray, maximum: int) -> list[tuple[float, float]]:
    remaining = mask.astype(np.uint8).copy()
    points: list[tuple[float, float]] = []
    for _ in range(maximum):
        if not remaining.any():
            break
        distance = cv2.distanceTransform(remaining, cv2.DIST_L2, 5)
        y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
        radius = max(2, round(float(distance[y, x]) * 1.6))
        points.append((float(x), float(y)))
        cv2.circle(remaining, (int(x), int(y)), radius, 0, -1)
    return points


def _nearest_points(
    mask: np.ndarray,
    targets: tuple[tuple[float, float], ...],
) -> list[tuple[float, float]]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return []
    coordinates = np.stack([xs, ys], axis=1).astype(np.float32)
    selected: list[tuple[float, float]] = []
    available = np.ones(len(coordinates), dtype=bool)
    for target_x, target_y in targets:
        if not available.any():
            break
        distances = (coordinates[:, 0] - target_x) ** 2 + (
            coordinates[:, 1] - target_y
        ) ** 2
        distances[~available] = np.inf
        index = int(np.argmin(distances))
        x, y = coordinates[index]
        selected.append((float(x), float(y)))
        radius = max(3.0, math.sqrt(float(mask.sum())) * 0.04)
        available &= (
            (coordinates[:, 0] - x) ** 2 + (coordinates[:, 1] - y) ** 2
        ) > radius**2
    return selected


def _component_prompts(
    semantic_name: str,
    coarse: np.ndarray,
    probability: np.ndarray,
    *,
    config: SemanticRefinementConfig,
) -> list[_Prompt]:
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        coarse.astype(np.uint8), 8
    )
    ranked = sorted(
        (
            (int(stats[index, cv2.CC_STAT_AREA]), index)
            for index in range(1, count)
            if int(stats[index, cv2.CC_STAT_AREA]) >= config.minimum_component_area
        ),
        reverse=True,
    )
    height, width = coarse.shape
    prompts: list[_Prompt] = []
    for component_id, (_, index) in enumerate(ranked):
        component = components == index
        x0, y0, x1, y1 = _mask_box(component)
        padding = max(3, round(max(x1 - x0, y1 - y0) * config.box_padding_ratio))
        x0, y0 = max(0, x0 - padding), max(0, y0 - padding)
        x1, y1 = min(width, x1 + padding), min(height, y1 + padding)
        positives = _positive_points(component, config.positive_point_count)
        radius = max(3, round(max(x1 - x0, y1 - y0) * config.ring_ratio))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        ring = cv2.dilate(component.astype(np.uint8), kernel).astype(bool) & ~component
        targets = (
            (float(x0), float(y0)),
            (float(x1 - 1), float(y0)),
            (float(x1 - 1), float(y1 - 1)),
            (float(x0), float(y1 - 1)),
        )
        negatives = _nearest_points(ring, targets)[: config.negative_point_count]
        points = [*positives, *negatives]
        labels = [1] * len(positives) + [0] * len(negatives)
        total = config.positive_point_count + config.negative_point_count
        while len(points) < total:
            points.append((0.0, 0.0))
            labels.append(-1)
        prompts.append(
            _Prompt(
                semantic_name=semantic_name,
                component_id=component_id,
                coarse_mask=component,
                probability=probability,
                box=(float(x0), float(y0), float(x1), float(y1)),
                points=tuple(points),
                labels=tuple(labels),
            )
        )
    return prompts


def _candidate_score(
    candidate: np.ndarray,
    prompt: _Prompt,
    root: np.ndarray,
    sam_quality: float,
    config: SemanticRefinementConfig,
) -> tuple[float, dict[str, float]]:
    coarse = prompt.coarse_mask
    intersection = int((candidate & coarse).sum())
    union = int((candidate | coarse).sum())
    agreement = intersection / max(1, union)
    coarse_recall = intersection / max(1, int(coarse.sum()))
    area_ratio = int(candidate.sum()) / max(1, int(coarse.sum()))
    inside = prompt.probability[candidate]
    outside = prompt.probability[root & ~candidate]
    support = float(inside.mean()) if len(inside) else 0.0
    background = float(outside.mean()) if len(outside) else 0.0
    contrast = max(0.0, support - background)
    size_penalty = min(2.0, abs(math.log(max(area_ratio, 1e-6))))
    valid = (
        coarse_recall >= config.minimum_coarse_recall
        and area_ratio <= config.maximum_area_expansion
    )
    score = (
        config.sam_quality_weight * sam_quality
        + config.coarse_agreement_weight * agreement
        + config.semantic_support_weight * contrast
        - config.size_penalty_weight * size_penalty
    )
    if not valid:
        score = -float("inf")
    return score, {
        "sam_quality": sam_quality,
        "coarse_iou": agreement,
        "coarse_recall": coarse_recall,
        "area_ratio": area_ratio,
        "semantic_contrast": contrast,
        "selection_score": score,
    }


class Sam2SemanticRefiner:
    """Use conditional masks as SAM2 prompts and retain auditable fallbacks."""

    def __init__(
        self,
        *,
        device: str = "cuda",
        config: SemanticRefinementConfig | None = None,
        processor: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self.device = device
        self.config = config or SemanticRefinementConfig()
        if (processor is None) != (model is None):
            raise ValueError("processor and model must be supplied together")
        if processor is None:
            try:
                from transformers import Sam2Model, Sam2Processor
            except ImportError as error:
                raise RuntimeError(
                    "Install the foundation extra before enabling SAM2 refinement"
                ) from error
            processor = Sam2Processor.from_pretrained(
                self.config.model_name,
                local_files_only=self.config.local_files_only,
            )
            model = Sam2Model.from_pretrained(
                self.config.model_name,
                local_files_only=self.config.local_files_only,
            ).to(device)
        self.processor = processor
        self.model = model
        self.model.eval()

    def refine(
        self,
        image: Image.Image,
        root: np.ndarray,
        masks: dict[str, np.ndarray],
        probabilities: dict[str, np.ndarray],
    ) -> dict[str, RefinedSemanticMask]:
        if root.shape != (image.height, image.width):
            raise ValueError("root mask must match the image")
        prompts: list[_Prompt] = []
        for semantic_name, coarse in sorted(masks.items()):
            probability = probabilities.get(semantic_name)
            if probability is None or probability.shape != root.shape:
                raise ValueError(
                    "every semantic mask requires an image-sized probability"
                )
            prompts.extend(
                _component_prompts(
                    semantic_name,
                    np.asarray(coarse, dtype=bool) & root,
                    np.asarray(probability, dtype=np.float32),
                    config=self.config,
                )
            )
        if not prompts:
            return {
                semantic_name: RefinedSemanticMask(
                    semantic_name,
                    np.asarray(mask, dtype=bool) & root,
                    probabilities[semantic_name],
                    False,
                    0,
                    (),
                )
                for semantic_name, mask in masks.items()
            }

        inputs = self.processor(
            images=image.convert("RGB"),
            input_points=[
                [[list(point) for point in prompt.points] for prompt in prompts]
            ],
            input_labels=[
                [[int(label) for label in prompt.labels] for prompt in prompts]
            ],
            input_boxes=[[list(prompt.box) for prompt in prompts]],
            return_tensors="pt",
        ).to(self.device)
        with (
            torch.inference_mode(),
            torch.amp.autocast("cuda", enabled=self.device.startswith("cuda")),
        ):
            outputs = self.model(**inputs)
        try:
            processed = self.processor.post_process_masks(
                outputs.pred_masks.detach().cpu(),
                inputs["original_sizes"].detach().cpu(),
                inputs["reshaped_input_sizes"].detach().cpu(),
                binarize=True,
            )[0]
        except (KeyError, TypeError):
            processed = self.processor.post_process_masks(
                outputs.pred_masks.detach().cpu(),
                inputs["original_sizes"].detach().cpu(),
                binarize=True,
            )[0]
        candidates = np.asarray(processed)
        if candidates.ndim == 3:
            candidates = candidates[:, None]
        if candidates.ndim != 4 or candidates.shape[0] != len(prompts):
            raise RuntimeError(f"unexpected SAM2 mask shape: {candidates.shape}")
        score_tensor = getattr(outputs, "iou_scores", None)
        if score_tensor is None:
            score_tensor = getattr(outputs, "pred_iou_scores", None)
        if score_tensor is None:
            qualities = np.full(candidates.shape[:2], 0.5, dtype=np.float32)
        else:
            qualities = score_tensor.detach().float().cpu().numpy()
            while qualities.ndim > 2:
                qualities = qualities[0]
            if qualities.ndim == 1:
                qualities = qualities[None]

        selected_by_semantic: dict[str, list[np.ndarray]] = {}
        diagnostics_by_semantic: dict[str, list[dict[str, object]]] = {}
        for prompt_index, prompt in enumerate(prompts):
            best_mask = prompt.coarse_mask
            best_score = -float("inf")
            best_diagnostics: dict[str, object] = {
                "component_id": prompt.component_id,
                "used_sam2": False,
                "fallback_reason": "no_valid_sam2_candidate",
            }
            for candidate_index in range(candidates.shape[1]):
                candidate = (
                    candidates[prompt_index, candidate_index].astype(bool) & root
                )
                score, row = _candidate_score(
                    candidate,
                    prompt,
                    root,
                    float(qualities[prompt_index, candidate_index]),
                    self.config,
                )
                if score > best_score:
                    best_score = score
                    best_mask = candidate
                    best_diagnostics = {
                        "component_id": prompt.component_id,
                        "candidate_index": candidate_index,
                        "used_sam2": True,
                        **row,
                    }
            selected_by_semantic.setdefault(prompt.semantic_name, []).append(best_mask)
            diagnostics_by_semantic.setdefault(prompt.semantic_name, []).append(
                best_diagnostics
            )

        output: dict[str, RefinedSemanticMask] = {}
        for semantic_name, coarse in masks.items():
            selected = selected_by_semantic.get(semantic_name, [])
            merged = np.zeros(root.shape, dtype=bool)
            for mask in selected:
                merged |= mask
            if not selected:
                merged = np.asarray(coarse, dtype=bool) & root
            rows = tuple(diagnostics_by_semantic.get(semantic_name, []))
            output[semantic_name] = RefinedSemanticMask(
                semantic_name=semantic_name,
                mask=merged,
                probability=probabilities[semantic_name],
                used_sam2=any(bool(row.get("used_sam2")) for row in rows),
                component_count=len(selected),
                diagnostics=rows,
            )
        return output


def exclusive_semantic_assignment(
    root: np.ndarray,
    refined: dict[str, RefinedSemanticMask],
    *,
    activation_threshold: float,
    small_region_bonus: float = 0.045,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Resolve sibling overlaps into one visible owner per root pixel."""

    names = sorted(refined)
    if not names:
        return {}, root.copy()
    root_area = max(1, int(root.sum()))
    score_maps: list[np.ndarray] = []
    for name in names:
        item = refined[name]
        mask = np.asarray(item.mask, dtype=bool) & root
        probability = np.asarray(item.probability, dtype=np.float32)
        if probability.shape != root.shape:
            raise ValueError("semantic probabilities must match the root mask")
        area = max(1, int(mask.sum()))
        bonus = small_region_bonus * min(3.0, math.log(root_area / area + 1.0))
        score = probability - activation_threshold + bonus
        score[~mask] = -np.inf
        score_maps.append(score)
    stacked = np.stack(score_maps, axis=0)
    winners = np.argmax(stacked, axis=0)
    valid = np.isfinite(stacked.max(axis=0)) & root
    assigned: dict[str, np.ndarray] = {}
    for index, name in enumerate(names):
        assigned[name] = valid & (winners == index)
    residual = root & ~valid
    return assigned, residual
