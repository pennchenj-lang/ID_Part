from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms import functional as TF

from .instances import PartInstance, semantic_to_part_ids
from .taxonomy import Taxonomy


@dataclass(frozen=True)
class SplitPrediction:
    semantic_map: np.ndarray
    instance_map: np.ndarray
    instances: tuple[PartInstance, ...]
    fine_probabilities: np.ndarray
    parent_probabilities: np.ndarray
    boundary_probability: np.ndarray


def _normalized_tensor(
    image: Image.Image, height: int
) -> tuple[torch.Tensor, tuple[int, int]]:
    width, native_height = image.size
    eval_width = max(1, round(width * height / native_height))
    tensor = TF.pil_to_tensor(image).float() / 255.0
    tensor = F.interpolate(
        tensor[None], size=(height, eval_width), mode="bilinear", align_corners=False
    )[0]
    tensor = TF.normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return tensor, (native_height, width)


def _forward_probabilities(
    model: torch.nn.Module,
    image: Image.Image,
    device: str,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor, native_size = _normalized_tensor(image, height)
    model.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device == "cuda"):
        outputs = model(tensor[None].to(device))
        fine = F.interpolate(
            outputs["fine"], size=native_size, mode="bilinear", align_corners=False
        ).softmax(dim=1)
        parent = F.interpolate(
            outputs["parent"], size=native_size, mode="bilinear", align_corners=False
        ).softmax(dim=1)
        boundary = F.interpolate(
            outputs["boundary"], size=native_size, mode="bilinear", align_corners=False
        ).sigmoid()
    return (
        fine[0].float().cpu().numpy(),
        parent[0].float().cpu().numpy(),
        boundary[0, 0].float().cpu().numpy(),
    )


def parent_condition(
    fine_probabilities: np.ndarray,
    parent_probabilities: np.ndarray,
    taxonomy: Taxonomy,
    strength: float = 0.65,
) -> np.ndarray:
    compatibility = parent_probabilities[np.asarray(taxonomy.fine_to_parent)]
    log_score = np.log(fine_probabilities + 1e-7) + strength * np.log(
        compatibility + 1e-7
    )
    log_score -= log_score.max(axis=0, keepdims=True)
    score = np.exp(log_score)
    return score / score.sum(axis=0, keepdims=True).clip(min=1e-7)


def boundary_aware_smooth(
    probabilities: np.ndarray, boundary: np.ndarray
) -> np.ndarray:
    blend = np.clip(boundary * 1.4, 0.0, 1.0)[None]
    smoothed = np.empty_like(probabilities)
    for class_id, channel in enumerate(probabilities):
        smoothed[class_id] = cv2.bilateralFilter(
            channel.astype(np.float32), 5, 0.08, 3.0
        )
    output = blend * probabilities + (1.0 - blend) * smoothed
    return output / output.sum(axis=0, keepdims=True).clip(min=1e-7)


def _bbox(mask: np.ndarray, padding: float) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) < 24:
        return None
    height, width = mask.shape
    x0, x1 = int(xs.min()), int(xs.max() + 1)
    y0, y1 = int(ys.min()), int(ys.max() + 1)
    px = round((x1 - x0) * padding)
    py = round((y1 - y0) * padding)
    return max(0, x0 - px), max(0, y0 - py), min(width, x1 + px), min(height, y1 + py)


def recursive_detail_refinement(
    model: torch.nn.Module,
    image: Image.Image,
    fine: np.ndarray,
    parent: np.ndarray,
    taxonomy: Taxonomy,
    device: str,
    *,
    roi_height: int = 640,
    blend_weight: float = 0.55,
    maximum_rois_per_parent: int = 4,
    maximum_total_rois: int = 16,
    minimum_roi_fraction: float = 0.0005,
) -> tuple[np.ndarray, np.ndarray]:
    parent_map = parent.argmax(axis=0)
    refined = fine.copy()
    refined_boundary = np.zeros(parent_map.shape, dtype=np.float32)
    used_rois = 0
    minimum_roi_area = max(48, round(parent_map.size * minimum_roi_fraction))
    for parent_id in range(1, taxonomy.num_parent_classes):
        child_ids = taxonomy.child_ids(parent_id, include_parent_fallback=False)
        if not child_ids:
            continue
        count, components, stats, _ = cv2.connectedComponentsWithStats(
            (parent_map == parent_id).astype(np.uint8), 8
        )
        component_ids = [
            component_id
            for component_id in range(1, count)
            if int(stats[component_id, cv2.CC_STAT_AREA]) >= minimum_roi_area
        ]
        component_ids.sort(
            key=lambda component_id: int(stats[component_id, cv2.CC_STAT_AREA]),
            reverse=True,
        )
        for component_id in component_ids[:maximum_rois_per_parent]:
            if used_rois >= maximum_total_rois:
                break
            box = _bbox(components == component_id, 0.16)
            if box is None:
                continue
            x0, y0, x1, y1 = box
            crop = image.crop(box)
            local_fine, _, local_boundary = _forward_probabilities(
                model, crop, device, roi_height
            )
            region = refined[list(child_ids), y0:y1, x0:x1]
            local = local_fine[list(child_ids)]
            combined = np.exp(
                (1.0 - blend_weight) * np.log(region + 1e-7)
                + blend_weight * np.log(local + 1e-7)
            )
            refined[list(child_ids), y0:y1, x0:x1] = combined
            refined_boundary[y0:y1, x0:x1] = np.maximum(
                refined_boundary[y0:y1, x0:x1], local_boundary
            )
            used_rois += 1
        if used_rois >= maximum_total_rois:
            break
    refined /= refined.sum(axis=0, keepdims=True).clip(min=1e-7)
    return refined, refined_boundary


def clean_semantic_map(
    labels: np.ndarray, taxonomy: Taxonomy, minimum_area: int = 12
) -> np.ndarray:
    cleaned = labels.copy()
    parent_root_fine = {
        parent_id: taxonomy.fine_names.index(parent_name)
        for parent_id, parent_name in enumerate(taxonomy.parent_names)
        if parent_name in taxonomy.fine_names
    }
    for class_id in range(1, taxonomy.num_fine_classes):
        count, components, stats, _ = cv2.connectedComponentsWithStats(
            (cleaned == class_id).astype(np.uint8), 8
        )
        replacement = parent_root_fine.get(taxonomy.fine_to_parent[class_id], 0)
        threshold = (
            minimum_area if class_id in taxonomy.detail_ids else max(20, minimum_area)
        )
        for component_id in range(1, count):
            if int(stats[component_id, cv2.CC_STAT_AREA]) < threshold:
                cleaned[components == component_id] = replacement
    return cleaned


def predict(
    model: torch.nn.Module,
    image: Image.Image,
    taxonomy: Taxonomy,
    *,
    device: str,
    evaluation_height: int = 768,
    recursive: bool = True,
    use_parent_conditioning: bool = True,
    use_boundary_refinement: bool = True,
) -> SplitPrediction:
    fine, parent, boundary = _forward_probabilities(
        model, image, device, evaluation_height
    )
    if use_parent_conditioning:
        fine = parent_condition(fine, parent, taxonomy)
    if recursive:
        fine, local_boundary = recursive_detail_refinement(
            model, image, fine, parent, taxonomy, device
        )
        boundary = np.maximum(boundary, local_boundary)
    if use_boundary_refinement:
        fine = boundary_aware_smooth(fine, boundary)
    semantic = clean_semantic_map(fine.argmax(axis=0).astype(np.uint8), taxonomy)
    instance_map, instances = semantic_to_part_ids(semantic, taxonomy)
    return SplitPrediction(
        semantic_map=semantic,
        instance_map=instance_map,
        instances=tuple(instances),
        fine_probabilities=fine,
        parent_probabilities=parent,
        boundary_probability=boundary,
    )
