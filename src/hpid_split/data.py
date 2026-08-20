from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .taxonomy import Taxonomy

DEFAULT_RGB_TO_NAME = {
    (255, 214, 176): "skin",
    (202, 66, 139): "eyes",
    (202, 66, 111): "eyebrow",
    (230, 145, 173): "eyebrow",
    (201, 145, 230): "eyelash",
    (90, 46, 20): "hair",
    (185, 106, 44): "front_hair",
    (58, 31, 15): "back_hair",
    (201, 138, 58): "side_hair",
    (30, 136, 229): "upper_cloth",
    (0, 229, 255): "collar",
    (13, 71, 161): "torso_cloth",
    (100, 181, 246): "sleeve",
    (24, 255, 255): "cuff",
    (83, 109, 254): "hem",
    (181, 255, 240): "inner_cloth",
    (142, 36, 170): "lower_cloth",
    (85, 85, 85): "shoes",
    (255, 140, 0): "shoe_upper",
    (224, 224, 224): "shoe_sole",
    (255, 214, 0): "shoe_tongue",
    (255, 23, 68): "shoelace",
    (121, 85, 72): "heel",
    (255, 255, 255): "sock",
    (0, 200, 83): "accessory",
}


def decode_rgb_labels(path: Path, taxonomy: Taxonomy) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.int32)
    names = {name: index for index, name in enumerate(taxonomy.fine_names)}
    palette_rgb = np.asarray([(0, 0, 0), *DEFAULT_RGB_TO_NAME.keys()], dtype=np.int32)
    palette_ids = np.asarray(
        [0, *(names[DEFAULT_RGB_TO_NAME[color]] for color in DEFAULT_RGB_TO_NAME)],
        dtype=np.uint8,
    )
    flat = rgb.reshape(-1, 3)
    decoded = np.empty(flat.shape[0], dtype=np.uint8)
    for start in range(0, len(flat), 100_000):
        pixels = flat[start : start + 100_000]
        distances = ((pixels[:, None] - palette_rgb[None]) ** 2).sum(axis=2)
        decoded[start : start + len(pixels)] = palette_ids[distances.argmin(axis=1)]
    labels = decoded.reshape(rgb.shape[:2])
    for class_id in np.unique(labels):
        if class_id == 0:
            continue
        count, components, stats, _ = cv2.connectedComponentsWithStats(
            (labels == class_id).astype(np.uint8),
            8,
        )
        for component_id in range(1, count):
            if int(stats[component_id, cv2.CC_STAT_AREA]) < 12:
                labels[components == component_id] = 0
    return labels


def load_label_map(path: Path, taxonomy: Taxonomy) -> np.ndarray:
    image = Image.open(path)
    if image.mode in {"L", "I", "I;16"}:
        labels = np.asarray(image, dtype=np.int64)
        if labels.max(initial=0) >= taxonomy.num_fine_classes:
            raise ValueError(f"Label map {path} contains an unknown class index")
        return labels.astype(np.uint8)
    return decode_rgb_labels(path, taxonomy)


def _component_boxes(
    labels: np.ndarray, detail_ids: Sequence[int]
) -> list[tuple[int, int, int, int, int]]:
    boxes: list[tuple[int, int, int, int, int]] = []
    for class_id in detail_ids:
        count, _, stats, _ = cv2.connectedComponentsWithStats(
            (labels == class_id).astype(np.uint8),
            8,
        )
        for component_id in range(1, count):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < 8:
                continue
            x = int(stats[component_id, cv2.CC_STAT_LEFT])
            y = int(stats[component_id, cv2.CC_STAT_TOP])
            w = int(stats[component_id, cv2.CC_STAT_WIDTH])
            h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
            boxes.append((class_id, x, y, x + w, y + h))
    return boxes


def _detail_crop(
    image: Image.Image,
    labels: np.ndarray,
    boxes: Sequence[tuple[int, int, int, int, int]],
    generator: torch.Generator,
) -> tuple[Image.Image, np.ndarray]:
    selected = boxes[int(torch.randint(len(boxes), (), generator=generator))]
    _, x0, y0, x1, y1 = selected
    width, height = image.size
    component_w = max(1, x1 - x0)
    component_h = max(1, y1 - y0)
    scale = float(torch.empty(1).uniform_(3.2, 6.0, generator=generator))
    side = int(
        max(96, min(max(width, height), round(max(component_w, component_h) * scale)))
    )
    jitter_x = (
        float(torch.empty(1).uniform_(-0.25, 0.25, generator=generator)) * component_w
    )
    jitter_y = (
        float(torch.empty(1).uniform_(-0.25, 0.25, generator=generator)) * component_h
    )
    center_x = (x0 + x1) / 2 + jitter_x
    center_y = (y0 + y1) / 2 + jitter_y
    left = round(center_x - side / 2)
    top = round(center_y - side / 2)
    left = max(0, min(width - side, left)) if side <= width else 0
    top = max(0, min(height - side, top)) if side <= height else 0
    right = min(width, left + side)
    bottom = min(height, top + side)
    return image.crop((left, top, right, bottom)), labels[top:bottom, left:right].copy()


class LayeredAssetDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        image_dir: Path,
        label_dir: Path,
        role_ids: Sequence[str],
        taxonomy: Taxonomy,
        *,
        repeats: int = 8,
        output_size: tuple[int, int] = (448, 288),
        focus_probability: float = 0.65,
        seed: int = 20260811,
    ) -> None:
        self.role_ids = tuple(role_ids)
        self.taxonomy = taxonomy
        self.repeats = repeats
        self.output_size = output_size
        self.focus_probability = focus_probability
        self.seed = seed
        self.epoch = 0
        self.cache: dict[
            str, tuple[Image.Image, np.ndarray, list[tuple[int, int, int, int, int]]]
        ] = {}
        for role_id in self.role_ids:
            image = Image.open(image_dir / f"{role_id}.png").convert("RGB")
            label_path = label_dir / f"{role_id}_final.png"
            if not label_path.exists():
                label_path = label_dir / f"{role_id}.png"
            labels = load_label_map(label_path, taxonomy)
            boxes = _component_boxes(labels, taxonomy.detail_ids)
            self.cache[role_id] = image, labels, boxes

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.role_ids) * self.repeats

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        role_id = self.role_ids[index % len(self.role_ids)]
        image, labels, boxes = self.cache[role_id]
        image = image.copy()
        labels = labels.copy()
        generator = torch.Generator().manual_seed(
            self.seed + self.epoch * 1_000_003 + index * 97
        )
        if (
            boxes
            and torch.rand((), generator=generator).item() < self.focus_probability
        ):
            image, labels = _detail_crop(image, labels, boxes, generator)

        image_tensor = TF.pil_to_tensor(image).float() / 255.0
        target_tensor = torch.from_numpy(labels.astype(np.int64))
        image_tensor = F_resize_image(image_tensor, self.output_size)
        target_tensor = F_resize_target(target_tensor, self.output_size)

        if torch.rand((), generator=generator).item() < 0.5:
            image_tensor = torch.flip(image_tensor, dims=(2,))
            target_tensor = torch.flip(target_tensor, dims=(1,))
        angle = float(torch.empty(1).uniform_(-8.0, 8.0, generator=generator))
        scale = float(torch.empty(1).uniform_(0.90, 1.10, generator=generator))
        translate = [
            int(
                float(torch.empty(1).uniform_(-0.035, 0.035, generator=generator))
                * self.output_size[1]
            ),
            int(
                float(torch.empty(1).uniform_(-0.035, 0.035, generator=generator))
                * self.output_size[0]
            ),
        ]
        image_tensor = TF.affine(
            image_tensor,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=1.0,
        )
        target_tensor = TF.affine(
            target_tensor[None].float(),
            angle=angle,
            translate=translate,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=0.0,
        )[0].long()
        image_tensor = TF.adjust_brightness(
            image_tensor,
            float(torch.empty(1).uniform_(0.78, 1.20, generator=generator)),
        )
        image_tensor = TF.adjust_contrast(
            image_tensor,
            float(torch.empty(1).uniform_(0.80, 1.22, generator=generator)),
        )
        image_tensor = TF.adjust_saturation(
            image_tensor,
            float(torch.empty(1).uniform_(0.82, 1.18, generator=generator)),
        ).clamp(0, 1)
        image_tensor = TF.normalize(
            image_tensor,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return image_tensor, target_tensor


# Backward-compatible alias for existing experiment commands.
CharacterPartDataset = LayeredAssetDataset


def F_resize_image(image: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return torch.nn.functional.interpolate(
        image[None],
        size=size,
        mode="bilinear",
        align_corners=False,
    )[0]


def F_resize_target(target: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return torch.nn.functional.interpolate(
        target[None, None].float(),
        size=size,
        mode="nearest",
    )[0, 0].long()


def class_weights(
    datasets: Sequence[tuple[Image.Image, np.ndarray]],
    class_count: int,
    *,
    background_weight: float = 0.12,
    maximum: float = 12.0,
) -> torch.Tensor:
    counts = np.ones(class_count, dtype=np.float64)
    for _, labels in datasets:
        counts += np.bincount(labels.ravel(), minlength=class_count)
    frequency = counts / counts.sum()
    weights = 1.0 / np.sqrt(frequency + 1e-9)
    weights /= np.median(weights[1:])
    weights = np.clip(weights, 0.25, maximum)
    weights[0] = background_weight
    return torch.tensor(weights, dtype=torch.float32)
