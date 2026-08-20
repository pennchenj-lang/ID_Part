from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torch.nn import functional as F
from torch.utils.data import Dataset

from .paco_semantics import normalize_paco_name


@dataclass(frozen=True)
class ConditionalPartSample:
    sample_id: str
    asset_id: str
    asset_label: str
    asset_domain: str
    semantic_name: str
    prompts: tuple[str, ...]
    image_path: Path
    object_mask_path: Path
    part_mask_paths: tuple[Path, ...]
    positive: bool


@dataclass(frozen=True)
class ConditionalBatch:
    model_inputs: dict[str, torch.Tensor]
    targets: torch.Tensor
    root_masks: torch.Tensor
    sample_ids: tuple[str, ...]
    positive: torch.Tensor


@dataclass(frozen=True)
class GroupedConditionalBatch:
    model_inputs: dict[str, torch.Tensor]
    targets: torch.Tensor
    root_masks: torch.Tensor
    sample_ids: tuple[str, ...]
    positive: torch.Tensor
    asset_id: str


def _humanize(value: str) -> str:
    return " ".join(token for token in value.replace("-", "_").split("_") if token)


def _stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _load_case_samples(entry: dict[str, object]) -> list[ConditionalPartSample]:
    case_path = Path(str(entry["paco_case"]))
    payload = json.loads(case_path.read_text(encoding="utf-8"))
    mapping = {
        normalize_paco_name(str(source)): str(target)
        for source, target in dict(entry.get("part_name_mapping", {})).items()
    }
    grouped_paths: dict[str, list[Path]] = defaultdict(list)
    grouped_names: dict[str, set[str]] = defaultdict(set)
    for row in payload.get("parts", []):
        raw_name = normalize_paco_name(str(row["part_name"]))
        semantic_name = mapping.get(raw_name)
        if semantic_name is None:
            continue
        grouped_paths[semantic_name].append(case_path.parent / str(row["mask_crop"]))
        grouped_names[semantic_name].add(_humanize(raw_name))

    asset_id = str(entry["asset_id"])
    asset_label = _humanize(str(entry["asset_label"]))
    samples: list[ConditionalPartSample] = []
    for semantic_name, paths in sorted(grouped_paths.items()):
        semantic_display = _humanize(
            semantic_name.removeprefix(f"{entry['asset_domain']}_")
        )
        prompts = tuple(
            dict.fromkeys(
                [
                    *(sorted(grouped_names[semantic_name])),
                    *(
                        f"{name} of {asset_label}"
                        for name in sorted(grouped_names[semantic_name])
                    ),
                    semantic_display,
                    f"{semantic_display} of {asset_label}",
                ]
            )
        )
        samples.append(
            ConditionalPartSample(
                sample_id=f"{asset_id}/{semantic_name}",
                asset_id=asset_id,
                asset_label=asset_label,
                asset_domain=str(entry["asset_domain"]),
                semantic_name=semantic_name,
                prompts=prompts,
                image_path=case_path.parent / "source_crop.png",
                object_mask_path=case_path.parent / "object_mask_crop.png",
                part_mask_paths=tuple(paths),
                positive=True,
            )
        )
    return samples


def load_conditional_part_samples(
    manifest_path: Path,
    *,
    negative_ratio: float = 0.25,
    seed: int = 20260813,
) -> list[ConditionalPartSample]:
    """Load leakage-auditable conditional masks from a reviewed manifest."""

    if not 0.0 <= negative_ratio <= 1.0:
        raise ValueError("negative_ratio must be in [0, 1]")
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    entries = list(payload.get("entries", []))
    positives: list[ConditionalPartSample] = []
    by_asset: dict[str, list[ConditionalPartSample]] = {}
    inventory: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        samples = _load_case_samples(entry)
        if not samples:
            continue
        positives.extend(samples)
        by_asset[samples[0].asset_id] = samples
        inventory[samples[0].asset_label].update(
            sample.semantic_name for sample in samples
        )

    negatives: list[ConditionalPartSample] = []
    for asset_id, samples in sorted(by_asset.items()):
        present = {sample.semantic_name for sample in samples}
        available = sorted(inventory[samples[0].asset_label] - present)
        if not available:
            continue
        count = min(
            len(available),
            max(1, round(len(samples) * negative_ratio)),
        )
        available.sort(key=lambda name: _stable_fraction(f"{asset_id}:{name}", seed))
        template = samples[0]
        for semantic_name in available[:count]:
            display = _humanize(semantic_name.removeprefix(f"{template.asset_domain}_"))
            negatives.append(
                ConditionalPartSample(
                    sample_id=f"{asset_id}/negative/{semantic_name}",
                    asset_id=asset_id,
                    asset_label=template.asset_label,
                    asset_domain=template.asset_domain,
                    semantic_name=semantic_name,
                    prompts=(display, f"{display} of {template.asset_label}"),
                    image_path=template.image_path,
                    object_mask_path=template.object_mask_path,
                    part_mask_paths=(),
                    positive=False,
                )
            )
    return sorted(
        [*positives, *negatives],
        key=lambda sample: sample.sample_id,
    )


def split_conditional_samples(
    samples: Sequence[ConditionalPartSample],
    *,
    validation_fraction: float = 0.17,
    seed: int = 20260813,
) -> tuple[list[ConditionalPartSample], list[ConditionalPartSample]]:
    """Split by asset inside each object category, never by individual mask."""

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    assets_by_label: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        assets_by_label[sample.asset_label].add(sample.asset_id)
    validation_assets: set[str] = set()
    for label, assets in sorted(assets_by_label.items()):
        ordered = sorted(
            assets,
            key=lambda asset_id: _stable_fraction(f"{label}:{asset_id}", seed),
        )
        if len(ordered) <= 1:
            continue
        count = max(1, min(len(ordered) - 1, round(len(ordered) * validation_fraction)))
        validation_assets.update(ordered[:count])
    train = [sample for sample in samples if sample.asset_id not in validation_assets]
    validation = [sample for sample in samples if sample.asset_id in validation_assets]
    if not train or not validation:
        raise ValueError("conditional split requires at least two independent assets")
    return train, validation


class ConditionalPartDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        samples: Sequence[ConditionalPartSample],
        *,
        training: bool,
        seed: int,
        root_suppression_probability: float = 0.45,
    ) -> None:
        self.samples = tuple(samples)
        self.training = training
        self.seed = seed
        self.epoch = 0
        self.root_suppression_probability = root_suppression_probability

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @staticmethod
    def _mask(path: Path) -> np.ndarray:
        return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) >= 128

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        rng = random.Random(f"{self.seed}:{self.epoch}:{sample.sample_id}")
        image = Image.open(sample.image_path).convert("RGB")
        root = self._mask(sample.object_mask_path)
        target = np.zeros(root.shape, dtype=bool)
        for path in sample.part_mask_paths:
            target |= self._mask(path)
        target &= root
        if self.training and rng.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            root = np.fliplr(root).copy()
            target = np.fliplr(target).copy()
        if self.training:
            image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.88, 1.12))
            image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.88, 1.12))
            image = ImageEnhance.Color(image).enhance(rng.uniform(0.90, 1.10))
        if self.training and rng.random() < self.root_suppression_probability:
            rgb = np.asarray(image, dtype=np.uint8).copy()
            neutral = rng.randint(105, 150)
            rgb[~root] = neutral
            image = Image.fromarray(rgb, mode="RGB")
        prompt = sample.prompts[rng.randrange(len(sample.prompts))]
        return {
            "sample": sample,
            "image": image,
            "root": root,
            "target": target,
            "prompt": prompt,
        }


class ConditionalPartCollator:
    def __init__(self, processor: object) -> None:
        self.processor = processor

    def __call__(self, rows: Sequence[dict[str, object]]) -> ConditionalBatch:
        images = [row["image"] for row in rows]
        prompts = [str(row["prompt"]) for row in rows]
        values = self.processor(
            text=prompts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        height, width = values["pixel_values"].shape[-2:]
        targets = torch.stack(
            [
                torch.from_numpy(
                    cv2.resize(
                        np.asarray(row["target"], dtype=np.uint8),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(np.float32)
                )
                for row in rows
            ]
        )
        roots = torch.stack(
            [
                torch.from_numpy(
                    cv2.resize(
                        np.asarray(row["root"], dtype=np.uint8),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(np.float32)
                )
                for row in rows
            ]
        )
        samples = [row["sample"] for row in rows]
        return ConditionalBatch(
            model_inputs=dict(values),
            targets=targets,
            root_masks=roots,
            sample_ids=tuple(sample.sample_id for sample in samples),
            positive=torch.tensor(
                [sample.positive for sample in samples], dtype=torch.bool
            ),
        )


class GroupedConditionalPartDataset(Dataset[dict[str, object]]):
    """Return all sampled semantic queries for one physical asset together."""

    def __init__(
        self,
        samples: Sequence[ConditionalPartSample],
        *,
        training: bool,
        seed: int,
        maximum_queries: int = 24,
        root_suppression_probability: float = 0.45,
    ) -> None:
        grouped: dict[str, list[ConditionalPartSample]] = defaultdict(list)
        for sample in samples:
            grouped[sample.asset_id].append(sample)
        self.groups = tuple(
            (asset_id, tuple(sorted(rows, key=lambda row: row.sample_id)))
            for asset_id, rows in sorted(grouped.items())
        )
        self.training = training
        self.seed = seed
        self.epoch = 0
        self.maximum_queries = maximum_queries
        self.root_suppression_probability = root_suppression_probability
        if maximum_queries < 2:
            raise ValueError("grouped training requires at least two queries")

    def __len__(self) -> int:
        return len(self.groups)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    @staticmethod
    def _mask(path: Path) -> np.ndarray:
        return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) >= 128

    def _selected(
        self,
        rows: tuple[ConditionalPartSample, ...],
        rng: random.Random,
    ) -> tuple[ConditionalPartSample, ...]:
        positives = [sample for sample in rows if sample.positive]
        negatives = [sample for sample in rows if not sample.positive]
        if self.training:
            rng.shuffle(positives)
            rng.shuffle(negatives)
        selected = [*positives[: self.maximum_queries]]
        selected.extend(negatives[: max(0, self.maximum_queries - len(selected))])
        return tuple(sorted(selected, key=lambda sample: sample.sample_id))

    def __getitem__(self, index: int) -> dict[str, object]:
        asset_id, all_rows = self.groups[index]
        rng = random.Random(f"{self.seed}:{self.epoch}:{asset_id}")
        rows = self._selected(all_rows, rng)
        template = rows[0]
        image = Image.open(template.image_path).convert("RGB")
        root = self._mask(template.object_mask_path)
        targets: list[np.ndarray] = []
        for sample in rows:
            target = np.zeros(root.shape, dtype=bool)
            for path in sample.part_mask_paths:
                target |= self._mask(path)
            targets.append(target & root)
        if self.training and rng.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            root = np.fliplr(root).copy()
            targets = [np.fliplr(target).copy() for target in targets]
        if self.training:
            image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.88, 1.12))
            image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.88, 1.12))
            image = ImageEnhance.Color(image).enhance(rng.uniform(0.90, 1.10))
        if self.training and rng.random() < self.root_suppression_probability:
            rgb = np.asarray(image, dtype=np.uint8).copy()
            rgb[~root] = rng.randint(105, 150)
            image = Image.fromarray(rgb, mode="RGB")
        return {
            "asset_id": asset_id,
            "samples": rows,
            "image": image,
            "root": root,
            "targets": tuple(targets),
            "prompts": tuple(
                sample.prompts[rng.randrange(len(sample.prompts))] for sample in rows
            ),
        }


class GroupedConditionalPartCollator:
    """Collate one variable-sized sibling query group."""

    def __init__(self, processor: object) -> None:
        self.processor = processor

    def __call__(self, rows: Sequence[dict[str, object]]) -> GroupedConditionalBatch:
        if len(rows) != 1:
            raise ValueError("grouped conditional batches require batch_size=1")
        row = rows[0]
        prompts = list(row["prompts"])
        image = row["image"]
        values = self.processor(
            text=prompts,
            images=[image] * len(prompts),
            padding=True,
            return_tensors="pt",
        )
        height, width = values["pixel_values"].shape[-2:]
        targets = torch.stack(
            [
                torch.from_numpy(
                    cv2.resize(
                        np.asarray(target, dtype=np.uint8),
                        (width, height),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(np.float32)
                )
                for target in row["targets"]
            ]
        )
        root = torch.from_numpy(
            cv2.resize(
                np.asarray(row["root"], dtype=np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.float32)
        )
        samples = tuple(row["samples"])
        return GroupedConditionalBatch(
            model_inputs=dict(values),
            targets=targets,
            root_masks=root[None].expand(len(samples), -1, -1),
            sample_ids=tuple(sample.sample_id for sample in samples),
            positive=torch.tensor(
                [sample.positive for sample in samples], dtype=torch.bool
            ),
            asset_id=str(row["asset_id"]),
        )


def conditional_part_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    root_masks: torch.Tensor,
    *,
    dice_weight: float = 0.65,
    outside_root_weight: float = 0.08,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Small-part-aware BCE and Dice constrained by physical root ownership."""

    if logits.shape != targets.shape or targets.shape != root_masks.shape:
        raise ValueError("logits, targets, and root masks must share a shape")
    root_area = root_masks.sum(dim=(1, 2)).clamp_min(1.0)
    positive_area = targets.sum(dim=(1, 2))
    ratio = (root_area - positive_area).clamp_min(1.0) / positive_area.clamp_min(1.0)
    positive_weight = ratio.sqrt().clamp(1.0, 12.0)
    pixel_weight = (root_masks + outside_root_weight * (1.0 - root_masks)) * (
        1.0 + targets * (positive_weight[:, None, None] - 1.0)
    )
    bce_map = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    bce = (bce_map * pixel_weight).sum() / pixel_weight.sum().clamp_min(1.0)

    probabilities = logits.sigmoid() * root_masks
    intersection = (probabilities * targets).sum(dim=(1, 2))
    denominator = probabilities.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
    positive_rows = targets.sum(dim=(1, 2)) > 0
    if bool(positive_rows.any()):
        dice = (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0))[
            positive_rows
        ].mean()
    else:
        dice = logits.new_zeros(())
    loss = bce + dice_weight * dice
    return loss, {
        "loss": float(loss.detach().cpu()),
        "bce": float(bce.detach().cpu()),
        "dice": float(dice.detach().cpu()),
        "mean_positive_weight": float(positive_weight.mean().detach().cpu()),
    }


def grouped_conditional_part_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    root_masks: torch.Tensor,
    positive: torch.Tensor,
    *,
    competition_weight: float = 0.30,
    overlap_weight: float = 0.16,
    negative_weight: float = 0.12,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Jointly learn sibling semantics and discourage duplicate visible owners."""

    base, components = conditional_part_loss(logits, targets, root_masks)
    root = root_masks[0].bool()
    probabilities = logits.sigmoid() * root_masks
    target_count = targets.sum(dim=0)
    exclusive_truth = (target_count == 1) & root
    if bool(exclusive_truth.any()):
        truth_index = targets.argmax(dim=0)
        log_probabilities = torch.log_softmax(logits, dim=0)
        competition = -log_probabilities.gather(0, truth_index[None])[0][
            exclusive_truth
        ].mean()
    else:
        competition = logits.new_zeros(())

    surplus = (probabilities.sum(dim=0) - probabilities.max(dim=0).values).clamp_min(
        0.0
    )
    overlap = surplus[root].mean() if bool(root.any()) else logits.new_zeros(())
    negative_rows = ~positive.to(logits.device)
    negative = (
        probabilities[negative_rows].mean()
        if bool(negative_rows.any())
        else logits.new_zeros(())
    )
    loss = (
        base
        + competition_weight * competition
        + overlap_weight * overlap
        + negative_weight * negative
    )
    return loss, {
        **components,
        "loss": float(loss.detach().cpu()),
        "competition": float(competition.detach().cpu()),
        "overlap": float(overlap.detach().cpu()),
        "negative": float(negative.detach().cpu()),
    }


def balanced_sample_weights(
    samples: Sequence[ConditionalPartSample],
) -> torch.Tensor:
    counts = Counter(sample.semantic_name for sample in samples)
    values = [1.0 / math.sqrt(counts[sample.semantic_name]) for sample in samples]
    mean = sum(values) / max(1, len(values))
    return torch.tensor([value / mean for value in values], dtype=torch.double)


def threshold_metrics(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    roots: torch.Tensor,
    positives: torch.Tensor,
    thresholds: Iterable[float],
) -> dict[float, dict[str, float]]:
    output: dict[float, dict[str, float]] = {}
    probabilities = probabilities.detach().float().cpu()
    targets = targets.detach().bool().cpu()
    roots = roots.detach().bool().cpu()
    positives = positives.detach().bool().cpu()
    for threshold in thresholds:
        predicted = (probabilities >= threshold) & roots
        intersection = (predicted & targets).sum(dim=(1, 2)).float()
        union = (predicted | targets).sum(dim=(1, 2)).float()
        iou = intersection / union.clamp_min(1.0)
        positive_iou = float(iou[positives].mean()) if bool(positives.any()) else 0.0
        negative_rows = ~positives
        false_fraction = (
            float(
                (
                    predicted[negative_rows].sum(dim=(1, 2)).float()
                    / roots[negative_rows].sum(dim=(1, 2)).float().clamp_min(1.0)
                ).mean()
            )
            if bool(negative_rows.any())
            else 0.0
        )
        output[float(threshold)] = {
            "positive_mean_iou": positive_iou,
            "negative_false_area_fraction": false_fraction,
            "selection_score": positive_iou - 0.25 * false_fraction,
        }
    return output
