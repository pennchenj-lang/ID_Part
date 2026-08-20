from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from hpid_split.conditional_parts import (
    conditional_part_loss,
    grouped_conditional_part_loss,
    load_conditional_part_samples,
    split_conditional_samples,
    threshold_metrics,
)


def _asset(root: Path, asset_id: str, image_id: int, part: str) -> dict[str, object]:
    asset = root / asset_id
    asset.mkdir()
    image = np.full((16, 20, 3), 127, dtype=np.uint8)
    object_mask = np.zeros((16, 20), dtype=np.uint8)
    object_mask[2:14, 3:18] = 255
    part_mask = np.zeros((16, 20), dtype=np.uint8)
    part_mask[4:9, 5:11] = 255
    Image.fromarray(image).save(asset / "source_crop.png")
    Image.fromarray(object_mask).save(asset / "object_mask_crop.png")
    (asset / "parts_crop").mkdir()
    Image.fromarray(part_mask).save(asset / "parts_crop" / "part.png")
    case = {
        "image_id": image_id,
        "parts": [{"part_name": part, "mask_crop": "parts_crop/part.png"}],
    }
    (asset / "case.json").write_text(json.dumps(case), encoding="utf-8")
    return {
        "asset_id": asset_id,
        "asset_label": "calculator",
        "asset_domain": "device",
        "paco_case": str(asset / "case.json"),
        "part_name_mapping": {part: f"device_{part}"},
    }


def test_conditional_samples_are_grouped_and_split_by_asset(tmp_path: Path) -> None:
    entries = [
        _asset(tmp_path, "a1", 1, "key"),
        _asset(tmp_path, "a2", 2, "key"),
        _asset(tmp_path, "a3", 3, "key"),
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"entries": entries}), encoding="utf-8")

    samples = load_conditional_part_samples(manifest, negative_ratio=0.0)
    train, validation = split_conditional_samples(
        samples, validation_fraction=0.34, seed=7
    )

    assert len(samples) == 3
    assert {sample.asset_id for sample in train}.isdisjoint(
        sample.asset_id for sample in validation
    )
    assert all("calculator" in sample.prompts[-1] for sample in samples)


def test_conditional_loss_rewards_the_correct_small_part() -> None:
    target = torch.zeros((1, 8, 8))
    target[:, 2:4, 3:5] = 1
    root = torch.ones_like(target)
    good = torch.full_like(target, -5.0)
    good[target.bool()] = 5.0
    bad = -good

    good_loss, _ = conditional_part_loss(good, target, root)
    bad_loss, _ = conditional_part_loss(bad, target, root)

    assert good_loss < bad_loss


def test_threshold_metrics_penalize_false_positive_negative_queries() -> None:
    targets = torch.zeros((2, 4, 4))
    targets[0, 1:3, 1:3] = 1
    roots = torch.ones_like(targets)
    probabilities = targets.clone()
    probabilities[1] = 0.8

    metrics = threshold_metrics(
        probabilities,
        targets,
        roots,
        torch.tensor([True, False]),
        [0.5, 0.9],
    )

    assert metrics[0.5]["negative_false_area_fraction"] == 1.0
    assert metrics[0.9]["negative_false_area_fraction"] == 0.0


def test_grouped_loss_prefers_exclusive_sibling_ownership() -> None:
    targets = torch.zeros((2, 8, 8))
    targets[0, :, :4] = 1
    targets[1, :, 4:] = 1
    roots = torch.ones_like(targets)
    exclusive = torch.full_like(targets, -4.0)
    exclusive[targets.bool()] = 4.0
    overlapping = torch.full_like(targets, 4.0)

    exclusive_loss, exclusive_rows = grouped_conditional_part_loss(
        exclusive,
        targets,
        roots,
        torch.tensor([True, True]),
    )
    overlapping_loss, overlapping_rows = grouped_conditional_part_loss(
        overlapping,
        targets,
        roots,
        torch.tensor([True, True]),
    )

    assert exclusive_loss < overlapping_loss
    assert exclusive_rows["overlap"] < overlapping_rows["overlap"]


def test_grouped_loss_suppresses_absent_query() -> None:
    targets = torch.zeros((2, 8, 8))
    targets[0, 2:6, 2:6] = 1
    roots = torch.ones_like(targets)
    clean = torch.full_like(targets, -4.0)
    clean[0, 2:6, 2:6] = 4.0
    false_absent = clean.clone()
    false_absent[1] = 4.0

    clean_loss, _ = grouped_conditional_part_loss(
        clean,
        targets,
        roots,
        torch.tensor([True, False]),
    )
    false_loss, _ = grouped_conditional_part_loss(
        false_absent,
        targets,
        roots,
        torch.tensor([True, False]),
    )

    assert clean_loss < false_loss
