from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import binary_fill_holes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--candidate-cache", type=Path, required=True)
    parser.add_argument("--parent-semantic", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="CIDAS/clipseg-rd64-refined")
    parser.add_argument("--sam-model", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--padding", type=float, default=0.15)
    parser.add_argument("--points-per-prompt", type=int, default=2)
    return parser.parse_args()


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def padded_box(mask: np.ndarray, padding: float) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("selected parent candidate has an empty mask")
    x0, x1 = int(xs.min()), int(xs.max() + 1)
    y0, y1 = int(ys.min()), int(ys.max() + 1)
    px = round((x1 - x0) * padding)
    py = round((y1 - y0) * padding)
    height, width = mask.shape
    return max(0, x0 - px), max(0, y0 - py), min(width, x1 + px), min(
        height, y1 + py
    )


def select_points(
    probability: np.ndarray,
    allowed: np.ndarray,
    maximum: int,
) -> list[tuple[int, int, float]]:
    smoothed = cv2.GaussianBlur(probability.astype(np.float32), (0, 0), 2.0)
    work = np.where(allowed, smoothed, -np.inf)
    values = work[np.isfinite(work)]
    if not len(values):
        return []
    peak = float(values.max())
    median = float(np.median(values))
    if peak < 0.08 or peak - median < 0.035:
        return []
    threshold = max(float(np.quantile(values, 0.92)), median + 0.035, peak * 0.52)
    radius = max(4, round(min(probability.shape) * 0.12))
    points: list[tuple[int, int, float]] = []
    for _ in range(maximum):
        flat_index = int(np.argmax(work))
        y, x = np.unravel_index(flat_index, work.shape)
        score = float(work[y, x])
        if not np.isfinite(score) or score < threshold:
            break
        points.append((int(x), int(y), score))
        cv2.circle(work, (int(x), int(y)), radius, -np.inf, -1)
    return points


def select_boxes(
    probability: np.ndarray,
    allowed: np.ndarray,
    maximum: int,
) -> list[tuple[int, int, int, int, float]]:
    smoothed = cv2.GaussianBlur(probability.astype(np.float32), (0, 0), 2.0)
    values = smoothed[allowed]
    if not len(values):
        return []
    peak = float(values.max())
    median = float(np.median(values))
    if peak < 0.08 or peak - median < 0.035:
        return []
    threshold = max(float(np.quantile(values, 0.88)), median + 0.035, peak * 0.46)
    active = ((smoothed >= threshold) & allowed).astype(np.uint8)
    active = cv2.morphologyEx(active, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, components, stats, _ = cv2.connectedComponentsWithStats(active, 8)
    boxes: list[tuple[int, int, int, int, float]] = []
    height, width = probability.shape
    for component_id in range(1, count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < 8:
            continue
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        w = int(stats[component_id, cv2.CC_STAT_WIDTH])
        h = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        pad = max(3, round(max(w, h) * 0.30))
        component = components == component_id
        boxes.append(
            (
                max(0, x - pad),
                max(0, y - pad),
                min(width, x + w + pad),
                min(height, y + h + pad),
                float(smoothed[component].max()),
            )
        )
    boxes.sort(key=lambda item: item[4], reverse=True)
    return boxes[:maximum]


def segment_points(
    image: Image.Image,
    points: list[tuple[int, int, float]],
    processor: object,
    model: object,
    device: str,
) -> list[tuple[np.ndarray, float]]:
    if not points:
        return []
    object_points = [[[float(x), float(y)]] for x, y, _ in points]
    input_points = [object_points]
    input_labels = [[[1] for _ in points]]
    inputs = processor(
        images=image,
        input_points=input_points,
        input_labels=input_labels,
        return_tensors="pt",
    ).to(device)
    with (
        torch.inference_mode(),
        torch.amp.autocast("cuda", enabled=device == "cuda"),
    ):
        outputs = model(**inputs)
    try:
        processed = processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            inputs["original_sizes"].detach().cpu(),
            inputs["reshaped_input_sizes"].detach().cpu(),
            binarize=True,
        )[0]
    except KeyError:
        processed = processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            inputs["original_sizes"].detach().cpu(),
            binarize=True,
        )[0]
    masks = np.asarray(processed)
    scores = outputs.iou_scores.detach().float().cpu().numpy()[0]
    return [
        (
            masks[index, int(scores[index].argmax())].astype(bool),
            float(scores[index].max()),
        )
        for index in range(len(points))
    ]


def segment_boxes(
    image: Image.Image,
    boxes: list[tuple[int, int, int, int, float]],
    processor: object,
    model: object,
    device: str,
) -> list[tuple[np.ndarray, float]]:
    if not boxes:
        return []
    input_boxes = [[[float(value) for value in box[:4]] for box in boxes]]
    inputs = processor(
        images=image,
        input_boxes=input_boxes,
        return_tensors="pt",
    ).to(device)
    with (
        torch.inference_mode(),
        torch.amp.autocast("cuda", enabled=device == "cuda"),
    ):
        outputs = model(**inputs)
    try:
        processed = processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            inputs["original_sizes"].detach().cpu(),
            inputs["reshaped_input_sizes"].detach().cpu(),
            binarize=True,
        )[0]
    except KeyError:
        processed = processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            inputs["original_sizes"].detach().cpu(),
            binarize=True,
        )[0]
    masks = np.asarray(processed)
    scores = outputs.iou_scores.detach().float().cpu().numpy()[0]
    return [
        (
            masks[index, int(scores[index].argmax())].astype(bool),
            float(scores[index].max()),
        )
        for index in range(len(boxes))
    ]


def main() -> int:
    args = parse_args()
    from transformers import (
        AutoProcessor,
        CLIPSegForImageSegmentation,
        Sam2Model,
        Sam2Processor,
    )

    payload = np.load(args.candidate_cache, allow_pickle=False)
    masks = payload["masks"].astype(bool)
    metadata = json.loads(str(payload["metadata"]))
    matches = [
        (index, item)
        for index, item in enumerate(metadata)
        if item["semantic_name"] == args.parent_semantic
    ]
    if not matches:
        raise ValueError(f"no candidate found for {args.parent_semantic!r}")
    index, parent_metadata = max(matches, key=lambda item: item[1]["score"])
    parent_mask = masks[index]
    box = padded_box(parent_mask, args.padding)
    source = Image.open(args.image).convert("RGB")
    crop = source.crop(box)
    x0, y0, x1, y1 = box
    local_parent = parent_mask[y0:y1, x0:x1]
    prompts = [value.strip() for value in args.prompts.split(",") if value.strip()]
    if not prompts:
        raise ValueError("at least one prompt is required")

    processor = AutoProcessor.from_pretrained(args.model)
    model = CLIPSegForImageSegmentation.from_pretrained(args.model).to(args.device)
    model.eval()
    inputs = processor(
        text=prompts,
        images=[crop] * len(prompts),
        padding=True,
        return_tensors="pt",
    ).to(args.device)
    with (
        torch.inference_mode(),
        torch.amp.autocast("cuda", enabled=args.device == "cuda"),
    ):
        logits = model(**inputs).logits[:, None]
        logits = torch.nn.functional.interpolate(
            logits,
            size=(crop.height, crop.width),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        probabilities = logits.sigmoid().float().cpu().numpy()
    sam_processor = Sam2Processor.from_pretrained(args.sam_model)
    sam_model = Sam2Model.from_pretrained(args.sam_model).to(args.device)
    sam_model.eval()

    args.output.mkdir(parents=True, exist_ok=True)
    crop.save(args.output / "crop.png")
    Image.fromarray(local_parent.astype(np.uint8) * 255).save(
        args.output / "parent_mask.png"
    )
    radius = max(3, round(min(local_parent.shape) * 0.06))
    envelope = cv2.dilate(
        local_parent.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * radius + 1, 2 * radius + 1),
        ),
    ).astype(bool)
    envelope = binary_fill_holes(envelope)
    Image.fromarray(envelope.astype(np.uint8) * 255).save(
        args.output / "parent_envelope.png"
    )
    prompt_stats = []
    for prompt, probability in zip(prompts, probabilities, strict=True):
        raw_heatmap = np.clip(probability * 255.0, 0, 255).astype(np.uint8)
        gated = np.where(envelope, probability, 0.0)
        heatmap = np.clip(gated * 255.0, 0, 255).astype(np.uint8)
        threshold = gated >= 0.5
        name = safe_name(prompt)
        Image.fromarray(raw_heatmap).save(args.output / f"{name}_raw_heatmap.png")
        Image.fromarray(heatmap).save(args.output / f"{name}_heatmap.png")
        Image.fromarray(threshold.astype(np.uint8) * 255).save(
            args.output / f"{name}_mask_050.png"
        )
        points = select_points(probability, envelope, args.points_per_prompt)
        segmented = segment_points(
            crop,
            points,
            sam_processor,
            sam_model,
            args.device,
        )
        boxes = select_boxes(probability, envelope, args.points_per_prompt)
        box_segmented = segment_boxes(
            crop,
            boxes,
            sam_processor,
            sam_model,
            args.device,
        )
        for point_index, (mask, quality) in enumerate(segmented, start=1):
            Image.fromarray((mask & envelope).astype(np.uint8) * 255).save(
                args.output / f"{name}_sam_{point_index:02d}.png"
            )
        for box_index, (mask, quality) in enumerate(box_segmented, start=1):
            Image.fromarray((mask & envelope).astype(np.uint8) * 255).save(
                args.output / f"{name}_box_sam_{box_index:02d}.png"
            )
        prompt_stats.append(
            {
                "prompt": prompt,
                "maximum_probability": float(gated.max()),
                "mean_parent_probability": float(gated[local_parent].mean()),
                "pixels_above_0_5": int(np.count_nonzero(threshold)),
                "points": [
                    {"xy": [x, y], "dense_score": score}
                    for x, y, score in points
                ],
                "sam_qualities": [quality for _, quality in segmented],
                "boxes": [
                    {"xyxy": list(box[:4]), "dense_score": box[4]}
                    for box in boxes
                ],
                "box_sam_qualities": [quality for _, quality in box_segmented],
            }
        )
    manifest = {
        "model": args.model,
        "sam_model": args.sam_model,
        "image": str(args.image.resolve()),
        "candidate_cache": str(args.candidate_cache.resolve()),
        "parent_semantic": args.parent_semantic,
        "parent_candidate": parent_metadata,
        "crop_xyxy": list(box),
        "prompts": prompt_stats,
        "ground_truth_used": False,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
