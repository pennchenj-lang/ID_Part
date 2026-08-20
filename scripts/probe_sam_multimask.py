from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export every SAM2 multimask candidate for one box prompt."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--box", type=int, nargs=4, required=True)
    parser.add_argument("--model", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    from transformers import Sam2Model, Sam2Processor

    image = Image.open(args.image).convert("RGB")
    processor = Sam2Processor.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    model = Sam2Model.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    ).to(args.device)
    model.eval()
    inputs = processor(
        images=image,
        input_boxes=[[[float(value) for value in args.box]]],
        return_tensors="pt",
    ).to(args.device)
    with (
        torch.inference_mode(),
        torch.amp.autocast("cuda", enabled=args.device == "cuda"),
    ):
        outputs = model(**inputs)
    try:
        processed = processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            inputs["original_sizes"].detach().cpu(),
            inputs["reshaped_input_sizes"].detach().cpu(),
            binarize=True,
        )[0]
    except (KeyError, TypeError):
        processed = processor.post_process_masks(
            outputs.pred_masks.detach().cpu(),
            inputs["original_sizes"].detach().cpu(),
            binarize=True,
        )[0]
    masks = np.asarray(processed)
    if masks.ndim == 3:
        masks = masks[:, None]
    scores = getattr(outputs, "iou_scores", None)
    if scores is None:
        scores = getattr(outputs, "pred_iou_scores", None)
    score_array = (
        scores.detach().float().cpu().numpy()
        if scores is not None
        else np.full(masks.shape[:2], 0.5, dtype=np.float32)
    )
    while score_array.ndim > 2:
        score_array = score_array[0]
    if score_array.ndim == 1:
        score_array = score_array[None, :]

    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    source = np.asarray(image, dtype=np.uint8)
    for index in range(masks.shape[1]):
        mask = masks[0, index].astype(bool)
        area = int(np.count_nonzero(mask))
        mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")
        mask_image.save(args.output / f"mask_{index:02d}.png")
        overlay = source.copy()
        overlay[mask] = np.clip(
            0.52 * overlay[mask] + 0.48 * np.array([255, 50, 50]),
            0,
            255,
        ).astype(np.uint8)
        Image.fromarray(overlay, mode="RGB").save(
            args.output / f"overlay_{index:02d}.png"
        )
        rows.append(
            {
                "candidate_index": index,
                "predicted_iou": float(score_array[0, index]),
                "area_px": area,
                "image_fraction": area / (image.width * image.height),
            }
        )
    manifest = {
        "model": args.model,
        "image": str(args.image.resolve()),
        "box_xyxy": args.box,
        "candidates": rows,
        "ground_truth_used": False,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
