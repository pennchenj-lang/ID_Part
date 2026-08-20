from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from hpid_split.vlm_parts import Qwen3VlPartPlanner, Qwen3VlPlannerConfig
from hpid_split.vlm_roots import build_root_localization_prompt, parse_root_plan


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    prompt = build_root_localization_prompt(args.target)
    planner = Qwen3VlPartPlanner(
        device=args.device,
        config=Qwen3VlPlannerConfig(
            model_name=args.model,
            local_files_only=True,
            load_in_4bit=args.load_in_4bit,
        ),
    )
    response = planner.generate_response(image, prompt)
    planner.release()
    roots, diagnostics = parse_root_plan(response, image_size=image.size)

    args.output.mkdir(parents=True, exist_ok=True)
    preview = image.copy()
    draw = ImageDraw.Draw(preview)
    for index, root in enumerate(roots, start=1):
        draw.rectangle(root.box_xyxy, outline=(255, 48, 48), width=3)
        draw.text(
            (root.box_xyxy[0] + 3, root.box_xyxy[1] + 3),
            f"{index}: {root.confidence:.2f}",
            fill=(255, 48, 48),
            stroke_width=1,
            stroke_fill=(255, 255, 255),
        )
    preview.save(args.output / "root_boxes.png")
    (args.output / "prompt.txt").write_text(prompt, encoding="utf-8")
    (args.output / "response.txt").write_text(response, encoding="utf-8")
    manifest = {
        "target": args.target,
        "roots": [
            {
                "bbox_xyxy": list(root.box_xyxy),
                "confidence": root.confidence,
                "instance_hint": root.instance_hint,
            }
            for root in roots
        ],
        "diagnostics": diagnostics,
        "ground_truth_used": False,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
