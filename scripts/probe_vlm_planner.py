from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from hpid_split.prompt_bank import PromptBank
from hpid_split.vlm_parts import (
    Qwen3VlPartPlanner,
    Qwen3VlPlannerConfig,
    build_part_planner_prompt,
    parse_part_plan,
)


def _mask_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("root mask is empty")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--prompt-bank", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--object-label", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--semantic", action="append", default=[])
    parser.add_argument("--per-semantic", action="store_true")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    root_mask = (
        np.asarray(Image.open(args.mask).convert("L")) > 0
        if args.mask is not None
        else np.ones((image.height, image.width), dtype=bool)
    )
    if root_mask.shape != (image.height, image.width):
        raise ValueError("root mask dimensions do not match source image")
    x0, y0, x1, y1 = _mask_box(root_mask)
    padding = max(2, round(max(x1 - x0, y1 - y0) * 0.06))
    x0, y0 = max(0, x0 - padding), max(0, y0 - padding)
    x1, y1 = min(image.width, x1 + padding), min(image.height, y1 + padding)
    crop = image.crop((x0, y0, x1, y1))
    local_root = root_mask[y0:y1, x0:x1]
    crop_array = np.asarray(crop, dtype=np.uint8).copy()
    crop_array[~local_root] = 127
    crop = Image.fromarray(crop_array, mode="RGB")

    prompt_bank = PromptBank.from_json(args.prompt_bank)
    domain = next(item for item in prompt_bank.domains if item.name == args.domain)
    parts, selected_profile, profile_diagnostics = domain.select_parts(
        args.object_label,
        profile_hint=args.profile,
        profile_hint_source="probe" if args.profile else None,
    )
    if args.semantic:
        requested = set(args.semantic)
        parts = tuple(part for part in parts if part.semantic_name in requested)
        missing = requested - {part.semantic_name for part in parts}
        if missing:
            raise ValueError(f"requested semantics are unavailable: {sorted(missing)}")
    planner = Qwen3VlPartPlanner(
        device=args.device,
        config=Qwen3VlPlannerConfig(
            model_name=args.model,
            local_files_only=True,
            load_in_4bit=args.load_in_4bit,
        ),
    )
    query_groups = [(part,) for part in parts] if args.per_semantic else [parts]
    prompts: list[str] = []
    responses: list[str] = []
    planned_parts = []
    parse_diagnostics = []
    for query_parts in query_groups:
        prompt = build_part_planner_prompt(
            object_label=args.object_label,
            domain_name=domain.name,
            parts=query_parts,
            context_parts=parts,
        )
        response = planner.generate_response(crop, prompt)
        parsed = parse_part_plan(
            response,
            image_size=crop.size,
            allowed_parts={part.semantic_name: part for part in query_parts},
        )
        prompts.append(prompt)
        responses.append(response)
        planned_parts.extend(parsed.parts)
        parse_diagnostics.append(parsed.diagnostics)
    planner.release()

    args.output.mkdir(parents=True, exist_ok=True)
    crop.save(args.output / "planner_input.png")
    (args.output / "planner_prompt.txt").write_text(
        "\n\n===== QUERY =====\n\n".join(prompts), encoding="utf-8"
    )
    (args.output / "planner_response.txt").write_text(
        "\n\n===== RESPONSE =====\n\n".join(responses), encoding="utf-8"
    )
    manifest = {
        "model": args.model,
        "domain": args.domain,
        "object_label": args.object_label,
        "selected_profile": selected_profile,
        "profile_selection": profile_diagnostics,
        "parts": [
            {
                "semantic_name": item.semantic_name,
                "bbox_xyxy": list(item.box_xyxy),
                "confidence": item.confidence,
                "instance_hint": item.instance_hint,
            }
            for item in planned_parts
        ],
        "diagnostics": parse_diagnostics,
        "ground_truth_used": False,
    }
    (args.output / "planner_plan.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    preview = crop.copy()
    draw = ImageDraw.Draw(preview)
    for item in planned_parts:
        draw.rectangle(item.box_xyxy, outline=(255, 50, 50), width=2)
        draw.text(
            (item.box_xyxy[0] + 2, item.box_xyxy[1] + 2),
            item.semantic_name,
            fill=(255, 50, 50),
            stroke_width=1,
            stroke_fill=(255, 255, 255),
        )
    preview.save(args.output / "planner_boxes.png")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
