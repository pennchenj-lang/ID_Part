from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from hpid_split.prompt_bank import PromptBank
from hpid_split.vlm_parts import (
    PartRegion,
    PlannedPart,
    assign_plans_to_regions,
)


def _root_crop_offset(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("root mask is empty")
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
    padding = max(2, round(max(x1 - x0, y1 - y0) * 0.06))
    return max(0, x0 - padding), max(0, y0 - padding)


def _root_key(metadata: dict[str, object]) -> str:
    return (
        f"{metadata.get('root_origin', 'legacy')}::"
        f"{metadata.get('root_index', 'unknown')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prompt-bank", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--object-label", required=True)
    parser.add_argument("--profile")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    image = Image.open(args.package / "source.png").convert("RGB")
    candidate_rows = json.loads(
        (args.package / "candidates.json").read_text(encoding="utf-8")
    )
    root_row = next(
        row
        for row in candidate_rows
        if row["semantic_name"] == args.domain
        and row["semantic_parent"] == args.domain
        and row["metadata"].get("parent_candidate_key") is None
    )
    root_mask = (
        np.asarray(Image.open(args.package / root_row["mask_path"]).convert("L"))
        > 0
    )
    offset_x, offset_y = _root_crop_offset(root_mask)
    plan_payload = json.loads(args.plan.read_text(encoding="utf-8"))
    plans = tuple(
        PlannedPart(
            str(row["semantic_name"]),
            (
                int(row["bbox_xyxy"][0]) + offset_x,
                int(row["bbox_xyxy"][1]) + offset_y,
                int(row["bbox_xyxy"][2]) + offset_x,
                int(row["bbox_xyxy"][3]) + offset_y,
            ),
            float(row["confidence"]),
            row.get("instance_hint"),
        )
        for row in plan_payload["parts"]
    )
    root_key = _root_key(root_row["metadata"])
    regions: list[PartRegion] = []
    for row in candidate_rows:
        metadata = row["metadata"]
        if _root_key(metadata) != root_key or row is root_row:
            continue
        visual = bool(metadata.get("visual_region"))
        if visual and "generic_visual_region" in metadata:
            generic = bool(metadata["generic_visual_region"])
        else:
            generic = visual and (
                "_visual_" in str(row["semantic_name"])
                or row["semantic_name"] == args.domain
            )
        mask = (
            np.asarray(Image.open(args.package / row["mask_path"]).convert("L"))
            > 0
        )
        regions.append(
            PartRegion(
                mask=mask,
                source=str(row["source"]),
                quality=float(metadata.get("sam_quality", row["score"])),
                candidate_key=str(
                    metadata.get("candidate_key", row["candidate_index"])
                ),
                semantic_name=None if generic else str(row["semantic_name"]),
                generic=generic,
                region_kind=(
                    str(metadata["visual_region_kind"])
                    if metadata.get("visual_region_kind") is not None
                    else None
                ),
            )
        )

    prompt_bank = PromptBank.from_json(args.prompt_bank)
    domain = next(item for item in prompt_bank.domains if item.name == args.domain)
    parts, _, _ = domain.select_parts(
        args.object_label,
        profile_hint=args.profile,
        profile_hint_source="probe" if args.profile else None,
    )
    allowed = {part.semantic_name: part for part in parts}
    plans = tuple(plan for plan in plans if plan.semantic_name in allowed)
    assignments, diagnostics = assign_plans_to_regions(
        plans,
        regions,
        allowed_parts=allowed,
        root_mask=root_mask,
    )
    rows = [
        {
            "semantic_name": plans[item.plan_index].semantic_name,
            "plan_box_xyxy": list(plans[item.plan_index].box_xyxy),
            "region_key": regions[item.region_index].candidate_key,
            "region_source": regions[item.region_index].source,
            "region_previous_semantic": regions[item.region_index].semantic_name,
            "score": item.score,
            "box_containment": item.box_containment,
            "box_fill": item.box_fill,
            "box_iou": item.box_iou,
            "area_prior": item.area_prior,
            "semantic_support": item.semantic_support,
        }
        for item in assignments
    ]
    output = {
        "assignments": rows,
        "diagnostics": diagnostics,
        "ground_truth_used": False,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "assignments.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    preview = np.asarray(image, dtype=np.uint8).copy()
    palette = [
        (255, 75, 75),
        (50, 210, 255),
        (255, 205, 50),
        (95, 225, 95),
        (210, 100, 255),
    ]
    for index, item in enumerate(assignments):
        mask = regions[item.region_index].mask
        color = np.asarray(palette[index % len(palette)], dtype=np.float32)
        preview[mask] = np.clip(
            preview[mask].astype(np.float32) * 0.40 + color * 0.60,
            0,
            255,
        ).astype(np.uint8)
    preview_image = Image.fromarray(preview, mode="RGB")
    draw = ImageDraw.Draw(preview_image)
    for item in assignments:
        plan = plans[item.plan_index]
        draw.rectangle(plan.box_xyxy, outline=(255, 255, 255), width=2)
        draw.text(
            (plan.box_xyxy[0] + 2, plan.box_xyxy[1] + 2),
            plan.semantic_name,
            fill=(255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    preview_image.save(args.output / "assignment_preview.png")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
