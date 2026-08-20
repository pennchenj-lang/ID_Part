from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from hpid_split.dense_semantic import DenseSemanticProposer
from hpid_split.fusion import MaskCandidate
from hpid_split.mask_refinement import refine_candidate_masks
from hpid_split.semantic_refinement import Sam2SemanticRefiner


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) >= 128


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe CLIPSeg-guided SAM2 refinement for one root mask."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--root-mask", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--activation-quantile", type=float, default=0.70)
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    root = _load_mask(args.root_mask)
    dense = DenseSemanticProposer(
        "CIDAS/clipseg-rd64-refined",
        device=args.device,
        local_files_only=args.local_files_only,
    )
    maps = dense.probability_maps(image, [args.prompt, *args.negative_prompt])

    def normalize(probability: np.ndarray) -> np.ndarray:
        low, high = np.quantile(probability[root], [0.10, 0.95])
        return np.clip((probability - low) / max(1e-6, high - low), 0.0, 1.0)

    target_probability = normalize(maps[0])
    if len(maps) > 1:
        competitors = np.maximum.reduce([normalize(item) for item in maps[1:]])
        discriminative = target_probability - competitors
        probability = np.clip(0.5 + 0.5 * discriminative, 0.0, 1.0)
        values = discriminative[root]
        threshold = max(
            float(np.quantile(values, args.activation_quantile)),
            0.04,
        )
        coarse = (discriminative >= threshold) & root
    else:
        probability = target_probability
        values = probability[root]
        threshold = max(
            float(np.quantile(values, args.activation_quantile)),
            float(np.median(values) + 0.06),
            float(values.max() * 0.52),
        )
        coarse = (probability >= threshold) & root
    coarse = cv2.morphologyEx(
        coarse.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)
    ).astype(bool)
    refiner = Sam2SemanticRefiner(device=args.device)
    refined = refiner.refine(
        image,
        root,
        {"target": coarse},
        {"target": probability},
    )["target"]
    edge_refined = refine_candidate_masks(
        image,
        [
            MaskCandidate(
                "target",
                "target",
                coarse,
                1.0,
                "probe/semantic-root",
            )
        ],
    ).candidates[0].mask

    args.output.mkdir(parents=True, exist_ok=True)
    source = np.asarray(image, dtype=np.uint8)
    for name, mask in (
        ("root", root),
        ("coarse", coarse),
        ("edge_refined", edge_refined),
        ("refined", refined.mask),
    ):
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
            args.output / f"{name}_mask.png"
        )
        overlay = source.copy()
        overlay[mask] = np.clip(
            0.52 * overlay[mask] + 0.48 * np.array([255, 50, 50]),
            0,
            255,
        ).astype(np.uint8)
        Image.fromarray(overlay, mode="RGB").save(
            args.output / f"{name}_overlay.png"
        )
    payload = {
        "image": str(args.image.resolve()),
        "root_mask": str(args.root_mask.resolve()),
        "prompt": args.prompt,
        "negative_prompts": list(args.negative_prompt),
        "activation_quantile": args.activation_quantile,
        "threshold": threshold,
        "root_area_px": int(root.sum()),
        "coarse_area_px": int(coarse.sum()),
        "edge_refined_area_px": int(edge_refined.sum()),
        "refined_area_px": int(refined.mask.sum()),
        "used_sam2": refined.used_sam2,
        "component_count": refined.component_count,
        "diagnostics": list(refined.diagnostics),
        "ground_truth_used": False,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
