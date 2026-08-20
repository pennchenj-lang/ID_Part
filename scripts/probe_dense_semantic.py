from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from hpid_split.dense_semantic import DenseSemanticProposer


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render one CLIPSeg text probability map for diagnosis."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="CIDAS/clipseg-rd64-refined")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    image = Image.open(args.image).convert("RGB")
    proposer = DenseSemanticProposer(
        args.model,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    probability = proposer.probability_maps(image, [args.prompt])[0]
    low, high = np.quantile(probability, [0.05, 0.99])
    normalized = np.clip((probability - low) / max(1e-6, high - low), 0.0, 1.0)
    source = np.asarray(image, dtype=np.float32)
    heat = np.zeros_like(source)
    heat[..., 0] = 255.0
    heat[..., 1] = 48.0 + 128.0 * (1.0 - normalized)
    alpha = (0.12 + 0.58 * normalized)[..., None]
    overlay = np.clip((1.0 - alpha) * source + alpha * heat, 0, 255).astype(
        np.uint8
    )

    args.output.mkdir(parents=True, exist_ok=True)
    Image.fromarray((normalized * 255).astype(np.uint8), mode="L").save(
        args.output / "probability.png"
    )
    Image.fromarray(overlay, mode="RGB").save(args.output / "overlay.png")
    payload = {
        "image": str(args.image.resolve()),
        "prompt": args.prompt,
        "model": args.model,
        "minimum_probability": float(probability.min()),
        "median_probability": float(np.median(probability)),
        "maximum_probability": float(probability.max()),
        "normalization_quantiles": [float(low), float(high)],
        "ground_truth_used": False,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
