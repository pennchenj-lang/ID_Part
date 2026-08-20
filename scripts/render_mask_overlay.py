from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a binary-mask overlay for visual diagnostics."
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--color", type=int, nargs=3, default=(255, 48, 48))
    parser.add_argument("--opacity", type=float, default=0.52)
    args = parser.parse_args()

    image = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8).copy()
    mask = np.asarray(Image.open(args.mask).convert("L"), dtype=np.uint8) > 0
    if mask.shape != image.shape[:2]:
        raise ValueError(
            f"mask shape {mask.shape} does not match image shape {image.shape[:2]}"
        )
    opacity = float(np.clip(args.opacity, 0.0, 1.0))
    color = np.asarray(args.color, dtype=np.float32)
    image[mask] = np.clip(
        (1.0 - opacity) * image[mask].astype(np.float32) + opacity * color,
        0,
        255,
    ).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
