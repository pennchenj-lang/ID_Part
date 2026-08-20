from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.package_root))
    os.environ["TORCH_HOME"] = str(args.model_cache)
    from simple_lama_inpainting import SimpleLama

    request = json.loads(args.request.read_text(encoding="utf-8"))
    source = Image.open(request["source"]).convert("RGB")
    visible = np.asarray(Image.open(request["visible_mask"]).convert("L")) >= 128
    full = np.asarray(Image.open(request["proposed_full_mask"]).convert("L")) >= 128
    hidden = full & ~visible
    if hidden.any():
        result = SimpleLama()(
            source, Image.fromarray(hidden.astype(np.uint8) * 255)
        ).convert("RGBA")
    else:
        result = source.convert("RGBA")
    Image.fromarray(full.astype(np.uint8) * 255).save(request["full_mask_output"])
    result.save(request["generated_rgba_output"])
    Path(request["metadata_output"]).write_text(
        json.dumps(
            {
                "confidence": min(
                    0.45, float(request.get("proposed_shape_confidence", 0.0))
                ),
                "amodal_shape_source": "HPID geometric fallback",
                "hidden_appearance_source": "LaMa",
                "warning": (
                    "LaMa supplies appearance only; the amodal mask is not a "
                    "learned physical-structure estimate"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
