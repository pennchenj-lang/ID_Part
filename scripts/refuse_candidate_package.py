from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from hpid_split.export import export_prediction
from hpid_split.fusion import FusionConfig, MaskCandidate, fuse_candidates
from hpid_split.inference import SplitPrediction


def _load_candidates(package: Path) -> list[MaskCandidate]:
    rows = json.loads((package / "candidates.json").read_text(encoding="utf-8"))
    candidates = []
    for row in rows:
        mask = (
            np.asarray(
                Image.open(package / row["mask_path"]).convert("L"),
                dtype=np.uint8,
            )
            > 0
        )
        candidates.append(
            MaskCandidate(
                semantic_name=str(row["semantic_name"]),
                semantic_parent=str(row["semantic_parent"]),
                mask=mask,
                score=float(row["score"]),
                source=str(row["source"]),
                prompt=str(row.get("prompt", "")),
                source_reliability=float(row.get("source_reliability", 1.0)),
                metadata=dict(row.get("metadata", {})),
            )
        )
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-run HPID ownership fusion on a frozen candidate package."
    )
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-dense-hierarchy", action="store_true")
    args = parser.parse_args()

    image = Image.open(args.package / "source.png").convert("RGB")
    candidates = _load_candidates(args.package)
    dense_hierarchy = not args.no_dense_hierarchy
    result = fuse_candidates(
        candidates,
        image_shape=(image.height, image.width),
        config=FusionConfig(
            use_parent_envelope=dense_hierarchy,
            use_transitive_residual=dense_hierarchy,
        ),
    )
    prediction = SplitPrediction(
        semantic_map=result.semantic_map,
        instance_map=result.instance_map,
        instances=result.instances,
        fine_probabilities=result.evidence,
        parent_probabilities=np.zeros(
            (result.taxonomy.num_parent_classes, image.height, image.width),
            dtype=np.float32,
        ),
        boundary_probability=np.zeros(
            (image.height, image.width), dtype=np.float32
        ),
    )
    diagnostics = {
        "refusion": {
            "source_package": str(args.package.resolve()),
            "ground_truth_used": False,
        },
        "fusion": result.diagnostics,
    }
    export_prediction(
        image,
        prediction,
        result.taxonomy,
        args.output,
        diagnostics=diagnostics,
        candidates=candidates,
    )
    print(f"parts={len(result.instances)} output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
