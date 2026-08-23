from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from hpid_split.export import export_prediction, load_previous_package
from hpid_split.fusion import MaskCandidate
from hpid_split.inference import SplitPrediction
from hpid_split.physical_groups import build_physical_groups
from hpid_split.taxonomy import Taxonomy


def _load_candidates(package: Path) -> list[MaskCandidate]:
    rows = json.loads((package / "candidates.json").read_text(encoding="utf-8"))
    candidates: list[MaskCandidate] = []
    for row in rows:
        mask = (
            np.asarray(
                Image.open(package / str(row["mask_path"])).convert("L"),
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


def regroup_package(package: Path, output: Path) -> dict[str, object]:
    """Rebuild only the editable-group layer of one frozen package."""

    package = package.resolve()
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output directory is not empty: {output}")

    image = Image.open(package / "source.png").convert("RGB")
    instance_map, records = load_previous_package(package)
    semantic_map = np.asarray(Image.open(package / "semantic_ids.png"), dtype=np.uint8)
    taxonomy = Taxonomy.from_json(package / "taxonomy.json")
    candidates = _load_candidates(package)
    diagnostics_path = package / "inference_diagnostics.json"
    diagnostics = (
        json.loads(diagnostics_path.read_text(encoding="utf-8"))
        if diagnostics_path.is_file()
        else {}
    )
    diagnostics["physical_group_regeneration"] = {
        "source_package": str(package),
        "fine_part_map_changed": False,
        "candidate_generation_rerun": False,
        "ground_truth_used": False,
    }

    height, width = instance_map.shape
    prediction = SplitPrediction(
        semantic_map=semantic_map,
        instance_map=instance_map,
        instances=tuple(records),
        fine_probabilities=np.zeros(
            (taxonomy.num_fine_classes, height, width), dtype=np.float32
        ),
        parent_probabilities=np.zeros(
            (taxonomy.num_parent_classes, height, width), dtype=np.float32
        ),
        boundary_probability=np.zeros((height, width), dtype=np.float32),
    )
    physical_groups = build_physical_groups(
        instance_map,
        records,
        candidates=candidates,
        image=image,
    )
    diagnostics["physical_grouping"] = physical_groups.diagnostics
    return export_prediction(
        image,
        prediction,
        taxonomy,
        output,
        records=records,
        diagnostics=diagnostics,
        candidates=candidates,
        physical_groups=physical_groups,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild editable physical groups from a frozen HPID Part-ID package "
            "without rerunning candidate generation or ownership fusion."
        )
    )
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    manifest = regroup_package(args.package, output)
    print(
        f"parts={manifest['part_count']} groups={manifest['group_count']} "
        f"output={output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
