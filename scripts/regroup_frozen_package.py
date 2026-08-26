from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from hpid_split.export import (
    colorize_part_ids,
    load_previous_package,
    render_source_overlay,
)
from hpid_split.fusion import MaskCandidate
from hpid_split.physical_groups import build_physical_groups


def _load_candidates(package_dir: Path) -> tuple[MaskCandidate, ...]:
    rows = json.loads((package_dir / "candidates.json").read_text(encoding="utf-8"))
    candidates: list[MaskCandidate] = []
    for row in rows:
        mask = np.asarray(
            Image.open(package_dir / str(row["mask_path"])).convert("L"),
            dtype=np.uint8,
        ) > 0
        candidates.append(
            MaskCandidate(
                semantic_name=str(row["semantic_name"]),
                semantic_parent=str(row["semantic_parent"]),
                mask=mask,
                score=float(row["score"]),
                source=str(row["source"]),
                prompt=str(row.get("prompt") or ""),
                source_reliability=float(row.get("source_reliability", 1.0)),
                metadata=dict(row.get("metadata") or {}),
            )
        )
    return tuple(candidates)


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "nonzero": int(np.count_nonzero(value)),
        }
    if isinstance(value, np.generic):
        return value.item()
    return value


def regroup(package_dir: Path, output_dir: Path) -> None:
    instance_map, records = load_previous_package(package_dir)
    candidates = _load_candidates(package_dir)
    image = Image.open(package_dir / "source.png").convert("RGB")
    result = build_physical_groups(
        instance_map,
        records,
        candidates=candidates,
        image=image,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    image.save(output_dir / "source.png")
    shutil.copy2(package_dir / "part_id_map.tiff", output_dir / "part_id_map.tiff")
    Image.fromarray(result.group_map.astype(np.uint16)).save(
        output_dir / "group_id_map.tiff"
    )
    colorize_part_ids(result.group_map).save(output_dir / "group_id_preview.png")
    render_source_overlay(image, result.group_map).save(output_dir / "group_overlay.png")
    (output_dir / "groups.json").write_text(
        json.dumps([group.to_dict() for group in result.groups], indent=2),
        encoding="utf-8",
    )
    (output_dir / "regroup_diagnostics.json").write_text(
        json.dumps(_json_safe(result.diagnostics), indent=2),
        encoding="utf-8",
    )
    original_parts = json.loads(
        (package_dir / "parts.json").read_text(encoding="utf-8")
    )
    group_by_part = {record.part_id: record.group_id for record in result.records}
    for row in original_parts:
        row["group_id"] = group_by_part.get(str(row["part_id"]), row.get("group_id"))
    (output_dir / "parts.json").write_text(
        json.dumps(original_parts, indent=2), encoding="utf-8"
    )
    for name in ("quality_report.json",):
        source = package_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)
    (output_dir / "source_package.txt").write_text(
        str(package_dir.resolve()) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute public physical groups from a frozen HPID candidate package."
    )
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    regroup(args.package, args.output)


if __name__ == "__main__":
    main()
