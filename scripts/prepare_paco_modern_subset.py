from __future__ import annotations

import argparse
import json
from pathlib import Path

from hpid_split.paco import (
    load_paco_cases,
    materialize_paco_case,
    sha256_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize an auditable modern-object subset from PACO-LVIS."
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-distinct-parts", type=int, default=2)
    parser.add_argument("--minimum-bbox-side", type=int, default=64)
    parser.add_argument("--crop-padding", type=float, default=0.10)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    categories = [str(row["category"]) for row in config["categories"]]
    metadata = {str(row["category"]): row for row in config["categories"]}
    alternatives = load_paco_cases(
        args.annotations,
        categories,
        minimum_distinct_parts=args.minimum_distinct_parts,
        minimum_bbox_side=args.minimum_bbox_side,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    annotation_sha = sha256_file(args.annotations)
    dataset_split = args.annotations.stem.removeprefix("paco_lvis_v1_")
    rows: list[dict[str, object]] = []
    used_image_ids: set[int] = set()
    for category in categories:
        candidates = alternatives.get(category, [])
        candidates.sort(key=lambda case: case.image_id in used_image_ids)
        errors: list[str] = []
        for case in candidates:
            case_id = category.replace("_(", "_").replace(")", "")
            case_dir = args.output / case_id
            try:
                payload = materialize_paco_case(
                    case,
                    case_dir,
                    annotation_sha256=annotation_sha,
                    crop_padding=args.crop_padding,
                    dataset_split=dataset_split,
                )
            except (OSError, TypeError, ValueError) as error:
                errors.append(f"object_annotation_id={case.object_annotation_id}: {error}")
                continue
            used_image_ids.add(case.image_id)
            rows.append(
                {
                    "case_id": case_id,
                    "expected_domain": metadata[category]["expected_domain"],
                    "expected_profile": metadata[category].get("expected_profile"),
                    "case_path": str((case_dir / "case.json").resolve()),
                    **payload,
                }
            )
            print(f"{category}: image_id={case.image_id} parts={payload['part_count']}")
            break
        else:
            rows.append(
                {
                    "case_id": category,
                    "object_category": category,
                    "status": "not_materialized",
                    "errors": errors or ["no eligible annotation"],
                }
            )
            print(f"{category}: not materialized")
    manifest = {
        "format": "HPID PACO modern-object subset",
        "format_version": "0.1.0",
        "source_annotations": str(args.annotations.resolve()),
        "source_annotation_sha256": annotation_sha,
        "selection_uses_ground_truth_metadata": True,
        "selection_metadata_is_not_available_to_hpid_inference": True,
        "case_count": sum("case_path" in row for row in rows),
        "cases": rows,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0 if manifest["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
