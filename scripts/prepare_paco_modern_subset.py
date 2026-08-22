from __future__ import annotations

import argparse
import hashlib
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
    parser.add_argument("--instances-per-category", type=int, default=1)
    parser.add_argument("--alternatives-per-category", type=int, default=32)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help=(
            "Exclude every object annotation and source image recorded in an "
            "earlier materialized manifest. May be supplied more than once."
        ),
    )
    args = parser.parse_args()

    if args.instances_per_category < 1:
        raise ValueError("--instances-per-category must be positive")
    if args.alternatives_per_category < args.instances_per_category:
        raise ValueError(
            "--alternatives-per-category must be at least "
            "--instances-per-category"
        )

    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    categories = [str(row["category"]) for row in config["categories"]]
    metadata = {str(row["category"]): row for row in config["categories"]}
    alternatives = load_paco_cases(
        args.annotations,
        categories,
        minimum_distinct_parts=args.minimum_distinct_parts,
        minimum_bbox_side=args.minimum_bbox_side,
        alternatives_per_category=args.alternatives_per_category,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    annotation_sha = sha256_file(args.annotations)
    dataset_split = args.annotations.stem.removeprefix("paco_lvis_v1_")
    excluded_object_ids: set[int] = set()
    excluded_image_ids: set[int] = set()
    exclusion_rows: list[dict[str, object]] = []
    for manifest_path in args.exclude_manifest:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in payload.get("cases", []):
            if row.get("object_annotation_id") is not None:
                excluded_object_ids.add(int(row["object_annotation_id"]))
            if row.get("image_id") is not None:
                excluded_image_ids.add(int(row["image_id"]))
        exclusion_rows.append(
            {
                "path": str(manifest_path.resolve()),
                "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            }
        )

    rows: list[dict[str, object]] = []
    used_image_ids: set[int] = set()
    for category in categories:
        candidates = alternatives.get(category, [])
        candidates.sort(key=lambda case: case.image_id in used_image_ids)
        errors: list[str] = []
        materialized = 0
        for case in candidates:
            if (
                case.object_annotation_id in excluded_object_ids
                or case.image_id in excluded_image_ids
                or case.image_id in used_image_ids
            ):
                continue
            suffix = (
                f"__{materialized + 1:02d}"
                if args.instances_per_category > 1
                else ""
            )
            case_id = (
                category.replace("_(", "_").replace(")", "") + suffix
            )
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
            materialized += 1
            print(
                f"{category}[{materialized}/{args.instances_per_category}]: "
                f"image_id={case.image_id} parts={payload['part_count']}"
            )
            if materialized >= args.instances_per_category:
                break
        if materialized < args.instances_per_category:
            rows.append(
                {
                    "case_id": f"{category}__missing_{materialized + 1:02d}",
                    "object_category": category,
                    "status": "not_materialized",
                    "requested_instances": args.instances_per_category,
                    "materialized_instances": materialized,
                    "errors": errors or ["no eligible annotation"],
                }
            )
            print(
                f"{category}: materialized {materialized}/"
                f"{args.instances_per_category}"
            )
    manifest = {
        "format": "HPID PACO modern-object subset",
        "format_version": "0.1.0",
        "source_annotations": str(args.annotations.resolve()),
        "source_annotation_sha256": annotation_sha,
        "selection_uses_ground_truth_metadata": True,
        "selection_metadata_is_not_available_to_hpid_inference": True,
        "instances_per_category_requested": args.instances_per_category,
        "alternatives_per_category_considered": args.alternatives_per_category,
        "excluded_manifests": exclusion_rows,
        "excluded_object_annotation_count": len(excluded_object_ids),
        "excluded_source_image_count": len(excluded_image_ids),
        "source_images_are_unique_within_manifest": True,
        "case_count": sum("case_path" in row for row in rows),
        "cases": rows,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0 if manifest["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
