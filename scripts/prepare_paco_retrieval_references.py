from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from hpid_split.paco import (
    load_paco_cases,
    materialize_paco_case,
    sha256_file,
)
from hpid_split.paco_semantics import canonical_semantic_name, normalize_paco_name
from hpid_split.prompt_bank import DomainPrompt, PromptBank
from hpid_split.retrieval import REFERENCE_FORMAT


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _excluded_image_ids(manifests: list[Path]) -> set[int]:
    excluded: set[int] = set()
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        for row in payload.get("cases", []):
            if isinstance(row, dict) and row.get("image_id") is not None:
                excluded.add(int(row["image_id"]))
    return excluded


def _domain_inventory(
    domain: DomainPrompt,
    profile_name: str,
    object_label: str,
) -> tuple[set[str], tuple[str, ...]]:
    selected, resolved_profile, _ = domain.select_parts(
        object_label,
        profile_hint=profile_name,
        profile_hint_source="paco_training_config",
    )
    if resolved_profile != profile_name:
        raise ValueError(
            f"failed to resolve profile {profile_name!r} in {domain.name!r}"
        )
    profile = next(item for item in domain.part_profiles if item.name == profile_name)
    return (
        {part.semantic_name for part in selected},
        tuple(profile.root_hints),
    )


def _part_mapping(
    part_categories: tuple[str, ...],
    *,
    domain: str,
    object_category: str,
    allowed_semantics: set[str],
) -> tuple[dict[str, str], set[str]]:
    mapping: dict[str, str] = {}
    excluded: set[str] = set()
    for category in part_categories:
        raw_name = category.split(":", maxsplit=1)[-1]
        source_name = normalize_paco_name(raw_name)
        semantic_name = canonical_semantic_name(
            source_name,
            domain,
            object_category=object_category,
            allowed_semantics=allowed_semantics,
        )
        if semantic_name is None:
            excluded.add(source_name)
        else:
            mapping[source_name] = semantic_name
    return mapping, excluded


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a leakage-safe PACO training reference set for HPID's learned "
            "modern-object prototype layer."
        )
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt-bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets-per-category", type=int, default=3)
    parser.add_argument("--minimum-mapped-parts", type=int, default=2)
    parser.add_argument("--minimum-bbox-side", type=int, default=64)
    parser.add_argument("--crop-padding", type=float, default=0.10)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help="Repeatable benchmark manifest whose image IDs must not enter training.",
    )
    args = parser.parse_args()
    if args.assets_per_category < 1:
        parser.error("--assets-per-category must be positive")
    if args.minimum_mapped_parts < 1:
        parser.error("--minimum-mapped-parts must be positive")

    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    category_rows = {
        str(row["category"]): row for row in config.get("categories", [])
    }
    if not category_rows:
        raise ValueError("category config is empty")
    bank = PromptBank.from_json(args.prompt_bank)
    domain_by_name = {domain.name: domain for domain in bank.domains}
    excluded_image_ids = _excluded_image_ids(args.exclude_manifest)
    alternatives = load_paco_cases(
        args.annotations,
        list(category_rows),
        minimum_distinct_parts=1,
        minimum_bbox_side=args.minimum_bbox_side,
        alternatives_per_category=max(12, args.assets_per_category * 5),
    )

    args.output.mkdir(parents=True, exist_ok=True)
    assets_dir = args.output / "assets"
    assets_dir.mkdir(exist_ok=True)
    annotation_sha = sha256_file(args.annotations)
    dataset_split = args.annotations.stem.removeprefix("paco_lvis_v1_")
    used_image_ids: set[int] = set(excluded_image_ids)
    entries: list[dict[str, object]] = []
    category_diagnostics: list[dict[str, object]] = []

    for object_category, metadata in category_rows.items():
        domain_name = str(metadata["expected_domain"])
        profile_name = str(metadata["expected_profile"])
        domain = domain_by_name.get(domain_name)
        if domain is None:
            raise ValueError(f"unknown prompt-bank domain: {domain_name}")
        object_label = object_category.replace("_", " ")
        allowed_semantics, _ = _domain_inventory(
            domain,
            profile_name,
            object_label,
        )
        accepted = 0
        skipped_overlap = 0
        skipped_mapping = 0
        unmapped_names: set[str] = set()
        materialization_errors: list[str] = []
        for case in alternatives.get(object_category, []):
            if accepted >= args.assets_per_category:
                break
            if case.image_id in used_image_ids:
                skipped_overlap += 1
                continue
            mapping, excluded_parts = _part_mapping(
                case.part_categories,
                domain=domain_name,
                object_category=object_category,
                allowed_semantics=allowed_semantics,
            )
            if len(set(mapping.values())) < args.minimum_mapped_parts:
                skipped_mapping += 1
                unmapped_names.update(excluded_parts)
                continue
            ordinal = accepted + 1
            asset_dir = assets_dir / (
                f"{_safe_name(object_category)}_{ordinal:02d}_"
                f"ann{case.object_annotation_id}"
            )
            case_path = asset_dir / "case.json"
            try:
                if case_path.is_file():
                    payload = json.loads(case_path.read_text(encoding="utf-8"))
                    if int(payload.get("object_annotation_id", -1)) != (
                        case.object_annotation_id
                    ):
                        raise ValueError(
                            "existing case has a different object annotation ID"
                        )
                else:
                    payload = materialize_paco_case(
                        case,
                        asset_dir,
                        annotation_sha256=annotation_sha,
                        crop_padding=args.crop_padding,
                        dataset_split=dataset_split,
                    )
            except (OSError, TypeError, ValueError) as error:
                materialization_errors.append(
                    f"object_annotation_id={case.object_annotation_id}: {error}"
                )
                continue
            used_image_ids.add(case.image_id)
            accepted += 1
            unmapped_names.update(excluded_parts)
            aliases = sorted(
                {
                    object_label,
                    *(str(value) for value in metadata.get("aliases", [])),
                }
            )
            entries.append(
                {
                    "asset_id": (
                        f"paco-{dataset_split}-{_safe_name(object_category)}-"
                        f"{case.object_annotation_id}"
                    ),
                    "asset_label": object_label,
                    "asset_domain": domain_name,
                    "asset_profile": profile_name,
                    "aliases": aliases,
                    "paco_case": str(case_path.resolve()),
                    "part_name_mapping": dict(sorted(mapping.items())),
                    "exclude_parts": sorted(excluded_parts),
                    "reviewed": True,
                    "review_source": "PACO-LVIS public human annotation",
                    "ground_truth_role": "offline_prototype_training_only",
                    "image_id": case.image_id,
                    "object_annotation_id": case.object_annotation_id,
                    "mapped_part_semantics": sorted(set(mapping.values())),
                    "materialized_part_count": int(payload["part_count"]),
                }
            )
            print(
                f"{object_category}: {accepted}/{args.assets_per_category} "
                f"image_id={case.image_id} mapped={len(set(mapping.values()))}"
            )
        category_diagnostics.append(
            {
                "object_category": object_category,
                "expected_domain": domain_name,
                "expected_profile": profile_name,
                "accepted_asset_count": accepted,
                "requested_asset_count": args.assets_per_category,
                "skipped_image_overlap_count": skipped_overlap,
                "skipped_insufficient_mapping_count": skipped_mapping,
                "unmapped_part_names": sorted(unmapped_names),
                "materialization_errors": materialization_errors,
            }
        )

    reference_manifest = {
        "format": REFERENCE_FORMAT,
        "format_version": "0.2.0",
        "prompt_bank": str(args.prompt_bank.resolve()),
        "source_dataset": "PACO-LVIS v1",
        "source_split": dataset_split,
        "source_annotations": str(args.annotations.resolve()),
        "source_annotation_sha256": annotation_sha,
        "supervision": "public human-annotated object and part masks",
        "ground_truth_used_for_offline_index_building": True,
        "ground_truth_available_during_query_inference": False,
        "excluded_image_ids": sorted(excluded_image_ids),
        "entries": entries,
    }
    manifest_path = args.output / "reference_manifest.json"
    manifest_path.write_text(
        json.dumps(reference_manifest, indent=2), encoding="utf-8"
    )
    report = {
        "format": "HPID PACO prototype preparation report",
        "format_version": "0.1.0",
        "reference_manifest": str(manifest_path.resolve()),
        "category_count": len(category_rows),
        "asset_count": len(entries),
        "complete_category_count": sum(
            int(row["accepted_asset_count"]) == args.assets_per_category
            for row in category_diagnostics
        ),
        "categories": category_diagnostics,
    }
    (args.output / "preparation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return 0 if entries else 1


if __name__ == "__main__":
    raise SystemExit(main())
