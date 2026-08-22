from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from analyze_cross_domain_fusion_ablation import _load_candidates
from analyze_public_baselines import (
    _exclusive_ownership,
    _grounded_predictions,
    _load_sam_cache,
    _nms,
)
from PIL import Image

from hpid_split.fusion import FusionConfig, fuse_candidates
from hpid_split.paper_audit import (
    overlap_excess_fraction,
    unassigned_root_fraction,
)
from hpid_split.validation import validate_package

SEED = 20260822
EXPECTED_CASES = 226
METHODS = (
    "sam2_raw",
    "sam2_nms",
    "sam2_max_ownership",
    "grounded_sam2_same_inventory",
    "clipseg_ovparts_style",
    "hpid_split_a3",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) >= 128


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _public_part_masks(package_dir: Path) -> list[np.ndarray]:
    rows = json.loads((package_dir / "parts.json").read_text(encoding="utf-8"))
    return [
        _load_mask(package_dir / str(row["mask_visible_path"])) for row in rows
    ]


def _package_relations(package_dir: Path) -> dict[str, int]:
    parts = json.loads((package_dir / "parts.json").read_text(encoding="utf-8"))
    groups = json.loads((package_dir / "groups.json").read_text(encoding="utf-8"))
    part_ids = [str(row.get("part_id", "")) for row in parts]
    group_ids = [str(row.get("group_id", "")) for row in groups]
    known_parts = set(part_ids)
    known_groups = set(group_ids)
    invalid_parents = sum(
        row.get("assembly_parent_id") is not None
        and str(row.get("assembly_parent_id")) not in known_parts
        for row in parts
    )
    unresolved_part_groups = sum(
        str(row.get("group_id", "")) not in known_groups for row in parts
    )
    invalid_group_members = sum(
        not set(map(str, row.get("member_part_ids", []))) <= known_parts
        for row in groups
    )
    return {
        "duplicate_part_id_count": len(part_ids) - len(set(part_ids)),
        "duplicate_group_id_count": len(group_ids) - len(set(group_ids)),
        "invalid_parent_reference_count": int(invalid_parents),
        "unresolved_part_group_count": int(unresolved_part_groups),
        "invalid_group_membership_count": int(invalid_group_members),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit exclusive ownership and HPID package invariants."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--sam-cache", type=Path, required=True)
    parser.add_argument("--baseline-cases", type=Path, required=True)
    parser.add_argument("--clipseg-evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nms-iou", type=float, default=0.80)
    parser.add_argument("--nms-containment", type=float, default=0.92)
    parser.add_argument("--permutation-count", type=int, default=3)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    manifest_cases = {
        str(row["case_id"]): row
        for row in manifest.get("cases", [])
        if row.get("case_path")
    }
    if len(manifest_cases) != EXPECTED_CASES:
        raise RuntimeError(
            f"manifest cases={len(manifest_cases)}, expected {EXPECTED_CASES}"
        )
    baseline_rows = _read_csv(args.baseline_cases)
    baseline_index = {
        (str(row["case_id"]), str(row["method"])): row for row in baseline_rows
    }
    expected_baseline_pairs = {
        (case_id, method)
        for case_id in manifest_cases
        for method in METHODS
        if method != "clipseg_ovparts_style"
    }
    if set(baseline_index) != expected_baseline_pairs:
        raise RuntimeError("public baseline case coverage is incomplete or duplicated")
    clipseg_payload = json.loads(
        args.clipseg_evaluation.read_text(encoding="utf-8")
    )
    clipseg_index = {
        str(row["case_id"]): row for row in clipseg_payload.get("cases", [])
    }
    if set(clipseg_index) != set(manifest_cases):
        raise RuntimeError("CLIPSeg and manifest case identities differ")

    metric_rows: list[dict[str, object]] = []
    package_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for case_number, case_id in enumerate(sorted(manifest_cases), start=1):
        manifest_case = manifest_cases[case_id]
        expected_domain = str(manifest_case["expected_domain"])
        case_path = Path(str(manifest_case["case_path"]))
        case = json.loads(case_path.read_text(encoding="utf-8"))
        object_category = str(case["object_category"])
        root = _load_mask(case_path.parent / "object_mask_crop.png")
        package_dir = args.benchmark_root / case_id

        raw_masks, raw_scores = _load_sam_cache(
            args.sam_cache / "cases" / f"{case_id}.npz"
        )
        raw_semantics: list[str | None] = [None] * len(raw_masks)
        nms_masks, nms_scores, nms_semantics = _nms(
            raw_masks,
            raw_scores,
            raw_semantics,
            iou_threshold=args.nms_iou,
            containment_threshold=args.nms_containment,
            semantic_aware=False,
        )
        owned_masks, _owned_semantics = _exclusive_ownership(
            nms_masks, nms_scores, nms_semantics
        )
        grounded_masks, _grounded_semantics = _grounded_predictions(
            package_dir,
            expected_domain,
            object_category,
            nms_iou=args.nms_iou,
            nms_containment=args.nms_containment,
        )
        hpid_masks = _public_part_masks(package_dir)
        mask_sets = {
            "sam2_raw": raw_masks,
            "sam2_nms": nms_masks,
            "sam2_max_ownership": owned_masks,
            "grounded_sam2_same_inventory": grounded_masks,
            "hpid_split_a3": hpid_masks,
        }
        for method, masks in mask_sets.items():
            overlap = overlap_excess_fraction(masks, root)
            metric_rows.append(
                {
                    "case_id": case_id,
                    "method": method,
                    "overlap_excess_root_fraction": overlap,
                    "unassigned_root_fraction": unassigned_root_fraction(
                        masks, root
                    ),
                    "exclusive_ownership": overlap <= 1e-12,
                    "native_persistent_part_ids": method == "hpid_split_a3",
                    "native_hierarchy_metadata": method == "hpid_split_a3",
                    "native_group_metadata": method == "hpid_split_a3",
                    "native_versioned_package": method == "hpid_split_a3",
                }
            )
        clipseg = clipseg_index[case_id]
        clipseg_overlap = float(clipseg["overlap_root_fraction"])
        metric_rows.append(
            {
                "case_id": case_id,
                "method": "clipseg_ovparts_style",
                "overlap_excess_root_fraction": clipseg_overlap,
                "unassigned_root_fraction": 1.0
                - float(clipseg["object_recall"]),
                "exclusive_ownership": clipseg_overlap <= 1e-12,
                "native_persistent_part_ids": False,
                "native_hierarchy_metadata": False,
                "native_group_metadata": False,
                "native_versioned_package": False,
            }
        )

        validation = validate_package(package_dir)
        evaluator_sidecar_error = (
            "payload file is absent from manifest: paco_evaluation.json"
        )
        payload_errors = [
            error
            for error in validation["errors"]
            if error != evaluator_sidecar_error
        ]
        relation_counts = _package_relations(package_dir)
        stored_map = np.asarray(
            Image.open(package_dir / "part_id_map.tiff"), dtype=np.uint16
        )
        candidates = _load_candidates(package_dir)
        canonical = fuse_candidates(
            candidates,
            image_shape=stored_map.shape,
            config=FusionConfig(),
        )
        canonical_exact = bool(np.array_equal(canonical.instance_map, stored_map))
        order_matches = 0
        foreground_matches = 0
        for permutation_index in range(args.permutation_count):
            generator = np.random.default_rng(
                SEED + case_number * 100 + permutation_index
            )
            shuffled = list(candidates)
            generator.shuffle(shuffled)
            permuted = fuse_candidates(
                shuffled,
                image_shape=stored_map.shape,
                config=FusionConfig(),
            )
            order_matches += int(
                np.array_equal(permuted.instance_map, canonical.instance_map)
            )
            foreground_matches += int(
                np.array_equal(
                    permuted.instance_map > 0,
                    canonical.instance_map > 0,
                )
            )
        package_rows.append(
            {
                "case_id": case_id,
                "raw_directory_valid": bool(validation["valid"]),
                "export_payload_valid": not payload_errors,
                "validation_error_count": len(validation["errors"]),
                "export_payload_error_count": len(payload_errors),
                "posthoc_evaluation_sidecar_present": (
                    evaluator_sidecar_error in validation["errors"]
                ),
                "checked_part_count": int(validation["checked_parts"]),
                "checked_group_count": int(validation["checked_groups"]),
                "canonical_rerun_exact_map": canonical_exact,
                "candidate_order_exact_matches": order_matches,
                "candidate_order_foreground_matches": foreground_matches,
                "candidate_order_trials": args.permutation_count,
                **relation_counts,
            }
        )
        print(f"[{case_number}/{EXPECTED_CASES}] {case_id}", flush=True)

    summary_rows: list[dict[str, object]] = []
    for method in METHODS:
        selected = [row for row in metric_rows if row["method"] == method]
        if len(selected) != EXPECTED_CASES:
            raise RuntimeError(f"identity rows for {method} are incomplete")
        summary_rows.append(
            {
                "method": method,
                "case_count": len(selected),
                "mean_overlap_excess_root_fraction": float(
                    np.mean(
                        [float(row["overlap_excess_root_fraction"]) for row in selected]
                    )
                ),
                "mean_unassigned_root_fraction": float(
                    np.mean(
                        [float(row["unassigned_root_fraction"]) for row in selected]
                    )
                ),
                "exclusive_ownership_case_rate": float(
                    np.mean([bool(row["exclusive_ownership"]) for row in selected])
                ),
                "native_persistent_part_ids": method == "hpid_split_a3",
                "native_hierarchy_metadata": method == "hpid_split_a3",
                "native_group_metadata": method == "hpid_split_a3",
                "native_versioned_package": method == "hpid_split_a3",
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "identity_representation_cases.csv", metric_rows)
    _write_csv(args.output / "identity_representation_summary.csv", summary_rows)
    _write_csv(args.output / "hpid_package_audit_cases.csv", package_rows)
    package_summary = {
        "case_count": EXPECTED_CASES,
        "raw_directory_valid_count": sum(
            bool(row["raw_directory_valid"]) for row in package_rows
        ),
        "export_payload_valid_count": sum(
            bool(row["export_payload_valid"]) for row in package_rows
        ),
        "export_payload_valid_rate": float(
            np.mean([bool(row["export_payload_valid"]) for row in package_rows])
        ),
        "total_raw_directory_validation_errors": sum(
            int(row["validation_error_count"]) for row in package_rows
        ),
        "total_export_payload_errors": sum(
            int(row["export_payload_error_count"]) for row in package_rows
        ),
        "posthoc_evaluation_sidecar_count": sum(
            bool(row["posthoc_evaluation_sidecar_present"])
            for row in package_rows
        ),
        "total_duplicate_part_ids": sum(
            int(row["duplicate_part_id_count"]) for row in package_rows
        ),
        "total_duplicate_group_ids": sum(
            int(row["duplicate_group_id_count"]) for row in package_rows
        ),
        "total_invalid_parent_references": sum(
            int(row["invalid_parent_reference_count"]) for row in package_rows
        ),
        "total_unresolved_part_groups": sum(
            int(row["unresolved_part_group_count"]) for row in package_rows
        ),
        "total_invalid_group_memberships": sum(
            int(row["invalid_group_membership_count"]) for row in package_rows
        ),
        "canonical_rerun_exact_map_rate": float(
            np.mean([bool(row["canonical_rerun_exact_map"]) for row in package_rows])
        ),
        "candidate_order_exact_map_rate": sum(
            int(row["candidate_order_exact_matches"]) for row in package_rows
        )
        / (EXPECTED_CASES * args.permutation_count),
        "candidate_order_foreground_rate": sum(
            int(row["candidate_order_foreground_matches"]) for row in package_rows
        )
        / (EXPECTED_CASES * args.permutation_count),
    }
    report = {
        "format": "HPID-Split identity representation and export audit",
        "format_version": "1.0.0",
        "case_count": EXPECTED_CASES,
        "methods": list(METHODS),
        "package_audit": package_summary,
        "permutation_count_per_case": args.permutation_count,
        "elapsed_seconds": time.perf_counter() - started,
        "metric_scope": {
            "overlap_excess_root_fraction": (
                "Repeated mask ownership divided by supplied-root pixels."
            ),
            "unassigned_root_fraction": (
                "Supplied-root pixels not covered by any public prediction mask."
            ),
            "native_capabilities": (
                "A dash must be reported for baselines that do not emit a persistent "
                "identity package; absence is not scored as segmentation failure."
            ),
            "candidate_order_consistency": (
                "Exact reruns after fixed candidate-list permutations; this is an "
                "implementation-invariance audit, not stochastic model determinism."
            ),
            "export_payload_validation": (
                "The algorithm-authored manifest payload is validated separately "
                "from paco_evaluation.json, a post-inference evaluator sidecar added "
                "to every case directory and intentionally absent from that manifest."
            ),
        },
        "ground_truth_used_in_prediction": False,
        "manifest_sha256": _sha256(args.manifest),
        "baseline_cases_sha256": _sha256(args.baseline_cases),
        "clipseg_evaluation_sha256": _sha256(args.clipseg_evaluation),
    }
    (args.output / "identity_layer_audit_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
