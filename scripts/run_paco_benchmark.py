from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from hpid_split.paco_eval import evaluate_paco_package


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    successful = [row for row in rows if row.get("return_code") == 0]
    metric_names = (
        "object_iou",
        "object_precision",
        "object_recall",
        "part_discovery_precision_at_025",
        "part_discovery_recall_at_025",
        "part_discovery_f1_at_025",
        "mean_matched_iou",
        "mean_matched_boundary_f1",
        "semantic_part_recall",
        "oversegmentation_ratio",
    )
    metrics = {
        name: float(np.mean([float(row[name]) for row in successful]))
        if successful
        else 0.0
        for name in metric_names
    }
    return {
        "case_count": len(rows),
        "successful_case_count": len(successful),
        "failed_case_count": len(rows) - len(successful),
        "domain_accuracy": (
            sum(bool(row["domain_correct"]) for row in successful) / len(successful)
            if successful
            else 0.0
        ),
        "profile_accuracy": (
            sum(bool(row["profile_correct"]) for row in successful)
            / max(
                1,
                sum(row.get("profile_correct") is not None for row in successful),
            )
        ),
        "macro_metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run and evaluate the fixed HPID PACO modern-object suite."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-bank", type=Path, required=True)
    parser.add_argument(
        "--retrieval-index",
        type=Path,
        help=(
            "Optional train-only visual prototype index. Benchmark ground-truth "
            "part masks remain unavailable during inference."
        ),
    )
    parser.add_argument(
        "--dense-semantic-model",
        default="CIDAS/clipseg-rd64-refined",
        help="Conditional region-text model used by the fixed inference pipeline.",
    )
    parser.add_argument(
        "--asset-router-index",
        type=Path,
        help="Train-only SigLIP 2 asset routing index used in automatic mode.",
    )
    parser.add_argument(
        "--asset-router-model",
        default="",
        help="Optional local SigLIP 2 model path for asset routing.",
    )
    parser.add_argument(
        "--routing-condition",
        choices=("automatic", "known-domain"),
        default="automatic",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--visual-crop-layers", type=int, default=1)
    parser.add_argument("--visual-points-per-crop", type=int, default=18)
    parser.add_argument(
        "--florence-parts",
        action="store_true",
        help="Enable the root-constrained Florence-2 retrieval supplement.",
    )
    parser.add_argument(
        "--additional-grounding-model",
        action="append",
        default=[],
        help="Additional detector used by the genuine sequential ensemble condition.",
    )
    parser.add_argument("--no-profile-refinement", action="store_true")
    parser.add_argument(
        "--asset-prompt-from-category",
        action="store_true",
        help=(
            "Pass the public PACO object category as an object-level prompt. "
            "This is a prompted condition, not automatic object retrieval."
        ),
    )
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reuse successful cases from an existing benchmark_summary.json. "
            "The fixed manifest and inference configuration must match."
        ),
    )
    parser.add_argument(
        "--force-case",
        action="append",
        default=[],
        help="Rerun this case even when --resume finds a valid saved result.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("format") != "HPID PACO modern-object subset":
        raise ValueError("--manifest is not an HPID PACO modern-object subset")
    requested = set(args.case)
    cases = [
        row
        for row in manifest.get("cases", [])
        if "case_path" in row and (not requested or str(row["case_id"]) in requested)
    ]
    missing = requested - {str(row["case_id"]) for row in cases}
    if missing:
        raise ValueError(f"unknown benchmark cases: {sorted(missing)}")
    forced = set(args.force_case)
    missing_forced = forced - {str(row["case_id"]) for row in cases}
    if missing_forced:
        raise ValueError(f"unknown forced benchmark cases: {sorted(missing_forced)}")

    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    summary_path = args.output / "benchmark_summary.json"
    resumed_rows: dict[str, dict[str, object]] = {}
    if args.resume and summary_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        expected_resume_fields = {
            "source_manifest_sha256": _sha256(args.manifest),
            "prompt_bank_sha256": _sha256(args.prompt_bank),
            "retrieval_index_manifest_sha256": (
                _sha256(args.retrieval_index / "index.json")
                if args.retrieval_index is not None
                else None
            ),
            "dense_semantic_model": args.dense_semantic_model,
            "asset_router_index_manifest_sha256": (
                _sha256(args.asset_router_index / "index.json")
                if args.asset_router_index is not None
                else None
            ),
            "asset_router_model": args.asset_router_model.strip() or None,
            "routing_condition": args.routing_condition,
            "asset_prompt_condition": bool(args.asset_prompt_from_category),
            "profile_refinement_enabled": not args.no_profile_refinement,
            "additional_grounding_models": list(args.additional_grounding_model),
            "florence_parts_enabled": bool(args.florence_parts),
        }
        mismatches = {
            key: {"previous": previous.get(key), "requested": value}
            for key, value in expected_resume_fields.items()
            if previous.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "--resume configuration does not match the existing benchmark: "
                f"{mismatches}"
            )
        for row in previous.get("cases", []):
            case_id = str(row.get("case_id", ""))
            output_dir = args.output / case_id
            if (
                case_id
                and row.get("return_code") == 0
                and (output_dir / "package_manifest.json").is_file()
                and (output_dir / "paco_evaluation.json").is_file()
            ):
                resumed_rows[case_id] = row
    for case in cases:
        case_id = str(case["case_id"])
        case_path = Path(str(case["case_path"]))
        image_path = case_path.parent / "source_crop.png"
        output_dir = args.output / case_id
        if case_id in resumed_rows and case_id not in forced:
            rows.append(resumed_rows[case_id])
            print(f"{case_id}: resumed", flush=True)
            continue
        command = [
            sys.executable,
            "-m",
            "hpid_split.cli",
            "auto",
            "--image",
            str(image_path),
            "--output",
            str(output_dir),
            "--prompt-bank",
            str(args.prompt_bank),
            "--dense-semantic-fallback",
            "--dense-semantic-model",
            args.dense_semantic_model,
            "--decomposition-mode",
            "automatic",
            "--root-mode",
            "primary",
            "--visual-crop-layers",
            str(args.visual_crop_layers),
            "--visual-points-per-crop",
            str(args.visual_points_per_crop),
            "--device",
            args.device,
        ]
        if args.routing_condition == "known-domain":
            command.extend(["--domains", str(case["expected_domain"])])
        if args.retrieval_index is not None:
            command.extend(["--retrieval-index", str(args.retrieval_index)])
        if args.asset_router_index is not None:
            command.extend(["--asset-router-index", str(args.asset_router_index)])
            if args.asset_router_model.strip():
                command.extend(
                    ["--asset-router-model", args.asset_router_model.strip()]
                )
        if args.florence_parts:
            if args.retrieval_index is None:
                raise ValueError("--florence-parts requires --retrieval-index")
            command.append("--florence-parts")
        asset_prompt = None
        if args.asset_prompt_from_category:
            asset_prompt = re.sub(
                r"\s+",
                " ",
                re.sub(r"[()_]+", " ", str(case["object_category"])),
            ).strip()
            command.extend(["--asset-prompt", asset_prompt])
        if args.local_files_only:
            command.append("--local-files-only")
        for grounding_model in args.additional_grounding_model:
            command.extend(["--additional-grounding-model", grounding_model])
        if args.no_profile_refinement:
            command.append("--no-profile-refinement")
        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )
        row: dict[str, object] = {
            "case_id": case_id,
            "object_category": case.get("object_category"),
            "expected_domain": case["expected_domain"],
            "expected_profile": case.get("expected_profile"),
            "routing_condition": args.routing_condition,
            "crop_uses_ground_truth_bbox": True,
            "part_masks_available_to_inference": False,
            "asset_prompt": asset_prompt,
            "asset_prompt_condition": bool(args.asset_prompt_from_category),
            "return_code": completed.returncode,
            "command": command,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-4000:],
        }
        if completed.returncode == 0:
            evaluation = evaluate_paco_package(
                output_dir,
                case_path,
                expected_domain=str(case["expected_domain"]),
                expected_profile=(
                    str(case["expected_profile"])
                    if case.get("expected_profile") is not None
                    else None
                ),
            )
            (output_dir / "paco_evaluation.json").write_text(
                json.dumps(evaluation, indent=2), encoding="utf-8"
            )
            row.update(
                {
                    key: value
                    for key, value in evaluation.items()
                    if key
                    not in {
                        "format",
                        "format_version",
                        "package",
                        "case",
                        "matches",
                        "semantic_matches",
                    }
                }
            )
        rows.append(row)
        summary = {
            "format": "HPID PACO modern-object benchmark",
            "format_version": "0.1.0",
            "source_manifest": str(args.manifest.resolve()),
            "source_manifest_sha256": _sha256(args.manifest),
            "prompt_bank": str(args.prompt_bank.resolve()),
            "prompt_bank_sha256": _sha256(args.prompt_bank),
            "retrieval_index": (
                str(args.retrieval_index.resolve())
                if args.retrieval_index is not None
                else None
            ),
            "retrieval_index_manifest_sha256": (
                _sha256(args.retrieval_index / "index.json")
                if args.retrieval_index is not None
                else None
            ),
            "dense_semantic_model": args.dense_semantic_model,
            "asset_router_index": (
                str(args.asset_router_index.resolve())
                if args.asset_router_index is not None
                else None
            ),
            "asset_router_index_manifest_sha256": (
                _sha256(args.asset_router_index / "index.json")
                if args.asset_router_index is not None
                else None
            ),
            "asset_router_model": args.asset_router_model.strip() or None,
            "routing_condition": args.routing_condition,
            "asset_prompt_condition": bool(args.asset_prompt_from_category),
            "profile_refinement_enabled": not args.no_profile_refinement,
            "additional_grounding_models": list(args.additional_grounding_model),
            "florence_parts_enabled": bool(args.florence_parts),
            "visual_crop_layers": args.visual_crop_layers,
            "visual_points_per_crop": args.visual_points_per_crop,
            "crop_uses_ground_truth_bbox": True,
            "part_masks_available_to_inference": False,
            "evaluation_reads_ground_truth_after_inference": True,
            "aggregate": _aggregate(rows),
            "cases": rows,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if completed.returncode == 0:
            print(
                f"{case_id}: domain={row['selected_domain']} "
                f"object_iou={row['object_iou']:.4f} "
                f"part_f1={row['part_discovery_f1_at_025']:.4f} "
                f"semantic_recall={row['semantic_part_recall']:.4f}",
                flush=True,
            )
        else:
            print(f"{case_id}: failed return_code={completed.returncode}", flush=True)
    return 0 if all(row["return_code"] == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
