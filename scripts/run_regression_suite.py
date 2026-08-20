from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _read_part_rows(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("parts", [])
    if not isinstance(payload, list):
        raise TypeError(f"unexpected parts payload in {path}")
    return [row for row in payload if isinstance(row, dict)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one fixed HPID configuration over an audited local suite."
    )
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-bank", type=Path, required=True)
    parser.add_argument("--retrieval-index", type=Path)
    parser.add_argument(
        "--dense-semantic-model",
        default="CIDAS/clipseg-rd64-refined",
    )
    parser.add_argument("--asset-router-index", type=Path)
    parser.add_argument("--asset-router-model", default="")
    parser.add_argument(
        "--ontology-router-model",
        default="",
        help="Optional local image-text model for scene ontology consensus.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--visual-crop-layers", type=int, default=1)
    parser.add_argument("--visual-points-per-crop", type=int, default=20)
    parser.add_argument("--vlm-model")
    parser.add_argument("--vlm-load-in-4bit", action="store_true")
    parser.add_argument(
        "--vlm-query-mode",
        choices=("bulk", "per-semantic"),
        default="per-semantic",
    )
    parser.add_argument("--vlm-maximum-queries", type=int, default=12)
    parser.add_argument("--vlm-maximum-total-queries", type=int, default=24)
    parser.add_argument("--vlm-maximum-roots", type=int, default=8)
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if audit.get("format") != "HPID local regression suite audit":
        raise ValueError("--audit is not an HPID local regression suite audit")
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for case in audit.get("cases", []):
        case_id = str(case["case_id"])
        output_dir = args.output / case_id
        command = [
            sys.executable,
            "-m",
            "hpid_split.cli",
            "auto",
            "--image",
            str(case["image"]),
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
            str(case.get("root_mode", "primary")),
            "--visual-crop-layers",
            str(args.visual_crop_layers),
            "--visual-points-per-crop",
            str(args.visual_points_per_crop),
            "--device",
            args.device,
        ]
        if args.retrieval_index is not None:
            command.extend(["--retrieval-index", str(args.retrieval_index)])
        if args.asset_router_index is not None:
            command.extend(["--asset-router-index", str(args.asset_router_index)])
            if args.asset_router_model.strip():
                command.extend(
                    ["--asset-router-model", args.asset_router_model.strip()]
                )
        if args.ontology_router_model.strip():
            command.extend(
                ["--ontology-router-model", args.ontology_router_model.strip()]
            )
        if str(case.get("asset_prompt", "")).strip():
            command.extend(["--asset-prompt", str(case["asset_prompt"]).strip()])
        if case.get("target_point_xy") is not None:
            point = case["target_point_xy"]
            if not isinstance(point, list) or len(point) != 2:
                raise ValueError(f"{case_id}: target_point_xy must be [x, y]")
            command.extend(["--target-point", str(point[0]), str(point[1])])
        if args.local_files_only:
            command.append("--local-files-only")
        if args.vlm_model:
            command.extend(
                [
                    "--vlm-parts",
                    "--vlm-model",
                    args.vlm_model,
                    "--vlm-query-mode",
                    args.vlm_query_mode,
                    "--vlm-maximum-queries",
                    str(args.vlm_maximum_queries),
                    "--vlm-maximum-total-queries",
                    str(args.vlm_maximum_total_queries),
                    "--vlm-maximum-roots",
                    str(args.vlm_maximum_roots),
                ]
            )
            if args.vlm_load_in_4bit:
                command.append("--vlm-load-in-4bit")
        completed = subprocess.run(
            command, text=True, capture_output=True, check=False
        )
        row: dict[str, object] = {
            "case_id": case_id,
            "image_sha256": case["image_sha256"],
            "evidence": case.get("evidence", "qualitative_only"),
            "return_code": completed.returncode,
            "status_line": completed.stdout.strip().splitlines()[-1]
            if completed.stdout.strip()
            else "",
            "stderr_tail": completed.stderr[-2000:],
        }
        if completed.returncode == 0:
            diagnostics = json.loads(
                (output_dir / "inference_diagnostics.json").read_text(
                    encoding="utf-8"
                )
            )
            parts = _read_part_rows(output_dir / "parts.json")
            consensus = (
                diagnostics.get("visual_region_generation", {})
                .get("multi_view_consensus", {})
            )
            retrieval = diagnostics.get("prototype_retrieval") or {}
            vlm = diagnostics.get("vlm_part_generation") or {}
            row.update(
                {
                    "part_count": len(parts),
                    "generic_part_count": sum(
                        "_visual_" in str(part.get("semantic_name", ""))
                        for part in parts
                    ),
                    "raw_visual_proposal_count": consensus.get(
                        "raw_proposal_count"
                    ),
                    "consolidated_visual_proposal_count": consensus.get(
                        "consolidated_proposal_count"
                    ),
                    "duplicate_view_proposals_removed": consensus.get(
                        "duplicate_view_proposals_removed"
                    ),
                    "retrieval_accepted_root_count": retrieval.get(
                        "accepted_root_count"
                    ),
                    "vlm_candidate_count": vlm.get("candidate_count"),
                    "vlm_processed_root_count": vlm.get("root_count"),
                }
            )
        rows.append(row)
        (args.output / "regression_summary.json").write_text(
            json.dumps(
                {
                    "format": "HPID local regression run",
                    "source_audit": str(args.audit.resolve()),
                    "cases": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"{case_id}: return_code={completed.returncode}", flush=True)
    return 0 if all(row["return_code"] == 0 for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
