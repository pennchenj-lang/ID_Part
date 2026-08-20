from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from hpid_split.fusion import MaskCandidate
from hpid_split.paco_semantics import (
    canonical_semantic_name,
    normalize_paco_name,
)
from hpid_split.retrieval import (
    CLIPSegEmbeddingEncoder,
    PrototypeIndex,
    PrototypeRetriever,
    RetrievalConfig,
)


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) >= 128


def _label_matches(expected: str, actual: str | None) -> bool:
    if actual is None:
        return False
    return normalize_paco_name(expected) == normalize_paco_name(actual)


def _macro(rows: list[dict[str, object]]) -> dict[str, float]:
    if not rows:
        return {}
    return {
        "acceptance_rate": float(np.mean([bool(row["accepted"]) for row in rows])),
        "accepted_label_accuracy": float(
            np.mean(
                [
                    bool(row["selected_label_correct"])
                    for row in rows
                    if bool(row["accepted"])
                ]
            )
        )
        if any(bool(row["accepted"]) for row in rows)
        else 0.0,
        "correct_retrieval_coverage": float(
            np.mean([bool(row["selected_label_correct"]) for row in rows])
        ),
        "nearest_label_accuracy": float(
            np.mean([bool(row["nearest_label_correct"]) for row in rows])
        ),
        "mean_top_similarity": float(
            np.mean([float(row["top_similarity"]) for row in rows])
        ),
        "mean_part_inventory_recall": float(
            np.mean([float(row["part_inventory_recall"]) for row in rows])
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit a train-only HPID prototype index on independent PACO cases. "
            "Ground-truth object masks are used only to isolate retrieval quality."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--condition",
        choices=("automatic", "prompted", "oracle-profile"),
        default="automatic",
    )
    parser.add_argument("--minimum-similarity", type=float, default=0.58)
    parser.add_argument(
        "--minimum-prompted-similarity", type=float, default=0.32
    )
    parser.add_argument("--minimum-profiled-similarity", type=float, default=0.50)
    parser.add_argument("--profile-bonus", type=float, default=0.035)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    requested = set(args.case)
    cases = [
        row
        for row in manifest.get("cases", [])
        if isinstance(row, dict)
        and row.get("case_path")
        and (not requested or str(row.get("case_id")) in requested)
    ]
    missing = requested - {str(row.get("case_id")) for row in cases}
    if missing:
        raise ValueError(f"unknown cases: {sorted(missing)}")

    index = PrototypeIndex.load(args.index)
    encoder = CLIPSegEmbeddingEncoder(
        model_name=index.encoder_model_name,
        device=args.device,
        local_files_only=args.local_files_only,
        batch_size=16,
    )
    retriever = PrototypeRetriever(
        index,
        encoder,
        config=RetrievalConfig(
            minimum_asset_similarity=args.minimum_similarity,
            minimum_prompted_asset_similarity=args.minimum_prompted_similarity,
            minimum_profiled_asset_similarity=args.minimum_profiled_similarity,
            profile_similarity_bonus=args.profile_bonus,
        ),
    )

    rows: list[dict[str, object]] = []
    by_domain: dict[str, list[dict[str, object]]] = defaultdict(list)
    for case_row in cases:
        case_path = Path(str(case_row["case_path"]))
        case = json.loads(case_path.read_text(encoding="utf-8"))
        image = Image.open(case_path.parent / "source_crop.png").convert("RGB")
        root_mask = _load_mask(case_path.parent / "object_mask_crop.png")
        expected_domain = str(case_row["expected_domain"])
        expected_profile = str(case_row.get("expected_profile") or "")
        category = str(case_row.get("object_category") or case["object_category"])
        profile_metadata = (
            {
                "selected_part_profile": expected_profile,
                "profile_hint_source": (
                    "user_asset_prompt"
                    if args.condition == "prompted"
                    else "oracle_profile_audit"
                ),
                "profile_resolution_status": "accepted",
            }
            if args.condition in {"prompted", "oracle-profile"}
            and expected_profile
            else {}
        )
        root = MaskCandidate(
            semantic_name=expected_domain,
            semantic_parent=expected_domain,
            mask=root_mask,
            score=1.0,
            source="paco/retrieval-audit-root",
            metadata={
                "root_origin": "paco-retrieval-audit",
                "root_index": str(case_row["case_id"]),
                **profile_metadata,
                "ground_truth_object_mask_used_for_retrieval_audit": True,
            },
        )
        result = retriever.query(
            image,
            [root],
            asset_hint=(category if args.condition == "prompted" else None),
        )
        plan = result.plans[0]
        nearest_label = (
            str(plan.nearest_assets[0]["asset_label"])
            if plan.nearest_assets
            else None
        )
        truth_semantics = {
            semantic
            for part in case.get("parts", [])
            if (
                semantic := canonical_semantic_name(
                    str(part["part_name"]),
                    expected_domain,
                    object_category=category,
                )
            )
            is not None
        }
        retrieved_semantics = {
            prior.output_semantic_name for prior in plan.part_priors
        }
        inventory_recall = len(truth_semantics & retrieved_semantics) / max(
            1, len(truth_semantics)
        )
        row = {
            "case_id": str(case_row["case_id"]),
            "object_category": category,
            "expected_domain": expected_domain,
            "expected_profile": expected_profile,
            "accepted": plan.accepted,
            "reason": plan.reason,
            "top_similarity": plan.top_similarity,
            "selected_asset_label": plan.asset_label,
            "selected_asset_domain": plan.asset_domain,
            "selected_label_correct": _label_matches(category, plan.asset_label),
            "nearest_asset_label": nearest_label,
            "nearest_label_correct": _label_matches(category, nearest_label),
            "supporting_asset_count": plan.supporting_asset_count,
            "truth_semantic_count": len(truth_semantics),
            "retrieved_semantic_count": len(retrieved_semantics),
            "part_inventory_recall": inventory_recall,
            "nearest_assets": list(plan.nearest_assets),
        }
        rows.append(row)
        by_domain[expected_domain].append(row)
        print(
            f"{row['case_id']}: accepted={plan.accepted} "
            f"label={plan.asset_label} similarity={plan.top_similarity:.4f} "
            f"inventory_recall={inventory_recall:.4f}",
            flush=True,
        )

    payload = {
        "format": "HPID PACO prototype retrieval audit",
        "format_version": "0.1.0",
        "condition": args.condition,
        "source_manifest": str(args.manifest.resolve()),
        "prototype_index": str(args.index.resolve()),
        "ground_truth_object_mask_used": True,
        "ground_truth_part_masks_used_during_retrieval": False,
        "ground_truth_part_names_used_only_for_post_retrieval_inventory_audit": True,
        "expected_profile_available_to_retrieval": args.condition
        in {"prompted", "oracle-profile"},
        "oracle_profile_condition": args.condition == "oracle-profile",
        "aggregate": _macro(rows),
        "by_domain": {
            domain: _macro(domain_rows)
            for domain, domain_rows in sorted(by_domain.items())
        },
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
