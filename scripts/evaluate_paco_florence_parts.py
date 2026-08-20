from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from hpid_split.florence_parts import FlorencePartConfig, FlorencePartGenerator
from hpid_split.fusion import MaskCandidate
from hpid_split.metrics import binary_iou, boundary_f1
from hpid_split.paco_semantics import canonical_part_token, normalize_paco_name
from hpid_split.retrieval import (
    CLIPSegEmbeddingEncoder,
    PrototypeIndex,
    PrototypeRetriever,
    RetrievalConfig,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) >= 128


def _asset_hint(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[()_]+", " ", value)).strip()


def _match_metrics(
    truth_masks: list[np.ndarray],
    candidates: list[MaskCandidate],
    *,
    threshold: float,
) -> dict[str, object]:
    predicted = [candidate.mask.astype(bool) for candidate in candidates]
    matrix = np.zeros((len(truth_masks), len(predicted)), dtype=np.float32)
    for row, truth in enumerate(truth_masks):
        for column, prediction in enumerate(predicted):
            matrix[row, column] = binary_iou(truth, prediction)
    matches: list[tuple[int, int, float]] = []
    if matrix.size:
        rows, columns = linear_sum_assignment(1.0 - matrix)
        matches = [
            (int(row), int(column), float(matrix[row, column]))
            for row, column in zip(rows, columns, strict=True)
        ]
    accepted = [match for match in matches if match[2] >= threshold]
    precision = len(accepted) / max(1, len(predicted))
    recall = len(accepted) / max(1, len(truth_masks))
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "prediction_count": len(predicted),
        "matched_count": len(accepted),
        "precision_at_025": precision,
        "recall_at_025": recall,
        "f1_at_025": f1,
        "mean_matched_iou": (
            float(np.mean([item[2] for item in accepted])) if accepted else 0.0
        ),
        "mean_matched_boundary_f1": (
            float(
                np.mean(
                    [
                        boundary_f1(
                            predicted[prediction_index],
                            truth_masks[truth_index],
                            tolerance=3,
                        )
                        for truth_index, prediction_index, _ in accepted
                    ]
                )
            )
            if accepted
            else 0.0
        ),
        "matches": [
            {
                "truth_index": truth_index,
                "prediction_index": prediction_index,
                "iou": overlap,
                "accepted": overlap >= threshold,
            }
            for truth_index, prediction_index, overlap in matches
        ],
    }


def _semantic_recall(
    truth_rows: list[dict[str, object]],
    truth_masks: list[np.ndarray],
    candidates: list[MaskCandidate],
    *,
    expected_domain: str,
    object_category: str,
    threshold: float,
) -> tuple[float, list[dict[str, object]]]:
    truth_tokens = [
        canonical_part_token(
            str(row["part_name"]),
            expected_domain,
            object_category=object_category,
        )
        for row in truth_rows
    ]
    predicted_tokens = [
        normalize_paco_name(candidate.semantic_name).removeprefix(
            f"{expected_domain}_"
        )
        for candidate in candidates
    ]
    accepted = 0
    rows: list[dict[str, object]] = []
    for truth_index, truth_token in enumerate(truth_tokens):
        best_iou = 0.0
        best_prediction = None
        for prediction_index, prediction_token in enumerate(predicted_tokens):
            if prediction_token != truth_token:
                continue
            overlap = binary_iou(
                truth_masks[truth_index],
                candidates[prediction_index].mask.astype(bool),
            )
            if overlap > best_iou:
                best_iou = overlap
                best_prediction = prediction_index
        matched = best_iou >= threshold
        accepted += int(matched)
        rows.append(
            {
                "truth_part": truth_token,
                "best_same_semantic_iou": best_iou,
                "accepted": matched,
                "prediction_index": best_prediction,
            }
        )
    return accepted / max(1, len(truth_rows)), rows


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    successful = [row for row in rows if row.get("status") == "ok"]
    names = (
        "raw_f1_at_025",
        "raw_mean_matched_iou",
        "raw_mean_matched_boundary_f1",
        "reranked_f1_at_025",
        "reranked_mean_matched_iou",
        "reranked_mean_matched_boundary_f1",
        "semantic_recall_at_025",
    )
    return {
        "case_count": len(rows),
        "successful_case_count": len(successful),
        "failed_case_count": len(rows) - len(successful),
        **{
            name: (
                float(np.mean([float(row[name]) for row in successful]))
                if successful
                else 0.0
            )
            for name in names
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Isolate Florence-2 part proposal quality on PACO modern objects."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--retrieval-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--phrases-per-part", type=int, default=2)
    parser.add_argument("--maximum-part-prompts", type=int, default=20)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    requested = set(args.case)
    cases = [
        row
        for row in manifest.get("cases", [])
        if "case_path" in row and (not requested or str(row["case_id"]) in requested)
    ]
    missing = requested - {str(row["case_id"]) for row in cases}
    if missing:
        raise ValueError(f"unknown cases: {sorted(missing)}")

    index = PrototypeIndex.load(args.retrieval_index)
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
            maximum_part_prompts=args.maximum_part_prompts,
        ),
    )
    florence = FlorencePartGenerator(
        device=args.device,
        config=FlorencePartConfig(
            local_files_only=args.local_files_only,
            phrases_per_part=args.phrases_per_part,
        ),
    )

    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    summary_path = args.output / "summary.json"
    for case in cases:
        case_id = str(case["case_id"])
        case_path = Path(str(case["case_path"]))
        case_dir = case_path.parent
        case_payload = json.loads(case_path.read_text(encoding="utf-8"))
        expected_domain = str(case.get("expected_domain") or "").strip()
        if not expected_domain:
            rows.append(
                {
                    "case_id": case_id,
                    "object_category": case.get("object_category"),
                    "status": "skipped_missing_domain_mapping",
                }
            )
            continue
        image = Image.open(case_dir / "source_crop.png").convert("RGB")
        root_mask = _load_mask(case_dir / "object_mask_crop.png")
        root = MaskCandidate(
            expected_domain,
            expected_domain,
            root_mask,
            1.0,
            "paco/oracle-object-root",
            metadata={
                "root_origin": "paco-oracle-object-root",
                "root_index": 0,
                "candidate_key": f"{case_id}/root",
            },
        )
        hint = _asset_hint(str(case["object_category"]))
        retrieval = retriever.query(image, [root], asset_hint=hint)
        plan = retrieval.plans[0]
        if not plan.accepted:
            rows.append(
                {
                    "case_id": case_id,
                    "object_category": case.get("object_category"),
                    "status": "retrieval_rejected",
                    "retrieval_reason": plan.reason,
                    "retrieval_similarity": plan.top_similarity,
                }
            )
            continue
        prompts = tuple(prior.guided_spec() for prior in plan.part_priors)
        generated = florence.generate(image, [root], prompts)
        reranked, rerank_diagnostics = retriever.rerank_candidates(
            image,
            root,
            generated.candidates,
            plan.part_priors,
        )

        # Part annotations enter only after proposal generation and reranking.
        truth_rows = list(case_payload["parts"])
        truth_masks = [
            _load_mask(case_dir / str(row["mask_crop"])) for row in truth_rows
        ]
        raw_metrics = _match_metrics(
            truth_masks,
            list(generated.candidates),
            threshold=0.25,
        )
        reranked_metrics = _match_metrics(
            truth_masks,
            reranked,
            threshold=0.25,
        )
        semantic_recall, semantic_rows = _semantic_recall(
            truth_rows,
            truth_masks,
            reranked,
            expected_domain=expected_domain,
            object_category=str(case["object_category"]),
            threshold=0.25,
        )
        row = {
            "case_id": case_id,
            "object_category": case.get("object_category"),
            "expected_domain": expected_domain,
            "status": "ok",
            "retrieved_asset_label": plan.asset_label,
            "retrieval_similarity": plan.top_similarity,
            "prompt_count": len(prompts),
            "truth_part_count": len(truth_masks),
            "raw_candidate_count": len(generated.candidates),
            "reranked_candidate_count": len(reranked),
            "raw_f1_at_025": raw_metrics["f1_at_025"],
            "raw_mean_matched_iou": raw_metrics["mean_matched_iou"],
            "raw_mean_matched_boundary_f1": raw_metrics[
                "mean_matched_boundary_f1"
            ],
            "reranked_f1_at_025": reranked_metrics["f1_at_025"],
            "reranked_mean_matched_iou": reranked_metrics["mean_matched_iou"],
            "reranked_mean_matched_boundary_f1": reranked_metrics[
                "mean_matched_boundary_f1"
            ],
            "semantic_recall_at_025": semantic_recall,
            "generation": generated.diagnostics,
            "reranking": rerank_diagnostics,
            "raw_matches": raw_metrics["matches"],
            "reranked_matches": reranked_metrics["matches"],
            "semantic_matches": semantic_rows,
        }
        rows.append(row)
        summary = {
            "format": "HPID PACO Florence part proposal audit",
            "format_version": "0.1.0",
            "source_manifest": str(args.manifest.resolve()),
            "source_manifest_sha256": _sha256(args.manifest),
            "retrieval_index": str(args.retrieval_index.resolve()),
            "retrieval_index_sha256": _sha256(args.retrieval_index / "index.json"),
            "condition": "prompted-category + oracle-domain + oracle-object-mask",
            "ground_truth_object_mask_used_as_root": True,
            "ground_truth_part_masks_used_during_inference": False,
            "ground_truth_part_names_used_during_inference": False,
            "ground_truth_parts_loaded_only_after_reranking": True,
            "aggregate": _aggregate(rows),
            "cases": rows,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            f"{case_id}: raw_f1={row['raw_f1_at_025']:.4f} "
            f"reranked_f1={row['reranked_f1_at_025']:.4f} "
            f"semantic_recall={row['semantic_recall_at_025']:.4f}",
            flush=True,
        )
    florence.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
