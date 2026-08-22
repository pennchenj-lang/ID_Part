from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from hpid_split.dense_semantic import DenseSemanticProposer
from hpid_split.fusion import MaskCandidate
from hpid_split.metrics import binary_iou, boundary_f1
from hpid_split.paco_semantics import canonical_part_token, normalize_paco_name
from hpid_split.paper_eval import evaluate_part_predictions
from hpid_split.retrieval import (
    CLIPSegEmbeddingEncoder,
    PrototypeIndex,
    PrototypeRetriever,
    RetrievalConfig,
)
from hpid_split.semantic_refinement import (
    Sam2SemanticRefiner,
    SemanticRefinementConfig,
    exclusive_semantic_assignment,
)


@dataclass(frozen=True)
class ModelSpec:
    label: str
    path: str


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


def _token(value: str, domain: str) -> str:
    return normalize_paco_name(value).removeprefix(f"{domain}_")


def _clean_mask(
    mask: np.ndarray,
    root: np.ndarray,
    *,
    maximum_instances: int,
) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool) & root
    if not mask.any():
        return mask
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    count, components, stats, _ = cv2.connectedComponentsWithStats(cleaned, 8)
    minimum_area = max(6, round(float(root.sum()) * 0.0005))
    ranked = sorted(
        (
            (int(stats[index, cv2.CC_STAT_AREA]), index)
            for index in range(1, count)
            if int(stats[index, cv2.CC_STAT_AREA]) >= minimum_area
        ),
        reverse=True,
    )
    output = np.zeros(root.shape, dtype=bool)
    for _, component_id in ranked[: max(1, maximum_instances)]:
        output |= components == component_id
    return output & root


def _truth_by_semantic(
    case: dict[str, object],
    *,
    case_dir: Path,
    expected_domain: str,
    object_category: str,
) -> dict[str, np.ndarray]:
    grouped: dict[str, np.ndarray] = {}
    for row in case.get("parts", []):
        token = canonical_part_token(
            str(row["part_name"]),
            expected_domain,
            object_category=object_category,
        )
        mask = _load_mask(case_dir / str(row["mask_crop"]))
        grouped[token] = grouped.get(token, np.zeros(mask.shape, dtype=bool)) | mask
    return grouped


def _truth_instances(
    case: dict[str, object],
    *,
    case_dir: Path,
    expected_domain: str,
    object_category: str,
) -> tuple[list[np.ndarray], list[str]]:
    masks: list[np.ndarray] = []
    semantics: list[str] = []
    for row in case.get("parts", []):
        masks.append(_load_mask(case_dir / str(row["mask_crop"])))
        semantics.append(
            canonical_part_token(
                str(row["part_name"]),
                expected_domain,
                object_category=object_category,
            )
        )
    return masks, semantics


def _prediction_instances(
    predicted: dict[str, np.ndarray],
) -> tuple[list[np.ndarray], list[str]]:
    masks: list[np.ndarray] = []
    semantics: list[str] = []
    for semantic_name, semantic_mask in sorted(predicted.items()):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            np.asarray(semantic_mask, dtype=np.uint8), 8
        )
        components = sorted(
            (
                (int(stats[index, cv2.CC_STAT_AREA]), index)
                for index in range(1, count)
                if int(stats[index, cv2.CC_STAT_AREA]) >= 6
            ),
            reverse=True,
        )
        for _area, component_id in components:
            masks.append(labels == component_id)
            semantics.append(semantic_name)
    return masks, semantics


def _measure(
    truth: dict[str, np.ndarray],
    predicted: dict[str, np.ndarray],
    *,
    root: np.ndarray,
    queried: set[str],
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    accepted = 0
    for semantic_name, truth_mask in sorted(truth.items()):
        prediction = predicted.get(
            semantic_name, np.zeros(truth_mask.shape, dtype=bool)
        )
        overlap = binary_iou(truth_mask, prediction)
        boundary = boundary_f1(prediction, truth_mask, tolerance=3)
        accepted += int(overlap >= 0.25)
        rows.append(
            {
                "semantic_name": semantic_name,
                "queried": semantic_name in queried,
                "predicted": bool(prediction.any()),
                "iou": overlap,
                "boundary_f1_tolerance_3": boundary,
                "accepted_at_025": overlap >= 0.25,
            }
        )
    prediction_count = int(sum(bool(mask.any()) for mask in predicted.values()))
    precision = accepted / max(1, prediction_count)
    recall = accepted / max(1, len(truth))
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    masks = [mask for mask in predicted.values() if mask.any()]
    if masks:
        stacked = np.stack(masks, axis=0)
        overlap_pixels = np.maximum(stacked.sum(axis=0) - 1, 0).sum()
        predicted_union = stacked.any(axis=0)
    else:
        overlap_pixels = 0
        predicted_union = np.zeros(root.shape, dtype=bool)
    return {
        "truth_semantic_count": len(truth),
        "queried_semantic_count": len(queried),
        "predicted_semantic_count": prediction_count,
        "query_recall": len(set(truth) & queried) / max(1, len(truth)),
        "mean_iou_all_truth": float(np.mean([row["iou"] for row in rows])),
        "mean_boundary_f1_all_truth": float(
            np.mean([row["boundary_f1_tolerance_3"] for row in rows])
        ),
        "precision_at_025": precision,
        "recall_at_025": recall,
        "f1_at_025": f1,
        "predicted_union_root_fraction": float(
            predicted_union.sum() / max(1, root.sum())
        ),
        "overlap_root_fraction": float(overlap_pixels / max(1, root.sum())),
        "parts": rows,
    }


def _palette(index: int) -> tuple[int, int, int]:
    colors = (
        (46, 204, 113),
        (52, 152, 219),
        (231, 76, 60),
        (241, 196, 15),
        (155, 89, 182),
        (26, 188, 156),
        (230, 126, 34),
        (236, 112, 99),
        (93, 173, 226),
        (88, 214, 141),
    )
    return colors[index % len(colors)]


def _save_preview(
    path: Path,
    image: Image.Image,
    root: np.ndarray,
    predicted: dict[str, np.ndarray],
) -> None:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    canvas = np.zeros_like(rgb)
    canvas[~root] = 20
    for index, (_, mask) in enumerate(sorted(predicted.items())):
        canvas[mask] = _palette(index)
    preview = np.concatenate([rgb, canvas], axis=1)
    Image.fromarray(preview).save(path)


def _model_threshold(proposer: DenseSemanticProposer, fallback: float) -> float:
    metadata = getattr(proposer.model.config, "hpid_conditional_parts", None)
    if isinstance(metadata, dict):
        return float(metadata.get("calibrated_threshold", fallback))
    return fallback


def _predict_case(
    proposer: DenseSemanticProposer,
    image: Image.Image,
    root: np.ndarray,
    priors: tuple[object, ...],
    *,
    expected_domain: str,
    threshold: float,
    phrases_per_part: int,
    inference_batch_size: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    set[str],
    list[dict[str, object]],
]:
    prompt_rows: list[tuple[str, str, int]] = []
    for prior_index, prior in enumerate(priors):
        phrases = tuple(dict.fromkeys(prior.phrases))[:phrases_per_part]
        if not phrases:
            phrases = (prior.display_name,)
        for phrase in phrases:
            prompt_rows.append((prior.output_semantic_name, phrase, prior_index))
    probability_batches = [
        proposer.probability_maps(
            image,
            [row[1] for row in prompt_rows[start : start + inference_batch_size]],
        )
        for start in range(0, len(prompt_rows), inference_batch_size)
    ]
    probabilities = (
        np.concatenate(probability_batches, axis=0)
        if probability_batches
        else np.zeros((0, image.height, image.width), dtype=np.float32)
    )
    grouped: dict[int, list[np.ndarray]] = {}
    for (_, _, prior_index), probability in zip(
        prompt_rows, probabilities, strict=True
    ):
        grouped.setdefault(prior_index, []).append(probability)
    predicted: dict[str, np.ndarray] = {}
    semantic_probabilities: dict[str, np.ndarray] = {}
    diagnostics: list[dict[str, object]] = []
    queried: set[str] = set()
    for prior_index, prior in enumerate(priors):
        semantic_name = _token(prior.output_semantic_name, expected_domain)
        queried.add(semantic_name)
        maps = grouped.get(prior_index, [])
        probability = (
            np.max(np.stack(maps, axis=0), axis=0)
            if maps
            else np.zeros(root.shape, dtype=np.float32)
        )
        probability = np.asarray(probability, dtype=np.float32)
        probability[~root] = 0.0
        mask = _clean_mask(
            probability >= threshold,
            root,
            maximum_instances=int(prior.maximum_instances),
        )
        if semantic_name in predicted:
            predicted[semantic_name] |= mask
            semantic_probabilities[semantic_name] = np.maximum(
                semantic_probabilities[semantic_name], probability
            )
        else:
            predicted[semantic_name] = mask
            semantic_probabilities[semantic_name] = probability
        root_values = probability[root]
        diagnostics.append(
            {
                "semantic_name": semantic_name,
                "display_name": prior.display_name,
                "phrases": [row[1] for row in prompt_rows if row[2] == prior_index],
                "support_count": int(prior.support_count),
                "prevalence": float(prior.prevalence),
                "retrieval_score": float(prior.retrieval_score),
                "peak_probability": float(root_values.max())
                if len(root_values)
                else 0.0,
                "mean_probability": float(root_values.mean())
                if len(root_values)
                else 0.0,
                "mask_area": int(mask.sum()),
            }
        )
    return predicted, semantic_probabilities, queried, diagnostics


def _aggregate(rows: list[dict[str, object]]) -> dict[str, object]:
    successful = [row for row in rows if row.get("status") == "ok"]
    legacy_metric_names = (
        "query_recall",
        "mean_iou_all_truth",
        "mean_boundary_f1_all_truth",
        "precision_at_025",
        "recall_at_025",
        "f1_at_025",
        "predicted_union_root_fraction",
        "overlap_root_fraction",
    )
    strict_prefixes = (
        "truth_part_count",
        "predicted_part_count",
        "oversegmentation_ratio",
        "object_",
        "part_",
        "mean_matched_",
        "semantic_",
    )
    metric_names = tuple(
        dict.fromkeys(
            [*legacy_metric_names]
            + sorted(
                {
                    key
                    for row in successful
                    for key, value in row.items()
                    if isinstance(value, (int, float))
                    and key.startswith(strict_prefixes)
                }
            )
        )
    )
    return {
        "case_count": len(rows),
        "successful_case_count": len(successful),
        "failed_or_rejected_case_count": len(rows) - len(successful),
        **{
            name: (
                float(np.mean([float(row[name]) for row in successful]))
                if successful
                else 0.0
            )
            for name in metric_names
        },
    }


def _parse_model(value: str) -> ModelSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must use LABEL=PATH")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("model must use non-empty LABEL=PATH")
    return ModelSpec(label.strip(), path.strip())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate text-conditioned part masks on an independent PACO split. "
            "Ground-truth object crops and roots isolate part segmentation; part "
            "annotations are read only after inference."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--retrieval-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=_parse_model, action="append", required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument(
        "--override-model-threshold",
        action="store_true",
        help="Use --threshold instead of checkpoint calibration for development audit.",
    )
    parser.add_argument("--phrases-per-part", type=int, default=2)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--maximum-part-prompts", type=int, default=24)
    parser.add_argument("--sam-refine", action="store_true")
    parser.add_argument("--no-previews", action="store_true")
    parser.add_argument("--sam-model", default="facebook/sam2.1-hiera-tiny")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.threshold < 1.0:
        parser.error("--threshold must be in (0, 1)")
    if args.phrases_per_part < 1:
        parser.error("--phrases-per-part must be positive")
    if args.inference_batch_size < 1:
        parser.error("--inference-batch-size must be positive")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    requested = set(args.case)
    cases = [
        row
        for row in manifest.get("cases", [])
        if "case_path" in row and (not requested or str(row["case_id"]) in requested)
    ]
    if requested - {str(row["case_id"]) for row in cases}:
        parser.error("one or more requested cases are absent from the manifest")

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
        config=RetrievalConfig(maximum_part_prompts=args.maximum_part_prompts),
    )
    prepared: list[dict[str, object]] = []
    for case in cases:
        case_path = Path(str(case["case_path"]))
        case_dir = case_path.parent
        case_payload = json.loads(case_path.read_text(encoding="utf-8"))
        image = Image.open(case_dir / "source_crop.png").convert("RGB")
        root = _load_mask(case_dir / "object_mask_crop.png")
        expected_domain = str(case.get("expected_domain") or "").strip()
        root_candidate = MaskCandidate(
            expected_domain,
            expected_domain,
            root,
            1.0,
            "paco/oracle-object-root",
            metadata={"candidate_key": f"{case['case_id']}/root"},
        )
        plan = retriever.query(
            image,
            [root_candidate],
            asset_hint=_asset_hint(str(case["object_category"])),
        ).plans[0]
        prepared.append(
            {
                "case": case,
                "case_payload": case_payload,
                "case_dir": case_dir,
                "image": image,
                "root": root,
                "plan": plan,
            }
        )

    args.output.mkdir(parents=True, exist_ok=True)
    refiner = (
        Sam2SemanticRefiner(
            device=args.device,
            config=SemanticRefinementConfig(
                model_name=args.sam_model,
                local_files_only=args.local_files_only,
            ),
        )
        if args.sam_refine
        else None
    )
    summaries: dict[str, object] = {}
    for spec in args.model:
        proposer = DenseSemanticProposer(
            spec.path,
            device=args.device,
            local_files_only=args.local_files_only,
        )
        threshold = (
            args.threshold
            if args.override_model_threshold
            else _model_threshold(proposer, args.threshold)
        )
        model_dir = args.output / spec.label
        model_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, object]] = []
        for item in prepared:
            case = item["case"]
            case_id = str(case["case_id"])
            plan = item["plan"]
            if not plan.accepted or not plan.part_priors:
                rows.append(
                    {
                        "case_id": case_id,
                        "object_category": case["object_category"],
                        "status": "retrieval_rejected",
                        "retrieval_reason": plan.reason,
                        "retrieval_similarity": plan.top_similarity,
                    }
                )
                continue
            expected_domain = str(case["expected_domain"])
            truth = _truth_by_semantic(
                item["case_payload"],
                case_dir=item["case_dir"],
                expected_domain=expected_domain,
                object_category=str(case["object_category"]),
            )
            truth_masks, truth_semantics = _truth_instances(
                item["case_payload"],
                case_dir=item["case_dir"],
                expected_domain=expected_domain,
                object_category=str(case["object_category"]),
            )
            predicted, probabilities, queried, prompt_diagnostics = _predict_case(
                proposer,
                item["image"],
                item["root"],
                plan.part_priors,
                expected_domain=expected_domain,
                threshold=threshold,
                phrases_per_part=args.phrases_per_part,
                inference_batch_size=args.inference_batch_size,
            )
            coarse_metrics = _measure(
                truth,
                predicted,
                root=item["root"],
                queried=queried,
            )
            selected = predicted
            refinement: dict[str, object] | None = None
            if refiner is not None:
                refined_items = refiner.refine(
                    item["image"], item["root"], predicted, probabilities
                )
                refined_masks = {
                    name: refined.mask for name, refined in refined_items.items()
                }
                refined_metrics = _measure(
                    truth,
                    refined_masks,
                    root=item["root"],
                    queried=queried,
                )
                selected, residual = exclusive_semantic_assignment(
                    item["root"],
                    refined_items,
                    activation_threshold=threshold,
                )
                refinement = {
                    "sam2_model": args.sam_model,
                    "coarse_metrics": coarse_metrics,
                    "sam2_independent_metrics": refined_metrics,
                    "residual_root_fraction": float(
                        residual.sum() / max(1, item["root"].sum())
                    ),
                    "semantics": {
                        name: {
                            "used_sam2": value.used_sam2,
                            "component_count": value.component_count,
                            "diagnostics": value.diagnostics,
                        }
                        for name, value in refined_items.items()
                    },
                }
            metrics = _measure(
                truth,
                selected,
                root=item["root"],
                queried=queried,
            )
            prediction_masks, prediction_semantics = _prediction_instances(
                selected
            )
            strict_metrics = evaluate_part_predictions(
                truth_masks=truth_masks,
                truth_semantics=truth_semantics,
                prediction_masks=prediction_masks,
                prediction_semantics=prediction_semantics,
                truth_object_mask=item["root"],
            )
            row = {
                "case_id": case_id,
                "object_category": case["object_category"],
                "expected_domain": expected_domain,
                "status": "ok",
                "retrieved_asset_label": plan.asset_label,
                "retrieval_similarity": plan.top_similarity,
                "threshold": threshold,
                **metrics,
                **strict_metrics,
                "prompt_diagnostics": prompt_diagnostics,
                "refinement": refinement,
            }
            rows.append(row)
            if not args.no_previews:
                _save_preview(
                    model_dir / f"{case_id}_preview.png",
                    item["image"],
                    item["root"],
                    selected,
                )
        summary = _aggregate(rows)
        payload = {
            "format": "HPID PACO object-conditioned part baseline evaluation",
            "format_version": "0.2.0",
            "evidence_scope": (
                "Object-conditioned images with oracle object crop/root. Part "
                "names and masks are unavailable during inference; the "
                "retrieved object-part inventory supplies text prompts."
            ),
            "baseline_interpretation": (
                "CLIPSeg object-part prompting in the one-stage open-vocabulary "
                "part-segmentation style used by OV-PARTS baselines. Connected "
                "components are evaluated as part instances."
            ),
            "not_claimed": [
                "full-image object detection quality",
                "amodal or hidden-region completion quality",
                "state-of-the-art open-vocabulary part segmentation",
            ],
            "sam2_refinement_enabled": args.sam_refine,
            "output_masks_are_sibling_exclusive": args.sam_refine,
            "model": asdict(spec),
            "calibrated_threshold": threshold,
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": _sha256(args.manifest),
            "retrieval_index": str(args.retrieval_index.resolve()),
            "retrieval_index_manifest_sha256": _sha256(
                args.retrieval_index / "index.json"
            ),
            "summary": summary,
            "cases": rows,
        }
        (model_dir / "evaluation.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        summaries[spec.label] = summary
        del proposer

    comparison = {
        "format": "HPID conditional-part model comparison",
        "format_version": "0.1.0",
        "manifest_sha256": _sha256(args.manifest),
        "models": [asdict(spec) for spec in args.model],
        "summaries": summaries,
    }
    (args.output / "comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
