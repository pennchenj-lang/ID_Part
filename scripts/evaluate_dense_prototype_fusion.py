from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from hpid_split.fusion import FusionConfig, MaskCandidate, fuse_candidates
from hpid_split.paco_eval import _normalize
from hpid_split.paper_eval import DEFAULT_IOU_THRESHOLDS, evaluate_part_predictions
from hpid_split.retrieval import (
    CLIPSegEmbeddingEncoder,
    PrototypeIndex,
    PrototypeRetriever,
    RetrievalConfig,
)
from hpid_split.visual_regions import (
    VisualMaskProposal,
    VisualRegionConfig,
    _boundary_alignment,
    _edge_strength,
    visual_region_candidates_from_masks,
)


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) >= 128


def _load_candidates(package: Path) -> list[MaskCandidate]:
    rows = json.loads((package / "candidates.json").read_text(encoding="utf-8"))
    return [
        MaskCandidate(
            semantic_name=str(row["semantic_name"]),
            semantic_parent=str(row["semantic_parent"]),
            mask=_load_mask(package / str(row["mask_path"])),
            score=float(row["score"]),
            source=str(row["source"]),
            prompt=str(row.get("prompt", "")),
            source_reliability=float(row.get("source_reliability", 1.0)),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in rows
    ]


def _load_dense_proposals(cache: Path, image: Image.Image) -> list[VisualMaskProposal]:
    with np.load(cache, allow_pickle=False) as arrays:
        width = int(arrays["width"][0])
        masks = np.unpackbits(arrays["packed_masks"], axis=2)[:, :, :width].astype(
            bool
        )
        scores = arrays["scores"].astype(np.float32)
        boxes = arrays["boxes_xyxy"].astype(np.int32)
    edges = _edge_strength(np.asarray(image.convert("RGB"))[:, :, ::-1])
    return [
        VisualMaskProposal(
            mask=mask,
            score=float(score),
            bbox_xyxy=tuple(int(value) for value in box),
            boundary_alignment=_boundary_alignment(edges, mask),
            geometric_support=_boundary_alignment(edges, mask),
            source="sam2-dense-development/point-grid",
        )
        for mask, score, box in zip(masks, scores, boxes, strict=True)
    ]


def _root_key(candidate: MaskCandidate) -> str:
    return PrototypeRetriever._root_key(candidate)


def _evaluate(
    *,
    result,
    case: dict[str, object],
    case_dir: Path,
    domain: str,
) -> dict[str, float]:
    category = str(case["object_category"])
    truth_rows = list(case["parts"])
    truth_masks = [_load_mask(case_dir / str(row["mask_crop"])) for row in truth_rows]
    truth_semantics = [
        _normalize(str(row["part_name"]), domain, object_category=category)
        for row in truth_rows
    ]
    truth_names = set(truth_semantics)
    include_root_body = "body" in truth_names
    selected = [
        (record, result.instance_map == record.instance_index)
        for record in result.instances
        if record.semantic_name != domain or include_root_body
    ]
    prediction_masks = [mask for _record, mask in selected]
    prediction_semantics = [
        (
            "body"
            if record.semantic_name == domain
            else _normalize(
                record.semantic_name.removeprefix(f"{domain}_"),
                domain,
                object_category=category,
            )
        )
        for record, _mask in selected
    ]
    metrics = evaluate_part_predictions(
        truth_masks=truth_masks,
        truth_semantics=truth_semantics,
        prediction_masks=prediction_masks,
        prediction_semantics=prediction_semantics,
        truth_object_mask=_load_mask(case_dir / "object_mask_crop.png"),
        thresholds=DEFAULT_IOU_THRESHOLDS,
    )
    return {key: float(value) for key, value in metrics.items()}


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate dense SAM2 proposals with train-only prototype labels."
    )
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--dense-cache", type=Path, required=True)
    parser.add_argument("--retrieval-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    benchmark = json.loads(
        (args.benchmark_root / "benchmark_summary.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(Path(str(benchmark["source_manifest"])).read_text())
    case_paths = {
        str(row["case_id"]): Path(str(row["case_path"]))
        for row in manifest["cases"]
        if row.get("case_path")
    }
    index = PrototypeIndex.load(args.retrieval_index)
    encoder = CLIPSegEmbeddingEncoder(
        model_name=index.encoder_model_name,
        device=args.device,
        local_files_only=args.local_files_only,
        batch_size=32,
    )
    retriever = PrototypeRetriever(
        index,
        encoder,
        config=RetrievalConfig(
            minimum_part_similarity=0.38,
            minimum_raw_part_similarity=0.54,
            minimum_geometry_compatibility=0.12,
            prototype_label_margin=0.015,
        ),
    )
    visual_config = VisualRegionConfig(
        maximum_regions_per_root=64,
        maximum_root_area_fraction=0.72,
        minimum_root_containment=0.72,
    )
    rows: list[dict[str, object]] = []
    for raw in benchmark["cases"]:
        if int(raw.get("return_code", 1)) != 0:
            continue
        case_id = str(raw["case_id"])
        package = args.benchmark_root / case_id
        image = Image.open(package / "source.png").convert("RGB")
        original = _load_candidates(package)
        roots = [
            candidate
            for candidate in original
            if candidate.semantic_name == candidate.semantic_parent
        ]
        dense = _load_dense_proposals(
            args.dense_cache / "cases" / f"{case_id}.npz", image
        )
        converted = visual_region_candidates_from_masks(
            dense,
            roots,
            original,
            config=visual_config,
            source="sam2-dense-development/point-grid",
            candidate_namespace="dense",
        )
        automatic_candidates = {
            _root_key(root): tuple(
                str(value)
                for value in root.metadata.get("asset_router_candidate_labels", [])
            )
            for root in roots
        }
        plan = retriever.query(
            image,
            roots,
            asset_candidates_by_root=automatic_candidates,
        )
        labelled, label_diagnostics = retriever.label_visual_candidates(
            image,
            roots,
            converted.candidates,
            plan.plans,
            existing_candidates=original,
        )
        named_dense = [
            candidate
            for candidate in labelled
            if not bool(candidate.metadata.get("generic_visual_region"))
        ]
        variants = {
            "B0_original": original,
            "B1_dense_named": [*original, *named_dense],
            "B2_dense_all": [*original, *labelled],
        }
        predictions = {}
        timings = {}
        for name, candidates in variants.items():
            started = time.perf_counter()
            predictions[name] = fuse_candidates(
                candidates,
                image_shape=(image.height, image.width),
                config=FusionConfig(),
            )
            timings[name] = time.perf_counter() - started

        # Labels are opened only after every variant has completed inference.
        case_path = case_paths[case_id]
        case = json.loads(case_path.read_text(encoding="utf-8"))
        for name, result in predictions.items():
            metrics = _evaluate(
                result=result,
                case=case,
                case_dir=case_path.parent,
                domain=str(raw["expected_domain"]),
            )
            rows.append(
                {
                    "case_id": case_id,
                    "expected_domain": raw["expected_domain"],
                    "object_category": raw["object_category"],
                    "variant": name,
                    "dense_proposal_count": len(dense),
                    "converted_dense_count": len(converted.candidates),
                    "prototype_labelled_count": int(
                        label_diagnostics["labelled_count"]
                    ),
                    "candidate_count": len(variants[name]),
                    "fusion_seconds": timings[name],
                    "ground_truth_used_in_inference": False,
                    **metrics,
                }
            )
        print(
            f"{case_id}: dense={len(dense)} converted={len(converted.candidates)} "
            f"labelled={label_diagnostics['labelled_count']}",
            flush=True,
        )

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "cases.csv", rows)
    summary = []
    for variant in ("B0_original", "B1_dense_named", "B2_dense_all"):
        selected = [row for row in rows if row["variant"] == variant]
        summary.append(
            {
                "variant": variant,
                "case_count": len(selected),
                **{
                    f"mean_{key}": float(np.mean([float(row[key]) for row in selected]))
                    for key in (
                        "object_iou",
                        "part_f1_at_025",
                        "part_recall_at_025",
                        "part_precision_at_025",
                        "semantic_f1_at_025",
                        "semantic_recall_at_025",
                        "predicted_part_count",
                    )
                },
            }
        )
    _write_csv(args.output / "summary.csv", summary)
    (args.output / "report.json").write_text(
        json.dumps(
            {
                "format": "HPID-Split v0.3.2 dense-prototype development audit",
                "case_count": len({str(row["case_id"]) for row in rows}),
                "variants": [row["variant"] for row in summary],
                "inference_uses_ground_truth": False,
                "development_only": True,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
