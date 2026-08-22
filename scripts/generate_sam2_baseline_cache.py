from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from hpid_split.visual_regions import Sam2VisualRegionProposer, VisualRegionConfig


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_case(
    output: Path,
    masks: np.ndarray,
    scores: np.ndarray,
    boxes: np.ndarray,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    packed = np.packbits(masks.astype(np.uint8), axis=2)
    np.savez_compressed(
        output,
        packed_masks=packed,
        width=np.asarray([masks.shape[2]], dtype=np.int32),
        scores=scores.astype(np.float32),
        boxes_xyxy=boxes.astype(np.int32),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a reusable, label-free SAM2 point-grid proposal cache. "
            "No object or part annotation is read during proposal generation."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="facebook/sam2.1-hiera-tiny",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--points-per-crop", type=int, default=18)
    parser.add_argument("--points-per-batch", type=int, default=32)
    parser.add_argument("--predicted-iou-threshold", type=float, default=0.78)
    parser.add_argument("--stability-score-threshold", type=float, default=0.82)
    parser.add_argument("--nms-threshold", type=float, default=0.65)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()

    try:
        from transformers import Sam2Model, Sam2Processor
    except ImportError as error:
        raise RuntimeError("Install the foundation dependency group") from error

    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    requested = set(args.case)
    cases = [
        row
        for row in manifest.get("cases", [])
        if row.get("case_path")
        and (not requested or str(row["case_id"]) in requested)
    ]
    missing = requested - {str(row["case_id"]) for row in cases}
    if missing:
        parser.error(f"unknown cases: {sorted(missing)}")

    args.output.mkdir(parents=True, exist_ok=True)
    processor = Sam2Processor.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    model = Sam2Model.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    ).to(args.device)
    model.eval()
    config = VisualRegionConfig(
        points_per_crop=args.points_per_crop,
        points_per_batch=args.points_per_batch,
        crops_n_layers=0,
        predicted_iou_threshold=args.predicted_iou_threshold,
        stability_score_threshold=args.stability_score_threshold,
        crops_nms_threshold=args.nms_threshold,
    )
    proposer = Sam2VisualRegionProposer(
        processor,
        model,
        segmentation_model=args.model,
        device=args.device,
        config=config,
    )

    rows: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        case_id = str(case["case_id"])
        cache_path = args.output / "cases" / f"{case_id}.npz"
        metadata_path = args.output / "cases" / f"{case_id}.json"
        if args.resume and cache_path.is_file() and metadata_path.is_file():
            row = json.loads(metadata_path.read_text(encoding="utf-8"))
            rows.append(row)
            print(f"[{index}/{len(cases)}] {case_id}: resumed", flush=True)
            continue

        case_path = Path(str(case["case_path"]))
        image_path = case_path.parent / "source_crop.png"
        image = Image.open(image_path).convert("RGB")
        started = time.perf_counter()
        proposals = proposer._pipeline_proposals(
            image,
            points_per_crop=args.points_per_crop,
            offset_xy=(0, 0),
            output_shape=(image.height, image.width),
            scale_level=0,
            view_id="global",
        )
        elapsed = time.perf_counter() - started
        masks = np.stack([row.mask for row in proposals], axis=0) if proposals else np.zeros((0, image.height, image.width), dtype=bool)
        scores = np.asarray([row.score for row in proposals], dtype=np.float32)
        boxes = np.asarray(
            [row.bbox_xyxy or (0, 0, 0, 0) for row in proposals],
            dtype=np.int32,
        ).reshape(-1, 4)
        _write_case(cache_path, masks, scores, boxes)
        row = {
            "case_id": case_id,
            "object_category": case.get("object_category"),
            "proposal_count": len(proposals),
            "image_width": image.width,
            "image_height": image.height,
            "runtime_seconds": elapsed,
            "source_image_sha256": _sha256(image_path),
            "cache_path": str(cache_path.resolve()),
            "cache_sha256": _sha256(cache_path),
            "ground_truth_used": False,
        }
        metadata_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
        rows.append(row)
        print(
            f"[{index}/{len(cases)}] {case_id}: "
            f"{len(proposals)} proposals in {elapsed:.2f}s",
            flush=True,
        )

    payload = {
        "format": "HPID-Split SAM2 baseline proposal cache",
        "format_version": "1.0.0",
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": _sha256(args.manifest),
        "model": args.model,
        "device": args.device,
        "parameters": {
            "points_per_crop": args.points_per_crop,
            "points_per_batch": args.points_per_batch,
            "predicted_iou_threshold": args.predicted_iou_threshold,
            "stability_score_threshold": args.stability_score_threshold,
            "nms_threshold": args.nms_threshold,
        },
        "case_count": len(rows),
        "mean_runtime_seconds": float(
            np.mean([float(row["runtime_seconds"]) for row in rows])
        )
        if rows
        else 0.0,
        "ground_truth_used": False,
        "cases": rows,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in payload.items() if k != "cases"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
