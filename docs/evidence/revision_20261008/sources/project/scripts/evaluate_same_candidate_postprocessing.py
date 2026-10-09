from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from hpid_split.fusion import FusionConfig, MaskCandidate, fuse_candidates
from hpid_split.metrics import binary_iou
from hpid_split.paco_eval import _normalize
from hpid_split.paper_eval import DEFAULT_IOU_THRESHOLDS, evaluate_part_predictions
from hpid_split.postprocess_baselines import (
    BaselineInstance,
    BaselinePrediction,
    dbscan_proposal_fusion,
    greedy_nms_ownership,
    local_pairwise_crf,
    overlap_excess,
    raw_proposals,
    unassigned_fraction,
)

SEED = 20260828
METHODS = (
    "raw_proposals",
    "greedy_nms",
    "dbscan_fusion",
    "local_pairwise_crf",
    "hpid_split_a3",
)
SUMMARY_METRICS = (
    "part_f1_at_025",
    "part_f1_at_050",
    "part_f1_at_075",
    "part_f1_mean_025_075",
    "mean_matched_iou_at_025",
    "mean_matched_boundary_f1_at_025",
    "semantic_f1_at_025",
    "object_iou",
    "oversegmentation_ratio",
    "predicted_part_count",
    "root_foreground_iou",
    "overlap_excess_root_fraction",
    "unassigned_root_fraction",
    "runtime_seconds",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) >= 128


def _load_candidates(package_dir: Path) -> list[MaskCandidate]:
    rows = json.loads((package_dir / "candidates.json").read_text(encoding="utf-8"))
    return [
        MaskCandidate(
            semantic_name=str(row["semantic_name"]),
            semantic_parent=str(row["semantic_parent"]),
            mask=_load_mask(package_dir / str(row["mask_path"])),
            score=float(row["score"]),
            source=str(row["source"]),
            prompt=str(row.get("prompt", "")),
            source_reliability=float(row.get("source_reliability", 1.0)),
            metadata=dict(row.get("metadata") or {}),
        )
        for row in rows
    ]


def _hpid_prediction(package_dir: Path, root_mask: np.ndarray) -> BaselinePrediction:
    instance_map = np.asarray(Image.open(package_dir / "part_id_map.tiff"))
    rows = json.loads((package_dir / "parts.json").read_text(encoding="utf-8"))
    instances = tuple(
        BaselineInstance(
            identity=str(row["part_id"]),
            semantic_name=str(row["semantic_name"]),
            semantic_parent=str(row["semantic_parent"]),
            mask=instance_map == int(row["instance_index"]),
            confidence=1.0,
        )
        for row in rows
    )
    return BaselinePrediction(
        method="hpid_split_a3",
        instances=instances,
        root_mask=root_mask,
        diagnostics={"source": "stored frozen v0.3.1 public Part-ID map"},
    )


def _coverage_iou(masks: list[np.ndarray], truth: np.ndarray) -> float:
    union = (
        np.logical_or.reduce(masks)
        if masks
        else np.zeros(truth.shape, dtype=bool)
    )
    return binary_iou(union, truth)


def _evaluation_rows(
    prediction: BaselinePrediction,
    *,
    expected_domain: str,
    object_category: str,
    truth_names: set[str],
) -> tuple[list[np.ndarray], list[str]]:
    include_root_body = "body" in truth_names
    masks: list[np.ndarray] = []
    semantics: list[str] = []
    for instance in prediction.instances:
        is_root = instance.semantic_name == expected_domain
        if is_root and not include_root_body:
            continue
        masks.append(np.asarray(instance.mask, dtype=bool))
        if is_root:
            semantics.append("body")
        else:
            raw_name = instance.semantic_name.removeprefix(f"{expected_domain}_")
            semantics.append(
                _normalize(
                    raw_name,
                    expected_domain,
                    object_category=object_category,
                )
            )
    return masks, semantics


def _evaluate(
    prediction: BaselinePrediction,
    *,
    case: dict[str, object],
    case_dir: Path,
    expected_domain: str,
) -> dict[str, float]:
    object_category = str(case["object_category"])
    truth_rows = list(case["parts"])
    truth_masks = [
        _load_mask(case_dir / str(row["mask_crop"])) for row in truth_rows
    ]
    truth_semantics = [
        _normalize(
            str(row["part_name"]),
            expected_domain,
            object_category=object_category,
        )
        for row in truth_rows
    ]
    truth_names = set(truth_semantics)
    prediction_masks, prediction_semantics = _evaluation_rows(
        prediction,
        expected_domain=expected_domain,
        object_category=object_category,
        truth_names=truth_names,
    )
    truth_object = _load_mask(case_dir / "object_mask_crop.png")
    metrics = evaluate_part_predictions(
        truth_masks=truth_masks,
        truth_semantics=truth_semantics,
        prediction_masks=prediction_masks,
        prediction_semantics=prediction_semantics,
        truth_object_mask=truth_object,
        thresholds=DEFAULT_IOU_THRESHOLDS,
    )
    all_masks = [np.asarray(item.mask, dtype=bool) for item in prediction.instances]
    metrics["root_foreground_iou"] = _coverage_iou(all_masks, truth_object)
    metrics["overlap_excess_root_fraction"] = overlap_excess(
        prediction.instances, prediction.root_mask
    )
    metrics["unassigned_root_fraction"] = unassigned_fraction(
        prediction.instances, prediction.root_mask
    )
    metrics["exclusive_ownership"] = float(
        metrics["overlap_excess_root_fraction"] <= 1e-12
    )
    return {key: float(value) for key, value in metrics.items()}


def _identity_signature(prediction: BaselinePrediction) -> dict[str, str]:
    signatures: dict[str, str] = {}
    for instance in prediction.instances:
        packed = np.packbits(np.asarray(instance.mask, dtype=np.uint8), axis=None)
        signatures[instance.identity] = hashlib.sha1(packed.tobytes()).hexdigest()
    return signatures


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap(values: list[float], seed: int, iterations: int) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indexes = generator.integers(0, len(array), size=(iterations, len(array)))
    means = array[indexes].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def _summarize(
    rows: list[dict[str, object]], iterations: int
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for method_index, method in enumerate(METHODS):
        selected = [row for row in rows if row["method"] == method]
        summary: dict[str, object] = {"method": method, "case_count": len(selected)}
        for metric_index, metric in enumerate(SUMMARY_METRICS):
            interval = _bootstrap(
                [float(row[metric]) for row in selected],
                SEED + method_index * 100 + metric_index,
                iterations,
            )
            for key, value in interval.items():
                summary[f"{key}_{metric}"] = value
        summary["exclusive_ownership_case_rate"] = float(
            np.mean([float(row["exclusive_ownership"]) for row in selected])
        )
        summary["candidate_order_identity_exact_rate"] = float(
            np.mean([bool(row["candidate_order_identity_exact"]) for row in selected])
        )
        summaries.append(summary)
    return summaries


def _paired_deltas(
    rows: list[dict[str, object]], iterations: int
) -> list[dict[str, object]]:
    indexed = {
        (str(row["case_id"]), str(row["method"])): row for row in rows
    }
    case_ids = sorted({str(row["case_id"]) for row in rows})
    output: list[dict[str, object]] = []
    for method_index, method in enumerate(METHODS[:-1]):
        for metric_index, metric in enumerate(SUMMARY_METRICS):
            differences = [
                float(indexed[(case_id, "hpid_split_a3")][metric])
                - float(indexed[(case_id, method)][metric])
                for case_id in case_ids
            ]
            interval = _bootstrap(
                differences,
                SEED + 1000 + method_index * 100 + metric_index,
                iterations,
            )
            output.append(
                {
                    "comparison": f"hpid_split_a3_minus_{method}",
                    "metric": metric,
                    "case_count": len(case_ids),
                    "mean_paired_difference": interval["mean"],
                    "ci95_low": interval["ci95_low"],
                    "ci95_high": interval["ci95_high"],
                }
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare final fusion with simple same-candidate postprocessing."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    manifest_cases = {
        str(row["case_id"]): row
        for row in manifest["cases"]
        if row.get("case_path")
    }
    benchmark = json.loads(
        (args.benchmark_root / "benchmark_summary.json").read_text(encoding="utf-8")
    )
    completed = [
        row for row in benchmark["cases"] if int(row.get("return_code", 1)) == 0
    ]
    if {str(row["case_id"]) for row in completed} != set(manifest_cases):
        raise RuntimeError("manifest and benchmark case identities do not match")

    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for case_number, raw in enumerate(completed, start=1):
        case_id = str(raw["case_id"])
        package_dir = args.benchmark_root / case_id
        candidates = _load_candidates(package_dir)
        source = np.asarray(Image.open(package_dir / "source.png").convert("RGB"))
        root_prediction = raw_proposals(candidates)
        methods: dict[str, BaselinePrediction] = {"raw_proposals": root_prediction}
        timings: dict[str, float] = {"raw_proposals": 0.0}

        for name, factory in (
            (
                "greedy_nms",
                lambda candidates=candidates: greedy_nms_ownership(candidates),
            ),
            (
                "dbscan_fusion",
                lambda candidates=candidates: dbscan_proposal_fusion(candidates),
            ),
            (
                "local_pairwise_crf",
                lambda candidates=candidates, source=source: local_pairwise_crf(
                    candidates, source
                ),
            ),
        ):
            started = time.perf_counter()
            methods[name] = factory()
            timings[name] = time.perf_counter() - started
        methods["hpid_split_a3"] = _hpid_prediction(
            package_dir, root_prediction.root_mask
        )
        timings["hpid_split_a3"] = float(raw.get("elapsed_seconds", 0.0))

        generator = np.random.default_rng(SEED + case_number)
        permuted_candidates = list(candidates)
        generator.shuffle(permuted_candidates)
        permuted: dict[str, BaselinePrediction] = {
            "raw_proposals": raw_proposals(permuted_candidates),
            "greedy_nms": greedy_nms_ownership(permuted_candidates),
            "dbscan_fusion": dbscan_proposal_fusion(permuted_candidates),
            "local_pairwise_crf": local_pairwise_crf(permuted_candidates, source),
        }
        hpid_permuted = fuse_candidates(
            permuted_candidates,
            image_shape=root_prediction.root_mask.shape,
            config=FusionConfig(),
        )
        permuted["hpid_split_a3"] = BaselinePrediction(
            method="hpid_split_a3",
            instances=tuple(
                BaselineInstance(
                    identity=record.part_id,
                    semantic_name=record.semantic_name,
                    semantic_parent=record.semantic_parent,
                    mask=hpid_permuted.instance_map == record.instance_index,
                    confidence=1.0,
                )
                for record in hpid_permuted.instances
            ),
            root_mask=root_prediction.root_mask,
            diagnostics={"source": "fixed candidate-order permutation"},
        )

        case_path = Path(str(manifest_cases[case_id]["case_path"]))
        case = json.loads(case_path.read_text(encoding="utf-8"))
        expected_domain = str(raw["expected_domain"])
        for method in METHODS:
            prediction = methods[method]
            metric = _evaluate(
                prediction,
                case=case,
                case_dir=case_path.parent,
                expected_domain=expected_domain,
            )
            signature = _identity_signature(prediction)
            permuted_signature = _identity_signature(permuted[method])
            rows.append(
                {
                    "case_id": case_id,
                    "object_category": case["object_category"],
                    "expected_domain": expected_domain,
                    "method": method,
                    "candidate_count": len(candidates),
                    "runtime_seconds": timings[method],
                    "candidate_order_identity_exact": signature == permuted_signature,
                    "native_persistent_identity_contract": method == "hpid_split_a3",
                    "ground_truth_used_in_inference": False,
                    **metric,
                }
            )
            diagnostics.append(
                {
                    "case_id": case_id,
                    "method": method,
                    **prediction.diagnostics,
                }
            )
        print(f"[{case_number}/{len(completed)}] {case_id}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    summaries = _summarize(rows, args.bootstrap_iterations)
    deltas = _paired_deltas(rows, args.bootstrap_iterations)
    _write_csv(args.output / "same_candidate_postprocessing_cases.csv", rows)
    _write_csv(args.output / "same_candidate_postprocessing_summary.csv", summaries)
    _write_csv(args.output / "same_candidate_postprocessing_paired.csv", deltas)
    _write_csv(args.output / "same_candidate_postprocessing_diagnostics.csv", diagnostics)
    report = {
        "format": "HPID-Split same-candidate simple postprocessing comparison",
        "format_version": "1.0.0",
        "case_count": len(completed),
        "methods": list(METHODS),
        "candidate_scope": (
            "The same frozen heterogeneous candidate masks are supplied to every "
            "method; proposal generation is not rerun."
        ),
        "method_scope": {
            "raw_proposals": "all frozen proposals without ownership resolution",
            "greedy_nms": "score-ordered NMS followed by maximum-score ownership",
            "dbscan_fusion": (
                "DBSCAN-style connected components in a fixed mask/semantic/geometry "
                "distance, followed by weighted voting and exclusive ownership"
            ),
            "local_pairwise_crf": (
                "local four-neighbour Potts CRF over semantic unary evidence; this is "
                "not DenseCRF"
            ),
            "hpid_split_a3": "full hierarchy-constrained, source-aware HPID fusion",
        },
        "identity_scope": (
            "Order consistency tests one fixed candidate-list permutation. Only "
            "HPID-Split natively exports the persistent hierarchy and package contract; "
            "baseline identifiers are evaluation adapters."
        ),
        "ground_truth_used_in_prediction": False,
        "manifest_sha256": _sha256(args.manifest),
        "benchmark_summary_sha256": _sha256(
            args.benchmark_root / "benchmark_summary.json"
        ),
        "seed": SEED,
    }
    (args.output / "same_candidate_postprocessing_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
