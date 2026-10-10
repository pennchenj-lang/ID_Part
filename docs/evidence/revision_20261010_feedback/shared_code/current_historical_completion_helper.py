from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from hpid_split.completion_frontend import (
    corrupt_region,
    evaluate_completion_handoff,
    make_controlled_defect,
    visible_lock_region,
)
from hpid_split.fusion import MaskCandidate
from hpid_split.paco_eval import _normalize
from hpid_split.postprocess_baselines import (
    BaselineInstance,
    BaselinePrediction,
    dbscan_proposal_fusion,
    greedy_nms_ownership,
    local_pairwise_crf,
    raw_proposals,
)
from hpid_split.restoration import BackendProvenance, TargetPackageLamaBackend

SEED = 20260828
METHODS = (
    "raw_proposals",
    "greedy_nms",
    "dbscan_fusion",
    "local_pairwise_crf",
    "hpid_split_group_ids",
)
SUMMARY_METRICS = (
    "query_resolved",
    "query_unique",
    "query_match_count",
    "review_option_count",
    "selected_part_iou",
    "semantic_target_hit_at_025",
    "completion_request_recall",
    "completion_request_precision",
    "visible_lock_exact",
    "changed_outside_request_fraction",
    "hidden_region_mae",
    "hidden_region_psnr",
    "repeat_completion_exact",
    "process_rerun_identity_exact",
    "process_rerun_mask_iou",
    "runtime_seconds",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) >= 128


def _mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    union = int(np.count_nonzero(first | second))
    return intersection / union if union else 0.0


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
    instance_map = np.asarray(Image.open(package_dir / "group_id_map.tiff"))
    rows = json.loads((package_dir / "groups.json").read_text(encoding="utf-8"))
    return BaselinePrediction(
        method="hpid_split_group_ids",
        instances=tuple(
            BaselineInstance(
                identity=str(row["group_id"]),
                semantic_name=str(row["semantic_name"]),
                semantic_parent=str(row["semantic_name"]),
                mask=instance_map == int(row["group_index"]),
                confidence=1.0,
            )
            for row in rows
        ),
        root_mask=root_mask,
        diagnostics={"source": "stored public Part-ID map"},
    )


def _predictions(package_dir: Path) -> dict[str, BaselinePrediction]:
    candidates = _load_candidates(package_dir)
    image = np.asarray(Image.open(package_dir / "source.png").convert("RGB"))
    raw = raw_proposals(candidates)
    return {
        "raw_proposals": raw,
        "greedy_nms": greedy_nms_ownership(candidates),
        "dbscan_fusion": dbscan_proposal_fusion(candidates),
        "local_pairwise_crf": local_pairwise_crf(candidates, image),
        "hpid_split_group_ids": _hpid_prediction(package_dir, raw.root_mask),
    }


def _normalized_name(
    instance: BaselineInstance,
    *,
    expected_domain: str,
    object_category: str,
) -> str:
    if instance.semantic_name == expected_domain:
        return "body"
    return _normalize(
        instance.semantic_name.removeprefix(f"{expected_domain}_"),
        expected_domain,
        object_category=object_category,
    )


def _resolve_target(
    prediction: BaselinePrediction,
    *,
    target_name: str,
    expected_domain: str,
    object_category: str,
) -> tuple[BaselineInstance | None, list[BaselineInstance]]:
    normalized_target = _normalize(
        target_name,
        expected_domain,
        object_category=object_category,
    )
    matches = [
        instance
        for instance in prediction.instances
        if _normalized_name(
            instance,
            expected_domain=expected_domain,
            object_category=object_category,
        )
        == normalized_target
    ]
    selected = (
        max(
            matches,
            key=lambda item: (
                float(item.confidence),
                int(np.count_nonzero(item.mask)),
                item.identity,
            ),
        )
        if matches
        else None
    )
    return selected, matches


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


def _summaries(
    rows: list[dict[str, object]], iterations: int
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
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
        output.append(summary)
    return output


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
                float(indexed[(case_id, "hpid_split_group_ids")][metric])
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
                    "comparison": f"hpid_split_group_ids_minus_{method}",
                    "metric": metric,
                    "case_count": len(case_ids),
                    "mean_paired_difference": interval["mean"],
                    "ci95_low": interval["ci95_low"],
                    "ci95_high": interval["ci95_high"],
                }
            )
    return output


def _save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255).save(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate HPID-Split as a semantic part-completion front-end."
    )
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--inference-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--rerun-a", type=Path, required=True)
    parser.add_argument("--rerun-b", type=Path, required=True)
    parser.add_argument("--lama-package-root", type=Path, required=True)
    parser.add_argument("--lama-model-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    args = parser.parse_args()

    manifest = json.loads(args.target_manifest.read_text(encoding="utf-8"))
    backend = TargetPackageLamaBackend(
        args.lama_package_root,
        args.lama_model_cache,
        BackendProvenance(
            name="LaMa via simple-lama-inpainting",
            version="simple-lama-inpainting 0.1.2 / big-lama.pt",
            implementation_url=(
                "https://github.com/enesmsahin/simple-lama-inpainting"
            ),
            publication_url=(
                "https://openaccess.thecvf.com/content/WACV2022/html/"
                "Suvorov_Resolution-Robust_Large_Mask_Inpainting_With_Fourier_"
                "Convolutions_WACV_2022_paper.html"
            ),
            license="Apache-2.0; upstream checkpoint terms apply",
            is_hpid_split_method=False,
        ),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    cases_root = args.output / "cases"
    cases_root.mkdir(exist_ok=True)
    rows: list[dict[str, object]] = []

    for case_number, target in enumerate(manifest["selected_cases"], start=1):
        case_id = str(target["case_id"])
        object_category = str(target["object_category"])
        expected_domain = str(target["expected_domain"])
        target_name = str(target["target_part_name"])
        package_dir = args.inference_root / case_id
        source = Image.open(package_dir / "source.png").convert("RGB")
        truth_part = _load_mask(
            args.reference_root / str(target["target_mask_relative_path"])
        )
        defect = make_controlled_defect(
            truth_part,
            seed=int(target["defect_seed"]),
            fraction=float(target["defect_fraction"]),
            mode=str(target["defect_mode"]),
        )
        corrupted = corrupt_region(source, defect.mask)
        predictions = _predictions(package_dir)
        rerun_a = _predictions(args.rerun_a / case_id)
        rerun_b = _predictions(args.rerun_b / case_id)

        case_output = cases_root / case_id
        case_output.mkdir(exist_ok=True)
        source.save(case_output / "source.png")
        corrupted.save(case_output / "corrupted.png")
        _save_mask(case_output / "truth_part.png", truth_part)
        _save_mask(case_output / "controlled_defect.png", defect.mask)

        for method in METHODS:
            prediction = predictions[method]
            selected, matches = _resolve_target(
                prediction,
                target_name=target_name,
                expected_domain=expected_domain,
                object_category=object_category,
            )
            selected_mask = (
                np.asarray(selected.mask, dtype=bool)
                if selected is not None
                else np.zeros(truth_part.shape, dtype=bool)
            )
            request_mask = defect.mask & selected_mask
            started = time.perf_counter()
            if request_mask.any():
                generated = backend.inpaint_region(
                    corrupted, request_mask, target_name
                )
                repeated = backend.inpaint_region(
                    corrupted, request_mask, target_name
                )
            else:
                generated = np.asarray(corrupted.convert("RGBA"), dtype=np.uint8)
                repeated = generated.copy()
            completed = visible_lock_region(corrupted, generated, request_mask)
            repeated_completed = visible_lock_region(
                corrupted, repeated, request_mask
            )
            runtime = time.perf_counter() - started
            metrics = evaluate_completion_handoff(
                original=source,
                corrupted=corrupted,
                completed_rgba=completed,
                requested_mask=request_mask,
                truth_defect_mask=defect.mask,
                selected_part_mask=selected_mask,
                truth_part_mask=truth_part,
            )

            selected_a, _matches_a = _resolve_target(
                rerun_a[method],
                target_name=target_name,
                expected_domain=expected_domain,
                object_category=object_category,
            )
            selected_b, _matches_b = _resolve_target(
                rerun_b[method],
                target_name=target_name,
                expected_domain=expected_domain,
                object_category=object_category,
            )
            rerun_identity_exact = bool(
                selected_a is not None
                and selected_b is not None
                and selected_a.identity == selected_b.identity
            )
            rerun_iou = (
                _mask_iou(selected_a.mask, selected_b.mask)
                if selected_a is not None and selected_b is not None
                else 0.0
            )
            method_dir = case_output / method
            method_dir.mkdir(exist_ok=True)
            _save_mask(method_dir / "selected_part.png", selected_mask)
            _save_mask(method_dir / "completion_request.png", request_mask)
            Image.fromarray(completed, mode="RGBA").save(
                method_dir / "completed.png"
            )
            (method_dir / "handoff.json").write_text(
                json.dumps(
                    {
                        "target_part_name": target_name,
                        "selected_identity": (
                            selected.identity if selected is not None else None
                        ),
                        "query_match_count": len(matches),
                        "request_area_px": int(np.count_nonzero(request_mask)),
                        "metrics": asdict(metrics),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            selected_iou = metrics.target_mask_iou
            rows.append(
                {
                    "case_id": case_id,
                    "object_category": object_category,
                    "expected_domain": expected_domain,
                    "target_part_name": target_name,
                    "defect_mode": defect.mode,
                    "defect_fraction": defect.actual_fraction,
                    "method": method,
                    "query_resolved": float(selected is not None),
                    "query_unique": float(len(matches) == 1),
                    "query_match_count": len(matches),
                    "review_option_count": (
                        len(matches) if matches else len(prediction.instances)
                    ),
                    "selected_identity": (
                        selected.identity if selected is not None else ""
                    ),
                    "native_persistent_identity_contract": (
                        method == "hpid_split_group_ids"
                    ),
                    "selected_part_iou": selected_iou,
                    "semantic_target_hit_at_025": float(selected_iou >= 0.25),
                    "completion_request_recall": metrics.request_recall,
                    "completion_request_precision": metrics.request_precision,
                    "visible_lock_exact": float(metrics.visible_lock_exact),
                    "changed_outside_request_fraction": (
                        metrics.changed_outside_request_fraction
                    ),
                    "hidden_region_mae": metrics.hidden_region_mae,
                    "hidden_region_psnr": metrics.hidden_region_psnr,
                    "repeat_completion_exact": float(
                        _array_sha256(completed)
                        == _array_sha256(repeated_completed)
                    ),
                    "process_rerun_identity_exact": float(rerun_identity_exact),
                    "process_rerun_mask_iou": rerun_iou,
                    "runtime_seconds": runtime,
                    "ground_truth_used_in_hpid_inference": False,
                    "reference_used_to_construct_and_score_defect": True,
                }
            )
        print(
            f"[{case_number}/{manifest['case_count']}] {case_id} / {target_name}",
            flush=True,
        )

    summaries = _summaries(rows, args.bootstrap_iterations)
    paired = _paired_deltas(rows, args.bootstrap_iterations)
    _write_csv(args.output / "completion_frontend_cases.csv", rows)
    _write_csv(args.output / "completion_frontend_summary.csv", summaries)
    _write_csv(args.output / "completion_frontend_paired.csv", paired)
    report = {
        "format": "HPID-Split controlled semantic-completion front-end benchmark",
        "format_version": "1.0.0",
        "case_count": manifest["case_count"],
        "methods": list(METHODS),
        "task": (
            "Resolve a requested physical part, intersect its selected identity mask "
            "with a controlled missing region, run a fixed LaMa completion backend, "
            "and lock every pixel outside that request."
        ),
        "evidence_boundary": (
            "This is controlled 2D photo-region recovery and identity handoff. It is "
            "not real amodal ground truth, full-image detection, 3D reconstruction, "
            "out-of-frame prediction, or agricultural validation."
        ),
        "selection_burden_definition": (
            "review_option_count counts matching semantic choices, or all exported "
            "instances when the requested semantic is absent. It is an objective "
            "option-count proxy, not measured human time."
        ),
        "completion_backend": asdict(backend.provenance),
        "target_manifest_sha256": _sha256(args.target_manifest),
        "reference_masks_used_in_hpid_inference": False,
        "reference_masks_used_for_benchmark_construction_and_scoring": True,
        "seed": SEED,
    }
    (args.output / "completion_frontend_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
