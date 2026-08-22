from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from analyze_cross_domain_fusion_ablation import (
    METRICS,
    SEED,
    _bootstrap_mean,
    _evaluate_result,
    _load_candidates,
    _sha256,
    _write_csv,
)
from PIL import Image

from hpid_split.fusion import FusionConfig, fuse_candidates

PARAMETERS = (
    "full_agreement_overlap",
    "specificity_minimum_containment",
    "specificity_root_minimum_candidate_score",
    "specificity_host_suppression",
    "remainder_merge_distance_ratio",
)
FACTORS = (0.8, 0.9, 1.0, 1.1, 1.2)


def _variant_name(parameter: str, factor: float) -> str:
    return f"{parameter}__{factor:.1f}x"


def _evaluate_case(
    raw: dict[str, object],
    manifest_case: dict[str, object],
    benchmark_root: Path,
    configurations: dict[str, tuple[str, float, FusionConfig]],
) -> list[dict[str, object]]:
    case_id = str(raw["case_id"])
    package_dir = benchmark_root / case_id
    candidates = _load_candidates(package_dir)
    stored_map = np.asarray(Image.open(package_dir / "part_id_map.tiff"))

    # Complete every sensitivity prediction before opening this case's labels.
    predictions: dict[str, object] = {}
    timings: dict[str, float] = {}
    for variant, (_parameter, _factor, config) in configurations.items():
        started = time.perf_counter()
        predictions[variant] = fuse_candidates(
            candidates,
            image_shape=stored_map.shape,
            config=config,
        )
        timings[variant] = time.perf_counter() - started

    case_path = Path(str(manifest_case["case_path"]))
    case = json.loads(case_path.read_text(encoding="utf-8"))
    expected_domain = str(raw["expected_domain"])
    rows: list[dict[str, object]] = []
    for variant, result in predictions.items():
        parameter, factor, config = configurations[variant]
        metrics = _evaluate_result(
            result=result,
            case=case,
            case_dir=case_path.parent,
            expected_domain=expected_domain,
        )
        rows.append(
            {
                "case_id": case_id,
                "object_category": raw["object_category"],
                "expected_domain": expected_domain,
                "variant": variant,
                "parameter": parameter,
                "factor": factor,
                "base_value": float(getattr(FusionConfig(), parameter)),
                "tested_value": float(getattr(config, parameter)),
                "candidate_count": len(candidates),
                "fusion_seconds": timings[variant],
                "ground_truth_used_in_inference": False,
                **metrics,
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run prespecified one-factor-at-a-time sensitivity analysis on "
            "frozen HPID-Split candidates."
        )
    )
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Independent case workers; predictions within each case remain serial.",
    )
    args = parser.parse_args()

    benchmark_path = args.benchmark_root / "benchmark_summary.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    manifest_path = args.manifest or Path(str(benchmark["source_manifest"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    manifest_cases = {
        str(row["case_id"]): row
        for row in manifest["cases"]
        if row.get("case_path")
    }
    base = FusionConfig()
    configurations = {
        _variant_name(parameter, factor): (
            parameter,
            factor,
            replace(
                base,
                **{parameter: float(getattr(base, parameter)) * factor},
            ),
        )
        for parameter in PARAMETERS
        for factor in FACTORS
    }
    valid_cases = [
        raw
        for raw in benchmark["cases"]
        if int(raw.get("return_code", 1)) == 0
        and str(raw["case_id"]) in manifest_cases
    ]
    rows: list[dict[str, object]] = []
    if args.workers <= 1:
        for benchmark_index, raw in enumerate(valid_cases, start=1):
            case_id = str(raw["case_id"])
            rows.extend(
                _evaluate_case(
                    raw,
                    manifest_cases[case_id],
                    args.benchmark_root,
                    configurations,
                )
            )
            print(f"[{benchmark_index}/{len(valid_cases)}] {case_id}", flush=True)
    else:
        completed = 0
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.workers
        ) as executor:
            future_to_case = {
                executor.submit(
                    _evaluate_case,
                    raw,
                    manifest_cases[str(raw["case_id"])],
                    args.benchmark_root,
                    configurations,
                ): str(raw["case_id"])
                for raw in valid_cases
            }
            for future in concurrent.futures.as_completed(future_to_case):
                case_id = future_to_case[future]
                rows.extend(future.result())
                completed += 1
                print(f"[{completed}/{len(valid_cases)}] {case_id}", flush=True)
        rows.sort(key=lambda row: (str(row["case_id"]), str(row["variant"])))

    summary_rows: list[dict[str, object]] = []
    for parameter_index, parameter in enumerate(PARAMETERS):
        for factor_index, factor in enumerate(FACTORS):
            selected = [
                row
                for row in rows
                if row["parameter"] == parameter and row["factor"] == factor
            ]
            summary: dict[str, object] = {
                "parameter": parameter,
                "factor": factor,
                "base_value": float(getattr(base, parameter)),
                "tested_value": float(getattr(base, parameter)) * factor,
                "case_count": len(selected),
            }
            for metric_index, metric in enumerate(METRICS):
                interval = _bootstrap_mean(
                    [float(row[metric]) for row in selected],
                    seed=(
                        SEED
                        + parameter_index * 1000
                        + factor_index * 100
                        + metric_index
                    ),
                    iterations=args.bootstrap_iterations,
                )
                summary[f"mean_{metric}"] = interval["mean"]
                summary[f"ci95_low_{metric}"] = interval["ci95_low"]
                summary[f"ci95_high_{metric}"] = interval["ci95_high"]
            summary_rows.append(summary)

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "fusion_sensitivity_cases.csv", rows)
    _write_csv(args.output / "fusion_sensitivity_summary.csv", summary_rows)
    report = {
        "format": "HPID-Split frozen-candidate fusion sensitivity analysis",
        "format_version": "1.0.0",
        "case_count": len({str(row["case_id"]) for row in rows}),
        "parameters": list(PARAMETERS),
        "factors": list(FACTORS),
        "base_config": asdict(base),
        "benchmark_sha256": _sha256(benchmark_path),
        "manifest_sha256": _sha256(manifest_path),
        "prediction_protocol": (
            "All five perturbations were prespecified for every parameter; "
            "all predictions for a case were completed before labels were read."
        ),
        "selection_protocol": (
            "No sensitivity variant replaces the frozen release setting and "
            "no test-set result is used to retune a threshold."
        ),
        "case_workers": max(1, args.workers),
        "seed": SEED,
    }
    (args.output / "fusion_sensitivity_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
