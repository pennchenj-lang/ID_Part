from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
import psutil
import torch
from PIL import Image

TIMING_PATTERN = re.compile(r"timings=(\{[^\r\n]+\})")


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _git_commit(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.strip()


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "standard_deviation": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "p95": float(np.quantile(array, 0.95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export hardware, software, input-size, and stage-runtime audit."
    )
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sam-cache-manifest", type=Path)
    args = parser.parse_args()

    benchmark_path = args.benchmark_root / "benchmark_summary.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    stage_values: dict[str, list[float]] = defaultdict(list)
    sizes: list[tuple[int, int]] = []
    successful = 0
    for row in benchmark.get("cases", []):
        if int(row.get("return_code", 1)) != 0:
            continue
        successful += 1
        match = TIMING_PATTERN.search(str(row.get("stdout_tail") or ""))
        if match:
            for stage, seconds in json.loads(match.group(1)).items():
                stage_values[str(stage)].append(float(seconds))
        image_path = args.benchmark_root / str(row["case_id"]) / "source.png"
        with Image.open(image_path) as image:
            sizes.append((image.width, image.height))

    width_values = [float(width) for width, _height in sizes]
    height_values = [float(height) for _width, height in sizes]
    area_values = [float(width * height) for width, height in sizes]
    gpu = (
        {
            "name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "cuda_runtime_reported_by_torch": torch.version.cuda,
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        }
        if torch.cuda.is_available()
        else {"name": "not available", "device_count": 0}
    )
    sam_cache = (
        json.loads(args.sam_cache_manifest.read_text(encoding="utf-8"))
        if args.sam_cache_manifest
        else None
    )
    report = {
        "format": "HPID-Split runtime and environment audit",
        "format_version": "1.0.0",
        "successful_case_count": successful,
        "hardware": {
            "cpu": platform.processor() or platform.machine(),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "ram_bytes": psutil.virtual_memory().total,
            "gpu": gpu,
        },
        "software": {
            "operating_system": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": _version("torchvision"),
            "transformers": _version("transformers"),
            "opencv_python": _version("opencv-python"),
            "numpy": np.__version__,
            "git_commit": _git_commit(Path.cwd()),
        },
        "models": {
            "grounding": "IDEA-Research/grounding-dino-tiny",
            "segmentation": "facebook/sam2.1-hiera-tiny",
            "dense_semantic": "CIDAS/clipseg-rd64-refined",
            "asset_router": "google/siglip2-base-patch16-224 compatible local snapshot",
        },
        "execution": {
            "device": "cuda",
            "precision_mode": "model defaults; no explicit FP16/BF16 autocast override",
            "sam2_segmentation_batch_size": 16,
            "visual_points_per_crop": 18,
            "visual_crop_layers": 1,
            "warmup": "none",
            "model_loading": (
                "included in each command-level total; the benchmark runner "
                "launches one isolated process per case"
            ),
            "grounding_and_sam2_included": True,
            "fast_mode_45_second_statement": (
                "an internal implementation target, not a guaranteed latency bound"
            ),
        },
        "input_crop_resolution": {
            "width": _summary(width_values),
            "height": _summary(height_values),
            "pixel_area": _summary(area_values),
        },
        "stage_runtime_seconds": {
            stage: _summary(values) for stage, values in sorted(stage_values.items())
        },
        "sam2_raw_cache": sam_cache,
        "scope": (
            "Timings describe ground-truth-box object crops and include model "
            "loading in total. They do not measure full-image object detection."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "runtime_environment_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
