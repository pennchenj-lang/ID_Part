from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from .resource_paths import user_completion_config, user_hpid_home


def completion_configuration(
    package_root: Path, model_cache: Path
) -> dict[str, object]:
    return {
        "kind": "target-package-lama-sam2",
        "package_root": str(package_root.resolve()),
        "model_cache": str(model_cache.resolve()),
        "segmentation_model": "facebook/sam2.1-hiera-tiny",
        "local_files_only": False,
        "maximum_hypotheses": 3,
        "maximum_target_parts": 12,
        "minimum_target_area_ratio": 0.001,
        "provenance": {
            "name": "LaMa via isolated simple-lama-inpainting adapter",
            "version": "simple-lama-inpainting 0.1.2",
            "implementation_url": (
                "https://github.com/enesmsahin/simple-lama-inpainting"
            ),
            "publication_url": (
                "https://openaccess.thecvf.com/content/WACV2022/html/"
                "Suvorov_Resolution-Robust_Large_Mask_Inpainting_With_"
                "Fourier_Convolutions_WACV_2022_paper.html"
            ),
            "license": "Apache-2.0; verify upstream checkpoint terms",
            "is_hpid_split_method": False,
        },
        "pipeline_provenance": {
            "name": "HPID evidence-gated amodal completion",
            "version": "0.2.0",
            "implementation_url": "",
            "publication_url": "",
            "license": "repository license plus upstream model licenses",
            "is_hpid_split_method": True,
        },
    }


def setup_completion_backend(
    *,
    package_root: Path | None = None,
    model_cache: Path | None = None,
    config_path: Path | None = None,
    python_executable: str = sys.executable,
    skip_install: bool = False,
) -> Path:
    home = user_hpid_home()
    package_root = (package_root or home / "lama_package").expanduser().resolve()
    model_cache = (model_cache or home / "lama_model_cache").expanduser().resolve()
    config_path = (config_path or user_completion_config()).expanduser().resolve()
    package_root.mkdir(parents=True, exist_ok=True)
    model_cache.mkdir(parents=True, exist_ok=True)
    if not skip_install:
        temp_root = home / "pip-temp"
        temp_root.mkdir(parents=True, exist_ok=True)
        environment = os.environ.copy()
        environment.update(
            {
                "TEMP": str(temp_root),
                "TMP": str(temp_root),
                "PIP_CACHE_DIR": str(home / "pip-cache"),
                "PYTHONPATH": str(package_root)
                + (
                    os.pathsep + environment["PYTHONPATH"]
                    if environment.get("PYTHONPATH")
                    else ""
                ),
            }
        )
        subprocess.run(
            [
                python_executable,
                "-m",
                "pip",
                "install",
                "simple-lama-inpainting==0.1.2",
                "--no-deps",
                "--no-cache-dir",
                "--upgrade",
                "--target",
                str(package_root),
            ],
            check=True,
            env=environment,
        )
        subprocess.run(
            [
                python_executable,
                "-c",
                "from simple_lama_inpainting import SimpleLama; print(SimpleLama)",
            ],
            check=True,
            env=environment,
        )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(completion_configuration(package_root, model_cache), indent=2)
        + "\n",
        encoding="utf-8",
    )
    return config_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install the isolated LaMa adapter and write a local HPID config."
    )
    parser.add_argument("--package-root", type=Path)
    parser.add_argument("--model-cache", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-install", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = setup_completion_backend(
        package_root=args.package_root,
        model_cache=args.model_cache,
        config_path=args.config,
        python_executable=args.python,
        skip_install=args.skip_install,
    )
    print(f"completion_config={config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
