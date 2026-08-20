from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from hpid_split.asset_routing import (
    AssetRouter,
    AssetRouterConfig,
    AssetRoutingIndex,
    Siglip2AssetEncoder,
    route_to_dict,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) >= 128


def _normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _aggregate(rows: list[dict[str, object]]) -> dict[str, float | int]:
    if not rows:
        return {"case_count": 0}
    accepted = [row for row in rows if bool(row["accepted"])]
    return {
        "case_count": len(rows),
        "acceptance_rate": len(accepted) / len(rows),
        "top1_accuracy_all": float(
            np.mean([bool(row["top1_correct"]) for row in rows])
        ),
        "top1_accuracy_accepted": (
            float(np.mean([bool(row["top1_correct"]) for row in accepted]))
            if accepted
            else 0.0
        ),
        "top5_accuracy": float(np.mean([bool(row["top5_correct"]) for row in rows])),
        "candidate_set_recall": float(
            np.mean([bool(row["candidate_set_correct"]) for row in rows])
        ),
        "mean_candidate_set_size": float(
            np.mean([len(row["candidate_labels"]) for row in rows])
        ),
        "domain_accuracy": float(
            np.mean([bool(row["domain_correct"]) for row in rows])
        ),
        "profile_accuracy": float(
            np.mean([bool(row["profile_correct"]) for row in rows])
        ),
        "mean_score": float(np.mean([float(row["score"]) for row in rows])),
        "mean_margin": float(np.mean([float(row["margin"]) for row in rows])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate automatic object-category routing on independent PACO images. "
            "Oracle object crops and masks isolate the category router."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prototype-weight", type=float, default=0.20)
    parser.add_argument("--nearest-asset-weight", type=float, default=0.05)
    parser.add_argument("--minimum-score", type=float, default=0.18)
    parser.add_argument("--minimum-margin", type=float, default=0.020)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument(
        "--mask-mode",
        choices=("oracle-object", "full-image"),
        default="oracle-object",
        help=(
            "Use the annotated object mask to isolate router accuracy, or use "
            "the full image to evaluate automatic pre-detection proposals."
        ),
    )
    args = parser.parse_args()
    if not 0.0 <= args.prototype_weight <= 1.0:
        parser.error("--prototype-weight must be in [0, 1]")
    if not 0.0 <= args.nearest_asset_weight <= 1.0:
        parser.error("--nearest-asset-weight must be in [0, 1]")

    source = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    requested = set(args.case)
    cases = [
        row
        for row in source.get("cases", [])
        if row.get("case_path")
        and (not requested or str(row.get("case_id")) in requested)
    ]
    missing = requested - {str(row.get("case_id")) for row in cases}
    if missing:
        parser.error(f"unknown materialized cases: {sorted(missing)}")

    index = AssetRoutingIndex.load(args.index)
    encoder = Siglip2AssetEncoder(
        args.model,
        device=args.device,
        local_files_only=args.local_files_only,
        batch_size=args.batch_size,
    )
    router = AssetRouter(
        index,
        encoder,
        config=AssetRouterConfig(
            prototype_weight=args.prototype_weight,
            text_weight=1.0 - args.prototype_weight,
            nearest_asset_weight=args.nearest_asset_weight,
            minimum_score=args.minimum_score,
            minimum_margin=args.minimum_margin,
            maximum_alternatives=len(index.labels),
        ),
    )

    rows: list[dict[str, object]] = []
    by_domain: dict[str, list[dict[str, object]]] = defaultdict(list)
    for case in cases:
        case_path = Path(str(case["case_path"]))
        image = Image.open(case_path.parent / "source_crop.png").convert("RGB")
        root = (
            _load_mask(case_path.parent / "object_mask_crop.png")
            if args.mask_mode == "oracle-object"
            else np.ones((image.height, image.width), dtype=bool)
        )
        route = router.route(image, root)
        expected_label = str(case["object_category"])
        alternatives = list(route.alternatives)
        row = {
            "case_id": str(case["case_id"]),
            "expected_label": expected_label,
            "expected_domain": str(case["expected_domain"]),
            "expected_profile": str(case.get("expected_profile") or ""),
            **route_to_dict(route),
            "top1_correct": _normalize(alternatives[0]["asset_label"])
            == _normalize(expected_label),
            "top5_correct": any(
                _normalize(item["asset_label"]) == _normalize(expected_label)
                for item in alternatives[:5]
            ),
            "candidate_set_correct": any(
                _normalize(item) == _normalize(expected_label)
                for item in route.candidate_labels
            ),
            "domain_correct": str(alternatives[0]["asset_domain"])
            == str(case["expected_domain"]),
            "profile_correct": str(alternatives[0].get("asset_profile") or "")
            == str(case.get("expected_profile") or ""),
        }
        rows.append(row)
        by_domain[str(case["expected_domain"])].append(row)
        print(
            f"{row['case_id']}: selected={alternatives[0]['asset_label']} "
            f"correct={row['top1_correct']} score={route.score:.4f} "
            f"margin={route.margin:.4f}",
            flush=True,
        )

    payload = {
        "format": "HPID independent PACO asset-router evaluation",
        "format_version": "0.1.0",
        "evidence_scope": (
            "oracle object crop/root; no category label enters inference"
            if args.mask_mode == "oracle-object"
            else "full image proposal; no object mask or category label enters inference"
        ),
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": _sha256(args.manifest),
        "index": str(args.index.resolve()),
        "index_manifest_sha256": _sha256(args.index / "index.json"),
        "model": args.model,
        "configuration": {
            "prototype_weight": args.prototype_weight,
            "text_weight": 1.0 - args.prototype_weight,
            "nearest_asset_weight": args.nearest_asset_weight,
            "minimum_score": args.minimum_score,
            "minimum_margin": args.minimum_margin,
            "mask_mode": args.mask_mode,
        },
        "aggregate": _aggregate(rows),
        "by_domain": {
            domain: _aggregate(domain_rows)
            for domain, domain_rows in sorted(by_domain.items())
        },
        "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
