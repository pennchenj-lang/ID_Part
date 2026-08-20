from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from hpid_split.data import load_label_map
from hpid_split.foundation import FoundationCandidateGenerator, FoundationConfig
from hpid_split.fusion import FusionConfig, FusionResult, MaskCandidate, fuse_candidates
from hpid_split.metrics import evaluate_semantic
from hpid_split.prompt_bank import PromptBank
from hpid_split.relational import propose_relational_candidates
from hpid_split.taxonomy import Taxonomy

SEED = 20260811
COMMON_NAMES = (
    "background",
    "skin",
    "eye",
    "hair",
    "upper_clothing",
    "sleeve",
    "lower_clothing",
    "shoe",
    "accessory",
    "unresolved_foreground",
)
COMMON_TAXONOMY = Taxonomy(
    fine_names=COMMON_NAMES,
    parent_names=("background", "foreground"),
    fine_to_parent=(0, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    detail_names=("eye", "accessory"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

FINE_NAME_MAPPING = {
    "character": "skin",
    "character_head": "skin",
    "character_torso": "skin",
    "character_arm": "skin",
    "character_hand": "skin",
    "character_leg": "skin",
    "character_eye": "eyes",
    "character_eyebrow": "eyebrow",
    "character_eyelash": "eyelash",
    "character_hair": "hair",
    "character_front_hair": "front_hair",
    "character_back_hair": "back_hair",
    "character_side_hair": "side_hair",
    "character_upper_clothing": "upper_cloth",
    "character_torso_cloth": "torso_cloth",
    "character_collar": "collar",
    "character_sleeve": "sleeve",
    "character_cuff": "cuff",
    "character_hem": "hem",
    "character_inner_clothing": "inner_cloth",
    "character_lower_clothing": "lower_cloth",
    "character_shoe": "shoes",
    "character_shoe_upper": "shoe_upper",
    "character_shoe_sole": "shoe_sole",
    "character_shoe_tongue": "shoe_tongue",
    "character_shoelace": "shoelace",
    "character_heel": "heel",
    "character_sock": "sock",
    "character_accessory": "accessory",
}

VARIANTS = {
    "max_no_constraints": FusionConfig(
        use_consensus=False,
        use_parent_support=False,
        use_parent_residual=False,
        use_direct_gate=False,
        use_boundary_ownership=False,
        detail_bonus=0.0,
    ),
    "parent_residual": FusionConfig(
        use_consensus=False,
        use_parent_support=False,
        use_parent_residual=True,
        use_direct_gate=False,
        use_boundary_ownership=False,
        detail_bonus=0.0,
    ),
    "consensus_residual": FusionConfig(
        use_consensus=True,
        use_parent_support=False,
        use_parent_residual=True,
        use_direct_gate=False,
        use_boundary_ownership=False,
        detail_bonus=0.0,
    ),
    "hierarchy_support": FusionConfig(
        use_consensus=True,
        use_parent_support=True,
        use_parent_residual=True,
        use_direct_gate=False,
        use_boundary_ownership=False,
        detail_bonus=0.0,
    ),
    "released_fusion": FusionConfig(
        use_consensus=True,
        use_parent_support=True,
        use_parent_residual=True,
        use_direct_gate=True,
        use_boundary_ownership=False,
    ),
    "parent_envelope": FusionConfig(
        use_parent_envelope=True,
        use_transitive_residual=True,
    ),
    "boundary_ownership": FusionConfig(use_boundary_ownership=True),
    "fixed_cleanup": FusionConfig(
        use_parent_envelope=True,
        use_transitive_residual=True,
        standard_component_fraction=0.0,
        detail_component_fraction=0.0,
        cleanup_passes=1,
    ),
    "no_source_agreement": FusionConfig(
        use_parent_envelope=True,
        use_transitive_residual=True,
        uncorroborated_source_penalty=1.0,
    ),
    "agreement_078": FusionConfig(
        use_parent_envelope=True,
        use_transitive_residual=True,
        uncorroborated_source_penalty=0.78,
    ),
    "agreement_086": FusionConfig(
        use_parent_envelope=True,
        use_transitive_residual=True,
        uncorroborated_source_penalty=0.86,
    ),
    "agreement_090": FusionConfig(
        use_parent_envelope=True,
        use_transitive_residual=True,
        uncorroborated_source_penalty=0.90,
    ),
    "agreement_094": FusionConfig(
        use_parent_envelope=True,
        use_transitive_residual=True,
        uncorroborated_source_penalty=0.94,
    ),
    "no_remainder_attachment": FusionConfig(
        use_parent_envelope=True,
        use_transitive_residual=True,
        use_remainder_attachment=False,
    ),
}


def _save_candidates(
    path: Path,
    candidates: tuple[MaskCandidate, ...],
    generator_signature: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    masks = np.stack([candidate.mask.astype(np.uint8) for candidate in candidates])
    metadata = []
    for candidate in candidates:
        item = asdict(candidate)
        item.pop("mask")
        metadata.append(item)
    np.savez_compressed(
        path,
        masks=masks,
        metadata=json.dumps(metadata),
        generator_signature=json.dumps(generator_signature, sort_keys=True),
    )


def _read_candidate_cache(
    path: Path,
) -> tuple[list[MaskCandidate], dict[str, object]]:
    payload = np.load(path, allow_pickle=False)
    if "generator_signature" not in payload:
        raise ValueError(
            f"candidate cache {path} predates model fingerprinting; regenerate it"
        )
    actual_signature = json.loads(str(payload["generator_signature"]))
    metadata = json.loads(str(payload["metadata"]))
    candidates = [
        MaskCandidate(mask=mask.astype(bool), **item)
        for mask, item in zip(payload["masks"], metadata, strict=True)
    ]
    return candidates, actual_signature


def _load_candidates(
    path: Path, expected_signature: dict[str, object]
) -> list[MaskCandidate]:
    candidates, actual_signature = _read_candidate_cache(path)
    if actual_signature != expected_signature:
        raise ValueError(
            f"candidate cache {path} was generated by a different model/config"
        )
    return candidates


def _map_prediction(result: FusionResult) -> np.ndarray:
    output = np.zeros(result.semantic_map.shape, dtype=np.uint8)
    mapping = {
        "character": "unresolved_foreground",
        "character_head": "skin",
        "character_torso": "skin",
        "character_arm": "skin",
        "character_hand": "skin",
        "character_leg": "skin",
        "character_eye": "eye",
        "character_eyebrow": "eye",
        "character_eyelash": "eye",
        "character_hair": "hair",
        "character_front_hair": "hair",
        "character_back_hair": "hair",
        "character_side_hair": "hair",
        "character_upper_clothing": "upper_clothing",
        "character_torso_cloth": "upper_clothing",
        "character_collar": "upper_clothing",
        "character_hem": "upper_clothing",
        "character_inner_clothing": "upper_clothing",
        "character_cuff": "sleeve",
        "character_sleeve": "sleeve",
        "character_lower_clothing": "lower_clothing",
        "character_shoe": "shoe",
        "character_shoe_upper": "shoe",
        "character_shoe_sole": "shoe",
        "character_shoe_tongue": "shoe",
        "character_shoelace": "shoe",
        "character_heel": "shoe",
        "character_sock": "shoe",
        "character_accessory": "accessory",
    }
    common_lookup = {name: index for index, name in enumerate(COMMON_NAMES)}
    for source_id, source_name in enumerate(result.taxonomy.fine_names):
        if source_name == "background":
            continue
        target_name = mapping.get(source_name, "unresolved_foreground")
        output[result.semantic_map == source_id] = common_lookup[target_name]
    return output


def _map_truth(truth: np.ndarray, taxonomy: Taxonomy) -> np.ndarray:
    output = np.zeros(truth.shape, dtype=np.uint8)
    mapping = {
        "skin": "skin",
        "eyes": "eye",
        "eyebrow": "eye",
        "eyelash": "eye",
        "hair": "hair",
        "front_hair": "hair",
        "back_hair": "hair",
        "side_hair": "hair",
        "upper_cloth": "upper_clothing",
        "collar": "upper_clothing",
        "torso_cloth": "upper_clothing",
        "hem": "upper_clothing",
        "inner_cloth": "upper_clothing",
        "sleeve": "sleeve",
        "cuff": "sleeve",
        "lower_cloth": "lower_clothing",
        "shoes": "shoe",
        "shoe_upper": "shoe",
        "shoe_sole": "shoe",
        "shoe_tongue": "shoe",
        "shoelace": "shoe",
        "heel": "shoe",
        "sock": "shoe",
        "accessory": "accessory",
    }
    common_lookup = {name: index for index, name in enumerate(COMMON_NAMES)}
    for class_id, name in enumerate(taxonomy.fine_names):
        if class_id == 0:
            continue
        output[truth == class_id] = common_lookup[
            mapping.get(name, "unresolved_foreground")
        ]
    return output


def _map_prediction_fine(result: FusionResult, target_taxonomy: Taxonomy) -> np.ndarray:
    output = np.zeros(result.semantic_map.shape, dtype=np.uint8)
    target_lookup = {
        name: index for index, name in enumerate(target_taxonomy.fine_names)
    }
    for source_id, source_name in enumerate(result.taxonomy.fine_names):
        if source_id == 0:
            continue
        target_name = FINE_NAME_MAPPING.get(source_name)
        if target_name in target_lookup:
            output[result.semantic_map == source_id] = target_lookup[target_name]
    return output


def _mask_precision_recall(
    prediction: np.ndarray, truth: np.ndarray
) -> tuple[float, float]:
    true_positive = int(np.count_nonzero(prediction & truth))
    predicted = int(np.count_nonzero(prediction))
    expected = int(np.count_nonzero(truth))
    precision = true_positive / predicted if predicted else 0.0
    recall = true_positive / expected if expected else 0.0
    return precision, recall


def _candidate_union_for_truth_class(
    candidates: list[MaskCandidate] | tuple[MaskCandidate, ...],
    truth_name: str,
    image_shape: tuple[int, int],
) -> tuple[np.ndarray, int]:
    matching = [
        candidate.mask
        for candidate in candidates
        if FINE_NAME_MAPPING.get(candidate.semantic_name) == truth_name
    ]
    if not matching:
        return np.zeros(image_shape, dtype=bool), 0
    return np.logical_or.reduce(matching), len(matching)


def _bottleneck_label(raw_recall: float, accepted_recall: float, final_recall: float) -> str:
    if raw_recall < 0.25:
        return "proposal_generation"
    if accepted_recall + 0.10 < raw_recall:
        return "candidate_filtering"
    if final_recall + 0.10 < accepted_recall:
        return "fusion_or_ownership"
    if final_recall < 0.50:
        return "partial_localization"
    return "retained"


def _colorize(labels: np.ndarray) -> Image.Image:
    colors = np.asarray(
        [
            (0, 0, 0),
            (242, 176, 135),
            (215, 65, 140),
            (54, 40, 35),
            (48, 141, 221),
            (99, 186, 238),
            (135, 50, 173),
            (244, 133, 35),
            (16, 174, 105),
            (130, 130, 130),
        ],
        dtype=np.uint8,
    )
    return Image.fromarray(colors[labels])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument(
        "--taxonomy",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "character_parts.json",
    )
    parser.add_argument(
        "--prompt-bank",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "general_asset_prompts.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate-cache-dir",
        type=Path,
        help=(
            "Read/write foundation candidates in this directory. Supplying the "
            "same immutable cache makes fusion ablations directly comparable."
        ),
    )
    parser.add_argument(
        "--frozen-cache-only",
        action="store_true",
        help=(
            "Require an existing fingerprinted cache and audit the signature "
            "stored in that cache instead of regenerating candidates."
        ),
    )
    parser.add_argument(
        "--additional-candidate-cache-dir",
        type=Path,
        action="append",
        default=[],
        help=(
            "Read a second or later fingerprinted candidate source and fuse it "
            "with the primary source. May be supplied more than once."
        ),
    )
    parser.add_argument(
        "--roles", default=",".join(f"{index:04d}" for index in range(1, 11))
    )
    parser.add_argument(
        "--methods",
        default=",".join(VARIANTS),
        help="Comma-separated fusion variants to evaluate.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--grounding-model", default="IDEA-Research/grounding-dino-tiny"
    )
    parser.add_argument(
        "--segmentation-model", default="facebook/sam2.1-hiera-tiny"
    )
    parser.add_argument("--dense-semantic-fallback", action="store_true")
    parser.add_argument("--relational-appearance", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    roles = [value.strip() for value in args.roles.split(",") if value.strip()]
    method_names = [value.strip() for value in args.methods.split(",") if value.strip()]
    unknown_methods = sorted(set(method_names) - set(VARIANTS))
    if unknown_methods:
        raise ValueError(f"unknown fusion variants: {unknown_methods}")
    if not method_names:
        raise ValueError("at least one fusion variant is required")
    variants = {name: VARIANTS[name] for name in method_names}
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device == "auto":
        device = "cpu"
    cache_root = args.candidate_cache_dir or args.output / "candidate_cache"
    generator_signature = {
        "grounding_model": args.grounding_model,
        "segmentation_model": args.segmentation_model,
        "dense_semantic_fallback": args.dense_semantic_fallback,
        "prompt_bank_path": str(args.prompt_bank.resolve()),
        "prompt_bank_sha256": _sha256(args.prompt_bank),
    }
    full_bank = PromptBank.from_json(args.prompt_bank)
    character_domain = next(
        domain for domain in full_bank.domains if domain.name == "character"
    )
    character_bank = PromptBank((character_domain,))
    generator: FoundationCandidateGenerator | None = None
    frozen_primary_signature: dict[str, object] | None = None
    additional_signatures: dict[str, dict[str, object]] = {}
    source_taxonomy = Taxonomy.from_json(args.taxonomy)
    rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for role in roles:
        image = Image.open(args.image_dir / f"{role}.png").convert("RGB")
        cache_path = cache_root / f"{role}.npz"
        generation_started = time.perf_counter()
        if cache_path.exists():
            if args.frozen_cache_only:
                candidates, actual_signature = _read_candidate_cache(cache_path)
                if frozen_primary_signature is None:
                    frozen_primary_signature = actual_signature
                elif frozen_primary_signature != actual_signature:
                    raise ValueError(
                        "primary candidate source changed configuration inside "
                        f"{cache_root}"
                    )
            else:
                candidates = _load_candidates(cache_path, generator_signature)
            cache_hit = True
        else:
            if args.frozen_cache_only:
                raise FileNotFoundError(
                    f"frozen candidate cache is missing: {cache_path}"
                )
            if generator is None:
                generator = FoundationCandidateGenerator(
                    character_bank,
                    device=device,
                    config=FoundationConfig(
                        grounding_model=args.grounding_model,
                        segmentation_model=args.segmentation_model,
                        use_dense_semantic_fallback=args.dense_semantic_fallback,
                        local_files_only=args.local_files_only,
                    ),
                )
            generated = generator.generate(image)
            candidates = list(generated.candidates)
            _save_candidates(cache_path, generated.candidates, generator_signature)
            cache_hit = False
        source_candidate_counts = [len(candidates)]
        for additional_dir in args.additional_candidate_cache_dir:
            additional_path = additional_dir / f"{role}.npz"
            if not additional_path.is_file():
                raise FileNotFoundError(
                    f"additional candidate cache is missing: {additional_path}"
                )
            additional, signature = _read_candidate_cache(additional_path)
            signature_key = str(additional_dir.resolve())
            previous_signature = additional_signatures.setdefault(
                signature_key, signature
            )
            if previous_signature != signature:
                raise ValueError(
                    f"candidate source changed configuration inside {additional_dir}"
                )
            candidates.extend(additional)
            source_candidate_counts.append(len(additional))
        relational_diagnostics: dict[str, object] | None = None
        relational_candidate_count = 0
        if args.relational_appearance:
            preliminary = fuse_candidates(
                candidates,
                image_shape=(image.height, image.width),
                config=VARIANTS["parent_envelope"],
            )
            relational = propose_relational_candidates(
                image,
                preliminary.semantic_map,
                preliminary.taxonomy,
                character_bank,
            )
            relational_diagnostics = relational.diagnostics
            relational_candidate_count = len(relational.candidates)
            candidates.extend(relational.candidates)
            if relational_candidate_count:
                source_candidate_counts.append(relational_candidate_count)
        generation_seconds = time.perf_counter() - generation_started

        predictions: dict[str, np.ndarray] = {}
        variant_results: dict[str, FusionResult] = {}
        for method, config in variants.items():
            inference_started = time.perf_counter()
            result = fuse_candidates(
                candidates,
                image_shape=(image.height, image.width),
                config=config,
            )
            inference_seconds = time.perf_counter() - inference_started
            mapped = _map_prediction(result)
            predictions[method] = mapped
            variant_results[method] = result
            rows.append(
                {
                    "role": role,
                    "method": method,
                    "candidate_count": len(candidates),
                    "candidate_source_count": len(source_candidate_counts),
                    "source_candidate_counts": json.dumps(source_candidate_counts),
                    "part_count": len(result.instances),
                    "generation_seconds": generation_seconds,
                    "fusion_seconds": inference_seconds,
                    "cache_hit": cache_hit,
                    "relational_candidate_count": relational_candidate_count,
                    "relational_ground_truth_used": (
                        relational_diagnostics.get("ground_truth_used")
                        if relational_diagnostics is not None
                        else None
                    ),
                    "inference_uses_ground_truth": False,
                }
            )

        # Labels are opened only after every prediction for this role exists.
        truth_path = args.label_dir / f"{role}_final.png"
        if not truth_path.exists():
            truth_path = args.label_dir / f"{role}.png"
        native_truth = load_label_map(truth_path, source_taxonomy)
        truth = _map_truth(native_truth, source_taxonomy)
        role_dir = args.output / "visuals" / role
        role_dir.mkdir(parents=True, exist_ok=True)
        image.save(role_dir / "source.png")
        _colorize(truth).save(role_dir / "ground_truth_common.png")
        for method, prediction in predictions.items():
            metrics = evaluate_semantic(
                prediction,
                truth,
                COMMON_TAXONOMY,
                boundary_tolerance=3,
                component_iou_threshold=0.25,
                small_part_fraction=0.01,
            )
            fine_metrics = evaluate_semantic(
                _map_prediction_fine(variant_results[method], source_taxonomy),
                native_truth,
                source_taxonomy,
                boundary_tolerance=3,
                component_iou_threshold=0.25,
                small_part_fraction=0.01,
            )
            row = rows[-len(variants) + list(variants).index(method)]
            row.update(metrics)
            row.update({f"fine_{key}": value for key, value in fine_metrics.items()})
            fine_prediction = _map_prediction_fine(
                variant_results[method], source_taxonomy
            )
            accepted_candidates = variant_results[method].accepted_candidates
            for class_id, class_name in enumerate(source_taxonomy.fine_names):
                if class_id == 0:
                    continue
                truth_mask = native_truth == class_id
                truth_area = int(np.count_nonzero(truth_mask))
                if not truth_area:
                    continue
                raw_union, raw_count = _candidate_union_for_truth_class(
                    candidates, class_name, truth_mask.shape
                )
                accepted_union, accepted_count = _candidate_union_for_truth_class(
                    accepted_candidates, class_name, truth_mask.shape
                )
                final_mask = fine_prediction == class_id
                raw_precision, raw_recall = _mask_precision_recall(
                    raw_union, truth_mask
                )
                accepted_precision, accepted_recall = _mask_precision_recall(
                    accepted_union, truth_mask
                )
                final_precision, final_recall = _mask_precision_recall(
                    final_mask, truth_mask
                )
                class_rows.append(
                    {
                        "role": role,
                        "method": method,
                        "class_name": class_name,
                        "truth_area_px": truth_area,
                        "raw_candidate_count": raw_count,
                        "accepted_candidate_count": accepted_count,
                        "raw_candidate_precision": raw_precision,
                        "raw_candidate_recall": raw_recall,
                        "accepted_candidate_precision": accepted_precision,
                        "accepted_candidate_recall": accepted_recall,
                        "final_precision": final_precision,
                        "final_recall": final_recall,
                        "bottleneck": _bottleneck_label(
                            raw_recall, accepted_recall, final_recall
                        ),
                        "truth_used_after_inference": True,
                    }
                )
            _colorize(prediction).save(role_dir / f"{method}.png")
        print(
            json.dumps(
                {
                    "role": role,
                    "candidate_count": len(candidates),
                    "candidate_source_count": len(source_candidate_counts),
                    "reported_method": method_names[-1],
                    "reported_part_count": len(
                        variant_results[method_names[-1]].instances
                    ),
                    "cache_hit": cache_hit,
                }
            ),
            flush=True,
        )

    cases = pd.DataFrame(rows)
    cases.to_csv(args.output / "case_metrics.csv", index=False)
    metric_names = [
        "foreground_iou",
        "coarse_miou",
        "semantic_miou",
        "boundary_f1",
        "small_part_recall",
        "component_f1",
        "part_count_abs_error",
        "fine_foreground_iou",
        "fine_coarse_miou",
        "fine_semantic_miou",
        "fine_boundary_f1",
        "fine_small_part_recall",
        "fine_component_f1",
        "fine_part_count_abs_error",
    ]
    summary = cases.groupby("method", as_index=False)[metric_names].agg(
        ["mean", "std", "median"]
    )
    summary.to_csv(args.output / "summary.csv")
    class_cases = pd.DataFrame(class_rows)
    class_cases.to_csv(args.output / "fine_class_diagnostics.csv", index=False)
    class_summary = (
        class_cases.groupby(["method", "class_name"], as_index=False)
        .agg(
            present_roles=("role", "count"),
            truth_area_px=("truth_area_px", "sum"),
            raw_candidate_count=("raw_candidate_count", "sum"),
            accepted_candidate_count=("accepted_candidate_count", "sum"),
            raw_candidate_recall=("raw_candidate_recall", "mean"),
            accepted_candidate_recall=("accepted_candidate_recall", "mean"),
            final_recall=("final_recall", "mean"),
            final_precision=("final_precision", "mean"),
        )
        .sort_values(["method", "final_recall", "class_name"])
    )
    class_summary.to_csv(args.output / "fine_class_summary.csv", index=False)
    manifest = {
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "roles": roles,
        "role_count": len(roles),
        "variants": {name: asdict(config) for name, config in variants.items()},
        "common_taxonomy": COMMON_TAXONOMY.to_dict(),
        "truth_usage": "post-inference metric computation only",
        "inference_uses_ground_truth": False,
        "candidate_cache_dir": str(cache_root.resolve()),
        "candidate_generator": (
            frozen_primary_signature
            if frozen_primary_signature is not None
            else generator_signature
        ),
        "configured_candidate_generator": generator_signature,
        "frozen_cache_only": args.frozen_cache_only,
        "relational_appearance": args.relational_appearance,
        "relational_algorithm": (
            "hpid-relational-appearance-v1"
            if args.relational_appearance
            else None
        ),
        "additional_candidate_generators": additional_signatures,
        "elapsed_seconds": time.perf_counter() - started,
        "seed": SEED,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
