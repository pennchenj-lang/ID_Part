from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from .metrics import binary_iou, boundary_f1
from .paco_semantics import canonical_part_token, normalize_paco_name


def _normalize(
    value: str,
    expected_domain: str | None = None,
    *,
    object_category: str | None = None,
) -> str:
    normalized = normalize_paco_name(value)
    if expected_domain is None:
        return normalized
    return canonical_part_token(
        normalized,
        expected_domain,
        object_category=object_category,
    )


def _predicted_part_token(semantic_name: str, expected_domain: str) -> str:
    prefix = f"{expected_domain}_"
    semantic_name = semantic_name.removeprefix(prefix)
    return _normalize(semantic_name, expected_domain)


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) >= 128


def _hungarian_matches(
    truth_masks: list[np.ndarray], prediction_masks: list[np.ndarray]
) -> list[tuple[int, int, float]]:
    matrix = np.zeros((len(truth_masks), len(prediction_masks)), dtype=np.float32)
    for row, truth in enumerate(truth_masks):
        for column, prediction in enumerate(prediction_masks):
            matrix[row, column] = binary_iou(truth, prediction)
    if not matrix.size:
        return []
    rows, columns = linear_sum_assignment(1.0 - matrix)
    return [
        (int(row), int(column), float(matrix[row, column]))
        for row, column in zip(rows, columns, strict=True)
    ]


def _semantic_recall(
    truth_rows: list[dict[str, object]],
    truth_masks: list[np.ndarray],
    predicted_rows: list[dict[str, object]],
    predicted_masks: list[np.ndarray],
    *,
    expected_domain: str,
    object_category: str,
    threshold: float,
) -> tuple[float, list[dict[str, object]]]:
    matched = 0
    rows: list[dict[str, object]] = []
    truth_groups: dict[str, list[int]] = defaultdict(list)
    prediction_groups: dict[str, list[int]] = defaultdict(list)
    for truth_index, truth_row in enumerate(truth_rows):
        truth_groups[
            _normalize(
                str(truth_row["part_name"]),
                expected_domain,
                object_category=object_category,
            )
        ].append(truth_index)
    for prediction_index, prediction_row in enumerate(predicted_rows):
        semantic_name = str(prediction_row["semantic_name"])
        token = (
            "body"
            if semantic_name == expected_domain
            else _predicted_part_token(semantic_name, expected_domain)
        )
        prediction_groups[token].append(prediction_index)

    assigned: dict[int, tuple[int, float]] = {}
    for semantic_name, truth_indices in truth_groups.items():
        prediction_indices = prediction_groups.get(semantic_name, [])
        matches = _hungarian_matches(
            [truth_masks[index] for index in truth_indices],
            [predicted_masks[index] for index in prediction_indices],
        )
        for local_truth, local_prediction, overlap in matches:
            assigned[truth_indices[local_truth]] = (
                prediction_indices[local_prediction],
                overlap,
            )

    for truth_index, truth_row in enumerate(truth_rows):
        truth_name = _normalize(
            str(truth_row["part_name"]),
            expected_domain,
            object_category=object_category,
        )
        prediction_indices = prediction_groups.get(truth_name, [])
        assignment = assigned.get(truth_index)
        overlap = assignment[1] if assignment is not None else 0.0
        accepted = overlap >= threshold
        matched += int(accepted)
        rows.append(
            {
                "truth_part": truth_name,
                "best_same_semantic_iou": overlap,
                "accepted": accepted,
                "candidate_count": len(prediction_indices),
                "assigned_prediction": (
                    str(predicted_rows[assignment[0]]["semantic_name"])
                    if assignment is not None
                    else None
                ),
            }
        )
    return matched / max(1, len(truth_masks)), rows


def _semantic_union_metrics(
    truth_rows: list[dict[str, object]],
    truth_masks: list[np.ndarray],
    predicted_rows: list[dict[str, object]],
    predicted_masks: list[np.ndarray],
    *,
    expected_domain: str,
    object_category: str,
    threshold: float,
    boundary_tolerance: int,
) -> dict[str, object]:
    """Compare unions per semantic while retaining strict instance metrics."""

    truth_groups: dict[str, list[np.ndarray]] = defaultdict(list)
    prediction_groups: dict[str, list[np.ndarray]] = defaultdict(list)
    for row, mask in zip(truth_rows, truth_masks, strict=True):
        token = _normalize(
            str(row["part_name"]),
            expected_domain,
            object_category=object_category,
        )
        truth_groups[token].append(mask)
    for row, mask in zip(predicted_rows, predicted_masks, strict=True):
        semantic_name = str(row["semantic_name"])
        token = (
            "body"
            if semantic_name == expected_domain
            else _predicted_part_token(semantic_name, expected_domain)
        )
        prediction_groups[token].append(mask)

    rows: list[dict[str, object]] = []
    accepted = 0
    overlaps: list[float] = []
    boundaries: list[float] = []
    for token, token_truth_masks in sorted(truth_groups.items()):
        truth_union = np.logical_or.reduce(token_truth_masks)
        token_predictions = prediction_groups.get(token, [])
        prediction_union = (
            np.logical_or.reduce(token_predictions)
            if token_predictions
            else np.zeros(truth_union.shape, dtype=bool)
        )
        overlap = binary_iou(truth_union, prediction_union)
        accepted_semantic = overlap >= threshold
        accepted += int(accepted_semantic)
        overlaps.append(overlap)
        boundary = (
            boundary_f1(
                prediction_union,
                truth_union,
                tolerance=boundary_tolerance,
            )
            if token_predictions
            else 0.0
        )
        boundaries.append(boundary)
        rows.append(
            {
                "semantic": token,
                "truth_instance_count": len(token_truth_masks),
                "predicted_instance_count": len(token_predictions),
                "union_iou": overlap,
                "union_boundary_f1": boundary,
                "accepted": accepted_semantic,
            }
        )
    predicted_tokens = set(prediction_groups)
    truth_tokens = set(truth_groups)
    return {
        "semantic_union_recall": accepted / max(1, len(truth_groups)),
        "semantic_union_precision": (
            len(predicted_tokens & truth_tokens) / max(1, len(predicted_tokens))
        ),
        "mean_semantic_union_iou": float(np.mean(overlaps)) if overlaps else 0.0,
        "mean_semantic_union_boundary_f1": (
            float(np.mean(boundaries)) if boundaries else 0.0
        ),
        "semantic_union_matches": rows,
    }


def evaluate_paco_package(
    package_dir: Path,
    case_path: Path,
    *,
    expected_domain: str,
    expected_profile: str | None = None,
    match_iou_threshold: float = 0.25,
    boundary_tolerance: int = 3,
) -> dict[str, object]:
    """Evaluate one exported package against PACO after inference is complete."""

    case = json.loads(case_path.read_text(encoding="utf-8"))
    parts = json.loads((package_dir / "parts.json").read_text(encoding="utf-8"))
    diagnostics = json.loads(
        (package_dir / "inference_diagnostics.json").read_text(encoding="utf-8")
    )
    selected_domain = str(diagnostics["root_routing"].get("selected_semantic", ""))
    profile_resolution = diagnostics.get("profile_root_resolution") or {}
    selected_profiles = {
        str(profile)
        for profile in profile_resolution.get("selected_profiles", [])
        if profile
    }
    profile_rows = (
        (profile_resolution.get("profile_consensus") or {}).get("roots", [])
    )
    selected_profiles.update(
        str(row.get("selected_profile"))
        for row in profile_rows
        if isinstance(row, dict)
        and row.get("selected_profile")
        and row.get("status", "accepted") == "accepted"
    )

    case_dir = case_path.parent
    object_category = str(case["object_category"])
    truth_rows = list(case["parts"])
    truth_masks = [_load_mask(case_dir / str(row["mask_crop"])) for row in truth_rows]
    truth_object = _load_mask(case_dir / "object_mask_crop.png")

    all_predicted_rows = list(parts)
    object_prediction_masks = [
        _load_mask(package_dir / str(row["mask_visible_path"]))
        for row in all_predicted_rows
    ]
    if object_prediction_masks:
        prediction_union = np.logical_or.reduce(object_prediction_masks)
    else:
        prediction_union = np.zeros(truth_object.shape, dtype=bool)

    normalized_truth_names = {
        _normalize(
            str(row["part_name"]),
            expected_domain,
            object_category=object_category,
        )
        for row in truth_rows
    }
    include_root_body = "body" in normalized_truth_names
    predicted_rows = [
        row
        for row in parts
        if str(row.get("semantic_name", "")) != expected_domain or include_root_body
    ]
    predicted_masks = [
        _load_mask(package_dir / str(row["mask_visible_path"]))
        for row in predicted_rows
    ]

    matches = _hungarian_matches(truth_masks, predicted_masks)
    accepted_matches = [match for match in matches if match[2] >= match_iou_threshold]
    discovery_precision = (
        len(accepted_matches) / len(predicted_masks) if predicted_masks else 0.0
    )
    discovery_recall = len(accepted_matches) / len(truth_masks) if truth_masks else 0.0
    discovery_f1 = (
        2.0
        * discovery_precision
        * discovery_recall
        / (discovery_precision + discovery_recall)
        if discovery_precision + discovery_recall
        else 0.0
    )

    matched_boundaries: list[float] = []
    match_rows: list[dict[str, object]] = []
    for truth_index, prediction_index, overlap in matches:
        truth_name = _normalize(
            str(truth_rows[truth_index]["part_name"]),
            expected_domain,
            object_category=object_category,
        )
        prediction_name = _predicted_part_token(
            str(predicted_rows[prediction_index]["semantic_name"]),
            expected_domain,
        )
        if str(predicted_rows[prediction_index]["semantic_name"]) == expected_domain:
            prediction_name = "body"
        semantic_match = truth_name == prediction_name
        accepted = overlap >= match_iou_threshold
        if accepted:
            matched_boundaries.append(
                boundary_f1(
                    predicted_masks[prediction_index],
                    truth_masks[truth_index],
                    tolerance=boundary_tolerance,
                )
            )
        match_rows.append(
            {
                "truth_part": truth_name,
                "predicted_semantic": str(
                    predicted_rows[prediction_index]["semantic_name"]
                ),
                "predicted_part": prediction_name,
                "iou": overlap,
                "accepted": accepted,
                "semantic_match": semantic_match,
            }
        )

    semantic_part_recall, semantic_rows = _semantic_recall(
        truth_rows,
        truth_masks,
        predicted_rows,
        predicted_masks,
        expected_domain=expected_domain,
        object_category=object_category,
        threshold=match_iou_threshold,
    )
    semantic_union = _semantic_union_metrics(
        truth_rows,
        truth_masks,
        predicted_rows,
        predicted_masks,
        expected_domain=expected_domain,
        object_category=object_category,
        threshold=match_iou_threshold,
        boundary_tolerance=boundary_tolerance,
    )
    prediction_inside_object = np.count_nonzero(prediction_union & truth_object)
    prediction_area = np.count_nonzero(prediction_union)
    return {
        "format": "HPID PACO package evaluation",
        "format_version": "0.1.0",
        "package": str(package_dir.resolve()),
        "case": str(case_path.resolve()),
        "object_category": object_category,
        "expected_domain": expected_domain,
        "selected_domain": selected_domain,
        "domain_correct": selected_domain == expected_domain,
        "expected_profile": expected_profile,
        "selected_profiles": sorted(selected_profiles),
        "profile_correct": (
            expected_profile in selected_profiles
            if expected_profile is not None
            else None
        ),
        "object_iou": binary_iou(prediction_union, truth_object),
        "object_precision": (
            prediction_inside_object / prediction_area if prediction_area else 0.0
        ),
        "object_recall": (
            prediction_inside_object / max(1, np.count_nonzero(truth_object))
        ),
        "truth_part_count": len(truth_masks),
        "predicted_part_count": len(predicted_masks),
        "generic_predicted_part_count": sum(
            "_visual_" in str(row.get("semantic_name", "")) for row in predicted_rows
        ),
        "matched_part_count": len(accepted_matches),
        "part_discovery_precision_at_025": discovery_precision,
        "part_discovery_recall_at_025": discovery_recall,
        "part_discovery_f1_at_025": discovery_f1,
        "mean_matched_iou": (
            float(np.mean([match[2] for match in accepted_matches]))
            if accepted_matches
            else 0.0
        ),
        "mean_matched_boundary_f1": (
            float(np.mean(matched_boundaries)) if matched_boundaries else 0.0
        ),
        "semantic_part_recall": semantic_part_recall,
        "semantic_matches": semantic_rows,
        **semantic_union,
        "oversegmentation_ratio": len(predicted_masks) / max(1, len(truth_masks)),
        "matches": match_rows,
        "evaluation_reads_ground_truth_after_inference": True,
        "inference_uses_ground_truth": False,
    }
