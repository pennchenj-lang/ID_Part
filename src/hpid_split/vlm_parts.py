from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

from .foundation import CandidateGeneration, Detection, SegmentProposal
from .fusion import MaskCandidate, mask_iou
from .prompt_bank import DomainPrompt, PartPrompt


@dataclass(frozen=True)
class VlmPartConfig:
    crop_padding_ratio: float = 0.06
    minimum_confidence: float = 0.30
    minimum_box_side_px: int = 3
    same_semantic_box_nms_iou: float = 0.84
    same_semantic_mask_nms_iou: float = 0.86
    minimum_raw_root_containment: float = 0.20
    minimum_parent_overlap: float = 0.05
    parent_dilation_ratio: float = 0.025
    minimum_area_px: int = 12
    minimum_area_tolerance: float = 0.35
    maximum_area_tolerance: float = 1.30
    source_reliability: float = 0.82
    maximum_inventory_size: int = 48
    maximum_planner_queries: int = 12
    maximum_total_planner_queries: int = 24
    maximum_roots: int = 8
    use_region_label_queries: bool = True
    use_batched_region_label_queries: bool = False
    region_label_batch_size: int = 8
    maximum_region_label_queries: int = 12
    region_label_minimum_confidence: float = 0.65
    region_label_corroboration_iou: float = 0.18
    region_label_corroboration_containment: float = 0.62
    maximum_exterior_contact_fraction: float = 0.48
    exterior_contact_dilation_px: int = 2
    weak_semantic_maximum_probability: float = 0.45
    weak_semantic_maximum_margin: float = 0.08
    use_box_planner_queries: bool = False
    use_per_semantic_queries: bool = True
    query_established_semantics: bool = False
    assignment_minimum_score: float = 0.48
    assignment_minimum_box_containment: float = 0.45
    assignment_semantic_bonus: float = 0.18
    assignment_direct_sam_penalty: float = 0.08
    assignment_panel_minimum_box_fill: float = 0.02
    assignment_panel_minimum_box_iou: float = 0.06
    allow_direct_sam_regions: bool = False
    use_dynamic_inventory: bool = False
    maximum_dynamic_inventory_roots: int = 2
    maximum_dynamic_parts: int = 10
    dynamic_part_minimum_confidence: float = 0.72
    dynamic_object_minimum_confidence: float = 0.75


@dataclass(frozen=True)
class PlannedPart:
    semantic_name: str
    box_xyxy: tuple[int, int, int, int]
    confidence: float
    instance_hint: str | None = None


@dataclass(frozen=True)
class ParsedPartPlan:
    parts: tuple[PlannedPart, ...]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class DynamicPartInventory:
    parts: tuple[PartPrompt, ...]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class DynamicObjectIdentity:
    label: str | None
    confidence: float
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class RegionLabelPlan:
    semantic_name: str | None
    confidence: float
    diagnostics: dict[str, object]
    entity_kind: str | None = None


@dataclass(frozen=True)
class RegionOwnershipPlan:
    is_semantic_part: bool
    region_kind: str | None
    confidence: float
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class PartRegion:
    mask: np.ndarray
    source: str
    quality: float
    candidate_key: str
    semantic_name: str | None = None
    generic: bool = True
    direct_plan_index: int | None = None
    region_kind: str | None = None


@dataclass(frozen=True)
class PlanRegionAssignment:
    plan_index: int
    region_index: int
    score: float
    box_containment: float
    box_fill: float
    box_iou: float
    area_prior: float
    semantic_support: bool


class VlmPlanner(Protocol):
    backend_id: str

    def generate_response(self, image: Image.Image, prompt: str) -> str: ...


BoxSegmenter = Callable[
    [Image.Image, list[Detection]],
    list[SegmentProposal],
]


def _normalized_semantic(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _extract_json_object(text: str) -> dict[str, object] | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _box_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(
        0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _area_prior_score(area_fraction: float, part: PartPrompt) -> float:
    minimum = part.minimum_parent_fraction
    maximum = part.maximum_parent_fraction
    if minimum <= area_fraction <= maximum:
        return 1.0
    if area_fraction < minimum:
        return max(0.0, area_fraction / max(minimum, 1e-8))
    return max(0.0, maximum / max(area_fraction, 1e-8))


def assign_plans_to_regions(
    plans: Sequence[PlannedPart],
    regions: Sequence[PartRegion],
    *,
    allowed_parts: Mapping[str, PartPrompt],
    root_mask: np.ndarray,
    config: VlmPartConfig | None = None,
) -> tuple[tuple[PlanRegionAssignment, ...], dict[str, object]]:
    """Globally match semantic plans to masks with one-region ownership."""

    config = config or VlmPartConfig()
    if not plans or not regions:
        return (), {
            "algorithm": "hpid-vlm-region-bipartite-matching-v1",
            "plan_count": len(plans),
            "region_count": len(regions),
            "assignment_count": 0,
            "ground_truth_used": False,
        }
    root_area = max(1, int(np.count_nonzero(root_mask)))
    score_matrix = np.full((len(plans), len(regions)), -1.0, dtype=np.float64)
    feature_rows: dict[tuple[int, int], tuple[float, float, float, float, bool]] = {}
    for plan_index, plan in enumerate(plans):
        part = allowed_parts[plan.semantic_name]
        x0, y0, x1, y1 = plan.box_xyxy
        box_mask = np.zeros(root_mask.shape, dtype=bool)
        box_mask[y0:y1, x0:x1] = True
        box_area = max(1, int(np.count_nonzero(box_mask)))
        for region_index, region in enumerate(regions):
            if (
                region.direct_plan_index is not None
                and region.direct_plan_index != plan_index
            ):
                continue
            semantic_support = region.semantic_name == plan.semantic_name
            if (
                region.semantic_name is not None
                and not region.generic
                and not semantic_support
            ):
                continue
            if (
                region.generic
                and region.region_kind == "detail"
                and not part.detail
            ):
                continue
            mask = region.mask.astype(bool) & root_mask
            mask_area = int(np.count_nonzero(mask))
            if mask_area < config.minimum_area_px:
                continue
            intersection = int(np.count_nonzero(mask & box_mask))
            box_containment = intersection / mask_area
            if (
                box_containment < config.assignment_minimum_box_containment
                and not (
                    semantic_support
                    and box_containment
                    >= config.minimum_parent_overlap
                )
            ):
                continue
            box_fill = intersection / box_area
            region_box = _mask_bbox(mask)
            region_box_iou = _box_iou(region_box, plan.box_xyxy)
            if (
                region.generic
                and not part.detail
                and box_fill < config.assignment_panel_minimum_box_fill
                and region_box_iou < config.assignment_panel_minimum_box_iou
            ):
                continue
            centroid_y, centroid_x = np.mean(np.argwhere(mask), axis=0)
            center_inside = float(
                x0 <= centroid_x < x1 and y0 <= centroid_y < y1
            )
            area_prior = _area_prior_score(mask_area / root_area, part)
            score = (
                0.42 * box_containment
                + 0.12 * min(1.0, box_fill * 3.0)
                + 0.13 * region_box_iou
                + 0.12 * center_inside
                + 0.13 * area_prior
                + config.assignment_semantic_bonus * float(semantic_support)
                + 0.08 * float(np.clip(region.quality, 0.0, 1.0))
                - config.assignment_direct_sam_penalty
                * float(region.direct_plan_index is not None)
            )
            score_matrix[plan_index, region_index] = score
            feature_rows[(plan_index, region_index)] = (
                box_containment,
                box_fill,
                region_box_iou,
                area_prior,
                semantic_support,
            )

    # One private dummy column per plan permits an honest unmatched result.
    augmented = np.concatenate(
        [score_matrix, np.zeros((len(plans), len(plans)), dtype=np.float64)],
        axis=1,
    )
    row_indices, column_indices = linear_sum_assignment(augmented, maximize=True)
    assignments: list[PlanRegionAssignment] = []
    rejected_below_threshold = 0
    for plan_index, column_index in zip(row_indices, column_indices, strict=True):
        if column_index >= len(regions):
            continue
        score = float(score_matrix[plan_index, column_index])
        if score < config.assignment_minimum_score:
            rejected_below_threshold += 1
            continue
        features = feature_rows[(plan_index, column_index)]
        assignments.append(
            PlanRegionAssignment(
                plan_index,
                column_index,
                score,
                features[0],
                features[1],
                features[2],
                features[3],
                features[4],
            )
        )
    assignments.sort(key=lambda item: item.plan_index)
    return tuple(assignments), {
        "algorithm": "hpid-vlm-region-bipartite-matching-v1",
        "plan_count": len(plans),
        "region_count": len(regions),
        "feasible_pair_count": int(np.count_nonzero(score_matrix >= 0.0)),
        "assignment_count": len(assignments),
        "unmatched_plan_count": len(plans) - len(assignments),
        "rejected_below_threshold_count": rejected_below_threshold,
        "ground_truth_used": False,
    }


def _convert_box(
    values: object,
    *,
    coordinate_system: str,
    image_size: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        return None
    try:
        coordinates = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in coordinates):
        return None
    width, height = image_size
    if coordinate_system == "normalized_1000":
        x0, y0, x1, y1 = (
            coordinates[0] * width / 1000.0,
            coordinates[1] * height / 1000.0,
            coordinates[2] * width / 1000.0,
            coordinates[3] * height / 1000.0,
        )
    elif coordinate_system == "pixels":
        x0, y0, x1, y1 = coordinates
    else:
        return None
    converted = (
        max(0, min(width, round(x0))),
        max(0, min(height, round(y0))),
        max(0, min(width, round(x1))),
        max(0, min(height, round(y1))),
    )
    if converted[2] <= converted[0] or converted[3] <= converted[1]:
        return None
    return converted


def parse_part_plan(
    response: str,
    *,
    image_size: tuple[int, int],
    allowed_parts: Mapping[str, PartPrompt],
    config: VlmPartConfig | None = None,
) -> ParsedPartPlan:
    """Parse a bounded VLM proposal without inventing ontology labels."""

    config = config or VlmPartConfig()
    payload = _extract_json_object(response)
    rejection_counts = {
        "malformed_row": 0,
        "unknown_semantic": 0,
        "not_visible": 0,
        "low_confidence": 0,
        "invalid_box": 0,
        "box_too_small": 0,
        "box_area_outside_prior": 0,
        "duplicate_box": 0,
        "instance_cap": 0,
    }
    if payload is None:
        return ParsedPartPlan(
            (),
            {
                "status": "invalid_json",
                "response_character_count": len(response),
                "rejection_counts": rejection_counts,
                "ground_truth_used": False,
            },
        )
    coordinate_system = str(payload.get("coordinate_system", ""))
    if coordinate_system not in {"normalized_1000", "pixels"}:
        return ParsedPartPlan(
            (),
            {
                "status": "invalid_coordinate_system",
                "coordinate_system": coordinate_system,
                "response_character_count": len(response),
                "rejection_counts": rejection_counts,
                "ground_truth_used": False,
            },
        )
    raw_parts = payload.get("parts")
    if not isinstance(raw_parts, list):
        return ParsedPartPlan(
            (),
            {
                "status": "missing_parts_array",
                "coordinate_system": coordinate_system,
                "response_character_count": len(response),
                "rejection_counts": rejection_counts,
                "ground_truth_used": False,
            },
        )

    canonical = {_normalized_semantic(name): name for name in allowed_parts}
    width, height = image_size
    image_area = max(1, width * height)
    accepted: list[PlannedPart] = []
    for row in raw_parts:
        if not isinstance(row, dict):
            rejection_counts["malformed_row"] += 1
            continue
        raw_name = row.get("semantic_name")
        if not isinstance(raw_name, str):
            rejection_counts["malformed_row"] += 1
            continue
        semantic_name = canonical.get(_normalized_semantic(raw_name))
        if semantic_name is None:
            rejection_counts["unknown_semantic"] += 1
            continue
        if row.get("visible") is False:
            rejection_counts["not_visible"] += 1
            continue
        try:
            confidence = float(row["confidence"])
        except (KeyError, TypeError, ValueError):
            rejection_counts["malformed_row"] += 1
            continue
        if not math.isfinite(confidence) or confidence < config.minimum_confidence:
            rejection_counts["low_confidence"] += 1
            continue
        box = _convert_box(
            row.get("bbox_2d"),
            coordinate_system=coordinate_system,
            image_size=image_size,
        )
        if box is None:
            rejection_counts["invalid_box"] += 1
            continue
        box_width = box[2] - box[0]
        box_height = box[3] - box[1]
        if min(box_width, box_height) < config.minimum_box_side_px:
            rejection_counts["box_too_small"] += 1
            continue
        part = allowed_parts[semantic_name]
        box_fraction = box_width * box_height / image_area
        if (
            box_fraction
            < part.minimum_parent_fraction * config.minimum_area_tolerance
            or box_fraction
            > min(
                1.0,
                part.maximum_parent_fraction * config.maximum_area_tolerance,
            )
        ):
            rejection_counts["box_area_outside_prior"] += 1
            continue
        duplicate = any(
            item.semantic_name == semantic_name
            and _box_iou(item.box_xyxy, box)
            >= config.same_semantic_box_nms_iou
            for item in accepted
        )
        if duplicate:
            rejection_counts["duplicate_box"] += 1
            continue
        instance_hint = row.get("instance_hint")
        accepted.append(
            PlannedPart(
                semantic_name,
                box,
                float(np.clip(confidence, 0.0, 1.0)),
                str(instance_hint) if instance_hint is not None else None,
            )
        )

    bounded: list[PlannedPart] = []
    for semantic_name in allowed_parts:
        semantic_parts = sorted(
            (item for item in accepted if item.semantic_name == semantic_name),
            key=lambda item: item.confidence,
            reverse=True,
        )
        maximum_instances = allowed_parts[semantic_name].maximum_instances
        bounded.extend(semantic_parts[:maximum_instances])
        rejection_counts["instance_cap"] += max(
            0, len(semantic_parts) - maximum_instances
        )
    return ParsedPartPlan(
        tuple(bounded),
        {
            "status": "parsed",
            "coordinate_system": coordinate_system,
            "raw_part_count": len(raw_parts),
            "accepted_part_count": len(bounded),
            "rejection_counts": rejection_counts,
            "response_character_count": len(response),
            "ground_truth_used": False,
        },
    )


_DYNAMIC_PART_BANNED_TOKENS = {
    "background",
    "blur",
    "color",
    "glare",
    "highlight",
    "lighting",
    "material",
    "pattern",
    "reflection",
    "shadow",
    "texture",
    "watermark",
}


def _inventory_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if token not in {"part", "component", "object", "asset"}
    }


def build_dynamic_object_identity_prompt(
    *,
    candidate_object_label: str,
    domain_name: str,
) -> str:
    """Build a short, inventory-independent object identity request."""

    return f"""Identify the single visible physical object in the masked crop. Its
broad domain is {domain_name!r}. A retrieval system suggested
{candidate_object_label!r}, but that candidate is untrusted and may be wrong. Judge the
image independently. Return a short concrete object type, not a component, material,
style, background, or generic word such as object/item. Do not list parts or explain.

Return JSON only on one line:
{{"object":"short concrete object name","object_confidence":0.0,
"candidate_label_matches":false}}
"""


def build_dynamic_part_inventory_prompt(
    *,
    object_label: str,
    domain_name: str,
    existing_parts: Sequence[PartPrompt],
    maximum_parts: int,
) -> str:
    existing_rows = []
    for part in existing_parts:
        aliases = ", ".join(part.prompts[:3])
        existing_rows.append(f"- {part.semantic_name}: {aliases}")
    existing_inventory = "\n".join(existing_rows) or "- none"
    return f"""Inspect the single masked object from broad domain {domain_name!r}. The
retrieval system suggested {object_label!r}, but that candidate is untrusted and may be
wrong. First identify the visible physical object yourself using a short concrete noun
phrase. Then propose only visible, physically meaningful component TYPES missing from
the existing inventory below. If the suggested object is wrong, ignore its irrelevant
inventory entries and describe components of the object actually shown. A component
must have its own coherent boundary and be useful as an independently editable Part ID.
In short, find physical components missing from the existing inventory.
Do not return the whole object, background, holes, gaps, shadows, reflections,
highlights, printed texture, color patches, material regions, or synonyms of an
existing component. Repeated physical parts such as wheels or buttons use one type with
maximum_instances greater than one. Omit uncertain and hidden components. Return at
most {maximum_parts} types.

Existing inventory:
{existing_inventory}

Return JSON only:
{{"object":"short concrete object name","object_confidence":0.0,
"candidate_label_matches":false,"parts":[
{{"name":"short physical noun phrase","description":"precise visible physical
definition","confidence":0.0,"visible":true,"physical":true,
"detail":false,"maximum_instances":1}}
]}}
"""


def parse_dynamic_object_identity(
    response: str,
    *,
    config: VlmPartConfig | None = None,
) -> DynamicObjectIdentity:
    """Parse an open-set object name without allowing it to change the broad domain."""

    config = config or VlmPartConfig()
    payload = _extract_json_object(response)
    diagnostics: dict[str, object] = {
        "response_character_count": len(response),
        "ground_truth_used": False,
    }
    if payload is None:
        return DynamicObjectIdentity(
            None,
            0.0,
            {**diagnostics, "status": "invalid_json"},
        )
    raw_label = payload.get("object")
    try:
        confidence = float(payload.get("object_confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if not isinstance(raw_label, str):
        return DynamicObjectIdentity(
            None,
            confidence,
            {**diagnostics, "status": "missing_object_label"},
        )
    label = re.sub(r"\s+", " ", raw_label.strip().lower())
    tokens = _inventory_tokens(label)
    generic_tokens = {
        "asset",
        "item",
        "object",
        "other",
        "thing",
        "uncertain",
        "unknown",
    }
    valid_label = bool(
        tokens
        and not tokens <= generic_tokens
        and len(tokens) <= 7
        and len(label) <= 80
    )
    accepted = bool(
        valid_label
        and math.isfinite(confidence)
        and confidence >= config.dynamic_object_minimum_confidence
    )
    return DynamicObjectIdentity(
        label if accepted else None,
        confidence if math.isfinite(confidence) else 0.0,
        {
            **diagnostics,
            "status": "accepted" if accepted else "rejected",
            "proposed_label": label or None,
            "confidence": confidence if math.isfinite(confidence) else 0.0,
            "candidate_label_matches": payload.get("candidate_label_matches"),
            "minimum_confidence": config.dynamic_object_minimum_confidence,
        },
    )


def _candidate_root_identity_key(candidate: MaskCandidate) -> str:
    return (
        f"{candidate.metadata.get('root_origin', 'legacy')}::"
        f"{candidate.metadata.get('root_index', 'unknown')}"
    )


def apply_dynamic_object_profile_corrections(
    candidates: Sequence[MaskCandidate],
    root_rows: Sequence[Mapping[str, object]],
    domains: Mapping[str, DomainPrompt],
) -> CandidateGeneration:
    """Remove candidates produced by a profile that open-set vision disproved."""

    corrections: dict[str, dict[str, object]] = {}
    for row in root_rows:
        root_key = str(row.get("root_key") or "")
        domain_name = str(row.get("domain") or "")
        initial_profile = row.get("initial_selected_profile")
        selected_profile = row.get("selected_profile")
        dynamic_inventory = row.get("dynamic_inventory")
        inventory = (
            dynamic_inventory if isinstance(dynamic_inventory, Mapping) else {}
        )
        object_identity = inventory.get("object_identity")
        identity = object_identity if isinstance(object_identity, Mapping) else {}
        resolved_label = str(inventory.get("resolved_object_label") or "").strip()
        domain = domains.get(domain_name)
        if (
            not root_key
            or domain is None
            or identity.get("status") != "accepted"
            or not resolved_label
        ):
            continue
        selected_parts, confirmed_profile, _ = domain.select_parts(
            resolved_label,
            profile_hint=(str(selected_profile) if selected_profile else None),
            profile_hint_source="vlm_open_set_object_identity",
        )
        final_profile = confirmed_profile or (
            str(selected_profile) if selected_profile else None
        )
        corrections[root_key] = {
            "domain": domain_name,
            "resolved_object_label": resolved_label,
            "initial_profile": (
                str(initial_profile) if initial_profile is not None else None
            ),
            "selected_profile": final_profile,
            "profile_changed": final_profile
            != (str(initial_profile) if initial_profile is not None else None),
            "allowed_semantics": {
                domain_name,
                *(part.semantic_name for part in selected_parts),
            },
        }
    if not corrections:
        return CandidateGeneration(
            tuple(candidates),
            {
                "algorithm": "hpid-vlm-open-set-profile-correction-v1",
                "corrected_root_count": 0,
                "dropped_candidate_count": 0,
                "ground_truth_used": False,
            },
        )

    output: list[MaskCandidate] = []
    dropped_rows: list[dict[str, object]] = []
    for candidate in candidates:
        root_key = _candidate_root_identity_key(candidate)
        correction = corrections.get(root_key)
        if correction is None:
            output.append(candidate)
            continue
        is_root = bool(
            candidate.metadata.get("parent_candidate_key") is None
            and candidate.semantic_name == candidate.semantic_parent
        )
        is_generic_visual = bool(candidate.metadata.get("generic_visual_region"))
        allowed_semantics = correction["allowed_semantics"]
        assert isinstance(allowed_semantics, set)
        profile_changed = bool(correction["profile_changed"])
        if (
            profile_changed
            and not is_root
            and not is_generic_visual
            and candidate.semantic_name not in allowed_semantics
        ):
            dropped_rows.append(
                {
                    "root_key": root_key,
                    "semantic_name": candidate.semantic_name,
                    "source": candidate.source,
                }
            )
            continue
        metadata = dict(candidate.metadata)
        metadata.update(
            {
                "resolved_object_label": correction["resolved_object_label"],
                "selected_part_profile": correction["selected_profile"],
                "profile_hint_source": "vlm_open_set_object_identity",
                "vlm_object_identity_applied": True,
                "ground_truth_used": False,
            }
        )
        output.append(
            MaskCandidate(
                candidate.semantic_name,
                candidate.semantic_parent,
                candidate.mask,
                candidate.score,
                candidate.source,
                prompt=(
                    str(correction["resolved_object_label"])
                    if is_root
                    else candidate.prompt
                ),
                source_reliability=candidate.source_reliability,
                metadata=metadata,
            )
        )
    correction_rows = [
        {
            key: value
            for key, value in correction.items()
            if key != "allowed_semantics"
        }
        | {"root_key": root_key}
        for root_key, correction in corrections.items()
    ]
    return CandidateGeneration(
        tuple(output),
        {
            "algorithm": "hpid-vlm-open-set-profile-correction-v1",
            "corrected_root_count": sum(
                bool(row["profile_changed"]) for row in correction_rows
            ),
            "identity_updated_root_count": len(correction_rows),
            "dropped_candidate_count": len(dropped_rows),
            "corrections": correction_rows,
            "dropped_candidates": dropped_rows,
            "ground_truth_used": False,
        },
    )


def parse_dynamic_part_inventory(
    response: str,
    *,
    domain_name: str,
    existing_parts: Sequence[PartPrompt],
    config: VlmPartConfig | None = None,
) -> DynamicPartInventory:
    """Parse open labels; masks still require independent region-level evidence."""

    config = config or VlmPartConfig()
    payload = _extract_json_object(response)
    rejection_counts = {
        "malformed_row": 0,
        "not_visible_or_physical": 0,
        "low_confidence": 0,
        "invalid_name": 0,
        "incidental_surface": 0,
        "existing_synonym": 0,
        "duplicate_name": 0,
        "capacity": 0,
    }
    if payload is None or not isinstance(payload.get("parts"), list):
        return DynamicPartInventory(
            (),
            {
                "status": "invalid_json_or_parts",
                "response_character_count": len(response),
                "rejection_counts": rejection_counts,
                "ground_truth_used": False,
            },
        )

    raw_parts = payload["parts"]
    assert isinstance(raw_parts, list)
    existing_token_sets: list[set[str]] = []
    for part in existing_parts:
        names = [part.semantic_name, *part.prompts, *part.aliases]
        tokens = set().union(*(_inventory_tokens(name) for name in names))
        tokens.discard(domain_name)
        if tokens:
            existing_token_sets.append(tokens)

    accepted: list[PartPrompt] = []
    accepted_slugs: set[str] = set()
    for row in raw_parts:
        if len(accepted) >= config.maximum_dynamic_parts:
            rejection_counts["capacity"] += 1
            continue
        if not isinstance(row, dict):
            rejection_counts["malformed_row"] += 1
            continue
        if row.get("visible") is False or row.get("physical") is not True:
            rejection_counts["not_visible_or_physical"] += 1
            continue
        try:
            confidence = float(row.get("confidence", 0.0))
        except (TypeError, ValueError):
            rejection_counts["malformed_row"] += 1
            continue
        if (
            not math.isfinite(confidence)
            or confidence < config.dynamic_part_minimum_confidence
        ):
            rejection_counts["low_confidence"] += 1
            continue
        raw_name = row.get("name")
        if not isinstance(raw_name, str):
            rejection_counts["invalid_name"] += 1
            continue
        name = re.sub(r"\s+", " ", raw_name.strip().lower())
        tokens = _inventory_tokens(name)
        if not tokens or len(tokens) > 5 or len(name) > 64:
            rejection_counts["invalid_name"] += 1
            continue
        if tokens & _DYNAMIC_PART_BANNED_TOKENS:
            rejection_counts["incidental_surface"] += 1
            continue
        if any(
            tokens <= existing
            or existing <= tokens
            or len(tokens & existing) / max(1, len(tokens | existing)) >= 0.72
            for existing in existing_token_sets
        ):
            rejection_counts["existing_synonym"] += 1
            continue
        slug = _normalized_semantic(name)
        if not slug or slug in accepted_slugs:
            rejection_counts["duplicate_name"] += 1
            continue
        try:
            maximum_instances = int(row.get("maximum_instances", 1))
        except (TypeError, ValueError):
            maximum_instances = 1
        maximum_instances = int(np.clip(maximum_instances, 1, 8))
        detail = bool(row.get("detail", False))
        raw_description = row.get("description")
        description = (
            re.sub(r"\s+", " ", raw_description.strip())[:240]
            if isinstance(raw_description, str) and raw_description.strip()
            else name
        )
        semantic_name = f"{domain_name}_dynamic_{slug}"
        accepted.append(
            PartPrompt(
                semantic_name=semantic_name,
                prompts=(name,),
                semantic_parent=domain_name,
                planner_description=description,
                minimum_parent_fraction=0.00005 if detail else 0.0005,
                maximum_parent_fraction=0.24 if detail else 0.72,
                maximum_instances=maximum_instances,
                detail=detail,
                priority=0.96,
            )
        )
        accepted_slugs.add(slug)

    return DynamicPartInventory(
        tuple(accepted),
        {
            "status": "ok",
            "response_character_count": len(response),
            "raw_part_count": len(raw_parts),
            "accepted_part_count": len(accepted),
            "accepted_semantics": [part.semantic_name for part in accepted],
            "rejection_counts": rejection_counts,
            "ground_truth_used": False,
        },
    )


def build_part_planner_prompt(
    *,
    object_label: str,
    domain_name: str,
    parts: Sequence[PartPrompt],
    context_parts: Sequence[PartPrompt] | None = None,
) -> str:
    inventory_rows = []
    for part in parts:
        aliases = ", ".join(part.prompts[:3])
        description = (
            f"; physical definition={part.planner_description}"
            if part.planner_description
            else ""
        )
        exclusions = (
            f"; do not confuse with [{', '.join(part.planner_exclusions)}]"
            if part.planner_exclusions
            else ""
        )
        inventory_rows.append(
            f'- {part.semantic_name}: visible name hints [{aliases}]; '
            f"parent={part.semantic_parent or domain_name}; "
            f"maximum_instances={part.maximum_instances}"
            f"{description}{exclusions}"
        )
    inventory = "\n".join(inventory_rows)
    requested_names = {part.semantic_name for part in parts}
    context_rows = []
    for part in context_parts or ():
        if part.semantic_name in requested_names:
            continue
        aliases = ", ".join(part.prompts[:2])
        context_rows.append(f"- {part.semantic_name}: [{aliases}]")
    context_inventory = (
        "\n\nVisible neighboring labels for disambiguation only; do not emit them "
        "in this query:\n" + "\n".join(context_rows)
        if context_rows
        else ""
    )
    return f"""You are a visual component planner for an editable asset pipeline.
Inspect the single masked object named {object_label!r} in domain {domain_name!r}.
Propose tight boxes for distinct, physically meaningful, visibly separable parts.
Repeated physical parts must be separate rows. Do not propose background, shadows,
reflections, highlights, printed texture, or a whole-object duplicate. Use only the
canonical semantic_name values in the allowed inventory. Omit uncertain or hidden
parts. Compare each target against the neighboring labels before answering. A high
confidence value cannot compensate for an imprecise or whole-object box. Never invent
a label.

Requested output inventory:
{inventory}{context_inventory}

Return JSON only, using exactly this schema:
{{"coordinate_system":"normalized_1000","object":"{object_label}","parts":[
{{"semantic_name":"allowed_key","bbox_2d":[x1,y1,x2,y2],
"confidence":0.0,"visible":true,"instance_hint":"optional short identifier"}}
]}}
Coordinates are integers from 0 to 1000 relative to the supplied crop.
"""


def build_region_ownership_prompt(
    *,
    object_label: str,
    domain_name: str,
    proposal_kind: str | None,
) -> str:
    return f"""Audit one red-outlined candidate inside the masked object
{object_label!r} from domain {domain_name!r}. The left panel shows the full object and
the right panel enlarges exactly the same candidate. Other pixels are dimmed.

Set is_semantic_part=true only when the red boundary follows one coherent, meaningful
editable part of this object. A complete physical component remains valid even when it
contains text, reflections, or texture. A complete named eye, symbol, or emblem may be
a semantic_surface_feature.

Set is_semantic_part=false for:
- background visible through an opening or gap;
- pixels from another object or the surroundings;
- only a watermark, text fragment, reflection, shadow, color patch, or material grain;
- a mixed region spanning multiple components;
- an uncertain or badly bounded fragment.

Examples: a whole wooden gun handguard is physical_component; the wood grain alone is
incidental_surface_pattern. A whole display rectangle is physical_component; watermark
letters inside it are incidental_surface_pattern. Bright background seen between chair
legs is background_through_opening, not a seat. A detached object beside a stool is
other_object_or_background, not a leg.

The proposal generator called this a {proposal_kind or 'unknown'!r}; this is only a
shape hint and may be wrong. Return JSON only:
{{"is_semantic_part":true,
"region_kind":"physical_component|semantic_surface_feature|incidental_surface_pattern|background_through_opening|other_object_or_background|mixed_region|uncertain",
"confidence":0.0}}
"""


def parse_region_ownership_plan(
    response: str,
    *,
    minimum_confidence: float = 0.55,
) -> RegionOwnershipPlan:
    payload = _extract_json_object(response)
    diagnostics: dict[str, object] = {
        "response_character_count": len(response),
        "ground_truth_used": False,
    }
    if payload is None:
        return RegionOwnershipPlan(
            False,
            None,
            0.0,
            {**diagnostics, "status": "invalid_json"},
        )
    is_semantic_part = payload.get("is_semantic_part")
    if not isinstance(is_semantic_part, bool):
        return RegionOwnershipPlan(
            False,
            None,
            0.0,
            {**diagnostics, "status": "missing_semantic_part_gate"},
        )
    raw_kind = payload.get("region_kind")
    if not isinstance(raw_kind, str):
        return RegionOwnershipPlan(
            False,
            None,
            0.0,
            {**diagnostics, "status": "missing_region_kind"},
        )
    region_kind = _normalized_semantic(raw_kind)
    allowed_kinds = {
        "physical_component",
        "semantic_surface_feature",
        "incidental_surface_pattern",
        "background_through_opening",
        "other_object_or_background",
        "mixed_region",
        "uncertain",
    }
    if region_kind not in allowed_kinds:
        return RegionOwnershipPlan(
            False,
            region_kind,
            0.0,
            {**diagnostics, "status": "unknown_region_kind"},
        )
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = (
        float(np.clip(confidence, 0.0, 1.0))
        if math.isfinite(confidence)
        else 0.0
    )
    accepted_kind = region_kind in {
        "physical_component",
        "semantic_surface_feature",
    }
    if not is_semantic_part or not accepted_kind:
        return RegionOwnershipPlan(
            False,
            region_kind,
            confidence,
            {**diagnostics, "status": "nonsemantic_region"},
        )
    if confidence < minimum_confidence:
        return RegionOwnershipPlan(
            False,
            region_kind,
            confidence,
            {**diagnostics, "status": "low_confidence"},
        )
    return RegionOwnershipPlan(
        True,
        region_kind,
        confidence,
        {**diagnostics, "status": "accepted"},
    )


def build_region_label_prompt(
    *,
    object_label: str,
    domain_name: str,
    parts: Sequence[PartPrompt],
    region_kind: str | None,
) -> str:
    rows = []
    for part in parts:
        aliases = ", ".join(part.prompts[:3])
        description = part.planner_description or aliases
        exclusions = (
            f"; exclude [{', '.join(part.planner_exclusions)}]"
            if part.planner_exclusions
            else ""
        )
        rows.append(
            f"- {part.semantic_name}: {description}; hints [{aliases}]"
            f"{exclusions}"
        )
    inventory = "\n".join(rows)
    return f"""Classify one red-outlined candidate region of the masked object
{object_label!r} in domain {domain_name!r}. The left panel gives full context and the
right panel enlarges the exact same boundary. The proposed region kind is
{region_kind or 'unknown'!r}.

Choose exactly one canonical label only when the highlighted boundary follows one
coherent, meaningful editable component and matches its physical definition, location,
and exclusions. Return unknown for background through an opening, another object,
watermarks, text fragments, reflections, shadows, isolated color or material patterns,
mixed components, imprecise boundaries, and unmatched regions. A complete component
remains valid when it contains texture or text. Do not infer hidden content, classify a
whole-object duplicate, or emit a label outside the inventory. Confidence cannot
compensate for a mismatched boundary.

Allowed labels:
{inventory}

Return JSON only:
{{"semantic_name":"allowed_key_or_unknown","confidence":0.0,
"matches_target_region":true}}
"""


def build_region_batch_label_prompt(
    *,
    object_label: str,
    domain_name: str,
    parts: Sequence[PartPrompt],
    region_specs: Sequence[tuple[str, str]],
) -> str:
    """Build one closed-world request for several independently proposed masks."""

    inventory_rows = []
    for part in parts:
        aliases = ", ".join(part.prompts[:3])
        description = part.planner_description or aliases
        exclusions = (
            f"; exclude [{', '.join(part.planner_exclusions)}]"
            if part.planner_exclusions
            else ""
        )
        inventory_rows.append(
            f"- {part.semantic_name}: {description}; hints [{aliases}]"
            f"{exclusions}"
        )
    inventory = "\n".join(inventory_rows)
    regions = "\n".join(
        f"- {region_id}: proposal kind {proposal_kind!r}"
        for region_id, proposal_kind in region_specs
    )
    return f"""Classify several numbered candidate regions of the same masked object
{object_label!r} in domain {domain_name!r}. Every tile is labelled R1, R2, and so on.
Each tile shows full-object context on the left and a close-up of exactly the same
red-outlined candidate on the right. Masks were proposed independently; you may name
them but must not invent, resize, merge, or split a mask.

Choose one canonical label only when the outlined boundary follows one coherent,
meaningful editable component and matches its physical definition, location, and
exclusions. Return unknown for background through an opening, another object,
watermarks, text fragments, reflections, shadows, isolated color/material patterns,
mixed components, imprecise boundaries, whole-object duplicates, and unmatched
regions. A complete component remains valid when it contains texture or text.

Regions:
{regions}

Allowed labels:
{inventory}

Return exactly one row for every region ID as JSON only:
{{"regions":[
{{"region_id":"R1","semantic_name":"allowed_key_or_unknown",
"region_kind":"physical_component|semantic_surface_feature|incidental_surface_pattern|background_through_opening|other_object_or_background|mixed_region|uncertain",
"confidence":0.0,"matches_target_region":true}}
]}}
"""


def parse_region_batch_label_plan(
    response: str,
    *,
    region_ids: Sequence[str],
    allowed_parts: Mapping[str, PartPrompt],
    minimum_confidence: float = 0.55,
) -> tuple[dict[str, RegionLabelPlan], dict[str, object]]:
    """Parse a bounded region-to-label map and reject non-physical assignments."""

    expected = {str(region_id).upper(): str(region_id) for region_id in region_ids}
    payload = _extract_json_object(response)
    diagnostics: dict[str, object] = {
        "response_character_count": len(response),
        "requested_region_count": len(expected),
        "ground_truth_used": False,
    }
    if payload is None or not isinstance(payload.get("regions"), list):
        missing = {
            original: RegionLabelPlan(
                None,
                0.0,
                {"status": "invalid_batch_json", "ground_truth_used": False},
                None,
            )
            for original in expected.values()
        }
        return missing, {**diagnostics, "status": "invalid_json_or_regions"}

    accepted_kinds = {"physical_component", "semantic_surface_feature"}
    parsed_by_id: dict[str, RegionLabelPlan] = {}
    duplicate_count = 0
    unknown_id_count = 0
    malformed_count = 0
    nonphysical_count = 0
    for raw in payload["regions"]:
        if not isinstance(raw, dict):
            malformed_count += 1
            continue
        raw_id = raw.get("region_id")
        if not isinstance(raw_id, str):
            malformed_count += 1
            continue
        normalized_id = raw_id.strip().upper()
        original_id = expected.get(normalized_id)
        if original_id is None:
            unknown_id_count += 1
            continue
        raw_kind = raw.get("region_kind")
        entity_kind = (
            _normalized_semantic(raw_kind)
            if isinstance(raw_kind, str)
            else "uncertain"
        )
        parsed = parse_region_label_plan(
            json.dumps(raw),
            allowed_parts=allowed_parts,
            minimum_confidence=minimum_confidence,
            entity_kind=entity_kind,
        )
        if entity_kind not in accepted_kinds and parsed.semantic_name is not None:
            nonphysical_count += 1
            parsed = RegionLabelPlan(
                None,
                parsed.confidence,
                {
                    **parsed.diagnostics,
                    "status": "nonphysical_region_kind",
                    "region_kind": entity_kind,
                },
                entity_kind,
            )
        previous = parsed_by_id.get(original_id)
        if previous is not None:
            duplicate_count += 1
            if previous.confidence >= parsed.confidence:
                continue
        parsed_by_id[original_id] = parsed

    for original_id in expected.values():
        parsed_by_id.setdefault(
            original_id,
            RegionLabelPlan(
                None,
                0.0,
                {"status": "missing_region_row", "ground_truth_used": False},
                None,
            ),
        )
    accepted_count = sum(
        item.semantic_name is not None for item in parsed_by_id.values()
    )
    return parsed_by_id, {
        **diagnostics,
        "status": "parsed",
        "returned_region_count": len(payload["regions"]),
        "accepted_region_count": accepted_count,
        "duplicate_region_count": duplicate_count,
        "unknown_region_id_count": unknown_id_count,
        "malformed_region_count": malformed_count,
        "nonphysical_region_count": nonphysical_count,
    }


def parse_region_label_plan(
    response: str,
    *,
    allowed_parts: Mapping[str, PartPrompt],
    minimum_confidence: float = 0.55,
    entity_kind: str | None = None,
) -> RegionLabelPlan:
    payload = _extract_json_object(response)
    diagnostics: dict[str, object] = {
        "response_character_count": len(response),
        "ground_truth_used": False,
    }
    if payload is None:
        return RegionLabelPlan(
            None,
            0.0,
            {**diagnostics, "status": "invalid_json"},
            entity_kind,
        )
    raw_name = payload.get("semantic_name")
    if not isinstance(raw_name, str):
        return RegionLabelPlan(
            None,
            0.0,
            {**diagnostics, "status": "missing_label"},
            entity_kind,
        )
    normalized = _normalized_semantic(raw_name)
    if normalized in {"unknown", "none", "uncertain", "unresolved"}:
        return RegionLabelPlan(
            None,
            0.0,
            {**diagnostics, "status": "unknown"},
            entity_kind,
        )
    canonical = {_normalized_semantic(name): name for name in allowed_parts}
    semantic_name = canonical.get(normalized)
    if semantic_name is None:
        return RegionLabelPlan(
            None,
            0.0,
            {**diagnostics, "status": "label_outside_inventory"},
            entity_kind,
        )
    if payload.get("matches_target_region") is False:
        return RegionLabelPlan(
            None,
            0.0,
            {**diagnostics, "status": "target_mismatch"},
            entity_kind,
        )
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    if not math.isfinite(confidence) or confidence < minimum_confidence:
        return RegionLabelPlan(
            None,
            0.0,
            {**diagnostics, "status": "low_confidence"},
            entity_kind,
        )
    return RegionLabelPlan(
        semantic_name,
        float(np.clip(confidence, 0.0, 1.0)),
        {**diagnostics, "status": "accepted"},
        entity_kind,
    )


def region_exterior_contact_fraction(
    region_mask: np.ndarray,
    root_mask: np.ndarray,
    *,
    dilation_px: int = 2,
) -> float:
    region = np.asarray(region_mask, dtype=bool)
    root = np.asarray(root_mask, dtype=bool)
    if region.shape != root.shape:
        raise ValueError("region and root masks must have the same shape")
    if dilation_px < 0:
        raise ValueError("dilation_px must be non-negative")
    region_area = int(np.count_nonzero(region))
    if region_area == 0:
        return 1.0
    outside = (~root).astype(np.uint8)
    if dilation_px:
        outside = cv2.dilate(
            outside,
            np.ones((3, 3), dtype=np.uint8),
            iterations=dilation_px,
        )
    return float(np.count_nonzero(region & outside.astype(bool)) / region_area)


def make_region_query_image(
    image: Image.Image,
    *,
    root_mask: np.ndarray,
    region_mask: np.ndarray,
    padding_ratio: float = 0.06,
) -> Image.Image:
    image = image.convert("RGB")
    expected_shape = (image.height, image.width)
    root = np.asarray(root_mask, dtype=bool)
    region = np.asarray(region_mask, dtype=bool) & root
    if root.shape != expected_shape or region.shape != expected_shape:
        raise ValueError("region-query masks must match the image dimensions")
    if not np.any(root) or not np.any(region):
        raise ValueError("region-query masks must be non-empty")
    context_box = _mask_box(
        root,
        image_size=image.size,
        padding_ratio=padding_ratio,
    )
    detail_box = _mask_box(
        region,
        image_size=image.size,
        padding_ratio=1.20,
    )
    context_width = context_box[2] - context_box[0]
    context_height = context_box[3] - context_box[1]
    panel_size = int(np.clip(max(context_width, context_height), 192, 640))
    image_array = np.asarray(image, dtype=np.uint8)
    context_panel = _render_region_panel(
        image_array,
        root,
        region,
        crop_box=context_box,
        panel_size=panel_size,
    )
    detail_panel = _render_region_panel(
        image_array,
        root,
        region,
        crop_box=detail_box,
        panel_size=panel_size,
    )
    divider = np.full((panel_size, 4, 3), 245, dtype=np.uint8)
    return Image.fromarray(
        np.concatenate((context_panel, divider, detail_panel), axis=1),
        mode="RGB",
    )


def make_region_batch_query_image(
    image: Image.Image,
    *,
    root_mask: np.ndarray,
    regions: Sequence[tuple[str, np.ndarray]],
    padding_ratio: float = 0.06,
    columns: int = 2,
    tile_size: tuple[int, int] = (320, 176),
) -> Image.Image:
    """Render numbered context/detail tiles for one bounded VLM request."""

    if not regions:
        raise ValueError("at least one region is required")
    if columns <= 0:
        raise ValueError("columns must be positive")
    tile_width, tile_height = tile_size
    if tile_width < 160 or tile_height < 96:
        raise ValueError("region batch tiles are too small")
    rows = math.ceil(len(regions) / columns)
    gap = 6
    canvas = np.full(
        (
            rows * tile_height + (rows - 1) * gap,
            columns * tile_width + (columns - 1) * gap,
            3,
        ),
        238,
        dtype=np.uint8,
    )
    for index, (region_id, region_mask) in enumerate(regions):
        query = np.asarray(
            make_region_query_image(
                image,
                root_mask=root_mask,
                region_mask=region_mask,
                padding_ratio=padding_ratio,
            ),
            dtype=np.uint8,
        )
        tile = cv2.resize(
            query,
            (tile_width, tile_height),
            interpolation=cv2.INTER_AREA,
        )
        cv2.rectangle(tile, (0, 0), (68, 32), (12, 12, 12), thickness=-1)
        cv2.putText(
            tile,
            str(region_id),
            (7, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        cv2.rectangle(
            tile,
            (0, 0),
            (tile_width - 1, tile_height - 1),
            (70, 70, 70),
            thickness=1,
        )
        row, column = divmod(index, columns)
        y0 = row * (tile_height + gap)
        x0 = column * (tile_width + gap)
        canvas[y0 : y0 + tile_height, x0 : x0 + tile_width] = tile
    return Image.fromarray(canvas, mode="RGB")


def make_root_query_image(
    image: Image.Image,
    *,
    root_mask: np.ndarray,
    padding_ratio: float = 0.18,
) -> Image.Image:
    """Show a highlighted scene context beside an isolated root close-up."""

    image = image.convert("RGB")
    expected_shape = (image.height, image.width)
    root = np.asarray(root_mask, dtype=bool)
    if root.shape != expected_shape:
        raise ValueError("root-query mask must match the image dimensions")
    if not np.any(root):
        raise ValueError("root-query mask must be non-empty")
    panel_size = int(np.clip(max(image.width, image.height) // 2, 256, 640))
    image_array = np.asarray(image, dtype=np.uint8)
    context_panel = _render_region_panel(
        image_array,
        np.ones_like(root, dtype=bool),
        root,
        crop_box=(0, 0, image.width, image.height),
        panel_size=panel_size,
    )
    detail_panel = _render_region_panel(
        image_array,
        root,
        root,
        crop_box=_mask_box(
            root,
            image_size=image.size,
            padding_ratio=padding_ratio,
        ),
        panel_size=panel_size,
    )
    divider = np.full((panel_size, 4, 3), 245, dtype=np.uint8)
    return Image.fromarray(
        np.concatenate((context_panel, divider, detail_panel), axis=1),
        mode="RGB",
    )


def _render_region_panel(
    image: np.ndarray,
    root_mask: np.ndarray,
    region_mask: np.ndarray,
    *,
    crop_box: tuple[int, int, int, int],
    panel_size: int,
) -> np.ndarray:
    x0, y0, x1, y1 = crop_box
    crop = image[y0:y1, x0:x1]
    root = root_mask[y0:y1, x0:x1].astype(np.uint8)
    region = region_mask[y0:y1, x0:x1].astype(np.uint8)
    if crop.size == 0:
        return np.full((panel_size, panel_size, 3), 127, dtype=np.uint8)
    height, width = crop.shape[:2]
    scale = min(panel_size / max(1, width), panel_size / max(1, height))
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized_image = cv2.resize(
        crop,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC,
    )
    resized_root = cv2.resize(
        root,
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    resized_region = cv2.resize(
        region,
        (resized_width, resized_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    rendered = resized_image.copy()
    dimmed = np.clip(
        rendered.astype(np.float32) * 0.28 + 127.0 * 0.72,
        0,
        255,
    ).astype(np.uint8)
    rendered[resized_root & ~resized_region] = dimmed[
        resized_root & ~resized_region
    ]
    rendered[~resized_root] = 127
    kernel = np.ones((3, 3), dtype=np.uint8)
    boundary = cv2.dilate(
        resized_region.astype(np.uint8), kernel, iterations=1
    ).astype(
        bool
    ) & ~resized_region
    rendered[boundary] = np.asarray([255, 40, 40], dtype=np.uint8)
    panel = np.full((panel_size, panel_size, 3), 127, dtype=np.uint8)
    offset_x = (panel_size - resized_width) // 2
    offset_y = (panel_size - resized_height) // 2
    panel[
        offset_y : offset_y + resized_height,
        offset_x : offset_x + resized_width,
    ] = rendered
    return panel


def _mask_box(
    mask: np.ndarray,
    *,
    image_size: tuple[int, int],
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0, 0, 0, 0
    width, height = image_size
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()) + 1, int(ys.max()) + 1
    padding = max(2, round(max(x1 - x0, y1 - y0) * padding_ratio))
    return (
        max(0, x0 - padding),
        max(0, y0 - padding),
        min(width, x1 + padding),
        min(height, y1 + padding),
    )


def _semantic_depth(
    semantic_name: str,
    parts: Mapping[str, PartPrompt],
    domain_name: str,
    active: frozenset[str] = frozenset(),
) -> int:
    if semantic_name in active:
        return 0
    part = parts[semantic_name]
    parent = part.semantic_parent
    if parent is None or parent == domain_name or parent not in parts:
        return 0
    return 1 + _semantic_depth(
        parent,
        parts,
        domain_name,
        active | {semantic_name},
    )


class VlmPartGenerator:
    """Use a VLM as soft semantics over independently proposed SAM2 regions."""

    def __init__(
        self,
        planner: VlmPlanner,
        segment_boxes: BoxSegmenter,
        *,
        config: VlmPartConfig | None = None,
    ) -> None:
        self.planner = planner
        self.segment_boxes = segment_boxes
        self.config = config or VlmPartConfig()

    def generate(
        self,
        image: Image.Image,
        roots: Sequence[MaskCandidate],
        domains: Mapping[str, DomainPrompt],
        existing_candidates: Sequence[MaskCandidate] = (),
    ) -> CandidateGeneration:
        image = image.convert("RGB")
        candidates: list[MaskCandidate] = []
        root_rows: list[dict[str, object]] = []
        total_query_count = 0
        dynamic_inventory_query_count = 0
        rejected = {
            "planner_error": 0,
            "segmenter_mismatch": 0,
            "outside_root": 0,
            "too_small": 0,
            "area_outside_prior": 0,
            "duplicate_mask": 0,
            "planner_unknown_or_rejected": 0,
            "ownership_geometry_rejected": 0,
        }
        for root in roots[: self.config.maximum_roots]:
            remaining_total_budget = max(
                0,
                self.config.maximum_total_planner_queries - total_query_count,
            )
            if remaining_total_budget == 0:
                break
            domain = domains.get(root.semantic_name)
            if domain is None:
                continue
            object_label = str(
                root.metadata.get("resolved_object_label")
                or root.metadata.get("root_model_label")
                or root.prompt
                or root.semantic_name.replace("_", " ")
            )
            profile_hint_value = root.metadata.get("selected_part_profile")
            profile_hint = (
                str(profile_hint_value) if profile_hint_value is not None else None
            )
            selected_parts, selected_profile, profile_diagnostics = (
                domain.select_parts(
                    object_label,
                    profile_hint=profile_hint,
                    profile_hint_source=(
                        str(root.metadata.get("profile_hint_source", "vlm_root"))
                        if profile_hint is not None
                        else None
                    ),
                )
            )
            selected_parts = list(
                selected_parts[: self.config.maximum_inventory_size]
            )
            initial_object_label = object_label
            initial_selected_profile = selected_profile
            dynamic_inventory_diagnostics: dict[str, object] = {
                "status": "disabled",
                "ground_truth_used": False,
            }
            if (
                self.config.use_dynamic_inventory
                and dynamic_inventory_query_count
                < self.config.maximum_dynamic_inventory_roots
                and total_query_count
                < self.config.maximum_total_planner_queries
            ):
                dynamic_inventory_query_count += 1
                total_query_count += 1
                crop_x0, crop_y0, crop_x1, crop_y1 = _mask_box(
                    root.mask,
                    image_size=image.size,
                    padding_ratio=self.config.crop_padding_ratio,
                )
                if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
                    dynamic_inventory_diagnostics = {
                        "status": "empty_root_crop",
                        "ground_truth_used": False,
                    }
                else:
                    inventory_crop = image.crop(
                        (crop_x0, crop_y0, crop_x1, crop_y1)
                    )
                    local_root = root.mask[
                        crop_y0:crop_y1, crop_x0:crop_x1
                    ].astype(bool)
                    crop_array = np.asarray(inventory_crop, dtype=np.uint8).copy()
                    crop_array[~local_root] = 127
                    inventory_crop = Image.fromarray(crop_array, mode="RGB")
                    identity_prompt = build_dynamic_object_identity_prompt(
                        candidate_object_label=object_label,
                        domain_name=domain.name,
                    )
                    try:
                        identity_response = self.planner.generate_response(
                            inventory_crop,
                            identity_prompt,
                        )
                    except (
                        RuntimeError,
                        ValueError,
                        OSError,
                        TypeError,
                        KeyError,
                    ) as error:
                        rejected["planner_error"] += 1
                        dynamic_inventory_diagnostics = {
                            "status": "object_identity_planner_error",
                            "error_type": type(error).__name__,
                            "ground_truth_used": False,
                        }
                    else:
                        object_identity = parse_dynamic_object_identity(
                            identity_response,
                            config=self.config,
                        )
                        if object_identity.label is not None:
                            object_label = object_identity.label
                            (
                                reselected_parts,
                                reselected_profile,
                                reselected_diagnostics,
                            ) = domain.select_parts(object_label)
                            selected_parts = list(
                                reselected_parts[: self.config.maximum_inventory_size]
                            )
                            selected_profile = reselected_profile
                            profile_diagnostics = {
                                **reselected_diagnostics,
                                "selection_reason": "vlm_open_set_object_identity",
                                "vlm_object_identity": object_identity.diagnostics,
                            }
                        if (
                            total_query_count
                            >= self.config.maximum_total_planner_queries
                        ):
                            dynamic_inventory_diagnostics = {
                                "status": "object_identity_only_budget_exhausted",
                                "object_identity": object_identity.diagnostics,
                                "resolved_object_label": object_label,
                                "ground_truth_used": False,
                            }
                        else:
                            total_query_count += 1
                            inventory_prompt = build_dynamic_part_inventory_prompt(
                                object_label=object_label,
                                domain_name=domain.name,
                                existing_parts=selected_parts,
                                maximum_parts=self.config.maximum_dynamic_parts,
                            )
                            try:
                                inventory_response = self.planner.generate_response(
                                    inventory_crop,
                                    inventory_prompt,
                                )
                            except (
                                RuntimeError,
                                ValueError,
                                OSError,
                                TypeError,
                                KeyError,
                            ) as error:
                                rejected["planner_error"] += 1
                                dynamic_inventory_diagnostics = {
                                    "status": "part_inventory_planner_error",
                                    "error_type": type(error).__name__,
                                    "object_identity": object_identity.diagnostics,
                                    "resolved_object_label": object_label,
                                    "ground_truth_used": False,
                                }
                            else:
                                if object_identity.label is None:
                                    fallback_identity = parse_dynamic_object_identity(
                                        inventory_response,
                                        config=self.config,
                                    )
                                    if fallback_identity.label is not None:
                                        object_identity = fallback_identity
                                        object_label = fallback_identity.label
                                        (
                                            reselected_parts,
                                            reselected_profile,
                                            reselected_diagnostics,
                                        ) = domain.select_parts(object_label)
                                        selected_parts = list(
                                            reselected_parts[
                                                : self.config.maximum_inventory_size
                                            ]
                                        )
                                        selected_profile = reselected_profile
                                        profile_diagnostics = {
                                            **reselected_diagnostics,
                                            "selection_reason": (
                                                "vlm_open_set_object_identity"
                                            ),
                                            "vlm_object_identity": (
                                                object_identity.diagnostics
                                            ),
                                        }
                                dynamic_inventory = parse_dynamic_part_inventory(
                                    inventory_response,
                                    domain_name=domain.name,
                                    existing_parts=selected_parts,
                                    config=self.config,
                                )
                                selected_parts.extend(dynamic_inventory.parts)
                                dynamic_inventory_diagnostics = {
                                    **dynamic_inventory.diagnostics,
                                    "object_identity": object_identity.diagnostics,
                                    "resolved_object_label": object_label,
                                }
            allowed_parts = {part.semantic_name: part for part in selected_parts}
            if not allowed_parts:
                continue
            root_key = self._root_key(root)
            scoped_existing = [
                candidate
                for candidate in existing_candidates
                if self._root_key(candidate) == root_key
                and candidate is not root
                and not (
                    candidate.semantic_name == candidate.semantic_parent
                    and candidate.metadata.get("parent_candidate_key") is None
                )
            ]
            (
                region_label_candidates,
                region_label_rows,
                region_label_query_count,
            ) = self._label_visual_regions(
                image=image,
                root=root,
                domain=domain,
                object_label=object_label,
                selected_parts=selected_parts,
                selected_profile=selected_profile,
                scoped_existing=scoped_existing,
                query_budget=min(
                    self.config.maximum_region_label_queries,
                    self.config.maximum_planner_queries,
                    max(
                        0,
                        self.config.maximum_total_planner_queries
                        - total_query_count,
                    ),
                ),
                rejected=rejected,
            )
            total_query_count += region_label_query_count
            candidates.extend(region_label_candidates)
            established_semantics = {
                candidate.semantic_name
                for candidate in [*scoped_existing, *region_label_candidates]
                if not self._is_generic_region(candidate, domain.name)
            }
            query_parts = (
                [
                    part
                    for part in selected_parts
                    if not self._is_root_fallback(part, domain.name)
                    and (
                        self.config.query_established_semantics
                        or part.semantic_name not in established_semantics
                    )
                ]
                if self.config.use_box_planner_queries
                else []
            )
            query_parts.sort(
                key=lambda part: (
                    part.semantic_name in established_semantics,
                    not part.detail,
                    -part.priority,
                    part.semantic_name,
                )
            )
            remaining_query_budget = max(
                0,
                self.config.maximum_planner_queries - region_label_query_count,
            )
            remaining_query_budget = min(
                remaining_query_budget,
                max(
                    0,
                    self.config.maximum_total_planner_queries
                    - total_query_count,
                ),
            )
            query_parts = query_parts[:remaining_query_budget]
            x0 = y0 = x1 = y1 = 0
            crop: Image.Image | None = None
            if query_parts:
                x0, y0, x1, y1 = _mask_box(
                    root.mask,
                    image_size=image.size,
                    padding_ratio=self.config.crop_padding_ratio,
                )
                if x1 <= x0 or y1 <= y0:
                    query_parts = []
                else:
                    crop = image.crop((x0, y0, x1, y1))
                    local_root = root.mask[y0:y1, x0:x1].astype(bool)
                    crop_array = np.asarray(crop, dtype=np.uint8).copy()
                    crop_array[~local_root] = 127
                    crop = Image.fromarray(crop_array, mode="RGB")
            query_groups = (
                [(part,) for part in query_parts]
                if self.config.use_per_semantic_queries
                else ([tuple(query_parts)] if query_parts else [])
            )
            planned_parts: list[PlannedPart] = []
            plan_rows: list[dict[str, object]] = []
            for query_group in query_groups:
                assert crop is not None
                total_query_count += 1
                prompt = build_part_planner_prompt(
                    object_label=object_label,
                    domain_name=domain.name,
                    parts=query_group,
                    context_parts=selected_parts,
                )
                try:
                    response = self.planner.generate_response(crop, prompt)
                except (
                    RuntimeError,
                    ValueError,
                    OSError,
                    TypeError,
                    KeyError,
                ) as error:
                    rejected["planner_error"] += 1
                    plan_rows.append(
                        {
                            "requested_semantics": [
                                part.semantic_name for part in query_group
                            ],
                            "status": "planner_error",
                            "error_type": type(error).__name__,
                            "ground_truth_used": False,
                        }
                    )
                    continue
                query_allowed = {
                    part.semantic_name: part for part in query_group
                }
                parsed = parse_part_plan(
                    response,
                    image_size=crop.size,
                    allowed_parts=query_allowed,
                    config=self.config,
                )
                planned_parts.extend(parsed.parts)
                rejected["planner_unknown_or_rejected"] += sum(
                    int(value)
                    for value in parsed.diagnostics.get(
                        "rejection_counts", {}
                    ).values()
                )
                plan_rows.append(
                    {
                        "requested_semantics": list(query_allowed),
                        "parse": parsed.diagnostics,
                        "ground_truth_used": False,
                    }
                )
            ordered_parts = sorted(
                planned_parts,
                key=lambda item: (
                    _semantic_depth(
                        item.semantic_name,
                        allowed_parts,
                        domain.name,
                    ),
                    -item.confidence,
                ),
            )
            global_plans = [
                PlannedPart(
                    item.semantic_name,
                    (
                        item.box_xyxy[0] + x0,
                        item.box_xyxy[1] + y0,
                        item.box_xyxy[2] + x0,
                        item.box_xyxy[3] + y0,
                    ),
                    item.confidence,
                    item.instance_hint,
                )
                for item in ordered_parts
            ]
            detections = [
                Detection(
                    item.semantic_name,
                    item.confidence,
                    item.box_xyxy,
                )
                for item in global_plans
            ]
            direct_indices = [
                index
                for index, item in enumerate(global_plans)
                if self.config.allow_direct_sam_regions
                and item.semantic_name not in established_semantics
            ]
            direct_detections = [detections[index] for index in direct_indices]
            segmentations = self.segment_boxes(image, direct_detections)
            if len(segmentations) != len(direct_detections):
                rejected["segmenter_mismatch"] += 1
            region_label_keys = {
                str(candidate.metadata["vlm_region_candidate_key"])
                for candidate in region_label_candidates
            }
            regions = [
                region
                for region in self._existing_regions(scoped_existing, domain.name)
                if region.candidate_key not in region_label_keys
            ]
            for plan_index, segmentation in zip(
                direct_indices,
                segmentations,
                strict=False,
            ):
                plan = global_plans[plan_index]
                regions.append(
                    PartRegion(
                        mask=segmentation.mask.astype(bool),
                        source="sam2/vlm-box-refinement",
                        quality=float(segmentation.quality),
                        candidate_key=f"vlm-direct:{plan_index}",
                        semantic_name=plan.semantic_name,
                        generic=False,
                        direct_plan_index=plan_index,
                        region_kind=(
                            "detail"
                            if allowed_parts[plan.semantic_name].detail
                            else "panel"
                        ),
                    )
                )
            assignments, assignment_diagnostics = assign_plans_to_regions(
                global_plans,
                regions,
                allowed_parts=allowed_parts,
                root_mask=root.mask.astype(bool),
                config=self.config,
            )
            accepted_by_semantic: dict[
                str, list[tuple[np.ndarray, str]]
            ] = self._parent_supports(
                [*scoped_existing, *region_label_candidates], domain.name
            )
            root_area = max(1, int(np.count_nonzero(root.mask)))
            emitted_for_root = len(region_label_candidates)
            semantic_counts: dict[str, int] = {}
            ordered_assignments = sorted(
                assignments,
                key=lambda assignment: (
                    _semantic_depth(
                        global_plans[assignment.plan_index].semantic_name,
                        allowed_parts,
                        domain.name,
                    ),
                    assignment.plan_index,
                ),
            )
            for assignment in ordered_assignments:
                item = global_plans[assignment.plan_index]
                region = regions[assignment.region_index]
                part = allowed_parts[item.semantic_name]
                raw_mask = region.mask.astype(bool)
                raw_area = int(np.count_nonzero(raw_mask))
                clipped = raw_mask & root.mask.astype(bool)
                clipped_area = int(np.count_nonzero(clipped))
                if (
                    raw_area < self.config.minimum_area_px
                    or clipped_area / max(1, raw_area)
                    < self.config.minimum_raw_root_containment
                ):
                    rejected["outside_root"] += 1
                    continue

                semantic_parent = part.semantic_parent or domain.name
                parent_candidate_key: str | None = None
                parent_options = accepted_by_semantic.get(semantic_parent, [])
                if semantic_parent != domain.name and parent_options:
                    overlaps = [
                        int(np.count_nonzero(clipped & parent_mask))
                        / max(1, clipped_area)
                        for parent_mask, _ in parent_options
                    ]
                    best_parent = int(np.argmax(overlaps))
                    if overlaps[best_parent] < self.config.minimum_parent_overlap:
                        rejected["outside_root"] += 1
                        continue
                    parent_mask, parent_candidate_key = parent_options[best_parent]
                    dilation = max(
                        1,
                        round(
                            min(image.size) * self.config.parent_dilation_ratio
                        ),
                    )
                    kernel = cv2.getStructuringElement(
                        cv2.MORPH_ELLIPSE,
                        (2 * dilation + 1, 2 * dilation + 1),
                    )
                    clipped &= cv2.dilate(
                        parent_mask.astype(np.uint8), kernel
                    ).astype(bool)
                    clipped_area = int(np.count_nonzero(clipped))
                if clipped_area < self.config.minimum_area_px:
                    rejected["too_small"] += 1
                    continue
                area_fraction = clipped_area / root_area
                if (
                    area_fraction
                    < part.minimum_parent_fraction
                    * self.config.minimum_area_tolerance
                    or area_fraction
                    > min(
                        1.0,
                        part.maximum_parent_fraction
                        * self.config.maximum_area_tolerance,
                    )
                ):
                    rejected["area_outside_prior"] += 1
                    continue
                if any(
                    existing.semantic_name == item.semantic_name
                    and mask_iou(existing.mask, clipped)
                    >= self.config.same_semantic_mask_nms_iou
                    for existing in candidates
                    if self._root_key(existing) == self._root_key(root)
                ):
                    rejected["duplicate_mask"] += 1
                    continue
                instance_number = semantic_counts.get(item.semantic_name, 0) + 1
                semantic_counts[item.semantic_name] = instance_number
                candidate_key = (
                    f"vlm:{root.metadata.get('root_index', 'unknown')}:"
                    f"{item.semantic_name}:{instance_number}"
                )
                score = item.confidence * assignment.score
                candidate = MaskCandidate(
                    semantic_name=item.semantic_name,
                    semantic_parent=semantic_parent,
                    mask=clipped,
                    score=score,
                    source=f"{self.planner.backend_id}+{region.source}/matched-part",
                    prompt=item.semantic_name.replace("_", " "),
                    source_reliability=self.config.source_reliability,
                    metadata={
                        "root_origin": root.metadata.get("root_origin", "legacy"),
                        "root_index": root.metadata.get("root_index"),
                        "candidate_key": candidate_key,
                        "parent_candidate_key": parent_candidate_key
                        or root.metadata.get("candidate_key"),
                        "assembly_parent_semantic": (
                            part.assembly_parent or semantic_parent
                        ),
                        "maximum_instances": part.maximum_instances,
                        "detail": part.detail,
                        "sam_quality": float(region.quality),
                        "planner_confidence": item.confidence,
                        "planner_box_xyxy": list(item.box_xyxy),
                        "planner_instance_hint": item.instance_hint,
                        "selected_part_profile": selected_profile,
                        "vlm_planner_backend": self.planner.backend_id,
                        "vlm_proposal_only": True,
                        "vlm_region_candidate_key": region.candidate_key,
                        "vlm_region_previous_semantic": region.semantic_name,
                        "vlm_assignment_score": assignment.score,
                        "vlm_assignment_box_containment": (
                            assignment.box_containment
                        ),
                        "vlm_assignment_box_fill": assignment.box_fill,
                        "vlm_assignment_box_iou": assignment.box_iou,
                        "vlm_assignment_area_prior": assignment.area_prior,
                        "vlm_assignment_semantic_support": (
                            assignment.semantic_support
                        ),
                        "ground_truth_used": False,
                    },
                )
                candidates.append(candidate)
                accepted_by_semantic.setdefault(item.semantic_name, []).append(
                    (clipped, candidate_key)
                )
                emitted_for_root += 1
            root_rows.append(
                {
                    "root_key": root_key,
                    "initial_object_label": initial_object_label,
                    "initial_selected_profile": initial_selected_profile,
                    "object_label": object_label,
                    "domain": domain.name,
                    "selected_profile": selected_profile,
                    "profile_selection": profile_diagnostics,
                    "dynamic_inventory": dynamic_inventory_diagnostics,
                    "region_label_queries": region_label_rows,
                    "region_label_query_count": region_label_query_count,
                    "total_planner_query_count_after_root": total_query_count,
                    "plan_queries": plan_rows,
                    "planned_part_count": len(global_plans),
                    "region_count": len(regions),
                    "assignment": assignment_diagnostics,
                    "accepted_mask_count": emitted_for_root,
                    "ground_truth_used": False,
                }
            )
        return CandidateGeneration(
            tuple(candidates),
            {
                "algorithm": "hpid-ontology-bounded-vlm-region-matching-v5",
                "planner_backend": self.planner.backend_id,
                "root_count": len(root_rows),
                "input_root_count": len(roots),
                "skipped_root_count": max(0, len(roots) - len(root_rows)),
                "candidate_count": len(candidates),
                "total_planner_query_count": total_query_count,
                "dynamic_inventory_query_count": (
                    dynamic_inventory_query_count
                ),
                "maximum_total_planner_queries": (
                    self.config.maximum_total_planner_queries
                ),
                "total_query_budget_exhausted": (
                    total_query_count
                    >= self.config.maximum_total_planner_queries
                ),
                "rejection_counts": rejected,
                "roots": root_rows,
                "final_part_ids_assigned_by_backend": False,
                "ground_truth_used": False,
            },
        )

    @staticmethod
    def _root_key(candidate: MaskCandidate) -> str:
        return _candidate_root_identity_key(candidate)

    @staticmethod
    def _is_root_fallback(part: PartPrompt, domain_name: str) -> bool:
        return part.semantic_name in {domain_name, f"{domain_name}_body"}

    @staticmethod
    def _is_generic_region(candidate: MaskCandidate, domain_name: str) -> bool:
        if not bool(candidate.metadata.get("visual_region")):
            return False
        if "generic_visual_region" in candidate.metadata:
            return bool(candidate.metadata["generic_visual_region"])
        return "_visual_" in candidate.semantic_name or (
            candidate.semantic_name == domain_name
        )

    def _is_weak_semantic_region(self, candidate: MaskCandidate) -> bool:
        metadata = candidate.metadata
        if not (
            bool(metadata.get("visual_region"))
            and bool(metadata.get("semantic_reranked"))
        ):
            return False
        if bool(metadata.get("semantic_axis_structure_rescue")):
            return False
        probability = float(metadata.get("semantic_rerank_probability", 1.0))
        margin = float(metadata.get("semantic_rerank_margin", 1.0))
        return (
            probability <= self.config.weak_semantic_maximum_probability
            and margin <= self.config.weak_semantic_maximum_margin
        )

    def _label_visual_regions(
        self,
        *,
        image: Image.Image,
        root: MaskCandidate,
        domain: DomainPrompt,
        object_label: str,
        selected_parts: Sequence[PartPrompt],
        selected_profile: str | None,
        scoped_existing: Sequence[MaskCandidate],
        query_budget: int,
        rejected: dict[str, int],
    ) -> tuple[list[MaskCandidate], list[dict[str, object]], int]:
        if not self.config.use_region_label_queries or query_budget <= 0:
            return [], [], 0
        root_area = max(1, int(np.count_nonzero(root.mask)))
        strong_semantics = {
            candidate.semantic_name
            for candidate in scoped_existing
            if not self._is_generic_region(candidate, domain.name)
            and not self._is_weak_semantic_region(candidate)
        }
        eligible = [
            candidate
            for candidate in scoped_existing
            if bool(candidate.metadata.get("visual_region"))
            and (
                self._is_generic_region(candidate, domain.name)
                or self._is_weak_semantic_region(candidate)
            )
            and not (
                self._is_weak_semantic_region(candidate)
                and candidate.semantic_name in strong_semantics
            )
        ]
        eligible.sort(
            key=lambda candidate: (
                str(candidate.metadata.get("visual_region_kind")) != "panel",
                candidate.semantic_name in strong_semantics,
                not self._is_weak_semantic_region(candidate),
                not bool(candidate.metadata.get("multi_view_confirmed")),
                -float(candidate.metadata.get("sam_quality", candidate.score)),
                -int(np.count_nonzero(candidate.mask)),
            )
        )
        batch_mode = bool(self.config.use_batched_region_label_queries)
        batch_results: dict[str, RegionLabelPlan] = {}
        batch_diagnostics: dict[str, dict[str, object]] = {}
        batch_selected_keys: set[str] = set()
        query_count = 0

        def region_key(candidate: MaskCandidate) -> str:
            return str(
                candidate.metadata.get(
                    "candidate_key",
                    f"{candidate.source}::{candidate.semantic_name}",
                )
            )

        if batch_mode:
            batch_eligible: list[
                tuple[MaskCandidate, str, tuple[PartPrompt, ...]]
            ] = []
            for region in eligible:
                region_kind = str(
                    region.metadata.get("visual_region_kind") or "unknown"
                )
                compatible_parts = tuple(
                    part
                    for part in selected_parts
                    if not self._is_root_fallback(part, domain.name)
                    and (region_kind == "detail") == bool(part.detail)
                )
                if not compatible_parts:
                    continue
                physicality_evidence = region.metadata.get(
                    "vlm_physicality_audit"
                )
                if (
                    isinstance(physicality_evidence, Mapping)
                    and physicality_evidence.get("decision")
                    == "nonphysical_supported"
                ):
                    continue
                exterior_contact_fraction = region_exterior_contact_fraction(
                    region.mask,
                    root.mask,
                    dilation_px=self.config.exterior_contact_dilation_px,
                )
                if (
                    exterior_contact_fraction
                    > self.config.maximum_exterior_contact_fraction
                ):
                    continue
                batch_eligible.append((region, region_kind, compatible_parts))

            maximum_regions = min(
                len(batch_eligible),
                self.config.maximum_region_label_queries,
                query_budget * max(1, self.config.region_label_batch_size),
            )
            batch_eligible = batch_eligible[:maximum_regions]
            batch_size = max(1, self.config.region_label_batch_size)
            for offset in range(0, len(batch_eligible), batch_size):
                if query_count >= query_budget:
                    break
                group = batch_eligible[offset : offset + batch_size]
                region_ids = [f"R{index + 1}" for index in range(len(group))]
                allowed_by_name: dict[str, PartPrompt] = {}
                for _, _, compatible_parts in group:
                    for part in compatible_parts:
                        allowed_by_name.setdefault(part.semantic_name, part)
                query_image = make_region_batch_query_image(
                    image,
                    root_mask=root.mask,
                    regions=[
                        (region_id, region.mask)
                        for region_id, (region, _, _) in zip(
                            region_ids, group, strict=True
                        )
                    ],
                    padding_ratio=self.config.crop_padding_ratio,
                )
                prompt = build_region_batch_label_prompt(
                    object_label=object_label,
                    domain_name=domain.name,
                    parts=tuple(allowed_by_name.values()),
                    region_specs=[
                        (region_id, region_kind)
                        for region_id, (_, region_kind, _) in zip(
                            region_ids, group, strict=True
                        )
                    ],
                )
                query_count += 1
                try:
                    response = self.planner.generate_response(query_image, prompt)
                except (
                    RuntimeError,
                    ValueError,
                    OSError,
                    TypeError,
                    KeyError,
                ) as error:
                    rejected["planner_error"] += 1
                    for region_id, (region, _, _) in zip(
                        region_ids, group, strict=True
                    ):
                        key = region_key(region)
                        batch_selected_keys.add(key)
                        batch_results[key] = RegionLabelPlan(
                            None,
                            0.0,
                            {
                                "status": "batch_label_planner_error",
                                "error_type": type(error).__name__,
                                "ground_truth_used": False,
                            },
                            None,
                        )
                        batch_diagnostics[key] = {
                            "region_id": region_id,
                            "batch_index": offset // batch_size,
                            "status": "planner_error",
                            "error_type": type(error).__name__,
                        }
                    continue
                parsed_batch, parsed_batch_diagnostics = (
                    parse_region_batch_label_plan(
                        response,
                        region_ids=region_ids,
                        allowed_parts=allowed_by_name,
                        minimum_confidence=(
                            self.config.region_label_minimum_confidence
                        ),
                    )
                )
                for region_id, (region, _, _) in zip(
                    region_ids, group, strict=True
                ):
                    key = region_key(region)
                    batch_selected_keys.add(key)
                    batch_results[key] = parsed_batch[region_id]
                    batch_diagnostics[key] = {
                        "region_id": region_id,
                        "batch_index": offset // batch_size,
                        "batch_parse": parsed_batch_diagnostics,
                    }
        proposals: list[
            tuple[float, MaskCandidate, PartPrompt, RegionLabelPlan]
        ] = []
        rows: list[dict[str, object]] = []
        for region in eligible:
            if not batch_mode and query_count >= query_budget:
                break
            region_kind = str(
                region.metadata.get("visual_region_kind") or "unknown"
            )
            compatible_parts = tuple(
                part
                for part in selected_parts
                if not self._is_root_fallback(part, domain.name)
                and (region_kind == "detail") == bool(part.detail)
            )
            if not compatible_parts:
                continue
            exterior_contact_fraction = region_exterior_contact_fraction(
                region.mask,
                root.mask,
                dilation_px=self.config.exterior_contact_dilation_px,
            )
            if (
                exterior_contact_fraction
                > self.config.maximum_exterior_contact_fraction
            ):
                rejected["ownership_geometry_rejected"] += 1
                rows.append(
                    {
                        "region_candidate_key": region.metadata.get(
                            "candidate_key"
                        ),
                        "previous_semantic": region.semantic_name,
                        "proposal_kind": region_kind,
                        "status": "exterior_fragment_rejected",
                        "exterior_contact_fraction": (
                            exterior_contact_fraction
                        ),
                        "ground_truth_used": False,
                    }
                )
                continue
            query_image = make_region_query_image(
                image,
                root_mask=root.mask,
                region_mask=region.mask,
                padding_ratio=self.config.crop_padding_ratio,
            )
            row: dict[str, object] = {
                "region_candidate_key": region.metadata.get("candidate_key"),
                "previous_semantic": region.semantic_name,
                "proposal_kind": region_kind,
                "exterior_contact_fraction": exterior_contact_fraction,
                "ground_truth_used": False,
            }
            rows.append(row)
            physicality_evidence = region.metadata.get("vlm_physicality_audit")
            if (
                isinstance(physicality_evidence, Mapping)
                and physicality_evidence.get("decision")
                == "nonphysical_supported"
            ):
                rejected["planner_unknown_or_rejected"] += 1
                row.update(
                    {
                        "ownership_kind": physicality_evidence.get("label"),
                        "ownership_confidence": physicality_evidence.get(
                            "confidence"
                        ),
                        "ownership_evidence_reused": True,
                        "ownership_parse": {
                            "status": "reused_nonphysicality_audit",
                            "ground_truth_used": False,
                        },
                        "status": "ownership_rejected",
                    }
                )
                continue
            reuse_physicality = bool(
                isinstance(physicality_evidence, Mapping)
                and physicality_evidence.get("decision") == "physical_supported"
                and float(physicality_evidence.get("confidence", 0.0))
                >= self.config.region_label_minimum_confidence
            )
            if reuse_physicality:
                region_entity_kind = str(
                    physicality_evidence.get("label") or "physical_component"
                )
                row.update(
                    {
                        "ownership_kind": region_entity_kind,
                        "ownership_confidence": float(
                            physicality_evidence.get("confidence", 0.0)
                        ),
                        "status": "reused_physicality_audit",
                        "ownership_evidence_reused": True,
                        "ownership_parse": {
                            "status": "reused_physicality_audit",
                            "planner_backend": physicality_evidence.get(
                                "planner_backend"
                            ),
                            "ground_truth_used": False,
                        },
                    }
                )
            else:
                region_entity_kind = (
                    "semantic_surface_feature"
                    if region_kind == "detail"
                    else "physical_component"
                )
                row.update(
                    {
                        "ownership_kind": region_entity_kind,
                        "ownership_confidence": None,
                        "ownership_evidence_reused": False,
                        "ownership_parse": {
                            "status": "joint_semantic_label_gate",
                            "ground_truth_used": False,
                        },
                    }
                )
            allowed = {part.semantic_name: part for part in compatible_parts}
            if batch_mode:
                key = region_key(region)
                if key not in batch_selected_keys:
                    continue
                parsed = batch_results[key]
                row["batch_region_label"] = batch_diagnostics.get(key)
                if (
                    parsed.semantic_name is not None
                    and parsed.semantic_name not in allowed
                ):
                    parsed = RegionLabelPlan(
                        None,
                        parsed.confidence,
                        {
                            **parsed.diagnostics,
                            "status": "label_outside_region_kind_inventory",
                        },
                        parsed.entity_kind,
                    )
            else:
                prompt = build_region_label_prompt(
                    object_label=object_label,
                    domain_name=domain.name,
                    parts=compatible_parts,
                    region_kind=region_entity_kind,
                )
                query_count += 1
                try:
                    response = self.planner.generate_response(query_image, prompt)
                except (
                    RuntimeError,
                    ValueError,
                    OSError,
                    TypeError,
                    KeyError,
                ) as error:
                    rejected["planner_error"] += 1
                    row["status"] = "label_planner_error"
                    row["error_type"] = type(error).__name__
                    continue
                parsed = parse_region_label_plan(
                    response,
                    allowed_parts=allowed,
                    minimum_confidence=(
                        self.config.region_label_minimum_confidence
                    ),
                    entity_kind=region_entity_kind,
                )
            row["predicted_semantic"] = parsed.semantic_name
            row["confidence"] = parsed.confidence
            row["label_parse"] = parsed.diagnostics
            if parsed.semantic_name is None:
                rejected["planner_unknown_or_rejected"] += 1
                continue
            if (
                parsed.semantic_name == region.semantic_name
                and not self._is_generic_region(region, domain.name)
            ):
                row["status"] = "confirmed_existing_semantic"
                continue
            part = allowed[parsed.semantic_name]
            area_fraction = int(np.count_nonzero(region.mask & root.mask)) / root_area
            if (
                area_fraction
                < part.minimum_parent_fraction * self.config.minimum_area_tolerance
                or area_fraction
                > min(
                    1.0,
                    part.maximum_parent_fraction
                    * self.config.maximum_area_tolerance,
                )
            ):
                rejected["area_outside_prior"] += 1
                row["status"] = "area_outside_prior"
                continue
            proposal_score = (
                parsed.confidence
                * float(region.metadata.get("sam_quality", region.score))
                * part.priority
            )
            proposals.append((proposal_score, region, part, parsed))
            row["status"] = "proposed"

        strong_existing_counts: dict[str, int] = {}
        for candidate in scoped_existing:
            if self._is_generic_region(candidate, domain.name) or (
                self._is_weak_semantic_region(candidate)
            ):
                continue
            strong_existing_counts[candidate.semantic_name] = (
                strong_existing_counts.get(candidate.semantic_name, 0) + 1
            )
        proposals.sort(key=lambda item: item[0], reverse=True)
        accepted: list[MaskCandidate] = []
        semantic_counts = dict(strong_existing_counts)
        corroborated_existing_keys: set[str] = set()
        for proposal_score, region, part, parsed in proposals:
            semantic_name = part.semantic_name
            clipped = region.mask.astype(bool) & root.mask.astype(bool)
            corroborated_existing: MaskCandidate | None = None
            if semantic_counts.get(semantic_name, 0) >= part.maximum_instances:
                same_semantic_existing = [
                    candidate
                    for candidate in scoped_existing
                    if candidate.semantic_name == semantic_name
                    and not self._is_generic_region(candidate, domain.name)
                ]
                scored_existing: list[
                    tuple[float, float, MaskCandidate]
                ] = []
                clipped_area = max(1, int(np.count_nonzero(clipped)))
                for existing in same_semantic_existing:
                    intersection = int(
                        np.count_nonzero(existing.mask.astype(bool) & clipped)
                    )
                    existing_area = max(
                        1, int(np.count_nonzero(existing.mask))
                    )
                    union = existing_area + clipped_area - intersection
                    overlap = intersection / max(1, union)
                    containment = intersection / min(existing_area, clipped_area)
                    if (
                        overlap >= self.config.region_label_corroboration_iou
                        or containment
                        >= self.config.region_label_corroboration_containment
                    ):
                        scored_existing.append(
                            (containment, overlap, existing)
                        )
                if scored_existing:
                    corroborated_existing = max(
                        scored_existing,
                        key=lambda item: (item[0], item[1]),
                    )[2]
                    existing_key = str(
                        corroborated_existing.metadata.get(
                            "candidate_key",
                            (
                                f"{corroborated_existing.source}:"
                                f"{corroborated_existing.semantic_name}"
                            ),
                        )
                    )
                    if existing_key in corroborated_existing_keys:
                        rejected["duplicate_mask"] += 1
                        continue
                    corroborated_existing_keys.add(existing_key)
                else:
                    rejected["duplicate_mask"] += 1
                    continue
            if any(
                item.semantic_name == semantic_name
                and mask_iou(item.mask, clipped)
                >= self.config.same_semantic_mask_nms_iou
                for item in accepted
            ):
                rejected["duplicate_mask"] += 1
                continue
            if corroborated_existing is None:
                semantic_counts[semantic_name] = (
                    semantic_counts.get(semantic_name, 0) + 1
                )
                instance_number = semantic_counts[semantic_name]
            else:
                instance_number = max(1, semantic_counts.get(semantic_name, 1))
            semantic_parent = part.semantic_parent or domain.name
            candidate_key = (
                f"vlm-region:{root.metadata.get('root_index', 'unknown')}:"
                f"{semantic_name}:{instance_number}"
            )
            accepted.append(
                MaskCandidate(
                    semantic_name=semantic_name,
                    semantic_parent=semantic_parent,
                    mask=clipped,
                    score=float(np.clip(proposal_score, 0.0, 1.0)),
                    source=(
                        f"{self.planner.backend_id}+{region.source}/region-label"
                    ),
                    prompt=semantic_name.replace("_", " "),
                    source_reliability=self.config.source_reliability,
                    metadata={
                        "root_origin": root.metadata.get("root_origin", "legacy"),
                        "root_index": root.metadata.get("root_index"),
                        "candidate_key": candidate_key,
                        "parent_candidate_key": root.metadata.get("candidate_key"),
                        "assembly_parent_semantic": (
                            part.assembly_parent or semantic_parent
                        ),
                        "maximum_instances": part.maximum_instances,
                        "detail": part.detail,
                        "sam_quality": float(
                            region.metadata.get("sam_quality", region.score)
                        ),
                        "selected_part_profile": selected_profile,
                        "vlm_planner_backend": self.planner.backend_id,
                        "vlm_proposal_only": True,
                        "vlm_region_label": True,
                        "vlm_region_candidate_key": region.metadata.get(
                            "candidate_key"
                        ),
                        "vlm_region_previous_semantic": region.semantic_name,
                        "vlm_region_label_confidence": parsed.confidence,
                        "vlm_region_entity_kind": parsed.entity_kind,
                        "vlm_region_corroborates_existing": (
                            corroborated_existing is not None
                        ),
                        "vlm_region_corroborated_candidate_key": (
                            corroborated_existing.metadata.get("candidate_key")
                            if corroborated_existing is not None
                            else None
                        ),
                        "vlm_region_label_parse": parsed.diagnostics,
                        "generic_visual_region": False,
                        "visual_region": True,
                        "visual_region_kind": region.metadata.get(
                            "visual_region_kind"
                        ),
                        "ground_truth_used": False,
                    },
                )
            )
        return accepted, rows, query_count

    def _existing_regions(
        self,
        candidates: Sequence[MaskCandidate],
        domain_name: str,
    ) -> list[PartRegion]:
        regions: list[PartRegion] = []
        for candidate in candidates:
            generic = self._is_generic_region(candidate, domain_name)
            regions.append(
                PartRegion(
                    mask=candidate.mask.astype(bool),
                    source=candidate.source,
                    quality=float(
                        candidate.metadata.get("sam_quality", candidate.score)
                    ),
                    candidate_key=str(
                        candidate.metadata.get("candidate_key", "unkeyed")
                    ),
                    semantic_name=None if generic else candidate.semantic_name,
                    generic=generic,
                    region_kind=(
                        str(candidate.metadata["visual_region_kind"])
                        if candidate.metadata.get("visual_region_kind") is not None
                        else None
                    ),
                )
            )
        return regions

    def _parent_supports(
        self,
        candidates: Sequence[MaskCandidate],
        domain_name: str,
    ) -> dict[str, list[tuple[np.ndarray, str]]]:
        supports: dict[str, list[tuple[np.ndarray, str]]] = {}
        for candidate in candidates:
            if self._is_generic_region(candidate, domain_name):
                continue
            candidate_key = str(
                candidate.metadata.get("candidate_key", "unkeyed")
            )
            supports.setdefault(candidate.semantic_name, []).append(
                (candidate.mask.astype(bool), candidate_key)
            )
        return supports


@dataclass(frozen=True)
class Qwen3VlPlannerConfig:
    model_name: str = "Qwen/Qwen3-VL-2B-Instruct"
    maximum_new_tokens: int = 384
    minimum_pixels: int = 64 * 28 * 28
    maximum_pixels: int = 512 * 28 * 28
    local_files_only: bool = False
    load_in_4bit: bool = False


class Qwen3VlPartPlanner:
    """Local deterministic Qwen3-VL adapter used only for labels and boxes."""

    backend_id = "qwen3-vl-2b-part-planner"

    def __init__(
        self,
        *,
        device: str,
        config: Qwen3VlPlannerConfig | None = None,
        processor: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self.device = device
        self.config = config or Qwen3VlPlannerConfig()
        if (processor is None) != (model is None):
            raise ValueError("Qwen processor and model must be supplied together")
        if processor is None:
            try:
                from transformers import (
                    AutoProcessor,
                    BitsAndBytesConfig,
                    Qwen3VLForConditionalGeneration,
                )
            except ImportError as error:
                raise RuntimeError(
                    "Qwen part planning requires the foundation extra: "
                    "pip install -e '.[foundation]'"
                ) from error
            processor = AutoProcessor.from_pretrained(
                self.config.model_name,
                min_pixels=self.config.minimum_pixels,
                max_pixels=self.config.maximum_pixels,
                local_files_only=self.config.local_files_only,
            )
            dtype = torch.float16 if device.startswith("cuda") else torch.float32
            model_kwargs: dict[str, object] = {
                "dtype": dtype,
                "attn_implementation": "sdpa",
                "local_files_only": self.config.local_files_only,
            }
            if self.config.load_in_4bit:
                if not device.startswith("cuda"):
                    raise ValueError("4-bit Qwen planning requires a CUDA device")
                model_kwargs.update(
                    {
                        "quantization_config": BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_compute_dtype=dtype,
                            bnb_4bit_use_double_quant=True,
                        ),
                        "device_map": {"": 0},
                    }
                )
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.config.model_name,
                    **model_kwargs,
                )
            else:
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.config.model_name,
                    **model_kwargs,
                ).to(device)
        self.processor = processor
        self.model = model
        self.quantized = bool(self.config.load_in_4bit)
        model_label = re.sub(
            r"[^a-z0-9]+",
            "-",
            self.config.model_name.replace("\\", "/").rsplit("/", 1)[-1].lower(),
        ).strip("-")
        self.backend_id = f"{model_label}-part-planner"
        self.model.eval()

    def activate(self) -> None:
        if not self.quantized:
            self.model.to(self.device)
        self.model.eval()

    def release(self) -> None:
        if self.device.startswith("cuda") and not self.quantized:
            self.model.to("cpu")
            torch.cuda.empty_cache()

    def generate_response(self, image: Image.Image, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a deterministic visual JSON API. Return exactly the "
                    "requested JSON object with no analysis, markdown, prose, or "
                    "thinking tags."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.config.maximum_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        input_ids = inputs["input_ids"]
        trimmed = [
            output[len(source) :]
            for source, output in zip(input_ids, generated, strict=True)
        ]
        return str(
            self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        )
