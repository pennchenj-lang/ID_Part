from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from .dense_semantic import DenseSemanticProposer, parent_envelope
from .fusion import MaskCandidate, mask_iou
from .guided_prompts import GuidedPromptSpec
from .prompt_bank import DomainPrompt, PartProfile, PartPrompt, PromptBank


@dataclass(frozen=True)
class AutomaticAssetQuery:
    """One full-image asset proposal routed without a target mask or label."""

    label: str
    domain: str
    profile: str | None = None
    score: float = 0.0
    rank: int = 1
    accepted: bool = False
    negative_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class FoundationConfig:
    grounding_model: str = "IDEA-Research/grounding-dino-tiny"
    segmentation_model: str = "facebook/sam2.1-hiera-tiny"
    box_threshold: float = 0.24
    text_threshold: float = 0.20
    root_nms_iou: float = 0.70
    maximum_roots_per_domain: int = 4
    maximum_total_roots: int = 16
    maximum_children_per_root: int = 36
    maximum_children_per_parent: int = 16
    part_detection_hypotheses_per_instance: int = 3
    maximum_hierarchy_depth: int = 4
    maximum_hierarchy_candidates_per_root: int = 72
    minimum_parent_containment: float = 0.10
    segmentation_batch_size: int = 16
    child_box_nms_iou: float = 0.76
    use_semantic_quota: bool = True
    isolate_semantic_prompts: bool = False
    crop_padding: float = 0.12
    part_box_padding_ratio: float = 0.16
    repeated_part_maximum_fraction_relative_slack: float = 0.12
    repeated_part_maximum_fraction_absolute_slack: float = 0.02
    use_dense_semantic_fallback: bool = False
    use_root_domain_arbitration: bool = True
    use_scene_profile_root_queries: bool = False
    dense_semantic_model: str = "CIDAS/clipseg-rd64-refined"
    dense_detail_only: bool = False
    dense_require_opt_in: bool = True
    dense_only_missing: bool = True
    dense_parent_dilation_ratio: float = 0.06
    dense_minimum_peak_probability: float = 0.08
    dense_minimum_peak_contrast: float = 0.035
    dense_activation_quantile: float = 0.88
    dense_peak_ratio: float = 0.46
    dense_box_padding_ratio: float = 0.30
    dense_minimum_sam_quality: float = 0.45
    dense_source_reliability: float = 0.74
    guided_dense_minimum_top_mean: float = 0.055
    guided_dense_minimum_contrast: float = 0.008
    maximum_profile_queries_per_domain: int = 8
    maximum_profile_roots_per_profile: int = 4
    asset_prompt: str = ""
    asset_prompt_domain: str = ""
    asset_prompt_profile: str = ""
    asset_prompt_label: str = ""
    asset_prompt_resolution_reason: str = ""
    automatic_asset_queries: tuple[AutomaticAssetQuery, ...] = ()
    local_files_only: bool = False
    use_semantic_root_multimask_selection: bool = True
    use_semantic_part_multimask_selection: bool = False
    semantic_multimask_quality_weight: float = 0.58
    semantic_multimask_minimum_target_probability: float = 0.20
    semantic_multimask_minimum_area_ratio: float = 0.30
    semantic_multimask_maximum_quality_drop: float = 0.08
    semantic_multimask_minimum_score_gain: float = 0.002
    lazy_grounding_model: bool = False


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box_xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class CandidateGeneration:
    candidates: tuple[MaskCandidate, ...]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class SegmentProposal:
    mask: np.ndarray
    quality: float
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class RootProposal:
    domain: DomainPrompt
    detection: Detection
    mask: np.ndarray
    sam_quality: float
    query_mode: str = "domain_inventory"
    profile_hint: str | None = None
    model_label: str = ""
    sam_selection: dict[str, object] = field(default_factory=dict)
    automatic_proposal_score: float = 0.0
    automatic_proposal_rank: int | None = None
    automatic_proposal_accepted: bool = False


def _select_semantic_multimask_index(
    qualities: np.ndarray,
    areas: np.ndarray,
    target_rows: list[dict[str, float | int]],
    *,
    quality_weight: float = 0.58,
    minimum_target_probability: float = 0.20,
    minimum_area_ratio: float = 0.30,
    maximum_quality_drop: float = 0.08,
    minimum_score_gain: float = 0.002,
) -> tuple[int, dict[str, object]]:
    """Select a semantically cleaner SAM alternative without accepting fragments."""

    baseline = int(np.argmax(qualities))
    baseline_area = max(1, int(areas[baseline]))
    semantic_weight = 1.0 - quality_weight
    scores: list[float] = []
    eligible: list[bool] = []
    for index, row in enumerate(target_rows):
        probability = float(row.get("probability", 0.0))
        rank = int(row.get("rank", 999))
        score = quality_weight * float(qualities[index]) + semantic_weight * probability
        scores.append(score)
        eligible.append(
            rank == 1
            and probability >= minimum_target_probability
            and float(qualities[index])
            >= float(qualities[baseline]) - maximum_quality_drop
            and int(areas[index]) >= round(baseline_area * minimum_area_ratio)
        )
    selectable = [index for index, allowed in enumerate(eligible) if allowed]
    selected = (
        max(selectable, key=lambda index: scores[index]) if selectable else baseline
    )
    if (
        selected != baseline
        and scores[selected] < scores[baseline] + minimum_score_gain
    ):
        selected = baseline
    return selected, {
        "algorithm": "hpid-text-conditioned-sam-multimask-selection-v1",
        "baseline_index": baseline,
        "selected_index": selected,
        "selection_changed": selected != baseline,
        "candidate_scores": scores,
        "candidate_eligible": eligible,
        "candidate_qualities": [float(value) for value in qualities],
        "candidate_areas": [int(value) for value in areas],
        "target_rows": target_rows,
        "ground_truth_used": False,
    }


def _prompt_text(phrases: tuple[str, ...] | list[str]) -> str:
    cleaned = [value.strip().rstrip(".") for value in phrases if value.strip()]
    return ". ".join(cleaned) + "."


def _box_iou(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    area_first = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    area_second = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union else 0.0


def _mask_containment(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    smaller = min(int(np.count_nonzero(first)), int(np.count_nonzero(second)))
    return intersection / smaller if smaller else 0.0


def _mask_area_coherence(first: np.ndarray, second: np.ndarray) -> float:
    first_area = int(np.count_nonzero(first))
    second_area = int(np.count_nonzero(second))
    larger = max(first_area, second_area)
    return min(first_area, second_area) / larger if larger else 0.0


def _profile_confusion_reassignment(
    source_part: PartPrompt,
    selected_parts: tuple[PartPrompt, ...],
    profile: PartProfile | None,
    *,
    root_area_fraction: float,
    root_containment: float,
    default_minimum_containment: float,
) -> PartPrompt | None:
    """Resolve one category-conditioned role when geometry leaves one choice."""

    if profile is None:
        return None
    group = profile.confusion_group_for(source_part.semantic_name)
    if not group:
        return None
    compatible: list[PartPrompt] = []
    for part in selected_parts:
        minimum_containment = (
            part.minimum_parent_containment
            if part.minimum_parent_containment is not None
            else default_minimum_containment
        )
        if (
            part.semantic_name != source_part.semantic_name
            and part.semantic_name in group
            and part.minimum_parent_fraction
            <= root_area_fraction
            <= part.maximum_parent_fraction
            and root_containment >= minimum_containment
        ):
            compatible.append(part)
    return compatible[0] if len(compatible) == 1 else None


def _padded_box_envelope(
    image_shape: tuple[int, int],
    box_xyxy: tuple[int, int, int, int],
    padding_ratio: float,
) -> np.ndarray:
    """Return a clipped mask around a detector box with proportional padding."""

    height, width = image_shape
    x0, y0, x1, y1 = box_xyxy
    box_width = max(1, x1 - x0)
    box_height = max(1, y1 - y0)
    padding_x = max(2, round(box_width * padding_ratio))
    padding_y = max(2, round(box_height * padding_ratio))
    envelope = np.zeros((height, width), dtype=bool)
    envelope[
        max(0, y0 - padding_y) : min(height, y1 + padding_y),
        max(0, x0 - padding_x) : min(width, x1 + padding_x),
    ] = True
    return envelope


def _surround_mask(
    root_mask: np.ndarray,
    anchor_mask: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, int]:
    """Derive a visible outer ring around an accepted internal anchor."""

    ys, xs = np.nonzero(anchor_mask)
    if not len(xs):
        return np.zeros(root_mask.shape, dtype=bool), 0
    anchor_width = int(xs.max() - xs.min() + 1)
    anchor_height = int(ys.max() - ys.min() + 1)
    radius = max(2, round(min(anchor_width, anchor_height) * scale))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius + 1, 2 * radius + 1),
    )
    expanded = cv2.dilate(anchor_mask.astype(np.uint8), kernel).astype(bool)
    ring = expanded & root_mask & ~anchor_mask
    if not ring.any():
        return ring, radius
    adjacency = cv2.dilate(
        anchor_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)
    ).astype(bool)
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        ring.astype(np.uint8), 8
    )
    connected = np.zeros(ring.shape, dtype=bool)
    for component_id in range(1, count):
        component = components == component_id
        if int(stats[component_id, cv2.CC_STAT_AREA]) < 12:
            continue
        if np.any(component & adjacency):
            connected |= component
    return connected, radius


def _terminal_complement_mask(
    root_mask: np.ndarray,
    anchor_mask: np.ndarray,
    *,
    anchor_position: float | None,
    target_position: float | None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Recover an endpoint part from the root remainder beside an ordered anchor."""

    root_y, root_x = np.nonzero(root_mask)
    anchor_y, anchor_x = np.nonzero(anchor_mask & root_mask)
    if len(root_x) < 8 or len(anchor_x) < 4:
        return np.zeros(root_mask.shape, dtype=bool), {"component_count": 0}
    points = np.column_stack((root_x, root_y)).astype(np.float64)
    center = points.mean(axis=0)
    covariance = np.cov((points - center).T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, int(np.argmax(eigenvalues))]
    projections = (points - center) @ direction
    low, high = float(projections.min()), float(projections.max())
    half_extent = max(1e-6, 0.5 * (high - low))
    midpoint = 0.5 * (low + high)

    anchor_points = np.column_stack((anchor_x, anchor_y)).astype(np.float64)
    anchor_coordinate = float(
        np.mean((anchor_points - center) @ direction - midpoint) / half_extent
    )
    expected_anchor = float(anchor_position if anchor_position is not None else -0.5)
    expected_target = float(target_position if target_position is not None else 0.7)
    orientation_sign = 1.0
    if abs(expected_anchor) >= 0.05 and anchor_coordinate * expected_anchor < 0.0:
        orientation_sign = -1.0

    yy, xx = np.indices(root_mask.shape)
    pixel_points = np.stack((xx - center[0], yy - center[1]), axis=-1)
    normalized = orientation_sign * (pixel_points @ direction - midpoint) / half_extent
    split = 0.5 * (expected_anchor + expected_target)
    target_side = (
        normalized >= split
        if expected_target >= expected_anchor
        else normalized <= split
    )
    residual = root_mask & ~anchor_mask & target_side

    count, components, stats, _ = cv2.connectedComponentsWithStats(
        residual.astype(np.uint8), 8
    )
    if count <= 1:
        return residual, {
            "component_count": int(max(0, count - 1)),
            "orientation_sign": float(orientation_sign),
            "split_coordinate": float(split),
        }
    adjacency_radius = max(2, round(min(root_mask.shape) * 0.01))
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * adjacency_radius + 1, 2 * adjacency_radius + 1),
    )
    anchor_neighborhood = cv2.dilate(anchor_mask.astype(np.uint8), kernel).astype(bool)
    ranked: list[tuple[float, int]] = []
    for component_id in range(1, count):
        component = components == component_id
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < 8:
            continue
        component_coordinate = float(np.mean(normalized[component]))
        endpoint_affinity = 1.0 - min(
            1.0,
            abs(component_coordinate - expected_target) / 2.0,
        )
        adjacency = float(np.any(component & anchor_neighborhood))
        score = np.log1p(area) + 0.75 * endpoint_affinity + 0.35 * adjacency
        ranked.append((float(score), component_id))
    if not ranked:
        return np.zeros(root_mask.shape, dtype=bool), {
            "component_count": int(count - 1),
            "orientation_sign": float(orientation_sign),
            "split_coordinate": float(split),
        }
    selected_id = max(ranked)[1]
    selected = components == selected_id
    return selected, {
        "component_count": int(count - 1),
        "selected_component": int(selected_id),
        "orientation_sign": float(orientation_sign),
        "split_coordinate": float(split),
        "anchor_coordinate": float(orientation_sign * anchor_coordinate),
        "adjacency_radius_px": int(adjacency_radius),
    }


def _remove_ambiguous_guided_candidates(
    candidates: list[MaskCandidate],
    *,
    overlap_threshold: float = 0.88,
    minimum_area_coherence: float = 0.72,
    confidence_margin: float = 0.08,
) -> tuple[list[MaskCandidate], int]:
    """Reject same-scale duplicate masks while preserving nested parts."""

    rejected: set[int] = set()
    replacements: dict[int, MaskCandidate] = {}
    for index, candidate in enumerate(candidates):
        conflicts = [
            other_index
            for other_index, other in enumerate(candidates)
            if other_index != index
            and other.semantic_name != candidate.semantic_name
            and other.metadata.get("root_origin")
            == candidate.metadata.get("root_origin")
            and other.metadata.get("root_index") == candidate.metadata.get("root_index")
            and _mask_containment(candidate.mask, other.mask) >= overlap_threshold
            and _mask_area_coherence(candidate.mask, other.mask)
            >= minimum_area_coherence
        ]
        if not conflicts:
            continue
        group = [index, *conflicts]
        hierarchy_anchors = [
            item
            for item in group
            if all(
                other == item
                or candidates[other].semantic_parent
                == candidates[item].semantic_name
                for other in group
            )
        ]
        if hierarchy_anchors:
            # A detector commonly returns the same physical component as, for
            # example, wheel, tire and rim.  The masks are not contradictory:
            # they are unresolved levels of one known hierarchy.  Preserve the
            # broad physical anchor and collapse its child labels instead of
            # deleting the complete component.
            retained = max(
                hierarchy_anchors,
                key=lambda item: candidates[item].score,
            )
            rejected.update(item for item in group if item != retained)
            replacements[retained] = replace(
                candidates[retained],
                metadata={
                    **candidates[retained].metadata,
                    "hierarchical_ambiguity_collapsed": True,
                    "collapsed_semantics": sorted(
                        {
                            candidates[item].semantic_name
                            for item in group
                            if item != retained
                        }
                    ),
                },
            )
            continue
        ranked = sorted(group, key=lambda item: candidates[item].score, reverse=True)
        if (
            candidates[ranked[0]].score - candidates[ranked[1]].score
            < confidence_margin
        ):
            rejected.update(group)
    return (
        [
            replacements.get(index, candidate)
            for index, candidate in enumerate(candidates)
            if index not in rejected
        ],
        len(rejected),
    )


def _effective_maximum_parent_fraction(
    part: PartPrompt,
    config: FoundationConfig,
) -> float:
    """Allow modest SAM-boundary slack for repeated physical components."""

    maximum = float(part.maximum_parent_fraction)
    if part.maximum_instances <= 1 or part.detail:
        return maximum
    slack = max(
        float(config.repeated_part_maximum_fraction_absolute_slack),
        maximum * float(config.repeated_part_maximum_fraction_relative_slack),
    )
    return min(1.0, maximum + slack)


def _nms_detections(
    detections: list[Detection], threshold: float, limit: int
) -> list[Detection]:
    kept: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.score, reverse=True):
        if any(
            _box_iou(detection.box_xyxy, existing.box_xyxy) >= threshold
            for existing in kept
        ):
            continue
        kept.append(detection)
        if len(kept) >= limit:
            break
    return kept


def _select_root_detections(
    detections: list[Detection], threshold: float, limit: int
) -> list[Detection]:
    """Keep spatially distinct root hypotheses without wasting SAM prompts.

    Grounding DINO often returns the same box once for every synonymous phrase.
    Allocating one slot per text label lets those duplicate boxes consume the
    entire SAM budget and can hide a lower-scoring, tighter object box.  Root
    identity is established downstream, so this stage first deduplicates boxes
    across labels and then distributes the remaining hypotheses across labels.
    Distinct instances and materially different scales are retained.
    """

    spatially_distinct: list[Detection] = []
    for detection in sorted(detections, key=lambda item: item.score, reverse=True):
        if any(
            _box_iou(detection.box_xyxy, existing.box_xyxy) >= threshold
            for existing in spatially_distinct
        ):
            continue
        spatially_distinct.append(detection)

    grouped: dict[str, list[Detection]] = {}
    for detection in spatially_distinct:
        label = re.sub(r"[^a-z0-9]+", " ", detection.label.lower()).strip()
        queue = grouped.setdefault(label, [])
        queue.append(detection)
    selected: list[Detection] = []
    depth = 0
    while len(selected) < limit:
        round_items = [queue[depth] for queue in grouped.values() if depth < len(queue)]
        if not round_items:
            break
        round_items.sort(key=lambda item: item.score, reverse=True)
        selected.extend(round_items[: limit - len(selected)])
        depth += 1
    return selected


def _select_child_detections(
    mapped: list[tuple[Detection, PartPrompt]],
    *,
    limit: int,
    nms_iou: float,
    use_semantic_quota: bool,
) -> tuple[list[tuple[Detection, PartPrompt]], dict[str, object]]:
    """Allocate the SAM budget across semantic classes before segmentation."""
    grouped: dict[str, list[tuple[Detection, PartPrompt]]] = {}
    for item in sorted(mapped, key=lambda value: value[0].score, reverse=True):
        detection, part = item
        queue = grouped.setdefault(part.semantic_name, [])
        if len(queue) >= part.maximum_instances:
            continue
        if any(
            _box_iou(detection.box_xyxy, existing[0].box_xyxy) >= nms_iou
            for existing in queue
        ):
            continue
        queue.append(item)

    if use_semantic_quota:
        selected: list[tuple[Detection, PartPrompt]] = []
        depth = 0
        while len(selected) < limit:
            round_items = [
                queue[depth] for queue in grouped.values() if depth < len(queue)
            ]
            if not round_items:
                break
            round_items.sort(
                key=lambda item: item[0].score * item[1].priority,
                reverse=True,
            )
            selected.extend(round_items[: limit - len(selected)])
            depth += 1
    else:
        selected = sorted(
            (item for queue in grouped.values() for item in queue),
            key=lambda item: item[0].score,
            reverse=True,
        )[:limit]

    diagnostics: dict[str, object] = {
        "mapped_detection_count": len(mapped),
        "post_nms_detection_count": sum(len(queue) for queue in grouped.values()),
        "selected_detection_count": len(selected),
        "detected_semantic_count": len(grouped),
        "selected_semantic_count": len({part.semantic_name for _, part in selected}),
        "selected_semantics": sorted({part.semantic_name for _, part in selected}),
        "semantic_quota_enabled": use_semantic_quota,
    }
    return selected, diagnostics


def _crop_box(
    box: tuple[int, int, int, int],
    image_size: tuple[int, int],
    padding: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    width, height = image_size
    px = round((x1 - x0) * padding)
    py = round((y1 - y0) * padding)
    return (
        max(0, x0 - px),
        max(0, y0 - py),
        min(width, x1 + px),
        min(height, y1 + py),
    )


def _mask_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("cannot crop an empty candidate mask")
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


class FoundationCandidateGenerator:
    """Grounded box proposals plus SAM2 masks; no label map enters this class."""

    def __init__(
        self,
        prompt_bank: PromptBank,
        *,
        device: str = "cuda",
        config: FoundationConfig | None = None,
        sam_processor: Any | None = None,
        sam_model: Any | None = None,
        dense_proposer: DenseSemanticProposer | None = None,
    ) -> None:
        try:
            from transformers import (
                AutoModelForZeroShotObjectDetection,
                AutoProcessor,
                Sam2Model,
                Sam2Processor,
            )
        except ImportError as error:
            raise RuntimeError(
                "Install the foundation extra: pip install 'hpid-split[foundation]'"
            ) from error
        self.prompt_bank = prompt_bank
        self.device = device
        self.config = config or FoundationConfig()
        self.grounding_processor = None
        self.grounding_model = None
        if not self.config.lazy_grounding_model:
            self.grounding_processor = AutoProcessor.from_pretrained(
                self.config.grounding_model,
                local_files_only=self.config.local_files_only,
            )
            self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.config.grounding_model,
                local_files_only=self.config.local_files_only,
            ).to(device)
        if (sam_processor is None) != (sam_model is None):
            raise ValueError("sam_processor and sam_model must be shared together")
        if sam_processor is None:
            self.sam_processor = Sam2Processor.from_pretrained(
                self.config.segmentation_model,
                local_files_only=self.config.local_files_only,
            )
            self.sam_model = Sam2Model.from_pretrained(
                self.config.segmentation_model,
                local_files_only=self.config.local_files_only,
            ).to(device)
        else:
            self.sam_processor = sam_processor
            self.sam_model = sam_model
        if self.grounding_model is not None:
            self.grounding_model.eval()
        self.sam_model.eval()
        self._grounding_model_active = self.grounding_model is not None
        self._grounding_output_mismatch_count = 0
        self._asset_prompt_diagnostics: dict[str, object] | None = None
        self._automatic_asset_diagnostics: dict[str, object] | None = None
        if (
            self.config.use_dense_semantic_fallback
            or self.config.use_root_domain_arbitration
        ):
            self.dense_proposer = dense_proposer or DenseSemanticProposer(
                self.config.dense_semantic_model,
                device=device,
                local_files_only=self.config.local_files_only,
            )
        else:
            self.dense_proposer = None

    def _grounded_source(self, stage: str) -> str:
        return (
            f"grounded-sam2[{self.config.grounding_model}|"
            f"{self.config.segmentation_model}]/{stage}"
        )

    def _dense_source(self, stage: str) -> str:
        return (
            f"clipseg-sam2[{self.config.dense_semantic_model}|"
            f"{self.config.segmentation_model}]/{stage}"
        )

    def _root_origin(self) -> str:
        return (
            f"grounded-sam2[{self.config.grounding_model}|"
            f"{self.config.segmentation_model}]"
        )

    def release_grounding_model(self) -> None:
        """Move the current detector off GPU before loading another source."""
        if self.grounding_model is None or not self.device.startswith("cuda"):
            return
        self.grounding_model.to("cpu")
        self._grounding_model_active = False
        torch.cuda.empty_cache()

    def activate_grounding_model(self) -> None:
        """Restore the detector after sequential ensemble or SAM3 offloading."""

        if self.grounding_model is None or self.grounding_processor is None:
            try:
                from transformers import (
                    AutoModelForZeroShotObjectDetection,
                    AutoProcessor,
                )
            except ImportError as error:
                raise RuntimeError(
                    "Install the foundation extra: pip install 'hpid-split[foundation]'"
                ) from error
            self.grounding_processor = AutoProcessor.from_pretrained(
                self.config.grounding_model,
                local_files_only=self.config.local_files_only,
            )
            self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.config.grounding_model,
                local_files_only=self.config.local_files_only,
            ).to(self.device)
        self.grounding_model.to(self.device)
        self.grounding_model.eval()
        self._grounding_model_active = True
        if self.dense_proposer is not None:
            self.dense_proposer.model.to(self.device)
            self.dense_proposer.model.eval()

    def prepare_for_completion(self) -> None:
        """Free proposal-model VRAM while retaining the shared SAM2 refiner."""
        self.release_grounding_model()
        if not self.device.startswith("cuda"):
            return
        if self.dense_proposer is not None:
            self.dense_proposer.model.to("cpu")
        torch.cuda.empty_cache()

    def attach_root_domain_evidence(
        self,
        image: Image.Image,
        candidates: list[MaskCandidate],
    ) -> tuple[list[MaskCandidate], dict[str, object] | None]:
        """Compare root categories in the frozen image-text embedding space."""

        if self.dense_proposer is None:
            return candidates, None
        regions: list[tuple[str, np.ndarray]] = []
        for candidate in candidates:
            if not (
                candidate.semantic_name == candidate.semantic_parent
                and candidate.metadata.get("root_index") is not None
                and candidate.metadata.get("parent_candidate_key") is None
            ):
                continue
            root_key = (
                f"{candidate.metadata.get('root_origin', 'legacy')}::"
                f"{candidate.metadata.get('root_index', 'unknown')}"
            )
            regions.append((root_key, candidate.mask))
        regions = list(dict(regions).items())
        labels = [
            (
                domain.name,
                domain.classifier_prompt
                or f"a complete {domain.name.replace('_', ' ')} object",
            )
            for domain in self.prompt_bank.domains
        ]
        rankings = self.dense_proposer.rank_regions_labels(
            image,
            regions,
            labels,
            masked_weight=0.90,
        )
        scores: dict[str, dict[str, dict[str, float | str | int]]] = {}
        for root_key, label_rows in rankings.items():
            scores[root_key] = {}
            for domain_name, row in label_rows.items():
                similarity = float(row["combined_similarity"])
                best_other = max(
                    (
                        float(other["combined_similarity"])
                        for name, other in label_rows.items()
                        if name != domain_name
                    ),
                    default=similarity,
                )
                scores[root_key][domain_name] = {
                    **row,
                    "margin_to_best_other": max(0.0, similarity - best_other),
                }
        enriched: list[MaskCandidate] = []
        for candidate in candidates:
            root_key = (
                f"{candidate.metadata.get('root_origin', 'legacy')}::"
                f"{candidate.metadata.get('root_index', 'unknown')}"
            )
            score = scores.get(root_key, {}).get(candidate.semantic_name)
            if score is None or candidate.semantic_name != candidate.semantic_parent:
                enriched.append(candidate)
                continue
            enriched.append(
                replace(
                    candidate,
                    metadata={
                        **candidate.metadata,
                        "domain_evidence_score": float(score["probability"]),
                        "domain_evidence_contrast": float(
                            score["margin_to_best_other"]
                        ),
                        "domain_evidence_prompt": str(score["prompt"]),
                        "domain_evidence_similarity": float(
                            score["combined_similarity"]
                        ),
                        "domain_evidence_rank": int(score["rank"]),
                    },
                )
            )
        return enriched, {
            "algorithm": "clipseg-embedding-root-arbitration-v3",
            "region_count": len(regions),
            "label_count": len(labels),
            "scores": scores,
            "ground_truth_used": False,
        }

    def generate_isolated_profile_roots(
        self,
        image: Image.Image,
        routed_roots: list[MaskCandidate],
        domains: dict[str, DomainPrompt],
    ) -> CandidateGeneration:
        """Resolve an asset subtype with one independent text query per profile.

        The first pass only establishes physical roots and broad domains. Long
        category inventories can make open-vocabulary detectors concatenate or
        truncate labels, so this second pass compares isolated category queries
        only for domains retained by routing. Geometry still comes from SAM2.
        """

        image = image.convert("RGB")
        selected_domain_names = sorted(
            {
                root.semantic_name
                for root in routed_roots
                if root.semantic_name in domains
            }
        )
        candidates: list[MaskCandidate] = []
        rows: list[dict[str, object]] = []
        profile_query_count = 0
        root_index = 0
        origin = f"{self._root_origin()}/isolated-profile"
        for domain_name in selected_domain_names:
            domain = domains[domain_name]
            domain_roots = [
                root for root in routed_roots if root.semantic_name == domain_name
            ]
            classifier_rows_by_root: dict[str, dict[str, object]] = {}
            if self.dense_proposer is not None:
                labels = [
                    (
                        profile.name,
                        profile.classifier_prompt or " or ".join(profile.root_hints),
                    )
                    for profile in domain.part_profiles
                ]
                for broad_root in domain_roots:
                    broad_key = (
                        f"{broad_root.metadata.get('root_origin', 'legacy')}::"
                        f"{broad_root.metadata.get('root_index', 'unknown')}"
                    )
                    classifier_rows_by_root[broad_key] = (
                        self.dense_proposer.rank_region_labels(
                            image,
                            broad_root.mask,
                            labels,
                        )
                    )
            ranked_profile_names: set[str] = set()
            for broad_root in domain_roots:
                broad_key = (
                    f"{broad_root.metadata.get('root_origin', 'legacy')}::"
                    f"{broad_root.metadata.get('root_index', 'unknown')}"
                )
                classifier_rows = classifier_rows_by_root.get(broad_key, {})
                classifier_ranking = sorted(
                    classifier_rows,
                    key=lambda name: int(classifier_rows[name].get("rank", 999)),
                )
                ranked_profile_names.update(
                    classifier_ranking[: self.config.maximum_profile_queries_per_domain]
                )
                label_profile = domain.select_parts(broad_root.prompt)[1]
                if label_profile is not None:
                    ranked_profile_names.add(label_profile)
            if not ranked_profile_names:
                ranked_profile_names.update(
                    profile.name
                    for profile in domain.part_profiles[
                        : self.config.maximum_profile_queries_per_domain
                    ]
                )
            queried_profiles = [
                profile
                for profile in domain.part_profiles
                if profile.name in ranked_profile_names
            ]
            candidate_count_before_domain = len(candidates)
            for profile in queried_profiles:
                profile_query_count += 1
                detections = _select_root_detections(
                    self._ground(image, list(profile.root_hints)),
                    self.config.root_nms_iou,
                    self.config.maximum_profile_roots_per_profile,
                )
                segmentations = self._segment_part_boxes(
                    image,
                    detections,
                    semantic_prompts=[detection.label for detection in detections],
                    semantic_domain_name=domain.name,
                )
                accepted = 0
                for detection, segmentation in zip(
                    detections, segmentations, strict=True
                ):
                    area = int(np.count_nonzero(segmentation.mask))
                    if area < 20:
                        continue
                    matched_root: MaskCandidate | None = None
                    matched_affinity = 0.0
                    for broad_root in domain_roots:
                        broad_area = max(1, int(np.count_nonzero(broad_root.mask)))
                        intersection = int(
                            np.count_nonzero(segmentation.mask & broad_root.mask)
                        )
                        containment = intersection / max(1, min(area, broad_area))
                        affinity = max(
                            mask_iou(segmentation.mask, broad_root.mask),
                            0.85 * containment,
                        )
                        if affinity > matched_affinity:
                            matched_affinity = float(affinity)
                            matched_root = broad_root
                    classifier_root_key = None
                    classifier_row: dict[str, object] = {}
                    if matched_root is not None:
                        classifier_root_key = (
                            f"{matched_root.metadata.get('root_origin', 'legacy')}::"
                            f"{matched_root.metadata.get('root_index', 'unknown')}"
                        )
                        classifier_row = dict(
                            dict(
                                classifier_rows_by_root.get(classifier_root_key, {})
                            ).get(profile.name, {})
                        )
                    detector_normalized = float(
                        np.clip(detection.score / 0.65, 0.0, 1.0)
                    )
                    classifier_probability = float(
                        classifier_row.get("probability", 0.0)
                    )
                    classifier_rank = int(classifier_row.get("rank", 999))
                    consensus_score = float(
                        0.45 * detector_normalized
                        + 0.35 * classifier_probability
                        + 0.20 * (classifier_rank == 1)
                    )
                    root_index += 1
                    accepted += 1
                    _, selected_profile, profile_diagnostics = domain.select_parts(
                        profile.root_hints[0],
                        profile_hint=profile.name,
                        profile_hint_source="isolated_profile_query",
                    )
                    candidates.append(
                        MaskCandidate(
                            semantic_name=domain.name,
                            semantic_parent=domain.name,
                            mask=segmentation.mask,
                            score=detection.score,
                            source=self._grounded_source("isolated-profile-root"),
                            prompt=profile.root_hints[0],
                            source_reliability=0.80 + 0.20 * segmentation.quality,
                            metadata={
                                "root_origin": origin,
                                "root_index": root_index,
                                "candidate_key": f"profile-root:{root_index}",
                                "parent_candidate_key": None,
                                "sam_quality": float(segmentation.quality),
                                "box_xyxy": list(detection.box_xyxy),
                                "root_label_specificity": 1.0,
                                "part_profile_specificity": 1.0,
                                "selected_part_profile": selected_profile,
                                "part_profile_selection": profile_diagnostics,
                                "root_query_mode": "isolated_profile_query",
                                "root_model_label": detection.label,
                                "profile_hint_source": ("isolated_profile_query"),
                                "profile_detector_score": float(detection.score),
                                "profile_classifier": classifier_row,
                                "profile_classifier_inventory_count": len(
                                    domain.part_profiles
                                ),
                                "profile_classifier_root_key": (classifier_root_key),
                                "profile_classifier_root_affinity": (matched_affinity),
                                "profile_consensus_score": consensus_score,
                                "ground_truth_used": False,
                            },
                        )
                    )
                accepted_rows = [
                    candidate
                    for candidate in candidates[candidate_count_before_domain:]
                    if candidate.semantic_name == domain.name
                    and candidate.metadata.get("root_query_mode")
                    == "isolated_profile_query"
                    and candidate.metadata.get("selected_part_profile") == profile.name
                ]
                rows.append(
                    {
                        "domain": domain.name,
                        "profile": profile.name,
                        "query": list(profile.root_hints),
                        "detection_count": len(detections),
                        "accepted_root_count": accepted,
                        "accepted_roots": [
                            {
                                "root_index": candidate.metadata.get("root_index"),
                                "detector_score": candidate.metadata.get(
                                    "profile_detector_score"
                                ),
                                "classifier_rank": dict(
                                    candidate.metadata.get("profile_classifier", {})
                                ).get("rank"),
                                "classifier_probability": dict(
                                    candidate.metadata.get("profile_classifier", {})
                                ).get("probability"),
                                "classifier_similarity": dict(
                                    candidate.metadata.get("profile_classifier", {})
                                ).get("combined_similarity"),
                                "consensus_score": candidate.metadata.get(
                                    "profile_consensus_score"
                                ),
                                "root_affinity": candidate.metadata.get(
                                    "profile_classifier_root_affinity"
                                ),
                                "area_px": int(np.count_nonzero(candidate.mask)),
                                "box_xyxy": candidate.metadata.get("box_xyxy"),
                                "model_label": candidate.metadata.get(
                                    "root_model_label"
                                ),
                            }
                            for candidate in accepted_rows
                        ],
                        "classifier": {
                            key: value[profile.name]
                            for key, value in classifier_rows_by_root.items()
                            if profile.name in value
                        },
                    }
                )

            domain_profile_candidates = candidates[candidate_count_before_domain:]
            if self.dense_proposer is not None and domain_profile_candidates:
                labels = [
                    (
                        profile.name,
                        profile.classifier_prompt or " or ".join(profile.root_hints),
                    )
                    for profile in domain.part_profiles
                ]
                region_keys = [
                    str(candidate.metadata.get("candidate_key"))
                    for candidate in domain_profile_candidates
                ]
                self_classifier_rows = self.dense_proposer.rank_regions_labels(
                    image,
                    [
                        (key, candidate.mask)
                        for key, candidate in zip(
                            region_keys, domain_profile_candidates, strict=True
                        )
                    ],
                    labels,
                )
                enriched_profile_candidates: list[MaskCandidate] = []
                self_classification_diagnostics: list[dict[str, object]] = []
                for key, candidate in zip(
                    region_keys, domain_profile_candidates, strict=True
                ):
                    profile_name = str(
                        candidate.metadata.get("selected_part_profile") or ""
                    )
                    label_rows = self_classifier_rows.get(key, {})
                    self_classifier = dict(label_rows.get(profile_name, {}))
                    ordered = sorted(
                        label_rows.values(),
                        key=lambda row: int(row.get("rank", 999)),
                    )
                    runner_up_probability = (
                        float(ordered[1].get("probability", 0.0))
                        if len(ordered) > 1
                        else 0.0
                    )
                    runner_up_similarity = (
                        float(ordered[1].get("combined_similarity", 0.0))
                        if len(ordered) > 1
                        else 0.0
                    )
                    self_probability = float(self_classifier.get("probability", 0.0))
                    self_similarity = float(
                        self_classifier.get("combined_similarity", 0.0)
                    )
                    self_rank = int(self_classifier.get("rank", 999))
                    probability_ratio = self_probability / max(
                        1e-8, runner_up_probability
                    )
                    similarity_margin = self_similarity - runner_up_similarity
                    context_classifier = dict(
                        candidate.metadata.get("profile_classifier", {})
                    )
                    detector_normalized = float(
                        np.clip(
                            float(
                                candidate.metadata.get(
                                    "profile_detector_score", candidate.score
                                )
                            )
                            / 0.65,
                            0.0,
                            1.0,
                        )
                    )
                    consensus_score = float(
                        0.42 * detector_normalized
                        + 0.28 * self_probability
                        + 0.18 * (self_rank == 1)
                        + 0.08 * float(context_classifier.get("probability", 0.0))
                        + 0.04 * (int(context_classifier.get("rank", 999)) == 1)
                    )
                    metadata = {
                        **candidate.metadata,
                        "profile_context_classifier": context_classifier,
                        "profile_self_classifier": self_classifier,
                        "profile_classifier": self_classifier,
                        "profile_classifier_root_key": key,
                        "profile_classifier_probability_ratio": probability_ratio,
                        "profile_classifier_similarity_margin": similarity_margin,
                        "profile_candidate_self_classified": True,
                        "profile_consensus_score": consensus_score,
                    }
                    enriched_profile_candidates.append(
                        replace(candidate, metadata=metadata)
                    )
                    self_classification_diagnostics.append(
                        {
                            "candidate_key": key,
                            "profile": profile_name,
                            "rank": self_rank,
                            "probability": self_probability,
                            "probability_ratio": probability_ratio,
                            "similarity": self_similarity,
                            "similarity_margin": similarity_margin,
                            "consensus_score": consensus_score,
                        }
                    )
                candidates[candidate_count_before_domain:] = enriched_profile_candidates
                rows.append(
                    {
                        "domain": domain.name,
                        "selection_stage": "candidate_self_classification",
                        "candidate_count": len(enriched_profile_candidates),
                        "candidates": self_classification_diagnostics,
                    }
                )
            virtual_profile_count = 0
            profiles_by_name = {
                profile.name: profile for profile in domain.part_profiles
            }
            for broad_root in domain_roots:
                broad_key = (
                    f"{broad_root.metadata.get('root_origin', 'legacy')}::"
                    f"{broad_root.metadata.get('root_index', 'unknown')}"
                )
                classifier_rows = classifier_rows_by_root.get(broad_key, {})
                matched_existing_profiles: set[str] = set()
                broad_area = max(1, int(np.count_nonzero(broad_root.mask)))
                for candidate in candidates[candidate_count_before_domain:]:
                    profile_name = candidate.metadata.get("selected_part_profile")
                    if not profile_name:
                        continue
                    candidate_area = max(1, int(np.count_nonzero(candidate.mask)))
                    intersection = int(
                        np.count_nonzero(candidate.mask & broad_root.mask)
                    )
                    containment = intersection / max(1, min(candidate_area, broad_area))
                    affinity = max(
                        mask_iou(candidate.mask, broad_root.mask),
                        0.85 * containment,
                    )
                    if affinity >= 0.12:
                        matched_existing_profiles.add(str(profile_name))
                label_profile = domain.select_parts(
                    str(
                        broad_root.metadata.get("root_model_label") or broad_root.prompt
                    )
                )[1]
                ranked_names = sorted(
                    classifier_rows,
                    key=lambda name: int(classifier_rows[name].get("rank", 999)),
                )
                top_name = ranked_names[0] if ranked_names else None
                virtual_names = {
                    name
                    for name in (label_profile, top_name)
                    if name is not None
                    and name in ranked_profile_names
                    and name not in matched_existing_profiles
                }
                for profile_name in sorted(virtual_names):
                    profile = profiles_by_name[profile_name]
                    classifier_row = dict(classifier_rows.get(profile_name, {}))
                    classifier_probability = float(
                        classifier_row.get("probability", 0.0)
                    )
                    classifier_rank = int(classifier_row.get("rank", 999))
                    label_support = profile_name == label_profile
                    consensus_score = float(
                        0.35 * classifier_probability
                        + 0.20 * (classifier_rank == 1)
                        + 0.22 * label_support
                    )
                    if consensus_score < 0.20:
                        continue
                    _, selected_profile, profile_diagnostics = domain.select_parts(
                        profile.root_hints[0],
                        profile_hint=profile.name,
                        profile_hint_source="classifier_or_root_label_hypothesis",
                    )
                    root_index += 1
                    virtual_profile_count += 1
                    candidates.append(
                        MaskCandidate(
                            semantic_name=domain.name,
                            semantic_parent=domain.name,
                            mask=broad_root.mask,
                            score=broad_root.score,
                            source=self._grounded_source("profile-hypothesis"),
                            prompt=profile.root_hints[0],
                            source_reliability=broad_root.source_reliability,
                            metadata={
                                **broad_root.metadata,
                                "root_origin": origin,
                                "root_index": root_index,
                                "candidate_key": f"profile-hypothesis:{root_index}",
                                "parent_candidate_key": None,
                                "selected_part_profile": selected_profile,
                                "part_profile_selection": profile_diagnostics,
                                "root_query_mode": "classifier_or_root_label_hypothesis",
                                "root_model_label": broad_root.metadata.get(
                                    "root_model_label"
                                ),
                                "profile_hint_source": (
                                    "classifier_or_root_label_hypothesis"
                                ),
                                "profile_detector_score": 0.0,
                                "profile_classifier": classifier_row,
                                "profile_classifier_inventory_count": len(
                                    domain.part_profiles
                                ),
                                "profile_classifier_root_key": broad_key,
                                "profile_classifier_root_affinity": 1.0,
                                "profile_consensus_score": consensus_score,
                                "profile_virtual_hypothesis": True,
                                "profile_root_label_support": label_support,
                                "ground_truth_used": False,
                            },
                        )
                    )
            rows.append(
                {
                    "domain": domain.name,
                    "query_budget": self.config.maximum_profile_queries_per_domain,
                    "available_profile_count": len(domain.part_profiles),
                    "queried_profiles": [profile.name for profile in queried_profiles],
                    "skipped_profiles": [
                        profile.name
                        for profile in domain.part_profiles
                        if profile.name not in ranked_profile_names
                    ],
                    "selection_stage": "classifier_top_k_plus_root_label",
                    "virtual_profile_hypothesis_count": virtual_profile_count,
                }
            )
        return CandidateGeneration(
            tuple(candidates),
            {
                "algorithm": "hpid-isolated-profile-root-resolution-v1",
                "selected_domains": selected_domain_names,
                "profile_query_count": profile_query_count,
                "candidate_count": len(candidates),
                "profiles": rows,
                "ground_truth_used": False,
            },
        )

    def generate_guided_parts(
        self,
        image: Image.Image,
        roots: list[MaskCandidate],
        prompts: tuple[GuidedPromptSpec, ...],
        *,
        require_dense_gate: bool = True,
    ) -> CandidateGeneration:
        """Ground user-named parts inside routed roots and segment their boundaries."""

        image = image.convert("RGB")
        candidates: list[MaskCandidate] = []
        root_rows: list[dict[str, object]] = []
        for root in roots:
            root_area = max(1, int(np.count_nonzero(root.mask)))
            crop_box = _crop_box(
                _mask_box(root.mask), image.size, self.config.crop_padding
            )
            crop = image.crop(crop_box)
            mapped: list[tuple[Detection, GuidedPromptSpec]] = []
            per_prompt_detections: dict[str, int] = {}
            for spec in prompts:
                detections = _nms_detections(
                    self._ground(crop, list(spec.phrases)),
                    self.config.child_box_nms_iou,
                    spec.maximum_instances,
                )
                per_prompt_detections[spec.slug] = len(detections)
                mapped.extend((detection, spec) for detection in detections)
            mapped.sort(key=lambda item: item[0].score, reverse=True)
            mapped = mapped[: self.config.maximum_children_per_root]
            segmentations = self._segment_part_boxes(
                crop,
                [detection for detection, _ in mapped],
                semantic_prompts=[spec.phrases[0] for _, spec in mapped],
                semantic_domain_name=root.semantic_name,
            )
            x0, y0, x1, y1 = crop_box
            root_envelope = parent_envelope(root.mask, 0.025)
            semantic_ordinals: dict[str, int] = {}
            accepted_for_root = 0
            for (detection, spec), segmentation in zip(
                mapped, segmentations, strict=True
            ):
                full_mask = np.zeros((image.height, image.width), dtype=bool)
                full_mask[y0:y1, x0:x1] = segmentation.mask
                local_box_envelope = _padded_box_envelope(
                    (crop.height, crop.width),
                    detection.box_xyxy,
                    self.config.part_box_padding_ratio,
                )
                full_box_envelope = np.zeros((image.height, image.width), dtype=bool)
                full_box_envelope[y0:y1, x0:x1] = local_box_envelope
                original_area = int(np.count_nonzero(full_mask))
                containment = int(np.count_nonzero(full_mask & root_envelope)) / max(
                    1, original_area
                )
                full_mask &= root_envelope & full_box_envelope
                area = int(np.count_nonzero(full_mask))
                fraction = area / root_area
                if area < 12 or containment < 0.48 or not 0.0001 <= fraction <= 0.82:
                    continue
                semantic_name = f"{root.semantic_name}_guided_{spec.slug}"
                semantic_ordinals[semantic_name] = (
                    semantic_ordinals.get(semantic_name, 0) + 1
                )
                root_candidate_key = str(
                    root.metadata.get(
                        "candidate_key", f"root:{root.metadata.get('root_index')}"
                    )
                )
                candidate_key = (
                    f"{root_candidate_key}/{semantic_name}:"
                    f"{semantic_ordinals[semantic_name]:02d}"
                )
                candidates.append(
                    MaskCandidate(
                        semantic_name=semantic_name,
                        semantic_parent=root.semantic_name,
                        mask=full_mask,
                        score=detection.score,
                        source=self._grounded_source("guided-part"),
                        prompt=detection.label,
                        source_reliability=(
                            0.86 * (0.55 + 0.45 * segmentation.quality)
                        ),
                        metadata={
                            "source_family": self._root_origin(),
                            "root_origin": root.metadata.get("root_origin"),
                            "root_index": root.metadata.get("root_index"),
                            "candidate_key": candidate_key,
                            "parent_candidate_key": root_candidate_key,
                            "assembly_parent_semantic": root.semantic_name,
                            "assembly_parent_candidate_key": root_candidate_key,
                            "guided_prompt": True,
                            "guided_prompt_label": spec.label,
                            "guided_prompt_slug": spec.slug,
                            "guided_prompt_phrases": list(spec.phrases),
                            "sam_quality": float(segmentation.quality),
                            "root_containment": float(containment),
                            "root_area_fraction": float(fraction),
                            "maximum_instances": spec.maximum_instances,
                            "box_xyxy_local": list(detection.box_xyxy),
                            "crop_xyxy": list(crop_box),
                            "ground_truth_used": False,
                        },
                    )
                )
                accepted_for_root += 1
            root_rows.append(
                {
                    "root_origin": str(root.metadata.get("root_origin")),
                    "root_index": root.metadata.get("root_index"),
                    "root_semantic": root.semantic_name,
                    "detected_by_prompt": per_prompt_detections,
                    "accepted_candidate_count": accepted_for_root,
                }
            )
        candidates, ambiguous_rejections = _remove_ambiguous_guided_candidates(
            candidates
        )
        dense_gate_rows: list[dict[str, object]] = []
        dense_gate_rejections = 0
        if (
            require_dense_gate
            and getattr(self, "dense_proposer", None) is not None
            and candidates
        ):
            queries = []
            for index, candidate in enumerate(candidates):
                phrases = candidate.metadata.get("guided_prompt_phrases", [])
                prompt = str(phrases[0]) if phrases else candidate.prompt
                queries.append((str(index), prompt, candidate.mask))
            dense_scores = self.dense_proposer.score_regions(
                image, queries, top_fraction=0.12
            )
            gated_candidates: list[MaskCandidate] = []
            for index, candidate in enumerate(candidates):
                row = dense_scores[str(index)]
                top_mean = float(row["top_mean"])
                contrast = float(row["contrast"])
                accepted = (
                    top_mean >= self.config.guided_dense_minimum_top_mean
                    and contrast >= self.config.guided_dense_minimum_contrast
                )
                dense_gate_rows.append(
                    {
                        "semantic_name": candidate.semantic_name,
                        "prompt": str(row["prompt"]),
                        "top_mean": top_mean,
                        "contrast": contrast,
                        "accepted": accepted,
                    }
                )
                if not accepted:
                    dense_gate_rejections += 1
                    continue
                gated_candidates.append(
                    replace(
                        candidate,
                        metadata={
                            **candidate.metadata,
                            "guided_dense_top_mean": top_mean,
                            "guided_dense_contrast": contrast,
                            "guided_dense_gate": True,
                        },
                    )
                )
            candidates = gated_candidates
        return CandidateGeneration(
            tuple(candidates),
            {
                "algorithm": "hpid-guided-grounding-fusion-v1",
                "grounding_model": self.config.grounding_model,
                "prompt_count": len(prompts),
                "prompt_labels": [spec.label for spec in prompts],
                "root_count": len(roots),
                "candidate_count": len(candidates),
                "ambiguous_semantic_mask_rejection_count": ambiguous_rejections,
                "dense_semantic_gate_rejection_count": dense_gate_rejections,
                "dense_semantic_gate_required": require_dense_gate,
                "dense_semantic_gate_rows": dense_gate_rows,
                "roots": root_rows,
                "automatic_visual_supplement": True,
                "ground_truth_used": False,
            },
        )

    def refine_profile_parts(
        self,
        image: Image.Image,
        roots: list[MaskCandidate],
        domains: dict[str, DomainPrompt],
    ) -> CandidateGeneration:
        """Run isolated canonical part queries after physical root routing.

        Initial root discovery must remain broad enough to avoid missing assets.
        Once routing fixes one physical entity and category, broad multi-part
        prompts become unnecessary and often merge labels. This pass queries
        only the selected category profile, one semantic at a time, and keeps
        the existing geometric and dense semantic gates.
        """

        image = image.convert("RGB")
        config = getattr(self, "config", None)
        grounding_model = str(
            getattr(config, "grounding_model", "unavailable-without-model")
        )
        grounding_model_key = re.sub(r"[^a-zA-Z0-9._-]+", "_", grounding_model)
        candidates: list[MaskCandidate] = []
        root_rows: list[dict[str, object]] = []
        for root in roots:
            domain = domains.get(root.semantic_name)
            if domain is None:
                continue
            root_label = str(
                root.metadata.get("resolved_object_label")
                or root.metadata.get("root_model_label")
                or root.prompt
                or ""
            )
            profile_hint = (
                str(root.metadata["selected_part_profile"])
                if root.metadata.get("profile_resolution_status") == "accepted"
                and root.metadata.get("selected_part_profile")
                else None
            )
            if (
                domain.part_profiles
                and root.metadata.get("profile_resolution_status") == "unresolved"
            ):
                root_rows.append(
                    {
                        "root_index": root.metadata.get("root_index"),
                        "root_semantic": root.semantic_name,
                        "selected_profile": None,
                        "skipped_reason": "profile_consensus_unresolved",
                    }
                )
                continue
            selected_parts, selected_profile, profile_diagnostics = domain.select_parts(
                root_label,
                profile_hint=profile_hint,
                profile_hint_source=(
                    "isolated_profile_consensus" if profile_hint else None
                ),
            )
            if selected_profile is None:
                root_rows.append(
                    {
                        "root_index": root.metadata.get("root_index"),
                        "root_semantic": root.semantic_name,
                        "selected_profile": None,
                        "skipped_reason": "no_confident_category_profile",
                        "profile": profile_diagnostics,
                    }
                )
                continue
            selected_profile_definition = next(
                (
                    profile
                    for profile in domain.part_profiles
                    if profile.name == selected_profile
                ),
                None,
            )
            root_area = max(1, int(np.count_nonzero(root.mask)))
            crop_box = _crop_box(
                _mask_box(root.mask), image.size, self.config.crop_padding
            )
            crop = image.crop(crop_box)
            root_envelope = parent_envelope(root.mask, 0.025)
            root_candidate_key = str(
                root.metadata.get(
                    "candidate_key", f"root:{root.metadata.get('root_index')}"
                )
            )
            mapped: list[tuple[Detection, PartPrompt]] = []
            detected_by_semantic: dict[str, int] = {}
            for part in selected_parts:
                if part.semantic_name.endswith("_body"):
                    continue
                if part.topology_relation is not None:
                    detected_by_semantic[part.semantic_name] = 0
                    continue
                detections = _nms_detections(
                    self._ground(crop, list(dict.fromkeys(part.phrases))),
                    self.config.child_box_nms_iou,
                    min(
                        self.config.maximum_children_per_parent,
                        max(
                            part.maximum_instances,
                            part.maximum_instances
                            * self.config.part_detection_hypotheses_per_instance,
                        ),
                    ),
                )
                detected_by_semantic[part.semantic_name] = len(detections)
                mapped.extend((detection, part) for detection in detections)
            mapped.sort(
                key=lambda item: item[0].score * item[1].priority,
                reverse=True,
            )
            mapped = mapped[: self.config.maximum_hierarchy_candidates_per_root]
            segmentations = self._segment_part_boxes(
                crop,
                [detection for detection, _ in mapped],
                semantic_prompts=[part.phrases[0] for _, part in mapped],
                semantic_domain_name=domain.name,
            )
            x0, y0, x1, y1 = crop_box
            provisional: list[MaskCandidate] = []
            semantic_ordinals: dict[str, int] = {}
            geometry_rejections = 0
            geometry_rejection_rows: list[dict[str, object]] = []
            semantic_reassignment_rows: list[dict[str, object]] = []
            for (detection, part), segmentation in zip(
                mapped, segmentations, strict=True
            ):
                full_mask = np.zeros((image.height, image.width), dtype=bool)
                local_box_envelope = _padded_box_envelope(
                    (crop.height, crop.width),
                    detection.box_xyxy,
                    self.config.part_box_padding_ratio,
                )
                unconstrained_area = int(np.count_nonzero(segmentation.mask))
                constrained_local_mask = segmentation.mask & local_box_envelope
                full_mask[y0:y1, x0:x1] = constrained_local_mask
                full_mask &= root_envelope
                area = int(np.count_nonzero(full_mask))
                fraction = area / root_area
                containment = int(np.count_nonzero(full_mask & root.mask)) / max(
                    1, area
                )
                minimum_containment = (
                    part.minimum_parent_containment
                    if part.minimum_parent_containment is not None
                    else max(0.48, self.config.minimum_parent_containment)
                )
                maximum_parent_fraction = _effective_maximum_parent_fraction(
                    part,
                    self.config,
                )
                rejection_reasons = []
                if area < 12:
                    rejection_reasons.append("minimum_area")
                if fraction < part.minimum_parent_fraction:
                    rejection_reasons.append("minimum_parent_fraction")
                if fraction > maximum_parent_fraction:
                    rejection_reasons.append("maximum_parent_fraction")
                if containment < minimum_containment:
                    rejection_reasons.append("minimum_parent_containment")
                original_part = part
                area_reasons = {
                    "minimum_parent_fraction",
                    "maximum_parent_fraction",
                }
                if rejection_reasons and set(rejection_reasons).issubset(area_reasons):
                    reassigned = _profile_confusion_reassignment(
                        part,
                        selected_parts,
                        selected_profile_definition,
                        root_area_fraction=fraction,
                        root_containment=containment,
                        default_minimum_containment=max(
                            0.48, self.config.minimum_parent_containment
                        ),
                    )
                    if reassigned is not None:
                        part = reassigned
                        minimum_containment = (
                            part.minimum_parent_containment
                            if part.minimum_parent_containment is not None
                            else max(0.48, self.config.minimum_parent_containment)
                        )
                        semantic_reassignment_rows.append(
                            {
                                "detector_semantic": original_part.semantic_name,
                                "assigned_semantic": part.semantic_name,
                                "detector_label": detection.label,
                                "detector_score": float(detection.score),
                                "root_area_fraction": float(fraction),
                                "root_containment": float(containment),
                                "confusion_group": list(
                                    selected_profile_definition.confusion_group_for(
                                        original_part.semantic_name
                                    )
                                ),
                                "reason": "unique_profile_geometry_role",
                                "ground_truth_used": False,
                            }
                        )
                        rejection_reasons = []
                if rejection_reasons:
                    geometry_rejections += 1
                    geometry_rejection_rows.append(
                        {
                            "semantic_name": part.semantic_name,
                            "detector_label": detection.label,
                            "detector_score": float(detection.score),
                            "box_xyxy_local": list(detection.box_xyxy),
                            "area_px": area,
                            "sam_unconstrained_area_px": unconstrained_area,
                            "sam_box_constrained_area_px": int(
                                np.count_nonzero(constrained_local_mask)
                            ),
                            "root_area_fraction": float(fraction),
                            "minimum_parent_fraction": float(
                                part.minimum_parent_fraction
                            ),
                                "maximum_parent_fraction": float(
                                    maximum_parent_fraction
                                ),
                            "root_containment": float(containment),
                            "minimum_parent_containment": float(minimum_containment),
                            "reasons": rejection_reasons,
                        }
                    )
                    continue
                semantic_ordinals[part.semantic_name] = (
                    semantic_ordinals.get(part.semantic_name, 0) + 1
                )
                candidate_key = (
                    f"{root_candidate_key}/profile-refine:{grounding_model_key}:"
                    f"{part.semantic_name}:"
                    f"{semantic_ordinals[part.semantic_name]:02d}"
                )
                provisional.append(
                    MaskCandidate(
                        semantic_name=part.semantic_name,
                        semantic_parent=part.semantic_parent or domain.name,
                        mask=full_mask,
                        score=detection.score,
                        source=self._grounded_source("profile-refine"),
                        prompt=detection.label,
                        source_reliability=(
                            0.88 * (0.55 + 0.45 * segmentation.quality) * part.priority
                        ),
                        metadata={
                            "source_family": f"{self._root_origin()}/profile-refine",
                            "root_origin": root.metadata.get("root_origin"),
                            "root_index": root.metadata.get("root_index"),
                            "candidate_key": candidate_key,
                            "parent_candidate_key": root_candidate_key,
                            "query_parent_semantic": domain.name,
                            "assembly_parent_semantic": (
                                part.assembly_parent
                                or part.semantic_parent
                                or domain.name
                            ),
                            "assembly_parent_candidate_key": root_candidate_key,
                            "hierarchy_depth": 1,
                            "profile_refinement": True,
                            "selected_part_profile": selected_profile,
                            "detector_semantic_name": original_part.semantic_name,
                            "profile_semantic_reassignment": (
                                original_part.semantic_name != part.semantic_name
                            ),
                            "profile_semantic_reassignment_reason": (
                                "unique_profile_geometry_role"
                                if original_part.semantic_name != part.semantic_name
                                else None
                            ),
                            "sam_quality": float(segmentation.quality),
                            "root_containment": float(containment),
                            "root_area_fraction": float(fraction),
                            "maximum_instances": part.maximum_instances,
                            "box_xyxy_local": list(detection.box_xyxy),
                            "detector_box_envelope_applied": True,
                            "detector_box_padding_ratio": (
                                self.config.part_box_padding_ratio
                            ),
                            "sam_unconstrained_area_px": unconstrained_area,
                            "sam_box_constrained_area_px": int(
                                np.count_nonzero(constrained_local_mask)
                            ),
                            "crop_xyxy": list(crop_box),
                            "ground_truth_used": False,
                        },
                    )
                )

            dense_rows: list[dict[str, object]] = []
            dense_rejections = 0
            accepted = provisional
            if self.dense_proposer is not None and provisional:
                queries = [
                    (
                        str(index),
                        next(
                            part.prompts[0]
                            for part in selected_parts
                            if part.semantic_name == candidate.semantic_name
                        ),
                        candidate.mask,
                    )
                    for index, candidate in enumerate(provisional)
                ]
                dense_scores = self.dense_proposer.score_regions(
                    image, queries, top_fraction=0.12
                )
                accepted = []
                for index, candidate in enumerate(provisional):
                    row = dense_scores[str(index)]
                    top_mean = float(row["top_mean"])
                    contrast = float(row["contrast"])
                    passed = (
                        top_mean >= self.config.guided_dense_minimum_top_mean
                        and contrast >= self.config.guided_dense_minimum_contrast
                    )
                    dense_rows.append(
                        {
                            "semantic_name": candidate.semantic_name,
                            "prompt": str(row["prompt"]),
                            "top_mean": top_mean,
                            "contrast": contrast,
                            "accepted": passed,
                        }
                    )
                    if not passed:
                        dense_rejections += 1
                        continue
                    accepted.append(
                        replace(
                            candidate,
                            metadata={
                                **candidate.metadata,
                                "profile_dense_top_mean": top_mean,
                                "profile_dense_contrast": contrast,
                                "profile_dense_gate": True,
                            },
                        )
                    )
            dense_supplement_candidates: list[MaskCandidate] = []
            dense_supplement_rows: list[dict[str, object]] = []
            dense_supplement_diagnostics: dict[str, object] | None = None
            if self.dense_proposer is not None:
                accepted_counts: dict[str, int] = {}
                for candidate in accepted:
                    accepted_counts[candidate.semantic_name] = (
                        accepted_counts.get(candidate.semantic_name, 0) + 1
                    )
                dense_parts = [
                    part
                    for part in selected_parts
                    if part.dense_fallback
                    and not part.semantic_name.endswith("_body")
                    and part.topology_relation is None
                    and accepted_counts.get(part.semantic_name, 0)
                    < part.maximum_instances
                ]
                maximum_by_semantic = {
                    part.semantic_name: (
                        part.maximum_instances
                        - accepted_counts.get(part.semantic_name, 0)
                    )
                    for part in dense_parts
                }
                if dense_parts:
                    local_allowed = root_envelope[y0:y1, x0:x1]
                    dense_regions, proposal_diagnostics = self.dense_proposer.propose(
                        crop,
                        dense_parts,
                        local_allowed,
                        maximum_by_semantic,
                        minimum_peak_probability=(
                            self.config.dense_minimum_peak_probability
                        ),
                        minimum_peak_contrast=(self.config.dense_minimum_peak_contrast),
                        activation_quantile=self.config.dense_activation_quantile,
                        peak_ratio=self.config.dense_peak_ratio,
                        box_padding_ratio=self.config.dense_box_padding_ratio,
                    )
                    part_lookup = {part.semantic_name: part for part in dense_parts}
                    dense_regions.sort(
                        key=lambda item: (
                            item.score * part_lookup[item.semantic_name].priority
                        ),
                        reverse=True,
                    )
                    remaining = max(
                        0,
                        self.config.maximum_hierarchy_candidates_per_root
                        - len(accepted),
                    )
                    dense_regions = dense_regions[:remaining]
                    dense_segmentations = self._segment_boxes(
                        crop,
                        [
                            Detection(region.prompt, region.score, region.box_xyxy)
                            for region in dense_regions
                        ],
                    )
                    for region, segmentation in zip(
                        dense_regions,
                        dense_segmentations,
                        strict=True,
                    ):
                        part = part_lookup[region.semantic_name]
                        rejection_reasons: list[str] = []
                        if segmentation.quality < self.config.dense_minimum_sam_quality:
                            rejection_reasons.append("minimum_sam_quality")
                        local_region = np.zeros((crop.height, crop.width), dtype=bool)
                        rx0, ry0, rx1, ry1 = region.box_xyxy
                        local_region[ry0:ry1, rx0:rx1] = True
                        local_mask = segmentation.mask & local_region & local_allowed
                        full_mask = np.zeros((image.height, image.width), dtype=bool)
                        full_mask[y0:y1, x0:x1] = local_mask
                        area = int(np.count_nonzero(full_mask))
                        fraction = area / root_area
                        containment = int(
                            np.count_nonzero(full_mask & root.mask)
                        ) / max(1, area)
                        minimum_containment = (
                            part.minimum_parent_containment
                            if part.minimum_parent_containment is not None
                            else max(0.48, self.config.minimum_parent_containment)
                        )
                        maximum_parent_fraction = (
                            _effective_maximum_parent_fraction(part, self.config)
                        )
                        if area < 12:
                            rejection_reasons.append("minimum_area")
                        if fraction < part.minimum_parent_fraction:
                            rejection_reasons.append("minimum_parent_fraction")
                        if fraction > maximum_parent_fraction:
                            rejection_reasons.append("maximum_parent_fraction")
                        if containment < minimum_containment:
                            rejection_reasons.append("minimum_parent_containment")
                        dense_supplement_rows.append(
                            {
                                "semantic_name": part.semantic_name,
                                "prompt": region.prompt,
                                "score": float(region.score),
                                "peak_contrast": float(region.peak_contrast),
                                "box_xyxy_local": list(region.box_xyxy),
                                "area_px": area,
                                "root_area_fraction": float(fraction),
                                "maximum_parent_fraction": float(
                                    maximum_parent_fraction
                                ),
                                "root_containment": float(containment),
                                "sam_quality": float(segmentation.quality),
                                "accepted": not rejection_reasons,
                                "reasons": rejection_reasons,
                            }
                        )
                        if rejection_reasons:
                            continue
                        semantic_ordinals[part.semantic_name] = (
                            semantic_ordinals.get(part.semantic_name, 0) + 1
                        )
                        candidate_key = (
                            f"{root_candidate_key}/profile-dense:"
                            f"{part.semantic_name}:"
                            f"{semantic_ordinals[part.semantic_name]:02d}"
                        )
                        dense_supplement_candidates.append(
                            MaskCandidate(
                                semantic_name=part.semantic_name,
                                semantic_parent=(part.semantic_parent or domain.name),
                                mask=full_mask,
                                score=region.score,
                                source=self._dense_source("profile-dense-supplement"),
                                prompt=region.prompt,
                                source_reliability=(
                                    self.config.dense_source_reliability
                                    * (0.40 + 0.60 * region.score)
                                    * (0.55 + 0.45 * segmentation.quality)
                                    * part.priority
                                ),
                                metadata={
                                    "source_family": (
                                        f"{self._root_origin()}/profile-dense"
                                    ),
                                    "root_origin": root.metadata.get("root_origin"),
                                    "root_index": root.metadata.get("root_index"),
                                    "candidate_key": candidate_key,
                                    "parent_candidate_key": root_candidate_key,
                                    "query_parent_semantic": domain.name,
                                    "assembly_parent_semantic": (
                                        part.assembly_parent
                                        or part.semantic_parent
                                        or domain.name
                                    ),
                                    "assembly_parent_candidate_key": (
                                        root_candidate_key
                                    ),
                                    "hierarchy_depth": 1,
                                    "profile_refinement": True,
                                    "profile_dense_supplement": True,
                                    "selected_part_profile": selected_profile,
                                    "dense_score": float(region.score),
                                    "dense_peak_contrast": float(region.peak_contrast),
                                    "sam_quality": float(segmentation.quality),
                                    "root_containment": float(containment),
                                    "root_area_fraction": float(fraction),
                                    "maximum_instances": part.maximum_instances,
                                    "box_xyxy_local": list(region.box_xyxy),
                                    "crop_xyxy": list(crop_box),
                                    "ground_truth_used": False,
                                },
                            )
                        )
                    dense_supplement_diagnostics = {
                        "queried_semantics": [
                            part.semantic_name for part in dense_parts
                        ],
                        "proposed_region_count": len(dense_regions),
                        "accepted_candidate_count": len(dense_supplement_candidates),
                        "proposal_diagnostics": {
                            "queried_semantics": list(
                                proposal_diagnostics.queried_semantics
                            ),
                            "proposed_region_count": (
                                proposal_diagnostics.proposed_region_count
                            ),
                            "rejected_low_contrast_count": (
                                proposal_diagnostics.rejected_low_contrast_count
                            ),
                        },
                        "rows": dense_supplement_rows,
                    }
            accepted.extend(dense_supplement_candidates)
            accepted, ambiguous_rejections = _remove_ambiguous_guided_candidates(
                accepted
            )
            topology_candidates: list[MaskCandidate] = []
            topology_rows: list[dict[str, object]] = []
            topology_parts = [
                part for part in selected_parts if part.topology_relation is not None
            ]
            for part in topology_parts:
                anchors = [
                    candidate
                    for candidate in accepted
                    if candidate.semantic_name == part.topology_anchor
                ][: part.maximum_instances]
                anchor_definition = next(
                    (
                        candidate_part
                        for candidate_part in selected_parts
                        if candidate_part.semantic_name == part.topology_anchor
                    ),
                    None,
                )
                derived_for_part: list[MaskCandidate] = []
                for anchor_index, anchor in enumerate(anchors, start=1):
                    topology_diagnostics: dict[str, float | int]
                    if part.topology_relation == "surround":
                        mask, radius = _surround_mask(
                            root.mask,
                            anchor.mask,
                            part.topology_scale,
                        )
                        topology_diagnostics = {"radius_px": int(radius)}
                    elif part.topology_relation == "terminal_complement":
                        mask, topology_diagnostics = _terminal_complement_mask(
                            root.mask,
                            anchor.mask,
                            anchor_position=(
                                anchor_definition.axis_position
                                if anchor_definition is not None
                                else None
                            ),
                            target_position=part.axis_position,
                        )
                        radius = 0
                    else:
                        continue
                    area = int(np.count_nonzero(mask))
                    fraction = area / root_area
                    if not (
                        area >= 12
                        and part.minimum_parent_fraction
                        <= fraction
                        <= part.maximum_parent_fraction
                    ):
                        continue
                    candidate_key = (
                        f"{root_candidate_key}/profile-topology:{grounding_model_key}:"
                        f"{part.semantic_name}:{anchor_index:02d}"
                    )
                    derived_for_part.append(
                        MaskCandidate(
                            semantic_name=part.semantic_name,
                            semantic_parent=part.semantic_parent or domain.name,
                            mask=mask,
                            score=float(np.clip(anchor.score * 0.94, 0.0, 1.0)),
                            source=(f"hpid-topology-v2/{part.topology_relation}"),
                            prompt=part.prompts[0],
                            source_reliability=0.82 * part.priority,
                            metadata={
                                "source_family": "hpid-topology-v1",
                                "root_origin": root.metadata.get("root_origin"),
                                "root_index": root.metadata.get("root_index"),
                                "candidate_key": candidate_key,
                                "parent_candidate_key": root_candidate_key,
                                "query_parent_semantic": domain.name,
                                "assembly_parent_semantic": (
                                    part.assembly_parent
                                    or part.semantic_parent
                                    or domain.name
                                ),
                                "assembly_parent_candidate_key": root_candidate_key,
                                "hierarchy_depth": 1,
                                "profile_refinement": True,
                                "topology_refinement": True,
                                "topology_relation": part.topology_relation,
                                "topology_anchor": part.topology_anchor,
                                "topology_anchor_candidate_key": (
                                    anchor.metadata.get("candidate_key")
                                ),
                                "topology_radius_px": radius,
                                "topology_scale": part.topology_scale,
                                "topology_diagnostics": topology_diagnostics,
                                "maximum_instances": part.maximum_instances,
                                "selected_part_profile": selected_profile,
                                "root_area_fraction": float(fraction),
                                "ground_truth_used": False,
                            },
                        )
                    )
                semantic_rejections = 0
                if self.dense_proposer is not None and derived_for_part:
                    queries = [
                        (
                            str(index),
                            part.prompts[0],
                            candidate.mask,
                        )
                        for index, candidate in enumerate(derived_for_part)
                    ]
                    dense_scores = self.dense_proposer.score_regions(
                        image, queries, top_fraction=0.12
                    )
                    gated: list[MaskCandidate] = []
                    for index, candidate in enumerate(derived_for_part):
                        row = dense_scores[str(index)]
                        top_mean = float(row["top_mean"])
                        contrast = float(row["contrast"])
                        passed = (
                            top_mean >= self.config.guided_dense_minimum_top_mean
                            and contrast
                            >= self.config.guided_dense_minimum_contrast * 0.65
                        )
                        if not passed:
                            semantic_rejections += 1
                            continue
                        gated.append(
                            replace(
                                candidate,
                                metadata={
                                    **candidate.metadata,
                                    "topology_dense_top_mean": top_mean,
                                    "topology_dense_contrast": contrast,
                                    "topology_dense_gate": True,
                                },
                            )
                        )
                    derived_for_part = gated
                topology_candidates.extend(derived_for_part)
                topology_rows.append(
                    {
                        "semantic_name": part.semantic_name,
                        "relation": part.topology_relation,
                        "anchor_semantic": part.topology_anchor,
                        "anchor_candidate_count": len(anchors),
                        "accepted_candidate_count": len(derived_for_part),
                        "semantic_gate_rejection_count": semantic_rejections,
                    }
                )
            accepted.extend(topology_candidates)
            candidates.extend(accepted)
            root_rows.append(
                {
                    "root_index": root.metadata.get("root_index"),
                    "root_semantic": root.semantic_name,
                    "selected_profile": selected_profile,
                    "queried_semantic_count": len(detected_by_semantic),
                    "detected_by_semantic": detected_by_semantic,
                    "provisional_candidate_count": len(provisional),
                    "accepted_candidate_count": len(accepted),
                    "geometry_rejection_count": geometry_rejections,
                    "geometry_rejection_rows": geometry_rejection_rows,
                    "profile_semantic_reassignment_count": len(
                        semantic_reassignment_rows
                    ),
                    "profile_semantic_reassignment_rows": (semantic_reassignment_rows),
                    "dense_semantic_gate_rejection_count": dense_rejections,
                    "ambiguous_semantic_mask_rejection_count": ambiguous_rejections,
                    "dense_semantic_gate_rows": dense_rows,
                    "dense_profile_supplement": dense_supplement_diagnostics,
                    "topology_candidate_count": len(topology_candidates),
                    "topology_rows": topology_rows,
                    "profile": profile_diagnostics,
                }
            )
        return CandidateGeneration(
            tuple(candidates),
            {
                "algorithm": "hpid-routed-profile-refinement-v1",
                "grounding_model": grounding_model,
                "root_count": len(roots),
                "candidate_count": len(candidates),
                "roots": root_rows,
                "ground_truth_used": False,
            },
        )

    def _ground(self, image: Image.Image, phrases: list[str]) -> list[Detection]:
        if not getattr(self, "_grounding_model_active", True):
            self.activate_grounding_model()
        prompt = _prompt_text(phrases)
        inputs = self.grounding_processor(
            images=image, text=prompt, return_tensors="pt"
        ).to(self.device)
        with (
            torch.inference_mode(),
            torch.amp.autocast("cuda", enabled=self.device.startswith("cuda")),
        ):
            outputs = self.grounding_model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        kwargs = {
            "text_threshold": self.config.text_threshold,
            "target_sizes": target_sizes,
        }
        try:
            result = self.grounding_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.config.box_threshold,
                **kwargs,
            )[0]
        except TypeError:
            result = self.grounding_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=self.config.box_threshold,
                **kwargs,
            )[0]
        raw_labels = result.get("text_labels")
        if raw_labels is None:
            raise RuntimeError(
                "The installed Grounding DINO processor did not return text_labels"
            )
        boxes = result["boxes"]
        scores = result["scores"]
        aligned_count = min(len(boxes), len(scores), len(raw_labels))
        self._grounding_output_mismatch_count += (
            max(len(boxes), len(scores), len(raw_labels)) - aligned_count
        )
        detections: list[Detection] = []
        for index in range(aligned_count):
            box = boxes[index]
            score = scores[index]
            label = raw_labels[index]
            x0, y0, x1, y1 = (round(float(value)) for value in box.tolist())
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(image.width, x1), min(image.height, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            detections.append(Detection(str(label), float(score), (x0, y0, x1, y1)))
        return detections

    def _segment_boxes(
        self,
        image: Image.Image,
        detections: list[Detection],
        *,
        semantic_prompts: list[str] | None = None,
        semantic_domain_name: str | None = None,
    ) -> list[SegmentProposal]:
        if not detections:
            return []
        if semantic_prompts is not None and len(semantic_prompts) != len(detections):
            raise ValueError("semantic prompts must align with detections")
        if len(detections) > self.config.segmentation_batch_size:
            output: list[SegmentProposal] = []
            for start in range(0, len(detections), self.config.segmentation_batch_size):
                output.extend(
                    self._segment_boxes(
                        image,
                        detections[start : start + self.config.segmentation_batch_size],
                        semantic_prompts=(
                            semantic_prompts[
                                start : start + self.config.segmentation_batch_size
                            ]
                            if semantic_prompts is not None
                            else None
                        ),
                        semantic_domain_name=semantic_domain_name,
                    )
                )
            return output
        boxes = [[float(value) for value in item.box_xyxy] for item in detections]
        inputs = self.sam_processor(
            images=image, input_boxes=[boxes], return_tensors="pt"
        ).to(self.device)
        with (
            torch.inference_mode(),
            torch.amp.autocast("cuda", enabled=self.device.startswith("cuda")),
        ):
            outputs = self.sam_model(**inputs)
        try:
            processed = self.sam_processor.post_process_masks(
                outputs.pred_masks.detach().cpu(),
                inputs["original_sizes"].detach().cpu(),
                inputs["reshaped_input_sizes"].detach().cpu(),
                binarize=True,
            )[0]
        except (KeyError, TypeError):
            processed = self.sam_processor.post_process_masks(
                outputs.pred_masks.detach().cpu(),
                inputs["original_sizes"].detach().cpu(),
                binarize=True,
            )[0]
        masks = np.asarray(processed)
        if masks.ndim == 3:
            masks = masks[:, None]
        if masks.ndim != 4:
            raise RuntimeError(f"unexpected SAM2 mask shape: {masks.shape}")
        scores = getattr(outputs, "iou_scores", None)
        if scores is None:
            scores = getattr(outputs, "pred_iou_scores", None)
        if scores is None:
            score_array = np.full(masks.shape[:2], 0.5, dtype=np.float32)
        else:
            score_array = scores.detach().float().cpu().numpy()
            while score_array.ndim > 2:
                score_array = score_array[0]
            if score_array.ndim == 1:
                score_array = score_array[None, :]
        mask_areas = np.count_nonzero(masks, axis=(-2, -1))
        best_indices = np.zeros(len(detections), dtype=np.int64)
        for detection_index in range(len(detections)):
            available = np.flatnonzero(mask_areas[detection_index] > 0)
            best_indices[detection_index] = int(
                available[np.argmax(score_array[detection_index, available])]
                if len(available)
                else np.argmax(score_array[detection_index])
            )
        selection_rows: list[dict[str, object]] = [
            {
                "algorithm": "sam-predicted-iou-argmax",
                "baseline_index": int(best_indices[index]),
                "selected_index": int(best_indices[index]),
                "selection_changed": False,
                "ground_truth_used": False,
            }
            for index in range(len(detections))
        ]
        semantic_selection_enabled = bool(
            semantic_prompts is not None
            and getattr(self.config, "use_semantic_root_multimask_selection", True)
            and getattr(self, "dense_proposer", None) is not None
            and masks.shape[1] > 1
        )
        if semantic_selection_enabled:
            competitor_labels = [
                (
                    f"competitor_{domain.name}",
                    domain.classifier_prompt or domain.name.replace("_", " "),
                )
                for domain in self.prompt_bank.domains
                if domain.name != semantic_domain_name
            ]
            competitor_labels.append(
                (
                    "competitor_background",
                    "background ground floor sky wall scenery or support surface",
                )
            )
            assert semantic_prompts is not None
            for detection_index, semantic_prompt in enumerate(semantic_prompts):
                available_indices = [
                    mask_index
                    for mask_index in range(masks.shape[1])
                    if mask_areas[detection_index, mask_index] > 0
                ]
                region_rows = [
                    (
                        f"mask_{mask_index}",
                        masks[detection_index, mask_index].astype(bool),
                    )
                    for mask_index in available_indices
                ]
                if len(region_rows) < 2:
                    continue
                labels = [("target", semantic_prompt), *competitor_labels]
                try:
                    rankings = self.dense_proposer.rank_regions_labels(
                        image,
                        region_rows,
                        labels,
                        masked_weight=0.90,
                    )
                except (RuntimeError, ValueError, OSError, TypeError, KeyError):
                    continue
                target_rows: list[dict[str, float | int]] = []
                missing_ranking = False
                for mask_index in available_indices:
                    row = rankings.get(f"mask_{mask_index}", {}).get("target")
                    if not isinstance(row, dict):
                        missing_ranking = True
                        break
                    target_rows.append(
                        {
                            "probability": float(row.get("probability", 0.0)),
                            "combined_similarity": float(
                                row.get("combined_similarity", 0.0)
                            ),
                            "rank": int(row.get("rank", 999)),
                        }
                    )
                if missing_ranking:
                    continue
                areas = np.asarray(
                    [
                        mask_areas[detection_index, mask_index]
                        for mask_index in available_indices
                    ],
                    dtype=np.int64,
                )
                selected_local, diagnostics = _select_semantic_multimask_index(
                    score_array[detection_index, available_indices],
                    areas,
                    target_rows,
                    quality_weight=(self.config.semantic_multimask_quality_weight),
                    minimum_target_probability=(
                        self.config.semantic_multimask_minimum_target_probability
                    ),
                    minimum_area_ratio=(
                        self.config.semantic_multimask_minimum_area_ratio
                    ),
                    maximum_quality_drop=(
                        self.config.semantic_multimask_maximum_quality_drop
                    ),
                    minimum_score_gain=(
                        self.config.semantic_multimask_minimum_score_gain
                    ),
                )
                selected = available_indices[selected_local]
                baseline_local = int(diagnostics["baseline_index"])
                diagnostics["baseline_index"] = available_indices[baseline_local]
                diagnostics["selected_index"] = selected
                best_indices[detection_index] = selected
                selection_rows[detection_index] = {
                    **diagnostics,
                    "semantic_prompt": semantic_prompt,
                    "semantic_domain_name": semantic_domain_name,
                    "detection_label": detections[detection_index].label,
                }
        return [
            SegmentProposal(
                masks[index, int(best_indices[index])].astype(bool),
                float(
                    np.clip(
                        score_array[index, int(best_indices[index])],
                        0.0,
                        1.0,
                    )
                ),
                selection_rows[index],
            )
            for index in range(len(detections))
        ]

    def _segment_part_boxes(
        self,
        image: Image.Image,
        detections: list[Detection],
        *,
        semantic_prompts: list[str],
        semantic_domain_name: str,
    ) -> list[SegmentProposal]:
        if not self.config.use_semantic_part_multimask_selection:
            return self._segment_boxes(image, detections)
        return self._segment_boxes(
            image,
            detections,
            semantic_prompts=semantic_prompts,
            semantic_domain_name=semantic_domain_name,
        )

    def _segment_root_boxes(
        self,
        image: Image.Image,
        detections: list[Detection],
        *,
        semantic_prompts: list[str],
        semantic_domain_name: str,
    ) -> list[SegmentProposal]:
        if not (
            getattr(self.config, "use_semantic_root_multimask_selection", True)
            and getattr(self, "dense_proposer", None) is not None
            and hasattr(self, "sam_model")
        ):
            return self._segment_boxes(image, detections)
        return self._segment_boxes(
            image,
            detections,
            semantic_prompts=semantic_prompts,
            semantic_domain_name=semantic_domain_name,
        )

    def _root_candidates(self, image: Image.Image) -> list[RootProposal]:
        prompted: list[RootProposal] = []
        prompted_domain: str | None = self.config.asset_prompt_domain.strip() or None
        prompted_profile: str | None = self.config.asset_prompt_profile.strip() or None
        asset_prompt = self.config.asset_prompt.strip()
        if asset_prompt:
            prompted = self._asset_prompt_root_candidates(image, asset_prompt)
            if prompted:
                prompted_domain = prompted[0].domain.name
                prompted_profile = prompted[0].profile_hint
        automatic = (
            self._automatic_asset_root_candidates(image)
            if not asset_prompt and self.config.automatic_asset_queries
            else []
        )
        roots: list[RootProposal] = [*prompted, *automatic]
        for domain in self.prompt_bank.domains:
            if prompted_domain is not None and domain.name != prompted_domain:
                continue
            raw_domain_detections = self._ground(image, list(domain.root_prompts))
            if self.config.use_scene_profile_root_queries:
                for profile in domain.part_profiles:
                    for query_group in profile.scene_root_query_groups:
                        raw_domain_detections.extend(
                            self._ground(image, list(query_group))
                        )
            domain_detections = _select_root_detections(
                raw_domain_detections,
                self.config.root_nms_iou,
                self.config.maximum_roots_per_domain,
            )
            for detection, segmentation in zip(
                domain_detections,
                self._segment_root_boxes(
                    image,
                    domain_detections,
                    semantic_prompts=[item.label for item in domain_detections],
                    semantic_domain_name=domain.name,
                ),
                strict=True,
            ):
                if np.count_nonzero(segmentation.mask) >= 20:
                    prompt_label_score = (
                        PartProfile._match_phrases(
                            detection.label,
                            tuple(
                                value
                                for value in (
                                    asset_prompt,
                                    self.config.asset_prompt_label.strip(),
                                )
                                if value
                            ),
                        )
                        if asset_prompt and prompted_domain is not None
                        else 0.0
                    )
                    prompt_support = bool(prompt_label_score >= 0.50)
                    if self.config.asset_prompt_domain.strip() and not prompt_support:
                        continue
                    roots.append(
                        RootProposal(
                            domain=domain,
                            detection=detection,
                            mask=segmentation.mask,
                            sam_quality=segmentation.quality,
                            query_mode=(
                                "user_asset_prompt_support"
                                if prompt_support
                                else "domain_inventory"
                            ),
                            profile_hint=(prompted_profile if prompt_support else None),
                            model_label=detection.label,
                            sam_selection=segmentation.metadata,
                        )
                    )

        if asset_prompt and self._asset_prompt_diagnostics is not None:
            support_count = sum(
                root.query_mode == "user_asset_prompt_support" for root in roots
            )
            self._asset_prompt_diagnostics["inventory_support_candidate_count"] = (
                support_count
            )
            if not prompted and support_count:
                self._asset_prompt_diagnostics["status"] = (
                    "resolved_with_inventory_support"
                )

        if prompted:
            prompt_keys = {
                (
                    root.detection.label,
                    root.detection.box_xyxy,
                    root.query_mode,
                )
                for root in prompted
            }
            compatible: list[RootProposal] = []
            for root in roots:
                root_key = (
                    root.detection.label,
                    root.detection.box_xyxy,
                    root.query_mode,
                )
                if root_key in prompt_keys or any(
                    mask_iou(root.mask, target.mask) >= 0.05
                    or _mask_containment(root.mask, target.mask) >= 0.55
                    for target in prompted
                ):
                    compatible.append(root)
            roots = compatible
            if self._asset_prompt_diagnostics is not None:
                self._asset_prompt_diagnostics["geometry_support_candidate_count"] = (
                    max(0, len(roots) - len(prompted))
                )
                self._asset_prompt_diagnostics[
                    "geometry_support_requires_prompt_overlap"
                ] = True

        roots.sort(
            key=lambda root: (
                root.detection.score
                + 0.10
                * (
                    1.0
                    if root.profile_hint is not None
                    else root.domain.profile_specificity(root.detection.label)
                ),
                root.detection.score,
            ),
            reverse=True,
        )
        kept: list[RootProposal] = []
        for root in roots:
            if any(
                existing.domain.name == root.domain.name
                and mask_iou(existing.mask, root.mask) >= self.config.root_nms_iou
                for existing in kept
            ):
                continue
            kept.append(root)
            if len(kept) >= self.config.maximum_total_roots:
                break
        return kept

    def _automatic_asset_root_candidates(
        self, image: Image.Image
    ) -> list[RootProposal]:
        """Detect full-image router proposals as isolated object hypotheses."""

        domains = {domain.name: domain for domain in self.prompt_bank.domains}
        roots: list[RootProposal] = []
        query_rows: list[dict[str, object]] = []
        for query in self.config.automatic_asset_queries:
            domain = domains.get(query.domain)
            if domain is None:
                query_rows.append(
                    {
                        "label": query.label,
                        "domain": query.domain,
                        "profile": query.profile,
                        "proposal_score": query.score,
                        "status": "skipped_unknown_domain",
                        "detection_count": 0,
                        "accepted_root_count": 0,
                    }
                )
                continue
            profile = next(
                (
                    profile
                    for profile in domain.part_profiles
                    if profile.name == query.profile
                ),
                None,
            )
            profile_hint = profile.name if profile is not None else None
            query_phrases = list(
                dict.fromkeys(
                    [
                        query.label,
                        *(profile.query_hints(query.label) if profile else ()),
                    ]
                )
            )[:4]
            raw_detections: list[Detection] = []
            phrase_rows: list[dict[str, object]] = []
            for phrase in query_phrases:
                queried = self._ground(image, [phrase])
                raw_detections.extend(queried)
                phrase_rows.append({"phrase": phrase, "detection_count": len(queried)})
            detections = _select_root_detections(
                raw_detections,
                self.config.root_nms_iou,
                min(2, self.config.maximum_roots_per_domain),
            )
            segmentations = self._segment_root_boxes(
                image,
                detections,
                semantic_prompts=[query.label] * len(detections),
                semantic_domain_name=domain.name,
            )
            accepted_count = 0
            for detection, segmentation in zip(detections, segmentations, strict=True):
                if np.count_nonzero(segmentation.mask) < 20:
                    continue
                roots.append(
                    RootProposal(
                        domain=domain,
                        detection=detection,
                        mask=segmentation.mask,
                        sam_quality=segmentation.quality,
                        query_mode="global_asset_proposal",
                        profile_hint=profile_hint,
                        model_label=query.label,
                        sam_selection=segmentation.metadata,
                        automatic_proposal_score=query.score,
                        automatic_proposal_rank=query.rank,
                        automatic_proposal_accepted=query.accepted,
                    )
                )
                accepted_count += 1
            query_rows.append(
                {
                    "label": query.label,
                    "domain": query.domain,
                    "profile": profile_hint,
                    "proposal_score": query.score,
                    "proposal_rank": query.rank,
                    "proposal_accepted": query.accepted,
                    "negative_labels": list(query.negative_labels),
                    "status": "queried",
                    "phrases": phrase_rows,
                    "raw_detection_count": len(raw_detections),
                    "detection_count": len(detections),
                    "accepted_root_count": accepted_count,
                }
            )
        self._automatic_asset_diagnostics = {
            "algorithm": "hpid-global-asset-proposal-grounding-v1",
            "query_count": len(self.config.automatic_asset_queries),
            "accepted_root_count": len(roots),
            "queries": query_rows,
            "ground_truth_used": False,
        }
        return roots

    def _asset_prompt_root_candidates(
        self, image: Image.Image, asset_prompt: str
    ) -> list[RootProposal]:
        configured_domain = self.config.asset_prompt_domain.strip()
        configured_profile = self.config.asset_prompt_profile.strip() or None
        configured_label = self.config.asset_prompt_label.strip() or None
        configured_route = bool(configured_domain)
        if configured_route:
            domain = next(
                (
                    item
                    for item in self.prompt_bank.domains
                    if item.name == configured_domain
                ),
                None,
            )
            if domain is None:
                raise ValueError(
                    f"asset prompt resolved to unavailable domain {configured_domain!r}"
                )
            if configured_profile is not None and configured_profile not in {
                profile.name for profile in domain.part_profiles
            }:
                raise ValueError(
                    "asset prompt resolved to unavailable profile "
                    f"{configured_profile!r} in {configured_domain!r}"
                )
            profile_hint = configured_profile
            best_score = 1.0
            best_profile_score = 1.0 if configured_profile is not None else 0.0
            runner_up = 0.0
            domain_forced = True
            resolved = True
        else:
            scored_domains: list[tuple[float, float, DomainPrompt, str | None]] = []
            for domain in self.prompt_bank.domains:
                profile_scores = [
                    (profile.match_score(asset_prompt), profile.name)
                    for profile in domain.part_profiles
                ]
                best_profile_score, best_profile = max(
                    profile_scores,
                    default=(0.0, None),
                    key=lambda item: item[0],
                )
                domain_score = domain.root_label_specificity(asset_prompt)
                scored_domains.append(
                    (
                        max(domain_score, best_profile_score),
                        best_profile_score,
                        domain,
                        best_profile if best_profile_score >= 0.50 else None,
                    )
                )
            scored_domains.sort(key=lambda item: (item[0], item[1]), reverse=True)
            best_score, best_profile_score, domain, profile_hint = scored_domains[0]
            runner_up = scored_domains[1][0] if len(scored_domains) > 1 else 0.0
            domain_forced = len(self.prompt_bank.domains) == 1
            resolved = bool(
                domain_forced
                or best_score >= 0.50
                and (best_score - runner_up >= 0.08 or best_score >= 0.86)
            )
        self._asset_prompt_diagnostics = {
            "algorithm": "user-asset-prompt-root-retrieval-v2",
            "asset_prompt": asset_prompt,
            "status": "resolved" if resolved else "automatic_fallback",
            "selected_domain": domain.name if resolved else None,
            "selected_profile": profile_hint if resolved else None,
            "selected_asset_label": configured_label if resolved else None,
            "domain_score": float(best_score),
            "profile_score": float(best_profile_score),
            "domain_margin": float(best_score - runner_up),
            "domain_forced_by_filter": domain_forced,
            "taxonomy_route_applied": configured_route,
            "taxonomy_route_reason": (
                self.config.asset_prompt_resolution_reason or None
            ),
            "ground_truth_used": False,
        }
        if not resolved:
            return []
        query_phrases = [asset_prompt]
        if profile_hint is not None:
            profile = next(
                (item for item in domain.part_profiles if item.name == profile_hint),
                None,
            )
            if profile is not None and profile.match_score(asset_prompt) > 0.0:
                query_phrases.extend(profile.query_hints(asset_prompt))
        unique_queries = list(
            dict.fromkeys(phrase.strip() for phrase in query_phrases if phrase.strip())
        )
        query_rows: list[dict[str, object]] = []
        raw_detections: list[Detection] = []
        for phrase in unique_queries:
            queried = self._ground(image, [phrase])
            raw_detections.extend(queried)
            query_rows.append(
                {
                    "phrase": phrase,
                    "detection_count": len(queried),
                }
            )
        detections = _select_root_detections(
            raw_detections,
            self.config.root_nms_iou,
            self.config.maximum_roots_per_domain,
        )
        segmentations = self._segment_root_boxes(
            image,
            detections,
            semantic_prompts=[asset_prompt] * len(detections),
            semantic_domain_name=domain.name,
        )
        roots = [
            RootProposal(
                domain=domain,
                detection=detection,
                mask=segmentation.mask,
                sam_quality=segmentation.quality,
                query_mode="user_asset_prompt",
                profile_hint=profile_hint,
                model_label=detection.label,
                sam_selection=segmentation.metadata,
            )
            for detection, segmentation in zip(detections, segmentations, strict=True)
            if np.count_nonzero(segmentation.mask) >= 20
        ]
        self._asset_prompt_diagnostics["detection_count"] = len(detections)
        self._asset_prompt_diagnostics["raw_detection_count"] = len(raw_detections)
        self._asset_prompt_diagnostics["queries"] = query_rows
        self._asset_prompt_diagnostics["resolved_query_hints"] = unique_queries
        self._asset_prompt_diagnostics["accepted_root_count"] = len(roots)
        if not roots:
            self._asset_prompt_diagnostics["status"] = (
                "resolved_without_direct_detection"
                if configured_route
                else "automatic_fallback"
            )
        return roots

    def _child_candidates(
        self,
        image: Image.Image,
        domain: DomainPrompt,
        root_index: int,
        root_box: tuple[int, int, int, int],
        root_mask: np.ndarray,
        root_label: str = "",
        profile_hint: str | None = None,
    ) -> tuple[list[MaskCandidate], dict[str, object]]:
        selected_parts, selected_profile, profile_diagnostics = domain.select_parts(
            root_label,
            profile_hint=profile_hint,
            profile_hint_source=("isolated_profile_query" if profile_hint else None),
        )
        if not selected_parts:
            return [], {
                "unmapped_detection_count": 0,
                "mapped_detection_count": 0,
                "selected_detection_count": 0,
                "part_profile": profile_diagnostics,
            }
        candidates: list[MaskCandidate] = []
        call_diagnostics: list[dict[str, object]] = []
        remaining_budget = [self.config.maximum_hierarchy_candidates_per_root]
        children_by_parent: dict[str, list[tuple[PartPrompt, bool]]] = {}
        for part in selected_parts:
            parent = part.query_parent or part.semantic_parent or domain.name
            children_by_parent.setdefault(parent, []).append((part, False))
            if (
                part.fallback_query_parent is not None
                and part.fallback_query_parent != parent
            ):
                children_by_parent.setdefault(part.fallback_query_parent, []).append(
                    (part, True)
                )

        def expand(
            parent_semantic: str,
            parent_box: tuple[int, int, int, int],
            parent_mask: np.ndarray,
            parent_candidate_key: str,
            depth: int,
        ) -> None:
            if depth > self.config.maximum_hierarchy_depth or remaining_budget[0] <= 0:
                return
            part_queries = children_by_parent.get(parent_semantic, [])
            if not part_queries:
                return
            parent_area = max(1, int(np.count_nonzero(parent_mask)))
            parts: list[PartPrompt] = []
            fallback_semantics: set[str] = set()
            fallback_skipped = 0
            for part, is_fallback in part_queries:
                if is_fallback:
                    existing = [
                        candidate.mask
                        for candidate in candidates
                        if candidate.semantic_name == part.semantic_name
                        and candidate.metadata.get("root_index") == root_index
                    ]
                    if existing:
                        covered = np.logical_or.reduce(existing)
                        coverage = np.count_nonzero(covered & parent_mask) / parent_area
                    else:
                        coverage = 0.0
                    if coverage >= part.fallback_if_coverage_below:
                        fallback_skipped += 1
                        continue
                    fallback_semantics.add(part.semantic_name)
                parts.append(part)
            if not parts:
                return
            crop_box = _crop_box(parent_box, image.size, self.config.crop_padding)
            crop = image.crop(crop_box)
            mapped: list[tuple[Detection, PartPrompt]] = []
            unmapped = 0
            if self.config.isolate_semantic_prompts:
                for part in parts:
                    phrases = list(dict.fromkeys(part.phrases))
                    mapped.extend(
                        (detection, part) for detection in self._ground(crop, phrases)
                    )
            else:
                phrases = list(
                    dict.fromkeys(phrase for part in parts for phrase in part.phrases)
                )
                for detection in self._ground(crop, phrases):
                    part = domain.match_part(detection.label, parts)
                    if part is None:
                        unmapped += 1
                        continue
                    mapped.append((detection, part))
            per_parent_limit = (
                self.config.maximum_children_per_root
                if depth == 1
                else self.config.maximum_children_per_parent
            )
            selected, selection = _select_child_detections(
                mapped,
                limit=min(per_parent_limit, remaining_budget[0]),
                nms_iou=self.config.child_box_nms_iou,
                use_semantic_quota=self.config.use_semantic_quota,
            )
            masks = self._segment_part_boxes(
                crop,
                [item[0] for item in selected],
                semantic_prompts=[item[1].phrases[0] for item in selected],
                semantic_domain_name=domain.name,
            )
            x0, y0, x1, y1 = crop_box
            provisional: list[tuple[MaskCandidate, PartPrompt]] = []
            semantic_ordinals: dict[str, int] = {}
            for (detection, part), segmentation in zip(selected, masks, strict=True):
                local_mask = segmentation.mask
                full_mask = np.zeros((image.height, image.width), dtype=bool)
                full_mask[y0:y1, x0:x1] = local_mask
                child_area = int(np.count_nonzero(full_mask))
                fraction = child_area / parent_area
                containment = np.count_nonzero(full_mask & parent_mask) / max(
                    1, child_area
                )
                minimum_containment = (
                    part.minimum_parent_containment
                    if part.minimum_parent_containment is not None
                    else self.config.minimum_parent_containment
                )
                is_fallback_query = part.semantic_name in fallback_semantics
                maximum_parent_fraction = (
                    part.fallback_maximum_parent_fraction
                    if is_fallback_query
                    and part.fallback_maximum_parent_fraction is not None
                    else part.maximum_parent_fraction
                )
                if not (
                    part.minimum_parent_fraction <= fraction <= maximum_parent_fraction
                    and containment >= minimum_containment
                ):
                    continue
                semantic_ordinals[part.semantic_name] = (
                    semantic_ordinals.get(part.semantic_name, 0) + 1
                )
                candidate_key = (
                    f"{parent_candidate_key}/{part.semantic_name}:"
                    f"{semantic_ordinals[part.semantic_name]:02d}"
                )
                candidate = MaskCandidate(
                    semantic_name=part.semantic_name,
                    semantic_parent=part.semantic_parent or domain.name,
                    mask=full_mask,
                    score=detection.score,
                    source=self._grounded_source(f"hierarchy-{depth}"),
                    prompt=detection.label,
                    source_reliability=(
                        max(0.78, 0.92 - 0.03 * depth)
                        * (0.55 + 0.45 * segmentation.quality)
                        * part.priority
                    ),
                    metadata={
                        "root_origin": self._root_origin(),
                        "root_index": root_index,
                        "candidate_key": candidate_key,
                        "parent_candidate_key": parent_candidate_key,
                        "query_parent_semantic": parent_semantic,
                        "fallback_query": is_fallback_query,
                        "maximum_parent_fraction_applied": (maximum_parent_fraction),
                        "assembly_parent_semantic": (
                            part.assembly_parent or part.semantic_parent or domain.name
                        ),
                        "assembly_parent_candidate_key": (
                            parent_candidate_key
                            if (
                                part.assembly_parent
                                or part.semantic_parent
                                or domain.name
                            )
                            == parent_semantic
                            else None
                        ),
                        "hierarchy_depth": depth,
                        "sam_quality": segmentation.quality,
                        "box_xyxy_local": list(detection.box_xyxy),
                        "crop_xyxy": list(crop_box),
                        "parent_area_fraction": fraction,
                        "parent_containment": containment,
                        "maximum_instances": part.maximum_instances,
                    },
                )
                provisional.append((candidate, part))

            def spatially_valid(candidate: MaskCandidate, part: PartPrompt) -> bool:
                if part.spatial_anchor is None or part.spatial_relation is None:
                    return True
                anchors = [
                    item
                    for item in [
                        *candidates,
                        *(candidate for candidate, _ in provisional),
                    ]
                    if item.semantic_name == part.spatial_anchor
                    and item.metadata.get("root_index") == root_index
                ]
                if not anchors:
                    return True
                candidate_box = _mask_box(candidate.mask)
                candidate_center = (
                    (candidate_box[0] + candidate_box[2]) / 2.0,
                    (candidate_box[1] + candidate_box[3]) / 2.0,
                )
                parent_width = max(1, parent_box[2] - parent_box[0])
                parent_height = max(1, parent_box[3] - parent_box[1])
                for anchor in anchors:
                    anchor_box = _mask_box(anchor.mask)
                    anchor_center = (
                        (anchor_box[0] + anchor_box[2]) / 2.0,
                        (anchor_box[1] + anchor_box[3]) / 2.0,
                    )
                    if part.spatial_relation == "below" and candidate_center[1] >= (
                        anchor_center[1] + part.spatial_tolerance * parent_height
                    ):
                        return True
                    if part.spatial_relation == "above" and candidate_center[1] <= (
                        anchor_center[1] - part.spatial_tolerance * parent_height
                    ):
                        return True
                    if part.spatial_relation == "right_of" and candidate_center[0] >= (
                        anchor_center[0] + part.spatial_tolerance * parent_width
                    ):
                        return True
                    if part.spatial_relation == "left_of" and candidate_center[0] <= (
                        anchor_center[0] - part.spatial_tolerance * parent_width
                    ):
                        return True
                    if (
                        part.spatial_relation == "overlap"
                        and mask_iou(candidate.mask, anchor.mask)
                        >= part.spatial_tolerance
                    ):
                        return True
                return False

            accepted_here: list[MaskCandidate] = []
            spatial_rejected = 0
            for candidate, part in provisional:
                if not spatially_valid(candidate, part):
                    spatial_rejected += 1
                    continue
                candidates.append(candidate)
                accepted_here.append(candidate)
                remaining_budget[0] -= 1

            dense_queried_semantics: list[str] = []
            dense_proposed_regions = 0
            dense_accepted_masks = 0
            dense_low_contrast_rejections = 0
            dense_low_quality_rejections = 0
            dense_proposer = getattr(self, "dense_proposer", None)
            if dense_proposer is not None and remaining_budget[0] > 0:
                accepted_counts: dict[str, int] = {}
                for candidate in accepted_here:
                    accepted_counts[candidate.semantic_name] = (
                        accepted_counts.get(candidate.semantic_name, 0) + 1
                    )
                dense_parts: list[PartPrompt] = []
                maximum_by_semantic: dict[str, int] = {}
                for part in parts:
                    if self.config.dense_detail_only and not part.detail:
                        continue
                    if self.config.dense_require_opt_in and not part.dense_fallback:
                        continue
                    existing_count = accepted_counts.get(part.semantic_name, 0)
                    maximum = (
                        max(0, part.maximum_instances - existing_count)
                        if self.config.dense_only_missing
                        else part.maximum_instances
                    )
                    if maximum <= 0 or part.semantic_name in maximum_by_semantic:
                        continue
                    dense_parts.append(part)
                    maximum_by_semantic[part.semantic_name] = maximum
                if dense_parts:
                    local_parent = parent_mask[y0:y1, x0:x1]
                    allowed = parent_envelope(
                        local_parent,
                        self.config.dense_parent_dilation_ratio,
                    )
                    dense_regions, dense_diagnostics = dense_proposer.propose(
                        crop,
                        dense_parts,
                        allowed,
                        maximum_by_semantic,
                        minimum_peak_probability=(
                            self.config.dense_minimum_peak_probability
                        ),
                        minimum_peak_contrast=(self.config.dense_minimum_peak_contrast),
                        activation_quantile=self.config.dense_activation_quantile,
                        peak_ratio=self.config.dense_peak_ratio,
                        box_padding_ratio=self.config.dense_box_padding_ratio,
                    )
                    dense_queried_semantics = list(dense_diagnostics.queried_semantics)
                    dense_proposed_regions = dense_diagnostics.proposed_region_count
                    dense_low_contrast_rejections = (
                        dense_diagnostics.rejected_low_contrast_count
                    )
                    part_lookup = {part.semantic_name: part for part in dense_parts}
                    dense_regions.sort(
                        key=lambda item: (
                            item.score * part_lookup[item.semantic_name].priority
                        ),
                        reverse=True,
                    )
                    dense_regions = dense_regions[: remaining_budget[0]]
                    dense_detections = [
                        Detection(
                            region.prompt,
                            region.score,
                            region.box_xyxy,
                        )
                        for region in dense_regions
                    ]
                    dense_segmentations = self._segment_boxes(crop, dense_detections)
                    full_allowed = np.zeros((image.height, image.width), dtype=bool)
                    full_allowed[y0:y1, x0:x1] = allowed
                    for region, segmentation in zip(
                        dense_regions,
                        dense_segmentations,
                        strict=True,
                    ):
                        if segmentation.quality < self.config.dense_minimum_sam_quality:
                            dense_low_quality_rejections += 1
                            continue
                        part = part_lookup[region.semantic_name]
                        full_mask = np.zeros((image.height, image.width), dtype=bool)
                        local_region = np.zeros((crop.height, crop.width), dtype=bool)
                        rx0, ry0, rx1, ry1 = region.box_xyxy
                        local_region[ry0:ry1, rx0:rx1] = True
                        full_mask[y0:y1, x0:x1] = segmentation.mask & local_region
                        full_mask &= full_allowed
                        child_area = int(np.count_nonzero(full_mask))
                        fraction = child_area / parent_area
                        containment = np.count_nonzero(full_mask & full_allowed) / max(
                            1, child_area
                        )
                        minimum_containment = (
                            part.minimum_parent_containment
                            if part.minimum_parent_containment is not None
                            else self.config.minimum_parent_containment
                        )
                        is_fallback_query = part.semantic_name in fallback_semantics
                        maximum_parent_fraction = (
                            part.fallback_maximum_parent_fraction
                            if is_fallback_query
                            and part.fallback_maximum_parent_fraction is not None
                            else part.maximum_parent_fraction
                        )
                        if not (
                            part.minimum_parent_fraction
                            <= fraction
                            <= maximum_parent_fraction
                            and containment >= minimum_containment
                        ):
                            continue
                        semantic_ordinals[part.semantic_name] = (
                            semantic_ordinals.get(part.semantic_name, 0) + 1
                        )
                        candidate_key = (
                            f"{parent_candidate_key}/{part.semantic_name}:"
                            f"{semantic_ordinals[part.semantic_name]:02d}"
                        )
                        candidate = MaskCandidate(
                            semantic_name=part.semantic_name,
                            semantic_parent=part.semantic_parent or domain.name,
                            mask=full_mask,
                            score=region.score,
                            source=self._dense_source(f"dense-hierarchy-{depth}"),
                            prompt=region.prompt,
                            source_reliability=(
                                self.config.dense_source_reliability
                                * (0.40 + 0.60 * region.score)
                                * (0.55 + 0.45 * segmentation.quality)
                                * part.priority
                            ),
                            metadata={
                                "root_origin": self._root_origin(),
                                "root_index": root_index,
                                "candidate_key": candidate_key,
                                "parent_candidate_key": parent_candidate_key,
                                "query_parent_semantic": parent_semantic,
                                "fallback_query": is_fallback_query,
                                "maximum_parent_fraction_applied": (
                                    maximum_parent_fraction
                                ),
                                "dense_semantic_fallback": True,
                                "dense_score": region.score,
                                "dense_peak_contrast": region.peak_contrast,
                                "assembly_parent_semantic": (
                                    part.assembly_parent
                                    or part.semantic_parent
                                    or domain.name
                                ),
                                "assembly_parent_candidate_key": (
                                    parent_candidate_key
                                    if (
                                        part.assembly_parent
                                        or part.semantic_parent
                                        or domain.name
                                    )
                                    == parent_semantic
                                    else None
                                ),
                                "hierarchy_depth": depth,
                                "sam_quality": segmentation.quality,
                                "box_xyxy_local": list(region.box_xyxy),
                                "crop_xyxy": list(crop_box),
                                "parent_area_fraction": fraction,
                                "parent_containment": containment,
                                "maximum_instances": part.maximum_instances,
                                "parent_support_mode": (
                                    "filled_dilated_visible_envelope"
                                ),
                            },
                        )
                        if not spatially_valid(candidate, part):
                            spatial_rejected += 1
                            continue
                        candidates.append(candidate)
                        accepted_here.append(candidate)
                        remaining_budget[0] -= 1
                        dense_accepted_masks += 1
                        if remaining_budget[0] <= 0:
                            break

            call_diagnostics.append(
                {
                    "parent_semantic": parent_semantic,
                    "parent_candidate_key": parent_candidate_key,
                    "hierarchy_depth": depth,
                    "semantic_prompt_isolation": (self.config.isolate_semantic_prompts),
                    **selection,
                    "unmapped_detection_count": unmapped,
                    "spatial_rejected_count": spatial_rejected,
                    "fallback_query_semantics": sorted(fallback_semantics),
                    "fallback_queries_skipped": fallback_skipped,
                    "dense_queried_semantics": dense_queried_semantics,
                    "dense_proposed_region_count": dense_proposed_regions,
                    "dense_accepted_mask_count": dense_accepted_masks,
                    "dense_low_contrast_rejection_count": (
                        dense_low_contrast_rejections
                    ),
                    "dense_low_quality_rejection_count": (dense_low_quality_rejections),
                    "accepted_mask_count": len(accepted_here),
                    "accepted_semantics": sorted(
                        {candidate.semantic_name for candidate in accepted_here}
                    ),
                }
            )
            for candidate in accepted_here:
                if remaining_budget[0] <= 0:
                    break
                expand(
                    candidate.semantic_name,
                    _mask_box(candidate.mask),
                    candidate.mask,
                    str(candidate.metadata["candidate_key"]),
                    depth + 1,
                )

        root_candidate_key = f"root:{root_index}"
        expand(
            domain.name,
            root_box,
            root_mask,
            root_candidate_key,
            1,
        )
        return candidates, {
            "unmapped_detection_count": sum(
                int(item["unmapped_detection_count"]) for item in call_diagnostics
            ),
            "mapped_detection_count": sum(
                int(item["mapped_detection_count"]) for item in call_diagnostics
            ),
            "selected_detection_count": sum(
                int(item["selected_detection_count"]) for item in call_diagnostics
            ),
            "accepted_mask_count": len(candidates),
            "hierarchy_call_count": len(call_diagnostics),
            "remaining_candidate_budget": remaining_budget[0],
            "selected_part_profile": selected_profile,
            "part_profile": profile_diagnostics,
            "hierarchy_calls": call_diagnostics,
        }

    def generate(self, image: Image.Image) -> CandidateGeneration:
        image = image.convert("RGB")
        mismatch_start = self._grounding_output_mismatch_count
        roots = self._root_candidates(image)
        candidates: list[MaskCandidate] = []
        child_diagnostics: list[dict[str, object]] = []
        for root_index, root in enumerate(roots, start=1):
            domain = root.domain
            detection = root.detection
            mask = root.mask
            sam_quality = root.sam_quality
            _, selected_profile, profile_diagnostics = domain.select_parts(
                detection.label,
                profile_hint=root.profile_hint,
                profile_hint_source=(
                    "isolated_profile_query" if root.profile_hint else None
                ),
            )
            accepted_profile_specificity = (
                float(profile_diagnostics.get("best_score", 0.0))
                if selected_profile is not None
                else 0.0
            )
            candidates.append(
                MaskCandidate(
                    semantic_name=domain.name,
                    semantic_parent=domain.name,
                    mask=mask,
                    score=detection.score,
                    source=self._grounded_source("root"),
                    prompt=detection.label,
                    source_reliability=0.80 + 0.20 * sam_quality,
                    metadata={
                        "root_origin": self._root_origin(),
                        "root_index": root_index,
                        "candidate_key": f"root:{root_index}",
                        "parent_candidate_key": None,
                        "sam_quality": sam_quality,
                        "sam_multimask_selection": root.sam_selection,
                        "box_xyxy": list(detection.box_xyxy),
                        "root_label_specificity": domain.root_label_specificity(
                            detection.label
                        ),
                        "part_profile_specificity": accepted_profile_specificity,
                        "selected_part_profile": selected_profile,
                        "part_profile_selection": profile_diagnostics,
                        "root_query_mode": root.query_mode,
                        "root_model_label": root.model_label,
                        "global_asset_proposal_score": (root.automatic_proposal_score),
                        "global_asset_proposal_rank": (root.automatic_proposal_rank),
                        "global_asset_proposal_accepted": (
                            root.automatic_proposal_accepted
                        ),
                        "profile_hint_source": (
                            "user_asset_prompt"
                            if root.query_mode.startswith("user_asset_prompt")
                            else "global_asset_proposal"
                            if root.query_mode == "global_asset_proposal"
                            else "isolated_profile_query"
                            if root.profile_hint is not None
                            else "specific_root_label"
                            if selected_profile is not None
                            else None
                        ),
                        "profile_resolution_status": (
                            "accepted"
                            if root.query_mode.startswith("user_asset_prompt")
                            and selected_profile is not None
                            else None
                        ),
                    },
                )
            )
            if domain.part_profiles:
                children = []
                root_child_diagnostics = {
                    "unmapped_detection_count": 0,
                    "mapped_detection_count": 0,
                    "selected_detection_count": 0,
                    "accepted_mask_count": 0,
                    "hierarchy_call_count": 0,
                    "remaining_candidate_budget": (
                        self.config.maximum_hierarchy_candidates_per_root
                    ),
                    "selected_part_profile": selected_profile,
                    "part_profile": profile_diagnostics,
                    "hierarchy_calls": [],
                    "deferred_until_isolated_profile_resolution": True,
                }
            else:
                children, root_child_diagnostics = self._child_candidates(
                    image,
                    domain,
                    root_index,
                    detection.box_xyxy,
                    mask,
                    detection.label,
                    root.profile_hint,
                )
            candidates.extend(children)
            child_diagnostics.append(
                {
                    "root_index": root_index,
                    "domain": domain.name,
                    **root_child_diagnostics,
                }
            )
        diagnostics: dict[str, object] = {
            "root_count": len(roots),
            "root_proposals": [
                {
                    "root_index": index,
                    "domain": root.domain.name,
                    "query_mode": root.query_mode,
                    "model_label": root.model_label,
                    "detection_label": root.detection.label,
                    "detection_score": root.detection.score,
                    "box_xyxy": list(root.detection.box_xyxy),
                    "mask_area_px": int(np.count_nonzero(root.mask)),
                    "mask_bbox_xyxy": list(_mask_box(root.mask)),
                    "sam_quality": root.sam_quality,
                    "profile_hint": root.profile_hint,
                    "global_asset_proposal_score": root.automatic_proposal_score,
                    "global_asset_proposal_rank": root.automatic_proposal_rank,
                    "global_asset_proposal_accepted": (
                        root.automatic_proposal_accepted
                    ),
                }
                for index, root in enumerate(roots, start=1)
            ],
            "isolated_profile_root_count": sum(
                root.query_mode == "isolated_profile_query" for root in roots
            ),
            "candidate_count": len(candidates),
            "child_generation": child_diagnostics,
            "unmapped_grounding_labels": sum(
                int(item["unmapped_detection_count"]) for item in child_diagnostics
            ),
            "discarded_unpaired_grounding_outputs": (
                self._grounding_output_mismatch_count - mismatch_start
            ),
            "models": asdict(self.config),
            "asset_prompt_resolution": self._asset_prompt_diagnostics,
            "automatic_asset_proposal_grounding": (self._automatic_asset_diagnostics),
            "ground_truth_used": False,
        }
        return CandidateGeneration(tuple(candidates), diagnostics)


def candidate_generator_signature() -> dict[str, Any]:
    """Machine-readable statement used by tests and package manifests."""
    return {
        "inputs": ["image", "prompt_bank", "model_configuration"],
        "forbidden_inputs": ["ground_truth", "target_mask", "reference_labels"],
    }
