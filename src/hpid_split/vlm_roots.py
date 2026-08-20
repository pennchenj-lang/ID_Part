from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .foundation import CandidateGeneration, Detection, SegmentProposal
from .fusion import MaskCandidate
from .vlm_parts import VlmPlanner, _extract_json_object


@dataclass(frozen=True)
class VlmRootConfig:
    minimum_confidence: float = 0.45
    minimum_box_side_px: int = 4
    maximum_instances: int = 4
    box_nms_iou: float = 0.82
    minimum_mask_pixels: int = 20
    source_reliability: float = 0.82


@dataclass(frozen=True)
class PlannedRoot:
    box_xyxy: tuple[int, int, int, int]
    confidence: float
    instance_hint: str | None = None


BoxSegmenter = Callable[
    [Image.Image, list[Detection]],
    list[SegmentProposal],
]


def build_root_localization_prompt(target_label: str) -> str:
    return f"""Locate every visible instance of the requested target object
{target_label!r} in this image. Draw one tight box around the complete visible extent of
each target instance. Include all connected or detached visible components that belong
to that same instance, such as a cable attached to headphones. A box may span an
occluder, but do not label the occluding person, contents, support surface, background,
or a nearby object as part of the target. Do not return a box for a context region that
merely contains the target. If the target is absent or too uncertain, return no boxes.

Return JSON only with normalized integer coordinates from 0 to 1000:
{{"target":"{target_label}","instances":[
{{"bbox_2d":[x1,y1,x2,y2],"confidence":0.0,"visible":true,
"instance_hint":"optional short identifier"}}
]}}
"""


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


def parse_root_plan(
    response: str,
    *,
    image_size: tuple[int, int],
    config: VlmRootConfig | None = None,
) -> tuple[tuple[PlannedRoot, ...], dict[str, object]]:
    config = config or VlmRootConfig()
    payload = _extract_json_object(response)
    diagnostics: dict[str, object] = {
        "response_character_count": len(response),
        "ground_truth_used": False,
    }
    if payload is None:
        return (), {**diagnostics, "status": "invalid_json", "raw_instance_count": 0}
    instances = payload.get("instances")
    if not isinstance(instances, list):
        return (), {
            **diagnostics,
            "status": "missing_instances",
            "raw_instance_count": 0,
        }
    width, height = image_size
    parsed: list[PlannedRoot] = []
    rejection_counts: dict[str, int] = {
        "not_visible": 0,
        "invalid_box": 0,
        "low_confidence": 0,
        "too_small": 0,
        "duplicate": 0,
    }
    for row in instances:
        if not isinstance(row, dict):
            rejection_counts["invalid_box"] += 1
            continue
        if row.get("visible") is False:
            rejection_counts["not_visible"] += 1
            continue
        raw_box = row.get("bbox_2d")
        if not (
            isinstance(raw_box, list)
            and len(raw_box) == 4
            and all(isinstance(value, (int, float)) for value in raw_box)
        ):
            rejection_counts["invalid_box"] += 1
            continue
        try:
            confidence = float(row.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        if not np.isfinite(confidence) or confidence < config.minimum_confidence:
            rejection_counts["low_confidence"] += 1
            continue
        x0 = round(float(raw_box[0]) * width / 1000.0)
        y0 = round(float(raw_box[1]) * height / 1000.0)
        x1 = round(float(raw_box[2]) * width / 1000.0)
        y1 = round(float(raw_box[3]) * height / 1000.0)
        x0, y0 = max(0, min(width, x0)), max(0, min(height, y0))
        x1, y1 = max(0, min(width, x1)), max(0, min(height, y1))
        if x1 <= x0 or y1 <= y0:
            rejection_counts["invalid_box"] += 1
            continue
        if (
            x1 - x0 < config.minimum_box_side_px
            or y1 - y0 < config.minimum_box_side_px
        ):
            rejection_counts["too_small"] += 1
            continue
        planned = PlannedRoot(
            (x0, y0, x1, y1),
            float(np.clip(confidence, 0.0, 1.0)),
            (
                str(row["instance_hint"]).strip()
                if row.get("instance_hint") is not None
                and str(row["instance_hint"]).strip()
                else None
            ),
        )
        if any(
            _box_iou(planned.box_xyxy, accepted.box_xyxy) >= config.box_nms_iou
            for accepted in parsed
        ):
            rejection_counts["duplicate"] += 1
            continue
        parsed.append(planned)
        parsed.sort(key=lambda item: item.confidence, reverse=True)
        parsed = parsed[: config.maximum_instances]
    return tuple(parsed), {
        **diagnostics,
        "status": "accepted" if parsed else "no_accepted_instances",
        "raw_instance_count": len(instances),
        "accepted_instance_count": len(parsed),
        "rejection_counts": rejection_counts,
    }


class VlmRootProposer:
    def __init__(
        self,
        planner: VlmPlanner,
        segment_boxes: BoxSegmenter,
        *,
        config: VlmRootConfig | None = None,
    ) -> None:
        self.planner = planner
        self.segment_boxes = segment_boxes
        self.config = config or VlmRootConfig()

    def generate(
        self,
        image: Image.Image,
        *,
        target_label: str,
        semantic_name: str,
        selected_profile: str | None = None,
    ) -> CandidateGeneration:
        prompt = build_root_localization_prompt(target_label)
        response = self.planner.generate_response(image, prompt)
        planned, parse_diagnostics = parse_root_plan(
            response,
            image_size=image.size,
            config=self.config,
        )
        detections = [
            Detection(target_label, item.confidence, item.box_xyxy) for item in planned
        ]
        segmentations = self.segment_boxes(image, detections) if detections else []
        source_root = f"vlm-sam2[{self.planner.backend_id}]"
        candidates: list[MaskCandidate] = []
        rejected_small_masks = 0
        for index, (plan, segmentation) in enumerate(
            zip(planned, segmentations, strict=True), start=1
        ):
            mask = np.asarray(segmentation.mask, dtype=bool)
            if int(np.count_nonzero(mask)) < self.config.minimum_mask_pixels:
                rejected_small_masks += 1
                continue
            score = float(np.sqrt(plan.confidence * max(0.0, segmentation.quality)))
            metadata: dict[str, object] = {
                "root_origin": source_root,
                "root_index": index,
                "candidate_key": f"vlm-root:{index}",
                "parent_candidate_key": None,
                "root_query_mode": "vlm_target_localization",
                "root_model_label": target_label,
                "root_label_specificity": 1.0,
                "sam_quality": float(segmentation.quality),
                "vlm_root_confidence": plan.confidence,
                "vlm_root_box_xyxy": list(plan.box_xyxy),
                "vlm_instance_hint": plan.instance_hint,
                "vlm_root_localization": True,
                "ground_truth_used": False,
                **(
                    {
                        "selected_part_profile": selected_profile,
                        "part_profile_specificity": 1.0,
                        "profile_hint_source": "vlm_target_localization",
                        "profile_resolution_status": "accepted",
                    }
                    if selected_profile is not None
                    else {}
                ),
            }
            if segmentation.metadata:
                metadata["sam_multimask_selection"] = dict(segmentation.metadata)
            candidates.append(
                MaskCandidate(
                    semantic_name=semantic_name,
                    semantic_parent=semantic_name,
                    mask=mask,
                    score=score,
                    source=f"{source_root}/root",
                    prompt=target_label,
                    source_reliability=self.config.source_reliability,
                    metadata=metadata,
                )
            )
        return CandidateGeneration(
            tuple(candidates),
            {
                "algorithm": "hpid-vlm-root-localization-sam2-v1",
                "planner_backend": self.planner.backend_id,
                "target_label": target_label,
                "semantic_name": semantic_name,
                "selected_profile": selected_profile,
                "prompt": prompt,
                "response": response,
                "parse": parse_diagnostics,
                "planned_instance_count": len(planned),
                "candidate_count": len(candidates),
                "rejected_small_mask_count": rejected_small_masks,
                "ground_truth_used": False,
            },
        )
