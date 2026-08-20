from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image

from .dense_semantic import parent_envelope
from .foundation import CandidateGeneration
from .fusion import MaskCandidate, mask_iou
from .guided_prompts import GuidedPromptSpec


@dataclass(frozen=True)
class Sam3TextConfig:
    model_name: str = "facebook/sam3"
    score_threshold: float = 0.34
    mask_threshold: float = 0.50
    minimum_root_containment: float = 0.52
    maximum_root_area_fraction: float = 0.82
    same_prompt_nms_iou: float = 0.84
    local_files_only: bool = False


def _root_key(candidate: MaskCandidate) -> str:
    return (
        f"{candidate.metadata.get('root_origin', 'legacy')}::"
        f"{candidate.metadata.get('root_index', 'unknown')}"
    )


class Sam3TextPartGenerator:
    """SAM3 text-prompt segmentation converted into routed HPID candidates."""

    def __init__(
        self,
        *,
        device: str,
        config: Sam3TextConfig | None = None,
        processor: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self.device = device
        self.config = config or Sam3TextConfig()
        if (processor is None) != (model is None):
            raise ValueError("SAM3 processor and model must be supplied together")
        if processor is None:
            try:
                from transformers import Sam3Model, Sam3Processor

                processor = Sam3Processor.from_pretrained(
                    self.config.model_name,
                    local_files_only=self.config.local_files_only,
                )
                model = Sam3Model.from_pretrained(
                    self.config.model_name,
                    local_files_only=self.config.local_files_only,
                ).to(device)
            except Exception as error:
                raise RuntimeError(
                    "SAM3 text segmentation is unavailable. Accept the model "
                    "terms at https://huggingface.co/facebook/sam3 and run "
                    "`hf auth login`, or use --guided-backend grounded-sam2."
                ) from error
        self.processor = processor
        self.model = model
        self.model.eval()

    def release(self) -> None:
        if self.device.startswith("cuda"):
            self.model.to("cpu")
            torch.cuda.empty_cache()

    def generate(
        self,
        image: Image.Image,
        roots: list[MaskCandidate],
        prompts: tuple[GuidedPromptSpec, ...],
    ) -> CandidateGeneration:
        image = image.convert("RGB")
        image_inputs = self.processor(images=image, return_tensors="pt").to(
            self.device
        )
        with torch.inference_mode():
            vision_features = self.model.get_vision_features(
                pixel_values=image_inputs.pixel_values
            )
        root_areas = {_root_key(root): int(np.count_nonzero(root.mask)) for root in roots}
        root_envelopes = {
            _root_key(root): parent_envelope(root.mask, 0.025) for root in roots
        }
        candidates: list[MaskCandidate] = []
        prompt_rows: list[dict[str, object]] = []
        for spec in prompts:
            proposed: list[tuple[float, np.ndarray, MaskCandidate, str]] = []
            for phrase in spec.phrases:
                text_inputs = self.processor(text=phrase, return_tensors="pt").to(
                    self.device
                )
                with torch.inference_mode():
                    outputs = self.model(
                        vision_embeds=vision_features,
                        **text_inputs,
                    )
                results = self.processor.post_process_instance_segmentation(
                    outputs,
                    threshold=self.config.score_threshold,
                    mask_threshold=self.config.mask_threshold,
                    target_sizes=image_inputs.get("original_sizes").tolist(),
                )[0]
                masks = results.get("masks", ())
                scores = results.get("scores", ())
                for raw_mask, raw_score in zip(masks, scores, strict=True):
                    mask = np.asarray(
                        raw_mask.detach().cpu()
                        if hasattr(raw_mask, "detach")
                        else raw_mask
                    ).astype(bool)
                    if mask.ndim > 2:
                        mask = np.squeeze(mask)
                    area = int(np.count_nonzero(mask))
                    if area < 12:
                        continue
                    root_matches: list[tuple[float, MaskCandidate]] = []
                    for root in roots:
                        key = _root_key(root)
                        intersection = int(np.count_nonzero(mask & root.mask))
                        containment = intersection / max(1, area)
                        fraction = area / max(1, root_areas[key])
                        if (
                            containment >= self.config.minimum_root_containment
                            and fraction <= self.config.maximum_root_area_fraction
                        ):
                            root_matches.append((containment, root))
                    if not root_matches:
                        continue
                    _, root = max(root_matches, key=lambda item: item[0])
                    mask &= root_envelopes[_root_key(root)]
                    proposed.append((float(raw_score), mask, root, phrase))

            kept: list[tuple[float, np.ndarray, MaskCandidate, str]] = []
            for proposal in sorted(proposed, key=lambda item: item[0], reverse=True):
                if any(
                    _root_key(proposal[2]) == _root_key(existing[2])
                    and mask_iou(proposal[1], existing[1])
                    >= self.config.same_prompt_nms_iou
                    for existing in kept
                ):
                    continue
                kept.append(proposal)
                if len(kept) >= spec.maximum_instances:
                    break
            for ordinal, (score, mask, root, phrase) in enumerate(kept, start=1):
                root_candidate_key = str(root.metadata.get("candidate_key"))
                semantic_name = f"{root.semantic_name}_guided_{spec.slug}"
                candidates.append(
                    MaskCandidate(
                        semantic_name=semantic_name,
                        semantic_parent=root.semantic_name,
                        mask=mask,
                        score=score,
                        source=f"sam3[{self.config.model_name}]/text-part",
                        prompt=phrase,
                        source_reliability=0.91,
                        metadata={
                            "source_family": f"sam3[{self.config.model_name}]",
                            "root_origin": root.metadata.get("root_origin"),
                            "root_index": root.metadata.get("root_index"),
                            "candidate_key": (
                                f"{root_candidate_key}/{semantic_name}:{ordinal:02d}"
                            ),
                            "parent_candidate_key": root_candidate_key,
                            "assembly_parent_semantic": root.semantic_name,
                            "assembly_parent_candidate_key": root_candidate_key,
                            "guided_prompt": True,
                            "guided_prompt_label": spec.label,
                            "guided_prompt_phrases": list(spec.phrases),
                            "guided_backend": "sam3-text",
                            "ground_truth_used": False,
                        },
                    )
                )
            prompt_rows.append(
                {
                    "label": spec.label,
                    "phrases": list(spec.phrases),
                    "proposal_count": len(proposed),
                    "accepted_candidate_count": len(kept),
                }
            )
        return CandidateGeneration(
            tuple(candidates),
            {
                "algorithm": "sam3-text-to-hpid-candidates-v1",
                "model": self.config.model_name,
                "prompt_count": len(prompts),
                "candidate_count": len(candidates),
                "prompts": prompt_rows,
                "ground_truth_used": False,
            },
        )
