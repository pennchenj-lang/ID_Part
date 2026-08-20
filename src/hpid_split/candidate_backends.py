from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from PIL import Image

from .foundation import FoundationCandidateGenerator
from .fusion import MaskCandidate
from .visual_regions import Sam2VisualRegionProposer, VisualRegionConfig


@dataclass(frozen=True)
class CandidateBackendResult:
    """Model-neutral evidence returned by one proposal backend."""

    candidates: tuple[MaskCandidate, ...]
    diagnostics: dict[str, object]


class CandidateBackend(Protocol):
    """Contract for external semantic, instance, or part proposal models.

    A backend may propose masks and labels, but it never assigns final Part IDs
    or exclusive pixel ownership. HPID fusion remains the only owner of those
    decisions, which makes it possible to compare and replace upstream models
    without changing the exported data contract.
    """

    backend_id: str

    def generate(
        self,
        image: Image.Image,
        existing_candidates: tuple[MaskCandidate, ...],
    ) -> CandidateBackendResult: ...


def validate_backend_result(
    result: CandidateBackendResult,
    *,
    image_shape: tuple[int, int],
) -> CandidateBackendResult:
    """Reject malformed or unverifiable proposals at the plugin boundary."""

    if result.diagnostics.get("ground_truth_used") is not False:
        raise ValueError("candidate backend must declare ground_truth_used=false")
    for candidate in result.candidates:
        if candidate.mask.shape != image_shape:
            raise ValueError(
                "candidate backend returned a mask that does not match the image"
            )
        if not candidate.source.strip():
            raise ValueError("candidate backend must record a source identifier")
    return result


class Sam2AutomaticMaskBackend:
    """HPID adapter around SAM2's label-free automatic mask generator."""

    backend_id = "sam2-automatic-mask"

    def __init__(
        self,
        generator: FoundationCandidateGenerator,
        *,
        config: VisualRegionConfig | None = None,
    ) -> None:
        self.proposer = Sam2VisualRegionProposer(
            generator.sam_processor,
            generator.sam_model,
            segmentation_model=generator.config.segmentation_model,
            device=generator.device,
            config=config,
        )

    def generate(
        self,
        image: Image.Image,
        existing_candidates: tuple[MaskCandidate, ...],
    ) -> CandidateBackendResult:
        generated = self.proposer.generate(image, list(existing_candidates))
        result = CandidateBackendResult(
            generated.candidates,
            {
                **generated.diagnostics,
                "backend_id": self.backend_id,
                "backend_role": "label_free_part_proposals",
                "final_part_ids_assigned_by_backend": False,
            },
        )
        return validate_backend_result(
            result,
            image_shape=(image.height, image.width),
        )
