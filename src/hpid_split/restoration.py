from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import torch
from PIL import Image

from .amodal import _complete_mask, _nearest_visible_texture
from .instances import PartInstance
from .occlusion import (
    rank_occluder_hypotheses,
    structural_non_occluder_indices,
    validate_amodal_proposal,
)

_CONFIG_PATH_KEYS = ("package_root", "model_cache", "cache_dir")
_UNEXPANDED_ENVIRONMENT = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*|\$\{[^}]+\}|%[^%]+%")


def _resolve_config_path(value: object, config_path: Path) -> Path:
    expanded = os.path.expanduser(os.path.expandvars(str(value)))
    if _UNEXPANDED_ENVIRONMENT.search(expanded):
        raise ValueError(f"completion config contains an undefined variable: {value}")
    resolved = Path(expanded)
    if not resolved.is_absolute():
        resolved = config_path.parent / resolved
    return resolved.resolve()


def _load_backend_payload(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in _CONFIG_PATH_KEYS:
        if key in payload:
            payload[key] = str(_resolve_config_path(payload[key], path))
    command = payload.get("command")
    if isinstance(command, list):
        payload["command"] = [
            os.path.expandvars(str(value)).replace("{config_dir}", str(path.parent))
            for value in command
        ]
    return payload


@dataclass(frozen=True)
class BackendProvenance:
    name: str
    version: str
    implementation_url: str
    publication_url: str
    license: str
    is_hpid_split_method: bool = False


@dataclass(frozen=True)
class CompletionOutput:
    full_mask: np.ndarray
    generated_rgba: np.ndarray
    confidence: float
    provenance: BackendProvenance
    metadata: dict[str, object]


@dataclass(frozen=True)
class CompletionRequest:
    image: Image.Image
    instance_map: np.ndarray
    target: PartInstance
    records: tuple[PartInstance, ...]

    @property
    def visible_mask(self) -> np.ndarray:
        return self.instance_map == self.target.instance_index

    @property
    def occupied_mask(self) -> np.ndarray:
        return (self.instance_map > 0) & ~self.visible_mask


@dataclass(frozen=True)
class RefinedMask:
    mask: np.ndarray
    quality: float


class AmodalMaskRefiner(Protocol):
    provenance: BackendProvenance

    def refine(
        self,
        image: Image.Image,
        visible_mask: np.ndarray,
        search_mask: np.ndarray,
    ) -> tuple[RefinedMask, ...]: ...


class CompletionBackend(Protocol):
    provenance: BackendProvenance

    def complete(self, request: CompletionRequest) -> CompletionOutput: ...


class RejectedCompletionError(RuntimeError):
    """A model output was produced but failed a non-negotiable safety gate."""


def visible_lock_compose(
    source: Image.Image,
    generated_rgba: np.ndarray,
    visible_mask: np.ndarray,
    full_mask: np.ndarray,
) -> np.ndarray:
    """Lock every visible source pixel and use generated pixels only when hidden."""
    source_rgba = np.asarray(source.convert("RGBA")).copy()
    generated = np.asarray(generated_rgba, dtype=np.uint8).copy()
    if generated.shape != source_rgba.shape:
        raise ValueError("generated RGBA image must match the source image size")
    full = full_mask.astype(bool) | visible_mask.astype(bool)
    output = np.zeros_like(source_rgba)
    hidden = full & ~visible_mask
    output[hidden] = generated[hidden]
    output[visible_mask] = source_rgba[visible_mask]
    output[..., 3] = full.astype(np.uint8) * 255
    if not np.array_equal(output[visible_mask], source_rgba[visible_mask]):
        raise AssertionError("visible-lock invariant was violated")
    return output


def _mask_crop(
    first: np.ndarray,
    second: np.ndarray,
    image_size: tuple[int, int],
    *,
    padding_ratio: float = 0.12,
) -> tuple[int, int, int, int]:
    union = first.astype(bool) | second.astype(bool)
    ys, xs = np.nonzero(union)
    if not len(xs):
        return 0, 0, image_size[0], image_size[1]
    width = int(xs.max() - xs.min() + 1)
    height = int(ys.max() - ys.min() + 1)
    padding = max(12, round(max(width, height) * padding_ratio))
    return (
        max(0, int(xs.min()) - padding),
        max(0, int(ys.min()) - padding),
        min(image_size[0], int(xs.max() + 1) + padding),
        min(image_size[1], int(ys.max() + 1) + padding),
    )


def _positive_points(mask: np.ndarray, maximum: int = 5) -> list[list[float]]:
    remaining = mask.astype(np.uint8).copy()
    points: list[list[float]] = []
    for _ in range(maximum):
        if not remaining.any():
            break
        distance = cv2.distanceTransform(remaining, cv2.DIST_L2, 5)
        y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
        if distance[y, x] <= 0:
            break
        points.append([float(x), float(y)])
        radius = max(3, round(float(distance[y, x]) * 1.8))
        cv2.circle(remaining, (int(x), int(y)), radius, 0, -1)
    if not points:
        ys, xs = np.nonzero(mask)
        if len(xs):
            points.append([float(np.median(xs)), float(np.median(ys))])
    return points


class Sam2AmodalRefiner:
    """Re-segment a target after an occluder region has been inpainted."""

    provenance = BackendProvenance(
        name="SAM 2.1 Hiera Tiny mask refiner",
        version="facebook/sam2.1-hiera-tiny",
        implementation_url="https://github.com/facebookresearch/sam2",
        publication_url="https://arxiv.org/abs/2408.00714",
        license="Apache-2.0; verify upstream checkpoint terms",
        is_hpid_split_method=False,
    )

    def __init__(self, processor: Any, model: Any, *, device: str) -> None:
        self.processor = processor
        self.model = model
        self.device = device

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "facebook/sam2.1-hiera-tiny",
        *,
        device: str = "cuda",
        local_files_only: bool = False,
    ) -> Sam2AmodalRefiner:
        try:
            from transformers import Sam2Model, Sam2Processor
        except ImportError as error:
            raise RuntimeError(
                "Install the foundation extra: pip install 'hpid-split[foundation]'"
            ) from error
        processor = Sam2Processor.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        model = Sam2Model.from_pretrained(
            model_name, local_files_only=local_files_only
        ).to(device)
        model.eval()
        return cls(processor, model, device=device)

    def refine(
        self,
        image: Image.Image,
        visible_mask: np.ndarray,
        search_mask: np.ndarray,
    ) -> tuple[RefinedMask, ...]:
        if not visible_mask.any() or not search_mask.any():
            return ()
        x0, y0, x1, y1 = _mask_crop(
            visible_mask, search_mask, image.size, padding_ratio=0.16
        )
        crop = image.crop((x0, y0, x1, y1)).convert("RGB")
        local_visible = visible_mask[y0:y1, x0:x1]
        local_search = search_mask[y0:y1, x0:x1]
        points = _positive_points(local_visible)
        if not points:
            return ()
        ys, xs = np.nonzero(local_visible | local_search)
        box = [
            float(xs.min()),
            float(ys.min()),
            float(xs.max() + 1),
            float(ys.max() + 1),
        ]
        inputs = self.processor(
            images=crop,
            input_points=[[points]],
            input_labels=[[[1] * len(points)]],
            input_boxes=[[box]],
            return_tensors="pt",
        ).to(self.device)
        with (
            torch.inference_mode(),
            torch.amp.autocast("cuda", enabled=self.device.startswith("cuda")),
        ):
            outputs = self.model(**inputs)
        try:
            processed = self.processor.post_process_masks(
                outputs.pred_masks.detach().cpu(),
                inputs["original_sizes"].detach().cpu(),
                inputs["reshaped_input_sizes"].detach().cpu(),
                binarize=True,
            )[0]
        except (KeyError, TypeError):
            processed = self.processor.post_process_masks(
                outputs.pred_masks.detach().cpu(),
                inputs["original_sizes"].detach().cpu(),
                binarize=True,
            )[0]
        masks = np.asarray(processed)
        if masks.ndim == 3:
            masks = masks[None]
        if masks.ndim != 4:
            raise RuntimeError(f"unexpected SAM2 mask shape: {masks.shape}")
        score_tensor = getattr(outputs, "iou_scores", None)
        if score_tensor is None:
            score_tensor = getattr(outputs, "pred_iou_scores", None)
        if score_tensor is None:
            qualities = np.full(masks.shape[1], 0.5, dtype=np.float32)
        else:
            scores = score_tensor.detach().float().cpu().numpy()
            while scores.ndim > 2:
                scores = scores[0]
            qualities = np.asarray(scores[0] if scores.ndim == 2 else scores)
        proposals: list[RefinedMask] = []
        for index in range(masks.shape[1]):
            full = np.zeros(visible_mask.shape, dtype=bool)
            full[y0:y1, x0:x1] = masks[0, index].astype(bool)
            quality = float(qualities[min(index, len(qualities) - 1)])
            proposals.append(RefinedMask(full, float(np.clip(quality, 0.0, 1.0))))
        return tuple(proposals)


class GeometricFallbackBackend:
    """Low-confidence no-model fallback; not part of the claimed AI method."""

    provenance = BackendProvenance(
        name="HPID geometric fallback",
        version="0.1.0",
        implementation_url="",
        publication_url="",
        license="repository license",
        is_hpid_split_method=False,
    )

    def complete(self, request: CompletionRequest) -> CompletionOutput:
        visible_mask = request.visible_mask
        full_mask, confidence, _ = _complete_mask(visible_mask, request.occupied_mask)
        generated = _nearest_visible_texture(
            np.asarray(request.image.convert("RGBA")).copy(), visible_mask, full_mask
        )
        return CompletionOutput(
            full_mask=full_mask,
            generated_rgba=generated,
            confidence=min(0.35, confidence),
            provenance=self.provenance,
            metadata={
                "semantic_name": request.target.semantic_name,
                "warning": "geometric fallback; no learned amodal evidence",
            },
        )


class ExternalProcessCompletionBackend:
    """Adapter contract for Pix2Gestalt, LaMa/MAT, or another cited backend.

    The external command receives paths through placeholders in ``command`` and
    must write ``full_mask.png`` plus ``generated_rgba.png``. External source code
    and checkpoints remain under their upstream licenses.
    """

    def __init__(
        self,
        command: tuple[str, ...],
        provenance: BackendProvenance,
        *,
        timeout_seconds: int = 900,
    ) -> None:
        if not command:
            raise ValueError("external completion command must not be empty")
        self.command = command
        self.provenance = provenance
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_json(cls, path: Path) -> ExternalProcessCompletionBackend:
        payload = _load_backend_payload(path)
        if payload.get("kind") != "external-command":
            raise ValueError("only external-command completion configs are supported")
        return cls(
            tuple(str(value) for value in payload["command"]),
            BackendProvenance(**payload["provenance"]),
            timeout_seconds=int(payload.get("timeout_seconds", 900)),
        )

    def complete(self, request: CompletionRequest) -> CompletionOutput:
        image = request.image
        visible_mask = request.visible_mask
        occupied_mask = request.occupied_mask
        semantic_name = request.target.semantic_name
        proposed_full, proposed_confidence, _ = _complete_mask(
            visible_mask, occupied_mask
        )
        with tempfile.TemporaryDirectory(prefix="hpid-completion-") as directory:
            work = Path(directory)
            source_path = work / "source.png"
            visible_path = work / "visible_mask.png"
            occupied_path = work / "occupied_mask.png"
            proposed_full_path = work / "proposed_full_mask.png"
            instance_map_path = work / "part_id_map.tiff"
            request_path = work / "request.json"
            full_path = work / "full_mask.png"
            generated_path = work / "generated_rgba.png"
            metadata_path = work / "completion.json"
            image.save(source_path)
            Image.fromarray(visible_mask.astype(np.uint8) * 255).save(visible_path)
            Image.fromarray(occupied_mask.astype(np.uint8) * 255).save(occupied_path)
            Image.fromarray(proposed_full.astype(np.uint8) * 255).save(
                proposed_full_path
            )
            Image.fromarray(request.instance_map.astype(np.uint16), mode="I;16").save(
                instance_map_path
            )
            request_path.write_text(
                json.dumps(
                    {
                        "semantic_name": semantic_name,
                        "target": request.target.to_dict(),
                        "parts": [record.to_dict() for record in request.records],
                        "source": str(source_path),
                        "visible_mask": str(visible_path),
                        "occupied_mask": str(occupied_path),
                        "part_id_map": str(instance_map_path),
                        "proposed_full_mask": str(proposed_full_path),
                        "proposed_shape_confidence": proposed_confidence,
                        "full_mask_output": str(full_path),
                        "generated_rgba_output": str(generated_path),
                        "metadata_output": str(metadata_path),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            replacements = {
                "request": str(request_path),
                "workdir": str(work),
                "source": str(source_path),
                "visible_mask": str(visible_path),
                "occupied_mask": str(occupied_path),
                "part_id_map": str(instance_map_path),
                "proposed_full_mask": str(proposed_full_path),
                "full_mask": str(full_path),
                "generated_rgba": str(generated_path),
                "metadata": str(metadata_path),
                "semantic_name": semantic_name,
                "hpid_repository": str(Path(__file__).resolve().parents[2]),
            }
            command = [value.format(**replacements) for value in self.command]
            subprocess.run(
                command,
                check=True,
                timeout=self.timeout_seconds,
                cwd=work,
                shell=False,
            )
            if not full_path.exists() or not generated_path.exists():
                raise RuntimeError(
                    "completion backend did not create full_mask.png and "
                    "generated_rgba.png"
                )
            full_mask = np.asarray(Image.open(full_path).convert("L")) >= 128
            generated = np.asarray(Image.open(generated_path).convert("RGBA"))
            if full_mask.shape != visible_mask.shape:
                raise ValueError("completion backend returned a mask with wrong size")
            metadata = (
                json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata_path.exists()
                else {}
            )
            confidence = float(metadata.get("confidence", 0.5))
            return CompletionOutput(
                full_mask=full_mask | visible_mask,
                generated_rgba=generated,
                confidence=float(np.clip(confidence, 0.0, 1.0)),
                provenance=self.provenance,
                metadata=metadata,
            )


class TargetPackageLamaBackend:
    """Load an isolated LaMa wrapper directory while reusing the main runtime."""

    def __init__(
        self,
        package_root: Path,
        model_cache: Path,
        provenance: BackendProvenance,
    ) -> None:
        if not package_root.exists():
            raise FileNotFoundError(
                f"LaMa target package is missing: {package_root}. Run "
                "scripts/setup_lama_backend.ps1 first."
            )
        model_cache.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(package_root))
        os.environ["TORCH_HOME"] = str(model_cache)
        try:
            from simple_lama_inpainting import SimpleLama
        except ImportError as error:
            raise RuntimeError(
                "The isolated LaMa package could not be imported"
            ) from error
        self.provenance = provenance
        self.model = SimpleLama()

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> TargetPackageLamaBackend:
        return cls(
            Path(str(payload["package_root"])),
            Path(str(payload["model_cache"])),
            BackendProvenance(**payload["provenance"]),
        )

    def _inpaint(
        self,
        image: Image.Image,
        mask: np.ndarray,
        semantic_name: str = "",
    ) -> np.ndarray:
        generated = np.asarray(image.convert("RGBA")).copy()
        if not mask.any():
            return generated
        x0, y0, x1, y1 = _mask_crop(mask, mask, image.size, padding_ratio=0.22)
        crop_image = image.crop((x0, y0, x1, y1)).convert("RGB")
        local_mask = mask[y0:y1, x0:x1]
        crop_mask = Image.fromarray(local_mask.astype(np.uint8) * 255)
        inpainted = np.asarray(self.model(crop_image, crop_mask).convert("RGBA"))[
            : crop_image.height, : crop_image.width
        ]
        local_generated = generated[y0:y1, x0:x1]
        local_generated[local_mask] = inpainted[local_mask]
        generated[y0:y1, x0:x1] = local_generated
        return generated

    def complete(self, request: CompletionRequest) -> CompletionOutput:
        visible_mask = request.visible_mask
        full_mask, shape_confidence, _ = _complete_mask(
            visible_mask, request.occupied_mask
        )
        hidden = full_mask & ~visible_mask
        generated = self._inpaint(request.image, hidden, request.target.semantic_name)
        return CompletionOutput(
            full_mask=full_mask,
            generated_rgba=generated,
            confidence=min(0.45, shape_confidence),
            provenance=self.provenance,
            metadata={
                "semantic_name": request.target.semantic_name,
                "amodal_shape_source": "HPID geometric fallback",
                "hidden_appearance_source": "LaMa",
                "warning": (
                    "LaMa supplies appearance only; physical internal structure "
                    "is not validated"
                ),
            },
        )


class TargetPackageLamaSamBackend(TargetPackageLamaBackend):
    """Evidence-gated amodal completion using LaMa removal and SAM2 recovery."""

    def __init__(
        self,
        package_root: Path,
        model_cache: Path,
        lama_provenance: BackendProvenance,
        refiner: AmodalMaskRefiner,
        *,
        maximum_hypotheses: int = 3,
        maximum_target_parts: int = 12,
        minimum_target_area_ratio: float = 0.001,
        pipeline_provenance: BackendProvenance | None = None,
    ) -> None:
        super().__init__(package_root, model_cache, lama_provenance)
        self.lama_provenance = lama_provenance
        self.refiner = refiner
        self.maximum_hypotheses = maximum_hypotheses
        self.maximum_target_parts = maximum_target_parts
        self.minimum_target_area_ratio = minimum_target_area_ratio
        self.provenance = pipeline_provenance or BackendProvenance(
            name="HPID evidence-gated amodal completion",
            version="0.2.0",
            implementation_url="",
            publication_url="",
            license="repository license plus upstream model licenses",
            is_hpid_split_method=True,
        )

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
        refiner: AmodalMaskRefiner,
    ) -> TargetPackageLamaSamBackend:
        pipeline_payload = payload.get("pipeline_provenance")
        return cls(
            Path(str(payload["package_root"])),
            Path(str(payload["model_cache"])),
            BackendProvenance(**payload["provenance"]),
            refiner,
            maximum_hypotheses=int(payload.get("maximum_hypotheses", 3)),
            maximum_target_parts=int(payload.get("maximum_target_parts", 12)),
            minimum_target_area_ratio=float(
                payload.get("minimum_target_area_ratio", 0.001)
            ),
            pipeline_provenance=(
                BackendProvenance(**pipeline_payload)
                if isinstance(pipeline_payload, dict)
                else None
            ),
        )

    def select_target_indices(
        self,
        instance_map: np.ndarray,
        records: tuple[PartInstance, ...],
    ) -> frozenset[int]:
        image_area = max(1, instance_map.size)
        ranked: list[tuple[float, int]] = []
        for record in records:
            if record.semantic_name == record.semantic_parent:
                continue
            visible = instance_map == record.instance_index
            area_ratio = np.count_nonzero(visible) / image_area
            if area_ratio < self.minimum_target_area_ratio:
                continue
            excluded = structural_non_occluder_indices(record, records)
            hypotheses = rank_occluder_hypotheses(
                visible,
                instance_map,
                record.instance_index,
                maximum_hypotheses=1,
                excluded_instance_indices=excluded,
            )
            if not hypotheses:
                continue
            area_weight = min(
                1.0,
                np.sqrt(area_ratio / max(self.minimum_target_area_ratio, 0.01)),
            )
            ranked.append((hypotheses[0].score * area_weight, record.instance_index))
        ranked.sort(reverse=True)
        return frozenset(
            index for _, index in ranked[: max(0, self.maximum_target_parts)]
        )

    def _shape_source_label(self) -> str:
        return (
            f"{self.refiner.provenance.name} after per-ID "
            f"{self.lama_provenance.name} appearance completion"
        )

    def _appearance_source_label(self) -> str:
        return self.lama_provenance.name

    def complete(self, request: CompletionRequest) -> CompletionOutput:
        visible = request.visible_mask
        excluded = structural_non_occluder_indices(request.target, request.records)
        hypotheses = rank_occluder_hypotheses(
            visible,
            request.instance_map,
            request.target.instance_index,
            maximum_hypotheses=self.maximum_hypotheses,
            excluded_instance_indices=excluded,
        )
        generated = np.asarray(request.image.convert("RGBA")).copy()
        full = visible.copy()
        accepted_occluders: list[int] = []
        accepted_scores: list[float] = []
        audit: list[dict[str, object]] = []
        for hypothesis in hypotheses:
            try:
                inpainted = self._inpaint(
                    request.image,
                    hypothesis.search_mask,
                    request.target.semantic_name,
                )
            except RejectedCompletionError as error:
                audit.append(
                    {
                        "occluder_instance_index": hypothesis.instance_index,
                        "contact_px": hypothesis.contact_px,
                        "hypothesis_score": hypothesis.score,
                        "search_area_px": int(np.count_nonzero(hypothesis.search_mask)),
                        "proposal_count": 0,
                        "accepted": False,
                        "decision": "appearance_backend_rejected",
                        "rejection_reason": str(error),
                    }
                )
                continue
            inpainted_image = Image.fromarray(inpainted, mode="RGBA").convert("RGB")
            proposals = self.refiner.refine(
                inpainted_image, visible, hypothesis.search_mask
            )
            evidence = [
                validate_amodal_proposal(
                    visible,
                    proposal.mask,
                    hypothesis.search_mask,
                    proposal.quality,
                )
                for proposal in proposals
            ]
            best = max(
                evidence,
                key=lambda item: (item.accepted, item.score),
                default=None,
            )
            entry: dict[str, object] = {
                "occluder_instance_index": hypothesis.instance_index,
                "contact_px": hypothesis.contact_px,
                "hypothesis_score": hypothesis.score,
                "search_area_px": int(np.count_nonzero(hypothesis.search_mask)),
                "proposal_count": len(proposals),
                "accepted": bool(best and best.accepted),
            }
            if best is not None:
                entry.update(
                    {
                        "evidence_score": best.score,
                        "visible_recall": best.visible_recall,
                        "search_precision": best.search_precision,
                        "orthogonal_precision": best.orthogonal_precision,
                        "orthogonal_span_ratio": best.orthogonal_span_ratio,
                        "added_area_px": best.added_area_px,
                        "decision": best.reason,
                    }
                )
            else:
                entry["decision"] = "no_refiner_proposal"
            audit.append(entry)
            if best is None or not best.accepted:
                continue
            added = best.added_mask & ~full
            if not added.any():
                continue
            generated[added] = inpainted[added]
            full |= added
            accepted_occluders.append(hypothesis.instance_index)
            accepted_scores.append(best.score)

        if not hypotheses:
            status = "no_adjacent_occluder_hypothesis"
            confidence = 1.0
        elif accepted_scores:
            status = "model_supported_completion"
            confidence = float(np.mean(accepted_scores))
        else:
            status = "no_model_supported_completion"
            confidence = 0.0
        return CompletionOutput(
            full_mask=full,
            generated_rgba=generated,
            confidence=confidence,
            provenance=self.provenance,
            metadata={
                "semantic_name": request.target.semantic_name,
                "status": status,
                "amodal_shape_source": self._shape_source_label(),
                "hidden_appearance_source": self._appearance_source_label(),
                "accepted_occluder_instance_indices": accepted_occluders,
                "hypotheses": audit,
                "components": {
                    "appearance_completion": asdict(self.lama_provenance),
                    "shape_refinement": asdict(self.refiner.provenance),
                },
                "warning": (
                    "The output is a learned 2D amodal hypothesis, not recovered "
                    "physical internal geometry."
                ),
            },
        )


class DiffusersInpaintSamBackend(TargetPackageLamaSamBackend):
    """Text-conditioned appearance completion with the same evidence gate."""

    def __init__(
        self,
        model_id_or_path: str,
        cache_dir: Path,
        refiner: AmodalMaskRefiner,
        *,
        device: str,
        resolution: int = 512,
        inference_steps: int = 24,
        guidance_scale: float = 6.0,
        strength: float = 0.98,
        maximum_hypotheses: int = 1,
        maximum_target_parts: int = 2,
        minimum_target_area_ratio: float = 0.002,
        local_files_only: bool = False,
        variant: str | None = "fp16",
    ) -> None:
        if resolution < 256 or resolution % 8:
            raise ValueError("diffusion resolution must be >=256 and divisible by 8")
        self.model_id_or_path = model_id_or_path
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.refiner = refiner
        self.device = device
        self.resolution = resolution
        self.inference_steps = inference_steps
        self.guidance_scale = guidance_scale
        self.strength = strength
        self.maximum_hypotheses = maximum_hypotheses
        self.maximum_target_parts = maximum_target_parts
        self.minimum_target_area_ratio = minimum_target_area_ratio
        self.local_files_only = local_files_only
        self.variant = variant
        self._pipe: Any | None = None
        self.lama_provenance = BackendProvenance(
            name="Stable Diffusion v1.5 inpainting",
            version=model_id_or_path,
            implementation_url=(
                "https://huggingface.co/stable-diffusion-v1-5/"
                "stable-diffusion-inpainting"
            ),
            publication_url="https://arxiv.org/abs/2112.10752",
            license="CreativeML OpenRAIL-M",
            is_hpid_split_method=False,
        )
        self.provenance = BackendProvenance(
            name="HPID evidence-gated diffusion amodal completion",
            version="0.2.0",
            implementation_url="",
            publication_url="",
            license="repository license plus CreativeML OpenRAIL-M",
            is_hpid_split_method=True,
        )

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
        refiner: AmodalMaskRefiner,
        *,
        device: str,
    ) -> DiffusersInpaintSamBackend:
        variant_value = payload.get("variant", "fp16")
        return cls(
            str(
                payload.get(
                    "model",
                    "stable-diffusion-v1-5/stable-diffusion-inpainting",
                )
            ),
            Path(str(payload["cache_dir"])),
            refiner,
            device=device,
            resolution=int(payload.get("resolution", 512)),
            inference_steps=int(payload.get("inference_steps", 24)),
            guidance_scale=float(payload.get("guidance_scale", 6.0)),
            strength=float(payload.get("strength", 0.98)),
            maximum_hypotheses=int(payload.get("maximum_hypotheses", 1)),
            maximum_target_parts=int(payload.get("maximum_target_parts", 2)),
            minimum_target_area_ratio=float(
                payload.get("minimum_target_area_ratio", 0.002)
            ),
            local_files_only=bool(payload.get("local_files_only", False)),
            variant=(str(variant_value) if variant_value is not None else None),
        )

    def _load_pipe(self) -> Any:
        if self._pipe is not None:
            return self._pipe
        try:
            from diffusers import AutoPipelineForInpainting
        except ImportError as error:
            raise RuntimeError(
                "Install the diffusion extra: pip install 'hpid-split[diffusion]'"
            ) from error
        kwargs: dict[str, object] = {
            "cache_dir": str(self.cache_dir),
            "torch_dtype": torch.float16,
            "use_safetensors": True,
            "local_files_only": self.local_files_only,
        }
        if self.variant is not None:
            kwargs["variant"] = self.variant
        pipe = AutoPipelineForInpainting.from_pretrained(
            self.model_id_or_path, **kwargs
        )
        if self.device.startswith("cuda"):
            pipe.enable_model_cpu_offload()
            pipe.enable_attention_slicing("max")
            pipe.vae.enable_slicing()
        else:
            pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        self._pipe = pipe
        return pipe

    def _square_crop(
        self, mask: np.ndarray, image_size: tuple[int, int]
    ) -> tuple[int, int, int, int]:
        ys, xs = np.nonzero(mask)
        if not len(xs):
            return 0, 0, image_size[0], image_size[1]
        width = int(xs.max() - xs.min() + 1)
        height = int(ys.max() - ys.min() + 1)
        side = min(
            image_size[0],
            image_size[1],
            max(64, round(max(width, height) * 1.7)),
        )
        cx = float(xs.min() + xs.max()) / 2.0
        cy = float(ys.min() + ys.max()) / 2.0
        x0 = int(np.clip(round(cx - side / 2), 0, image_size[0] - side))
        y0 = int(np.clip(round(cy - side / 2), 0, image_size[1] - side))
        return x0, y0, x0 + side, y0 + side

    def _inpaint(
        self,
        image: Image.Image,
        mask: np.ndarray,
        semantic_name: str = "",
    ) -> np.ndarray:
        generated = np.asarray(image.convert("RGBA")).copy()
        if not mask.any():
            return generated
        x0, y0, x1, y1 = self._square_crop(mask, image.size)
        crop = image.crop((x0, y0, x1, y1)).convert("RGB")
        local_mask = mask[y0:y1, x0:x1]
        source_size = crop.size
        model_image = crop.resize(
            (self.resolution, self.resolution), Image.Resampling.LANCZOS
        )
        model_mask = Image.fromarray(local_mask.astype(np.uint8) * 255).resize(
            (self.resolution, self.resolution), Image.Resampling.NEAREST
        )
        readable_name = semantic_name.replace("_", " ") or "object part"
        prompt = (
            f"complete the hidden continuation of the same {readable_name}, "
            "same object, same material, same color, same texture, seamless"
        )
        negative_prompt = (
            "new object, duplicate object, text, watermark, unrelated material, "
            "changed visible region"
        )
        seed = int.from_bytes(
            hashlib.sha256(semantic_name.encode("utf-8")).digest()[:4], "little"
        )
        pipeline_output = self._load_pipe()(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=model_image,
            mask_image=model_mask,
            height=self.resolution,
            width=self.resolution,
            num_inference_steps=self.inference_steps,
            guidance_scale=self.guidance_scale,
            strength=self.strength,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        )
        nsfw_flags = getattr(pipeline_output, "nsfw_content_detected", None)
        if nsfw_flags and any(bool(value) for value in nsfw_flags):
            raise RejectedCompletionError(
                "diffusion safety checker rejected the generated region"
            )
        result = pipeline_output.images[0]
        inpainted = np.asarray(
            result.convert("RGB").resize(source_size, Image.Resampling.LANCZOS)
        )
        if float(inpainted.mean()) < 2.0 and float(inpainted.std()) < 2.0:
            raise RejectedCompletionError("diffusion backend returned a blank image")
        local_generated = generated[y0:y1, x0:x1]
        local_generated[local_mask, :3] = inpainted[local_mask]
        local_generated[local_mask, 3] = 255
        generated[y0:y1, x0:x1] = local_generated
        return generated


def load_completion_backend(
    path: Path,
    *,
    sam_processor: Any | None = None,
    sam_model: Any | None = None,
    device: str = "cuda",
) -> CompletionBackend:
    payload = _load_backend_payload(path)
    kind = payload.get("kind")
    if kind == "external-command":
        return ExternalProcessCompletionBackend.from_json(path)
    if kind == "target-package-lama":
        return TargetPackageLamaBackend.from_dict(payload)
    if kind == "target-package-lama-sam2":
        if (sam_processor is None) != (sam_model is None):
            raise ValueError("sam_processor and sam_model must be provided together")
        refiner = (
            Sam2AmodalRefiner(sam_processor, sam_model, device=device)
            if sam_processor is not None and sam_model is not None
            else Sam2AmodalRefiner.from_pretrained(
                str(payload.get("segmentation_model", "facebook/sam2.1-hiera-tiny")),
                device=device,
                local_files_only=bool(payload.get("local_files_only", False)),
            )
        )
        return TargetPackageLamaSamBackend.from_dict(payload, refiner)
    if kind == "diffusers-inpaint-sam2":
        if (sam_processor is None) != (sam_model is None):
            raise ValueError("sam_processor and sam_model must be provided together")
        refiner = (
            Sam2AmodalRefiner(sam_processor, sam_model, device=device)
            if sam_processor is not None and sam_model is not None
            else Sam2AmodalRefiner.from_pretrained(
                str(payload.get("segmentation_model", "facebook/sam2.1-hiera-tiny")),
                device=device,
                local_files_only=bool(payload.get("local_files_only", False)),
            )
        )
        return DiffusersInpaintSamBackend.from_dict(payload, refiner, device=device)
    raise ValueError(f"unknown completion backend kind: {kind!r}")


def complete_and_export_parts(
    image: Image.Image,
    instance_map: np.ndarray,
    records: list[PartInstance],
    output_dir: Path,
    backend: CompletionBackend,
) -> dict[int, dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks_full"
    crops_dir = output_dir / "crops_completed"
    masks_dir.mkdir(exist_ok=True)
    crops_dir.mkdir(exist_ok=True)
    completion_records: dict[int, dict[str, object]] = {}
    occlusion_edges: list[dict[str, object]] = []
    record_tuple = tuple(records)
    selector = getattr(backend, "select_target_indices", None)
    selected_targets = (
        selector(instance_map, record_tuple) if callable(selector) else None
    )
    for record in records:
        visible = instance_map == record.instance_index
        request = CompletionRequest(image, instance_map, record, record_tuple)
        if (
            selected_targets is not None
            and record.instance_index not in selected_targets
        ):
            result = CompletionOutput(
                full_mask=visible.copy(),
                generated_rgba=np.asarray(image.convert("RGBA")).copy(),
                confidence=0.0,
                provenance=backend.provenance,
                metadata={
                    "semantic_name": record.semantic_name,
                    "status": "not_selected_by_completion_budget",
                    "accepted_occluder_instance_indices": [],
                    "warning": "No hidden region was generated for this Part-ID.",
                },
            )
        else:
            result = backend.complete(request)
        full = result.full_mask.astype(bool) | visible
        completed = visible_lock_compose(image, result.generated_rgba, visible, full)
        stem = f"{record.instance_index:04d}_{record.semantic_name}"
        mask_path = masks_dir / f"{stem}.png"
        crop_path = crops_dir / f"{stem}.png"
        Image.fromarray(full.astype(np.uint8) * 255).save(mask_path)
        ys, xs = np.nonzero(full)
        if len(xs):
            x0, x1 = int(xs.min()), int(xs.max() + 1)
            y0, y1 = int(ys.min()), int(ys.max() + 1)
            Image.fromarray(completed[y0:y1, x0:x1]).save(crop_path)
        else:
            x0 = y0 = x1 = y1 = 0
            Image.fromarray(completed).save(crop_path)
        accepted_key = "accepted_occluder_instance_indices"
        evidence_by_occluder = {
            int(item["occluder_instance_index"]): item
            for item in result.metadata.get("hypotheses", [])
            if isinstance(item, dict) and "occluder_instance_index" in item
        }
        if accepted_key in result.metadata:
            occluders = [int(value) for value in result.metadata.get(accepted_key, [])]
            relation = "learned-completion-supported occlusion"
        else:
            occupied = request.occupied_mask
            ring = (
                cv2.dilate(visible.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(
                    bool
                )
                & occupied
            )
            occluders = [
                int(value)
                for value in np.unique(instance_map[ring])
                if int(value) not in {0, record.instance_index}
            ]
            relation = "adjacency-supported hypothesis"
        for occluder in sorted(set(occluders)):
            evidence = evidence_by_occluder.get(occluder, {})
            occlusion_edges.append(
                {
                    "occluder_instance_index": occluder,
                    "occluded_instance_index": record.instance_index,
                    "relation": relation,
                    "confidence": float(
                        evidence.get("evidence_score", result.confidence)
                    ),
                }
            )
        completion_records[record.instance_index] = {
            "mask_full_path": mask_path.relative_to(output_dir).as_posix(),
            "crop_completed_path": crop_path.relative_to(output_dir).as_posix(),
            "bbox_full": [x0, y0, x1, y1],
            "added_area_px": int(np.count_nonzero(full & ~visible)),
            "completion_confidence": result.confidence,
            "completion_backend": asdict(result.provenance),
            "completion_metadata": result.metadata,
        }
    (output_dir / "occlusion_edges.json").write_text(
        json.dumps(occlusion_edges, indent=2), encoding="utf-8"
    )
    return completion_records
