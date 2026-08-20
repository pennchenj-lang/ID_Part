from __future__ import annotations

import hashlib
import json
import re
from colorsys import hsv_to_rgb
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import __version__
from .fusion import MaskCandidate
from .inference import SplitPrediction
from .instances import PartInstance
from .physical_groups import PhysicalGroupingResult, build_physical_groups
from .quality import assess_product_quality, render_target_candidates
from .taxonomy import Taxonomy


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _algorithm_metadata(
    diagnostics: dict[str, object] | None,
    checkpoint: Path | None,
) -> dict[str, object]:
    candidate_models: list[dict[str, object]] = []
    prompt_bank_sha256 = None
    fusion_config: dict[str, object] | None = None
    completion_backend = None
    relational_algorithm = None
    if diagnostics is not None:
        prompt_bank = diagnostics.get("prompt_bank")
        if isinstance(prompt_bank, dict):
            prompt_bank_sha256 = prompt_bank.get("sha256")
        generations = diagnostics.get("candidate_generations", [])
        if isinstance(generations, list):
            for generation in generations:
                if not isinstance(generation, dict):
                    continue
                models = generation.get("models")
                if not isinstance(models, dict):
                    continue
                candidate_models.append(
                    {
                        key: models.get(key)
                        for key in (
                            "grounding_model",
                            "segmentation_model",
                            "dense_semantic_model",
                        )
                        if models.get(key) is not None
                    }
                )
        fusion = diagnostics.get("fusion")
        if isinstance(fusion, dict) and isinstance(fusion.get("ablation"), dict):
            fusion_config = _json_safe(fusion["ablation"])
        completion = diagnostics.get("completion")
        if isinstance(completion, dict):
            completion_backend = completion.get("backend")
        relational = diagnostics.get("relational_candidate_generation")
        if isinstance(relational, dict):
            relational_algorithm = relational.get("algorithm")
    return {
        "name": "HPID-Split",
        "version": __version__,
        "mode": "foundation-fusion" if diagnostics is not None else "learned-checkpoint",
        "ground_truth_used": False,
        "prompt_bank_sha256": prompt_bank_sha256,
        "candidate_models": candidate_models,
        "relational_algorithm": relational_algorithm,
        "fusion_config": fusion_config,
        "completion_backend": completion_backend,
        "checkpoint_sha256": _sha256(checkpoint) if checkpoint is not None else None,
    }


def _export_candidate_audit(
    output_dir: Path, candidates: list[MaskCandidate] | tuple[MaskCandidate, ...]
) -> None:
    masks_dir = output_dir / "candidate_masks"
    masks_dir.mkdir(exist_ok=True)
    payload: list[dict[str, object]] = []
    for index, candidate in enumerate(candidates, start=1):
        stem = f"{index:04d}_{_safe_name(candidate.semantic_name)}"
        mask_path = masks_dir / f"{stem}.png"
        Image.fromarray(candidate.mask.astype(np.uint8) * 255, mode="L").save(mask_path)
        payload.append(
            {
                "candidate_index": index,
                "semantic_name": candidate.semantic_name,
                "semantic_parent": candidate.semantic_parent,
                "score": candidate.score,
                "source": candidate.source,
                "prompt": candidate.prompt,
                "source_reliability": candidate.source_reliability,
                "area_px": int(np.count_nonzero(candidate.mask)),
                "mask_path": mask_path.relative_to(output_dir).as_posix(),
                "metadata": _json_safe(candidate.metadata),
            }
        )
    (output_dir / "candidates.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def colorize_semantic(labels: np.ndarray, class_count: int) -> Image.Image:
    generator = np.random.default_rng(1907)
    colors = np.vstack(
        [
            np.zeros((1, 3), np.uint8),
            generator.integers(35, 240, size=(class_count - 1, 3), dtype=np.uint8),
        ]
    )
    return Image.fromarray(colors[labels])


def _instance_palette(maximum_index: int) -> np.ndarray:
    colors = np.zeros((maximum_index + 1, 3), dtype=np.uint8)
    for index in range(1, maximum_index + 1):
        hue = (0.11 + index * 0.618033988749895) % 1.0
        saturation = 0.64 + 0.12 * ((index * 7) % 3) / 2.0
        value = 0.82 + 0.12 * ((index * 5) % 4) / 3.0
        colors[index] = np.asarray(hsv_to_rgb(hue, saturation, value)) * 255
    return colors


def colorize_part_ids(instance_map: np.ndarray) -> Image.Image:
    """Render unique IDs while maximizing contrast between touching regions."""

    maximum_index = int(instance_map.max(initial=0))
    colors = np.zeros((maximum_index + 1, 3), dtype=np.uint8)
    active_ids = [int(value) for value in np.unique(instance_map) if value > 0]
    if not active_ids:
        return Image.fromarray(colors[instance_map], mode="RGB")

    adjacency = {part_id: set() for part_id in active_ids}
    for first, second in (
        (instance_map[:, :-1], instance_map[:, 1:]),
        (instance_map[:-1], instance_map[1:]),
    ):
        changed = (first != second) & (first > 0) & (second > 0)
        for left, right in zip(first[changed], second[changed], strict=True):
            left_id = int(left)
            right_id = int(right)
            adjacency[left_id].add(right_id)
            adjacency[right_id].add(left_id)

    candidates = _instance_palette(len(active_ids))[1:]
    candidate_lab = cv2.cvtColor(
        candidates[np.newaxis, :, :], cv2.COLOR_RGB2LAB
    )[0].astype(np.float32)
    assigned: dict[int, int] = {}
    available = set(range(len(active_ids)))
    while len(assigned) < len(active_ids):
        remaining = [part_id for part_id in active_ids if part_id not in assigned]
        part_id = max(
            remaining,
            key=lambda value: (
                sum(neighbor in assigned for neighbor in adjacency[value]),
                len(adjacency[value]),
                -value,
            ),
        )
        neighboring_colors = tuple(
            assigned[neighbor]
            for neighbor in adjacency[part_id]
            if neighbor in assigned
        )
        globally_used = tuple(assigned.values())

        def contrast_score(
            candidate_index: int,
            neighboring: tuple[int, ...] = neighboring_colors,
            used: tuple[int, ...] = globally_used,
        ) -> tuple[float, float, int]:
            neighbor_distance = (
                min(
                    float(
                        np.linalg.norm(
                            candidate_lab[candidate_index] - candidate_lab[index]
                        )
                    )
                    for index in neighboring
                )
                if neighboring
                else 0.0
            )
            global_distance = (
                min(
                    float(
                        np.linalg.norm(
                            candidate_lab[candidate_index] - candidate_lab[index]
                        )
                    )
                    for index in used
                )
                if used
                else 0.0
            )
            return neighbor_distance, global_distance, -candidate_index

        selected_color = max(available, key=contrast_score)
        assigned[part_id] = selected_color
        available.remove(selected_color)

    for part_id, color_index in assigned.items():
        colors[part_id] = candidates[color_index]
    return Image.fromarray(colors[instance_map], mode="RGB")


def render_source_overlay(
    image: Image.Image,
    instance_map: np.ndarray,
) -> Image.Image:
    """Overlay unique Part-ID colors while preserving inspectable source edges."""

    source = np.asarray(image.convert("RGB"), dtype=np.float32)
    colors = np.asarray(colorize_part_ids(instance_map), dtype=np.float32)
    foreground = instance_map > 0
    rendered = source.copy()
    rendered[foreground] = 0.56 * source[foreground] + 0.44 * colors[foreground]
    if np.any(foreground):
        boundary = np.zeros(instance_map.shape, dtype=bool)
        boundary[1:] |= instance_map[1:] != instance_map[:-1]
        boundary[:-1] |= instance_map[:-1] != instance_map[1:]
        boundary[:, 1:] |= instance_map[:, 1:] != instance_map[:, :-1]
        boundary[:, :-1] |= instance_map[:, :-1] != instance_map[:, 1:]
        rendered[boundary & foreground] = np.asarray((255, 255, 255))
    return Image.fromarray(np.clip(rendered, 0, 255).astype(np.uint8), mode="RGB")


def load_previous_package(path: Path) -> tuple[np.ndarray, list[PartInstance]]:
    instance_map = np.asarray(Image.open(path / "part_id_map.tiff"), dtype=np.uint16)
    payload = json.loads((path / "parts.json").read_text(encoding="utf-8"))
    records = [
        PartInstance(
            part_id=str(item["part_id"]),
            semantic_name=str(item["semantic_name"]),
            semantic_parent=str(item["semantic_parent"]),
            instance_index=int(item["instance_index"]),
            side=str(item["side"]),
            bbox_xyxy=tuple(int(value) for value in item["bbox_xyxy"]),
            centroid_xy=tuple(float(value) for value in item["centroid_xy"]),
            area_px=int(item["area_px"]),
            asset_id=str(item.get("asset_id", "object_001")),
            assembly_parent_id=(
                str(item["assembly_parent_id"])
                if item.get("assembly_parent_id") is not None
                else None
            ),
            group_id=(
                str(item["group_id"])
                if item.get("group_id") is not None
                else str(item["part_id"])
            ),
        )
        for item in payload
    ]
    return instance_map, records


def export_prediction(
    image: Image.Image,
    prediction: SplitPrediction,
    taxonomy: Taxonomy,
    output_dir: Path,
    *,
    records: list[PartInstance] | None = None,
    checkpoint: Path | None = None,
    diagnostics: dict[str, object] | None = None,
    completion_records: dict[int, dict[str, object]] | None = None,
    candidates: list[MaskCandidate] | tuple[MaskCandidate, ...] | None = None,
    physical_groups: PhysicalGroupingResult | None = None,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks_visible"
    crops_dir = output_dir / "crops"
    masks_dir.mkdir(exist_ok=True)
    crops_dir.mkdir(exist_ok=True)
    records = list(records if records is not None else prediction.instances)
    physical_groups = physical_groups or build_physical_groups(
        prediction.instance_map,
        records,
        candidates=(candidates or ()),
        image=image,
    )
    records = list(physical_groups.records)
    source_path = output_dir / "source.png"
    image.save(source_path)
    Image.fromarray(prediction.semantic_map.astype(np.uint8), mode="L").save(
        output_dir / "semantic_ids.png"
    )
    colorize_semantic(prediction.semantic_map, taxonomy.num_fine_classes).save(
        output_dir / "semantic_preview.png"
    )
    Image.fromarray(prediction.instance_map.astype(np.uint16), mode="I;16").save(
        output_dir / "part_id_map.tiff"
    )
    colorize_part_ids(prediction.instance_map).save(output_dir / "part_id_preview.png")
    render_source_overlay(image, prediction.instance_map).save(
        output_dir / "source_overlay.png"
    )
    Image.fromarray(physical_groups.group_map.astype(np.uint16), mode="I;16").save(
        output_dir / "group_id_map.tiff"
    )
    colorize_part_ids(physical_groups.group_map).save(
        output_dir / "group_id_preview.png"
    )
    render_source_overlay(image, physical_groups.group_map).save(
        output_dir / "group_overlay.png"
    )
    (output_dir / "groups.json").write_text(
        json.dumps([group.to_dict() for group in physical_groups.groups], indent=2),
        encoding="utf-8",
    )
    payload: list[dict[str, object]] = []
    source_rgba = image.convert("RGBA")
    source_array = np.asarray(source_rgba).copy()
    for record in records:
        mask = prediction.instance_map == record.instance_index
        stem = f"{record.instance_index:04d}_{_safe_name(record.part_id)}"
        mask_path = masks_dir / f"{stem}.png"
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
        x0, y0, x1, y1 = record.bbox_xyxy
        crop_array = source_array[y0:y1, x0:x1].copy()
        crop_array[..., 3] = mask[y0:y1, x0:x1].astype(np.uint8) * 255
        crop_path = crops_dir / f"{stem}.png"
        Image.fromarray(crop_array, mode="RGBA").save(crop_path)
        item = record.to_dict()
        item.update(
            {
                "bbox_visible": list(record.bbox_xyxy),
                "crop_offset": [x0, y0],
                "mask_visible_path": mask_path.relative_to(output_dir).as_posix(),
                "crop_path": crop_path.relative_to(output_dir).as_posix(),
            }
        )
        if completion_records is not None:
            item.update(completion_records.get(record.instance_index, {}))
        payload.append(item)
    (output_dir / "parts.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    taxonomy.to_json(output_dir / "taxonomy.json")
    if diagnostics is not None:
        (output_dir / "inference_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2), encoding="utf-8"
        )
    if candidates is not None:
        _export_candidate_audit(output_dir, candidates)
    quality_report = assess_product_quality(
        prediction.instance_map,
        records,
        diagnostics,
    )
    (output_dir / "quality_report.json").write_text(
        json.dumps(quality_report, indent=2), encoding="utf-8"
    )
    target_candidates = render_target_candidates(image, quality_report)
    target_candidates_path = None
    if target_candidates is not None:
        target_candidates_path = "target_candidates.png"
        target_candidates.save(output_dir / target_candidates_path)
    files = sorted(path for path in output_dir.rglob("*") if path.is_file())
    manifest = {
        "format": "HPID-Split package",
        "format_version": "0.3.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_size": list(image.size),
        "part_count": len(records),
        "group_count": len(physical_groups.groups),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "inference_uses_ground_truth": False,
        "algorithm": _algorithm_metadata(diagnostics, checkpoint),
        "diagnostics_path": (
            "inference_diagnostics.json" if diagnostics is not None else None
        ),
        "candidate_audit_path": ("candidates.json" if candidates is not None else None),
        "quality_report_path": "quality_report.json",
        "part_id_preview_path": "part_id_preview.png",
        "source_overlay_path": "source_overlay.png",
        "group_id_preview_path": "group_id_preview.png",
        "group_overlay_path": "group_overlay.png",
        "target_candidates_path": target_candidates_path,
        "quality_status": quality_report["status"],
        "evidence_grade": quality_report["evidence_grade"],
        "files": [
            {
                "path": path.relative_to(output_dir).as_posix(),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in files
        ],
    }
    (output_dir / "package_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest
