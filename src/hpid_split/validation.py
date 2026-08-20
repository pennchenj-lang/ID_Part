from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version_at_least(value: str, minimum: tuple[int, int, int]) -> bool:
    try:
        parts = tuple(int(part) for part in value.split(".")[:3])
    except ValueError:
        return False
    return (*parts, 0, 0, 0)[:3] >= minimum


def _validate_manifest_files(
    package_dir: Path,
    manifest: dict[str, object],
) -> list[str]:
    """Verify the manifest against every exported payload file."""

    rows = manifest.get("files")
    if not isinstance(rows, list):
        return ["manifest payload file list is missing"]
    root = package_dir.resolve()
    listed_paths: set[str] = set()
    errors: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            errors.append("manifest contains a malformed file record")
            continue
        relative = str(row.get("path", "")).replace("\\", "/")
        if not relative or relative == "package_manifest.json":
            errors.append(f"manifest contains an invalid payload path: {relative!r}")
            continue
        if relative in listed_paths:
            errors.append(f"manifest contains a duplicate payload path: {relative}")
            continue
        listed_paths.add(relative)
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            errors.append(f"manifest payload path escapes package: {relative}")
            continue
        if not path.is_file():
            errors.append(f"manifest payload file is missing: {relative}")
            continue
        expected_bytes = row.get("bytes")
        if not isinstance(expected_bytes, int) or path.stat().st_size != expected_bytes:
            errors.append(f"manifest byte count differs: {relative}")
        expected_sha256 = str(row.get("sha256", ""))
        if len(expected_sha256) != 64 or _sha256(path) != expected_sha256:
            errors.append(f"manifest SHA-256 differs: {relative}")

    actual_paths = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "package_manifest.json"
    }
    for relative in sorted(actual_paths - listed_paths):
        errors.append(f"payload file is absent from manifest: {relative}")
    return errors


def validate_package(package_dir: Path) -> dict[str, object]:
    errors: list[str] = []
    required = (
        "source.png",
        "part_id_map.tiff",
        "semantic_ids.png",
        "parts.json",
        "taxonomy.json",
        "package_manifest.json",
    )
    for relative in required:
        if not (package_dir / relative).is_file():
            errors.append(f"missing required file: {relative}")
    if errors:
        return {"valid": False, "errors": errors, "checked_parts": 0}

    source = np.asarray(Image.open(package_dir / "source.png").convert("RGBA"))
    instance_map = np.asarray(
        Image.open(package_dir / "part_id_map.tiff"), dtype=np.uint16
    )
    if instance_map.shape != source.shape[:2]:
        errors.append("part_id_map dimensions do not match source")
    parts = json.loads((package_dir / "parts.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (package_dir / "package_manifest.json").read_text(encoding="utf-8")
    )
    errors.extend(_validate_manifest_files(package_dir, manifest))
    format_version = str(manifest.get("format_version", "0.0.0"))
    if _version_at_least(format_version, (0, 2, 0)):
        for relative in (
            "part_id_preview.png",
            "source_overlay.png",
            "quality_report.json",
        ):
            if not (package_dir / relative).is_file():
                errors.append(f"missing product output: {relative}")
        for relative in ("part_id_preview.png", "source_overlay.png"):
            path = package_dir / relative
            if path.is_file():
                preview = np.asarray(Image.open(path).convert("RGB"))
                if preview.shape[:2] != instance_map.shape:
                    errors.append(f"{relative} dimensions do not match source")
        quality_path = package_dir / "quality_report.json"
        if quality_path.is_file():
            quality = json.loads(quality_path.read_text(encoding="utf-8"))
            if quality.get("ground_truth_used") is not False:
                errors.append("quality report does not confirm zero-GT inference")
            if quality.get("status") != manifest.get("quality_status"):
                errors.append("quality status differs between report and manifest")
            if quality.get("evidence_grade") != manifest.get("evidence_grade"):
                errors.append("evidence grade differs between report and manifest")
        target_candidates = manifest.get("target_candidates_path")
        if target_candidates is not None and not (
            package_dir / str(target_candidates)
        ).is_file():
            errors.append("target candidate review image is missing")
    algorithm = manifest.get("algorithm")
    if not isinstance(algorithm, dict):
        errors.append("manifest algorithm provenance is missing")
    else:
        if algorithm.get("name") != "HPID-Split" or not algorithm.get("version"):
            errors.append("manifest algorithm identity is invalid")
        if algorithm.get("ground_truth_used") is not False:
            errors.append("manifest algorithm provenance does not confirm zero-GT inference")
    if manifest.get("inference_uses_ground_truth") is not False:
        errors.append("manifest does not confirm zero-GT inference")
    if len(parts) != int(manifest.get("part_count", -1)):
        errors.append("part count differs between parts.json and manifest")
    part_ids = [str(part.get("part_id")) for part in parts]
    asset_ids = [str(part.get("asset_id", "")) for part in parts]
    indices = [int(part.get("instance_index", -1)) for part in parts]
    if len(part_ids) != len(set(part_ids)):
        errors.append("part_id values are not unique")
    invalid_asset_ids = {"", "none", "null"}
    if _version_at_least(format_version, (0, 2, 0)) and any(
        value.strip().casefold() in invalid_asset_ids for value in asset_ids
    ):
        errors.append("asset_id values are missing")
    if len(indices) != len(set(indices)) or any(index <= 0 for index in indices):
        errors.append("instance_index values are invalid or duplicated")
    known_ids = set(part_ids)
    asset_by_part_id = dict(zip(part_ids, asset_ids, strict=True))

    candidate_path = package_dir / "candidates.json"
    if candidate_path.is_file():
        candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidate_indices = [
            int(candidate.get("candidate_index", -1)) for candidate in candidates
        ]
        if len(candidate_indices) != len(set(candidate_indices)) or any(
            index <= 0 for index in candidate_indices
        ):
            errors.append("candidate indices are invalid or duplicated")
        for candidate in candidates:
            index = int(candidate.get("candidate_index", -1))
            mask_path = package_dir / str(candidate.get("mask_path", ""))
            if not mask_path.is_file():
                errors.append(f"candidate {index}: audit mask is missing")
                continue
            mask = np.asarray(Image.open(mask_path).convert("L")) >= 128
            if mask.shape != instance_map.shape:
                errors.append(f"candidate {index}: audit mask has wrong dimensions")
                continue
            if int(np.count_nonzero(mask)) != int(candidate.get("area_px", -1)):
                errors.append(f"candidate {index}: audit area is inconsistent")

    for part in parts:
        index = int(part["instance_index"])
        visible_path = package_dir / str(part["mask_visible_path"])
        if not visible_path.is_file():
            errors.append(f"part {index}: visible mask is missing")
            continue
        visible = np.asarray(Image.open(visible_path).convert("L")) >= 128
        if visible.shape != instance_map.shape:
            errors.append(f"part {index}: visible mask has wrong dimensions")
            continue
        if not np.array_equal(visible, instance_map == index):
            errors.append(f"part {index}: visible mask disagrees with part_id_map")
        parent_id = part.get("assembly_parent_id")
        if parent_id is not None and str(parent_id) not in known_ids:
            errors.append(f"part {index}: assembly_parent_id is unresolved")
        if (
            _version_at_least(format_version, (0, 2, 0))
            and parent_id is not None
            and asset_by_part_id.get(str(parent_id)) != str(part.get("asset_id", ""))
        ):
            errors.append(f"part {index}: assembly parent belongs to another asset")

        full_relative = part.get("mask_full_path")
        completed_relative = part.get("crop_completed_path")
        if full_relative is None and completed_relative is None:
            continue
        if full_relative is None or completed_relative is None:
            errors.append(f"part {index}: completion paths are incomplete")
            continue
        full_path = package_dir / str(full_relative)
        completed_path = package_dir / str(completed_relative)
        if not full_path.is_file() or not completed_path.is_file():
            errors.append(f"part {index}: completion output is missing")
            continue
        full = np.asarray(Image.open(full_path).convert("L")) >= 128
        if full.shape != visible.shape or np.any(visible & ~full):
            errors.append(f"part {index}: visible mask is not a subset of full mask")
            continue
        x0, y0, x1, y1 = (int(value) for value in part["bbox_full"])
        completed = np.asarray(Image.open(completed_path).convert("RGBA"))
        expected_shape = (y1 - y0, x1 - x0)
        if completed.shape[:2] != expected_shape:
            errors.append(f"part {index}: completed crop has wrong dimensions")
            continue
        local_visible = visible[y0:y1, x0:x1]
        local_source = source[y0:y1, x0:x1]
        if not np.array_equal(completed[local_visible], local_source[local_visible]):
            errors.append(f"part {index}: visible-lock pixels drifted")

    checked_groups = 0
    if _version_at_least(format_version, (0, 3, 0)):
        group_required = (
            "group_id_map.tiff",
            "groups.json",
            "group_id_preview.png",
            "group_overlay.png",
        )
        for relative in group_required:
            if not (package_dir / relative).is_file():
                errors.append(f"missing physical-group output: {relative}")
        group_map_path = package_dir / "group_id_map.tiff"
        groups_path = package_dir / "groups.json"
        if group_map_path.is_file() and groups_path.is_file():
            group_map = np.asarray(Image.open(group_map_path), dtype=np.uint16)
            groups = json.loads(groups_path.read_text(encoding="utf-8"))
            checked_groups = len(groups)
            if group_map.shape != instance_map.shape:
                errors.append("group_id_map dimensions do not match source")
            elif not np.array_equal(group_map > 0, instance_map > 0):
                errors.append("group_id_map foreground differs from part_id_map")
            group_indices = [int(group.get("group_index", -1)) for group in groups]
            group_ids = [str(group.get("group_id", "")) for group in groups]
            if (
                len(group_indices) != len(set(group_indices))
                or any(index <= 0 for index in group_indices)
            ):
                errors.append("group_index values are invalid or duplicated")
            if len(group_ids) != len(set(group_ids)) or any(not value for value in group_ids):
                errors.append("group_id values are missing or duplicated")
            if group_map.shape == instance_map.shape:
                map_indices = {int(value) for value in np.unique(group_map) if value > 0}
                if map_indices != set(group_indices):
                    errors.append("group records disagree with group_id_map")
            if len(groups) != int(manifest.get("group_count", -1)):
                errors.append("group count differs between groups.json and manifest")
            known_group_ids = set(group_ids)
            for part in parts:
                group_id = str(part.get("group_id", ""))
                if group_id not in known_group_ids:
                    errors.append(
                        f"part {part.get('instance_index')}: group_id is unresolved"
                    )
            part_id_set = set(part_ids)
            for group in groups:
                members = {str(value) for value in group.get("member_part_ids", [])}
                if not members or not members <= part_id_set:
                    errors.append(
                        f"group {group.get('group_index')}: member Part IDs are invalid"
                    )
        for relative in ("group_id_preview.png", "group_overlay.png"):
            path = package_dir / relative
            if path.is_file():
                preview = np.asarray(Image.open(path).convert("RGB"))
                if preview.shape[:2] != instance_map.shape:
                    errors.append(f"{relative} dimensions do not match source")

    for entry in manifest.get("files", []):
        path = package_dir / str(entry["path"])
        if not path.is_file():
            errors.append(f"manifest file is missing: {entry['path']}")
            continue
        if _sha256(path) != entry["sha256"]:
            errors.append(f"manifest hash mismatch: {entry['path']}")
        if path.stat().st_size != int(entry["bytes"]):
            errors.append(f"manifest size mismatch: {entry['path']}")

    return {
        "valid": not errors,
        "errors": errors,
        "checked_parts": len(parts),
        "checked_groups": checked_groups,
        "completed_parts": sum("mask_full_path" in part for part in parts),
        "audited_candidates": (
            len(json.loads(candidate_path.read_text(encoding="utf-8")))
            if candidate_path.is_file()
            else 0
        ),
        "ground_truth_used": False,
    }
