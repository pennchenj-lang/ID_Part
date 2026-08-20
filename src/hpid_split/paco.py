from __future__ import annotations

import hashlib
import json
import math
import time
import urllib.request
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PacoCase:
    object_annotation: dict[str, Any]
    image_record: dict[str, Any]
    object_category: str
    part_annotations: tuple[dict[str, Any], ...]
    part_categories: tuple[str, ...]

    @property
    def object_annotation_id(self) -> int:
        return int(self.object_annotation["id"])

    @property
    def image_id(self) -> int:
        return int(self.image_record["id"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode_compressed_rle(value: str) -> list[int]:
    counts: list[int] = []
    position = 0
    while position < len(value):
        number = 0
        shift = 0
        more = True
        while more:
            code = ord(value[position]) - 48
            position += 1
            number |= (code & 0x1F) << (5 * shift)
            more = bool(code & 0x20)
            if not more and code & 0x10:
                number |= -1 << (5 * (shift + 1))
            shift += 1
        if len(counts) > 2:
            number += counts[-2]
        if number < 0:
            raise ValueError("COCO RLE contains a negative run length")
        counts.append(number)
    return counts


def _decode_rle_counts(
    counts: Sequence[int], height: int, width: int
) -> np.ndarray:
    size = height * width
    flat = np.zeros(size, dtype=np.uint8)
    position = 0
    foreground = False
    for raw_count in counts:
        count = int(raw_count)
        if count < 0 or position + count > size:
            raise ValueError("COCO RLE run lengths exceed the declared mask size")
        if foreground and count:
            flat[position : position + count] = 1
        position += count
        foreground = not foreground
    if position != size:
        raise ValueError(
            f"COCO RLE decodes to {position} pixels, expected {size}"
        )
    return flat.reshape((height, width), order="F").astype(bool)


def decode_coco_segmentation(
    segmentation: object, height: int, width: int
) -> np.ndarray:
    """Decode polygon, compressed RLE, or uncompressed RLE without pycocotools."""

    if isinstance(segmentation, dict):
        declared_size = segmentation.get("size", [height, width])
        if [int(value) for value in declared_size] != [height, width]:
            raise ValueError("COCO RLE size does not match its image record")
        counts = segmentation.get("counts")
        if isinstance(counts, str):
            return _decode_rle_counts(
                _decode_compressed_rle(counts), height, width
            )
        if isinstance(counts, list):
            return _decode_rle_counts(counts, height, width)
        raise TypeError("unsupported COCO RLE counts payload")
    if not isinstance(segmentation, list):
        raise TypeError("unsupported COCO segmentation payload")
    mask = np.zeros((height, width), dtype=np.uint8)
    polygons: list[np.ndarray] = []
    for raw_polygon in segmentation:
        if not isinstance(raw_polygon, list) or len(raw_polygon) < 6:
            continue
        coordinates = np.asarray(raw_polygon, dtype=np.float32).reshape(-1, 2)
        coordinates[:, 0] = np.clip(coordinates[:, 0], 0, width - 1)
        coordinates[:, 1] = np.clip(coordinates[:, 1], 0, height - 1)
        polygons.append(np.rint(coordinates).astype(np.int32))
    if polygons:
        cv2.fillPoly(mask, polygons, color=1)
    return mask.astype(bool)


def load_paco_cases(
    annotation_path: Path,
    categories: Sequence[str],
    *,
    minimum_distinct_parts: int = 2,
    minimum_bbox_side: int = 64,
    alternatives_per_category: int = 8,
) -> dict[str, list[PacoCase]]:
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    category_by_id = {
        int(row["id"]): str(row["name"]) for row in payload["categories"]
    }
    image_by_id = {int(row["id"]): row for row in payload["images"]}
    parts_by_object: dict[int, list[dict[str, Any]]] = defaultdict(list)
    object_annotations: list[dict[str, Any]] = []
    for annotation in payload["annotations"]:
        category = category_by_id[int(annotation["category_id"])]
        if ":" in category:
            parts_by_object[int(annotation["obj_ann_id"])].append(annotation)
        else:
            object_annotations.append(annotation)

    requested = set(categories)
    ranked: dict[str, list[tuple[tuple[float, ...], PacoCase]]] = defaultdict(list)
    for annotation in object_annotations:
        category = category_by_id[int(annotation["category_id"])]
        if category not in requested:
            continue
        image = image_by_id[int(annotation["image_id"])]
        parts = parts_by_object.get(int(annotation["id"]), [])
        part_categories = tuple(
            category_by_id[int(part["category_id"])] for part in parts
        )
        distinct_parts = len(set(part_categories))
        x, y, width, height = (float(value) for value in annotation["bbox"])
        if (
            distinct_parts < minimum_distinct_parts
            or min(width, height) < minimum_bbox_side
        ):
            continue
        image_width = max(1.0, float(image["width"]))
        image_height = max(1.0, float(image["height"]))
        area_fraction = width * height / (image_width * image_height)
        center_x = (x + width / 2.0) / image_width
        center_y = (y + height / 2.0) / image_height
        center_score = 1.0 - min(
            1.0, math.hypot(center_x - 0.5, center_y - 0.5) / math.sqrt(0.5)
        )
        case = PacoCase(
            object_annotation=annotation,
            image_record=image,
            object_category=category,
            part_annotations=tuple(parts),
            part_categories=part_categories,
        )
        score = (
            float(distinct_parts),
            float(len(parts)),
            min(area_fraction, 0.50),
            center_score,
            float(width * height),
        )
        ranked[category].append((score, case))
    result: dict[str, list[PacoCase]] = {}
    for category in categories:
        rows = sorted(ranked.get(category, []), key=lambda item: item[0], reverse=True)
        result[category] = [case for _, case in rows[:alternatives_per_category]]
    return result


def _download(urls: Sequence[str], output: Path, retries: int = 2) -> str:
    output.parent.mkdir(parents=True, exist_ok=True)
    last_error: OSError | None = None
    for raw_url in urls:
        if not raw_url:
            continue
        url = raw_url.replace("http://", "https://", 1)
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "HPID-Split public benchmark adapter/0.1"},
        )
        for attempt in range(retries):
            try:
                with urllib.request.urlopen(request, timeout=20) as response:
                    output.write_bytes(response.read())
                return url
            except OSError as error:
                last_error = error
                output.unlink(missing_ok=True)
                if attempt + 1 < retries:
                    time.sleep(1.0 * (attempt + 1))
    assert last_error is not None
    raise last_error


def _crop_box(
    bbox: Sequence[float], image_size: tuple[int, int], padding: float
) -> tuple[int, int, int, int]:
    x, y, width, height = (float(value) for value in bbox)
    image_width, image_height = image_size
    return (
        max(0, math.floor(x - padding * width)),
        max(0, math.floor(y - padding * height)),
        min(image_width, math.ceil(x + width * (1.0 + padding))),
        min(image_height, math.ceil(y + height * (1.0 + padding))),
    )


def materialize_paco_case(
    case: PacoCase,
    output: Path,
    *,
    annotation_sha256: str,
    crop_padding: float = 0.10,
    dataset_split: str = "unspecified",
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    raw_image = output / "source_download.jpg"
    source_urls = [
        str(value)
        for value in (
            case.image_record.get("flickr_url"),
            case.image_record.get("coco_url"),
        )
        if value
    ]
    source_url = _download(
        source_urls,
        raw_image,
    )
    with Image.open(raw_image) as opened:
        image = opened.convert("RGB")
    expected_size = (
        int(case.image_record["width"]),
        int(case.image_record["height"]),
    )
    if image.size != expected_size:
        raise ValueError(
            f"downloaded image size {image.size} differs from {expected_size}"
        )
    image.save(output / "source_full.png")
    height, width = image.height, image.width
    object_mask = decode_coco_segmentation(
        case.object_annotation["segmentation"], height, width
    )
    if not object_mask.any():
        raise ValueError("selected PACO object has an empty mask")
    crop_box = _crop_box(case.object_annotation["bbox"], image.size, crop_padding)
    x0, y0, x1, y1 = crop_box
    image.crop(crop_box).save(output / "source_crop.png")
    Image.fromarray(object_mask.astype(np.uint8) * 255, mode="L").save(
        output / "object_mask_full.png"
    )
    Image.fromarray(
        object_mask[y0:y1, x0:x1].astype(np.uint8) * 255, mode="L"
    ).save(output / "object_mask_crop.png")

    part_rows: list[dict[str, object]] = []
    for index, (annotation, category) in enumerate(
        zip(case.part_annotations, case.part_categories, strict=True), start=1
    ):
        mask = decode_coco_segmentation(annotation["segmentation"], height, width)
        mask &= object_mask
        if not mask.any():
            continue
        part_name = category.split(":", maxsplit=1)[1]
        safe_name = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in part_name
        )
        full_relative = f"parts_full/{index:03d}_{safe_name}.png"
        crop_relative = f"parts_crop/{index:03d}_{safe_name}.png"
        (output / "parts_full").mkdir(exist_ok=True)
        (output / "parts_crop").mkdir(exist_ok=True)
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
            output / full_relative
        )
        Image.fromarray(
            mask[y0:y1, x0:x1].astype(np.uint8) * 255, mode="L"
        ).save(output / crop_relative)
        part_rows.append(
            {
                "annotation_id": int(annotation["id"]),
                "category": category,
                "part_name": part_name,
                "area_px": int(np.count_nonzero(mask)),
                "mask_full": full_relative,
                "mask_crop": crop_relative,
            }
        )
    if not part_rows:
        raise ValueError("selected PACO object has no decodable part masks")
    raw_image.unlink(missing_ok=True)
    payload: dict[str, object] = {
        "format": "HPID PACO benchmark case",
        "format_version": "0.1.0",
        "dataset": f"PACO-LVIS v1 {dataset_split}",
        "dataset_repository": "https://github.com/facebookresearch/paco",
        "source_annotation_sha256": annotation_sha256,
        "source_image_url": source_url,
        "source_image_urls_from_annotation": source_urls,
        "source_image_sha256": sha256_file(output / "source_full.png"),
        "image_id": case.image_id,
        "object_annotation_id": case.object_annotation_id,
        "object_category": case.object_category,
        "crop_box_xyxy": list(crop_box),
        "crop_uses_ground_truth_bbox": True,
        "full_image_uses_ground_truth_at_inference": False,
        "part_count": len(part_rows),
        "parts": part_rows,
    }
    (output / "case.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload
