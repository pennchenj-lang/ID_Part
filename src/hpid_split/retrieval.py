from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import torch
from PIL import Image

from .data import load_label_map
from .fusion import MaskCandidate
from .guided_prompts import GuidedPromptSpec
from .paco_semantics import normalize_paco_name
from .prompt_bank import PartPrompt, PromptBank
from .taxonomy import Taxonomy

INDEX_FORMAT = "HPID reviewed prototype index"
INDEX_VERSION = "0.1.0"
REFERENCE_FORMAT = "HPID reviewed reference set"
_GENERIC_PART = re.compile(r"(?:^|_)(?:visual|asset_\d+)(?:_|$)")


class EmbeddingEncoder(Protocol):
    model_name: str

    def encode(self, images: Sequence[Image.Image]) -> np.ndarray: ...


@dataclass(frozen=True)
class RetrievalConfig:
    top_k_assets: int = 5
    maximum_part_prompts: int = 28
    minimum_asset_similarity: float = 0.58
    minimum_prompted_asset_similarity: float = 0.32
    minimum_profiled_asset_similarity: float = 0.50
    minimum_asset_label_margin: float = 0.025
    maximum_similarity_drop: float = 0.12
    profile_similarity_bonus: float = 0.035
    minimum_part_prevalence: float = 0.20
    minimum_part_support_count: int = 2
    minimum_part_similarity: float = 0.42
    minimum_raw_part_similarity: float = 0.58
    minimum_geometry_compatibility: float = 0.16
    prototype_label_margin: float = 0.025
    prototype_duplicate_iou: float = 0.72
    prototype_duplicate_containment: float = 0.92
    domain_relabel_similarity: float = 0.76
    domain_relabel_margin: float = 0.10
    domain_relabel_minimum_support: int = 2
    allow_domain_relabel: bool = False


@dataclass(frozen=True)
class RetrievedPartPrior:
    semantic_name: str
    output_semantic_name: str
    display_name: str
    phrases: tuple[str, ...]
    semantic_parent: str
    assembly_parent_semantic: str | None
    maximum_instances: int
    support_count: int
    prevalence: float
    retrieval_score: float
    prototype_indices: tuple[int, ...]
    geometry_mean: tuple[float, float, float, float, float]
    geometry_std: tuple[float, float, float, float, float]
    geometry_samples: tuple[tuple[float, float, float, float, float], ...]

    @property
    def slug(self) -> str:
        value = re.sub(r"[^a-z0-9]+", "_", self.output_semantic_name.lower())
        return value.strip("_")[:64]

    def guided_spec(self) -> GuidedPromptSpec:
        return GuidedPromptSpec(
            label=self.display_name,
            slug=self.slug,
            phrases=self.phrases,
            maximum_instances=self.maximum_instances,
        )


@dataclass(frozen=True)
class RootRetrievalPlan:
    root_key: str
    root_semantic: str
    accepted: bool
    reason: str
    top_similarity: float
    asset_label: str | None
    asset_domain: str | None
    asset_label_margin: float
    supporting_asset_count: int
    nearest_assets: tuple[dict[str, object], ...]
    part_priors: tuple[RetrievedPartPrior, ...]


@dataclass(frozen=True)
class RetrievalResult:
    plans: tuple[RootRetrievalPlan, ...]
    diagnostics: dict[str, object]


def _merge_hierarchical_part_priors(
    profile_priors: Sequence[RetrievedPartPrior],
    inventory_priors: Sequence[RetrievedPartPrior],
) -> tuple[RetrievedPartPrior, ...]:
    """Keep subtype semantics while adding geometry modes from nearby assets."""

    merged = {prior.output_semantic_name: prior for prior in profile_priors}
    for fallback in inventory_priors:
        primary = merged.get(fallback.output_semantic_name)
        if primary is None:
            merged[fallback.output_semantic_name] = fallback
            continue
        geometry_samples = tuple(
            dict.fromkeys((*primary.geometry_samples, *fallback.geometry_samples))
        )
        geometry = np.asarray(geometry_samples, dtype=np.float32)
        geometry_std = np.maximum(geometry.std(axis=0), 0.04)
        merged[fallback.output_semantic_name] = replace(
            primary,
            phrases=tuple(dict.fromkeys((*primary.phrases, *fallback.phrases)))[:8],
            support_count=max(primary.support_count, fallback.support_count),
            prevalence=max(primary.prevalence, fallback.prevalence),
            retrieval_score=max(primary.retrieval_score, fallback.retrieval_score),
            prototype_indices=tuple(
                dict.fromkeys((*primary.prototype_indices, *fallback.prototype_indices))
            ),
            geometry_mean=tuple(float(value) for value in geometry.mean(axis=0)),
            geometry_std=tuple(float(value) for value in geometry_std),
            geometry_samples=geometry_samples,
        )
    return tuple(
        sorted(
            merged.values(),
            key=lambda prior: (
                prior.retrieval_score,
                prior.support_count,
                len(prior.prototype_indices),
            ),
            reverse=True,
        )
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _combined_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((item.resolve() for item in paths), key=str):
        digest.update(str(path).encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _mask_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0, 0, 0, 0
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _masked_view(
    image: Image.Image,
    mask: np.ndarray,
    *,
    padding: float = 0.10,
) -> Image.Image:
    image = image.convert("RGB")
    if mask.shape != (image.height, image.width):
        raise ValueError("reference mask size does not match its source image")
    x0, y0, x1, y1 = _mask_box(mask)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("cannot encode an empty mask")
    pad_x = max(2, round((x1 - x0) * padding))
    pad_y = max(2, round((y1 - y0) * padding))
    x0, y0 = max(0, x0 - pad_x), max(0, y0 - pad_y)
    x1, y1 = min(image.width, x1 + pad_x), min(image.height, y1 + pad_y)
    rgb = np.asarray(image, dtype=np.uint8)[y0:y1, x0:x1]
    local_mask = mask[y0:y1, x0:x1]
    background = np.full_like(rgb, 127)
    composited = np.where(local_mask[..., None], rgb, background)
    height, width = composited.shape[:2]
    side = max(height, width)
    square = np.full((side, side, 3), 127, dtype=np.uint8)
    offset_y = (side - height) // 2
    offset_x = (side - width) // 2
    square[offset_y : offset_y + height, offset_x : offset_x + width] = composited
    return Image.fromarray(square, mode="RGB")


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def _apply_metric(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return _normalize_rows(values * weights[None, :])


class CLIPSegEmbeddingEncoder:
    """Frozen CLIPSeg vision encoder shared with the dense semantic backend."""

    def __init__(
        self,
        *,
        model_name: str = "CIDAS/clipseg-rd64-refined",
        device: str = "cuda",
        local_files_only: bool = False,
        batch_size: int = 16,
        processor: object | None = None,
        model: torch.nn.Module | None = None,
    ) -> None:
        try:
            from transformers import (
                CLIPSegForImageSegmentation,
                CLIPSegProcessor,
            )
        except ImportError as error:
            raise RuntimeError(
                "Prototype retrieval requires the foundation extra: "
                "pip install -e '.[foundation]'"
            ) from error
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.processor = processor or CLIPSegProcessor.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.model = model or CLIPSegForImageSegmentation.from_pretrained(
            model_name, local_files_only=local_files_only
        ).to(device)
        self.model.eval()

    def encode(self, images: Sequence[Image.Image]) -> np.ndarray:
        if not images:
            return np.zeros((0, 0), dtype=np.float32)
        chunks: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            batch = list(images[start : start + self.batch_size])
            pixel_values = self.processor(images=batch, return_tensors="pt")[
                "pixel_values"
            ].to(self.device)
            with (
                torch.inference_mode(),
                torch.amp.autocast("cuda", enabled=str(self.device).startswith("cuda")),
            ):
                vision = self.model.clip.vision_model(pixel_values=pixel_values)
                features = self.model.clip.visual_projection(vision.pooler_output)
                features = torch.nn.functional.normalize(features.float(), dim=1)
            chunks.append(features.cpu().numpy().astype(np.float32))
        return np.concatenate(chunks, axis=0)


def _fit_metric_weights(
    embeddings: np.ndarray,
    labels: Sequence[str],
    groups: Sequence[int],
    *,
    epochs: int,
    device: str,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    dimension = int(embeddings.shape[1])
    label_lookup = {name: index for index, name in enumerate(sorted(set(labels)))}
    label_ids = np.asarray(
        [label_lookup[name] for name in labels],
        dtype=np.int64,
    )
    group_ids = np.asarray(groups, dtype=np.int64)
    positive = (label_ids[:, None] == label_ids[None, :]) & (
        group_ids[:, None] != group_ids[None, :]
    )
    negative = label_ids[:, None] != label_ids[None, :]
    valid = positive.any(axis=1) & negative.any(axis=1)
    if int(np.count_nonzero(valid)) < 4 or epochs <= 0:
        return np.ones(dimension, dtype=np.float32), {
            "trained": False,
            "reason": "insufficient repeated labels across independent assets",
            "valid_anchor_count": int(np.count_nonzero(valid)),
        }

    torch.manual_seed(seed)
    x = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    positive_t = torch.as_tensor(positive, dtype=torch.bool, device=device)
    negative_t = torch.as_tensor(negative, dtype=torch.bool, device=device)
    valid_t = torch.as_tensor(valid, dtype=torch.bool, device=device)
    logits = torch.nn.Parameter(torch.zeros(dimension, device=device))
    optimizer = torch.optim.AdamW([logits], lr=0.035, weight_decay=0.0)
    first_loss = None
    final_loss = None
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        weights = torch.nn.functional.softplus(logits) + 1e-4
        projected = torch.nn.functional.normalize(x * weights[None, :], dim=1)
        similarity = projected @ projected.T
        hardest_positive = similarity.masked_fill(~positive_t, 2.0).min(dim=1).values
        hardest_negative = similarity.masked_fill(~negative_t, -2.0).max(dim=1).values
        triplet = torch.relu(0.08 + hardest_negative - hardest_positive)
        regularizer = 0.01 * logits.square().mean()
        loss = triplet[valid_t].mean() + regularizer
        if first_loss is None:
            first_loss = float(loss.detach().cpu())
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    learned = (torch.nn.functional.softplus(logits) + 1e-4).detach().cpu().numpy()
    learned = np.clip(learned / max(1e-8, float(learned.mean())), 0.15, 6.0)
    return learned.astype(np.float32), {
        "trained": True,
        "valid_anchor_count": int(np.count_nonzero(valid)),
        "epochs": epochs,
        "seed": seed,
        "initial_loss": first_loss,
        "final_loss": final_loss,
    }


def _humanize(name: str, domain: str) -> str:
    prefix = f"{domain}_"
    value = name.removeprefix(prefix)
    return re.sub(r"[_-]+", " ", value).strip()


def _clean_phrases(values: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = re.sub(r"\s+", " ", str(value)).strip()
        key = cleaned.casefold()
        if cleaned and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return tuple(result)


def _phrase_tokens(value: str) -> frozenset[str]:
    normalized = normalize_paco_name(value)
    return frozenset(token for token in normalized.split("_") if token)


def _asset_hint_matches(asset: dict[str, object], hint: str) -> bool:
    hint_tokens = _phrase_tokens(hint)
    if not hint_tokens:
        return False
    phrases = (
        str(asset.get("asset_label", "")),
        *(str(value) for value in asset.get("aliases", [])),
    )
    for phrase in phrases:
        phrase_tokens = _phrase_tokens(phrase)
        if not phrase_tokens:
            continue
        if phrase_tokens == hint_tokens or phrase_tokens.issubset(hint_tokens):
            return True
        if len(hint_tokens) >= 2 and hint_tokens.issubset(phrase_tokens):
            return True
    return False


def _geometry(mask: np.ndarray, root_mask: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = _mask_box(root_mask)
    px0, py0, px1, py1 = _mask_box(mask)
    width = max(1, x1 - x0)
    height = max(1, y1 - y0)
    root_area = max(1, int(np.count_nonzero(root_mask)))
    return np.asarray(
        [
            ((px0 + px1) / 2 - x0) / width,
            ((py0 + py1) / 2 - y0) / height,
            (px1 - px0) / width,
            (py1 - py0) / height,
            int(np.count_nonzero(mask)) / root_area,
        ],
        dtype=np.float32,
    )


def _package_parts(
    package: Path,
) -> tuple[Image.Image, np.ndarray, list[dict[str, object]], list[Path]]:
    source_path = package / "source.png"
    parts_path = package / "parts.json"
    if not source_path.is_file() or not parts_path.is_file():
        raise ValueError(f"not an HPID package: {package}")
    image = Image.open(source_path).convert("RGB")
    payload = json.loads(parts_path.read_text(encoding="utf-8"))
    by_part_id = {str(item["part_id"]): item for item in payload}
    records: list[dict[str, object]] = []
    union = np.zeros((image.height, image.width), dtype=bool)
    audit_paths = [source_path, parts_path]
    for item in payload:
        mask_path = package / str(item["mask_visible_path"])
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
        union |= mask
        audit_paths.append(mask_path)
        semantic_name = str(item["semantic_name"])
        semantic_parent = str(item["semantic_parent"])
        if semantic_name == semantic_parent and item.get("assembly_parent_id") is None:
            continue
        assembly_parent_id = item.get("assembly_parent_id")
        assembly_parent = None
        if assembly_parent_id is not None:
            parent = by_part_id.get(str(assembly_parent_id))
            if parent is not None:
                assembly_parent = str(parent["semantic_name"])
        records.append(
            {
                "part_id": str(item["part_id"]),
                "semantic_name": semantic_name,
                "semantic_parent": semantic_parent,
                "assembly_parent_semantic": assembly_parent,
                "side": str(item.get("side", "center")),
                "mask": mask,
                "geometry_masks": [mask],
            }
        )
    return image, union, records, audit_paths


def _label_map_parts(
    image_path: Path,
    label_map_path: Path,
    taxonomy_path: Path,
) -> tuple[Image.Image, np.ndarray, list[dict[str, object]], list[Path]]:
    image = Image.open(image_path).convert("RGB")
    taxonomy = Taxonomy.from_json(taxonomy_path)
    labels = load_label_map(label_map_path, taxonomy)
    if labels.shape != (image.height, image.width):
        raise ValueError(f"label map size does not match image: {label_map_path}")
    records: list[dict[str, object]] = []
    for class_id in range(1, taxonomy.num_fine_classes):
        mask = labels == class_id
        if int(np.count_nonzero(mask)) < 12:
            continue
        semantic_name = taxonomy.fine_names[class_id]
        semantic_parent = taxonomy.parent_names[taxonomy.fine_to_parent[class_id]]
        count, components, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), 8
        )
        records.append(
            {
                "part_id": f"prototype/{semantic_name}",
                "semantic_name": semantic_name,
                "semantic_parent": semantic_parent,
                "assembly_parent_semantic": semantic_parent,
                "side": "aggregate",
                "mask": mask,
                "geometry_masks": [
                    components == component_id
                    for component_id in range(1, count)
                    if int(stats[component_id, cv2.CC_STAT_AREA]) >= 12
                ],
            }
        )
    return image, labels > 0, records, [image_path, label_map_path, taxonomy_path]


def _paco_case_parts(
    case_path: Path,
    asset_domain: str,
) -> tuple[Image.Image, np.ndarray, list[dict[str, object]], list[Path]]:
    """Load a materialized PACO case as supervised retrieval prototypes.

    PACO masks are consumed only while an index is built.  The serialized index
    contains embeddings, geometry summaries, provenance, and hashes; query-time
    inference never opens this case or its ground-truth masks.
    """

    payload = json.loads(case_path.read_text(encoding="utf-8"))
    if payload.get("format") != "HPID PACO benchmark case":
        raise ValueError(f"not a materialized PACO case: {case_path}")
    case_dir = case_path.parent
    image_path = case_dir / "source_crop.png"
    root_mask_path = case_dir / "object_mask_crop.png"
    if not image_path.is_file() or not root_mask_path.is_file():
        raise ValueError(f"PACO case is missing its crop assets: {case_path}")
    image = Image.open(image_path).convert("RGB")
    root_mask = np.asarray(Image.open(root_mask_path).convert("L"), dtype=np.uint8) > 0
    if root_mask.shape != (image.height, image.width):
        raise ValueError(f"PACO object mask size does not match image: {case_path}")
    audit_paths = [case_path, image_path, root_mask_path]
    records: list[dict[str, object]] = []
    for row in payload.get("parts", []):
        if not isinstance(row, dict):
            continue
        mask_path = case_dir / str(row["mask_crop"])
        if not mask_path.is_file():
            raise ValueError(f"PACO part mask is missing: {mask_path}")
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
        if mask.shape != root_mask.shape:
            raise ValueError(f"PACO part mask size does not match image: {mask_path}")
        mask &= root_mask
        audit_paths.append(mask_path)
        if int(np.count_nonzero(mask)) < 12:
            continue
        semantic_name = normalize_paco_name(str(row["part_name"]))
        records.append(
            {
                "part_id": f"paco/{int(row['annotation_id'])}",
                "semantic_name": semantic_name,
                "semantic_parent": asset_domain,
                "assembly_parent_semantic": asset_domain,
                "side": "aggregate",
                "mask": mask,
                "geometry_masks": [mask],
            }
        )
    return image, root_mask, records, audit_paths


def build_retrieval_index(
    reference_manifest: Path,
    output_dir: Path,
    encoder: EmbeddingEncoder,
    *,
    metric_epochs: int = 120,
    metric_device: str = "cpu",
    seed: int = 20260812,
) -> dict[str, object]:
    raw = json.loads(reference_manifest.read_text(encoding="utf-8"))
    if raw.get("format") != REFERENCE_FORMAT:
        raise ValueError(f"reference manifest format must be {REFERENCE_FORMAT!r}")
    entries = raw.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("reference manifest must contain at least one entry")
    base = reference_manifest.parent
    default_part_aliases = {
        str(key): tuple(str(value) for value in values)
        for key, values in dict(raw.get("part_aliases", {})).items()
    }
    default_name_mapping = {
        str(key): str(value)
        for key, value in dict(raw.get("part_name_mapping", {})).items()
    }
    default_excluded_parts = {str(value) for value in raw.get("exclude_parts", [])}
    canonical_parent_by_name: dict[str, str] = {}
    canonical_assembly_by_name: dict[str, str] = {}
    maximum_instances_by_name: dict[str, int] = {}
    if raw.get("prompt_bank") is not None:
        prompt_bank_path = _resolve(base, str(raw["prompt_bank"]))
        prompt_bank = PromptBank.from_json(prompt_bank_path)
        for domain in prompt_bank.domains:
            local_parts = {part.semantic_name: part for part in domain.parts}

            def root_instance_limit(
                name: str,
                active: set[str],
                *,
                parts: dict[str, PartPrompt] = local_parts,
                domain_name: str = domain.name,
            ) -> int:
                if name in active:
                    return 8
                part = parts[name]
                parent = part.semantic_parent or domain_name
                if parent not in parts:
                    return min(8, part.maximum_instances)
                return min(
                    8,
                    part.maximum_instances
                    * root_instance_limit(parent, active | {name}),
                )

            for part in domain.parts:
                canonical_parent_by_name[part.semantic_name] = (
                    part.semantic_parent or domain.name
                )
                canonical_assembly_by_name[part.semantic_name] = (
                    part.assembly_parent or part.semantic_parent or domain.name
                )
                maximum_instances_by_name[part.semantic_name] = root_instance_limit(
                    part.semantic_name, set()
                )
    asset_rows: list[dict[str, object]] = []
    part_rows: list[dict[str, object]] = []
    asset_views: list[Image.Image] = []
    part_views: list[Image.Image] = []
    seen_asset_ids: set[str] = set()
    for asset_index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise TypeError("every reference entry must be an object")
        if entry.get("reviewed") is not True:
            raise ValueError(
                f"reference {entry.get('asset_id', asset_index)!r} is not reviewed"
            )
        asset_id = str(entry.get("asset_id", "")).strip()
        asset_label = str(entry.get("asset_label", "")).strip()
        asset_domain = str(entry.get("asset_domain", "")).strip()
        asset_profile = str(entry.get("asset_profile", "")).strip() or None
        if not asset_id or not asset_label or not asset_domain:
            raise ValueError("asset_id, asset_label, and asset_domain are required")
        if asset_id in seen_asset_ids:
            raise ValueError(f"duplicate asset_id: {asset_id}")
        seen_asset_ids.add(asset_id)
        if entry.get("package") is not None:
            package = _resolve(base, str(entry["package"]))
            image, root_mask, records, audit_paths = _package_parts(package)
            source_path = package / "source.png"
            source_kind = "reviewed_package"
        elif entry.get("paco_case") is not None:
            paco_case = _resolve(base, str(entry["paco_case"]))
            image, root_mask, records, audit_paths = _paco_case_parts(
                paco_case,
                asset_domain,
            )
            source_path = paco_case.parent / "source_crop.png"
            source_kind = "public_human_annotated_paco_case"
        else:
            required = ("image", "label_map", "taxonomy")
            if any(entry.get(key) is None for key in required):
                raise ValueError(
                    "each entry needs package, paco_case, or "
                    "image + label_map + taxonomy"
                )
            source_path = _resolve(base, str(entry["image"]))
            label_map_path = _resolve(base, str(entry["label_map"]))
            taxonomy_path = _resolve(base, str(entry["taxonomy"]))
            image, root_mask, records, audit_paths = _label_map_parts(
                source_path, label_map_path, taxonomy_path
            )
            source_kind = "reviewed_label_map"
        if int(np.count_nonzero(root_mask)) < 20:
            raise ValueError(f"reference {asset_id!r} has no usable foreground")
        aliases_by_part = dict(default_part_aliases)
        aliases_by_part.update(
            {
                str(key): tuple(str(value) for value in values)
                for key, values in dict(entry.get("part_aliases", {})).items()
            }
        )
        name_mapping = dict(default_name_mapping)
        name_mapping.update(
            {
                str(key): str(value)
                for key, value in dict(entry.get("part_name_mapping", {})).items()
            }
        )
        excluded_parts = default_excluded_parts | {
            str(value) for value in entry.get("exclude_parts", [])
        }
        asset_views.append(_masked_view(image, root_mask, padding=0.08))
        accepted_part_count = 0
        for record in records:
            source_semantic_name = str(record["semantic_name"])
            if source_semantic_name in excluded_parts:
                continue
            semantic_name = name_mapping.get(source_semantic_name, source_semantic_name)
            if _GENERIC_PART.search(semantic_name) and not bool(
                entry.get("allow_generic_parts", False)
            ):
                continue
            mask = np.asarray(record["mask"], dtype=bool)
            if int(np.count_nonzero(mask)) < 12:
                continue
            human_name = _humanize(semantic_name, asset_domain)
            phrases = _clean_phrases(
                (
                    human_name,
                    *aliases_by_part.get(source_semantic_name, ()),
                    *aliases_by_part.get(semantic_name, ()),
                    f"{human_name} of {asset_label}",
                    f"{asset_label} {human_name}",
                )
            )
            source_parent = str(record["semantic_parent"])
            mapped_parent = name_mapping.get(source_parent, source_parent)
            assembly_source = record["assembly_parent_semantic"]
            mapped_assembly = (
                name_mapping.get(str(assembly_source), str(assembly_source))
                if assembly_source is not None
                else None
            )
            geometry_masks = [
                np.asarray(component, dtype=bool)
                for component in record.get("geometry_masks", [mask])
                if int(np.count_nonzero(component)) >= 12
            ]
            row: dict[str, object] = {
                "asset_index": asset_index,
                "source_part_id": str(record["part_id"]),
                "source_semantic_name": source_semantic_name,
                "semantic_name": semantic_name,
                "display_name": human_name,
                "phrases": list(phrases),
                "semantic_parent": canonical_parent_by_name.get(
                    semantic_name,
                    mapped_parent,
                ),
                "assembly_parent_semantic": canonical_assembly_by_name.get(
                    semantic_name,
                    mapped_assembly,
                ),
                "side": str(record["side"]),
                "geometry": _geometry(mask, root_mask).tolist(),
                "geometry_samples": [
                    _geometry(component, root_mask).tolist()
                    for component in geometry_masks
                ],
                "observed_instance_count": len(geometry_masks),
            }
            if semantic_name in maximum_instances_by_name:
                row["maximum_instances"] = maximum_instances_by_name[semantic_name]
            part_rows.append(row)
            part_views.append(_masked_view(image, mask, padding=0.18))
            accepted_part_count += 1
        if accepted_part_count == 0:
            raise ValueError(f"reference {asset_id!r} has no semantic parts")
        asset_rows.append(
            {
                "asset_index": asset_index,
                "asset_id": asset_id,
                "asset_label": asset_label,
                "asset_domain": asset_domain,
                "asset_profile": asset_profile,
                "aliases": list(
                    _clean_phrases(
                        (
                            asset_label,
                            *(str(value) for value in entry.get("aliases", [])),
                        )
                    )
                ),
                "source_kind": source_kind,
                "source_path": str(source_path),
                "source_sha256": _sha256(source_path),
                "annotation_sha256": _combined_sha256(audit_paths),
                "part_count": accepted_part_count,
                "reviewed": True,
            }
        )
    asset_embeddings = _normalize_rows(encoder.encode(asset_views))
    part_embeddings = _normalize_rows(encoder.encode(part_views))
    if asset_embeddings.shape[0] != len(asset_rows):
        raise RuntimeError("encoder returned the wrong number of asset embeddings")
    if part_embeddings.shape[0] != len(part_rows):
        raise RuntimeError("encoder returned the wrong number of part embeddings")
    if asset_embeddings.shape[1] != part_embeddings.shape[1]:
        raise RuntimeError("asset and part embedding dimensions differ")
    asset_weights, asset_metric = _fit_metric_weights(
        asset_embeddings,
        [str(row["asset_label"]) for row in asset_rows],
        list(range(len(asset_rows))),
        epochs=metric_epochs,
        device=metric_device,
        seed=seed,
    )
    part_weights, part_metric = _fit_metric_weights(
        part_embeddings,
        [str(row["semantic_name"]) for row in part_rows],
        [int(row["asset_index"]) for row in part_rows],
        epochs=metric_epochs,
        device=metric_device,
        seed=seed + 1,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "prototype_arrays.npz"
    np.savez_compressed(
        arrays_path,
        asset_embeddings=asset_embeddings,
        part_embeddings=part_embeddings,
        asset_metric_weights=asset_weights,
        part_metric_weights=part_weights,
        part_geometry=np.asarray(
            [row["geometry"] for row in part_rows], dtype=np.float32
        ),
    )
    manifest: dict[str, object] = {
        "format": INDEX_FORMAT,
        "format_version": INDEX_VERSION,
        "encoder": {
            "model_name": encoder.model_name,
            "embedding_dimension": int(asset_embeddings.shape[1]),
        },
        "training": {
            "reference_manifest": str(reference_manifest.resolve()),
            "reference_manifest_sha256": _sha256(reference_manifest),
            "reviewed_only": True,
            "ground_truth_used_for_index_building": True,
            "ground_truth_used_during_query_inference": False,
            "seed": seed,
            "asset_metric": asset_metric,
            "part_metric": part_metric,
        },
        "asset_count": len(asset_rows),
        "part_prototype_count": len(part_rows),
        "arrays_path": arrays_path.name,
        "arrays_sha256": _sha256(arrays_path),
        "assets": asset_rows,
        "parts": part_rows,
    }
    manifest_path = output_dir / "index.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


class PrototypeIndex:
    def __init__(self, root: Path, manifest: dict[str, object]) -> None:
        self.root = root
        self.manifest = manifest
        arrays_path = root / str(manifest["arrays_path"])
        if _sha256(arrays_path) != manifest.get("arrays_sha256"):
            raise ValueError("prototype array hash does not match index manifest")
        with np.load(arrays_path, allow_pickle=False) as arrays:
            self.asset_embeddings = arrays["asset_embeddings"].astype(np.float32)
            self.part_embeddings = arrays["part_embeddings"].astype(np.float32)
            self.asset_metric_weights = arrays["asset_metric_weights"].astype(
                np.float32
            )
            self.part_metric_weights = arrays["part_metric_weights"].astype(np.float32)
            self.part_geometry = arrays["part_geometry"].astype(np.float32)
        self.assets = tuple(dict(row) for row in manifest["assets"])
        self.parts = tuple(dict(row) for row in manifest["parts"])
        dimension = int(manifest["encoder"]["embedding_dimension"])
        if self.asset_embeddings.shape != (len(self.assets), dimension):
            raise ValueError("invalid asset embedding array shape")
        if self.part_embeddings.shape != (len(self.parts), dimension):
            raise ValueError("invalid part embedding array shape")

    @classmethod
    def load(cls, path: Path) -> PrototypeIndex:
        root = path if path.is_dir() else path.parent
        manifest_path = root / "index.json" if path.is_dir() else path
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != INDEX_FORMAT:
            raise ValueError(f"not an HPID prototype index: {manifest_path}")
        if manifest.get("format_version") != INDEX_VERSION:
            raise ValueError("unsupported HPID prototype index version")
        return cls(root, manifest)

    @property
    def encoder_model_name(self) -> str:
        return str(self.manifest["encoder"]["model_name"])


class PrototypeRetriever:
    def __init__(
        self,
        index: PrototypeIndex,
        encoder: EmbeddingEncoder,
        *,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.index = index
        self.encoder = encoder
        self.config = config or RetrievalConfig()
        if encoder.model_name != index.encoder_model_name:
            raise ValueError(
                "retrieval encoder does not match the encoder recorded in the index"
            )

    @staticmethod
    def _root_key(root: MaskCandidate) -> str:
        return (
            f"{root.metadata.get('root_origin', 'legacy')}::"
            f"{root.metadata.get('root_index', 'unknown')}"
        )

    def query(
        self,
        image: Image.Image,
        roots: Sequence[MaskCandidate],
        *,
        asset_hint: str | None = None,
        asset_candidates_by_root: dict[str, Sequence[str]] | None = None,
        allowed_part_semantics_by_root: dict[str, Sequence[str]] | None = None,
    ) -> RetrievalResult:
        if not roots:
            return RetrievalResult((), {"root_count": 0, "accepted_root_count": 0})
        views = [
            _masked_view(image, root.mask.astype(bool), padding=0.08) for root in roots
        ]
        queries = _apply_metric(
            self.encoder.encode(views), self.index.asset_metric_weights
        )
        indexed = _apply_metric(
            self.index.asset_embeddings, self.index.asset_metric_weights
        )
        plans: list[RootRetrievalPlan] = []
        for index, root in enumerate(roots):
            root_key = self._root_key(root)
            allowed_inventory = allowed_part_semantics_by_root or {}
            allowed_semantics = (
                frozenset(
                    str(value)
                    for value in allowed_inventory[root_key]
                    if str(value).strip()
                )
                if root_key in allowed_inventory
                else None
            )
            plans.append(
                self._plan_for_root(
                    root,
                    queries[index] @ indexed.T,
                    asset_hint=asset_hint,
                    asset_candidates=(
                        tuple(
                            str(value)
                            for value in (asset_candidates_by_root or {}).get(
                                root_key, ()
                            )
                            if str(value).strip()
                        )
                    ),
                    allowed_output_semantics=allowed_semantics,
                )
            )
        resolved_plans = tuple(plans)
        return RetrievalResult(
            resolved_plans,
            {
                "algorithm": "hpid-reviewed-prototype-retrieval-v1",
                "index_path": str(self.index.root.resolve()),
                "index_array_sha256": self.index.manifest["arrays_sha256"],
                "encoder_model": self.encoder.model_name,
                "root_count": len(roots),
                "accepted_root_count": sum(plan.accepted for plan in plans),
                "asset_hint": asset_hint,
                "automatic_asset_candidates_by_root": {
                    str(key): [str(value) for value in values]
                    for key, values in (asset_candidates_by_root or {}).items()
                },
                "allowed_part_semantics_by_root": {
                    str(key): [str(value) for value in values]
                    for key, values in (allowed_part_semantics_by_root or {}).items()
                },
                "ground_truth_used_during_query_inference": False,
                "plans": [
                    {
                        **asdict(plan),
                        "part_priors": [asdict(prior) for prior in plan.part_priors],
                    }
                    for plan in resolved_plans
                ],
            },
        )

    def _plan_for_root(
        self,
        root: MaskCandidate,
        similarities: np.ndarray,
        *,
        asset_hint: str | None,
        asset_candidates: Sequence[str] = (),
        allowed_output_semantics: frozenset[str] | None = None,
    ) -> RootRetrievalPlan:
        all_indices = np.arange(len(self.index.assets), dtype=np.int64)
        root_domain = root.semantic_name
        domain_indices = np.asarray(
            [
                index
                for index, asset in enumerate(self.index.assets)
                if str(asset.get("asset_domain", "")) == root_domain
            ],
            dtype=np.int64,
        )
        pool = domain_indices if len(domain_indices) else all_indices
        selection_scope = "root_domain" if len(domain_indices) else "global_fallback"
        selected_profile = str(root.metadata.get("selected_part_profile", "")).strip()
        profile_source = str(root.metadata.get("profile_hint_source", ""))
        profile_is_resolved = bool(
            selected_profile
            and (
                str(root.metadata.get("profile_resolution_status", "")) == "accepted"
                or profile_source
                in {
                    "isolated_profile_consensus",
                    "user_asset_prompt",
                }
                or str(root.metadata.get("root_query_mode", "")).startswith(
                    "user_asset_prompt"
                )
            )
        )
        resolved_profile_pool = np.asarray([], dtype=np.int64)
        candidate_inventory_pool = np.asarray([], dtype=np.int64)

        cleaned_hint = str(asset_hint or "").strip()
        hint_matched = False
        candidate_set_mode = False
        candidate_inventory_mode = False
        if cleaned_hint:
            hinted = np.asarray(
                [
                    int(index)
                    for index in all_indices
                    if _asset_hint_matches(self.index.assets[int(index)], cleaned_hint)
                ],
                dtype=np.int64,
            )
            if len(hinted):
                same_domain_hinted = np.intersect1d(
                    hinted,
                    domain_indices,
                    assume_unique=False,
                )
                pool = same_domain_hinted if len(same_domain_hinted) else hinted
                selection_scope = "user_asset_hint"
                hint_matched = True
            else:
                ranking = np.argsort(-similarities)[: self.config.top_k_assets]
                nearest = self._nearest_rows(
                    ranking,
                    similarities,
                    similarities,
                    selection_scope="asset_hint_not_indexed",
                )
                return RootRetrievalPlan(
                    self._root_key(root),
                    root.semantic_name,
                    False,
                    "asset_hint_not_indexed",
                    float(similarities[ranking[0]]),
                    None,
                    None,
                    0.0,
                    0,
                    nearest,
                    (),
                )
        elif asset_candidates:
            normalized_candidates = tuple(
                dict.fromkeys(
                    value.strip() for value in asset_candidates if value.strip()
                )
            )
            candidate_indices = np.asarray(
                [
                    int(index)
                    for index in all_indices
                    if any(
                        _asset_hint_matches(
                            self.index.assets[int(index)], candidate_label
                        )
                        for candidate_label in normalized_candidates
                    )
                ],
                dtype=np.int64,
            )
            if len(candidate_indices):
                same_domain_candidates = np.intersect1d(
                    candidate_indices,
                    domain_indices,
                    assume_unique=False,
                )
                pool = (
                    same_domain_candidates
                    if len(same_domain_candidates)
                    else candidate_indices
                )
                resolved_labels = {
                    str(self.index.assets[int(index)]["asset_label"]) for index in pool
                }
                hint_matched = True
                candidate_set_mode = len(resolved_labels) > 1
                candidate_inventory_mode = candidate_set_mode
                candidate_inventory_pool = pool.copy()
                selection_scope = (
                    "automatic_asset_candidate_set"
                    if candidate_set_mode
                    else "automatic_exact_asset_route"
                )
        if profile_is_resolved and not cleaned_hint:
            profile_source_pool = domain_indices if len(domain_indices) else all_indices
            profile_pool = np.asarray(
                [
                    int(index)
                    for index in profile_source_pool
                    if str(self.index.assets[int(index)].get("asset_profile", ""))
                    == selected_profile
                ],
                dtype=np.int64,
            )
            resolved_profile_pool = profile_pool
            if not len(profile_pool):
                fallback_order = np.argsort(-similarities[pool])[
                    : self.config.top_k_assets
                ]
                ranking = pool[fallback_order]
                nearest = self._nearest_rows(
                    ranking,
                    similarities,
                    similarities,
                    selection_scope="resolved_profile_not_indexed",
                )
                return RootRetrievalPlan(
                    self._root_key(root),
                    root.semantic_name,
                    False,
                    "resolved_profile_not_indexed",
                    float(similarities[ranking[0]]),
                    None,
                    None,
                    0.0,
                    0,
                    nearest,
                    (),
                )
            candidate_profile_pool = np.intersect1d(
                pool,
                profile_pool,
                assume_unique=False,
            )
            pool = (
                candidate_profile_pool if len(candidate_profile_pool) else profile_pool
            )
            resolved_labels = {
                str(self.index.assets[int(index)]["asset_label"]) for index in pool
            }
            candidate_set_mode = len(resolved_labels) > 1
            hint_matched = True
            selection_scope += "+resolved_profile"
        elif len(domain_indices) and self.config.allow_domain_relabel:
            global_top = int(np.argmax(similarities))
            domain_top = int(domain_indices[np.argmax(similarities[domain_indices])])
            global_domain = str(self.index.assets[global_top].get("asset_domain", ""))
            if (
                global_domain != root_domain
                and float(similarities[global_top])
                >= self.config.domain_relabel_similarity
                and float(similarities[global_top] - similarities[domain_top])
                >= self.config.domain_relabel_margin
            ):
                pool = np.asarray(
                    [
                        index
                        for index in all_indices
                        if str(self.index.assets[int(index)].get("asset_domain", ""))
                        == global_domain
                    ],
                    dtype=np.int64,
                )
                selection_scope = "high_confidence_cross_domain"

        adjusted = similarities.copy()
        if selected_profile and not profile_is_resolved:
            for index in pool:
                if (
                    str(self.index.assets[int(index)].get("asset_profile", ""))
                    == selected_profile
                ):
                    adjusted[int(index)] += self.config.profile_similarity_bonus
        local_order = np.argsort(-adjusted[pool])[: self.config.top_k_assets]
        ranking = pool[local_order]
        nearest = self._nearest_rows(
            ranking,
            similarities,
            adjusted,
            selection_scope=selection_scope,
        )
        top_index = int(ranking[0])
        top_similarity = float(similarities[top_index])
        minimum_similarity = (
            self.config.minimum_prompted_asset_similarity
            if hint_matched
            else self.config.minimum_profiled_asset_similarity
            if profile_is_resolved
            else self.config.minimum_asset_similarity
        )
        if top_similarity < minimum_similarity:
            return RootRetrievalPlan(
                self._root_key(root),
                root.semantic_name,
                False,
                "open_set_similarity_rejection",
                top_similarity,
                None,
                None,
                0.0,
                0,
                nearest,
                (),
            )
        if candidate_set_mode or candidate_inventory_mode:
            supporting = [int(index) for index in pool]
            top_adjusted = float(adjusted[top_index])
            weights = {
                index: math.exp(
                    max(
                        -8.0,
                        (float(adjusted[index]) - top_adjusted) / 0.055,
                    )
                )
                for index in supporting
            }
            domains = Counter(
                str(self.index.assets[index]["asset_domain"]) for index in supporting
            )
            output_domain = (
                root_domain if root_domain in domains else domains.most_common(1)[0][0]
            )
            priors = self._aggregate_part_priors(
                supporting,
                weights,
                output_domain=output_domain,
                asset_label="candidate object",
                top_similarity=top_similarity,
                allowed_output_semantics=allowed_output_semantics,
            )
            supporting_count = len(supporting)
            if (
                profile_is_resolved
                and candidate_inventory_mode
                and allowed_output_semantics is not None
                and len(candidate_inventory_pool)
            ):
                inventory_supporting = [
                    int(index) for index in candidate_inventory_pool
                ]
                inventory_top_adjusted = max(
                    float(adjusted[index]) for index in inventory_supporting
                )
                inventory_weights = {
                    index: math.exp(
                        max(
                            -8.0,
                            (float(adjusted[index]) - inventory_top_adjusted) / 0.055,
                        )
                    )
                    for index in inventory_supporting
                }
                inventory_domains = Counter(
                    str(self.index.assets[index]["asset_domain"])
                    for index in inventory_supporting
                )
                inventory_domain = (
                    root_domain
                    if root_domain in inventory_domains
                    else inventory_domains.most_common(1)[0][0]
                )
                inventory_priors = self._aggregate_part_priors(
                    inventory_supporting,
                    inventory_weights,
                    output_domain=inventory_domain,
                    asset_label="candidate object",
                    top_similarity=top_similarity,
                    allowed_output_semantics=allowed_output_semantics,
                )
                priors = _merge_hierarchical_part_priors(priors, inventory_priors)
                supporting_count = len({*supporting, *inventory_supporting})
            return RootRetrievalPlan(
                self._root_key(root),
                root.semantic_name,
                bool(priors),
                (
                    "accepted_ambiguous_candidate_inventory"
                    if priors
                    else "no_recurrent_semantic_parts"
                ),
                top_similarity,
                None,
                output_domain,
                0.0,
                supporting_count,
                nearest,
                priors,
            )
        usable = [
            int(index)
            for index in ranking
            if float(adjusted[index])
            >= float(adjusted[top_index]) - self.config.maximum_similarity_drop
        ]
        weights = {
            index: math.exp(
                (float(adjusted[index]) - float(adjusted[top_index])) / 0.055
            )
            for index in usable
        }
        label_scores: dict[str, float] = defaultdict(float)
        for index in usable:
            label_scores[str(self.index.assets[index]["asset_label"])] += weights[index]
        ordered_labels = sorted(
            label_scores.items(), key=lambda item: item[1], reverse=True
        )
        asset_label = ordered_labels[0][0]
        second_score = ordered_labels[1][1] if len(ordered_labels) > 1 else 0.0
        margin = (ordered_labels[0][1] - second_score) / max(1e-8, ordered_labels[0][1])
        if len(ordered_labels) > 1 and margin < self.config.minimum_asset_label_margin:
            return RootRetrievalPlan(
                self._root_key(root),
                root.semantic_name,
                False,
                "ambiguous_asset_label_rejection",
                top_similarity,
                asset_label,
                None,
                margin,
                0,
                nearest,
                (),
            )
        supporting = (
            [int(index) for index in resolved_profile_pool]
            if profile_is_resolved
            and allowed_output_semantics is not None
            and len(resolved_profile_pool)
            else [
                int(index)
                for index in pool
                if str(self.index.assets[index]["asset_label"]) == asset_label
            ]
        )
        for index in supporting:
            weights.setdefault(
                index,
                math.exp(
                    max(
                        -8.0,
                        (float(adjusted[index]) - float(adjusted[top_index])) / 0.055,
                    )
                ),
            )
        domain = Counter(
            str(self.index.assets[index]["asset_domain"]) for index in supporting
        ).most_common(1)[0][0]
        priors = self._aggregate_part_priors(
            supporting,
            weights,
            output_domain=domain,
            asset_label=asset_label,
            top_similarity=top_similarity,
            allowed_output_semantics=allowed_output_semantics,
        )
        return RootRetrievalPlan(
            self._root_key(root),
            root.semantic_name,
            bool(priors),
            "accepted" if priors else "no_recurrent_semantic_parts",
            top_similarity,
            asset_label,
            domain,
            margin,
            len(supporting),
            nearest,
            priors,
        )

    def _nearest_rows(
        self,
        ranking: Sequence[int],
        similarities: np.ndarray,
        adjusted: np.ndarray,
        *,
        selection_scope: str,
    ) -> tuple[dict[str, object], ...]:
        return tuple(
            {
                "asset_id": str(self.index.assets[int(index)]["asset_id"]),
                "asset_label": str(self.index.assets[int(index)]["asset_label"]),
                "asset_domain": str(self.index.assets[int(index)]["asset_domain"]),
                "asset_profile": self.index.assets[int(index)].get("asset_profile"),
                "similarity": float(similarities[int(index)]),
                "adjusted_similarity": float(adjusted[int(index)]),
                "selection_scope": selection_scope,
            }
            for index in ranking
        )

    def _aggregate_part_priors(
        self,
        asset_indices: Sequence[int],
        asset_weights: dict[int, float],
        *,
        output_domain: str,
        asset_label: str,
        top_similarity: float,
        allowed_output_semantics: frozenset[str] | None = None,
    ) -> tuple[RetrievedPartPrior, ...]:
        selected = set(asset_indices)
        by_semantic: dict[str, list[int]] = defaultdict(list)
        assets_by_semantic: dict[str, set[int]] = defaultdict(set)
        for index, part in enumerate(self.index.parts):
            asset_index = int(part["asset_index"])
            if asset_index not in selected:
                continue
            semantic = str(part["semantic_name"])
            by_semantic[semantic].append(index)
            assets_by_semantic[semantic].add(asset_index)
        priors: list[RetrievedPartPrior] = []
        total_weight = sum(asset_weights[index] for index in asset_indices)
        for semantic, prototype_indices in by_semantic.items():
            supporting_assets = assets_by_semantic[semantic]
            if len(supporting_assets) < self.config.minimum_part_support_count:
                continue
            prevalence = sum(asset_weights[index] for index in supporting_assets) / max(
                1e-8, total_weight
            )
            if prevalence < self.config.minimum_part_prevalence:
                continue
            rows = [self.index.parts[index] for index in prototype_indices]
            display = Counter(str(row["display_name"]) for row in rows).most_common(1)[
                0
            ][0]
            phrases = _clean_phrases(
                [phrase for row in rows for phrase in list(row.get("phrases", []))]
                + [display, f"{display} of {asset_label}"]
            )[:8]
            parents = Counter(str(row["semantic_parent"]) for row in rows)
            assembly = Counter(
                str(row["assembly_parent_semantic"])
                for row in rows
                if row.get("assembly_parent_semantic") is not None
            )
            geometry = self.index.part_geometry[prototype_indices]
            geometry_samples = np.asarray(
                [
                    sample
                    for row in rows
                    for sample in row.get("geometry_samples", [row["geometry"]])
                ],
                dtype=np.float32,
            )
            output_name = semantic
            if not output_name.startswith(f"{output_domain}_"):
                output_name = f"{output_domain}_{output_name}"
            if (
                allowed_output_semantics is not None
                and output_name not in allowed_output_semantics
            ):
                continue
            observed_by_asset: dict[int, int] = defaultdict(int)
            for row in rows:
                observed_by_asset[int(row["asset_index"])] += int(
                    row.get(
                        "observed_instance_count",
                        len(row.get("geometry_samples", [row["geometry"]])),
                    )
                )
            configured_limits = [
                int(row["maximum_instances"])
                for row in rows
                if row.get("maximum_instances") is not None
            ]
            observed_limit = max(
                1,
                round(float(np.quantile(list(observed_by_asset.values()), 0.90))),
            )
            root_instance_limit = min(
                8,
                min(max(configured_limits), observed_limit)
                if configured_limits
                else observed_limit,
            )
            priors.append(
                RetrievedPartPrior(
                    semantic_name=semantic,
                    output_semantic_name=output_name,
                    display_name=display,
                    phrases=phrases,
                    semantic_parent=parents.most_common(1)[0][0],
                    assembly_parent_semantic=(
                        assembly.most_common(1)[0][0] if assembly else None
                    ),
                    maximum_instances=root_instance_limit,
                    support_count=len(supporting_assets),
                    prevalence=float(prevalence),
                    retrieval_score=float(prevalence * (0.5 + 0.5 * top_similarity)),
                    prototype_indices=tuple(prototype_indices),
                    geometry_mean=tuple(
                        float(value) for value in geometry.mean(axis=0)
                    ),
                    geometry_std=tuple(
                        float(value) for value in np.maximum(geometry.std(axis=0), 0.04)
                    ),
                    geometry_samples=tuple(
                        tuple(float(value) for value in sample)
                        for sample in geometry_samples
                    ),
                )
            )
        priors.sort(
            key=lambda item: (
                item.retrieval_score,
                item.support_count,
                len(item.prototype_indices),
            ),
            reverse=True,
        )
        return tuple(priors[: self.config.maximum_part_prompts])

    def rerank_candidates(
        self,
        image: Image.Image,
        root: MaskCandidate,
        candidates: Sequence[MaskCandidate],
        priors: Sequence[RetrievedPartPrior],
        *,
        existing_candidates: Sequence[MaskCandidate] = (),
    ) -> tuple[list[MaskCandidate], dict[str, object]]:
        prior_by_slug = {prior.slug: prior for prior in priors}
        matched: list[tuple[MaskCandidate, RetrievedPartPrior]] = []
        for candidate in candidates:
            slug = str(candidate.metadata.get("guided_prompt_slug", ""))
            prior = prior_by_slug.get(slug)
            if prior is not None:
                matched.append((candidate, prior))
        if not matched:
            return [], {
                "candidate_count": 0,
                "accepted_count": 0,
                "rejected_count": 0,
            }
        views = [
            _masked_view(image, candidate.mask.astype(bool), padding=0.18)
            for candidate, _ in matched
        ]
        raw_candidate_embeddings = _normalize_rows(self.encoder.encode(views))
        candidate_embeddings = _apply_metric(
            raw_candidate_embeddings, self.index.part_metric_weights
        )
        raw_prototype_embeddings = _normalize_rows(self.index.part_embeddings)
        prototype_embeddings = _apply_metric(
            self.index.part_embeddings, self.index.part_metric_weights
        )
        accepted: list[MaskCandidate] = []
        accepted_by_semantic: Counter[str] = Counter()
        rows: list[dict[str, object]] = []
        for index, (candidate, prior) in enumerate(matched):
            similarities = (
                candidate_embeddings[index]
                @ prototype_embeddings[list(prior.prototype_indices)].T
            )
            visual_similarity = float(np.max(similarities))
            raw_visual_similarity = float(
                np.max(
                    raw_candidate_embeddings[index]
                    @ raw_prototype_embeddings[list(prior.prototype_indices)].T
                )
            )
            descriptor = _geometry(candidate.mask.astype(bool), root.mask.astype(bool))
            geometry_score = _geometry_compatibility(descriptor, prior)
            keep = not (
                visual_similarity < self.config.minimum_part_similarity
                or raw_visual_similarity < self.config.minimum_raw_part_similarity
                or geometry_score < self.config.minimum_geometry_compatibility
            )
            rejection_reason = None
            same_semantic_existing = [
                existing
                for existing in existing_candidates
                if self._root_key(existing) == self._root_key(root)
                and existing.semantic_name == prior.output_semantic_name
                and not bool(existing.metadata.get("generic_visual_region"))
            ]
            if keep and any(
                _mask_overlap(
                    candidate.mask.astype(bool),
                    existing.mask.astype(bool),
                )[0]
                >= self.config.prototype_duplicate_iou
                or _mask_overlap(
                    candidate.mask.astype(bool),
                    existing.mask.astype(bool),
                )[1]
                >= self.config.prototype_duplicate_containment
                for existing in same_semantic_existing
            ):
                keep = False
                rejection_reason = "existing_semantic_duplicate"
            existing_floor = int(bool(same_semantic_existing))
            if (
                keep
                and existing_floor + accepted_by_semantic[prior.output_semantic_name]
                >= prior.maximum_instances
            ):
                keep = False
                rejection_reason = "existing_semantic_instance_limit"
            rows.append(
                {
                    "semantic_name": prior.output_semantic_name,
                    "visual_similarity": visual_similarity,
                    "raw_visual_similarity": raw_visual_similarity,
                    "geometry_compatibility": geometry_score,
                    "support_count": prior.support_count,
                    "accepted": keep,
                    "rejection_reason": rejection_reason,
                }
            )
            if not keep:
                continue
            visual_weight = float(np.clip((visual_similarity + 1.0) / 2.0, 0.0, 1.0))
            score_scale = 0.58 + 0.27 * visual_weight + 0.15 * geometry_score
            accepted.append(
                replace(
                    candidate,
                    semantic_name=prior.output_semantic_name,
                    semantic_parent=prior.semantic_parent,
                    score=float(candidate.score * score_scale),
                    source=candidate.source.rsplit("/", maxsplit=1)[0]
                    + "/retrieved-part",
                    source_reliability=float(
                        candidate.source_reliability
                        * (0.72 + 0.28 * prior.retrieval_score)
                    ),
                    metadata={
                        **candidate.metadata,
                        "retrieval_prior": True,
                        "retrieval_source_semantic": prior.semantic_name,
                        "retrieval_output_semantic": prior.output_semantic_name,
                        "retrieval_support_count": prior.support_count,
                        "retrieval_prevalence": prior.prevalence,
                        "retrieval_score": prior.retrieval_score,
                        "retrieval_visual_similarity": visual_similarity,
                        "retrieval_raw_visual_similarity": raw_visual_similarity,
                        "retrieval_geometry_compatibility": geometry_score,
                        "retrieval_semantic_parent": prior.semantic_parent,
                        "retrieval_assembly_parent_semantic": (
                            prior.assembly_parent_semantic
                        ),
                    },
                )
            )
            accepted_by_semantic[prior.output_semantic_name] += 1
        return accepted, {
            "candidate_count": len(matched),
            "accepted_count": len(accepted),
            "rejected_count": len(matched) - len(accepted),
            "rows": rows,
        }

    def label_visual_candidates(
        self,
        image: Image.Image,
        roots: Sequence[MaskCandidate],
        candidates: Sequence[MaskCandidate],
        plans: Sequence[RootRetrievalPlan],
        *,
        existing_candidates: Sequence[MaskCandidate] = (),
    ) -> tuple[list[MaskCandidate], dict[str, object]]:
        """Assign reviewed semantics directly to compatible label-free regions."""

        root_by_key = {self._root_key(root): root for root in roots}
        plan_by_key = {
            plan.root_key: plan for plan in plans if plan.accepted and plan.part_priors
        }
        eligible: list[
            tuple[MaskCandidate, MaskCandidate, tuple[RetrievedPartPrior, ...]]
        ] = []
        untouched: list[MaskCandidate] = []
        for candidate in candidates:
            root_key = self._root_key(candidate)
            root = root_by_key.get(root_key)
            plan = plan_by_key.get(root_key)
            if (
                root is None
                or plan is None
                or not bool(candidate.metadata.get("generic_visual_region"))
            ):
                untouched.append(candidate)
                continue
            eligible.append((candidate, root, plan.part_priors))
        if not eligible:
            return list(candidates), {
                "candidate_count": 0,
                "labelled_count": 0,
                "rejected_count": 0,
                "rows": [],
            }

        views = [
            _masked_view(image, candidate.mask.astype(bool), padding=0.18)
            for candidate, _, _ in eligible
        ]
        raw_candidates = _normalize_rows(self.encoder.encode(views))
        metric_candidates = _apply_metric(
            raw_candidates, self.index.part_metric_weights
        )
        raw_prototypes = _normalize_rows(self.index.part_embeddings)
        metric_prototypes = _apply_metric(
            self.index.part_embeddings, self.index.part_metric_weights
        )
        candidate_options: list[
            tuple[
                MaskCandidate,
                str,
                list[tuple[float, float, float, RetrievedPartPrior]],
            ]
        ] = []
        for row_index, (candidate, root, priors) in enumerate(eligible):
            scores: list[
                tuple[
                    float,
                    float,
                    float,
                    RetrievedPartPrior,
                ]
            ] = []
            descriptor = _geometry(candidate.mask.astype(bool), root.mask.astype(bool))
            for prior in priors:
                prototype_indices = list(prior.prototype_indices)
                metric_similarity = float(
                    np.max(
                        metric_candidates[row_index]
                        @ metric_prototypes[prototype_indices].T
                    )
                )
                raw_similarity = float(
                    np.max(
                        raw_candidates[row_index] @ raw_prototypes[prototype_indices].T
                    )
                )
                geometry_score = _geometry_compatibility(descriptor, prior)
                if (
                    raw_similarity < self.config.minimum_raw_part_similarity
                    or geometry_score < self.config.minimum_geometry_compatibility
                ):
                    continue
                combined = (
                    0.46 * raw_similarity
                    + 0.32 * metric_similarity
                    + 0.22 * geometry_score
                )
                scores.append(
                    (
                        combined,
                        raw_similarity,
                        geometry_score,
                        prior,
                    )
                )
            scores.sort(key=lambda item: item[0], reverse=True)
            candidate_options.append((candidate, self._root_key(root), scores))

        assigned: dict[int, tuple[float, float, float, RetrievedPartPrior]] = {}
        semantic_counts: Counter[tuple[str, str]] = Counter()
        existing_by_semantic: dict[tuple[str, str], list[MaskCandidate]] = defaultdict(
            list
        )
        for existing in existing_candidates:
            if bool(existing.metadata.get("generic_visual_region")):
                continue
            existing_by_semantic[
                (self._root_key(existing), existing.semantic_name)
            ].append(existing)
        for key, existing in existing_by_semantic.items():
            if existing:
                semantic_counts[key] = 1
        rejection_reason: dict[int, str] = {}
        proposals: list[
            tuple[
                float,
                float,
                MaskCandidate,
                str,
                tuple[float, float, float, RetrievedPartPrior],
            ]
        ] = []
        for candidate, root_key, options in candidate_options:
            if not options:
                rejection_reason[id(candidate)] = "no_compatible_prototype"
                continue
            option = options[0]
            next_score = options[1][0] if len(options) > 1 else 0.0
            margin = option[0] - next_score
            if margin < self.config.prototype_label_margin:
                rejection_reason[id(candidate)] = "ambiguous_label_margin"
                continue
            proposals.append((option[0], margin, candidate, root_key, option))
        proposals.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for _, margin, candidate, root_key, option in proposals:
            candidate_id = id(candidate)
            if candidate_id in assigned:
                continue
            _, _, _, prior = option
            semantic_key = (root_key, prior.output_semantic_name)
            existing_semantic = existing_by_semantic.get(semantic_key, [])
            if any(
                _mask_overlap(
                    candidate.mask.astype(bool),
                    existing.mask.astype(bool),
                )[0]
                >= self.config.prototype_duplicate_iou
                or _mask_overlap(
                    candidate.mask.astype(bool),
                    existing.mask.astype(bool),
                )[1]
                >= self.config.prototype_duplicate_containment
                for existing in existing_semantic
            ):
                rejection_reason.setdefault(candidate_id, "existing_semantic_duplicate")
                continue
            if semantic_counts[semantic_key] >= prior.maximum_instances:
                rejection_reason.setdefault(candidate_id, "instance_limit")
                continue
            duplicate = False
            for assigned_id, assigned_option in assigned.items():
                if (
                    assigned_option[3].output_semantic_name
                    != prior.output_semantic_name
                ):
                    continue
                assigned_root_key = next(
                    option_root
                    for item, option_root, _ in candidate_options
                    if id(item) == assigned_id
                )
                if assigned_root_key != root_key:
                    continue
                assigned_candidate = next(
                    item for item, _, _ in candidate_options if id(item) == assigned_id
                )
                overlap, containment = _mask_overlap(
                    candidate.mask.astype(bool),
                    assigned_candidate.mask.astype(bool),
                )
                if (
                    overlap >= self.config.prototype_duplicate_iou
                    or containment >= self.config.prototype_duplicate_containment
                ):
                    duplicate = True
                    break
            if duplicate:
                rejection_reason.setdefault(candidate_id, "duplicate_semantic_region")
                continue
            assigned[candidate_id] = option
            semantic_counts[semantic_key] += 1

        labelled: dict[int, MaskCandidate] = {}
        rows: list[dict[str, object]] = []
        for candidate, root_key, options in candidate_options:
            best = assigned.get(id(candidate))
            best_raw = options[0] if options else None
            margin = (
                best_raw[0] - options[1][0]
                if best_raw is not None and len(options) > 1
                else (best_raw[0] if best_raw is not None else 0.0)
            )
            rows.append(
                {
                    "candidate_key": candidate.metadata.get("candidate_key"),
                    "root_key": root_key,
                    "accepted": best is not None,
                    "semantic_name": (
                        best[3].output_semantic_name if best is not None else None
                    ),
                    "combined_score": best[0] if best is not None else None,
                    "raw_visual_similarity": best[1] if best is not None else None,
                    "geometry_compatibility": best[2] if best is not None else None,
                    "label_margin": margin,
                    "rejection_reason": rejection_reason.get(id(candidate)),
                }
            )
            if best is None:
                continue
            combined, raw_similarity, geometry_score, prior = best
            labelled[id(candidate)] = replace(
                candidate,
                semantic_name=prior.output_semantic_name,
                semantic_parent=prior.semantic_parent,
                prompt=f"retrieved visual prototype for {prior.display_name}",
                source=candidate.source.rsplit("/", maxsplit=1)[0]
                + "/prototype-labelled-region",
                source_reliability=float(
                    candidate.source_reliability * (0.72 + 0.28 * combined)
                ),
                metadata={
                    **candidate.metadata,
                    "generic_visual_region": False,
                    "retrieval_prior": True,
                    "retrieval_region_label": True,
                    "retrieval_source_semantic": prior.semantic_name,
                    "retrieval_output_semantic": prior.output_semantic_name,
                    "retrieval_raw_visual_similarity": raw_similarity,
                    "retrieval_geometry_compatibility": geometry_score,
                    "retrieval_label_score": combined,
                    "retrieval_label_margin": margin,
                    "assembly_parent_semantic": (
                        prior.assembly_parent_semantic or prior.semantic_parent
                    ),
                },
            )
        output = [labelled.get(id(candidate), candidate) for candidate in candidates]
        aggregate_semantic_counts: Counter[str] = Counter()
        for (_, semantic_name), count in semantic_counts.items():
            aggregate_semantic_counts[semantic_name] += count
        return output, {
            "algorithm": "hpid-object-level-prototype-assignment-v2",
            "candidate_count": len(eligible),
            "labelled_count": len(labelled),
            "rejected_count": len(eligible) - len(labelled),
            "semantic_instance_counts": dict(sorted(aggregate_semantic_counts.items())),
            "root_semantic_instance_counts": {
                f"{root_key}|{semantic_name}": count
                for (root_key, semantic_name), count in sorted(semantic_counts.items())
            },
            "rows": rows,
        }


def _geometry_compatibility(descriptor: np.ndarray, prior: RetrievedPartPrior) -> float:
    samples = np.asarray(prior.geometry_samples, dtype=np.float32)
    scales = np.maximum(
        np.asarray(prior.geometry_std, dtype=np.float32),
        np.asarray((0.10, 0.10, 0.14, 0.14, 0.025), dtype=np.float32),
    )
    direct = np.exp(
        -0.5 * np.min(np.mean(((samples - descriptor[None, :]) / scales) ** 2, axis=1))
    )
    mirrored = descriptor.copy()
    mirrored[0] = 1.0 - mirrored[0]
    mirror_score = np.exp(
        -0.5 * np.min(np.mean(((samples - mirrored[None, :]) / scales) ** 2, axis=1))
    )
    return float(max(direct, mirror_score))


def _mask_overlap(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    intersection = int(np.count_nonzero(first & second))
    if not intersection:
        return 0.0, 0.0
    first_area = int(np.count_nonzero(first))
    second_area = int(np.count_nonzero(second))
    union = first_area + second_area - intersection
    return (
        intersection / max(1, union),
        intersection / max(1, min(first_area, second_area)),
    )


def apply_retrieval_domain_corrections(
    candidates: Sequence[MaskCandidate],
    result: RetrievalResult,
    *,
    config: RetrievalConfig,
) -> tuple[list[MaskCandidate], dict[str, object]]:
    corrections = {
        plan.root_key: plan
        for plan in result.plans
        if (
            config.allow_domain_relabel
            and plan.accepted
            and plan.asset_domain is not None
            and plan.asset_domain != plan.root_semantic
            and plan.top_similarity >= config.domain_relabel_similarity
            and plan.supporting_asset_count >= config.domain_relabel_minimum_support
        )
    }
    if not corrections:
        return list(candidates), {"correction_count": 0, "corrections": []}
    corrected: list[MaskCandidate] = []
    rows: list[dict[str, object]] = []
    for candidate in candidates:
        root_key = (
            f"{candidate.metadata.get('root_origin', 'legacy')}::"
            f"{candidate.metadata.get('root_index', 'unknown')}"
        )
        plan = corrections.get(root_key)
        if plan is None:
            corrected.append(candidate)
            continue
        is_root = (
            candidate.metadata.get("parent_candidate_key") is None
            and candidate.semantic_name == candidate.semantic_parent
        )
        if not is_root:
            continue
        corrected.append(
            replace(
                candidate,
                semantic_name=str(plan.asset_domain),
                semantic_parent=str(plan.asset_domain),
                metadata={
                    **candidate.metadata,
                    "retrieval_domain_correction": True,
                    "retrieval_previous_domain": candidate.semantic_name,
                    "retrieval_asset_label": plan.asset_label,
                    "retrieval_similarity": plan.top_similarity,
                },
            )
        )
        rows.append(
            {
                "root_key": root_key,
                "from": candidate.semantic_name,
                "to": plan.asset_domain,
                "asset_label": plan.asset_label,
                "similarity": plan.top_similarity,
                "supporting_asset_count": plan.supporting_asset_count,
            }
        )
    return corrected, {"correction_count": len(rows), "corrections": rows}
