from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from PIL import Image

from .prompt_bank import DomainPrompt


@dataclass(frozen=True)
class AssetRoute:
    accepted: bool
    asset_label: str | None
    asset_domain: str | None
    asset_profile: str | None
    score: float
    margin: float
    alternatives: tuple[dict[str, object], ...]
    candidate_labels: tuple[str, ...]
    candidate_domains: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class ProfileTextRoute:
    """Conservative zero-shot route into one domain's physical-part inventory."""

    accepted: bool
    profile: str | None
    score: float
    margin: float
    alternatives: tuple[dict[str, object], ...]
    reason: str


@dataclass(frozen=True)
class ProfileTextRouterConfig:
    minimum_score: float = 0.10
    minimum_margin: float = 0.025
    maximum_alternatives: int = 5


@dataclass(frozen=True)
class AssetDomainResolution:
    accepted: bool
    resolved_domain: str | None
    resolved_profile: str | None
    reason: str
    support_count: int
    candidate_count: int
    support_ratio: float
    vote_margin: int
    domain_votes: tuple[dict[str, object], ...]
    resolved_asset_label: str | None = None
    asset_label_reason: str | None = None


@dataclass(frozen=True)
class AssetRouterConfig:
    prototype_weight: float = 0.20
    text_weight: float = 0.80
    nearest_asset_weight: float = 0.05
    nearest_asset_count: int = 3
    minimum_score: float = 0.18
    minimum_margin: float = 0.020
    maximum_alternatives: int = 5
    maximum_candidate_labels: int = 5
    maximum_candidate_score_drop: float = 0.12
    minimum_domain_consensus_ratio: float = 0.60
    minimum_domain_consensus_votes: int = 2
    minimum_domain_vote_margin: int = 1
    maximum_current_domain_score_gap: float = 0.020
    maximum_cross_view_label_score_gap: float = 0.015


class ImageTextEncoder(Protocol):
    model_name: str

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray: ...

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _load_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) >= 128


def masked_asset_view(
    image: Image.Image,
    root_mask: np.ndarray,
    *,
    padding_ratio: float = 0.08,
) -> Image.Image:
    """Crop one physical root and suppress unrelated context."""

    image = image.convert("RGB")
    if root_mask.shape != (image.height, image.width):
        raise ValueError("root mask must match the source image")
    ys, xs = np.nonzero(root_mask)
    if not len(xs):
        raise ValueError("cannot route an empty object root")
    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)
    padding = max(2, round(max(x1 - x0, y1 - y0) * padding_ratio))
    x0, y0 = max(0, x0 - padding), max(0, y0 - padding)
    x1, y1 = min(image.width, x1 + padding), min(image.height, y1 + padding)
    rgb = np.asarray(image, dtype=np.uint8)[y0:y1, x0:x1]
    local_root = root_mask[y0:y1, x0:x1]
    muted = np.full_like(rgb, 127)
    muted[local_root] = rgb[local_root]
    return Image.fromarray(muted, mode="RGB")


def _label_prompts(label: str, aliases: Sequence[str]) -> tuple[str, ...]:
    names = tuple(
        dict.fromkeys(
            re.sub(r"\s+", " ", value.replace("_", " ")).strip().lower()
            for value in (label, *aliases)
            if value.strip()
        )
    )
    prompts: list[str] = []
    for name in names:
        prompts.extend(
            (
                f"a photo of one {name}",
                f"a product image of a {name}",
                f"the complete {name} object",
            )
        )
    return tuple(dict.fromkeys(prompts))


class Siglip2AssetEncoder:
    """SigLIP 2 image/text embeddings for automatic object-category routing."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "cuda",
        local_files_only: bool = False,
        batch_size: int = 16,
    ) -> None:
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as error:
            raise RuntimeError(
                "Install the foundation extra before enabling SigLIP 2 routing"
            ) from error
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.processor = AutoProcessor.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        self.model = AutoModel.from_pretrained(
            model_name, local_files_only=local_files_only
        ).to(device)
        self.model.eval()

    def activate(self, device: str | None = None) -> None:
        if device is not None:
            self.device = device
        self.model.to(self.device)
        self.model.eval()

    def release(self) -> None:
        if self.device.startswith("cuda"):
            self.model.to("cpu")
            torch.cuda.empty_cache()

    @staticmethod
    def _tensor(output: object) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        for name in ("pooler_output", "image_embeds", "text_embeds"):
            value = getattr(output, name, None)
            if isinstance(value, torch.Tensor):
                return value
        raise RuntimeError("image-text encoder did not return an embedding tensor")

    def encode_images(self, images: Sequence[Image.Image]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            inputs = self.processor(
                images=[
                    image.convert("RGB")
                    for image in images[start : start + self.batch_size]
                ],
                return_tensors="pt",
            ).to(self.device)
            with (
                torch.inference_mode(),
                torch.amp.autocast("cuda", enabled=self.device.startswith("cuda")),
            ):
                output = self.model.get_image_features(**inputs)
            rows.append(self._tensor(output).float().cpu().numpy())
        return (
            _normalize_rows(np.concatenate(rows, axis=0))
            if rows
            else np.zeros((0, 0), dtype=np.float32)
        )

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            inputs = self.processor(
                text=list(texts[start : start + self.batch_size]),
                padding="max_length",
                return_tensors="pt",
            ).to(self.device)
            with (
                torch.inference_mode(),
                torch.amp.autocast("cuda", enabled=self.device.startswith("cuda")),
            ):
                output = self.model.get_text_features(**inputs)
            rows.append(self._tensor(output).float().cpu().numpy())
        return (
            _normalize_rows(np.concatenate(rows, axis=0))
            if rows
            else np.zeros((0, 0), dtype=np.float32)
        )


def build_asset_routing_index(
    reference_manifest: Path,
    output_dir: Path,
    encoder: ImageTextEncoder,
) -> dict[str, object]:
    """Build a train-only category router from independently reviewed roots."""

    payload = json.loads(reference_manifest.read_text(encoding="utf-8-sig"))
    entries = [entry for entry in payload.get("entries", []) if entry.get("reviewed")]
    if not entries:
        raise ValueError("reference manifest contains no reviewed assets")
    base = reference_manifest.parent
    views: list[Image.Image] = []
    asset_rows: list[dict[str, object]] = []
    aliases_by_label: dict[str, set[str]] = defaultdict(set)
    metadata_by_label: dict[str, tuple[str, str | None]] = {}
    source_paths: list[Path] = [reference_manifest]
    for entry in entries:
        case_path = _resolve(base, str(entry["paco_case"]))
        image_path = case_path.parent / "source_crop.png"
        root_path = case_path.parent / "object_mask_crop.png"
        image = Image.open(image_path).convert("RGB")
        root = _load_mask(root_path)
        views.append(masked_asset_view(image, root))
        label = str(entry["asset_label"])
        aliases = tuple(str(value) for value in entry.get("aliases", []))
        aliases_by_label[label].update(aliases)
        metadata_by_label[label] = (
            str(entry["asset_domain"]),
            str(entry.get("asset_profile")) if entry.get("asset_profile") else None,
        )
        asset_rows.append(
            {
                "asset_id": str(entry["asset_id"]),
                "asset_label": label,
                "asset_domain": str(entry["asset_domain"]),
                "asset_profile": entry.get("asset_profile"),
                "image_id": entry.get("image_id"),
                "object_annotation_id": entry.get("object_annotation_id"),
            }
        )
        source_paths.extend((case_path, image_path, root_path))
    image_embeddings = encoder.encode_images(views)

    labels = sorted(aliases_by_label)
    text_prompts: list[str] = []
    prompt_ranges: dict[str, tuple[int, int]] = {}
    for label in labels:
        start = len(text_prompts)
        text_prompts.extend(_label_prompts(label, sorted(aliases_by_label[label])))
        prompt_ranges[label] = (start, len(text_prompts))
    prompt_embeddings = encoder.encode_texts(text_prompts)
    text_embeddings = _normalize_rows(
        np.stack(
            [
                prompt_embeddings[start:end].mean(axis=0)
                for start, end in (prompt_ranges[label] for label in labels)
            ],
            axis=0,
        )
    )

    label_asset_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(asset_rows):
        label_asset_indices[str(row["asset_label"])].append(index)
    label_prototypes = _normalize_rows(
        np.stack(
            [
                image_embeddings[label_asset_indices[label]].mean(axis=0)
                for label in labels
            ],
            axis=0,
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = output_dir / "router_arrays.npz"
    np.savez_compressed(
        arrays_path,
        image_embeddings=image_embeddings.astype(np.float32),
        text_embeddings=text_embeddings.astype(np.float32),
        label_prototypes=label_prototypes.astype(np.float32),
    )
    manifest = {
        "format": "HPID train-only SigLIP 2 asset router",
        "format_version": "0.1.0",
        "encoder_model_name": encoder.model_name,
        "reference_manifest": str(reference_manifest.resolve()),
        "reference_manifest_sha256": _sha256(reference_manifest),
        "arrays_file": arrays_path.name,
        "arrays_sha256": _sha256(arrays_path),
        "asset_count": len(asset_rows),
        "label_count": len(labels),
        "labels": labels,
        "label_metadata": {
            label: {
                "asset_domain": metadata_by_label[label][0],
                "asset_profile": metadata_by_label[label][1],
                "aliases": sorted(aliases_by_label[label]),
                "asset_indices": label_asset_indices[label],
                "text_prompts": text_prompts[slice(*prompt_ranges[label])],
            }
            for label in labels
        },
        "assets": asset_rows,
        "source_bundle_sha256": hashlib.sha256(
            "".join(
                _sha256(path) for path in sorted(set(source_paths), key=str)
            ).encode()
        ).hexdigest(),
        "ground_truth_used_for_offline_index_build": True,
        "ground_truth_available_at_inference": False,
    }
    (output_dir / "index.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


class AssetRoutingIndex:
    def __init__(self, root: Path, manifest: dict[str, object], arrays: object) -> None:
        self.root = root
        self.manifest = manifest
        self.labels = tuple(str(value) for value in manifest["labels"])
        self.assets = tuple(dict(value) for value in manifest["assets"])
        self.label_metadata = {
            str(key): dict(value)
            for key, value in dict(manifest["label_metadata"]).items()
        }
        self.image_embeddings = np.asarray(arrays["image_embeddings"], dtype=np.float32)
        self.text_embeddings = np.asarray(arrays["text_embeddings"], dtype=np.float32)
        self.label_prototypes = np.asarray(arrays["label_prototypes"], dtype=np.float32)

    @classmethod
    def load(cls, root: Path) -> AssetRoutingIndex:
        manifest_path = root / "index.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        arrays_path = root / str(manifest["arrays_file"])
        if _sha256(arrays_path) != str(manifest["arrays_sha256"]):
            raise ValueError("asset router array checksum mismatch")
        return cls(root, manifest, np.load(arrays_path, allow_pickle=False))


class AssetRouter:
    """Fuse zero-shot language and train-only visual prototypes for one root."""

    def __init__(
        self,
        index: AssetRoutingIndex,
        encoder: ImageTextEncoder,
        *,
        config: AssetRouterConfig | None = None,
    ) -> None:
        if encoder.model_name != str(index.manifest["encoder_model_name"]):
            raise ValueError("asset router encoder does not match its index")
        self.index = index
        self.encoder = encoder
        self.config = config or AssetRouterConfig()
        total = self.config.prototype_weight + self.config.text_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError("asset router score weights must sum to one")
        if not 0.0 <= self.config.nearest_asset_weight <= 1.0:
            raise ValueError("nearest asset weight must be in [0, 1]")
        if self.config.maximum_candidate_labels < 1:
            raise ValueError("maximum candidate labels must be positive")
        if self.config.maximum_candidate_score_drop < 0.0:
            raise ValueError("maximum candidate score drop must be non-negative")

    def encode_root(self, image: Image.Image, root_mask: np.ndarray) -> np.ndarray:
        """Encode one cleaned object root for every downstream routing decision."""

        return self.encoder.encode_images([masked_asset_view(image, root_mask)])[0]

    def route_embedding(self, embedding: np.ndarray) -> AssetRoute:
        """Route a precomputed root embedding without repeating image inference."""

        text_scores = self.index.text_embeddings @ embedding
        prototype_scores = self.index.label_prototypes @ embedding
        scores = (
            self.config.prototype_weight * prototype_scores
            + self.config.text_weight * text_scores
        )
        rows: list[dict[str, object]] = []
        for label_index, label in enumerate(self.index.labels):
            metadata = self.index.label_metadata[label]
            asset_indices = [int(value) for value in metadata["asset_indices"]]
            nearest = sorted(
                (
                    float(self.index.image_embeddings[index] @ embedding),
                    index,
                )
                for index in asset_indices
            )[-self.config.nearest_asset_count :]
            nearest_score = float(np.mean([value for value, _ in nearest]))
            final_score = (1.0 - self.config.nearest_asset_weight) * float(
                scores[label_index]
            ) + self.config.nearest_asset_weight * nearest_score
            rows.append(
                {
                    "asset_label": label,
                    "asset_domain": metadata["asset_domain"],
                    "asset_profile": metadata.get("asset_profile"),
                    "score": final_score,
                    "prototype_score": float(prototype_scores[label_index]),
                    "text_score": float(text_scores[label_index]),
                    "nearest_asset_score": nearest_score,
                    "nearest_asset_ids": [
                        self.index.assets[index]["asset_id"]
                        for _, index in reversed(nearest)
                    ],
                }
            )
        rows.sort(key=lambda row: float(row["score"]), reverse=True)
        top = rows[0]
        margin = float(top["score"]) - float(rows[1]["score"])
        accepted = (
            float(top["score"]) >= self.config.minimum_score
            and margin >= self.config.minimum_margin
        )
        candidate_rows: list[dict[str, object]] = []
        if accepted:
            candidate_rows = [top]
        elif float(top["score"]) >= self.config.minimum_score:
            candidate_rows = [
                row
                for row in rows
                if float(top["score"]) - float(row["score"])
                <= self.config.maximum_candidate_score_drop
            ][: self.config.maximum_candidate_labels]
        reason = (
            "accepted_exact_label"
            if accepted
            else "ambiguous_candidate_set"
            if candidate_rows
            else "open_set_score_rejection"
        )
        return AssetRoute(
            accepted=accepted,
            asset_label=str(top["asset_label"]) if accepted else None,
            asset_domain=str(top["asset_domain"]) if accepted else None,
            asset_profile=(
                str(top["asset_profile"])
                if accepted and top.get("asset_profile") is not None
                else None
            ),
            score=float(top["score"]),
            margin=margin,
            alternatives=tuple(rows[: self.config.maximum_alternatives]),
            candidate_labels=tuple(str(row["asset_label"]) for row in candidate_rows),
            candidate_domains=tuple(
                dict.fromkeys(str(row["asset_domain"]) for row in candidate_rows)
            ),
            reason=reason,
        )

    def route(self, image: Image.Image, root_mask: np.ndarray) -> AssetRoute:
        return self.route_embedding(self.encode_root(image, root_mask))


def route_profile_text_inventory(
    embedding: np.ndarray,
    domain: DomainPrompt,
    encoder: ImageTextEncoder,
    *,
    config: ProfileTextRouterConfig | None = None,
) -> ProfileTextRoute:
    """Select a physical-part inventory with one bounded batch of text scores.

    This stage only confirms an object profile. It never creates masks or Part IDs,
    and a weak or near-tied result stays unresolved.
    """

    config = config or ProfileTextRouterConfig()
    profiles = domain.part_profiles
    if not profiles:
        return ProfileTextRoute(
            accepted=False,
            profile=None,
            score=0.0,
            margin=0.0,
            alternatives=(),
            reason="no_profile_inventory",
        )
    prompts = [
        profile.classifier_prompt
        or " or ".join(profile.root_hints)
        or profile.name.replace("_", " ")
        for profile in profiles
    ]
    text_embeddings = encoder.encode_texts(prompts)
    scores = text_embeddings @ np.asarray(embedding, dtype=np.float32)
    order = np.argsort(scores)[::-1]
    rows = tuple(
        {
            "profile": profiles[int(index)].name,
            "score": float(scores[int(index)]),
            "classifier_prompt": prompts[int(index)],
        }
        for index in order[: config.maximum_alternatives]
    )
    top_score = float(scores[int(order[0])])
    runner_up = float(scores[int(order[1])]) if len(order) > 1 else -1.0
    margin = top_score - runner_up
    accepted = bool(
        top_score >= config.minimum_score and margin >= config.minimum_margin
    )
    return ProfileTextRoute(
        accepted=accepted,
        profile=profiles[int(order[0])].name if accepted else None,
        score=top_score,
        margin=margin,
        alternatives=rows,
        reason=(
            "accepted_profile_inventory"
            if accepted
            else "ambiguous_profile_inventory"
            if top_score >= config.minimum_score
            else "open_set_profile_rejection"
        ),
    )


def route_profile_text_inventories(
    embedding: np.ndarray,
    domains: Sequence[DomainPrompt],
    encoder: ImageTextEncoder,
    *,
    config: ProfileTextRouterConfig | None = None,
) -> dict[str, ProfileTextRoute]:
    """Score every domain inventory in one text batch, then route per domain.

    Domain routing and profile routing are deliberately separate.  The caller can
    therefore choose the inventory belonging to the final reconciled domain rather
    than accidentally retaining the inventory of an earlier broad-domain guess.
    """

    config = config or ProfileTextRouterConfig()
    rows = [
        (
            domain.name,
            profile.name,
            profile.classifier_prompt
            or " or ".join(profile.root_hints)
            or profile.name.replace("_", " "),
        )
        for domain in domains
        for profile in domain.part_profiles
    ]
    if not rows:
        return {}
    text_embeddings = encoder.encode_texts([row[2] for row in rows])
    scores = text_embeddings @ np.asarray(embedding, dtype=np.float32)
    output: dict[str, ProfileTextRoute] = {}
    for domain in domains:
        indices = [index for index, row in enumerate(rows) if row[0] == domain.name]
        if not indices:
            continue
        local_order = sorted(indices, key=lambda index: float(scores[index]), reverse=True)
        alternatives = tuple(
            {
                "profile": rows[index][1],
                "score": float(scores[index]),
                "classifier_prompt": rows[index][2],
            }
            for index in local_order[: config.maximum_alternatives]
        )
        top_index = local_order[0]
        top_score = float(scores[top_index])
        runner_up = float(scores[local_order[1]]) if len(local_order) > 1 else -1.0
        margin = top_score - runner_up
        accepted = bool(
            top_score >= config.minimum_score and margin >= config.minimum_margin
        )
        output[domain.name] = ProfileTextRoute(
            accepted=accepted,
            profile=rows[top_index][1] if accepted else None,
            score=top_score,
            margin=margin,
            alternatives=alternatives,
            reason=(
                "accepted_profile_inventory"
                if accepted
                else "ambiguous_profile_inventory"
                if top_score >= config.minimum_score
                else "open_set_profile_rejection"
            ),
        )
    return output


def profile_text_route_to_dict(route: ProfileTextRoute) -> dict[str, object]:
    return asdict(route)


def resolve_asset_domain(
    route: AssetRoute,
    current_domain: str,
    *,
    config: AssetRouterConfig | None = None,
) -> AssetDomainResolution:
    """Resolve a broad domain without inventing an ambiguous asset label."""

    config = config or AssetRouterConfig()
    if route.accepted and route.asset_domain is not None:
        return AssetDomainResolution(
            accepted=True,
            resolved_domain=route.asset_domain,
            resolved_profile=route.asset_profile,
            reason="accepted_exact_asset_domain",
            support_count=1,
            candidate_count=1,
            support_ratio=1.0,
            vote_margin=1,
            domain_votes=(
                {
                    "asset_domain": route.asset_domain,
                    "count": 1,
                    "score_sum": route.score,
                    "best_score": route.score,
                },
            ),
            resolved_asset_label=route.asset_label,
            asset_label_reason="accepted_exact_asset_label",
        )
    candidate_labels = set(route.candidate_labels)
    candidate_rows = [
        row
        for row in route.alternatives
        if str(row.get("asset_label", "")) in candidate_labels
    ]
    if not candidate_rows:
        return AssetDomainResolution(
            accepted=False,
            resolved_domain=None,
            resolved_profile=None,
            reason="no_asset_domain_evidence",
            support_count=0,
            candidate_count=0,
            support_ratio=0.0,
            vote_margin=0,
            domain_votes=(),
        )
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in candidate_rows:
        grouped[str(row["asset_domain"])].append(float(row["score"]))
    votes = sorted(
        (
            {
                "asset_domain": domain,
                "count": len(scores),
                "score_sum": float(sum(scores)),
                "best_score": float(max(scores)),
            }
            for domain, scores in grouped.items()
        ),
        key=lambda row: (
            int(row["count"]),
            float(row["score_sum"]),
            float(row["best_score"]),
            str(row["asset_domain"]),
        ),
        reverse=True,
    )
    best_candidate_score = max(float(row["score"]) for row in candidate_rows)
    current_vote = next(
        (row for row in votes if row["asset_domain"] == current_domain),
        None,
    )
    if (
        current_vote is not None
        and best_candidate_score - float(current_vote["best_score"])
        <= config.maximum_current_domain_score_gap
    ):
        support_count = int(current_vote["count"])
        candidate_count = len(candidate_rows)
        runner_up_count = max(
            (
                int(row["count"])
                for row in votes
                if row["asset_domain"] != current_domain
            ),
            default=0,
        )
        return AssetDomainResolution(
            accepted=True,
            resolved_domain=current_domain,
            resolved_profile=None,
            reason="retained_current_domain_under_ambiguity",
            support_count=support_count,
            candidate_count=candidate_count,
            support_ratio=support_count / max(1, candidate_count),
            vote_margin=support_count - runner_up_count,
            domain_votes=tuple(votes),
            resolved_asset_label=None,
            asset_label_reason="ambiguous_route_not_promoted_to_exact_label",
        )
    winner = votes[0]
    support_count = int(winner["count"])
    candidate_count = len(candidate_rows)
    support_ratio = support_count / max(1, candidate_count)
    runner_up_count = int(votes[1]["count"]) if len(votes) > 1 else 0
    vote_margin = support_count - runner_up_count
    single_domain = len(votes) == 1
    accepted = bool(
        support_ratio >= config.minimum_domain_consensus_ratio
        and (
            single_domain
            or support_count >= config.minimum_domain_consensus_votes
            and vote_margin >= config.minimum_domain_vote_margin
        )
    )
    winning_domain = str(winner["asset_domain"])
    reason = (
        "accepted_candidate_domain_consensus"
        if accepted
        else "ambiguous_cross_domain_candidates"
    )
    return AssetDomainResolution(
        accepted=accepted,
        resolved_domain=winning_domain if accepted else None,
        resolved_profile=None,
        reason=reason,
        support_count=support_count,
        candidate_count=candidate_count,
        support_ratio=float(support_ratio),
        vote_margin=vote_margin,
        domain_votes=tuple(votes),
    )


def reconcile_asset_routes(
    full_image_route: AssetRoute | None,
    root_crop_route: AssetRoute,
    *,
    config: AssetRouterConfig | None = None,
    root_global_proposal_rank: int | None = None,
) -> tuple[AssetRoute, dict[str, object]]:
    """Fuse category evidence from the full image and the selected root crop.

    The full-image view is better at naming the salient asset, while the masked
    root view is better at rejecting contextual objects. An exact label is
    promoted only when both views support the same label, or when the full-image
    label is effectively tied for first in the root crop and the selected root
    came from that rank-one global proposal. No annotation or target mask enters
    this decision.
    """

    config = config or AssetRouterConfig()
    diagnostics: dict[str, object] = {
        "algorithm": "hpid-cross-view-asset-consensus-v1",
        "status": "root_crop_only",
        "accepted": bool(root_crop_route.accepted),
        "full_image_label": None,
        "root_crop_label": None,
        "root_crop_support_rank": None,
        "root_crop_score_gap": None,
        "root_global_proposal_rank": root_global_proposal_rank,
        "ground_truth_used": False,
    }
    if full_image_route is None or not full_image_route.alternatives:
        return root_crop_route, diagnostics
    if not root_crop_route.alternatives:
        diagnostics["status"] = "missing_root_crop_evidence"
        return root_crop_route, diagnostics

    full_top = full_image_route.alternatives[0]
    crop_top = root_crop_route.alternatives[0]
    full_label = str(full_top.get("asset_label", ""))
    crop_label = str(crop_top.get("asset_label", ""))
    diagnostics["full_image_label"] = full_label or None
    diagnostics["root_crop_label"] = crop_label or None
    if not full_label:
        diagnostics["status"] = "missing_full_image_label"
        return root_crop_route, diagnostics

    crop_support = next(
        (
            (rank, row)
            for rank, row in enumerate(root_crop_route.alternatives, start=1)
            if str(row.get("asset_label", "")) == full_label
        ),
        None,
    )
    if crop_support is None:
        diagnostics["status"] = "cross_view_label_conflict"
        return root_crop_route, diagnostics

    crop_support_rank, crop_support_row = crop_support
    crop_top_score = float(crop_top.get("score", root_crop_route.score))
    crop_support_score = float(crop_support_row.get("score", 0.0))
    crop_score_gap = max(0.0, crop_top_score - crop_support_score)
    full_score = float(full_top.get("score", full_image_route.score))
    exact_top_agreement = crop_support_rank == 1
    rank_one_near_tie = bool(
        root_global_proposal_rank == 1
        and crop_score_gap <= config.maximum_cross_view_label_score_gap
    )
    accepted = bool(
        full_score >= config.minimum_score
        and crop_support_score >= config.minimum_score
        and (exact_top_agreement or rank_one_near_tie)
    )
    diagnostics.update(
        {
            "root_crop_support_rank": crop_support_rank,
            "root_crop_score_gap": crop_score_gap,
            "exact_top_agreement": exact_top_agreement,
            "rank_one_near_tie": rank_one_near_tie,
            "accepted": accepted,
        }
    )
    if not accepted:
        diagnostics["status"] = "insufficient_cross_view_consensus"
        return root_crop_route, diagnostics

    domain = str(
        full_top.get("asset_domain")
        or crop_support_row.get("asset_domain")
        or ""
    )
    profile_value = full_top.get("asset_profile") or crop_support_row.get(
        "asset_profile"
    )
    profile = str(profile_value) if profile_value else None
    if not domain:
        diagnostics["status"] = "consensus_missing_domain"
        diagnostics["accepted"] = False
        return root_crop_route, diagnostics

    combined_score = float((full_score + crop_support_score) / 2.0)
    support_row = {
        **dict(crop_support_row),
        **{
            key: value
            for key, value in dict(full_top).items()
            if key not in {"score"}
        },
        "asset_label": full_label,
        "asset_domain": domain,
        "asset_profile": profile,
        "score": combined_score,
        "full_image_score": full_score,
        "root_crop_score": crop_support_score,
        "cross_view_consensus": True,
    }
    remaining = tuple(
        row
        for row in root_crop_route.alternatives
        if str(row.get("asset_label", "")) != full_label
    )
    diagnostics["status"] = (
        "accepted_top_label_agreement"
        if exact_top_agreement
        else "accepted_rank_one_near_tie"
    )
    diagnostics["resolved_label"] = full_label
    diagnostics["resolved_domain"] = domain
    diagnostics["resolved_profile"] = profile
    return (
        AssetRoute(
            accepted=True,
            asset_label=full_label,
            asset_domain=domain,
            asset_profile=profile,
            score=combined_score,
            margin=max(
                0.0,
                combined_score
                - max(
                    (float(row.get("score", 0.0)) for row in remaining),
                    default=0.0,
                ),
            ),
            alternatives=(support_row, *remaining),
            candidate_labels=(full_label,),
            candidate_domains=(domain,),
            reason="accepted_cross_view_asset_consensus",
        ),
        diagnostics,
    )


def route_to_dict(route: AssetRoute) -> dict[str, object]:
    return asdict(route)
