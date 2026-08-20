from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
from PIL import Image, ImageDraw

from .fusion import MaskCandidate
from .prompt_bank import DomainPrompt
from .vlm_parts import make_region_query_image, region_exterior_contact_fraction


class PhysicalityPlanner(Protocol):
    backend_id: str

    def generate_response(self, image: Image.Image, prompt: str) -> str: ...


@dataclass(frozen=True)
class PhysicalRegionAuditConfig:
    maximum_queries: int = 4
    maximum_candidates: int = 24
    batch_size: int = 6
    minimum_root_area_fraction: float = 0.0008
    maximum_root_area_fraction: float = 0.40
    maximum_named_root_area_fraction: float = 0.90
    physical_acceptance_confidence: float = 0.70
    nonphysical_rejection_confidence: float = 0.85
    open_domains: tuple[str, ...] = (
        "character",
        "natural_object",
        "terrain",
        "structure",
    )


@dataclass(frozen=True)
class ParsedPhysicality:
    label: str
    confidence: float


@dataclass(frozen=True)
class PhysicalRegionAuditResult:
    candidates: tuple[MaskCandidate, ...]
    diagnostics: dict[str, object]


_LABELS = frozenset(
    {
        "physical_component",
        "surface_detail",
        "lighting_or_shadow",
        "background_or_noise",
        "uncertain",
    }
)


def _root_key(candidate: MaskCandidate) -> str:
    return (
        f"{candidate.metadata.get('root_origin', 'legacy')}::"
        f"{candidate.metadata.get('root_index', 'unknown')}"
    )


def _candidate_key(candidate: MaskCandidate) -> str:
    return str(candidate.metadata.get("candidate_key", ""))


def build_physical_region_audit_prompt(
    target_count: int,
    *,
    object_labels: Sequence[str] = (),
) -> str:
    object_context = "; ".join(
        f"{index}={label}" for index, label in enumerate(object_labels, start=1)
    )
    return f"""The contact sheet contains {target_count} numbered highlighted regions.
Each tile shows one region inside its complete object on the left and the same
region enlarged on the right. Classify the highlighted pixels, not nearby pixels.
Object context by target: {object_context or "not provided"}.

Use physical_component only for a bounded material or geometric subassembly that
can reasonably receive its own editable Part-ID, such as a handle, panel, button,
trigger, wheel, blade, garment piece, cap, or structural insert. A component need
not be detachable, but it must be a coherent physical region.
Use surface_detail for printed graphics, text, paint patches, texture, decals,
reflections, highlights, or color-only markings. Use lighting_or_shadow for
illumination or cast-shadow regions. Use background_or_noise for leakage, holes,
or unrelated pixels. Use uncertain instead of guessing.
For books, newspapers, posters, screens, and packages, printed text, pictures,
icons, colored blocks, and page graphics are surface_detail, never a physical
component. A page, cover, bezel, key, or button can be physical_component.

Return one row for every TARGET number and no extra rows. Return JSON only:
{{"objects":[{{"id":1,"label":"physical_component|surface_detail|lighting_or_shadow|background_or_noise|uncertain","certainty":"high|medium|low"}}]}}
"""


def _confidence(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(np.clip(float(value), 0.0, 1.0))
    return {
        "high": 0.95,
        "medium": 0.70,
        "low": 0.25,
    }.get(str(value or "").strip().casefold(), 0.0)


def parse_physical_region_audit_response(
    response: str,
    *,
    expected_count: int,
) -> tuple[dict[int, ParsedPhysicality], dict[str, object]]:
    payload: dict[str, object] | None = None
    decoder = json.JSONDecoder()
    for index, character in enumerate(response):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(response[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            payload = parsed
            break
    raw_rows = payload.get("objects") if payload is not None else None
    if not isinstance(raw_rows, list):
        return {}, {
            "status": "invalid_batch_json",
            "response_character_count": len(response),
            "ground_truth_used": False,
        }
    parsed_by_id: dict[int, ParsedPhysicality] = {}
    invalid_count = 0
    duplicate_count = 0
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            invalid_count += 1
            continue
        try:
            target_id = int(raw_row.get("id", 0))
        except (TypeError, ValueError):
            target_id = 0
        if not 1 <= target_id <= expected_count:
            invalid_count += 1
            continue
        label = re.sub(
            r"[^a-z]+", "_", str(raw_row.get("label", "uncertain")).casefold()
        ).strip("_")
        if label not in _LABELS:
            label = "uncertain"
            invalid_count += 1
        parsed = ParsedPhysicality(label, _confidence(raw_row.get("certainty")))
        incumbent = parsed_by_id.get(target_id)
        if incumbent is not None:
            duplicate_count += 1
            if incumbent.confidence >= parsed.confidence:
                continue
        parsed_by_id[target_id] = parsed
    return parsed_by_id, {
        "status": "accepted",
        "expected_count": expected_count,
        "parsed_count": len(parsed_by_id),
        "missing_count": expected_count - len(parsed_by_id),
        "invalid_count": invalid_count,
        "duplicate_count": duplicate_count,
        "response_character_count": len(response),
        "ground_truth_used": False,
    }


def make_physical_region_contact_sheet(
    image: Image.Image,
    targets: Sequence[tuple[MaskCandidate, MaskCandidate]],
    *,
    columns: int = 2,
    tile_width: int = 384,
    panel_height: int = 192,
    header_height: int = 28,
) -> Image.Image:
    if not targets:
        raise ValueError("physical-region contact sheet needs at least one target")
    columns = max(1, min(columns, len(targets)))
    rows = (len(targets) + columns - 1) // columns
    tile_height = header_height + panel_height
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(canvas)
    for target_id, (root, candidate) in enumerate(targets, start=1):
        panel = make_region_query_image(
            image,
            root_mask=root.mask,
            region_mask=candidate.mask,
        ).resize((tile_width, panel_height), Image.Resampling.LANCZOS)
        column = (target_id - 1) % columns
        row = (target_id - 1) // columns
        x0 = column * tile_width
        y0 = row * tile_height
        canvas.paste(panel, (x0, y0 + header_height))
        draw.rectangle(
            (x0, y0, x0 + tile_width - 1, y0 + header_height - 1),
            fill="white",
        )
        object_label = str(
            root.metadata.get("resolved_object_label")
            or root.metadata.get("root_model_label")
            or root.prompt
            or root.semantic_name
        ).replace("_", " ")
        draw.text(
            (x0 + 8, y0 + 7),
            f"TARGET {target_id} | {object_label[:32]}",
            fill="black",
        )
        draw.rectangle(
            (x0, y0, x0 + tile_width - 1, y0 + tile_height - 1),
            outline=(80, 80, 80),
            width=2,
        )
    return canvas


def _eligible_candidates(
    roots: Sequence[MaskCandidate],
    candidates: Sequence[MaskCandidate],
    domains: Mapping[str, DomainPrompt],
    config: PhysicalRegionAuditConfig,
) -> list[tuple[MaskCandidate, MaskCandidate, float]]:
    roots_by_key = {_root_key(root): root for root in roots}
    eligible: list[tuple[MaskCandidate, MaskCandidate, float]] = []
    for candidate in candidates:
        generic = bool(candidate.metadata.get("generic_visual_region"))
        named_visual = bool(candidate.metadata.get("visual_region")) and not generic
        if not generic and not named_visual:
            continue
        root = roots_by_key.get(_root_key(candidate))
        if root is None or root.semantic_name in config.open_domains:
            continue
        if root.semantic_name not in domains:
            continue
        if not str(root.metadata.get("selected_part_profile") or "").strip():
            continue
        fraction = float(candidate.metadata.get("root_area_fraction", 0.0))
        maximum_fraction = (
            config.maximum_root_area_fraction
            if generic
            else config.maximum_named_root_area_fraction
        )
        if not (
            config.minimum_root_area_fraction
            <= fraction
            <= maximum_fraction
        ):
            continue
        if not _candidate_key(candidate):
            continue
        exterior = region_exterior_contact_fraction(candidate.mask, root.mask)
        kind = str(candidate.metadata.get("visual_region_kind", "panel"))
        priority = (
            1.2 * float(named_visual)
            + 1.6 * float(kind == "detail")
            + 0.8 * (1.0 - exterior)
            + 0.8 * candidate.score
            + 0.4 * min(1.0, fraction / 0.08)
        )
        eligible.append((root, candidate, priority))
    eligible.sort(key=lambda row: row[2], reverse=True)
    maximum = min(
        config.maximum_candidates,
        config.maximum_queries * max(1, config.batch_size),
    )
    return eligible[:maximum]


class PhysicalRegionAuditor:
    """Audit SAM regions, including semantic reranks, without assigning IDs."""

    def __init__(
        self,
        planner: PhysicalityPlanner,
        *,
        config: PhysicalRegionAuditConfig | None = None,
    ) -> None:
        self.planner = planner
        self.config = config or PhysicalRegionAuditConfig()

    def audit(
        self,
        image: Image.Image,
        roots: Sequence[MaskCandidate],
        candidates: Sequence[MaskCandidate],
        domains: Mapping[str, DomainPrompt],
    ) -> PhysicalRegionAuditResult:
        eligible = _eligible_candidates(roots, candidates, domains, self.config)
        evidence_by_key: dict[str, dict[str, object]] = {}
        rows: list[dict[str, object]] = []
        query_count = 0
        batch_size = max(1, self.config.batch_size)
        for offset in range(0, len(eligible), batch_size):
            if query_count >= self.config.maximum_queries:
                break
            batch = eligible[offset : offset + batch_size]
            query_count += 1
            targets = [(root, candidate) for root, candidate, _ in batch]
            object_labels = [
                str(
                    root.metadata.get("resolved_object_label")
                    or root.metadata.get("root_model_label")
                    or root.prompt
                    or root.semantic_name
                ).replace("_", " ")
                for root, _ in targets
            ]
            try:
                response = self.planner.generate_response(
                    make_physical_region_contact_sheet(image, targets),
                    build_physical_region_audit_prompt(
                        len(batch), object_labels=object_labels
                    ),
                )
                parsed_by_id, parse = parse_physical_region_audit_response(
                    response,
                    expected_count=len(batch),
                )
            except (RuntimeError, ValueError, OSError, TypeError, KeyError) as error:
                for root, candidate, priority in batch:
                    rows.append(
                        {
                            "root_key": _root_key(root),
                            "candidate_key": _candidate_key(candidate),
                            "status": "planner_error",
                            "error_type": type(error).__name__,
                            "priority": priority,
                            "batch_query_index": query_count,
                            "ground_truth_used": False,
                        }
                    )
                continue
            for target_id, (root, candidate, priority) in enumerate(batch, start=1):
                parsed = parsed_by_id.get(target_id)
                if parsed is None:
                    rows.append(
                        {
                            "root_key": _root_key(root),
                            "candidate_key": _candidate_key(candidate),
                            "status": "missing_batch_prediction",
                            "priority": priority,
                            "batch_query_index": query_count,
                            "parse": parse,
                            "ground_truth_used": False,
                        }
                    )
                    continue
                if (
                    parsed.label == "physical_component"
                    and parsed.confidence
                    >= self.config.physical_acceptance_confidence
                ):
                    decision = "physical_supported"
                elif (
                    parsed.label
                    in {
                        "surface_detail",
                        "lighting_or_shadow",
                        "background_or_noise",
                    }
                    and parsed.confidence
                    >= self.config.nonphysical_rejection_confidence
                ):
                    decision = "nonphysical_supported"
                else:
                    decision = "uncertain"
                evidence = {
                    "algorithm": "hpid-vlm-region-physicality-audit-v1",
                    "planner_backend": self.planner.backend_id,
                    "label": parsed.label,
                    "confidence": parsed.confidence,
                    "decision": decision,
                    "batch_query_index": query_count,
                    "ground_truth_used": False,
                }
                evidence_by_key[_candidate_key(candidate)] = evidence
                rows.append(
                    {
                        "root_key": _root_key(root),
                        "candidate_key": _candidate_key(candidate),
                        "priority": priority,
                        **evidence,
                    }
                )
        output = tuple(
            replace(
                candidate,
                metadata={
                    **candidate.metadata,
                    "vlm_physicality_audit": evidence_by_key[
                        _candidate_key(candidate)
                    ],
                },
            )
            if _candidate_key(candidate) in evidence_by_key
            else candidate
            for candidate in candidates
        )
        return PhysicalRegionAuditResult(
            output,
            {
                "algorithm": "hpid-vlm-region-physicality-audit-v1",
                "planner_backend": self.planner.backend_id,
                "eligible_candidate_count": len(eligible),
                "query_count": query_count,
                "audited_candidate_count": len(evidence_by_key),
                "physical_supported_count": sum(
                    row.get("decision") == "physical_supported" for row in rows
                ),
                "nonphysical_supported_count": sum(
                    row.get("decision") == "nonphysical_supported" for row in rows
                ),
                "uncertain_count": sum(
                    row.get("decision") == "uncertain" for row in rows
                ),
                "rows": rows,
                "ground_truth_used": False,
            },
        )
