from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np
from PIL import Image

from .fusion import MaskCandidate, mask_iou
from .prompt_bank import DomainPrompt, PartPrompt
from .vlm_parts import make_region_query_image


class SemanticAuditPlanner(Protocol):
    backend_id: str

    def generate_response(self, image: Image.Image, prompt: str) -> str: ...


@dataclass(frozen=True)
class SemanticCandidateAuditConfig:
    maximum_queries: int = 4
    maximum_candidate_score: float = 0.50
    invalidation_minimum_confidence: float = 0.85
    cluster_iou: float = 0.52
    cluster_containment: float = 0.82
    crop_padding_ratio: float = 0.08


@dataclass(frozen=True)
class SemanticCandidateAuditResult:
    candidates: tuple[MaskCandidate, ...]
    diagnostics: dict[str, object]


def _root_key(candidate: MaskCandidate) -> str:
    return (
        f"{candidate.metadata.get('root_origin', 'legacy')}::"
        f"{candidate.metadata.get('root_index', 'unknown')}"
    )


def _candidate_key(candidate: MaskCandidate) -> str:
    return str(candidate.metadata.get("candidate_key", ""))


def _mask_containment(first: np.ndarray, second: np.ndarray) -> float:
    first_area = int(np.count_nonzero(first))
    second_area = int(np.count_nonzero(second))
    smaller = min(first_area, second_area)
    if smaller <= 0:
        return 0.0
    return int(np.count_nonzero(first & second)) / smaller


def _extract_json_object(text: str) -> dict[str, object] | None:
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _confidence(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(np.clip(float(value), 0.0, 1.0))
    normalized = str(value or "").strip().lower()
    return {
        "high": 0.95,
        "medium": 0.70,
        "low": 0.25,
    }.get(normalized, 0.0)


def parse_semantic_candidate_audit(text: str) -> tuple[str, float, dict[str, object]]:
    payload = _extract_json_object(text)
    if payload is None:
        return "uncertain", 0.0, {"status": "json_not_found", "raw": text}
    normalized = re.sub(
        r"[^a-z]+", "_", str(payload.get("verdict", "uncertain")).lower()
    ).strip("_")
    aliases = {
        "yes": "correct",
        "match": "correct",
        "correct": "correct",
        "no": "wrong",
        "incorrect": "wrong",
        "wrong": "wrong",
        "mixed": "wrong",
        "whole_object": "wrong",
        "other_object": "wrong",
        "background": "wrong",
        "unknown": "uncertain",
        "uncertain": "uncertain",
    }
    verdict = aliases.get(normalized, "uncertain")
    confidence = _confidence(payload.get("confidence"))
    return verdict, confidence, {
        "status": "parsed",
        "verdict_raw": payload.get("verdict"),
        "reason_code": payload.get("reason_code"),
        "confidence_raw": payload.get("confidence"),
    }


def build_semantic_candidate_audit_prompt(
    *,
    object_label: str,
    candidate: MaskCandidate,
    part: PartPrompt,
) -> str:
    description = part.planner_description or part.prompts[0]
    exclusions = ", ".join(part.planner_exclusions) or "none listed"
    return f"""Audit one proposed part mask inside a {object_label}.
The left panel shows the complete target object with the proposed region outlined
in red; the right panel enlarges exactly the same region. Judge the highlighted
physical pixels, not merely nearby context.

Candidate semantic ID: {candidate.semantic_name}
Expected part: {description}
Explicit exclusions: {exclusions}

Use correct only when the highlighted mask mostly follows that exact physical
part. Use wrong when it is mainly the whole object, main housing, an operator,
background, a different part, or an inseparable mixture. Use uncertain when the
image does not support a reliable decision. Be conservative.

Return exactly one JSON object:
{{"verdict":"correct|wrong|uncertain","confidence":"high|medium|low","reason_code":"exact_part|whole_object|main_body|other_part|operator_or_background|mixed|insufficient_detail"}}
"""


def _cluster_candidates(
    candidates: Sequence[MaskCandidate],
    config: SemanticCandidateAuditConfig,
) -> list[list[MaskCandidate]]:
    clusters: list[list[MaskCandidate]] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        for cluster in clusters:
            anchor = cluster[0]
            if candidate.semantic_name != anchor.semantic_name:
                continue
            if mask_iou(candidate.mask, anchor.mask) >= config.cluster_iou or (
                _mask_containment(candidate.mask, anchor.mask)
                >= config.cluster_containment
            ):
                cluster.append(candidate)
                break
        else:
            clusters.append([candidate])
    return clusters


def _suspicion_score(candidate: MaskCandidate, part: PartPrompt) -> float:
    fraction = float(candidate.metadata.get("root_area_fraction", 0.0))
    maximum = max(1e-6, part.maximum_parent_fraction)
    area_pressure = float(np.clip(fraction / maximum, 0.0, 1.5))
    detector_uncertainty = 1.0 - float(np.clip(candidate.score, 0.0, 1.0))
    single_source = not bool(candidate.metadata.get("multi_view_confirmed"))
    return detector_uncertainty + 0.45 * area_pressure + 0.08 * float(single_source)


class SemanticCandidateAuditor:
    """Conservatively invalidate low-confidence semantic labels, never masks."""

    def __init__(
        self,
        planner: SemanticAuditPlanner,
        *,
        config: SemanticCandidateAuditConfig | None = None,
    ) -> None:
        self.planner = planner
        self.config = config or SemanticCandidateAuditConfig()

    def audit(
        self,
        image: Image.Image,
        roots: Sequence[MaskCandidate],
        candidates: Sequence[MaskCandidate],
        domains: dict[str, DomainPrompt],
    ) -> SemanticCandidateAuditResult:
        roots_by_key = {_root_key(root): root for root in roots}
        parts_by_root: dict[str, dict[str, PartPrompt]] = {}
        object_labels: dict[str, str] = {}
        for root_key, root in roots_by_key.items():
            domain = domains.get(root.semantic_name)
            if domain is None:
                continue
            object_label = str(
                root.metadata.get("resolved_object_label")
                or root.metadata.get("root_model_label")
                or root.prompt
                or root.semantic_name.replace("_", " ")
            )
            profile_hint = root.metadata.get("selected_part_profile")
            parts, _, _ = domain.select_parts(
                object_label,
                profile_hint=str(profile_hint) if profile_hint is not None else None,
                profile_hint_source="semantic_candidate_audit",
            )
            parts_by_root[root_key] = {part.semantic_name: part for part in parts}
            object_labels[root_key] = object_label

        eligible: list[MaskCandidate] = []
        for candidate in candidates:
            root_key = _root_key(candidate)
            part = parts_by_root.get(root_key, {}).get(candidate.semantic_name)
            if part is None:
                continue
            if not bool(candidate.metadata.get("profile_refinement")):
                continue
            if candidate.score > self.config.maximum_candidate_score:
                continue
            if not _candidate_key(candidate):
                continue
            eligible.append(candidate)

        clusters = _cluster_candidates(eligible, self.config)
        clusters.sort(
            key=lambda cluster: _suspicion_score(
                cluster[0],
                parts_by_root[_root_key(cluster[0])][cluster[0].semantic_name],
            ),
            reverse=True,
        )
        clusters = clusters[: self.config.maximum_queries]
        invalidated_keys: set[str] = set()
        rows: list[dict[str, object]] = []
        for cluster in clusters:
            representative = cluster[0]
            root_key = _root_key(representative)
            root = roots_by_key[root_key]
            part = parts_by_root[root_key][representative.semantic_name]
            query_image = make_region_query_image(
                image,
                root_mask=root.mask,
                region_mask=representative.mask,
                padding_ratio=self.config.crop_padding_ratio,
            )
            prompt = build_semantic_candidate_audit_prompt(
                object_label=object_labels[root_key],
                candidate=representative,
                part=part,
            )
            row: dict[str, object] = {
                "root_key": root_key,
                "semantic_name": representative.semantic_name,
                "candidate_keys": [_candidate_key(item) for item in cluster],
                "candidate_score": float(representative.score),
                "root_area_fraction": float(
                    representative.metadata.get("root_area_fraction", 0.0)
                ),
                "ground_truth_used": False,
            }
            try:
                response = self.planner.generate_response(query_image, prompt)
            except (RuntimeError, ValueError, OSError, TypeError, KeyError) as error:
                row.update(
                    {
                        "status": "planner_error",
                        "error_type": type(error).__name__,
                    }
                )
                rows.append(row)
                continue
            verdict, confidence, parse = parse_semantic_candidate_audit(response)
            row.update(
                {
                    "verdict": verdict,
                    "confidence": confidence,
                    "parse": parse,
                    "status": "retained",
                }
            )
            if (
                verdict == "wrong"
                and confidence >= self.config.invalidation_minimum_confidence
            ):
                invalidated_keys.update(_candidate_key(item) for item in cluster)
                row["status"] = "invalidated_semantic_label"
            rows.append(row)

        output: list[MaskCandidate] = []
        genericized_count = 0
        generic_ordinals: dict[tuple[str, str], int] = {}
        for candidate in candidates:
            candidate_key = _candidate_key(candidate)
            support_key = str(
                candidate.metadata.get("semantic_support_candidate_key", "")
            )
            invalid_direct = candidate_key in invalidated_keys
            invalid_support = support_key in invalidated_keys
            if invalid_direct and not bool(candidate.metadata.get("visual_region")):
                continue
            if invalid_direct or invalid_support:
                root_key = _root_key(candidate)
                root = roots_by_key.get(root_key)
                if root is None or not bool(candidate.metadata.get("visual_region")):
                    continue
                kind = str(candidate.metadata.get("visual_region_kind", "panel"))
                ordinal_key = (root_key, kind)
                generic_ordinals[ordinal_key] = generic_ordinals.get(ordinal_key, 0) + 1
                semantic_name = (
                    f"{root.semantic_name}_visual_{kind}_audit_"
                    f"{generic_ordinals[ordinal_key]:02d}"
                )
                output.append(
                    replace(
                        candidate,
                        semantic_name=semantic_name,
                        semantic_parent=root.semantic_name,
                        prompt="automatic visual region after semantic audit",
                        source=f"{candidate.source}/semantic-audit-fallback",
                        source_reliability=min(candidate.source_reliability, 0.62),
                        metadata={
                            **candidate.metadata,
                            "generic_visual_region": True,
                            "semantic_audit_invalidated_previous_label": True,
                            "semantic_audit_previous_semantic": candidate.semantic_name,
                            "semantic_support_candidate_key": None,
                            "parent_candidate_key": root.metadata.get("candidate_key"),
                            "assembly_parent_semantic": root.semantic_name,
                            "assembly_parent_candidate_key": root.metadata.get(
                                "candidate_key"
                            ),
                            "ground_truth_used": False,
                        },
                    )
                )
                genericized_count += 1
                continue
            output.append(candidate)

        return SemanticCandidateAuditResult(
            tuple(output),
            {
                "algorithm": "hpid-vlm-semantic-candidate-audit-v1",
                "planner_backend": self.planner.backend_id,
                "eligible_candidate_count": len(eligible),
                "query_count": len(rows),
                "invalidated_candidate_count": len(invalidated_keys),
                "genericized_visual_support_count": genericized_count,
                "rows": rows,
                "ground_truth_used": False,
            },
        )
