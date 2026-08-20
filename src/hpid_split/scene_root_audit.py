from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from PIL import Image, ImageDraw

from .fusion import MaskCandidate
from .prompt_bank import DomainPrompt, PartProfile
from .root_routing import candidate_root_key
from .vlm_parts import VlmPlanner, make_root_query_image

_ROOT_LABEL_TARGETS: dict[str, tuple[str, str | None] | None] = {
    "character": ("character", None),
    "vehicle": ("vehicle", None),
    "furniture": ("furniture", None),
    "tool_or_weapon": ("tool_prop", None),
    "container": ("container", None),
    "device": ("device", None),
    "clothing_or_everyday_object": ("daily_object", None),
    "architecture": ("structure", "game_structure"),
    "natural_rock": ("natural_object", "rock"),
    "natural_tree": ("natural_object", "tree"),
    "natural_plant": ("natural_object", "bush"),
    "terrain_ground": ("terrain", "outdoor_level"),
    "terrain_cliff_or_rock_formation": ("terrain", "outdoor_level"),
    "other_object": None,
    "uncertain": None,
}

_MANMADE_SCENE_DOMAINS = frozenset(
    {
        "furniture",
        "tool_prop",
        "container",
        "device",
        "daily_object",
        "structure",
        "vehicle",
    }
)


@dataclass(frozen=True)
class SceneRootAuditConfig:
    maximum_queries: int = 12
    maximum_candidates: int = 48
    batch_size: int = 6
    minimum_confidence: float = 0.80
    minimum_priority: float = 4.0
    natural_scene_minimum_ratio: float = 0.55
    minority_domain_maximum_ratio: float = 0.24


@dataclass(frozen=True)
class ParsedRootAudit:
    label: str | None
    confidence: float
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class SceneRootAuditResult:
    roots: tuple[MaskCandidate, ...]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class CandidateAuditApplication:
    candidates: tuple[MaskCandidate, ...]
    diagnostics: dict[str, object]


def build_scene_root_audit_prompt() -> str:
    labels = ", ".join(_ROOT_LABEL_TARGETS)
    return (
        "The left panel shows the full scene with one red-outlined target. "
        "The right panel shows that same complete target enlarged. Classify the "
        "physical target itself, using scene context and not just its silhouette. "
        f"Choose exactly one label: {labels}. Classify the supporting physical "
        "body of the whole target, not a surface covering. A boulder or stone "
        "block is natural_rock even when rectangular. A connected or grass-topped "
        "game terrain block is terrain_cliff_or_rock_formation. Grass, moss, snow, "
        "paint, or foliage on top of stone is only a surface covering; do not call "
        "the supporting stone body natural_plant. Do not call stone furniture or "
        "a container merely because it is rectangular. Ignore text, watermarks, "
        "shadows, reflections, openings, and background. Return exactly one JSON "
        "object with certainty equal to high, medium, or low: "
        '{"label":"one_allowed_label","certainty":"high_or_medium_or_low"}.'
    )


def build_scene_root_batch_audit_prompt(target_count: int) -> str:
    labels = ", ".join(_ROOT_LABEL_TARGETS)
    return (
        f"The contact sheet contains {target_count} numbered physical targets. "
        "Each tile has scene context on the left and the same isolated target on "
        "the right. Classify every target independently. Use context, material, "
        "support, and neighboring objects rather than silhouette alone. Choose "
        f"only from these labels: {labels}. A boulder or stone block is "
        "natural_rock even when rectangular. A connected or grass-topped game "
        "terrain block is terrain_cliff_or_rock_formation. Grass, moss, snow, "
        "paint, and foliage are surface coverings, not the supporting object. "
        "Do not call stone furniture or a container merely because it is boxy. "
        "Return one row for every visible TARGET number and no extra rows. Return "
        "JSON only: {\"objects\":[{\"id\":1,\"label\":\"one_allowed_label\","
        "\"certainty\":\"high_or_medium_or_low\"}]}"
    )


def make_scene_root_batch_query_image(
    image: Image.Image,
    roots: Sequence[MaskCandidate],
    *,
    columns: int = 2,
    tile_width: int = 384,
    panel_height: int = 192,
    header_height: int = 28,
) -> Image.Image:
    if not roots:
        raise ValueError("scene-root batch requires at least one target")
    columns = max(1, min(columns, len(roots)))
    rows = (len(roots) + columns - 1) // columns
    tile_height = header_height + panel_height
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, root in enumerate(roots, start=1):
        panel = make_root_query_image(image, root_mask=root.mask)
        panel = panel.resize((tile_width, panel_height), Image.Resampling.LANCZOS)
        column = (index - 1) % columns
        row = (index - 1) // columns
        x0 = column * tile_width
        y0 = row * tile_height
        canvas.paste(panel, (x0, y0 + header_height))
        draw.rectangle((x0, y0, x0 + tile_width - 1, y0 + header_height - 1), fill="white")
        draw.text((x0 + 8, y0 + 7), f"TARGET {index}", fill="black")
        draw.rectangle(
            (x0, y0, x0 + tile_width - 1, y0 + tile_height - 1),
            outline=(80, 80, 80),
            width=2,
        )
    return canvas


def parse_scene_root_batch_audit_response(
    response: str,
    *,
    expected_count: int,
) -> tuple[dict[int, ParsedRootAudit], dict[str, object]]:
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
    parsed_by_id: dict[int, ParsedRootAudit] = {}
    duplicate_count = 0
    invalid_id_count = 0
    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            invalid_id_count += 1
            continue
        try:
            target_id = int(raw_row.get("id", 0))
        except (TypeError, ValueError):
            target_id = 0
        if not 1 <= target_id <= expected_count:
            invalid_id_count += 1
            continue
        parsed = parse_scene_root_audit_response(json.dumps(raw_row))
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
        "duplicate_count": duplicate_count,
        "invalid_id_count": invalid_id_count,
        "response_character_count": len(response),
        "ground_truth_used": False,
    }


def parse_scene_root_audit_response(response: str) -> ParsedRootAudit:
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
    if payload is None:
        return ParsedRootAudit(
            None,
            0.0,
            {
                "status": "invalid_json",
                "response_character_count": len(response),
                "ground_truth_used": False,
            },
        )
    label = re.sub(r"[^a-z0-9]+", "_", str(payload.get("label", "")).lower())
    label = label.strip("_")
    certainty = re.sub(
        r"[^a-z]+", "", str(payload.get("certainty", "")).lower()
    )
    certainty_confidence = {"high": 0.95, "medium": 0.75, "low": 0.25}.get(
        certainty
    )
    if certainty_confidence is not None:
        confidence = certainty_confidence
    else:
        try:
            confidence = float(payload.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
    confidence = float(max(0.0, min(1.0, confidence)))
    if label not in _ROOT_LABEL_TARGETS:
        return ParsedRootAudit(
            None,
            confidence,
            {
                "status": "unknown_label",
                "label": label or None,
                "certainty": certainty or None,
                "response_character_count": len(response),
                "ground_truth_used": False,
            },
        )
    return ParsedRootAudit(
        label,
        confidence,
        {
            "status": "accepted" if _ROOT_LABEL_TARGETS[label] else "abstained",
            "label": label,
            "certainty": certainty or None,
            "response_character_count": len(response),
            "ground_truth_used": False,
        },
    )


def _audit_priority(
    root: MaskCandidate,
    *,
    domain_counts: Counter[str],
    root_count: int,
    natural_scene: bool,
    config: SceneRootAuditConfig,
) -> tuple[float, tuple[str, ...]]:
    metadata = root.metadata
    has_ontology_evidence = bool(metadata.get("ontology_routing_algorithm"))
    current_domain = root.semantic_name
    global_domain = str(metadata.get("ontology_global_profile_domain") or "")
    visual_domain = str(metadata.get("ontology_visual_domain_winner") or "")
    global_agreement = bool(metadata.get("ontology_global_visual_agreement"))
    consensus = metadata.get("ontology_consensus_label")
    profile_decision = str(metadata.get("ontology_profile_decision") or "")
    priority = 0.0
    reasons: list[str] = []
    route_requires_audit = bool(metadata.get("asset_domain_audit_required"))
    if route_requires_audit:
        priority += 5.0
        reasons.append("asset_router_exact_label_uncertain")
    if has_ontology_evidence:
        if global_domain and global_domain != current_domain:
            priority += 3.0
            reasons.append("global_profile_domain_disagreement")
        if visual_domain and visual_domain != current_domain:
            priority += 2.0
            reasons.append("visual_domain_disagreement")
        if not global_agreement and consensus is None:
            priority += 2.0
            reasons.append("no_independent_confirmation")
        if profile_decision in {
            "ambiguous_profile",
            "conflicting_profile_rejected",
        }:
            priority += 2.0
            reasons.append("profile_conflict")
    else:
        priority += 1.5
        reasons.append("no_ontology_router_evidence")
        if metadata.get("profile_resolution_status") in {
            None,
            "unresolved",
            "ambiguous",
        }:
            priority += 1.0
            reasons.append("profile_resolution_uncertain")
    if natural_scene and current_domain in _MANMADE_SCENE_DOMAINS:
        priority += 4.0
        reasons.append("manmade_label_in_natural_scene")
    domain_ratio = domain_counts[current_domain] / max(1, root_count)
    if domain_ratio <= config.minority_domain_maximum_ratio:
        priority += 1.0
        reasons.append("minority_domain")
    if has_ontology_evidence and consensus is not None:
        priority -= 3.0
        reasons.append("scene_consensus_present")
    if (
        current_domain == "character"
        and not route_requires_audit
        and profile_decision not in {
        "ambiguous_profile",
        "conflicting_profile_rejected",
        }
    ):
        priority -= 5.0
        reasons.append("small_character_domain_guard")
    return priority, tuple(reasons)


def select_scene_root_audit_candidates(
    roots: Sequence[MaskCandidate],
    *,
    config: SceneRootAuditConfig | None = None,
) -> tuple[tuple[MaskCandidate, float, tuple[str, ...]], ...]:
    config = config or SceneRootAuditConfig()
    domain_counts = Counter(root.semantic_name for root in roots)
    root_count = len(roots)
    natural_ratio = (
        domain_counts["natural_object"] + domain_counts["terrain"]
    ) / max(1, root_count)
    natural_scene = (
        natural_ratio >= config.natural_scene_minimum_ratio
        or domain_counts["terrain"] > 0
        and natural_ratio >= 0.25
    )
    ranked = []
    for root in roots:
        priority, reasons = _audit_priority(
            root,
            domain_counts=domain_counts,
            root_count=root_count,
            natural_scene=natural_scene,
            config=config,
        )
        if priority >= config.minimum_priority:
            ranked.append((root, priority, reasons))
    ranked.sort(
        key=lambda row: (
            row[1],
            -domain_counts[row[0].semantic_name],
            row[0].score,
        ),
        reverse=True,
    )
    maximum_candidates = min(
        max(0, config.maximum_candidates),
        max(0, config.maximum_queries) * max(1, config.batch_size),
    )
    return tuple(ranked[:maximum_candidates])


def _profile_by_name(domain: DomainPrompt, name: str | None) -> PartProfile | None:
    if name is None:
        return None
    return next((profile for profile in domain.part_profiles if profile.name == name), None)


def _audited_root(
    root: MaskCandidate,
    *,
    label: str,
    confidence: float,
    domains: Mapping[str, DomainPrompt],
    planner_backend: str,
) -> MaskCandidate | None:
    target = _ROOT_LABEL_TARGETS[label]
    if target is None:
        return None
    domain_name, mapped_profile = target
    domain = domains.get(domain_name)
    if domain is None:
        return None
    profile_name = mapped_profile
    if profile_name is None and domain_name == root.semantic_name:
        value = root.metadata.get("selected_part_profile")
        profile_name = str(value) if value else None
    if profile_name is None:
        global_domain = str(root.metadata.get("ontology_global_profile_domain") or "")
        global_profile = root.metadata.get("ontology_global_profile_winner")
        if global_domain == domain_name and global_profile:
            profile_name = str(global_profile)
    profile = _profile_by_name(domain, profile_name)
    metadata = dict(root.metadata)
    for field in (
        "selected_part_profile",
        "resolved_object_label",
        "profile_evidence_root_key",
        "profile_evidence_score",
        "profile_evidence_margin",
    ):
        metadata.pop(field, None)
    if profile is not None:
        object_label = profile.root_hints[0]
        _, _, selection = domain.select_parts(
            object_label,
            profile_hint=profile.name,
            profile_hint_source="vlm_scene_root_audit",
        )
        metadata.update(
            {
                "selected_part_profile": profile.name,
                "resolved_object_label": object_label,
                "part_profile_specificity": 1.0,
                "part_profile_selection": selection,
                "profile_resolution_status": "accepted",
                "profile_hint_source": "vlm_scene_root_audit",
            }
        )
        prompt = object_label
    else:
        metadata.update(
            {
                "part_profile_specificity": 0.0,
                "profile_resolution_status": "unresolved",
                "profile_hint_source": "vlm_scene_root_audit",
            }
        )
        prompt = domain.name.replace("_", " ")
    metadata.update(
        {
            "vlm_root_audit_applied": True,
            "vlm_root_audit_original_domain": root.semantic_name,
            "vlm_root_audit_original_profile": root.metadata.get(
                "selected_part_profile"
            ),
            "vlm_root_audit_label": label,
            "vlm_root_audit_confidence": confidence,
            "vlm_root_audit_backend": planner_backend,
            "ground_truth_used": False,
        }
    )
    return MaskCandidate(
        domain_name,
        domain_name,
        root.mask,
        root.score,
        root.source,
        prompt=prompt,
        source_reliability=root.source_reliability,
        metadata=metadata,
    )


class SceneRootAuditor:
    def __init__(
        self,
        planner: VlmPlanner,
        *,
        config: SceneRootAuditConfig | None = None,
    ) -> None:
        self.planner = planner
        self.config = config or SceneRootAuditConfig()

    def audit(
        self,
        image: Image.Image,
        roots: Sequence[MaskCandidate],
        domains: Mapping[str, DomainPrompt],
    ) -> SceneRootAuditResult:
        selected = select_scene_root_audit_candidates(roots, config=self.config)
        updates: dict[str, MaskCandidate] = {}
        rows: list[dict[str, object]] = []
        query_count = 0
        batch_size = max(1, self.config.batch_size)
        for offset in range(0, len(selected), batch_size):
            batch = selected[offset : offset + batch_size]
            if query_count >= self.config.maximum_queries:
                break
            query_count += 1
            try:
                if len(batch) == 1:
                    response = self.planner.generate_response(
                        make_root_query_image(image, root_mask=batch[0][0].mask),
                        build_scene_root_audit_prompt(),
                    )
                    parsed_by_id = {1: parse_scene_root_audit_response(response)}
                    batch_parse = {"status": "single_target"}
                else:
                    response = self.planner.generate_response(
                        make_scene_root_batch_query_image(
                            image, [root for root, _, _ in batch]
                        ),
                        build_scene_root_batch_audit_prompt(len(batch)),
                    )
                    parsed_by_id, batch_parse = (
                        parse_scene_root_batch_audit_response(
                            response,
                            expected_count=len(batch),
                        )
                    )
            except Exception as error:  # noqa: BLE001  # model/runtime fallback
                for root, priority, reasons in batch:
                    rows.append(
                        {
                            "root_key": candidate_root_key(root),
                            "status": "planner_error",
                            "error": f"{type(error).__name__}: {error}",
                            "priority": priority,
                            "reasons": list(reasons),
                            "batch_query_index": query_count,
                            "ground_truth_used": False,
                        }
                    )
                continue
            for target_id, (root, priority, reasons) in enumerate(batch, start=1):
                root_key = candidate_root_key(root)
                if root_key is None:
                    continue
                parsed = parsed_by_id.get(target_id)
                if parsed is None:
                    rows.append(
                        {
                            "root_key": root_key,
                            "status": "missing_batch_prediction",
                            "original_domain": root.semantic_name,
                            "priority": priority,
                            "reasons": list(reasons),
                            "batch_query_index": query_count,
                            "batch_parse": batch_parse,
                            "ground_truth_used": False,
                        }
                    )
                    continue
                status = "abstained"
                updated = None
                if (
                    parsed.label is not None
                    and parsed.confidence >= self.config.minimum_confidence
                ):
                    updated = _audited_root(
                        root,
                        label=parsed.label,
                        confidence=parsed.confidence,
                        domains=domains,
                        planner_backend=self.planner.backend_id,
                    )
                    if updated is not None:
                        if (
                            updated.semantic_name == root.semantic_name
                            and updated.metadata.get("selected_part_profile")
                            == root.metadata.get("selected_part_profile")
                        ):
                            status = "confirmed"
                        else:
                            status = "corrected"
                            updates[root_key] = updated
                rows.append(
                    {
                        "root_key": root_key,
                        "status": status,
                        "original_domain": root.semantic_name,
                        "original_profile": root.metadata.get(
                            "selected_part_profile"
                        ),
                        "predicted_label": parsed.label,
                        "confidence": parsed.confidence,
                        "resolved_domain": (
                            updated.semantic_name
                            if updated is not None
                            else root.semantic_name
                        ),
                        "resolved_profile": (
                            updated.metadata.get("selected_part_profile")
                            if updated is not None
                            else root.metadata.get("selected_part_profile")
                        ),
                        "priority": priority,
                        "reasons": list(reasons),
                        "parse": parsed.diagnostics,
                        "batch_query_index": query_count,
                        "batch_parse": batch_parse,
                        "ground_truth_used": False,
                    }
                )
        output = tuple(
            updates.get(candidate_root_key(root) or "", root) for root in roots
        )
        return SceneRootAuditResult(
            output,
            {
                "algorithm": "hpid-vlm-scene-root-audit-v3",
                "planner_backend": self.planner.backend_id,
                "input_root_count": len(roots),
                "eligible_root_count": len(selected),
                "query_count": query_count,
                "audited_root_count": len(rows),
                "batch_size": batch_size,
                "correction_count": sum(row["status"] == "corrected" for row in rows),
                "confirmation_count": sum(row["status"] == "confirmed" for row in rows),
                "abstention_count": sum(row["status"] == "abstained" for row in rows),
                "rows": rows,
                "ground_truth_used": False,
            },
        )


def apply_scene_root_audit(
    candidates: Sequence[MaskCandidate],
    audited_roots: Sequence[MaskCandidate],
    domains: Mapping[str, DomainPrompt],
) -> CandidateAuditApplication:
    updates = {
        key: root
        for root in audited_roots
        if root.metadata.get("vlm_root_audit_applied")
        and (key := candidate_root_key(root)) is not None
    }
    if not updates:
        return CandidateAuditApplication(
            tuple(candidates),
            {
                "updated_root_count": 0,
                "dropped_incompatible_candidate_count": 0,
                "reset_visual_candidate_count": 0,
                "ground_truth_used": False,
            },
        )
    allowed_parts = {
        domain_name: {part.semantic_name for part in domain.parts}
        for domain_name, domain in domains.items()
    }
    output: list[MaskCandidate] = []
    visual_counters: Counter[str] = Counter()
    dropped = 0
    reset = 0
    replaced_roots: set[str] = set()
    for candidate in candidates:
        root_key = candidate_root_key(candidate)
        updated_root = updates.get(root_key or "")
        if updated_root is None:
            output.append(candidate)
            continue
        is_root = (
            candidate.metadata.get("parent_candidate_key") is None
            and candidate.semantic_name == candidate.semantic_parent
        )
        if is_root:
            if root_key not in replaced_roots:
                output.append(updated_root)
                replaced_roots.add(str(root_key))
            continue
        target_domain = updated_root.semantic_name
        if candidate.metadata.get("visual_region"):
            visual_counters[str(root_key)] += 1
            metadata = {
                key: value
                for key, value in candidate.metadata.items()
                if not key.startswith("semantic_rerank")
                and not key.startswith("prototype_")
            }
            metadata.update(
                {
                    "generic_visual_region": True,
                    "vlm_root_audit_reset": True,
                    "vlm_root_audit_target_domain": target_domain,
                    "ground_truth_used": False,
                }
            )
            kind = re.sub(
                r"[^a-z0-9]+",
                "_",
                str(metadata.get("visual_region_kind") or "region").lower(),
            ).strip("_")
            root_token = re.sub(
                r"[^a-z0-9]+",
                "_",
                str(metadata.get("scene_object_id") or metadata.get("root_index")),
            ).strip("_")
            semantic_name = (
                f"{target_domain}_{root_token}_audit_visual_{kind}_"
                f"{visual_counters[str(root_key)]:02d}"
            )
            output.append(
                MaskCandidate(
                    semantic_name,
                    target_domain,
                    candidate.mask,
                    candidate.score,
                    candidate.source,
                    prompt=candidate.prompt,
                    source_reliability=candidate.source_reliability,
                    metadata=metadata,
                )
            )
            reset += 1
            continue
        if candidate.semantic_name in allowed_parts.get(target_domain, set()):
            output.append(candidate)
            continue
        dropped += 1
    return CandidateAuditApplication(
        tuple(output),
        {
            "updated_root_count": len(replaced_roots),
            "dropped_incompatible_candidate_count": dropped,
            "reset_visual_candidate_count": reset,
            "ground_truth_used": False,
        },
    )
