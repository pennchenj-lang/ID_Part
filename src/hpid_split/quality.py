from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
from PIL import Image, ImageDraw

from .instances import PartInstance


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _rows(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _target_candidates(root_routing: dict[str, object]) -> list[dict[str, object]]:
    score_rows = {
        str(row.get("root_key")): row
        for row in _rows(root_routing.get("root_scores"))
    }
    selected_group = str(root_routing.get("selected_physical_group_id", ""))
    groups = sorted(
        _rows(root_routing.get("physical_groups")),
        key=lambda row: float(row.get("group_score", 0.0)),
        reverse=True,
    )
    output: list[dict[str, object]] = []
    for rank, group in enumerate(groups[:8], start=1):
        root_key = str(group.get("selected_root_key", ""))
        score_row = score_rows.get(root_key, {})
        box = score_row.get("bbox_xyxy")
        if not (
            isinstance(box, list)
            and len(box) == 4
            and all(isinstance(value, (int, float)) for value in box)
        ):
            continue
        group_id = str(group.get("physical_group_id", ""))
        output.append(
            {
                "rank": rank,
                "physical_group_id": group_id,
                "semantic_name": str(score_row.get("semantic_name", "asset")),
                "bbox_xyxy": [round(float(value)) for value in box],
                "group_score": float(group.get("group_score", 0.0)),
                "selected": group_id == selected_group,
            }
        )
    return output


def assess_product_quality(
    instance_map: np.ndarray,
    records: Sequence[PartInstance],
    diagnostics: dict[str, object] | None,
) -> dict[str, object]:
    """Create a ground-truth-free product review report for one exported run.

    The report is an evidence and ambiguity audit, not a calibrated estimate of
    segmentation accuracy. Its purpose is to prevent an uncertain automatic
    choice from being presented as an unquestionable result.
    """

    if instance_map.ndim != 2:
        raise ValueError("instance_map must be two-dimensional")
    diagnostics = diagnostics or {}
    root_routing = _mapping(diagnostics.get("root_routing"))
    initial_root_routing = _mapping(diagnostics.get("initial_root_routing"))
    final_groups = _rows(root_routing.get("physical_groups"))
    initial_groups = sorted(
        _rows(initial_root_routing.get("physical_groups")),
        key=lambda row: float(row.get("group_score", 0.0)),
        reverse=True,
    )
    initial_heterogeneous_choice = False
    final_point_requested = bool(
        _mapping(root_routing.get("target_point_routing")).get("requested", False)
    )
    if (
        not final_point_requested
        and len(final_groups) <= 1
        and len(initial_groups) > 1
    ):
        initial_top = float(initial_groups[0].get("group_score", 0.0))
        initial_runner = float(initial_groups[1].get("group_score", 0.0))
        initial_heterogeneous_choice = bool(
            str(initial_groups[0].get("selected_semantic", ""))
            != str(initial_groups[1].get("selected_semantic", ""))
            and initial_runner >= 0.55
            and initial_top - initial_runner <= 0.16
        )
    target_audit_routing = (
        initial_root_routing if initial_heterogeneous_choice else root_routing
    )
    target_routing = _mapping(target_audit_routing.get("target_point_routing"))
    fragment_consensus = _mapping(
        target_audit_routing.get("prompt_fragment_consensus")
    )
    groups = sorted(
        _rows(target_audit_routing.get("physical_groups")),
        key=lambda row: float(row.get("group_score", 0.0)),
        reverse=True,
    )
    top_score = float(groups[0].get("group_score", 0.0)) if groups else None
    runner_up_score = (
        float(groups[1].get("group_score", 0.0)) if len(groups) > 1 else None
    )
    score_margin = (
        top_score - runner_up_score
        if top_score is not None and runner_up_score is not None
        else None
    )
    point_requested = bool(target_routing.get("requested", False))
    consensus_selected = bool(
        fragment_consensus.get("selected_physical_group_id")
    )
    salience_ties = int(
        target_audit_routing.get("salience_tie_candidate_count", 1) or 1
    )
    prompt_groups = int(
        target_audit_routing.get("prompt_owned_group_count", 0) or 0
    )
    target_ambiguous = bool(
        target_audit_routing.get("mode") == "primary"
        and not point_requested
        and len(groups) > 1
        and not consensus_selected
        and (
            (salience_ties > 1 and score_margin is not None and score_margin <= 0.03)
            or (
                prompt_groups > 1
                and score_margin is not None
                and score_margin <= 0.04
            )
        )
    )
    close_physical_target_ambiguity = False
    heterogeneous_target_ambiguity = False
    if (
        target_audit_routing.get("mode") == "primary"
        and not point_requested
        and not consensus_selected
        and len(groups) > 1
        and top_score is not None
        and runner_up_score is not None
    ):
        top_semantic = str(groups[0].get("selected_semantic", ""))
        runner_up_semantic = str(groups[1].get("selected_semantic", ""))
        close_physical_target_ambiguity = bool(
            runner_up_score >= 0.55
            and score_margin is not None
            and score_margin <= 0.03
        )
        heterogeneous_target_ambiguity = bool(
            top_semantic
            and runner_up_semantic
            and top_semantic != runner_up_semantic
            and runner_up_score >= 0.55
            and score_margin is not None
            and score_margin <= 0.16
        )
    target_ambiguous = bool(
        target_ambiguous
        or close_physical_target_ambiguity
        or heterogeneous_target_ambiguity
    )

    image_area = max(1, int(instance_map.size))
    part_count = len(records)
    child_count = sum(
        record.semantic_name != record.semantic_parent for record in records
    )
    root_count = part_count - child_count
    generic_records = [
        record
        for record in records
        if "visual_" in record.semantic_name
        or record.semantic_name.endswith(("_region", "_detail"))
    ]
    generic_count = len(generic_records)
    tiny_threshold = max(20, round(image_area * 0.00012))
    tiny_count = sum(record.area_px <= tiny_threshold for record in records)
    generic_ratio = generic_count / max(1, part_count)
    tiny_ratio = tiny_count / max(1, part_count)
    exported_foreground_area = max(1, int(np.count_nonzero(instance_map)))
    generic_area_px = sum(record.area_px for record in generic_records)
    generic_area_ratio = generic_area_px / exported_foreground_area
    root_residual_area_px = sum(
        record.area_px
        for record in records
        if record.semantic_name == record.semantic_parent
    )
    root_residual_ratio = root_residual_area_px / exported_foreground_area
    semantic_areas: dict[tuple[str, str], list[int]] = {}
    for record in records:
        semantic_areas.setdefault(
            (record.asset_id, record.semantic_name), []
        ).append(record.area_px)
    repeated_part_imbalances = []
    for (asset_id, semantic_name), areas in sorted(semantic_areas.items()):
        if len(areas) < 2:
            continue
        smallest = min(areas)
        largest = max(areas)
        ratio = smallest / max(1, largest)
        if (
            smallest <= tiny_threshold
            and largest >= max(4 * tiny_threshold, 100)
            and ratio < 0.08
        ):
            repeated_part_imbalances.append(
                {
                    "asset_id": asset_id,
                    "semantic_name": semantic_name,
                    "instance_count": len(areas),
                    "smallest_area_px": smallest,
                    "largest_area_px": largest,
                    "smallest_to_largest_ratio": ratio,
                }
            )
    asset_counts = Counter(record.asset_id for record in records)
    record_by_part_id = {record.part_id: record for record in records}
    missing_parent_count = 0
    cross_asset_parent_count = 0
    for record in records:
        if record.assembly_parent_id is None:
            continue
        parent = record_by_part_id.get(record.assembly_parent_id)
        if parent is None:
            missing_parent_count += 1
        elif parent.asset_id != record.asset_id:
            cross_asset_parent_count += 1
    singleton_asset_count = sum(count == 1 for count in asset_counts.values())
    singleton_asset_ratio = singleton_asset_count / max(1, len(asset_counts))
    dominant_asset_ratio = max(asset_counts.values(), default=0) / max(1, part_count)

    completion_statuses = Counter()
    completion = _mapping(diagnostics.get("completion"))
    for row in _rows(completion.get("records")):
        completion_statuses[str(row.get("status", "unknown"))] += 1

    review_reasons: list[str] = []
    recommended_actions: list[str] = []
    asset_domain_routing = _mapping(diagnostics.get("asset_domain_routing"))
    vlm_root_audit = _mapping(diagnostics.get("vlm_root_audit"))
    audited_root_keys = {
        str(row.get("root_key", ""))
        for row in _rows(vlm_root_audit.get("rows"))
        if str(row.get("status", "")) in {"confirmed", "corrected"}
    }
    unresolved_domain_disagreement_count = 0
    routing_rows = _rows(asset_domain_routing.get("rows")) or _rows(
        asset_domain_routing.get("routes")
    )
    global_asset_route = _mapping(
        _mapping(diagnostics.get("global_asset_proposal")).get("route")
    )
    selected_root_scores = [
        row
        for row in _rows(root_routing.get("root_scores"))
        if bool(row.get("selected", False))
    ]
    weak_cross_view_consensus_count = 0
    for row in routing_rows:
        consensus = _mapping(row.get("cross_view_consensus"))
        crop_route = _mapping(row.get("root_crop_asset_route"))
        if (
            bool(consensus.get("accepted", False))
            and not bool(global_asset_route.get("accepted", False))
            and not bool(crop_route.get("accepted", False))
        ):
            weak_cross_view_consensus_count += 1

    weak_selected_root_evidence = False
    coherent_wrong_target_risk = False
    if (
        root_routing.get("mode") == "primary"
        and not point_requested
        and len(selected_root_scores) == 1
    ):
        selected_root = selected_root_scores[0]
        semantic_probability = selected_root.get("semantic_mask_probability")
        semantic_probability = (
            float(semantic_probability)
            if isinstance(semantic_probability, (int, float))
            else None
        )
        frame_extent = float(selected_root.get("frame_extent", 0.0) or 0.0)
        touched_sides = int(selected_root.get("touched_sides", 0) or 0)
        weak_selected_root_evidence = bool(
            semantic_probability is not None
            and semantic_probability < 0.25
            and frame_extent >= 0.65
            and weak_cross_view_consensus_count
        )
        coherent_wrong_target_risk = bool(
            weak_selected_root_evidence
            and semantic_probability is not None
            and semantic_probability < 0.20
            and touched_sides >= 4
            and child_count == 0
            and part_count == 1
        )
    profile_resolution = _mapping(diagnostics.get("profile_root_resolution"))
    profile_consensus = _mapping(profile_resolution.get("profile_consensus"))
    accepted_profile_rows = [
        row
        for row in _rows(profile_consensus.get("roots"))
        if str(row.get("status", "")) == "accepted"
        and bool(row.get("selected_profile"))
    ]
    accepted_profile = (
        str(accepted_profile_rows[0]["selected_profile"])
        if len(routing_rows) == 1 and len(accepted_profile_rows) == 1
        else None
    )
    exact_routed_profiles = {
        str(route.get("asset_profile"))
        for route in [global_asset_route]
        + [_mapping(row.get("asset_route")) for row in routing_rows]
        if bool(route.get("accepted", False))
        and str(route.get("reason", "")) == "accepted_exact_label"
        and route.get("asset_profile")
    }
    selected_profiles = {
        str(value)
        for value in profile_resolution.get("selected_profiles", [])
        if str(value).strip()
    }
    confirmed_profiles = exact_routed_profiles | selected_profiles
    if accepted_profile:
        confirmed_profiles.add(accepted_profile)
    decomposition_missing = bool(
        part_count == 1
        and child_count == 0
        and confirmed_profiles
        and root_routing.get("mode") == "primary"
    )
    dominant_root_residual = bool(
        root_routing.get("mode") == "primary"
        and confirmed_profiles
        and root_count == 1
        and child_count >= 3
        and root_residual_ratio > 0.55
    )
    root_domain_evidence = {
        str(row.get("root_key", "")): (
            float(row["domain_evidence_score"])
            if isinstance(row.get("domain_evidence_score"), (int, float))
            else None
        )
        for row in _rows(root_routing.get("root_scores"))
    }
    for row in routing_rows:
        root_key = str(row.get("root_key", ""))
        asset_route = _mapping(row.get("asset_route"))
        if (
            root_key in audited_root_keys
            or bool(asset_route.get("accepted", False))
        ):
            continue
        original_domain = str(row.get("original_domain", ""))
        candidate_domains = {
            str(value) for value in asset_route.get("candidate_domains", [])
        }
        if row.get("routing_applicable") is False:
            evidence = root_domain_evidence.get(root_key)
            if evidence is not None and evidence >= 0.50:
                continue
            if candidate_domains and original_domain not in candidate_domains:
                unresolved_domain_disagreement_count += 1
            continue
        alternatives = _rows(asset_route.get("alternatives"))
        top_alternative = alternatives[0] if alternatives else {}
        profile_confirms_top_route = bool(
            accepted_profile
            and str(top_alternative.get("asset_profile", "")) == accepted_profile
            and str(top_alternative.get("asset_domain", "")) == original_domain
            and bool(row.get("domain_accepted", False))
        )
        if profile_confirms_top_route:
            continue
        if (
            not bool(row.get("domain_accepted", False))
            or not candidate_domains
            or original_domain not in candidate_domains
            or str(asset_route.get("reason", "")) == "ambiguous_candidate_set"
        ):
            unresolved_domain_disagreement_count += 1
    if unresolved_domain_disagreement_count:
        review_reasons.append("asset_domain_uncertain")
        recommended_actions.append("review_asset_domain_or_add_asset_prompt")
    if weak_cross_view_consensus_count:
        review_reasons.append("weak_cross_view_asset_consensus")
        recommended_actions.append("review_asset_label_or_add_asset_prompt")
    if weak_selected_root_evidence:
        review_reasons.append("weak_root_mask_evidence")
        recommended_actions.append("inspect_root_mask_or_select_target_point")
    if coherent_wrong_target_risk:
        review_reasons.append("coherent_wrong_target_risk")
        recommended_actions.append("select_target_point")
    if missing_parent_count:
        review_reasons.append("missing_assembly_parent")
        recommended_actions.append("repair_asset_hierarchy")
    if cross_asset_parent_count:
        review_reasons.append("cross_asset_assembly_parent")
        recommended_actions.append("repair_asset_hierarchy")
    if target_ambiguous:
        review_reasons.append("ambiguous_primary_asset")
        recommended_actions.append("select_target_point")
    if decomposition_missing:
        review_reasons.append("no_part_decomposition")
        recommended_actions.append("add_part_prompts_or_use_ensemble")
    if dominant_root_residual:
        review_reasons.append("dominant_unassigned_root_residual")
        recommended_actions.append("review_root_mask_or_use_part_prompts")
    if part_count == 0:
        review_reasons.append("no_exported_parts")
        recommended_actions.append("change_domain_or_add_asset_prompt")
    if part_count >= 6 and generic_count >= 3 and generic_ratio > 0.15:
        review_reasons.append("many_generic_part_names")
        recommended_actions.append("add_part_prompts_for_semantic_ids")
    if generic_count and generic_area_ratio > 0.20:
        review_reasons.append("large_unresolved_generic_area")
        recommended_actions.append("review_or_relabel_generic_regions")
    if part_count >= 12 and tiny_ratio > 0.35:
        review_reasons.append("many_tiny_regions")
        recommended_actions.append("inspect_part_boundaries")
    elif part_count >= 12 and tiny_count >= 3 and tiny_ratio >= 0.05:
        review_reasons.append("several_tiny_regions")
        recommended_actions.append("inspect_part_boundaries")
    if repeated_part_imbalances:
        review_reasons.append("severe_repeated_part_area_imbalance")
        recommended_actions.append("inspect_repeated_part_identity")
    if part_count > 96 and root_routing.get("mode") != "scene":
        review_reasons.append("unusually_many_parts_for_primary_asset")
        recommended_actions.append("inspect_part_boundaries")
    if root_routing.get("mode") == "scene" and (
        part_count > 128
        or part_count > max(96, 8 * max(1, len(asset_counts)))
    ):
        review_reasons.append("unusually_many_scene_parts")
        recommended_actions.append("inspect_scene_object_ownership")
    if len(asset_counts) >= 4 and singleton_asset_ratio > 0.6:
        review_reasons.append("many_singleton_scene_assets")
        recommended_actions.append("inspect_scene_object_ownership")
    if len(asset_counts) >= 4 and dominant_asset_ratio > 0.7:
        review_reasons.append("scene_asset_ownership_imbalance")
        recommended_actions.append("inspect_scene_object_ownership")

    hierarchy_invalid = bool(missing_parent_count or cross_asset_parent_count)
    if hierarchy_invalid:
        status = "invalid_hierarchy"
        evidence_grade = "C"
    elif target_ambiguous or coherent_wrong_target_risk:
        status = "target_selection_required"
        evidence_grade = "C"
    elif part_count == 0:
        status = "no_parts_found"
        evidence_grade = "C"
    elif review_reasons:
        status = "review_recommended"
        evidence_grade = "B"
    else:
        status = "ready"
        evidence_grade = "A"

    return {
        "format": "HPID-Split product quality report",
        "format_version": "0.1.0",
        "status": status,
        "evidence_grade": evidence_grade,
        "review_reasons": review_reasons,
        "recommended_actions": list(dict.fromkeys(recommended_actions)),
        "target_selection": {
            "ambiguous": target_ambiguous or coherent_wrong_target_risk,
            "close_physical_target_ambiguity": close_physical_target_ambiguity,
            "heterogeneous_target_ambiguity": heterogeneous_target_ambiguity,
            "coherent_wrong_target_risk": coherent_wrong_target_risk,
            "target_point_requested": point_requested,
            "physical_group_count": len(groups),
            "salience_tie_candidate_count": salience_ties,
            "top_group_score": top_score,
            "runner_up_group_score": runner_up_score,
            "score_margin": score_margin,
            "prompt_fragment_consensus_selected": consensus_selected,
            "audit_stage": (
                "initial_root_routing"
                if initial_heterogeneous_choice
                else "final_root_routing"
            ),
            "candidates": _target_candidates(target_audit_routing),
        },
        "part_structure": {
            "part_count": part_count,
            "root_count": root_count,
            "child_count": child_count,
            "generic_name_count": generic_count,
            "generic_name_ratio": generic_ratio,
            "generic_area_px": generic_area_px,
            "generic_area_ratio": generic_area_ratio,
            "root_residual_area_px": root_residual_area_px,
            "root_residual_ratio": root_residual_ratio,
            "dominant_root_residual": dominant_root_residual,
            "tiny_region_count": tiny_count,
            "tiny_region_ratio": tiny_ratio,
            "tiny_region_threshold_px": tiny_threshold,
            "repeated_part_area_imbalances": repeated_part_imbalances,
            "asset_count": len(asset_counts),
            "parts_per_asset": dict(sorted(asset_counts.items())),
            "singleton_asset_count": singleton_asset_count,
            "singleton_asset_ratio": singleton_asset_ratio,
            "dominant_asset_ratio": dominant_asset_ratio,
            "missing_assembly_parent_count": missing_parent_count,
            "cross_asset_assembly_parent_count": cross_asset_parent_count,
            "confirmed_profiles": sorted(confirmed_profiles),
            "decomposition_missing": decomposition_missing,
        },
        "hidden_completion_status_counts": dict(sorted(completion_statuses.items())),
        "asset_domain_uncertainty": {
            "unresolved_disagreement_count": unresolved_domain_disagreement_count,
            "weak_cross_view_consensus_count": (
                weak_cross_view_consensus_count
            ),
            "weak_selected_root_evidence": weak_selected_root_evidence,
        },
        "ground_truth_used": False,
        "warning": (
            "This is a ground-truth-free evidence audit, not a calibrated "
            "segmentation accuracy probability."
        ),
    }


def render_target_candidates(
    image: Image.Image,
    report: dict[str, object],
) -> Image.Image | None:
    """Render auditable primary-asset alternatives for interactive correction."""

    target = _mapping(report.get("target_selection"))
    candidates = _rows(target.get("candidates"))
    if len(candidates) <= 1:
        return None
    rendered = image.convert("RGB").copy()
    draw = ImageDraw.Draw(rendered)
    palette = (
        (0, 196, 118),
        (245, 91, 67),
        (46, 134, 222),
        (229, 177, 0),
        (164, 94, 196),
        (0, 166, 166),
        (212, 87, 153),
        (92, 118, 138),
    )
    line_width = max(2, round(min(image.size) * 0.006))
    for row in candidates:
        rank = int(row.get("rank", 0))
        box = row.get("bbox_xyxy")
        if not isinstance(box, list) or len(box) != 4:
            continue
        color = palette[(rank - 1) % len(palette)]
        width = line_width * (2 if row.get("selected") else 1)
        draw.rectangle(tuple(int(value) for value in box), outline=color, width=width)
        x0, y0, _, _ = (int(value) for value in box)
        label = f"#{rank}"
        label_box = draw.textbbox((x0, y0), label)
        label_width = label_box[2] - label_box[0] + 8
        label_height = label_box[3] - label_box[1] + 6
        draw.rectangle(
            (x0, y0, x0 + label_width, y0 + label_height),
            fill=color,
        )
        draw.text((x0 + 4, y0 + 2), label, fill=(255, 255, 255))
    return rendered
