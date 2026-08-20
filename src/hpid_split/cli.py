from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from PIL import Image

from .adapters import semantic_prediction_candidates
from .appearance_graph import optimize_appearance_graph
from .appearance_proposals import (
    AppearanceProposalConfig,
    propose_appearance_regions,
)
from .asset_routing import (
    AssetRoute,
    AssetRouter,
    AssetRouterConfig,
    AssetRoutingIndex,
    ProfileTextRoute,
    Siglip2AssetEncoder,
    profile_text_route_to_dict,
    reconcile_asset_routes,
    resolve_asset_domain,
    route_profile_text_inventories,
    route_to_dict,
)
from .candidate_backends import Sam2AutomaticMaskBackend
from .ensemble_gate import filter_unresolved_ensemble_regions
from .export import export_prediction, load_previous_package
from .florence_parts import FlorencePartConfig, FlorencePartGenerator
from .foundation import (
    AutomaticAssetQuery,
    CandidateGeneration,
    FoundationCandidateGenerator,
    FoundationConfig,
)
from .fusion import FusionConfig, MaskCandidate, fuse_candidates
from .guided_prompts import parse_guided_prompts
from .inference import predict
from .mask_refinement import MaskRefinementConfig, refine_candidate_masks
from .ontology_routing import route_scene_ontology
from .physical_groups import build_physical_groups
from .physical_region_audit import (
    PhysicalRegionAuditConfig,
    PhysicalRegionAuditor,
)
from .profile_resolution import resolve_profile_roots
from .prompt_bank import DomainPrompt, PromptBank
from .proposal_first import ProposalFirstConfig, generate_proposal_first_roots
from .registry import preserve_part_ids
from .relational import propose_relational_candidates
from .resource_paths import DEFAULT_PROMPT_BANK
from .restoration import (
    GeometricFallbackBackend,
    complete_and_export_parts,
    load_completion_backend,
)
from .retrieval import (
    CLIPSegEmbeddingEncoder,
    PrototypeIndex,
    PrototypeRetriever,
    RetrievalConfig,
    apply_retrieval_domain_corrections,
    build_retrieval_index,
)
from .root_cleanup import clean_primary_roots
from .root_geometry import refine_root_geometry_from_parts
from .root_routing import (
    RootRoutingConfig,
    candidate_root_key,
    propagate_scene_object_identity,
    route_asset_roots,
)
from .scene_root_audit import (
    SceneRootAuditConfig,
    SceneRootAuditor,
    apply_scene_root_audit,
)
from .semantic_candidate_audit import (
    SemanticCandidateAuditConfig,
    SemanticCandidateAuditor,
)
from .shape_proposals import ShapeProposalConfig, propose_shape_regions
from .structural_fusion import refine_profile_structure
from .taxonomy import Taxonomy
from .text_segmentation import Sam3TextConfig, Sam3TextPartGenerator
from .training import load_checkpoint, train
from .validation import validate_package
from .visual_regions import (
    Sam2VisualRegionProposer,
    VisualMaskProposal,
    VisualRegionConfig,
    visual_region_candidates_from_masks,
)
from .visual_semantics import (
    VisualSemanticConfig,
    enforce_axis_consistency,
    filter_unresolved_visual_regions,
    rerank_visual_candidates,
)
from .vlm_parts import (
    Qwen3VlPartPlanner,
    Qwen3VlPlannerConfig,
    VlmPartConfig,
    VlmPartGenerator,
    apply_dynamic_object_profile_corrections,
)


def _device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return value


def _role_ids(image_dir: Path) -> list[str]:
    return sorted(path.stem for path in image_dir.glob("*.png"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _combine_profile_refinements(
    runs: list[CandidateGeneration],
) -> CandidateGeneration:
    """Combine sequential detector runs without hiding per-model provenance."""

    candidates = tuple(candidate for run in runs for candidate in run.candidates)
    root_rows = [
        {
            **row,
            "grounding_model": run.diagnostics.get("grounding_model"),
        }
        for run in runs
        for row in run.diagnostics.get("roots", [])
        if isinstance(row, dict)
    ]
    return CandidateGeneration(
        candidates,
        {
            "algorithm": "hpid-routed-profile-refinement-ensemble-v1",
            "model_run_count": len(runs),
            "grounding_models": [
                str(run.diagnostics.get("grounding_model", "unknown")) for run in runs
            ],
            "root_count": len(root_rows),
            "candidate_count": len(candidates),
            "roots": root_rows,
            "model_runs": [run.diagnostics for run in runs],
            "ground_truth_used": False,
        },
    )


def _ontology_router_model(args: argparse.Namespace) -> str:
    explicit = args.ontology_router_model.strip() or args.asset_router_model.strip()
    if explicit:
        return explicit
    if args.asset_router_index is None:
        return ""
    manifest_path = args.asset_router_index / "index.json"
    if not manifest_path.is_file():
        return ""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return str(manifest.get("encoder_model_name", "")).strip()


def _apply_asset_domain_routes(
    candidates: list[MaskCandidate],
    roots: list[MaskCandidate],
    routes_by_root: dict[str, AssetRoute],
    prompt_bank: PromptBank,
    *,
    config: AssetRouterConfig,
    supported_domains: set[str] | None = None,
    full_image_route: AssetRoute | None = None,
    profile_text_routes_by_root: dict[
        str, ProfileTextRoute | dict[str, ProfileTextRoute]
    ]
    | None = None,
) -> tuple[list[MaskCandidate], dict[str, object]]:
    """Apply label-safe asset-domain evidence before category-part queries."""

    domains = {domain.name: domain for domain in prompt_bank.domains}
    replacement_by_key: dict[str, MaskCandidate] = {}
    corrected_root_keys: set[str] = set()
    rows: list[dict[str, object]] = []
    for root in roots:
        root_key = candidate_root_key(root)
        if root_key is None or root_key not in routes_by_root:
            continue
        local_route = routes_by_root[root_key]
        proposal_rank_value = root.metadata.get("global_asset_proposal_rank")
        proposal_rank = (
            int(proposal_rank_value)
            if isinstance(proposal_rank_value, int)
            else None
        )
        route, cross_view_consensus = reconcile_asset_routes(
            full_image_route,
            local_route,
            config=config,
            root_global_proposal_rank=proposal_rank,
        )
        independently_accepted_exact_route = bool(
            route.accepted and route.reason == "accepted_exact_label"
        )
        routing_applicable = bool(
            supported_domains is None
            or root.semantic_name in supported_domains
            or independently_accepted_exact_route
        )
        resolution = resolve_asset_domain(
            route,
            root.semantic_name,
            config=config,
        )
        resolved_domain = resolution.resolved_domain
        existing_profile_value = root.metadata.get("selected_part_profile")
        existing_profile = (
            str(existing_profile_value)
            if existing_profile_value is not None
            and str(existing_profile_value).strip()
            else None
        )
        preserve_specific_root_domain = bool(
            resolution.accepted
            and resolved_domain is not None
            and resolved_domain != root.semantic_name
            and not independently_accepted_exact_route
            and existing_profile is not None
            and float(root.metadata.get("part_profile_specificity", 0.0)) >= 0.80
            and float(np.clip(root.score / 0.65, 0.0, 1.0)) >= 0.58
        )
        if preserve_specific_root_domain:
            resolved_domain = root.semantic_name
        accepted = bool(
            routing_applicable
            and
            resolution.accepted
            and resolved_domain is not None
            and resolved_domain in domains
        )
        corrected = bool(accepted and resolved_domain != root.semantic_name)
        selected_profile = None
        profile_hint_source = None
        proposed_profile = resolution.resolved_profile
        cross_view_exact_profile = bool(
            route.accepted
            and proposed_profile is not None
            and cross_view_consensus.get("status")
            == "accepted_top_label_agreement"
            and cross_view_consensus.get("exact_top_agreement") is True
        )
        profile_text_route_entry = (
            profile_text_routes_by_root.get(root_key)
            if profile_text_routes_by_root is not None
            else None
        )
        profile_text_route = (
            profile_text_route_entry.get(str(resolved_domain))
            if isinstance(profile_text_route_entry, dict)
            and resolved_domain is not None
            else profile_text_route_entry
            if isinstance(profile_text_route_entry, ProfileTextRoute)
            else None
        )
        # Cross-view agreement is useful for recovering a broad asset domain, but
        # it is not independent evidence for an exact part template.  Keeping the
        # profile unresolved here prevents two weak views from locking a rifle to
        # a visually similar screwdriver (or analogous same-domain mistakes).
        if preserve_specific_root_domain and existing_profile is not None:
            available_profiles = {
                profile.name for profile in domains[resolved_domain].part_profiles
            }
            if existing_profile in available_profiles:
                selected_profile = existing_profile
                profile_hint_source = str(
                    root.metadata.get("profile_hint_source")
                    or "specific_root_label"
                )
        elif (
            accepted
            and independently_accepted_exact_route
            and proposed_profile is not None
        ):
            available_profiles = {
                profile.name for profile in domains[resolved_domain].part_profiles
            }
            if proposed_profile in available_profiles:
                selected_profile = proposed_profile
                profile_hint_source = (
                    "exact_asset_router"
                    if route.accepted
                    else "domain_conditioned_asset_router"
                )
        elif (
            accepted
            and cross_view_exact_profile
            and proposed_profile is not None
            and (
                profile_text_route is None
                or not profile_text_route.accepted
                or profile_text_route.profile == proposed_profile
            )
        ):
            available_profiles = {
                profile.name for profile in domains[resolved_domain].part_profiles
            }
            if proposed_profile in available_profiles:
                selected_profile = proposed_profile
                profile_hint_source = "cross_view_exact_asset_router"
        elif (
            accepted
            and profile_text_route is not None
            and profile_text_route.accepted
            and profile_text_route.profile is not None
        ):
            available_profiles = {
                profile.name for profile in domains[resolved_domain].part_profiles
            }
            if profile_text_route.profile in available_profiles:
                selected_profile = profile_text_route.profile
                profile_hint_source = "lightweight_profile_text_router"
        row = {
            "root_key": root_key,
            "original_domain": root.semantic_name,
            "resolved_domain": resolved_domain if accepted else None,
            "selected_profile": selected_profile,
            "domain_accepted": accepted,
            "domain_corrected": corrected,
            "routing_applicable": routing_applicable,
            "independently_accepted_exact_route": (
                independently_accepted_exact_route
            ),
            "cross_view_exact_profile": cross_view_exact_profile,
            "specific_root_domain_preserved": preserve_specific_root_domain,
            "router_supported_domains": (
                sorted(supported_domains)
                if supported_domains is not None
                else None
            ),
            "domain_resolution": {
                "reason": (
                    "preserved_specific_root_domain_over_ambiguous_router"
                    if preserve_specific_root_domain
                    else resolution.reason
                    if routing_applicable
                    else "current_domain_outside_router_inventory"
                ),
                "support_count": resolution.support_count,
                "candidate_count": resolution.candidate_count,
                "support_ratio": resolution.support_ratio,
                "vote_margin": resolution.vote_margin,
                "domain_votes": list(resolution.domain_votes),
                "resolved_asset_label": resolution.resolved_asset_label,
                "asset_label_reason": resolution.asset_label_reason,
            },
            "asset_route": route_to_dict(route),
            "profile_text_route": (
                profile_text_route_to_dict(profile_text_route)
                if profile_text_route is not None
                else None
            ),
            "root_crop_asset_route": route_to_dict(local_route),
            "cross_view_consensus": cross_view_consensus,
        }
        rows.append(row)
        route_requires_audit = bool(routing_applicable and not route.accepted)
        if not accepted:
            metadata = dict(root.metadata)
            metadata.update(
                {
                    "asset_domain_routing_algorithm": (
                        "hpid-siglip2-preprofile-domain-consensus-v1"
                    ),
                    "asset_domain_routing_reason": resolution.reason,
                    "asset_domain_routing_original_domain": root.semantic_name,
                    "asset_domain_routing_resolved_domain": None,
                    "asset_domain_routing_accepted": False,
                    "asset_router_exact_label_accepted": (
                        independently_accepted_exact_route
                    ),
                    "asset_router_candidate_labels": list(route.candidate_labels),
                    "asset_router_candidate_domains": list(route.candidate_domains),
                    "asset_domain_audit_required": route_requires_audit,
                    "ground_truth_used": False,
                }
            )
            replacement_by_key[root_key] = MaskCandidate(
                semantic_name=root.semantic_name,
                semantic_parent=root.semantic_parent,
                mask=root.mask,
                score=root.score,
                source=root.source,
                prompt=root.prompt,
                source_reliability=root.source_reliability,
                metadata=metadata,
            )
            continue
        metadata = dict(root.metadata)
        for key in (
            "domain_evidence_score",
            "domain_evidence_contrast",
            "domain_evidence_prompt",
            "domain_evidence_similarity",
            "domain_evidence_rank",
            "part_profile_selection",
        ):
            metadata.pop(key, None)
        object_label = (
            str(selected_profile).replace("_", " ")
            if selected_profile is not None
            else str(
                root.metadata.get("root_model_label")
                or root.prompt
                or root.semantic_name.replace("_", " ")
            )
            if preserve_specific_root_domain
            else str(resolution.resolved_asset_label)
            if (independently_accepted_exact_route or cross_view_exact_profile)
            and resolution.resolved_asset_label is not None
            else str(resolved_domain).replace("_", " ")
        )
        metadata.update(
            {
                "asset_domain_routing_algorithm": (
                    "hpid-siglip2-preprofile-domain-consensus-v1"
                ),
                "asset_domain_routing_reason": (
                    "preserved_specific_root_domain_over_ambiguous_router"
                    if preserve_specific_root_domain
                    else resolution.reason
                ),
                "asset_domain_routing_original_domain": root.semantic_name,
                "asset_domain_routing_resolved_domain": resolved_domain,
                "asset_domain_routing_accepted": True,
                "asset_domain_routing_support_ratio": resolution.support_ratio,
                "asset_domain_routing_vote_margin": resolution.vote_margin,
                "asset_router_exact_label_accepted": (
                    independently_accepted_exact_route
                ),
                "asset_router_cross_view_exact_profile": (
                    cross_view_exact_profile
                ),
                "asset_router_candidate_labels": list(route.candidate_labels),
                "asset_router_candidate_domains": list(route.candidate_domains),
                "asset_domain_audit_required": route_requires_audit,
                "resolved_object_label": object_label,
                "root_model_label": object_label,
                "selected_part_profile": selected_profile,
                "profile_hint_source": (
                    profile_hint_source
                    if selected_profile is not None
                    else "asset_domain_consensus"
                ),
                "profile_resolution_status": (
                    "accepted" if selected_profile is not None else None
                ),
                "root_label_specificity": (
                    1.0
                    if (
                        independently_accepted_exact_route
                        or cross_view_exact_profile
                    )
                    and resolution.resolved_asset_label is not None
                    else 0.0
                ),
                "part_profile_specificity": (
                    1.0 if selected_profile is not None else 0.0
                ),
                "ground_truth_used": False,
            }
        )
        replacement_by_key[root_key] = MaskCandidate(
            semantic_name=str(resolved_domain),
            semantic_parent=str(resolved_domain),
            mask=root.mask,
            score=root.score,
            source=root.source,
            prompt=object_label,
            source_reliability=root.source_reliability,
            metadata=metadata,
        )
        if corrected:
            corrected_root_keys.add(root_key)

    if replacement_by_key:
        replaced: list[MaskCandidate] = []
        emitted_roots: set[str] = set()
        for candidate in candidates:
            root_key = candidate_root_key(candidate)
            if root_key is None or root_key not in replacement_by_key:
                replaced.append(candidate)
                continue
            is_root = (
                candidate.semantic_name == candidate.semantic_parent
                and candidate.metadata.get("parent_candidate_key") is None
            )
            if is_root and root_key not in emitted_roots:
                replaced.append(replacement_by_key[root_key])
                emitted_roots.add(root_key)
            elif root_key not in corrected_root_keys:
                replaced.append(candidate)
        candidates = replaced
    return candidates, {
        "algorithm": "hpid-siglip2-preprofile-domain-consensus-v1",
        "root_count": len(roots),
        "route_count": len(rows),
        "accepted_domain_count": sum(bool(row["domain_accepted"]) for row in rows),
        "corrected_domain_count": sum(bool(row["domain_corrected"]) for row in rows),
        "rows": rows,
        "ground_truth_used": False,
    }


def command_train(args: argparse.Namespace) -> int:
    taxonomy = Taxonomy.from_json(args.taxonomy)
    roles = (
        [value.strip() for value in args.roles.split(",") if value.strip()]
        if args.roles
        else _role_ids(args.image_dir)
    )
    train(
        args.image_dir,
        args.label_dir,
        roles,
        taxonomy,
        args.checkpoint,
        epochs=args.epochs,
        repeats=args.repeats,
        batch_size=args.batch_size,
        device=_device(args.device),
        seed=args.seed,
    )
    return 0


def command_predict(args: argparse.Namespace) -> int:
    device = _device(args.device)
    model, taxonomy, _ = load_checkpoint(args.checkpoint, device)
    image = Image.open(args.image).convert("RGB")
    result = predict(
        model,
        image,
        taxonomy,
        device=device,
        evaluation_height=args.evaluation_height,
        recursive=not args.no_recursive,
    )
    records = list(result.instances)
    if args.previous is not None:
        previous_map, previous_records = load_previous_package(args.previous)
        records = preserve_part_ids(
            result.instance_map,
            records,
            previous_map,
            previous_records,
        )
    manifest = export_prediction(
        image,
        result,
        taxonomy,
        args.output,
        records=records,
        checkpoint=args.checkpoint,
    )
    print(f"parts={manifest['part_count']} output={args.output}")
    return 0


def command_validate_taxonomy(args: argparse.Namespace) -> int:
    taxonomy = Taxonomy.from_json(args.taxonomy)
    print(
        f"valid fine_classes={taxonomy.num_fine_classes} "
        f"parent_classes={taxonomy.num_parent_classes} details={len(taxonomy.detail_ids)}"
    )
    return 0


def command_build_retrieval_index(args: argparse.Namespace) -> int:
    device = _device(args.device)
    encoder = CLIPSegEmbeddingEncoder(
        model_name=args.encoder_model,
        device=device,
        local_files_only=args.local_files_only,
        batch_size=args.embedding_batch_size,
    )
    manifest = build_retrieval_index(
        args.manifest,
        args.output,
        encoder,
        metric_epochs=args.metric_epochs,
        metric_device=device,
        seed=args.seed,
    )
    print(
        f"assets={manifest['asset_count']} "
        f"part_prototypes={manifest['part_prototype_count']} "
        f"output={args.output}"
    )
    return 0


def command_inspect_retrieval_index(args: argparse.Namespace) -> int:
    index = PrototypeIndex.load(args.index)
    labels = sorted({str(item["asset_label"]) for item in index.assets})
    domains = sorted({str(item["asset_domain"]) for item in index.assets})
    print(
        json.dumps(
            {
                "path": str(index.root.resolve()),
                "encoder": index.encoder_model_name,
                "asset_count": len(index.assets),
                "part_prototype_count": len(index.parts),
                "asset_labels": labels,
                "asset_domains": domains,
                "arrays_sha256": index.manifest["arrays_sha256"],
            },
            indent=2,
        )
    )
    return 0


def _routed_roots(
    candidates: list[MaskCandidate],
    routed_diagnostics: dict[str, object],
    root_mode: str,
) -> list[MaskCandidate]:
    roots = [
        candidate
        for candidate in candidates
        if candidate.semantic_name == candidate.semantic_parent
        and candidate.metadata.get("root_index") is not None
        and candidate.metadata.get("parent_candidate_key") is None
    ]
    if root_mode != "primary":
        return roots
    primary_root_keys = {
        str(row["root_key"])
        for row in routed_diagnostics.get("root_scores", [])
        if isinstance(row, dict) and row.get("selection_role") == "primary"
    }
    return [
        root
        for root in roots
        if (
            f"{root.metadata.get('root_origin', 'legacy')}::"
            f"{root.metadata.get('root_index', 'unknown')}"
        )
        in primary_root_keys
    ] or roots[:1]


def _requires_grounded_profile_refinement(
    root: MaskCandidate,
    domain_lookup: dict[str, DomainPrompt],
) -> bool:
    """Return whether a routed profile needs category-guided mask discovery."""

    profile_name = str(root.metadata.get("selected_part_profile") or "").strip()
    domain = domain_lookup.get(root.semantic_name)
    if domain is None:
        return False
    if not profile_name:
        required_profiles = [
            profile
            for profile in domain.part_profiles
            if profile.requires_grounded_refinement
        ]
        return len(domain.part_profiles) == 1 and len(required_profiles) == 1
    return any(
        profile.name == profile_name and profile.requires_grounded_refinement
        for profile in domain.part_profiles
    )


def _global_asset_proposals(
    args: argparse.Namespace,
    image: Image.Image,
    prompt_bank: PromptBank,
) -> tuple[
    tuple[AutomaticAssetQuery, ...],
    dict[str, object] | None,
    AssetRoute | None,
]:
    """Route the full image before root detection, without a target mask."""

    if not (
        args.decomposition_mode == "automatic"
        and args.root_mode == "primary"
        and args.asset_router_index is not None
        and not args.asset_prompt.strip()
    ):
        return (), None, None
    index = AssetRoutingIndex.load(args.asset_router_index)
    model_name = args.asset_router_model.strip() or str(
        index.manifest["encoder_model_name"]
    )
    router_device = _device(args.asset_router_device)
    encoder = Siglip2AssetEncoder(
        model_name,
        device=router_device,
        local_files_only=args.local_files_only,
        batch_size=args.asset_router_batch_size,
    )
    config = AssetRouterConfig(
        prototype_weight=args.asset_router_prototype_weight,
        text_weight=1.0 - args.asset_router_prototype_weight,
        nearest_asset_weight=args.asset_router_nearest_weight,
        minimum_score=args.asset_router_minimum_score,
        minimum_margin=args.asset_router_minimum_margin,
    )
    router = AssetRouter(index, encoder, config=config)
    full_image = np.ones((image.height, image.width), dtype=bool)
    route = router.route(image, full_image)
    alternatives = {
        str(row["asset_label"]): row for row in route.alternatives
    }
    supported_domains = {domain.name for domain in prompt_bank.domains}
    queries: list[AutomaticAssetQuery] = []
    ordered_alternatives = list(route.alternatives)
    for rank, label in enumerate(route.candidate_labels, start=1):
        row = alternatives.get(label, index.label_metadata.get(label, {}))
        domain = str(row.get("asset_domain", ""))
        if domain not in supported_domains:
            continue
        profile = row.get("asset_profile")
        queries.append(
            AutomaticAssetQuery(
                label=label,
                domain=domain,
                profile=str(profile) if profile else None,
                score=float(row.get("score", 0.0)),
                rank=rank,
                accepted=bool(route.accepted and label == route.asset_label),
                negative_labels=tuple(
                    str(other["asset_label"])
                    for other in ordered_alternatives
                    if str(other["asset_label"]) != label
                    and str(other.get("asset_domain", "")) != domain
                )[:3],
            )
        )
    diagnostics = {
        "algorithm": "hpid-global-asset-proposal-router-v1",
        "index": str(args.asset_router_index.resolve()),
        "index_manifest_sha256": _sha256(
            args.asset_router_index / "index.json"
        ),
        "model": model_name,
        "route": route_to_dict(route),
        "queries": [
            {
                "label": query.label,
                "domain": query.domain,
                "profile": query.profile,
                "score": query.score,
                "rank": query.rank,
                "accepted": query.accepted,
                "negative_labels": list(query.negative_labels),
            }
            for query in queries
        ],
        "ground_truth_used": False,
        "target_mask_used": False,
    }
    encoder.release()
    del router, encoder
    if router_device == "cuda":
        torch.cuda.empty_cache()
    return tuple(queries), diagnostics, route


def command_auto(args: argparse.Namespace) -> int:
    command_started = perf_counter()
    timing_checkpoint = command_started
    stage_timings: dict[str, float] = {}

    def finish_stage(name: str) -> None:
        nonlocal timing_checkpoint
        now = perf_counter()
        stage_timings[name] = round(now - timing_checkpoint, 4)
        timing_checkpoint = now

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _device(args.device)
    guided_text = ""
    if args.part_prompts_file is not None:
        guided_text = args.part_prompts_file.read_text(encoding="utf-8")
    elif args.part_prompts:
        guided_text = args.part_prompts
    guided_prompts = parse_guided_prompts(guided_text) if guided_text.strip() else ()
    if args.decomposition_mode == "prompt-guided" and not guided_prompts:
        raise ValueError(
            "prompt-guided decomposition requires --part-prompts or --part-prompts-file"
        )
    if args.decomposition_mode == "automatic" and guided_prompts:
        raise ValueError(
            "part prompts were supplied in automatic mode; select "
            "--decomposition-mode prompt-guided"
        )
    if args.proposal_first_fast and args.decomposition_mode != "automatic":
        raise ValueError("proposal-first fast path requires automatic mode")
    if args.proposal_first_fast and args.additional_grounding_model:
        raise ValueError(
            "proposal-first fast path cannot be combined with detector ensemble"
        )
    if args.florence_parts and (
        args.decomposition_mode != "automatic" or args.retrieval_index is None
    ):
        raise ValueError(
            "--florence-parts requires automatic mode and --retrieval-index"
        )
    use_dense_semantic = args.dense_semantic_fallback or bool(guided_prompts)
    prompt_bank = PromptBank.from_json(args.prompt_bank)
    if args.domains:
        selected = {value.strip() for value in args.domains.split(",") if value.strip()}
        prompt_bank = PromptBank(
            tuple(domain for domain in prompt_bank.domains if domain.name in selected)
        )
        missing = selected - {domain.name for domain in prompt_bank.domains}
        if missing:
            raise ValueError(f"unknown prompt-bank domains: {sorted(missing)}")
    image = Image.open(args.image).convert("RGB")
    scene_mode = args.root_mode == "scene"
    maximum_roots_per_domain = (
        args.maximum_roots_per_domain
        if args.maximum_roots_per_domain is not None
        else (24 if scene_mode else 4)
    )
    maximum_total_roots = (
        args.maximum_total_roots
        if args.maximum_total_roots is not None
        else (96 if scene_mode else 16)
    )
    if maximum_roots_per_domain < 1 or maximum_total_roots < 1:
        raise ValueError("root candidate limits must be positive")
    (
        automatic_asset_queries,
        global_asset_proposal_diagnostics,
        global_asset_route,
    ) = (
        _global_asset_proposals(args, image, prompt_bank)
    )
    generator = FoundationCandidateGenerator(
        prompt_bank,
        device=device,
        config=FoundationConfig(
            grounding_model=args.grounding_model,
            segmentation_model=args.segmentation_model,
            dense_semantic_model=args.dense_semantic_model,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            use_dense_semantic_fallback=use_dense_semantic,
            use_root_domain_arbitration=not args.no_root_domain_arbitration,
            use_scene_profile_root_queries=(
                scene_mode and not args.no_scene_profile_root_queries
            ),
            maximum_roots_per_domain=maximum_roots_per_domain,
            maximum_total_roots=maximum_total_roots,
            local_files_only=args.local_files_only,
            asset_prompt=args.asset_prompt,
            automatic_asset_queries=automatic_asset_queries,
            use_semantic_part_multimask_selection=(
                args.semantic_part_multimask
            ),
            lazy_grounding_model=args.proposal_first_fast,
        ),
    )
    proposal_first_proposals: tuple[VisualMaskProposal, ...] | None = None
    proposal_first_diagnostics: dict[str, object] | None = None
    if args.proposal_first_fast:
        proposal_first_backend = Sam2VisualRegionProposer(
            generator.sam_processor,
            generator.sam_model,
            segmentation_model=generator.config.segmentation_model,
            device=generator.device,
            config=VisualRegionConfig(
                points_per_crop=args.visual_points_per_crop,
                crops_n_layers=0,
            ),
        )
        raw_proposals, raw_proposal_diagnostics = (
            proposal_first_backend.propose_global(image)
        )
        proposal_first = generate_proposal_first_roots(
            image,
            raw_proposals,
            prompt_bank,
            generator.dense_proposer,
            preferred_route=global_asset_route,
            target_point_xy=(
                tuple(float(value) for value in args.target_point)
                if args.target_point is not None
                else None
            ),
            config=ProposalFirstConfig(
                root_mode=args.root_mode,
                maximum_roots=maximum_total_roots,
            ),
        )
        detector_fallback: CandidateGeneration | None = None
        fallback_recommended = bool(
            proposal_first.diagnostics.get("primary_root_quality", {}).get(
                "fallback_recommended", False
            )
        )
        if fallback_recommended and args.root_mode == "primary":
            generator.activate_grounding_model()
            detector_fallback = generator.generate(image)
            shape_roots = [
                candidate
                for candidate in detector_fallback.candidates
                if candidate.semantic_name == candidate.semantic_parent
                and candidate.metadata.get("parent_candidate_key") is None
            ]
        else:
            shape_roots = list(proposal_first.roots)
        shape_proposals = propose_shape_regions(
            shape_roots,
            existing_visual_proposals=raw_proposals,
            config=ShapeProposalConfig(
                minimum_outer_boundary_contact=0.58 if scene_mode else 0.50,
                maximum_markers=8 if scene_mode else 14,
                maximum_regions_per_root=5 if scene_mode else 12,
            ),
        )
        use_appearance_proposals = not scene_mode
        if use_appearance_proposals:
            character_roots = bool(
                shape_roots
                and all(root.semantic_name == "character" for root in shape_roots)
            )
            appearance_proposals = propose_appearance_regions(
                image,
                shape_roots,
                config=AppearanceProposalConfig(
                    analysis_maximum_dimension=480,
                    maximum_root_fraction=0.36 if character_roots else 0.58,
                    maximum_regions_per_root=10 if character_roots else 18,
                    detail_region_limit=2 if character_roots else 4,
                    small_region_limit=4 if character_roots else 5,
                    medium_region_limit=4 if character_roots else 6,
                    large_region_limit=0 if character_roots else 3,
                    use_graph_regions=not character_roots,
                    use_enclosed_interiors=not character_roots,
                ),
            )
        else:
            appearance_proposals = None
        proposal_first_proposals = (
            *proposal_first.proposals,
            *shape_proposals.proposals,
            *(
                appearance_proposals.proposals
                if appearance_proposals is not None
                else ()
            ),
        )
        proposal_first_diagnostics = {
            **proposal_first.diagnostics,
            "proposal_generation": raw_proposal_diagnostics,
            "shape_proposal_generation": shape_proposals.diagnostics,
            "appearance_proposal_generation": (
                appearance_proposals.diagnostics
                if appearance_proposals is not None
                else {
                    "algorithm": "hpid-multiscale-appearance-proposals-v2",
                    "status": "skipped_for_scene",
                    "proposal_count": 0,
                    "ground_truth_used": False,
                }
            ),
            "combined_proposal_count": len(proposal_first_proposals),
            "detector_fallback": (
                detector_fallback.diagnostics
                if detector_fallback is not None
                else None
            ),
        }
        if detector_fallback is not None:
            generated = CandidateGeneration(
                detector_fallback.candidates,
                {
                    "algorithm": "hpid-proposal-first-detector-fallback-v1",
                    "root_count": len(shape_roots),
                    "candidate_count": len(detector_fallback.candidates),
                    "proposal_first": proposal_first_diagnostics,
                    "detector_generation": detector_fallback.diagnostics,
                    "models": {
                        "segmentation_model": generator.config.segmentation_model,
                        "dense_semantic_model": generator.config.dense_semantic_model,
                        "grounding_model": generator.config.grounding_model,
                        "grounding_model_loaded": True,
                    },
                    "ground_truth_used": False,
                },
            )
        else:
            generated = CandidateGeneration(
                proposal_first.roots,
                {
                    "algorithm": "hpid-proposal-first-fast-candidate-generation-v2",
                    "root_count": len(proposal_first.roots),
                    "candidate_count": len(proposal_first.roots),
                    "proposal_first": proposal_first_diagnostics,
                    "models": {
                        "segmentation_model": generator.config.segmentation_model,
                        "dense_semantic_model": generator.config.dense_semantic_model,
                        "grounding_model_loaded": False,
                    },
                    "ground_truth_used": False,
                },
            )
    else:
        generated = generator.generate(image)
    finish_stage("proposal_and_root_generation")
    candidates = list(generated.candidates)
    candidate_generations = [generated.diagnostics]
    guided_fallback_generators = [generator]
    additional_models = list(dict.fromkeys(args.additional_grounding_model))
    if args.grounding_model in additional_models:
        additional_models.remove(args.grounding_model)
    for grounding_model in additional_models:
        generator.release_grounding_model()
        additional_generator = FoundationCandidateGenerator(
            prompt_bank,
            device=device,
            config=FoundationConfig(
                grounding_model=grounding_model,
                segmentation_model=args.segmentation_model,
                dense_semantic_model=args.dense_semantic_model,
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                use_dense_semantic_fallback=use_dense_semantic,
                use_root_domain_arbitration=not args.no_root_domain_arbitration,
                use_scene_profile_root_queries=(
                    scene_mode and not args.no_scene_profile_root_queries
                ),
                maximum_roots_per_domain=maximum_roots_per_domain,
                maximum_total_roots=maximum_total_roots,
                local_files_only=args.local_files_only,
                asset_prompt=args.asset_prompt,
                automatic_asset_queries=automatic_asset_queries,
                use_semantic_part_multimask_selection=(
                    args.semantic_part_multimask
                ),
                lazy_grounding_model=False,
            ),
            sam_processor=generator.sam_processor,
            sam_model=generator.sam_model,
            dense_proposer=generator.dense_proposer,
        )
        additional_generated = additional_generator.generate(image)
        candidates.extend(additional_generated.candidates)
        candidate_generations.append(additional_generated.diagnostics)
        additional_generator.release_grounding_model()
        guided_fallback_generators.append(additional_generator)
    learned_diagnostics: dict[str, object] | None = None
    if args.learned_checkpoint is not None:
        model, learned_taxonomy, checkpoint_payload = load_checkpoint(
            args.learned_checkpoint, device
        )
        learned_prediction = predict(
            model,
            image,
            learned_taxonomy,
            device=device,
            evaluation_height=args.learned_evaluation_height,
        )
        name_mapping = (
            json.loads(args.learned_name_mapping.read_text(encoding="utf-8"))
            if args.learned_name_mapping is not None
            else {}
        )
        learned_candidates = semantic_prediction_candidates(
            learned_prediction,
            learned_taxonomy,
            semantic_parent=args.learned_domain,
            name_mapping={str(key): str(value) for key, value in name_mapping.items()},
        )
        candidates.extend(learned_candidates)
        learned_diagnostics = {
            "checkpoint": str(args.learned_checkpoint),
            "candidate_count": len(learned_candidates),
            "training_role_count": len(
                checkpoint_payload.get(
                    "role_ids", checkpoint_payload.get("train_roles", [])
                )
            ),
            "ground_truth_used": False,
        }
    if not candidates:
        raise RuntimeError(
            "No asset candidate was found. Extend the prompt bank or lower the "
            "box threshold; no empty result was silently exported."
        )
    candidates, root_domain_diagnostics = generator.attach_root_domain_evidence(
        image, candidates
    )
    routed = route_asset_roots(
        candidates,
        image_shape=(image.height, image.width),
        image=image,
        config=RootRoutingConfig(
            mode=args.root_mode,
            target_point_xy=(
                tuple(float(value) for value in args.target_point)
                if args.target_point is not None
                else None
            ),
        ),
    )
    initial_root_routing_diagnostics = routed.diagnostics
    candidates = list(routed.candidates)
    asset_domain_routing_diagnostics: dict[str, object] | None = None
    early_asset_router_diagnostics: dict[str, object] | None = None
    early_automatic_asset_candidates: dict[str, tuple[str, ...]] = {}
    early_asset_routes_by_key: dict[str, AssetRoute] = {}
    early_profile_text_routes_by_key: dict[
        str, dict[str, ProfileTextRoute]
    ] = {}
    early_router_supported_domains: set[str] = set()
    if (
        args.decomposition_mode == "automatic"
        and args.root_mode == "primary"
        and args.asset_router_index is not None
        and not args.asset_prompt.strip()
    ):
        generator.release_grounding_model()
        router_index = AssetRoutingIndex.load(args.asset_router_index)
        early_router_supported_domains = {
            str(row["asset_domain"]) for row in router_index.assets
        }
        router_model = args.asset_router_model.strip() or str(
            router_index.manifest["encoder_model_name"]
        )
        router_device = _device(args.asset_router_device)
        router_encoder = Siglip2AssetEncoder(
            router_model,
            device=router_device,
            local_files_only=args.local_files_only,
            batch_size=args.asset_router_batch_size,
        )
        router_config = AssetRouterConfig(
            prototype_weight=args.asset_router_prototype_weight,
            text_weight=1.0 - args.asset_router_prototype_weight,
            nearest_asset_weight=args.asset_router_nearest_weight,
            minimum_score=args.asset_router_minimum_score,
            minimum_margin=args.asset_router_minimum_margin,
        )
        asset_router = AssetRouter(
            router_index,
            router_encoder,
            config=router_config,
        )
        initial_asset_roots = _routed_roots(
            candidates, routed.diagnostics, args.root_mode
        )
        route_rows: list[dict[str, object]] = []
        domains_by_name = {domain.name: domain for domain in prompt_bank.domains}
        for root in initial_asset_roots:
            root_key = candidate_root_key(root)
            if root_key is None:
                continue
            root_embedding = asset_router.encode_root(image, root.mask.astype(bool))
            asset_route = asset_router.route_embedding(root_embedding)
            early_asset_routes_by_key[root_key] = asset_route
            profile_domain_name = (
                str(asset_route.asset_domain)
                if asset_route.accepted and asset_route.asset_domain is not None
                else root.semantic_name
                if root.semantic_name in domains_by_name
                else asset_route.candidate_domains[0]
                if len(asset_route.candidate_domains) == 1
                else ""
            )
            profile_text_routes = route_profile_text_inventories(
                root_embedding,
                tuple(domains_by_name.values()),
                router_encoder,
            )
            early_profile_text_routes_by_key[root_key] = profile_text_routes
            profile_text_route = profile_text_routes.get(profile_domain_name)
            if (
                (
                    root.semantic_name in early_router_supported_domains
                    or asset_route.accepted
                )
                and asset_route.candidate_labels
            ):
                early_automatic_asset_candidates[root_key] = (
                    asset_route.candidate_labels
                )
            route_rows.append(
                {
                    "root_key": root_key,
                    **route_to_dict(asset_route),
                    "profile_text_route": (
                        profile_text_route_to_dict(profile_text_route)
                        if profile_text_route is not None
                        else None
                    ),
                }
            )
        candidates, asset_domain_routing_diagnostics = _apply_asset_domain_routes(
            candidates,
            initial_asset_roots,
            early_asset_routes_by_key,
            prompt_bank,
            config=router_config,
            supported_domains=early_router_supported_domains,
            full_image_route=global_asset_route,
            profile_text_routes_by_root=early_profile_text_routes_by_key,
        )
        rerouted = route_asset_roots(
            candidates,
            image_shape=(image.height, image.width),
            image=image,
            config=RootRoutingConfig(
                mode="primary",
                include_attached_roots=False,
                target_point_xy=(
                    tuple(float(value) for value in args.target_point)
                    if args.target_point is not None
                    else None
                ),
            ),
        )
        candidates = list(rerouted.candidates)
        routed = rerouted
        early_asset_router_diagnostics = {
            "algorithm": "hpid-siglip2-asset-router-v1",
            "stage": "preprofile-domain-routing",
            "index": str(args.asset_router_index.resolve()),
            "index_manifest_sha256": _sha256(
                args.asset_router_index / "index.json"
            ),
            "model": router_model,
            "root_count": len(initial_asset_roots),
            "exact_route_count": sum(bool(row["accepted"]) for row in route_rows),
            "candidate_set_route_count": sum(
                row["reason"] == "ambiguous_candidate_set" for row in route_rows
            ),
            "routes": route_rows,
            "ground_truth_used_during_query_inference": False,
        }
        router_encoder.release()
        del asset_router, router_encoder
        if router_device == "cuda":
            torch.cuda.empty_cache()
        if not args.proposal_first_fast:
            generator.activate_grounding_model()
    profile_root_resolution_diagnostics: dict[str, object] | None = None
    if args.decomposition_mode in {"automatic", "prompt-guided"}:
        broad_roots = _routed_roots(candidates, routed.diagnostics, args.root_mode)
        user_prompt_resolved = bool(
            args.asset_prompt.strip()
            and broad_roots
            and all(
                str(root.metadata.get("root_query_mode", "")).startswith(
                    "user_asset_prompt"
                )
                and root.metadata.get("selected_part_profile")
                for root in broad_roots
            )
        )
        asset_router_profile_resolved = bool(
            broad_roots
            and all(
                root.metadata.get("profile_resolution_status") == "accepted"
                and root.metadata.get("profile_hint_source")
                in {
                    "exact_asset_router",
                    "domain_conditioned_asset_router",
                    "lightweight_profile_text_router",
                }
                and root.metadata.get("selected_part_profile")
                for root in broad_roots
            )
        )
        profile_resolution_bypassed = bool(args.no_isolated_profile_resolution)
        profile_lock_resolved = user_prompt_resolved or asset_router_profile_resolved
        if profile_resolution_bypassed:
            profile_lock_resolved = True
            profile_root_resolution_diagnostics = {
                "algorithm": "hpid-broad-root-profile-pass-through-v1",
                "status": "isolated_profile_resolution_skipped",
                "root_count": len(broad_roots),
                "selected_profiles": sorted(
                    {
                        str(root.metadata["selected_part_profile"])
                        for root in broad_roots
                        if root.metadata.get("selected_part_profile")
                    }
                ),
                "ground_truth_used": False,
            }
        elif profile_lock_resolved:
            profile_root_resolution_diagnostics = {
                "algorithm": (
                    "user-asset-prompt-profile-lock-v1"
                    if user_prompt_resolved
                    else "asset-router-profile-lock-v2"
                ),
                "lock_source": (
                    "user_asset_prompt"
                    if user_prompt_resolved
                    else str(broad_roots[0].metadata.get("profile_hint_source"))
                ),
                "asset_prompt": args.asset_prompt.strip() or None,
                "root_count": len(broad_roots),
                "selected_profiles": sorted(
                    {
                        str(root.metadata["selected_part_profile"])
                        for root in broad_roots
                    }
                ),
                "ground_truth_used": False,
            }
        else:
            profile_roots = generator.generate_isolated_profile_roots(
                image,
                broad_roots,
                {domain.name: domain for domain in prompt_bank.domains},
            )
        if not profile_lock_resolved and profile_roots.candidates:
            # The broad-root router has already decided which physical object is
            # being decomposed.  Re-running primary-root competition with every
            # profile query here allowed a visually similar profile (for example,
            # staff) to replace a correctly grounded hammer, watch, or scissors.
            # Profile resolution already has a guarded isolated-root replacement
            # path for genuinely bad broad roots, so keep the selected physical
            # root stable and use profile candidates only as semantic evidence.
            profile_geometry_routing = {
                "algorithm": "hpid-profile-root-stability-gate-v1",
                "status": "broad_root_preserved",
                "broad_root_count": len(broad_roots),
                "profile_candidate_count": len(profile_roots.candidates),
                "ground_truth_used": False,
            }
            roots_for_resolution = broad_roots
            resolved_profiles = resolve_profile_roots(
                roots_for_resolution,
                list(profile_roots.candidates),
                image_shape=(image.height, image.width),
                domains={domain.name: domain for domain in prompt_bank.domains},
            )
            if args.root_mode == "primary":
                candidates = [
                    candidate
                    for candidate in candidates
                    if not (
                        candidate.semantic_name == candidate.semantic_parent
                        and candidate.metadata.get("parent_candidate_key") is None
                    )
                ]
                candidates.extend(resolved_profiles.roots)
            else:
                resolved_by_key = {
                    (
                        f"{root.metadata.get('root_origin', 'legacy')}::"
                        f"{root.metadata.get('root_index', 'unknown')}"
                    ): root
                    for root in resolved_profiles.roots
                }
                resolved_candidates: list[MaskCandidate] = []
                for candidate in candidates:
                    if not (
                        candidate.semantic_name == candidate.semantic_parent
                        and candidate.metadata.get("parent_candidate_key") is None
                    ):
                        resolved_candidates.append(candidate)
                        continue
                    root_key = (
                        f"{candidate.metadata.get('root_origin', 'legacy')}::"
                        f"{candidate.metadata.get('root_index', 'unknown')}"
                    )
                    resolved_candidates.append(resolved_by_key.get(root_key, candidate))
                candidates = resolved_candidates
            profile_consensus_diagnostics = resolved_profiles.diagnostics
            profile_domain_evidence = profile_geometry_routing
            if args.root_mode in {"primary", "scene"}:
                rerouted = route_asset_roots(
                    candidates,
                    image_shape=(image.height, image.width),
                    image=image,
                    config=RootRoutingConfig(
                        mode=args.root_mode,
                        include_attached_roots=args.root_mode == "scene",
                        target_point_xy=(
                            tuple(float(value) for value in args.target_point)
                            if args.target_point is not None
                            else None
                        ),
                    ),
                )
                candidates = list(rerouted.candidates)
                routed = rerouted
        elif not profile_lock_resolved:
            profile_consensus_diagnostics = None
            profile_domain_evidence = None
        if not profile_lock_resolved:
            profile_root_resolution_diagnostics = {
                **profile_roots.diagnostics,
                "root_domain_evidence": profile_domain_evidence,
                "profile_consensus": profile_consensus_diagnostics,
            }
    ontology_routing_diagnostics: dict[str, object] | None = None
    if (
        args.decomposition_mode == "automatic"
        and args.root_mode == "scene"
        and not args.no_ontology_scene_consensus
    ):
        ontology_model = _ontology_router_model(args)
        if ontology_model:
            generator.release_grounding_model()
            ontology_device = _device(args.asset_router_device)
            ontology_encoder = Siglip2AssetEncoder(
                ontology_model,
                device=ontology_device,
                local_files_only=args.local_files_only,
                batch_size=args.asset_router_batch_size,
            )
            scene_roots = _routed_roots(candidates, routed.diagnostics, args.root_mode)
            ontology_result = route_scene_ontology(
                image,
                scene_roots,
                prompt_bank,
                ontology_encoder,
            )
            ontology_encoder.release()
            del ontology_encoder
            ontology_by_key = {
                candidate_root_key(root): root for root in ontology_result.roots
            }
            candidates = [
                ontology_by_key.get(candidate_root_key(candidate), candidate)
                if candidate.semantic_name == candidate.semantic_parent
                and candidate.metadata.get("parent_candidate_key") is None
                else candidate
                for candidate in candidates
            ]
            rerouted = route_asset_roots(
                candidates,
                image_shape=(image.height, image.width),
                image=image,
                config=RootRoutingConfig(mode="scene"),
            )
            candidates = list(rerouted.candidates)
            routed = rerouted
            ontology_routing_diagnostics = ontology_result.diagnostics
        else:
            ontology_routing_diagnostics = {
                "algorithm": "hpid-ontology-scene-consensus-v1",
                "status": "skipped_no_siglip2_model",
                "ground_truth_used": False,
            }
    profile_refinement_diagnostics: dict[str, object] | None = None
    if args.decomposition_mode in {"automatic", "prompt-guided"} and (
        not args.no_profile_refinement or args.adaptive_profile_refinement
    ):
        profile_roots = _routed_roots(candidates, routed.diagnostics, args.root_mode)
        profile_refinement_runs: list[CandidateGeneration] = []
        domain_lookup = {domain.name: domain for domain in prompt_bank.domains}
        refinement_mode = "full"
        if args.no_profile_refinement:
            refinement_mode = "adaptive"
            profile_roots = [
                root
                for root in profile_roots
                if _requires_grounded_profile_refinement(root, domain_lookup)
            ]
        if profile_roots:
            for refinement_generator in guided_fallback_generators:
                refinement_generator.activate_grounding_model()
                profile_refinement_runs.append(
                    refinement_generator.refine_profile_parts(
                        image,
                        profile_roots,
                        domain_lookup,
                    )
                )
                refinement_generator.release_grounding_model()
            profile_refinement = _combine_profile_refinements(
                profile_refinement_runs
            )
            candidates.extend(profile_refinement.candidates)
            profile_refinement_diagnostics = {
                **profile_refinement.diagnostics,
                "mode": refinement_mode,
                "refined_root_count": len(profile_roots),
            }
        else:
            profile_refinement_diagnostics = {
                "algorithm": "hpid-adaptive-profile-refinement-v1",
                "status": "skipped_no_profile_requires_grounding",
                "mode": refinement_mode,
                "refined_root_count": 0,
                "ground_truth_used": False,
            }
    retrieval_diagnostics: dict[str, object] | None = None
    prototype_retriever: PrototypeRetriever | None = None
    prototype_result = None
    retrieval_roots: list[MaskCandidate] = []
    if args.retrieval_index is not None and args.decomposition_mode == "automatic":
        retrieval_index = PrototypeIndex.load(args.retrieval_index)
        retrieval_supported_domains = {
            str(row["asset_domain"]) for row in retrieval_index.assets
        }
        dense_proposer = getattr(generator, "dense_proposer", None)
        retrieval_encoder = CLIPSegEmbeddingEncoder(
            model_name=retrieval_index.encoder_model_name,
            device=device,
            local_files_only=args.local_files_only,
            batch_size=args.retrieval_embedding_batch_size,
            processor=(
                dense_proposer.processor if dense_proposer is not None else None
            ),
            model=dense_proposer.model if dense_proposer is not None else None,
        )
        retrieval_config = RetrievalConfig(
            top_k_assets=args.retrieval_top_k,
            maximum_part_prompts=args.retrieval_maximum_part_prompts,
            minimum_asset_similarity=args.retrieval_minimum_asset_similarity,
            minimum_prompted_asset_similarity=(
                args.retrieval_minimum_prompted_asset_similarity
            ),
            minimum_profiled_asset_similarity=(
                args.retrieval_minimum_profiled_asset_similarity
            ),
            minimum_part_similarity=args.retrieval_minimum_part_similarity,
            profile_similarity_bonus=args.retrieval_profile_similarity_bonus,
            allow_domain_relabel=args.retrieval_allow_domain_relabel,
        )
        retriever = PrototypeRetriever(
            retrieval_index,
            retrieval_encoder,
            config=retrieval_config,
        )
        prototype_retriever = retriever
        all_initial_roots = _routed_roots(
            candidates, routed.diagnostics, args.root_mode
        )
        initial_roots = [
            root
            for root in all_initial_roots
            if root.semantic_name in retrieval_supported_domains
        ]
        skipped_retrieval_roots = [
            {
                "root_key": PrototypeRetriever._root_key(root),
                "domain": root.semantic_name,
                "reason": "domain_outside_retrieval_inventory",
            }
            for root in all_initial_roots
            if root.semantic_name not in retrieval_supported_domains
        ]
        asset_router_diagnostics = early_asset_router_diagnostics
        eligible_retrieval_keys = {
            PrototypeRetriever._root_key(root) for root in initial_roots
        }
        automatic_asset_candidates = {
            key: values
            for key, values in early_automatic_asset_candidates.items()
            if key in eligible_retrieval_keys
        }
        if (
            args.asset_router_index is not None
            and not args.asset_prompt.strip()
            and not early_asset_routes_by_key
        ):
            generator.release_grounding_model()
            router_index = AssetRoutingIndex.load(args.asset_router_index)
            router_supported_domains = {
                str(row["asset_domain"]) for row in router_index.assets
            }
            router_model = args.asset_router_model.strip() or str(
                router_index.manifest["encoder_model_name"]
            )
            router_device = _device(args.asset_router_device)
            router_encoder = Siglip2AssetEncoder(
                router_model,
                device=router_device,
                local_files_only=args.local_files_only,
                batch_size=args.asset_router_batch_size,
            )
            asset_router = AssetRouter(
                router_index,
                router_encoder,
                config=AssetRouterConfig(
                    prototype_weight=args.asset_router_prototype_weight,
                    text_weight=1.0 - args.asset_router_prototype_weight,
                    nearest_asset_weight=args.asset_router_nearest_weight,
                    minimum_score=args.asset_router_minimum_score,
                    minimum_margin=args.asset_router_minimum_margin,
                ),
            )
            route_rows: list[dict[str, object]] = []
            for root in initial_roots:
                asset_route = asset_router.route(image, root.mask.astype(bool))
                root_key = PrototypeRetriever._root_key(root)
                if (
                    (
                        root.semantic_name in router_supported_domains
                        or asset_route.accepted
                    )
                    and asset_route.candidate_labels
                ):
                    automatic_asset_candidates[root_key] = asset_route.candidate_labels
                route_rows.append(
                    {
                        "root_key": root_key,
                        **route_to_dict(asset_route),
                    }
                )
            asset_router_diagnostics = {
                "algorithm": "hpid-siglip2-asset-router-v1",
                "index": str(args.asset_router_index.resolve()),
                "index_manifest_sha256": _sha256(
                    args.asset_router_index / "index.json"
                ),
                "model": router_model,
                "root_count": len(initial_roots),
                "exact_route_count": sum(bool(row["accepted"]) for row in route_rows),
                "candidate_set_route_count": sum(
                    row["reason"] == "ambiguous_candidate_set" for row in route_rows
                ),
                "routes": route_rows,
                "ground_truth_used_during_query_inference": False,
            }
            del asset_router, router_encoder
            if router_device == "cuda":
                torch.cuda.empty_cache()
        retrieval_result = retriever.query(
            image,
            initial_roots,
            asset_hint=args.asset_prompt.strip() or None,
            asset_candidates_by_root=automatic_asset_candidates,
        )
        prototype_result = retrieval_result
        candidates, domain_corrections = apply_retrieval_domain_corrections(
            candidates,
            retrieval_result,
            config=retrieval_config,
        )
        corrected_roots = _routed_roots(candidates, routed.diagnostics, args.root_mode)
        retrieval_roots = corrected_roots
        root_by_key = {
            (
                f"{root.metadata.get('root_origin', 'legacy')}::"
                f"{root.metadata.get('root_index', 'unknown')}"
            ): root
            for root in corrected_roots
        }
        retrieval_runs: list[dict[str, object]] = []
        retrieval_candidate_count = 0
        florence_generator: FlorencePartGenerator | None = None
        for plan in retrieval_result.plans:
            root = root_by_key.get(plan.root_key)
            if not plan.accepted or root is None or not plan.part_priors:
                continue
            prompt_specs = tuple(prior.guided_spec() for prior in plan.part_priors)
            generated_for_root: list[MaskCandidate] = []
            model_rows: list[dict[str, object]] = []
            for fallback_generator in guided_fallback_generators:
                fallback_generator.activate_grounding_model()
                retrieved_run = fallback_generator.generate_guided_parts(
                    image,
                    [root],
                    prompt_specs,
                    require_dense_gate=False,
                )
                fallback_generator.release_grounding_model()
                generated_for_root.extend(retrieved_run.candidates)
                model_rows.append(retrieved_run.diagnostics)
            if args.florence_parts:
                generator.prepare_for_completion()
                if florence_generator is None:
                    florence_generator = FlorencePartGenerator(
                        device=device,
                        config=FlorencePartConfig(
                            model_name=args.florence_model,
                            phrases_per_part=args.florence_phrases_per_part,
                            local_files_only=args.local_files_only,
                        ),
                    )
                else:
                    florence_generator.activate()
                try:
                    florence_run = florence_generator.generate(
                        image,
                        [root],
                        prompt_specs,
                    )
                finally:
                    florence_generator.release()
                generated_for_root.extend(florence_run.candidates)
                model_rows.append(florence_run.diagnostics)
                retrieval_encoder.model.to(device)
                retrieval_encoder.model.eval()
            reranked, rerank_diagnostics = retriever.rerank_candidates(
                image,
                root,
                generated_for_root,
                plan.part_priors,
                existing_candidates=candidates,
            )
            candidates.extend(reranked)
            retrieval_candidate_count += len(reranked)
            retrieval_runs.append(
                {
                    "root_key": plan.root_key,
                    "asset_label": plan.asset_label,
                    "asset_domain": plan.asset_domain,
                    "prompt_count": len(prompt_specs),
                    "model_runs": model_rows,
                    "reranking": rerank_diagnostics,
                }
            )
        retrieval_diagnostics = {
            **retrieval_result.diagnostics,
            "supported_domains": sorted(retrieval_supported_domains),
            "skipped_root_count": len(skipped_retrieval_roots),
            "skipped_roots": skipped_retrieval_roots,
            "automatic_asset_router": asset_router_diagnostics,
            "domain_corrections": domain_corrections,
            "retrieved_candidate_count": retrieval_candidate_count,
            "candidate_generation_runs": retrieval_runs,
        }
    guided_diagnostics: dict[str, object] | None = None
    resolved_guided_backend: str | None = None
    if guided_prompts:
        roots = _routed_roots(candidates, routed.diagnostics, args.root_mode)
        guided = None
        sam3_error: str | None = None
        if args.guided_backend in {"auto", "sam3"}:
            generator.prepare_for_completion()
            try:
                sam3_generator = Sam3TextPartGenerator(
                    device=device,
                    config=Sam3TextConfig(
                        model_name=args.sam3_model,
                        score_threshold=args.sam3_score_threshold,
                        local_files_only=args.local_files_only,
                    ),
                )
                guided = sam3_generator.generate(image, roots, guided_prompts)
                sam3_generator.release()
                resolved_guided_backend = "sam3"
            except RuntimeError as error:
                sam3_error = str(error)
                if args.guided_backend == "sam3":
                    raise
        if guided is None and args.guided_backend in {"auto", "grounded-sam2"}:
            guided_runs: list[CandidateGeneration] = []
            for fallback_generator in guided_fallback_generators:
                fallback_generator.activate_grounding_model()
                guided_runs.append(
                    fallback_generator.generate_guided_parts(
                        image, roots, guided_prompts
                    )
                )
                fallback_generator.release_grounding_model()
            guided = CandidateGeneration(
                tuple(candidate for run in guided_runs for candidate in run.candidates),
                {
                    "algorithm": "hpid-guided-grounded-sam2-ensemble-v1",
                    "model_run_count": len(guided_runs),
                    "candidate_count": sum(len(run.candidates) for run in guided_runs),
                    "model_runs": [run.diagnostics for run in guided_runs],
                    "automatic_visual_supplement": True,
                    "ground_truth_used": False,
                },
            )
            resolved_guided_backend = "grounded-sam2"
        if guided is None:
            raise RuntimeError("no guided segmentation backend produced candidates")
        candidates.extend(guided.candidates)
        guided_diagnostics = {
            **guided.diagnostics,
            "requested_backend": args.guided_backend,
            "resolved_backend": resolved_guided_backend,
            "fallback_reason": sam3_error,
        }
    visual_region_diagnostics: dict[str, object] | None = None
    if not args.no_visual_regions:
        visual_roots = _routed_roots(candidates, routed.diagnostics, args.root_mode)
        visual_config = VisualRegionConfig(
            points_per_crop=args.visual_points_per_crop,
            crops_n_layers=args.visual_crop_layers,
            crop_n_points_downscale_factor=args.visual_crop_downscale,
            use_isolated_root_crops=scene_mode,
        )
        appearance_supplement = None
        if proposal_first_proposals is not None:
            visual_regions = visual_region_candidates_from_masks(
                list(proposal_first_proposals),
                visual_roots,
                candidates,
                config=visual_config,
                source=(
                    f"sam2-amg[{generator.config.segmentation_model}]"
                    "/proposal-first"
                ),
            )
            visual_region_base_diagnostics = {
                **visual_regions.diagnostics,
                "proposal_first_reuse": proposal_first_diagnostics,
                "model_rerun_count": 0,
            }
        else:
            visual_backend = Sam2AutomaticMaskBackend(
                generator,
                config=visual_config,
            )
            visual_regions = visual_backend.generate(image, tuple(candidates))
            if visual_roots and all(
                root.semantic_name == "character" for root in visual_roots
            ):
                character_appearance = propose_appearance_regions(
                    image,
                    visual_roots,
                    config=AppearanceProposalConfig(
                        analysis_maximum_dimension=480,
                        maximum_root_fraction=0.36,
                        maximum_regions_per_root=10,
                        detail_region_limit=2,
                        small_region_limit=4,
                        medium_region_limit=4,
                        large_region_limit=0,
                        use_graph_regions=False,
                        use_enclosed_interiors=False,
                    ),
                )
                appearance_supplement = visual_region_candidates_from_masks(
                    list(character_appearance.proposals),
                    visual_roots,
                    candidates,
                    config=visual_config,
                    source="hpid-appearance-contour/character-supplement",
                    candidate_namespace="contour",
                )
            visual_region_base_diagnostics = {
                **visual_regions.diagnostics,
                "character_contour_supplement": (
                    {
                        **character_appearance.diagnostics,
                        "converted_candidate_count": len(
                            appearance_supplement.candidates
                        ),
                    }
                    if appearance_supplement is not None
                    else None
                ),
            }
        visual_candidates = [
            *visual_regions.candidates,
            *(
                appearance_supplement.candidates
                if appearance_supplement is not None
                else ()
            ),
        ]
        appearance_graph = optimize_appearance_graph(
            image,
            visual_candidates,
            visual_roots,
        )
        visual_candidates = list(appearance_graph.candidates)
        prototype_region_diagnostics: dict[str, object] | None = None
        if prototype_retriever is not None and prototype_result is not None:
            visual_candidates, prototype_region_diagnostics = (
                prototype_retriever.label_visual_candidates(
                    image,
                    retrieval_roots,
                    visual_candidates,
                    prototype_result.plans,
                    existing_candidates=candidates,
                )
            )
        prototype_semantic_constraints = {
            plan.root_key: {
                prior.output_semantic_name: prior.maximum_instances
                for prior in plan.part_priors
            }
            for plan in (prototype_result.plans if prototype_result is not None else ())
            if plan.accepted and plan.part_priors
        }
        semantic_rerank = rerank_visual_candidates(
            image,
            visual_candidates,
            visual_roots,
            candidates,
            {domain.name: domain for domain in prompt_bank.domains},
            getattr(generator, "dense_proposer", None),
            config=(
                VisualSemanticConfig(minimum_probability=0.12)
                if additional_models
                else None
            ),
            semantic_constraints=prototype_semantic_constraints,
        )
        visual_candidates = list(semantic_rerank.candidates)
        structural_fusion_diagnostics: dict[str, object] | None = None
        if not args.no_structural_fusion:
            structural_fusion = refine_profile_structure(
                visual_candidates,
                visual_roots,
                [*candidates, *visual_candidates],
                {domain.name: domain for domain in prompt_bank.domains},
            )
            visual_candidates = list(structural_fusion.candidates)
            structural_fusion_diagnostics = structural_fusion.diagnostics
        candidates.extend(visual_candidates)
        visual_region_diagnostics = {
            **visual_region_base_diagnostics,
            "appearance_graph": appearance_graph.diagnostics,
            "prototype_labelling": prototype_region_diagnostics,
            "semantic_reranking": semantic_rerank.diagnostics,
            "structural_fusion": structural_fusion_diagnostics,
        }
    vlm_root_audit_diagnostics: dict[str, object] | None = None
    vlm_semantic_audit_diagnostics: dict[str, object] | None = None
    vlm_physicality_audit_diagnostics: dict[str, object] | None = None
    vlm_part_diagnostics: dict[str, object] | None = None
    if args.vlm_parts:
        if args.decomposition_mode != "automatic":
            raise ValueError("--vlm-parts currently requires automatic mode")
        generator.prepare_for_completion()
        vlm_planner = Qwen3VlPartPlanner(
            device=device,
            config=Qwen3VlPlannerConfig(
                model_name=args.vlm_model,
                maximum_new_tokens=args.vlm_maximum_new_tokens,
                local_files_only=args.local_files_only,
                load_in_4bit=args.vlm_load_in_4bit,
            ),
        )
        domains_by_name = {domain.name: domain for domain in prompt_bank.domains}
        root_audit_query_count = 0
        if (
            args.root_mode in {"primary", "scene"}
            and args.vlm_maximum_root_audits > 0
        ):
            audit_roots = _routed_roots(candidates, routed.diagnostics, args.root_mode)
            root_auditor = SceneRootAuditor(
                vlm_planner,
                config=SceneRootAuditConfig(
                    maximum_queries=min(
                        args.vlm_maximum_root_audits,
                        args.vlm_maximum_total_queries,
                    ),
                    maximum_candidates=(48 if args.root_mode == "scene" else 1),
                    batch_size=(6 if args.root_mode == "scene" else 1),
                ),
            )
            root_audit = root_auditor.audit(
                image,
                audit_roots,
                domains_by_name,
            )
            applied_audit = apply_scene_root_audit(
                candidates,
                root_audit.roots,
                domains_by_name,
            )
            candidates = list(applied_audit.candidates)
            root_audit_query_count = int(root_audit.diagnostics["query_count"])
            vlm_root_audit_diagnostics = {
                **root_audit.diagnostics,
                "candidate_application": applied_audit.diagnostics,
            }
            if int(applied_audit.diagnostics["updated_root_count"]) > 0:
                rerouted = route_asset_roots(
                    candidates,
                    image_shape=(image.height, image.width),
                    image=image,
                    config=RootRoutingConfig(
                        mode=args.root_mode,
                        include_attached_roots=args.root_mode == "scene",
                        target_point_xy=(
                            tuple(float(value) for value in args.target_point)
                            if args.target_point is not None
                            else None
                        ),
                    ),
                )
                candidates = list(rerouted.candidates)
                routed = rerouted
        remaining_after_root_audit = max(
            0,
            args.vlm_maximum_total_queries - root_audit_query_count,
        )
        semantic_audit_query_count = 0
        if (
            args.vlm_maximum_semantic_audits > 0
            and remaining_after_root_audit > 0
        ):
            semantic_roots = _routed_roots(
                candidates, routed.diagnostics, args.root_mode
            )
            semantic_auditor = SemanticCandidateAuditor(
                vlm_planner,
                config=SemanticCandidateAuditConfig(
                    maximum_queries=min(
                        args.vlm_maximum_semantic_audits,
                        remaining_after_root_audit,
                    )
                ),
            )
            semantic_audit = semantic_auditor.audit(
                image,
                semantic_roots,
                candidates,
                domains_by_name,
            )
            candidates = list(semantic_audit.candidates)
            semantic_audit_query_count = int(
                semantic_audit.diagnostics["query_count"]
            )
            vlm_semantic_audit_diagnostics = semantic_audit.diagnostics
        remaining_after_semantic_audit = max(
            0,
            remaining_after_root_audit - semantic_audit_query_count,
        )
        physicality_audit_query_count = 0
        if (
            args.vlm_maximum_physicality_audits > 0
            and remaining_after_semantic_audit > 0
        ):
            physicality_roots = _routed_roots(
                candidates, routed.diagnostics, args.root_mode
            )
            physicality_auditor = PhysicalRegionAuditor(
                vlm_planner,
                config=PhysicalRegionAuditConfig(
                    maximum_queries=min(
                        args.vlm_maximum_physicality_audits,
                        remaining_after_semantic_audit,
                    )
                ),
            )
            physicality_audit = physicality_auditor.audit(
                image,
                physicality_roots,
                candidates,
                domains_by_name,
            )
            candidates = list(physicality_audit.candidates)
            physicality_audit_query_count = int(
                physicality_audit.diagnostics["query_count"]
            )
            vlm_physicality_audit_diagnostics = physicality_audit.diagnostics
        remaining_vlm_queries = max(
            0,
            remaining_after_semantic_audit - physicality_audit_query_count,
        )
        vlm_generator = VlmPartGenerator(
            vlm_planner,
            generator._segment_boxes,
            config=VlmPartConfig(
                maximum_planner_queries=args.vlm_maximum_queries,
                maximum_total_planner_queries=remaining_vlm_queries,
                maximum_roots=args.vlm_maximum_roots,
                use_batched_region_label_queries=True,
                use_box_planner_queries=args.vlm_box_planner,
                use_per_semantic_queries=args.vlm_query_mode == "per-semantic",
                query_established_semantics=(
                    args.vlm_query_established_semantics
                ),
                allow_direct_sam_regions=args.vlm_allow_direct_sam_regions,
                use_dynamic_inventory=args.vlm_dynamic_inventory,
            ),
        )
        vlm_roots = _routed_roots(candidates, routed.diagnostics, args.root_mode)
        vlm_roots.sort(
            key=lambda root: bool(root.metadata.get("vlm_root_audit_applied")),
            reverse=True,
        )
        vlm_run = vlm_generator.generate(
            image,
            vlm_roots,
            domains_by_name,
            existing_candidates=tuple(candidates),
        )
        profile_correction = apply_dynamic_object_profile_corrections(
            candidates,
            vlm_run.diagnostics.get("roots", ()),
            domains_by_name,
        )
        candidates = list(profile_correction.candidates)
        candidates.extend(vlm_run.candidates)
        vlm_part_diagnostics = {
            **vlm_run.diagnostics,
            "object_identity_application": profile_correction.diagnostics,
            "planner_query_budget_before_root_audit": (
                args.vlm_maximum_total_queries
            ),
            "root_audit_query_count": root_audit_query_count,
            "semantic_candidate_audit_query_count": (
                semantic_audit_query_count
            ),
            "physicality_audit_query_count": physicality_audit_query_count,
            "remaining_part_query_budget": remaining_vlm_queries,
            "combined_planner_query_count": (
                root_audit_query_count
                + semantic_audit_query_count
                + physicality_audit_query_count
                + int(vlm_run.diagnostics["total_planner_query_count"])
            ),
        }
        vlm_planner.release()
        del vlm_run, vlm_generator, vlm_planner
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    finish_stage("routing_and_semantic_candidates")
    scene_identity_runs: list[dict[str, object]] = []
    if scene_mode:
        scene_roots = _routed_roots(candidates, routed.diagnostics, args.root_mode)
        propagated, propagation_diagnostics = propagate_scene_object_identity(
            candidates,
            scene_roots,
        )
        candidates = list(propagated)
        scene_identity_runs.append(propagation_diagnostics)
    refinement_diagnostics: dict[str, object] | None = None
    if not args.no_mask_refinement:
        refined = refine_candidate_masks(
            image,
            candidates,
            config=MaskRefinementConfig(
                grabcut_iterations=args.grabcut_iterations,
                maximum_grabcut_candidates=args.maximum_grabcut_candidates,
            ),
        )
        candidates = list(refined.candidates)
        refinement_diagnostics = refined.diagnostics
    cleanup_roots = _routed_roots(candidates, routed.diagnostics, args.root_mode)
    root_cleanup = clean_primary_roots(
        candidates,
        cleanup_roots,
        target_point_xy=(
            tuple(float(value) for value in args.target_point)
            if args.target_point is not None
            else None
        ),
    )
    candidates = list(root_cleanup.candidates)
    axis_roots = _routed_roots(candidates, routed.diagnostics, args.root_mode)
    axis_consistency = enforce_axis_consistency(
        candidates,
        axis_roots,
        {domain.name: domain for domain in prompt_bank.domains},
    )
    candidates = list(axis_consistency.candidates)
    physical_region_gate = filter_unresolved_visual_regions(
        candidates,
        axis_roots,
        {domain.name: domain for domain in prompt_bank.domains},
    )
    candidates = list(physical_region_gate.candidates)
    root_geometry = refine_root_geometry_from_parts(
        candidates,
        axis_roots,
        {domain.name: domain for domain in prompt_bank.domains},
    )
    candidates = list(root_geometry.candidates)
    axis_roots = _routed_roots(candidates, routed.diagnostics, args.root_mode)
    ensemble_evidence_gate = None
    if additional_models:
        ensemble_evidence_gate = filter_unresolved_ensemble_regions(
            candidates,
            axis_roots,
        )
        candidates = list(ensemble_evidence_gate.candidates)
    fusion_config = FusionConfig(
        use_parent_envelope=use_dense_semantic,
        use_transitive_residual=use_dense_semantic,
    )
    preliminary = fuse_candidates(
        candidates,
        image_shape=(image.height, image.width),
        config=fusion_config,
    )
    relational_diagnostics: dict[str, object] | None = None
    if not args.no_relational_appearance:
        relational = propose_relational_candidates(
            image,
            preliminary.semantic_map,
            preliminary.taxonomy,
            prompt_bank,
            roots=axis_roots,
        )
        relational_diagnostics = relational.diagnostics
        relational_candidates = list(relational.candidates)
        if relational_candidates and not args.no_mask_refinement:
            relational_refined = refine_candidate_masks(
                image,
                relational_candidates,
                config=MaskRefinementConfig(
                    grabcut_iterations=args.grabcut_iterations,
                    maximum_grabcut_candidates=args.maximum_grabcut_candidates,
                ),
            )
            relational_candidates = list(relational_refined.candidates)
            relational_diagnostics = {
                **relational_diagnostics,
                "mask_refinement": relational_refined.diagnostics,
            }
        candidates.extend(relational_candidates)
        if scene_mode and relational_candidates:
            scene_roots = _routed_roots(candidates, routed.diagnostics, args.root_mode)
            propagated, propagation_diagnostics = propagate_scene_object_identity(
                candidates,
                scene_roots,
            )
            candidates = list(propagated)
            scene_identity_runs.append(propagation_diagnostics)
    fused = (
        fuse_candidates(
            candidates,
            image_shape=(image.height, image.width),
            config=fusion_config,
        )
        if relational_diagnostics is not None
        and int(relational_diagnostics["candidate_count"]) > 0
        else preliminary
    )
    records = list(fused.instances)
    if args.previous is not None:
        previous_map, previous_records = load_previous_package(args.previous)
        records = preserve_part_ids(
            fused.instance_map,
            records,
            previous_map,
            previous_records,
        )
    physical_groups = build_physical_groups(
        fused.instance_map,
        records,
        candidates=fused.accepted_candidates,
        image=image,
        provisional_scene_labels=(
            scene_mode and args.no_ontology_scene_consensus
        ),
    )
    records = list(physical_groups.records)
    finish_stage("root_cleanup_and_physical_fusion")
    diagnostics = {
        "prompt_bank": {
            "path": str(args.prompt_bank.resolve()),
            "sha256": _sha256(args.prompt_bank),
        },
        "candidate_generation": generated.diagnostics,
        "candidate_generations": candidate_generations,
        "candidate_source_count": len(candidate_generations),
        "learned_candidate_generation": learned_diagnostics,
        "root_domain_arbitration": root_domain_diagnostics,
        "global_asset_proposal": global_asset_proposal_diagnostics,
        "initial_root_routing": initial_root_routing_diagnostics,
        "root_routing": routed.diagnostics,
        "asset_domain_routing": asset_domain_routing_diagnostics,
        "profile_root_resolution": profile_root_resolution_diagnostics,
        "ontology_scene_routing": ontology_routing_diagnostics,
        "scene_object_identity": (
            {
                "algorithm": "hpid-scene-object-identity-propagation-v1",
                "runs": scene_identity_runs,
                "ground_truth_used": False,
            }
            if scene_identity_runs
            else None
        ),
        "profile_refinement": profile_refinement_diagnostics,
        "prototype_retrieval": retrieval_diagnostics,
        "decomposition": {
            "mode": args.decomposition_mode,
            "asset_prompt": args.asset_prompt.strip() or None,
            "target_point_xy": (
                [float(value) for value in args.target_point]
                if args.target_point is not None
                else None
            ),
            "guided_prompt_count": len(guided_prompts),
            "guided_backend": resolved_guided_backend,
            "dense_semantic_evidence_enabled": use_dense_semantic,
            "dense_semantic_evidence_forced_for_guided_mode": bool(
                guided_prompts and not args.dense_semantic_fallback
            ),
            "automatic_visual_supplement": True,
            "prototype_retrieval_enabled": retrieval_diagnostics is not None,
            "florence_part_supplement_enabled": bool(args.florence_parts),
            "ground_truth_used": False,
            "inference_seed": args.seed,
        },
        "guided_candidate_generation": guided_diagnostics,
        "visual_region_generation": visual_region_diagnostics,
        "vlm_root_audit": vlm_root_audit_diagnostics,
        "vlm_semantic_candidate_audit": vlm_semantic_audit_diagnostics,
        "vlm_physical_region_audit": vlm_physicality_audit_diagnostics,
        "vlm_part_generation": vlm_part_diagnostics,
        "mask_refinement": refinement_diagnostics,
        "root_cleanup": root_cleanup.diagnostics,
        "axis_consistency": axis_consistency.diagnostics,
        "physical_region_gate": physical_region_gate.diagnostics,
        "root_geometry_refinement": root_geometry.diagnostics,
        "ensemble_evidence_gate": (
            ensemble_evidence_gate.diagnostics
            if ensemble_evidence_gate is not None
            else None
        ),
        "relational_candidate_generation": relational_diagnostics,
        "fusion_pass_count": 2 if fused is not preliminary else 1,
        "fusion": fused.diagnostics,
        "physical_grouping": physical_groups.diagnostics,
        "stage_timings_seconds": stage_timings,
    }
    completion_records = None
    if args.completion_config is not None or args.geometric_fallback:
        if args.completion_config is not None:
            generator.prepare_for_completion()
            backend = load_completion_backend(
                args.completion_config,
                sam_processor=generator.sam_processor,
                sam_model=generator.sam_model,
                device=device,
            )
        else:
            backend = GeometricFallbackBackend()
        completion_records = complete_and_export_parts(
            image, fused.instance_map, records, args.output, backend
        )
        diagnostics["completion"] = {
            "backend": backend.provenance.name,
            "part_count": len(completion_records),
            "is_hpid_split_method": backend.provenance.is_hpid_split_method,
        }
    manifest = export_prediction(
        image,
        fused,
        fused.taxonomy,
        args.output,
        records=records,
        diagnostics=diagnostics,
        completion_records=completion_records,
        candidates=fused.accepted_candidates,
        physical_groups=physical_groups,
    )
    finish_stage("export")
    stage_timings["total"] = round(perf_counter() - command_started, 4)
    print(
        f"parts={manifest['part_count']} mode={args.decomposition_mode} "
        f"guided_backend={resolved_guided_backend or 'none'} "
        f"retrieval={'on' if retrieval_diagnostics is not None else 'off'} "
        f"output={args.output} timings={json.dumps(stage_timings, separators=(',', ':'))}"
    )
    return 0


def command_validate_package(args: argparse.Namespace) -> int:
    result = validate_package(args.package)
    print(json.dumps(result, indent=2))
    return 0 if result["valid"] else 1


def command_setup_completion(args: argparse.Namespace) -> int:
    from .setup_completion import setup_completion_backend

    config_path = setup_completion_backend(
        package_root=args.package_root,
        model_cache=args.model_cache,
        config_path=args.config,
        python_executable=args.python,
        skip_install=args.skip_install,
    )
    print(f"completion_config={config_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hpid-split")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train", help="Train a hierarchy-constrained model."
    )
    train_parser.add_argument("--image-dir", type=Path, required=True)
    train_parser.add_argument("--label-dir", type=Path, required=True)
    train_parser.add_argument("--taxonomy", type=Path, required=True)
    train_parser.add_argument("--checkpoint", type=Path, required=True)
    train_parser.add_argument("--roles", default="")
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--repeats", type=int, default=8)
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--seed", type=int, default=20260811)
    train_parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    train_parser.set_defaults(handler=command_train)

    predict_parser = subparsers.add_parser(
        "predict", help="Export an HPID package for one asset image."
    )
    predict_parser.add_argument("--image", type=Path, required=True)
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--previous", type=Path)
    predict_parser.add_argument("--evaluation-height", type=int, default=768)
    predict_parser.add_argument("--no-recursive", action="store_true")
    predict_parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    predict_parser.set_defaults(handler=command_predict)

    retrieval_build_parser = subparsers.add_parser(
        "build-retrieval-index",
        help="Learn an object/part prototype index from reviewed references.",
    )
    retrieval_build_parser.add_argument("--manifest", type=Path, required=True)
    retrieval_build_parser.add_argument("--output", type=Path, required=True)
    retrieval_build_parser.add_argument(
        "--encoder-model", default="CIDAS/clipseg-rd64-refined"
    )
    retrieval_build_parser.add_argument("--embedding-batch-size", type=int, default=16)
    retrieval_build_parser.add_argument("--metric-epochs", type=int, default=120)
    retrieval_build_parser.add_argument("--seed", type=int, default=20260812)
    retrieval_build_parser.add_argument("--local-files-only", action="store_true")
    retrieval_build_parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    retrieval_build_parser.set_defaults(handler=command_build_retrieval_index)

    retrieval_inspect_parser = subparsers.add_parser(
        "inspect-retrieval-index",
        help="Print the audited contents of a learned prototype index.",
    )
    retrieval_inspect_parser.add_argument("--index", type=Path, required=True)
    retrieval_inspect_parser.set_defaults(handler=command_inspect_retrieval_index)

    auto_parser = subparsers.add_parser(
        "auto", help="Split one image into hierarchy-constrained Part IDs."
    )
    auto_parser.add_argument("--image", type=Path, required=True)
    auto_parser.add_argument("--output", type=Path, required=True)
    auto_parser.add_argument(
        "--prompt-bank",
        type=Path,
        default=DEFAULT_PROMPT_BANK,
    )
    auto_parser.add_argument("--domains", default="")
    auto_parser.add_argument(
        "--asset-prompt",
        default="",
        help=(
            "Optional object-level prompt selecting which asset to decompose, "
            "for example 'serving tray' or 'power drill'."
        ),
    )
    auto_parser.add_argument(
        "--target-point",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help=(
            "Optional image-space point used to select one instance when an "
            "object prompt matches multiple objects."
        ),
    )
    auto_parser.add_argument(
        "--decomposition-mode",
        choices=("automatic", "prompt-guided"),
        default="automatic",
        help=(
            "Use open-set automatic IDs, or prioritize user-named parts while "
            "retaining automatic visual-region supplementation."
        ),
    )
    guided_group = auto_parser.add_mutually_exclusive_group()
    guided_group.add_argument(
        "--part-prompts",
        default="",
        help=("Comma/newline-separated parts; optional syntax: label=phrase|alias."),
    )
    guided_group.add_argument("--part-prompts-file", type=Path)
    auto_parser.add_argument(
        "--guided-backend",
        choices=("auto", "sam3", "grounded-sam2"),
        default="auto",
        help=(
            "Text segmentation backend. Auto prefers gated SAM3 and records a "
            "fallback to Grounded-SAM2 when SAM3 is unavailable."
        ),
    )
    auto_parser.add_argument("--sam3-model", default="facebook/sam3")
    auto_parser.add_argument("--sam3-score-threshold", type=float, default=0.34)
    auto_parser.add_argument(
        "--root-mode",
        choices=("primary", "scene", "all"),
        default="primary",
        help=(
            "Route one salient asset (primary), canonicalize every physical "
            "scene object before recursive splitting (scene), or retain every "
            "raw detected root for diagnostics only (all)."
        ),
    )
    auto_parser.add_argument("--previous", type=Path)
    auto_parser.add_argument("--seed", type=int, default=20260812)
    auto_parser.add_argument("--box-threshold", type=float, default=0.24)
    auto_parser.add_argument("--text-threshold", type=float, default=0.20)
    auto_parser.add_argument(
        "--grounding-model",
        default="IDEA-Research/grounding-dino-tiny",
        help="Hugging Face open-vocabulary detector used to propose part boxes.",
    )
    auto_parser.add_argument(
        "--additional-grounding-model",
        action="append",
        default=[],
        help=(
            "Additional open-vocabulary detector whose candidates are fused "
            "with the primary detector. May be supplied more than once."
        ),
    )
    auto_parser.add_argument(
        "--segmentation-model",
        default="facebook/sam2.1-hiera-tiny",
        help="Hugging Face SAM2 model used to turn boxes into masks.",
    )
    auto_parser.add_argument(
        "--dense-semantic-model",
        default="CIDAS/clipseg-rd64-refined",
        help=(
            "CLIPSeg-compatible conditional mask model used for dense part "
            "evidence. A reviewed HPID fine-tune may be supplied."
        ),
    )
    auto_parser.add_argument(
        "--dense-semantic-fallback",
        action="store_true",
        help="Use CLIPSeg heatmaps to recover detail IDs missed by box grounding.",
    )
    auto_parser.add_argument(
        "--semantic-part-multimask",
        action="store_true",
        help=(
            "Select among SAM2 part-mask alternatives using independent "
            "region-text evidence while retaining SAM quality guards."
        ),
    )
    auto_parser.add_argument(
        "--maximum-roots-per-domain",
        type=int,
        default=None,
        help="Optional runtime cap for retained root candidates in each domain.",
    )
    auto_parser.add_argument(
        "--maximum-total-roots",
        type=int,
        default=None,
        help="Optional runtime cap for retained root candidates across all domains.",
    )
    auto_parser.add_argument(
        "--no-scene-profile-root-queries",
        action="store_true",
        help=(
            "Use broad domain inventories only during scene discovery. This "
            "reduces detector calls; Ensemble mode should keep profile queries."
        ),
    )
    auto_parser.add_argument(
        "--proposal-first-fast",
        action="store_true",
        help=(
            "Reuse one label-free SAM2 proposal pool for root discovery and "
            "part decomposition; Grounding DINO remains available to other "
            "modes but is not loaded on this low-latency path."
        ),
    )
    auto_parser.add_argument(
        "--no-isolated-profile-resolution",
        action="store_true",
        help=(
            "Keep routed broad roots without category-by-category detector "
            "queries. Intended for low-latency scene decomposition."
        ),
    )
    auto_parser.add_argument(
        "--no-root-domain-arbitration",
        action="store_true",
        help=(
            "Disable image-text arbitration of broad object domains. Intended "
            "for ablation only; automatic product runs keep it enabled."
        ),
    )
    auto_parser.add_argument(
        "--no-relational-appearance",
        action="store_true",
        help=(
            "Disable the image-only second pass that derives configured detail "
            "candidates from stable anchor parts."
        ),
    )
    auto_parser.add_argument(
        "--no-mask-refinement",
        action="store_true",
        help="Disable narrow-band image-guided boundary refinement and cleanup.",
    )
    auto_parser.add_argument(
        "--grabcut-iterations",
        type=int,
        default=3,
        help="GrabCut iterations per scheduled candidate (default: 3).",
    )
    auto_parser.add_argument(
        "--maximum-grabcut-candidates",
        type=int,
        default=None,
        help=(
            "Maximum candidates receiving GrabCut; all candidates still receive "
            "lightweight morphology and component cleanup."
        ),
    )
    auto_parser.add_argument(
        "--no-visual-regions",
        action="store_true",
        help="Disable label-free SAM2 point-grid part proposals.",
    )
    auto_parser.add_argument(
        "--no-structural-fusion",
        action="store_true",
        help=(
            "Disable profile-constrained axial/residual part fusion. Intended "
            "for an algorithm ablation."
        ),
    )
    auto_parser.add_argument(
        "--no-profile-refinement",
        action="store_true",
        help=(
            "Disable routed category-profile part queries; useful as an "
            "ablation of semantic Part-ID refinement."
        ),
    )
    auto_parser.add_argument(
        "--visual-points-per-crop",
        type=int,
        default=20,
        help="SAM2 point-grid density used for open-set visual part proposals.",
    )
    auto_parser.add_argument(
        "--visual-crop-layers",
        type=int,
        default=0,
        help="Additional multiscale SAM2 crop layers for small visual parts.",
    )
    auto_parser.add_argument(
        "--visual-crop-downscale",
        type=int,
        default=2,
        help="Point-grid downscale factor used on deeper SAM2 crop layers.",
    )
    auto_parser.add_argument("--local-files-only", action="store_true")
    auto_parser.add_argument("--learned-checkpoint", type=Path)
    auto_parser.add_argument("--learned-domain", default="character")
    auto_parser.add_argument("--learned-name-mapping", type=Path)
    auto_parser.add_argument("--learned-evaluation-height", type=int, default=768)
    auto_parser.add_argument(
        "--retrieval-index",
        type=Path,
        help=(
            "Reviewed object/part prototype index used to derive automatic "
            "part queries and priors."
        ),
    )
    auto_parser.add_argument("--retrieval-top-k", type=int, default=5)
    auto_parser.add_argument("--retrieval-maximum-part-prompts", type=int, default=28)
    auto_parser.add_argument(
        "--retrieval-minimum-asset-similarity", type=float, default=0.58
    )
    auto_parser.add_argument(
        "--retrieval-minimum-prompted-asset-similarity",
        type=float,
        default=0.32,
    )
    auto_parser.add_argument(
        "--retrieval-minimum-profiled-asset-similarity",
        type=float,
        default=0.50,
    )
    auto_parser.add_argument(
        "--retrieval-minimum-part-similarity", type=float, default=0.42
    )
    auto_parser.add_argument(
        "--retrieval-profile-similarity-bonus", type=float, default=0.035
    )
    auto_parser.add_argument(
        "--retrieval-allow-domain-relabel",
        action="store_true",
        help=(
            "Allow exceptionally strong prototype evidence to override the "
            "resolved root domain. Disabled by default."
        ),
    )
    auto_parser.add_argument("--retrieval-embedding-batch-size", type=int, default=16)
    auto_parser.add_argument(
        "--asset-router-index",
        type=Path,
        help=(
            "Optional train-only SigLIP 2 category index. High-confidence "
            "routes lock one inventory; ambiguous routes retain a bounded "
            "candidate inventory."
        ),
    )
    auto_parser.add_argument(
        "--asset-router-model",
        default="",
        help="SigLIP 2 model path; defaults to the model recorded by the index.",
    )
    auto_parser.add_argument(
        "--ontology-router-model",
        default="",
        help=(
            "Optional SigLIP 2 model for scene object/profile verification and "
            "repeated-instance consensus. Defaults to the asset-router model."
        ),
    )
    auto_parser.add_argument(
        "--no-ontology-scene-consensus",
        action="store_true",
        help="Disable SigLIP 2 scene ontology verification for ablation.",
    )
    auto_parser.add_argument(
        "--asset-router-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    auto_parser.add_argument("--asset-router-batch-size", type=int, default=16)
    auto_parser.add_argument(
        "--asset-router-prototype-weight", type=float, default=0.20
    )
    auto_parser.add_argument("--asset-router-nearest-weight", type=float, default=0.05)
    auto_parser.add_argument("--asset-router-minimum-score", type=float, default=0.18)
    auto_parser.add_argument("--asset-router-minimum-margin", type=float, default=0.020)
    auto_parser.add_argument(
        "--florence-parts",
        action="store_true",
        help=(
            "Supplement retrieval-guided masks with root-constrained Florence-2 "
            "proposals before the prototype and geometry gates."
        ),
    )
    auto_parser.add_argument(
        "--florence-model",
        default="florence-community/Florence-2-base-ft",
    )
    auto_parser.add_argument(
        "--florence-phrases-per-part",
        type=int,
        default=2,
    )
    auto_parser.add_argument(
        "--vlm-parts",
        action="store_true",
        help=(
            "Use a local ontology-bounded VLM as soft box evidence over "
            "independent SAM2 regions. The VLM never assigns final Part IDs."
        ),
    )
    auto_parser.add_argument(
        "--adaptive-profile-refinement",
        action="store_true",
        help=(
            "When broad profile refinement is disabled, run category-guided "
            "mask discovery only for profiles that explicitly require it."
        ),
    )
    auto_parser.add_argument(
        "--vlm-model",
        default="Qwen/Qwen3-VL-2B-Instruct",
    )
    auto_parser.add_argument("--vlm-load-in-4bit", action="store_true")
    auto_parser.add_argument(
        "--vlm-query-mode",
        choices=("bulk", "per-semantic"),
        default="per-semantic",
    )
    auto_parser.add_argument("--vlm-maximum-queries", type=int, default=12)
    auto_parser.add_argument(
        "--vlm-maximum-total-queries",
        type=int,
        default=24,
        help=(
            "Global VLM request cap across every routed object in one image; "
            "prevents scene cost from growing with object count."
        ),
    )
    auto_parser.add_argument("--vlm-maximum-roots", type=int, default=8)
    auto_parser.add_argument(
        "--vlm-maximum-root-audits",
        type=int,
        default=12,
        help=(
            "Maximum uncertain scene objects audited by the VLM before part "
            "labelling; these requests share the global VLM budget."
        ),
    )
    auto_parser.add_argument(
        "--vlm-maximum-semantic-audits",
        type=int,
        default=4,
        help=(
            "Maximum low-confidence named part masks audited before VLM part "
            "labelling. Rejected labels fall back to anonymous visual IDs; "
            "these requests share the global VLM budget."
        ),
    )
    auto_parser.add_argument(
        "--vlm-maximum-physicality-audits",
        type=int,
        default=4,
        help=(
            "Maximum batched VLM requests used to distinguish physical SAM "
            "regions from texture, lighting, and noise. These requests share "
            "the global VLM budget and never assign final Part IDs."
        ),
    )
    auto_parser.add_argument("--vlm-maximum-new-tokens", type=int, default=384)
    auto_parser.add_argument(
        "--vlm-box-planner",
        action="store_true",
        help=(
            "Also ask the VLM to localize missing parts by boxes. Disabled in "
            "product mode because independent SAM-region evidence is safer."
        ),
    )
    auto_parser.add_argument(
        "--vlm-query-established-semantics",
        action="store_true",
        help="Query already-labelled semantics too; intended for diagnostics.",
    )
    auto_parser.add_argument(
        "--vlm-allow-direct-sam-regions",
        action="store_true",
        help=(
            "Allow a VLM box plus SAM2 mask without an independent visual "
            "region. Disabled in product mode because it weakens corroboration."
        ),
    )
    auto_parser.add_argument(
        "--vlm-dynamic-inventory",
        action="store_true",
        help=(
            "Let the VLM propose missing visible physical component types. "
            "Every dynamic label still requires an independently audited "
            "SAM2 region before it can become a Part-ID candidate."
        ),
    )
    completion_group = auto_parser.add_mutually_exclusive_group()
    completion_group.add_argument("--completion-config", type=Path)
    completion_group.add_argument("--geometric-fallback", action="store_true")
    auto_parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    auto_parser.set_defaults(handler=command_auto)

    taxonomy_parser = subparsers.add_parser(
        "validate-taxonomy", help="Validate a taxonomy JSON file."
    )
    taxonomy_parser.add_argument("--taxonomy", type=Path, required=True)
    taxonomy_parser.set_defaults(handler=command_validate_taxonomy)

    package_parser = subparsers.add_parser(
        "validate-package", help="Audit an exported HPID package."
    )
    package_parser.add_argument("--package", type=Path, required=True)
    package_parser.set_defaults(handler=command_validate_package)

    setup_parser = subparsers.add_parser(
        "setup-completion", help="Install LaMa and write a local completion config."
    )
    setup_parser.add_argument("--package-root", type=Path)
    setup_parser.add_argument("--model-cache", type=Path)
    setup_parser.add_argument("--config", type=Path)
    setup_parser.add_argument("--python", default=sys.executable)
    setup_parser.add_argument("--skip-install", action="store_true")
    setup_parser.set_defaults(handler=command_setup_completion)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
