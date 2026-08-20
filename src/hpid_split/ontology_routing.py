from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from .asset_routing import ImageTextEncoder, masked_asset_view
from .fusion import MaskCandidate
from .prompt_bank import DomainPrompt, PartProfile, PromptBank
from .root_routing import candidate_root_key


@dataclass(frozen=True)
class OntologyRoutingConfig:
    minimum_profile_margin: float = 0.015
    minimum_profile_override_gain: float = 0.015
    minimum_profile_rejection_gain: float = 0.012
    minimum_global_profile_margin: float = 0.015
    minimum_cross_domain_profile_score: float = 0.09
    minimum_cross_domain_profile_margin: float = 0.025
    minimum_cross_domain_profile_gain: float = 0.020
    minimum_domain_margin: float = 0.025
    minimum_domain_override_gain: float = 0.025
    minimum_scene_similarity: float = 0.82
    minimum_scene_mean_similarity: float = 0.84
    minimum_scene_anchor_count: int = 2
    minimum_scene_support_ratio: float = 0.62
    maximum_scene_anchors_per_label: int = 3
    scene_consensus_temperature: float = 0.03


@dataclass(frozen=True)
class OntologyRoutingResult:
    roots: tuple[MaskCandidate, ...]
    diagnostics: dict[str, object]


def _profile_name(root: MaskCandidate) -> str | None:
    value = root.metadata.get("selected_part_profile")
    return str(value) if value is not None and str(value).strip() else None


def _profile_prompt(profile: PartProfile) -> str:
    return profile.classifier_prompt or " or ".join(profile.root_hints)


def _domain_prompt(domain: DomainPrompt) -> str:
    return domain.classifier_prompt or f"one complete {domain.name.replace('_', ' ')}"


def _rank(values: np.ndarray) -> tuple[int, int | None, float]:
    ordering = np.argsort(-values)
    winner = int(ordering[0])
    runner_up = int(ordering[1]) if len(ordering) > 1 else None
    margin = float(values[winner] - values[runner_up]) if runner_up is not None else 1.0
    return winner, runner_up, margin


def _resolved_profile_root(
    root: MaskCandidate,
    domain: DomainPrompt,
    profile: PartProfile | None,
    *,
    source: str,
    routing_metadata: dict[str, object],
) -> MaskCandidate:
    metadata = {
        **root.metadata,
        **routing_metadata,
        "ontology_original_domain": root.semantic_name,
        "ontology_original_profile": _profile_name(root),
        "ground_truth_used": False,
    }
    if profile is None:
        for field in (
            "selected_part_profile",
            "resolved_object_label",
            "profile_evidence_root_key",
            "profile_evidence_score",
            "profile_evidence_margin",
        ):
            metadata.pop(field, None)
        metadata.update(
            {
                "part_profile_specificity": 0.0,
                "profile_resolution_status": "unresolved",
                "profile_hint_source": source,
            }
        )
        prompt = domain.name.replace("_", " ")
    else:
        object_label = profile.root_hints[0]
        _, _, profile_diagnostics = domain.select_parts(
            object_label,
            profile_hint=profile.name,
            profile_hint_source=source,
        )
        metadata.update(
            {
                "selected_part_profile": profile.name,
                "resolved_object_label": object_label,
                "part_profile_specificity": 1.0,
                "part_profile_selection": profile_diagnostics,
                "profile_resolution_status": "accepted",
                "profile_hint_source": source,
            }
        )
        prompt = object_label
    return MaskCandidate(
        semantic_name=domain.name,
        semantic_parent=domain.name,
        mask=root.mask,
        score=root.score,
        source=root.source,
        prompt=prompt,
        source_reliability=root.source_reliability,
        metadata=metadata,
    )


def route_scene_ontology(
    image: Image.Image,
    roots: list[MaskCandidate],
    prompt_bank: PromptBank,
    encoder: ImageTextEncoder,
    *,
    config: OntologyRoutingConfig | None = None,
) -> OntologyRoutingResult:
    """Resolve scene object types with visual semantics and repeated-instance consensus."""

    config = config or OntologyRoutingConfig()
    roots = [root for root in roots if candidate_root_key(root) is not None]
    if not roots:
        return OntologyRoutingResult(
            (),
            {
                "algorithm": "hpid-ontology-scene-consensus-v1",
                "root_count": 0,
                "ground_truth_used": False,
            },
        )
    domains = list(prompt_bank.domains)
    domains_by_name = {domain.name: domain for domain in domains}
    profiles: list[tuple[DomainPrompt, PartProfile]] = [
        (domain, profile) for domain in domains for profile in domain.part_profiles
    ]
    views = [masked_asset_view(image, root.mask.astype(bool)) for root in roots]
    image_embeddings = encoder.encode_images(views)
    domain_embeddings = encoder.encode_texts(
        [f"a clear isolated image of {_domain_prompt(domain)}" for domain in domains]
    )
    profile_embeddings = (
        encoder.encode_texts(
            [
                f"a clear isolated image of {_profile_prompt(profile)}"
                for _, profile in profiles
            ]
        )
        if profiles
        else np.zeros((0, image_embeddings.shape[1]), dtype=np.float32)
    )
    domain_scores = image_embeddings @ domain_embeddings.T
    profile_scores = image_embeddings @ profile_embeddings.T
    profile_indices_by_domain = {
        domain.name: [
            index
            for index, (owner, _) in enumerate(profiles)
            if owner.name == domain.name
        ]
        for domain in domains
    }

    provisional: list[dict[str, object]] = []
    for root_index, root in enumerate(roots):
        current_domain = domains_by_name.get(root.semantic_name)
        if current_domain is None:
            continue
        domain_winner_index, _, domain_margin = _rank(domain_scores[root_index])
        domain_winner = domains[domain_winner_index]
        current_domain_index = domains.index(current_domain)
        domain_gain = float(
            domain_scores[root_index, domain_winner_index]
            - domain_scores[root_index, current_domain_index]
        )
        selected_domain = current_domain
        domain_decision = "preserved_detector_domain"
        if (
            domain_winner.name != current_domain.name
            and domain_margin >= config.minimum_domain_margin
            and domain_gain >= config.minimum_domain_override_gain
        ):
            selected_domain = domain_winner
            domain_decision = "visual_domain_override"

        selected_indices = profile_indices_by_domain.get(selected_domain.name, [])
        current_profile = (
            _profile_name(root) if selected_domain.name == current_domain.name else None
        )
        detector_label = str(root.metadata.get("root_model_label") or root.prompt or "")
        detector_profile = selected_domain.select_parts(detector_label)[1]
        selected_profile: PartProfile | None = None
        visual_profile: PartProfile | None = None
        profile_margin = 0.0
        profile_gain = 0.0
        profile_decision = "domain_has_no_profiles"
        if selected_indices:
            local_scores = profile_scores[root_index, selected_indices]
            local_winner, _, profile_margin = _rank(local_scores)
            visual_index = selected_indices[local_winner]
            visual_profile = profiles[visual_index][1]
            profile_decision = "preserved_detector_profile"
            current_score = -1.0
            if current_profile is not None:
                current_index = next(
                    (
                        index
                        for index in selected_indices
                        if profiles[index][1].name == current_profile
                    ),
                    None,
                )
                if current_index is not None:
                    current_score = float(profile_scores[root_index, current_index])
            visual_score = float(profile_scores[root_index, visual_index])
            profile_gain = visual_score - current_score
            if current_profile == visual_profile.name:
                selected_profile = visual_profile
                profile_decision = "visual_profile_confirmed"
            elif (
                profile_margin >= config.minimum_profile_margin
                and profile_gain >= config.minimum_profile_override_gain
            ):
                selected_profile = visual_profile
                profile_decision = "visual_profile_override"
            elif current_profile is None:
                selected_profile = None
                profile_decision = "ambiguous_profile"
            elif profile_gain >= config.minimum_profile_rejection_gain:
                selected_profile = None
                profile_decision = "conflicting_profile_rejected"
            else:
                selected_profile = next(
                    (
                        profile
                        for owner, profile in profiles
                        if owner.name == selected_domain.name
                        and profile.name == current_profile
                    ),
                    None,
                )

        if profiles:
            global_winner_index, _, global_margin = _rank(profile_scores[root_index])
            global_domain, global_profile = profiles[global_winner_index]
            global_profile_score = float(
                profile_scores[root_index, global_winner_index]
            )
        else:
            global_margin = 0.0
            global_domain, global_profile = selected_domain, None
            global_profile_score = -1.0
        selected_domain_profile_score = (
            float(np.max(profile_scores[root_index, selected_indices]))
            if selected_indices
            else -1.0
        )
        global_profile_gain = global_profile_score - selected_domain_profile_score
        if (
            global_profile is not None
            and global_domain.name != selected_domain.name
            and global_profile_score >= config.minimum_cross_domain_profile_score
            and global_margin >= config.minimum_cross_domain_profile_margin
            and global_profile_gain >= config.minimum_cross_domain_profile_gain
        ):
            selected_domain = global_domain
            selected_profile = global_profile
            visual_profile = global_profile
            profile_margin = float(global_margin)
            profile_gain = float(global_profile_gain)
            domain_decision = "global_profile_domain_override"
            profile_decision = "global_profile_override"
            detector_profile = selected_domain.select_parts(detector_label)[1]
        detector_agreement = bool(
            selected_profile is not None
            and detector_profile == selected_profile.name
            and visual_profile is not None
            and visual_profile.name == selected_profile.name
        )
        global_visual_agreement = bool(
            selected_profile is not None
            and global_profile is not None
            and global_domain.name == selected_domain.name
            and global_profile.name == selected_profile.name
            and profile_margin >= config.minimum_profile_margin
            and global_margin >= config.minimum_global_profile_margin
        )
        # Detector/profile agreement is not independent evidence: both labels can
        # originate from the same open-vocabulary query. Only a profile that also
        # wins the global visual comparison may propagate through scene consensus.
        anchor = global_visual_agreement
        provisional.append(
            {
                "root": root,
                "root_index": root_index,
                "domain": selected_domain,
                "profile": selected_profile,
                "anchor": anchor,
                "domain_decision": domain_decision,
                "profile_decision": profile_decision,
                "domain_winner": domain_winner.name,
                "domain_margin": float(domain_margin),
                "domain_gain": domain_gain,
                "visual_profile": (
                    visual_profile.name if visual_profile is not None else None
                ),
                "profile_margin": float(profile_margin),
                "profile_gain": float(profile_gain),
                "global_profile_domain": global_domain.name,
                "global_profile": (
                    global_profile.name if global_profile is not None else None
                ),
                "global_profile_margin": float(global_margin),
                "global_profile_score": global_profile_score,
                "global_profile_gain": global_profile_gain,
                "detector_agreement": detector_agreement,
                "global_visual_agreement": global_visual_agreement,
            }
        )

    anchors = [
        row for row in provisional if bool(row["anchor"]) and row["profile"] is not None
    ]
    consensus_count = 0
    output: list[MaskCandidate] = []
    diagnostic_rows: list[dict[str, object]] = []
    for row in provisional:
        root_index = int(row["root_index"])
        consensus: tuple[str, str] | None = None
        consensus_rows: list[tuple[float, dict[str, object]]] = []
        consensus_group_rows: list[dict[str, object]] = []
        if not bool(row["anchor"]):
            consensus_rows = sorted(
                (
                    (
                        float(
                            image_embeddings[root_index]
                            @ image_embeddings[int(anchor["root_index"])]
                        ),
                        anchor,
                    )
                    for anchor in anchors
                    if int(anchor["root_index"]) != root_index
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            consensus_rows = [
                item
                for item in consensus_rows
                if item[0] >= config.minimum_scene_similarity
            ]
            grouped: dict[tuple[str, str], list[float]] = {}
            for similarity, anchor in consensus_rows:
                anchor_domain = anchor["domain"]
                anchor_profile = anchor["profile"]
                assert isinstance(anchor_domain, DomainPrompt)
                assert isinstance(anchor_profile, PartProfile)
                grouped.setdefault(
                    (anchor_domain.name, anchor_profile.name), []
                ).append(similarity)
            ranked_groups = sorted(
                (
                    (
                        label,
                        similarities,
                        float(
                            np.mean(
                                sorted(similarities, reverse=True)[
                                    : config.maximum_scene_anchors_per_label
                                ]
                            )
                        ),
                    )
                    for label, similarities in grouped.items()
                ),
                key=lambda item: (item[2], len(item[1]), max(item[1])),
                reverse=True,
            )
            if ranked_groups:
                label, similarities, mean_similarity = ranked_groups[0]
                scores = np.asarray(
                    [item[2] for item in ranked_groups], dtype=np.float64
                )
                temperature = max(1e-6, config.scene_consensus_temperature)
                weights = np.exp((scores - float(np.max(scores))) / temperature)
                supports = weights / max(1e-8, float(np.sum(weights)))
                support_ratio = float(supports[0])
                consensus_group_rows = [
                    {
                        "domain": item_label[0],
                        "profile": item_label[1],
                        "anchor_count": len(item_similarities),
                        "mean_top_similarity": item_mean,
                        "support": float(support),
                    }
                    for (
                        item_label,
                        item_similarities,
                        item_mean,
                    ), support in zip(ranked_groups, supports, strict=True)
                ]
                if (
                    len(similarities) >= config.minimum_scene_anchor_count
                    and mean_similarity >= config.minimum_scene_mean_similarity
                    and support_ratio >= config.minimum_scene_support_ratio
                ):
                    consensus = label

        domain = row["domain"]
        profile = row["profile"]
        assert isinstance(domain, DomainPrompt)
        assert profile is None or isinstance(profile, PartProfile)
        decision = str(row["profile_decision"])
        if consensus is not None:
            consensus_domain, consensus_profile = consensus
            domain = domains_by_name[consensus_domain]
            profile = next(
                item for item in domain.part_profiles if item.name == consensus_profile
            )
            decision = "scene_repeated_instance_consensus"
            consensus_count += 1
        root = row["root"]
        assert isinstance(root, MaskCandidate)
        routed_root = _resolved_profile_root(
            root,
            domain,
            profile,
            source="siglip2_ontology_scene_consensus",
            routing_metadata={
                "ontology_routing_algorithm": "hpid-ontology-scene-consensus-v1",
                "ontology_domain_decision": row["domain_decision"],
                "ontology_profile_decision": decision,
                "ontology_visual_domain_winner": row["domain_winner"],
                "ontology_visual_domain_margin": row["domain_margin"],
                "ontology_visual_domain_gain": row["domain_gain"],
                "ontology_visual_profile_winner": row["visual_profile"],
                "ontology_visual_profile_margin": row["profile_margin"],
                "ontology_visual_profile_gain": row["profile_gain"],
                "ontology_global_profile_domain": row["global_profile_domain"],
                "ontology_global_profile_winner": row["global_profile"],
                "ontology_global_profile_margin": row["global_profile_margin"],
                "ontology_global_profile_score": row["global_profile_score"],
                "ontology_global_profile_gain": row["global_profile_gain"],
                "ontology_detector_agreement": bool(row["detector_agreement"]),
                "ontology_global_visual_agreement": bool(
                    row["global_visual_agreement"]
                ),
                "ontology_anchor": bool(row["anchor"]),
                "ontology_consensus_label": (
                    list(consensus) if consensus is not None else None
                ),
            },
        )
        output.append(routed_root)
        diagnostic_rows.append(
            {
                "root_key": candidate_root_key(root),
                "original_domain": root.semantic_name,
                "original_profile": _profile_name(root),
                "resolved_domain": domain.name,
                "resolved_profile": profile.name if profile is not None else None,
                "domain_decision": row["domain_decision"],
                "profile_decision": decision,
                "anchor": bool(row["anchor"]),
                "detector_agreement": bool(row["detector_agreement"]),
                "global_visual_agreement": bool(row["global_visual_agreement"]),
                "consensus_label": (list(consensus) if consensus is not None else None),
                "consensus_groups": consensus_group_rows,
                "nearest_anchor_similarities": [
                    {
                        "similarity": similarity,
                        "domain": str(anchor["domain"].name),
                        "profile": str(anchor["profile"].name),
                    }
                    for similarity, anchor in consensus_rows[:5]
                ],
            }
        )
    return OntologyRoutingResult(
        tuple(output),
        {
            "algorithm": "hpid-ontology-scene-consensus-v1",
            "encoder_model": encoder.model_name,
            "root_count": len(roots),
            "anchor_count": len(anchors),
            "consensus_resolution_count": consensus_count,
            "rows": diagnostic_rows,
            "ground_truth_used": False,
        },
    )
