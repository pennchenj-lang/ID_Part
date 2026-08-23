from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from .fusion import MaskCandidate, mask_iou
from .prompt_bank import DomainPrompt


@dataclass(frozen=True)
class ProfileResolutionConfig:
    minimum_root_affinity: float = 0.12
    minimum_consensus_score: float = 0.30
    minimum_detector_only_margin: float = 0.06
    minimum_classifier_similarity_margin: float = 0.006
    minimum_classifier_margin_when_detector_disagrees: float = 0.014
    minimum_classifier_probability_ratio: float = 1.20
    minimum_classifier_uniform_multiplier: float = 1.30
    detector_override_margin: float = 0.10
    detector_only_minimum_margin: float = 0.16
    detector_only_maximum_classifier_rank: int = 3
    full_image_area_threshold: float = 0.88
    minimum_isolated_classifier_probability_ratio: float = 1.35
    minimum_isolated_classifier_uniform_multiplier: float = 1.40
    minimum_isolated_detector_score: float = 0.24
    minimum_specific_root_profile_score: float = 0.80


@dataclass(frozen=True)
class ProfileResolutionResult:
    roots: tuple[MaskCandidate, ...]
    diagnostics: dict[str, object]


def _root_key(candidate: MaskCandidate) -> str:
    return (
        f"{candidate.metadata.get('root_origin', 'legacy')}::"
        f"{candidate.metadata.get('root_index', 'unknown')}"
    )


def _profile_name(candidate: MaskCandidate) -> str | None:
    value = candidate.metadata.get("selected_part_profile")
    return str(value) if value is not None and str(value).strip() else None


def _trusted_global_asset_profile(candidate: MaskCandidate) -> bool:
    """Return whether whole-image routing and the selected root agree exactly."""

    return bool(
        candidate.metadata.get("global_asset_proposal_accepted", False)
        and candidate.metadata.get("global_asset_proposal_rank") == 1
        and candidate.metadata.get("root_query_mode") == "global_asset_proposal"
    )


def _profile_hint_conflict(
    candidate: MaskCandidate,
    profile: str,
    domain: DomainPrompt | None,
) -> bool:
    if domain is None:
        return False
    model_label = str(candidate.metadata.get("root_model_label") or "")
    if not model_label:
        return False
    resolved = domain.select_parts(model_label)[1]
    return resolved is not None and resolved != profile


def _affinity(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    if not intersection:
        return 0.0
    first_area = max(1, int(np.count_nonzero(first)))
    second_area = max(1, int(np.count_nonzero(second)))
    containment = max(intersection / first_area, intersection / second_area)
    return float(max(mask_iou(first, second), 0.85 * containment))


def _isolated_root_support(
    candidate: MaskCandidate,
    config: ProfileResolutionConfig,
) -> bool:
    if not candidate.metadata.get("profile_candidate_self_classified"):
        return False
    classifier = dict(candidate.metadata.get("profile_self_classifier", {}))
    inventory_count = max(
        1, int(candidate.metadata.get("profile_classifier_inventory_count", 1))
    )
    probability = float(classifier.get("probability", 0.0))
    return bool(
        int(classifier.get("rank", 999)) == 1
        and probability
        >= config.minimum_isolated_classifier_uniform_multiplier / inventory_count
        and float(candidate.metadata.get("profile_classifier_probability_ratio", 0.0))
        >= config.minimum_isolated_classifier_probability_ratio
        and float(candidate.metadata.get("profile_detector_score", candidate.score))
        >= config.minimum_isolated_detector_score
    )


def _geometry_metrics(
    candidate: MaskCandidate,
    image_shape: tuple[int, int],
    config: ProfileResolutionConfig,
) -> dict[str, float | int | list[int]]:
    height, width = image_shape
    ys, xs = np.nonzero(candidate.mask)
    if not len(xs):
        return {
            "score": 0.0,
            "area_fraction": 0.0,
            "bbox_fraction": 0.0,
            "coherence": 0.0,
            "border_sides": 0,
            "bbox_xyxy": [0, 0, 0, 0],
        }
    area = len(xs)
    image_area = max(1, height * width)
    x0, y0, x1, y1 = (
        int(xs.min()),
        int(ys.min()),
        int(xs.max() + 1),
        int(ys.max() + 1),
    )
    area_fraction = area / image_area
    bbox_fraction = (x1 - x0) * (y1 - y0) / image_area
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        candidate.mask.astype(np.uint8), 8
    )
    coherence = (
        float(stats[1:, cv2.CC_STAT_AREA].max() / area) if count > 1 and area else 0.0
    )
    border_sides = sum((x0 == 0, y0 == 0, x1 == width, y1 == height))
    detector = float(np.clip(candidate.score / 0.65, 0.0, 1.0))
    sam = float(np.clip(candidate.metadata.get("sam_quality", 0.5), 0.0, 1.0))
    area_salience = float(np.clip(np.sqrt(area_fraction / 0.20), 0.0, 1.0))
    bbox_salience = float(np.clip(np.sqrt(bbox_fraction / 0.20), 0.0, 1.0))
    oversize_penalty = float(
        0.34
        * np.clip(
            (area_fraction - config.full_image_area_threshold)
            / max(1e-6, 1.0 - config.full_image_area_threshold),
            0.0,
            1.0,
        )
    )
    border_penalty = 0.035 * border_sides
    score = (
        0.30 * bbox_salience
        + 0.23 * area_salience
        + 0.18 * sam
        + 0.14 * detector
        + 0.15 * coherence
        - oversize_penalty
        - border_penalty
    )
    return {
        "score": float(score),
        "area_fraction": float(area_fraction),
        "bbox_fraction": float(bbox_fraction),
        "coherence": coherence,
        "border_sides": int(border_sides),
        "bbox_xyxy": [x0, y0, x1, y1],
    }


def _clear_untrusted_profile(root: MaskCandidate) -> MaskCandidate:
    metadata = dict(root.metadata)
    for key in (
        "selected_part_profile",
        "profile_hint_source",
        "profile_evidence_root_key",
        "profile_evidence_score",
        "profile_evidence_margin",
    ):
        metadata.pop(key, None)
    metadata["part_profile_specificity"] = 0.0
    metadata["profile_resolution_status"] = "unresolved"
    return MaskCandidate(
        semantic_name=root.semantic_name,
        semantic_parent=root.semantic_parent,
        mask=root.mask,
        score=root.score,
        source=root.source,
        prompt=root.semantic_name.replace("_", " "),
        source_reliability=root.source_reliability,
        metadata=metadata,
    )


def resolve_profile_roots(
    broad_roots: list[MaskCandidate],
    profile_candidates: list[MaskCandidate],
    *,
    image_shape: tuple[int, int],
    domains: dict[str, DomainPrompt] | None = None,
    config: ProfileResolutionConfig | None = None,
) -> ProfileResolutionResult:
    """Resolve category profile and geometry without using target annotations."""

    config = config or ProfileResolutionConfig()
    resolved: list[MaskCandidate] = []
    diagnostics: list[dict[str, object]] = []
    allow_isolated_replacement = len(broad_roots) == 1
    for broad_root in broad_roots:
        domain = (domains or {}).get(broad_root.semantic_name)
        broad_profile = _profile_name(broad_root)
        broad_profile_score = float(
            broad_root.metadata.get("part_profile_specificity", 0.0)
        )
        if domain is not None:
            _, inferred_profile, inference = domain.select_parts(
                str(broad_root.metadata.get("root_model_label") or broad_root.prompt)
            )
            inferred_score = float(inference.get("best_score", 0.0))
            if inferred_profile is not None and inferred_score > broad_profile_score:
                broad_profile = inferred_profile
                broad_profile_score = inferred_score
        profile_source = str(broad_root.metadata.get("profile_hint_source") or "")
        trusted_global_profile = _trusted_global_asset_profile(broad_root)
        if (
            broad_profile is not None
            and broad_profile_score >= config.minimum_specific_root_profile_score
            and (
                profile_source != "isolated_profile_query" or trusted_global_profile
            )
        ):
            metadata = {
                **broad_root.metadata,
                "selected_part_profile": broad_profile,
                "part_profile_specificity": broad_profile_score,
                "profile_resolution_status": "accepted",
                "profile_hint_source": (
                    "accepted_global_asset_profile"
                    if trusted_global_profile
                    else "specific_root_label"
                ),
                "ground_truth_used": False,
            }
            resolved.append(replace(broad_root, metadata=metadata))
            diagnostics.append(
                {
                    "root_key": _root_key(broad_root),
                    "root_semantic": broad_root.semantic_name,
                    "status": "accepted",
                    "selected_profile": broad_profile,
                    "resolution_source": (
                        "accepted_global_asset_profile"
                        if trusted_global_profile
                        else "specific_root_label"
                    ),
                    "root_profile_specificity": broad_profile_score,
                    "root_label_profile": broad_profile,
                    "root_label_support": True,
                    "ranking": [],
                }
            )
            continue
        evidence_rows: list[dict[str, object]] = []
        for evidence in profile_candidates:
            profile = _profile_name(evidence)
            if profile is None or evidence.semantic_name != broad_root.semantic_name:
                continue
            if _profile_hint_conflict(evidence, profile, domain):
                continue
            affinity = _affinity(broad_root.mask, evidence.mask)
            isolated_root_support = bool(
                allow_isolated_replacement
                and affinity < config.minimum_root_affinity
                and _isolated_root_support(evidence, config)
            )
            if affinity < config.minimum_root_affinity and not isolated_root_support:
                continue
            classifier = dict(evidence.metadata.get("profile_classifier", {}))
            evidence_rows.append(
                {
                    "candidate": evidence,
                    "profile": profile,
                    "affinity": affinity,
                    "consensus_score": float(
                        evidence.metadata.get("profile_consensus_score", evidence.score)
                    ),
                    "detector_score": float(
                        evidence.metadata.get("profile_detector_score", evidence.score)
                    ),
                    "classifier_rank": int(classifier.get("rank", 999)),
                    "classifier_probability": float(classifier.get("probability", 0.0)),
                    "classifier_similarity": float(
                        classifier.get("combined_similarity", 0.0)
                    ),
                    "classifier_inventory_count": int(
                        evidence.metadata.get("profile_classifier_inventory_count", 1)
                    ),
                    "classifier_probability_ratio": float(
                        evidence.metadata.get(
                            "profile_classifier_probability_ratio", 0.0
                        )
                    ),
                    "classifier_similarity_margin": float(
                        evidence.metadata.get(
                            "profile_classifier_similarity_margin", 0.0
                        )
                    ),
                    "isolated_root_replacement": isolated_root_support,
                }
            )

        by_profile: dict[str, dict[str, object]] = {}
        for row in evidence_rows:
            previous = by_profile.get(str(row["profile"]))
            if previous is None or float(row["consensus_score"]) > float(
                previous["consensus_score"]
            ):
                by_profile[str(row["profile"])] = row
        ranking = sorted(
            by_profile.values(),
            key=lambda row: (
                float(row["consensus_score"]),
                float(row["affinity"]),
            ),
            reverse=True,
        )
        winner = ranking[0] if ranking else None
        runner_up_score = (
            float(ranking[1]["consensus_score"]) if len(ranking) > 1 else 0.0
        )
        margin = (
            float(winner["consensus_score"]) - runner_up_score
            if winner is not None
            else 0.0
        )
        classifier_ranking = sorted(
            ranking,
            key=lambda row: float(row["classifier_similarity"]),
            reverse=True,
        )
        classifier_margin = (
            float(classifier_ranking[0]["classifier_similarity"])
            - float(classifier_ranking[1]["classifier_similarity"])
            if len(classifier_ranking) > 1
            else 1.0
        )
        classifier_probability_runner_up = (
            float(classifier_ranking[1]["classifier_probability"])
            if len(classifier_ranking) > 1
            else 0.0
        )
        aggregate_classifier_probability_ratio = (
            float(winner["classifier_probability"])
            / max(1e-8, classifier_probability_runner_up)
            if winner is not None
            else 0.0
        )
        classifier_probability_ratio = (
            float(winner["classifier_probability_ratio"])
            if winner is not None
            and float(winner["classifier_probability_ratio"]) > 0.0
            else aggregate_classifier_probability_ratio
        )
        winner_classifier_margin = (
            float(winner["classifier_similarity_margin"])
            if winner is not None and bool(winner["isolated_root_replacement"])
            else classifier_margin
        )
        classifier_uniform_probability = 1.0 / max(
            1,
            int(winner["classifier_inventory_count"]) if winner is not None else 1,
        )
        classifier_support = bool(
            winner is not None
            and int(winner["classifier_rank"]) == 1
            and float(winner["classifier_probability"])
            >= (
                config.minimum_classifier_uniform_multiplier
                * classifier_uniform_probability
            )
            and classifier_probability_ratio
            >= config.minimum_classifier_probability_ratio
        )
        detector_ranking = sorted(
            ranking,
            key=lambda row: float(row["detector_score"]),
            reverse=True,
        )
        detector_margin = (
            float(detector_ranking[0]["detector_score"])
            - float(detector_ranking[1]["detector_score"])
            if len(detector_ranking) > 1
            else 1.0
        )
        classifier_contradicted_by_detector = bool(
            winner is not None
            and classifier_support
            and detector_ranking
            and detector_ranking[0]["profile"] != winner["profile"]
            and detector_margin >= config.detector_override_margin
            and winner_classifier_margin
            < config.minimum_classifier_margin_when_detector_disagrees
        )
        detector_only_support = bool(
            winner is not None
            and detector_ranking
            and detector_ranking[0]["profile"] == winner["profile"]
            and detector_margin >= config.detector_only_minimum_margin
            and int(winner["classifier_rank"])
            <= config.detector_only_maximum_classifier_rank
        )
        label_profile: str | None = None
        if domain is not None:
            label_profile = domain.select_parts(
                str(broad_root.metadata.get("root_model_label") or broad_root.prompt)
            )[1]
        root_label_support = bool(
            winner is not None
            and label_profile == winner["profile"]
            and margin >= config.minimum_detector_only_margin
        )
        accepted = bool(
            winner is not None
            and float(winner["consensus_score"]) >= config.minimum_consensus_score
            and (
                (
                    classifier_support
                    and winner_classifier_margin
                    >= config.minimum_classifier_similarity_margin
                    and not classifier_contradicted_by_detector
                )
                or (
                    margin >= config.minimum_detector_only_margin
                    and detector_only_support
                )
                or root_label_support
            )
        )
        if not accepted or winner is None:
            unresolved = _clear_untrusted_profile(broad_root)
            resolved.append(unresolved)
            diagnostics.append(
                {
                    "root_key": _root_key(broad_root),
                    "root_semantic": broad_root.semantic_name,
                    "status": "unresolved",
                    "selected_profile": None,
                    "margin": margin,
                    "classifier_support": classifier_support,
                    "classifier_margin": classifier_margin,
                    "winner_classifier_margin": winner_classifier_margin,
                    "classifier_probability_ratio": classifier_probability_ratio,
                    "detector_margin": detector_margin,
                    "classifier_contradicted_by_detector": (
                        classifier_contradicted_by_detector
                    ),
                    "detector_only_support": detector_only_support,
                    "root_label_profile": label_profile,
                    "root_label_support": root_label_support,
                    "ranking": [
                        {key: value for key, value in row.items() if key != "candidate"}
                        for row in ranking
                    ],
                }
            )
            continue

        profile = str(winner["profile"])
        profile_geometry = [
            row["candidate"] for row in evidence_rows if row["profile"] == profile
        ]
        winner_isolated_replacement = bool(winner["isolated_root_replacement"])
        geometry_pool = (
            [
                row["candidate"]
                for row in evidence_rows
                if row["profile"] == profile and bool(row["isolated_root_replacement"])
            ]
            if winner_isolated_replacement
            else [broad_root, *profile_geometry]
        )
        geometry_rows = [
            {
                "candidate": candidate,
                "metrics": _geometry_metrics(candidate, image_shape, config),
            }
            for candidate in geometry_pool
        ]
        geometry_row = max(
            geometry_rows,
            key=lambda row: float(dict(row["metrics"])["score"]),
        )
        geometry = geometry_row["candidate"]
        assert isinstance(geometry, MaskCandidate)
        evidence = winner["candidate"]
        assert isinstance(evidence, MaskCandidate)
        resolved_object_label = str(
            evidence.metadata.get("resolved_object_label")
            or evidence.metadata.get("root_model_label")
            or evidence.prompt
            or broad_root.metadata.get("root_model_label")
            or broad_root.prompt
        )
        _, _, resolved_profile_diagnostics = (
            domain.select_parts(
                resolved_object_label,
                profile_hint=profile,
                profile_hint_source="isolated_profile_consensus",
            )
            if domain is not None
            else ((), None, {})
        )
        metadata = {
            **broad_root.metadata,
            "sam_quality": geometry.metadata.get("sam_quality", 0.5),
            "box_xyxy": dict(geometry_row["metrics"])["bbox_xyxy"],
            "selected_part_profile": profile,
            "resolved_object_label": resolved_object_label,
            "part_profile_specificity": 1.0,
            "part_profile_selection": resolved_profile_diagnostics
            or evidence.metadata.get("part_profile_selection"),
            "profile_hint_source": "isolated_profile_consensus",
            "profile_resolution_status": "accepted",
            "profile_evidence_root_key": _root_key(evidence),
            "profile_evidence_score": float(winner["consensus_score"]),
            "profile_evidence_margin": margin,
            "profile_classifier": evidence.metadata.get("profile_classifier"),
            "profile_geometry_source": _root_key(geometry),
            "profile_geometry_metrics": geometry_row["metrics"],
            "ground_truth_used": False,
        }
        resolved.append(
            MaskCandidate(
                semantic_name=broad_root.semantic_name,
                semantic_parent=broad_root.semantic_parent,
                mask=geometry.mask,
                score=max(broad_root.score, evidence.score),
                source=geometry.source,
                prompt=resolved_object_label,
                source_reliability=max(
                    broad_root.source_reliability,
                    evidence.source_reliability,
                    geometry.source_reliability,
                ),
                metadata=metadata,
            )
        )
        diagnostics.append(
            {
                "root_key": _root_key(broad_root),
                "root_semantic": broad_root.semantic_name,
                "status": "accepted",
                "selected_profile": profile,
                "consensus_score": float(winner["consensus_score"]),
                "margin": margin,
                "classifier_support": classifier_support,
                "classifier_margin": classifier_margin,
                "winner_classifier_margin": winner_classifier_margin,
                "classifier_probability_ratio": classifier_probability_ratio,
                "detector_margin": detector_margin,
                "detector_only_support": detector_only_support,
                "root_label_profile": label_profile,
                "root_label_support": root_label_support,
                "geometry_source": _root_key(geometry),
                "isolated_root_replacement": winner_isolated_replacement,
                "geometry_candidates": [
                    {
                        "root_key": _root_key(row["candidate"]),
                        **dict(row["metrics"]),
                    }
                    for row in geometry_rows
                ],
                "ranking": [
                    {key: value for key, value in row.items() if key != "candidate"}
                    for row in ranking
                ],
            }
        )
    return ProfileResolutionResult(
        tuple(resolved),
        {
            "algorithm": "hpid-profile-consensus-resolution-v1",
            "root_count": len(broad_roots),
            "profile_candidate_count": len(profile_candidates),
            "accepted_root_count": sum(
                row["status"] == "accepted" for row in diagnostics
            ),
            "roots": diagnostics,
            "ground_truth_used": False,
        },
    )
