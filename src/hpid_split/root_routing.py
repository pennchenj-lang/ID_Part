from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np
from PIL import Image

from .fusion import MaskCandidate, mask_iou


@dataclass(frozen=True)
class RootRoutingConfig:
    """Controls image-only routing from scene proposals to asset roots."""

    mode: str = "primary"
    target_point_xy: tuple[float, float] | None = None
    maximum_target_point_distance_ratio: float = 0.10
    border_band_ratio: float = 0.018
    physical_hypothesis_iou: float = 0.58
    physical_hypothesis_box_iou: float = 0.86
    physical_hypothesis_box_minimum_area_ratio: float = 0.30
    physical_hypothesis_box_maximum_centroid_ratio: float = 0.08
    physical_hypothesis_containment: float = 0.90
    physical_hypothesis_minimum_area_ratio: float = 0.42
    physical_hypothesis_maximum_centroid_ratio: float = 0.12
    cross_source_match_iou: float = 0.16
    cross_source_match_containment: float = 0.55
    maximum_cross_source_centroid_ratio: float = 0.62
    include_attached_roots: bool = True
    maximum_attached_area_ratio: float = 0.48
    minimum_attached_seed_overlap: float = 0.005
    maximum_attached_seed_containment: float = 0.72
    minimum_attached_coherence: float = 0.72
    maximum_attached_distance_ratio: float = 0.012
    minimum_attached_domain_evidence: float = 0.012
    minimum_attached_domain_contrast: float = 0.004
    maximum_weak_attached_border_fraction: float = 0.12
    maximum_weak_attached_touched_sides: int = 2
    minimum_scene_root_pixels: int = 24
    minimum_scene_root_area_ratio: float = 0.00035
    minimum_scene_group_score: float = 0.10
    maximum_scene_roots: int = 48
    minimum_scene_layer_area_ratio: float = 0.02


@dataclass(frozen=True)
class RootRoutingResult:
    candidates: tuple[MaskCandidate, ...]
    diagnostics: dict[str, object]


def _root_origin(candidate: MaskCandidate) -> str:
    explicit = candidate.metadata.get("root_origin")
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    root_index = candidate.metadata.get("root_index")
    if candidate.semantic_name == candidate.semantic_parent:
        return candidate.source.rsplit("/", maxsplit=1)[0]
    return f"legacy-root-origin:{root_index}"


def candidate_root_key(candidate: MaskCandidate) -> str | None:
    root_index = candidate.metadata.get("root_index")
    if root_index is None:
        return None
    return f"{_root_origin(candidate)}::{root_index}"


def propagate_scene_object_identity(
    candidates: list[MaskCandidate],
    roots: list[MaskCandidate],
) -> tuple[tuple[MaskCandidate, ...], dict[str, object]]:
    """Copy canonical scene ownership from each routed root to its descendants."""

    identity_fields = ("scene_object_id", "physical_group_id", "scene_role")
    roots_by_key = {
        key: root
        for root in roots
        if (key := candidate_root_key(root)) is not None
        and root.metadata.get("scene_object_id") is not None
    }
    propagated: list[MaskCandidate] = []
    propagated_count = 0
    unresolved_count = 0
    for candidate in candidates:
        if candidate.metadata.get("scene_object_id") is not None:
            propagated.append(candidate)
            continue
        root = roots_by_key.get(candidate_root_key(candidate))
        if root is None:
            propagated.append(candidate)
            unresolved_count += 1
            continue
        metadata = {**candidate.metadata}
        for field in identity_fields:
            value = root.metadata.get(field)
            if value is not None:
                metadata[field] = value
        metadata["scene_identity_propagated"] = True
        propagated.append(replace(candidate, metadata=metadata))
        propagated_count += 1
    return tuple(propagated), {
        "algorithm": "hpid-scene-object-identity-propagation-v1",
        "canonical_root_count": len(roots_by_key),
        "candidate_count": len(candidates),
        "propagated_candidate_count": propagated_count,
        "unresolved_candidate_count": unresolved_count,
        "ground_truth_used": False,
    }


def _geometry(mask: np.ndarray) -> tuple[float, float, int, tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return 0.0, 0.0, 0, (0, 0, 0, 0)
    return (
        float(xs.mean()),
        float(ys.mean()),
        len(xs),
        (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)),
    )


def _profile_name(candidate: MaskCandidate) -> str | None:
    value = candidate.metadata.get("selected_part_profile")
    return str(value) if value is not None and str(value).strip() else None


def _mask_containment(first: np.ndarray, second: np.ndarray) -> float:
    intersection = int(np.count_nonzero(first & second))
    smaller = min(int(np.count_nonzero(first)), int(np.count_nonzero(second)))
    return intersection / smaller if smaller else 0.0


def _box_iou(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _coherence(mask: np.ndarray) -> float:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    area = int(np.count_nonzero(mask))
    if count <= 1 or area == 0:
        return 0.0
    largest = int(stats[1:, cv2.CC_STAT_AREA].max())
    return float(np.clip(largest / area, 0.0, 1.0))


def _edge_strength(image: Image.Image) -> np.ndarray:
    image_bgr = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    smoothed = cv2.GaussianBlur(image_bgr, (0, 0), 0.65)
    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB).astype(np.float32)
    gradients: list[np.ndarray] = []
    for channel in cv2.split(lab):
        dx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        gradients.append(cv2.magnitude(dx, dy))
    return np.maximum.reduce(gradients)


def _boundary_alignment(mask: np.ndarray, edge_strength: np.ndarray | None) -> float:
    if edge_strength is None or not np.any(mask):
        return 0.5
    binary = mask.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    boundary = cv2.dilate(binary, kernel).astype(bool) ^ cv2.erode(
        binary, kernel
    ).astype(bool)
    if not np.any(boundary):
        return 0.0
    reference = float(np.quantile(edge_strength, 0.90))
    if reference <= 1e-6:
        return 0.0
    return float(np.clip(edge_strength[boundary].mean() / reference, 0.0, 1.25))


def _semantic_mask_probability(root: MaskCandidate) -> float:
    """Read the target-text support for the selected SAM multimask candidate."""

    diagnostics = root.metadata.get("sam_multimask_selection")
    if not isinstance(diagnostics, dict):
        return 0.5
    rows = diagnostics.get("target_rows")
    selected = diagnostics.get("selected_index")
    if not isinstance(rows, list) or not isinstance(selected, int):
        return 0.5
    if not 0 <= selected < len(rows) or not isinstance(rows[selected], dict):
        return 0.5
    probability = rows[selected].get("probability")
    if not isinstance(probability, (float, int)):
        return 0.5
    return float(np.clip(probability, 0.0, 1.0))


def _root_score(
    root: MaskCandidate,
    children: list[MaskCandidate],
    image_shape: tuple[int, int],
    config: RootRoutingConfig,
    edge_strength: np.ndarray | None = None,
) -> tuple[float, dict[str, float | int | list[int]]]:
    height, width = image_shape
    center_x, center_y, area, box = _geometry(root.mask)
    image_area = max(1, height * width)
    area_fraction = area / image_area
    x0, y0, x1, y1 = box
    bbox_fraction = max(0, x1 - x0) * max(0, y1 - y0) / image_area
    width_span = max(0, x1 - x0) / max(1, width)
    height_span = max(0, y1 - y0) / max(1, height)
    frame_extent = float(np.clip(max(width_span, height_span), 0.0, 1.0))
    detector_score = float(np.clip(root.score / 0.65, 0.0, 1.0))
    sam_score = float(np.clip(root.metadata.get("sam_quality", 0.5), 0.0, 1.0))
    proposal_rank = root.metadata.get("global_asset_proposal_rank")
    proposal_score = float(root.metadata.get("global_asset_proposal_score", 0.0))
    if isinstance(proposal_rank, int) and proposal_rank > 0:
        proposal_priority = float(
            1.0
            if root.metadata.get("global_asset_proposal_accepted")
            and proposal_rank == 1
            else 0.65 / proposal_rank
            + 0.35 * np.clip((proposal_score - 0.15) / 0.20, 0.0, 1.0)
        )
    else:
        proposal_priority = 0.0
    semantic_mask_probability = _semantic_mask_probability(root)
    center_distance = np.hypot(
        (center_x / max(1, width) - 0.5) / 0.5,
        (center_y / max(1, height) - 0.5) / 0.5,
    ) / np.sqrt(2.0)
    center_score = float(np.clip(1.0 - center_distance, 0.0, 1.0))
    area_score = float(np.exp(-0.70 * abs(np.log(max(area_fraction, 1e-5) / 0.20))))
    area_salience = float(np.clip(np.sqrt(area_fraction / 0.20), 0.0, 1.0))
    unique_children = len(
        {
            candidate.semantic_name
            for candidate in children
            if candidate.semantic_name != root.semantic_name
        }
    )
    child_score = float(
        np.clip(0.65 * unique_children / 10.0 + 0.35 * len(children) / 20.0, 0.0, 1.0)
    )
    coherence = _coherence(root.mask)
    boundary_alignment = _boundary_alignment(root.mask, edge_strength)
    domain_evidence = root.metadata.get("domain_evidence_score")
    band = max(1, round(min(height, width) * config.border_band_ratio))
    border = np.zeros(root.mask.shape, dtype=bool)
    border[:band] = True
    border[-band:] = True
    border[:, :band] = True
    border[:, -band:] = True
    border_fraction = np.count_nonzero(root.mask & border) / max(1, area)
    touched_sides = sum(
        (
            x0 <= band,
            y0 <= band,
            x1 >= width - band,
            y1 >= height - band,
        )
    )
    border_penalty = float(
        np.clip(border_fraction / 0.035, 0.0, 1.0) * 0.08
        + min(0.10, touched_sides * 0.025)
    )
    oversize_penalty = float(0.24 * np.clip((area_fraction - 0.82) / 0.18, 0.0, 1.0))
    bbox_salience = float(np.clip(np.sqrt(bbox_fraction / 0.20), 0.0, 1.0))
    physical_salience = float(
        0.14 * detector_score
        + 0.12 * sam_score
        + 0.15 * area_salience
        + 0.14 * bbox_salience
        + 0.07 * center_score
        + 0.12 * coherence
        + 0.12 * float(np.clip(boundary_alignment, 0.0, 1.0))
        + 0.07 * frame_extent
        + 0.07 * semantic_mask_probability
        - border_penalty
        - oversize_penalty
    )
    if domain_evidence is None:
        score = (
            0.25 * detector_score
            + 0.10 * sam_score
            + 0.18 * area_score
            + 0.17 * center_score
            + 0.20 * child_score
            + 0.10 * coherence
            - border_penalty
        )
    else:
        score = (
            0.10 * detector_score
            + 0.08 * sam_score
            + 0.15 * area_score
            + 0.15 * center_score
            + 0.20 * child_score
            + 0.08 * coherence
            + 0.28 * float(np.clip(domain_evidence, 0.0, 1.0))
            - border_penalty
        )
    return float(score), {
        "detector_score": detector_score,
        "sam_score": sam_score,
        "area_fraction": area_fraction,
        "area_score": area_score,
        "area_salience": area_salience,
        "bbox_fraction": float(bbox_fraction),
        "bbox_salience": bbox_salience,
        "frame_extent": frame_extent,
        "center_score": center_score,
        "child_semantic_count": unique_children,
        "child_candidate_count": len(children),
        "child_score": child_score,
        "coherence": coherence,
        "boundary_alignment": boundary_alignment,
        "semantic_mask_probability": semantic_mask_probability,
        "global_asset_proposal_priority": proposal_priority,
        "global_asset_proposal_score": proposal_score,
        "global_asset_proposal_rank": (
            int(proposal_rank) if isinstance(proposal_rank, int) else None
        ),
        "global_asset_proposal_accepted": bool(
            root.metadata.get("global_asset_proposal_accepted", False)
        ),
        "domain_evidence_score": (
            float(domain_evidence) if domain_evidence is not None else None
        ),
        "domain_evidence_contrast": (
            float(root.metadata.get("domain_evidence_contrast", 0.0))
            if domain_evidence is not None
            else None
        ),
        "border_fraction": float(border_fraction),
        "touched_sides": int(touched_sides),
        "physical_salience_score": physical_salience,
        "oversize_penalty": oversize_penalty,
        "bbox_xyxy": list(box),
    }


def _propagate_isolated_profile_hypotheses(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Pass category evidence from a subregion to an enclosing same-domain root."""

    profile_rows = [
        row
        for row in rows
        if isinstance(row["candidate"], MaskCandidate)
        and row["candidate"].metadata.get("profile_hint_source")
        == "isolated_profile_query"
        and _profile_name(row["candidate"]) is not None
    ]
    for row in rows:
        geometry = row["candidate"]
        assert isinstance(geometry, MaskCandidate)
        supports: list[tuple[float, float, dict[str, object]]] = []
        for profile_row in profile_rows:
            evidence = profile_row["candidate"]
            assert isinstance(evidence, MaskCandidate)
            if evidence.semantic_name != geometry.semantic_name:
                continue
            if str(profile_row["root_key"]) == str(row["root_key"]):
                continue
            evidence_area = max(1, int(np.count_nonzero(evidence.mask)))
            containment = (
                int(np.count_nonzero(evidence.mask & geometry.mask)) / evidence_area
            )
            overlap = mask_iou(evidence.mask, geometry.mask)
            if containment < 0.62 and overlap < 0.12:
                continue
            profile_score = float(
                evidence.metadata.get("profile_consensus_score", evidence.score)
            )
            supports.append((profile_score, containment, profile_row))
        if not supports:
            continue
        supports.sort(key=lambda item: (item[0], item[1]), reverse=True)
        best_score, best_containment, best_row = supports[0]
        runner_up = supports[1][0] if len(supports) > 1 else 0.0
        evidence = best_row["candidate"]
        assert isinstance(evidence, MaskCandidate)
        row["candidate"] = MaskCandidate(
            semantic_name=geometry.semantic_name,
            semantic_parent=geometry.semantic_parent,
            mask=geometry.mask,
            score=max(geometry.score, evidence.score),
            source=geometry.source,
            prompt=evidence.prompt,
            source_reliability=max(
                geometry.source_reliability, evidence.source_reliability
            ),
            metadata={
                **geometry.metadata,
                "selected_part_profile": _profile_name(evidence),
                "part_profile_specificity": 1.0,
                "profile_hint_source": "isolated_profile_query",
                "profile_evidence_root_key": str(best_row["root_key"]),
                "profile_evidence_containment": float(best_containment),
                "profile_evidence_score": float(best_score),
                "profile_evidence_margin": float(best_score - runner_up),
                "profile_classifier": evidence.metadata.get("profile_classifier"),
                "root_model_label": evidence.metadata.get("root_model_label"),
            },
        )
        row["profile_evidence_root_key"] = str(best_row["root_key"])
        row["profile_hypothesis_score"] = float(best_score)
        row["profile_hypothesis_margin"] = float(best_score - runner_up)
    return rows


def _same_physical_hypothesis(
    first: MaskCandidate,
    second: MaskCandidate,
    *,
    image_shape: tuple[int, int],
    config: RootRoutingConfig,
) -> bool:
    """Return whether two category proposals describe the same visible entity."""

    overlap = mask_iou(first.mask, second.mask)
    if overlap >= config.physical_hypothesis_iou:
        return True
    first_x, first_y, first_area, first_box = _geometry(first.mask)
    second_x, second_y, second_area, second_box = _geometry(second.mask)
    diagonal = max(1.0, float(np.hypot(*image_shape)))
    centroid_ratio = float(np.hypot(first_x - second_x, first_y - second_y) / diagonal)
    area_ratio = min(first_area, second_area) / max(1, first_area, second_area)
    if (
        _box_iou(first_box, second_box) >= config.physical_hypothesis_box_iou
        and area_ratio >= config.physical_hypothesis_box_minimum_area_ratio
        and centroid_ratio <= config.physical_hypothesis_box_maximum_centroid_ratio
    ):
        return True
    containment = _mask_containment(first.mask, second.mask)
    if containment < config.physical_hypothesis_containment:
        return False
    if area_ratio < config.physical_hypothesis_minimum_area_ratio:
        return False
    return centroid_ratio <= config.physical_hypothesis_maximum_centroid_ratio


def _cluster_physical_hypotheses(
    rows: list[dict[str, object]],
    *,
    image_shape: tuple[int, int],
    config: RootRoutingConfig,
) -> list[list[dict[str, object]]]:
    """Cluster cross-domain root masks before assigning an asset category."""

    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[second_root] = first_root

    for first_index, first_row in enumerate(rows):
        first = first_row["candidate"]
        assert isinstance(first, MaskCandidate)
        for second_index in range(first_index + 1, len(rows)):
            second = rows[second_index]["candidate"]
            assert isinstance(second, MaskCandidate)
            if _same_physical_hypothesis(
                first,
                second,
                image_shape=image_shape,
                config=config,
            ):
                union(first_index, second_index)

    by_parent: dict[int, list[dict[str, object]]] = {}
    for index, row in enumerate(rows):
        by_parent.setdefault(find(index), []).append(row)
    ordered = sorted(
        by_parent.values(),
        key=lambda group: min(str(row["root_key"]) for row in group),
    )
    for group_index, group in enumerate(ordered, start=1):
        group_id = f"physical:{group_index:02d}"
        for row in group:
            row["physical_group_id"] = group_id
    return ordered


def _semantic_arbitration(
    group: list[dict[str, object]],
) -> tuple[dict[str, object], list[dict[str, object]], float]:
    """Choose a category, then its best full-entity geometric representative."""

    evidence_values: list[float | None] = []
    for row in group:
        metrics = row["metrics"]
        assert isinstance(metrics, dict)
        evidence = metrics.get("domain_evidence_score")
        if evidence is None:
            evidence_values.append(None)
        else:
            evidence_values.append(
                float(evidence)
                + 0.50 * float(metrics.get("domain_evidence_contrast") or 0.0)
            )
    finite = [value for value in evidence_values if value is not None]
    evidence_min = min(finite) if finite else 0.0
    evidence_max = max(finite) if finite else 0.0

    scored: list[dict[str, object]] = []
    for row, evidence in zip(group, evidence_values, strict=True):
        candidate = row["candidate"]
        metrics = row["metrics"]
        assert isinstance(candidate, MaskCandidate)
        assert isinstance(metrics, dict)
        if evidence is None or evidence_max - evidence_min < 1e-6:
            evidence_rank = 0.50
        else:
            evidence_rank = (evidence - evidence_min) / (evidence_max - evidence_min)
        root_specificity = float(
            np.clip(candidate.metadata.get("root_label_specificity", 0.0), 0.0, 1.0)
        )
        profile_specificity = float(
            np.clip(candidate.metadata.get("part_profile_specificity", 0.0), 0.0, 1.0)
        )
        profile_detector_score = float(
            np.clip(float(row.get("profile_hypothesis_score", 0.0)) / 0.65, 0.0, 1.0)
        )
        semantic_score = float(
            0.26 * max(float(metrics["detector_score"]), profile_detector_score)
            + 0.30 * evidence_rank
            + 0.16 * root_specificity
            + 0.10 * profile_specificity
            + 0.04 * float(metrics["child_score"])
            + 0.14 * float(metrics["global_asset_proposal_priority"])
        )
        row["domain_evidence_rank"] = float(evidence_rank)
        row["semantic_arbitration_score"] = semantic_score
        row["root_label_specificity"] = root_specificity
        row["part_profile_specificity"] = profile_specificity
        scored.append(row)

    semantic_representatives: list[dict[str, object]] = []
    maximum_group_area = max(
        float(dict(row["metrics"])["area_fraction"]) for row in scored
    )
    for semantic_name in sorted({str(row["semantic_name"]) for row in scored}):
        semantic_rows = [
            row for row in scored if str(row["semantic_name"]) == semantic_name
        ]
        unique_origins = {str(row["origin"]) for row in semantic_rows}
        coverage_weight = 0.24 if len(unique_origins) == 1 else 0.18
        category_representative = max(
            semantic_rows,
            key=lambda row: (
                float(row["semantic_arbitration_score"]),
                float(row["routing_score"]),
            ),
        )
        category_score = float(category_representative["semantic_arbitration_score"])
        for row in semantic_rows:
            metrics = row["metrics"]
            assert isinstance(metrics, dict)
            relative_coverage = float(
                np.clip(
                    float(metrics["area_fraction"]) / max(1e-6, maximum_group_area),
                    0.0,
                    1.0,
                )
            )
            geometry_score = float(
                0.27 * float(metrics["physical_salience_score"])
                + min(0.20, coverage_weight) * relative_coverage
                + 0.14 * float(row["part_profile_specificity"])
                + 0.06 * float(row["root_label_specificity"])
                + 0.08 * float(metrics["detector_score"])
                + 0.09 * float(metrics["sam_score"])
                + 0.04 * float(row["domain_evidence_rank"])
                + 0.14 * float(metrics["global_asset_proposal_priority"])
            )
            row["semantic_category_score"] = category_score
            row["geometry_representative_score"] = geometry_score
            row["physical_group_relative_coverage"] = relative_coverage
        geometry_representative = max(
            semantic_rows,
            key=lambda row: (
                float(row["geometry_representative_score"]),
                float(row["routing_score"]),
            ),
        )
        category_candidate = category_representative["candidate"]
        geometry_candidate = geometry_representative["candidate"]
        assert isinstance(category_candidate, MaskCandidate)
        assert isinstance(geometry_candidate, MaskCandidate)
        category_profile = _profile_name(category_candidate)
        geometry_profile = _profile_name(geometry_candidate)
        geometry_profile_specificity = float(
            geometry_candidate.metadata.get("part_profile_specificity", 0.0)
        )
        if geometry_profile is not None and geometry_profile_specificity >= 0.80:
            category_profile = geometry_profile
            category_candidate = geometry_candidate
        if category_profile is not None:
            profile_hint_source = str(
                category_candidate.metadata.get("profile_hint_source")
                or "specific_root_label"
            )
            propagated = MaskCandidate(
                semantic_name=geometry_candidate.semantic_name,
                semantic_parent=geometry_candidate.semantic_parent,
                mask=geometry_candidate.mask,
                score=max(category_candidate.score, geometry_candidate.score),
                source=geometry_candidate.source,
                prompt=category_candidate.prompt,
                source_reliability=max(
                    category_candidate.source_reliability,
                    geometry_candidate.source_reliability,
                ),
                metadata={
                    **geometry_candidate.metadata,
                    "selected_part_profile": category_profile,
                    "part_profile_specificity": 1.0,
                    "profile_hint_source": profile_hint_source,
                    "profile_evidence_root_key": str(
                        category_representative["root_key"]
                    ),
                    "root_model_label": category_candidate.metadata.get(
                        "root_model_label"
                    ),
                },
            )
            geometry_representative = {
                **geometry_representative,
                "candidate": propagated,
                "profile_evidence_root_key": str(category_representative["root_key"]),
            }
        semantic_representatives.append(geometry_representative)

    semantic_representatives.sort(
        key=lambda row: (
            float(row["semantic_category_score"]),
            float(row["geometry_representative_score"]),
            str(row["root_key"]),
        ),
        reverse=True,
    )
    # Crop classifiers are vulnerable to context when the target is small.  If
    # two category hypotheses describe the same physical mask and one has a
    # substantially stronger, specific detector match, let that direct evidence
    # override the crop-domain rank.  This is intentionally a high-margin gate;
    # ordinary close calls continue through the consensus score above.
    specific_rows = [
        row
        for row in scored
        if float(row.get("root_label_specificity", 0.0)) >= 0.80
        and float(row.get("part_profile_specificity", 0.0)) >= 0.80
    ]
    detector_override: dict[str, object] | None = None
    detector_margin = 0.0
    if specific_rows:
        detector_ranked = sorted(
            specific_rows,
            key=lambda row: float(dict(row["metrics"])["detector_score"]),
            reverse=True,
        )
        default_winner = semantic_representatives[0]
        default_detector = float(
            dict(default_winner["metrics"])["detector_score"]
        )
        detector_margin = float(
            dict(detector_ranked[0]["metrics"])["detector_score"]
            - (
                dict(detector_ranked[1]["metrics"])["detector_score"]
                if len(detector_ranked) >= 2
                else default_detector
            )
        )
        default_profile_specificity = float(
            default_winner.get("part_profile_specificity", 0.0)
        )
        direct_over_unspecific_margin = float(
            dict(detector_ranked[0]["metrics"])["detector_score"]
            - default_detector
        )
        if (
            float(dict(detector_ranked[0]["metrics"])["detector_score"]) >= 0.58
            and (
                detector_margin >= 0.22
                or (
                    default_profile_specificity < 0.80
                    and direct_over_unspecific_margin >= 0.18
                )
            )
        ):
            winning_semantic = str(detector_ranked[0]["semantic_name"])
            detector_override = next(
                row
                for row in semantic_representatives
                if str(row["semantic_name"]) == winning_semantic
            )
            detector_override["semantic_arbitration_override"] = (
                "specific_detector_margin"
            )
            override_candidate = detector_override["candidate"]
            assert isinstance(override_candidate, MaskCandidate)
            detector_override["candidate"] = replace(
                override_candidate,
                metadata={
                    **override_candidate.metadata,
                    "semantic_arbitration_override": (
                        "specific_detector_margin"
                    ),
                    "semantic_detector_margin": detector_margin,
                    "semantic_detector_margin_over_default": (
                        direct_over_unspecific_margin
                    ),
                },
            )
    margin = (
        float(semantic_representatives[0]["semantic_category_score"])
        - float(semantic_representatives[1]["semantic_category_score"])
        if len(semantic_representatives) > 1
        else 1.0
    )
    scored.sort(
        key=lambda row: (
            float(row["semantic_category_score"]),
            float(row["geometry_representative_score"]),
        ),
        reverse=True,
    )
    if detector_override is not None:
        return detector_override, scored, detector_margin
    return semantic_representatives[0], scored, margin


def _resolve_profile_geometry(
    winner: dict[str, object],
    rows: list[dict[str, object]],
) -> dict[str, object]:
    """Use isolated category evidence without replacing full-object geometry."""

    candidate = winner["candidate"]
    assert isinstance(candidate, MaskCandidate)
    profile = _profile_name(candidate)
    if profile is None:
        return winner
    same_domain = [
        row for row in rows if str(row["semantic_name"]) == candidate.semantic_name
    ]
    profile_mask = candidate.mask
    compatible = [
        row
        for row in same_domain
        if (
            mask_iou(row["candidate"].mask, profile_mask) >= 0.42
            or _mask_containment(row["candidate"].mask, profile_mask) >= 0.75
        )
    ]
    pool = compatible or [winner]
    maximum_bbox_fraction = max(
        float(dict(row["metrics"])["bbox_fraction"]) for row in pool
    )
    for row in pool:
        metrics = row["metrics"]
        assert isinstance(metrics, dict)
        relative_bbox = float(metrics["bbox_fraction"]) / max(
            1e-6, maximum_bbox_fraction
        )
        row["profile_geometry_score"] = float(
            0.42 * float(metrics["physical_salience_score"])
            + 0.20 * relative_bbox
            + 0.14 * float(metrics["sam_score"])
            + 0.10 * float(metrics["detector_score"])
            + 0.09 * float(metrics["coherence"])
        )
    geometry = max(
        pool,
        key=lambda row: (
            float(row.get("profile_geometry_score", 0.0)),
            float(row["routing_score"]),
        ),
    )
    geometry_candidate = geometry["candidate"]
    assert isinstance(geometry_candidate, MaskCandidate)
    if geometry is winner:
        return winner
    geometry_profile = _profile_name(geometry_candidate)
    geometry_profile_specificity = float(
        geometry_candidate.metadata.get("part_profile_specificity", 0.0)
    )
    resolved_profile = (
        geometry_profile
        if geometry_profile is not None and geometry_profile_specificity >= 0.80
        else profile
    )
    profile_source_candidate = (
        geometry_candidate if resolved_profile == geometry_profile else candidate
    )
    metadata = {
        **geometry_candidate.metadata,
        "selected_part_profile": resolved_profile,
        "part_profile_specificity": 1.0,
        "profile_hint_source": profile_source_candidate.metadata.get(
            "profile_hint_source"
        )
        or "specific_root_label",
        "profile_evidence_root_key": str(
            geometry["root_key"]
            if profile_source_candidate is geometry_candidate
            else winner["root_key"]
        ),
        "geometry_source_root_key": str(geometry["root_key"]),
        "root_model_label": candidate.metadata.get("root_model_label"),
        **(
            {
                "semantic_arbitration_override": candidate.metadata[
                    "semantic_arbitration_override"
                ],
                "semantic_detector_margin": candidate.metadata.get(
                    "semantic_detector_margin", 0.0
                ),
            }
            if candidate.metadata.get("semantic_arbitration_override")
            else {}
        ),
    }
    resolved_candidate = MaskCandidate(
        semantic_name=geometry_candidate.semantic_name,
        semantic_parent=geometry_candidate.semantic_parent,
        mask=geometry_candidate.mask,
        score=max(candidate.score, geometry_candidate.score),
        source=geometry_candidate.source,
        prompt=candidate.prompt,
        source_reliability=max(
            candidate.source_reliability, geometry_candidate.source_reliability
        ),
        metadata=metadata,
    )
    return {
        **geometry,
        "candidate": resolved_candidate,
        "semantic_arbitration_score": winner["semantic_arbitration_score"],
        "semantic_category_score": winner.get(
            "semantic_category_score", winner["semantic_arbitration_score"]
        ),
        "profile_evidence_root_key": str(winner["root_key"]),
        "profile_geometry_resolved": True,
    }


def _select_salient_group(
    groups: list[dict[str, object]],
    *,
    score_tolerance: float = 0.015,
) -> tuple[dict[str, object], int]:
    """Break near-ties with geometric evidence instead of semantic confidence.

    Open-vocabulary confidence can be high for a contextual surface or a crop
    that contains the named object. When physical salience is effectively tied,
    the root with the stronger geometric representative is the safer canonical
    mask. The tolerance is absolute and does not inspect evaluation labels.
    """

    best_group_score = max(float(group["group_score"]) for group in groups)
    competitive = [
        group
        for group in groups
        if best_group_score - float(group["group_score"]) <= score_tolerance
    ]
    globally_owned: list[dict[str, object]] = []
    for group in groups:
        winner = dict(group["winner"])
        metrics = dict(winner["metrics"])
        proposal_rank = metrics.get("global_asset_proposal_rank")
        proposal_priority = float(metrics.get("global_asset_proposal_priority", 0.0))
        proposal_accepted = bool(
            metrics.get("global_asset_proposal_accepted", False)
        )
        frame_extent = float(metrics.get("frame_extent", 0.0))
        bbox_salience = float(metrics.get("bbox_salience", 0.0))
        area_salience = float(metrics.get("area_salience", 0.0))
        physical_score = float(
            group.get("physical_score", metrics.get("physical_salience_score", 0.0))
        )
        maximum_gap = 0.23 if proposal_accepted else 0.18
        unaccepted_geometry_supported = bool(
            frame_extent >= 0.80
            or bbox_salience >= 0.95
            and area_salience >= 0.70
            or (
                best_group_score - float(group["group_score"]) <= 0.06
                and frame_extent >= 0.72
                and area_salience >= 0.55
            )
        )
        if (
            proposal_rank == 1
            and proposal_priority >= 0.65
            and physical_score >= 0.45
            and best_group_score - float(group["group_score"]) <= maximum_gap
            and (proposal_accepted or unaccepted_geometry_supported)
        ):
            globally_owned.append(group)
    if globally_owned:
        nondegenerate_owned = [
            group
            for group in globally_owned
            if not (
                float(
                    dict(dict(group["winner"])["metrics"]).get(
                        "area_fraction", 0.0
                    )
                )
                >= 0.90
                and int(
                    dict(dict(group["winner"])["metrics"]).get(
                        "touched_sides", 0
                    )
                )
                    >= 3
                    and any(
                        str(other["winner"]["semantic_name"])
                        == str(group["winner"]["semantic_name"])
                    and float(
                        dict(dict(other["winner"])["metrics"])[
                            "physical_salience_score"
                        ]
                    )
                    >= float(
                        dict(dict(group["winner"])["metrics"])[
                            "physical_salience_score"
                        ]
                    )
                    + 0.08
                    and float(
                        dict(dict(other["winner"])["metrics"]).get(
                            "area_fraction", 1.0
                        )
                    )
                    <= 0.80
                    for other in globally_owned
                    if other is not group
                )
            )
        ]
        if nondegenerate_owned:
            globally_owned = nondegenerate_owned
        selected = max(
            globally_owned,
            key=lambda group: (
                bool(
                    dict(dict(group["winner"])["metrics"]).get(
                        "global_asset_proposal_accepted", False
                    )
                ),
                float(
                    dict(dict(group["winner"])["metrics"]).get(
                        "global_asset_proposal_priority", 0.0
                    )
                ),
                float(dict(group["winner"]).get("semantic_category_score", 0.0)),
                float(group["group_score"]),
            ),
        )
        return selected, max(len(competitive), len(globally_owned))
    selected = max(
        competitive,
        key=lambda group: (
            0.25 * float(dict(dict(group["winner"])["metrics"])["frame_extent"])
            + 0.17 * float(dict(dict(group["winner"])["metrics"])["bbox_salience"])
            + 0.17 * float(dict(dict(group["winner"])["metrics"])["area_salience"])
            + 0.13
            * float(dict(dict(group["winner"])["metrics"])["physical_salience_score"])
            + 0.08 * float(dict(dict(group["winner"])["metrics"])["sam_score"])
            + 0.05
            * float(dict(dict(group["winner"])["metrics"])["detector_score"])
            + 0.15
            * float(
                dict(dict(group["winner"])["metrics"]).get(
                    "global_asset_proposal_priority", 0.0
                )
            ),
            float(dict(group["winner"])["routing_score"]),
            float(group["group_score"]),
        ),
    )
    return selected, len(competitive)


def _select_prompt_consensus_group(
    groups: list[dict[str, object]],
) -> tuple[dict[str, object] | None, dict[str, object]]:
    """Find one direct-prompt root that consolidates disjoint visible fragments."""

    representatives: list[tuple[dict[str, object], dict[str, object]]] = []
    for group in groups:
        direct_rows = [
            row
            for row in group["rows"]
            if dict(row["candidate"].metadata).get("root_query_mode")
            == "user_asset_prompt"
        ]
        if not direct_rows:
            continue
        representative = max(
            direct_rows,
            key=lambda row: (
                float(dict(row["metrics"])["semantic_mask_probability"]),
                float(row["routing_score"]),
            ),
        )
        representatives.append((group, representative))

    scored: list[
        tuple[int, float, float, dict[str, object], dict[str, object]]
    ] = []
    public_rows: list[dict[str, object]] = []

    def eligible_metrics(metrics: dict[str, object]) -> bool:
        return bool(
            float(metrics["semantic_mask_probability"]) >= 0.50
            and float(metrics["area_fraction"]) <= 0.72
            and int(metrics["touched_sides"]) <= 2
        )

    for group, row in representatives:
        candidate = row["candidate"]
        metrics = row["metrics"]
        assert isinstance(candidate, MaskCandidate)
        assert isinstance(metrics, dict)
        semantic_support = float(metrics["semantic_mask_probability"])
        eligible = eligible_metrics(metrics)
        supporting_groups: list[str] = []
        if eligible:
            for other_group, other_row in representatives:
                if other_group is group:
                    continue
                other = other_row["candidate"]
                other_metrics = other_row["metrics"]
                assert isinstance(other, MaskCandidate)
                assert isinstance(other_metrics, dict)
                if not eligible_metrics(other_metrics):
                    continue
                if _mask_containment(candidate.mask, other.mask) >= 0.38:
                    supporting_groups.append(str(other_group["physical_group_id"]))
        support_count = len(supporting_groups)
        span_score = float(
            0.50 * semantic_support
            + 0.28 * min(1.0, float(metrics["bbox_fraction"]) / 0.55)
            + 0.22 * float(group["group_score"])
        )
        public_rows.append(
            {
                "physical_group_id": str(group["physical_group_id"]),
                "representative_root_key": str(row["root_key"]),
                "eligible": eligible,
                "support_count": support_count,
                "supporting_physical_group_ids": sorted(supporting_groups),
                "semantic_support": semantic_support,
                "span_score": span_score,
            }
        )
        if eligible:
            scored.append((support_count, span_score, semantic_support, group, row))

    winner = max(scored, default=None, key=lambda item: item[:3])
    selected = winner[3] if winner is not None and winner[0] >= 2 else None
    return selected, {
        "algorithm": "direct-prompt-fragment-consensus-v1",
        "candidate_group_count": len(representatives),
        "selected_physical_group_id": (
            str(selected["physical_group_id"]) if selected is not None else None
        ),
        "minimum_support_count": 2,
        "rows": public_rows,
        "ground_truth_used": False,
    }


def _rewrite_candidate_reference(
    value: object,
    *,
    old_root_candidate_key: str,
    canonical_root_candidate_key: str,
) -> object:
    if not isinstance(value, str):
        return value
    if value == old_root_candidate_key:
        return canonical_root_candidate_key
    prefix = f"{old_root_candidate_key}/"
    if value.startswith(prefix):
        return f"{canonical_root_candidate_key}/{value[len(prefix) :]}"
    return value


def _remap_candidate_to_canonical_root(
    candidate: MaskCandidate,
    *,
    evidence_root_key: str,
    canonical_root_key: str,
    canonical_root: MaskCandidate,
) -> MaskCandidate:
    """Attach evidence from another detector to one physical root identity."""

    candidate_key = candidate.metadata.get("candidate_key")
    old_root_candidate_key = str(
        candidate.metadata.get("root_candidate_key")
        or (
            str(candidate_key).split("/", maxsplit=1)[0]
            if candidate_key is not None
            else None
        )
        or f"root:{candidate.metadata.get('root_index')}"
    )
    canonical_root_candidate_key = str(
        canonical_root.metadata.get("candidate_key")
        or f"root:{canonical_root.metadata.get('root_index')}"
    )
    metadata = dict(candidate.metadata)
    for field_name in (
        "candidate_key",
        "parent_candidate_key",
        "assembly_parent_candidate_key",
        "topology_anchor_candidate_key",
    ):
        if field_name in metadata:
            metadata[field_name] = _rewrite_candidate_reference(
                metadata[field_name],
                old_root_candidate_key=old_root_candidate_key,
                canonical_root_candidate_key=canonical_root_candidate_key,
            )
    metadata.update(
        {
            "root_origin": canonical_root.metadata.get("root_origin"),
            "root_index": canonical_root.metadata.get("root_index"),
            "canonical_root_key": canonical_root_key,
            "root_evidence_key": evidence_root_key,
            "root_remapped": evidence_root_key != canonical_root_key,
            "ground_truth_used": False,
        }
    )
    return replace(candidate, metadata=metadata)


def _tag_scene_candidate(
    candidate: MaskCandidate,
    *,
    object_id: str,
    physical_group_id: str,
    scene_role: str,
) -> MaskCandidate:
    return replace(
        candidate,
        metadata={
            **candidate.metadata,
            "scene_object_id": object_id,
            "physical_group_id": physical_group_id,
            "scene_role": scene_role,
            "ground_truth_used": False,
        },
    )


def _route_scene_groups(
    candidates: list[MaskCandidate],
    rows: list[dict[str, object]],
    physical_group_rows: list[dict[str, object]],
    *,
    image_shape: tuple[int, int],
    config: RootRoutingConfig,
) -> RootRoutingResult:
    """Canonicalize every physical object before recursive part fusion."""

    image_area = max(1, int(image_shape[0] * image_shape[1]))
    accepted_groups: list[dict[str, object]] = []
    rejected_groups: list[dict[str, object]] = []
    for group in physical_group_rows:
        winner = _resolve_profile_geometry(dict(group["winner"]), list(group["rows"]))
        candidate = winner["candidate"]
        metrics = winner["metrics"]
        assert isinstance(candidate, MaskCandidate)
        assert isinstance(metrics, dict)
        area = int(np.count_nonzero(candidate.mask))
        area_ratio = area / image_area
        rejection_reason: str | None = None
        if area < config.minimum_scene_root_pixels:
            rejection_reason = "too_few_pixels"
        elif area_ratio < config.minimum_scene_root_area_ratio:
            rejection_reason = "too_small_for_scene_root"
        elif float(group["group_score"]) < config.minimum_scene_group_score:
            rejection_reason = "weak_physical_group"
        payload = {**group, "winner": winner}
        if rejection_reason is None:
            accepted_groups.append(payload)
        else:
            rejected_groups.append(
                {
                    "physical_group_id": str(group["physical_group_id"]),
                    "semantic_name": candidate.semantic_name,
                    "area_ratio": float(area_ratio),
                    "group_score": float(group["group_score"]),
                    "reason": rejection_reason,
                }
            )

    accepted_groups.sort(
        key=lambda group: (
            float(group["group_score"]),
            int(np.count_nonzero(dict(group["winner"])["candidate"].mask)),
            str(group["physical_group_id"]),
        ),
        reverse=True,
    )
    if len(accepted_groups) > config.maximum_scene_roots:
        for group in accepted_groups[config.maximum_scene_roots :]:
            candidate = dict(group["winner"])["candidate"]
            assert isinstance(candidate, MaskCandidate)
            rejected_groups.append(
                {
                    "physical_group_id": str(group["physical_group_id"]),
                    "semantic_name": candidate.semantic_name,
                    "group_score": float(group["group_score"]),
                    "reason": "scene_root_limit",
                }
            )
        accepted_groups = accepted_groups[: config.maximum_scene_roots]

    def spatial_key(group: dict[str, object]) -> tuple[float, float, str, float]:
        candidate = dict(group["winner"])["candidate"]
        assert isinstance(candidate, MaskCandidate)
        center_x, center_y, area, _ = _geometry(candidate.mask)
        return center_y, center_x, candidate.semantic_name, -float(area)

    ordered_groups = sorted(accepted_groups, key=spatial_key)
    canonical_by_evidence_key: dict[str, tuple[str, str, str, MaskCandidate]] = {}
    public_groups: list[dict[str, object]] = []
    selected_keys: set[str] = set()
    for object_index, group in enumerate(ordered_groups, start=1):
        winner = dict(group["winner"])
        canonical = winner["candidate"]
        assert isinstance(canonical, MaskCandidate)
        physical_group_id = str(group["physical_group_id"])
        object_id = f"object_{object_index:03d}"
        canonical_area_ratio = float(np.count_nonzero(canonical.mask) / image_area)
        scene_role = (
            "scene_layer"
            if canonical.semantic_name == "terrain"
            and canonical_area_ratio >= config.minimum_scene_layer_area_ratio
            else "object"
        )
        canonical = _tag_scene_candidate(
            canonical,
            object_id=object_id,
            physical_group_id=physical_group_id,
            scene_role=scene_role,
        )
        canonical_key = str(winner["root_key"])
        evidence_rows = [
            row
            for row in group["rows"]
            if str(row["semantic_name"]) == canonical.semantic_name
        ]
        if not evidence_rows:
            evidence_rows = [winner]
        evidence_keys = {str(row["root_key"]) for row in evidence_rows}
        evidence_keys.add(canonical_key)
        selected_keys.update(evidence_keys)
        for evidence_key in evidence_keys:
            canonical_by_evidence_key[evidence_key] = (
                object_id,
                physical_group_id,
                canonical_key,
                canonical,
            )
        public_groups.append(
            {
                "scene_object_id": object_id,
                "physical_group_id": physical_group_id,
                "scene_role": scene_role,
                "semantic_name": canonical.semantic_name,
                "canonical_root_key": canonical_key,
                "evidence_root_keys": sorted(evidence_keys),
                "semantic_names_considered": list(group["semantic_names"]),
                "semantic_margin": float(group["semantic_margin"]),
                "group_score": float(group["group_score"]),
                "area_fraction": canonical_area_ratio,
            }
        )

    emitted_roots: set[str] = set()
    routed: list[MaskCandidate] = []
    for candidate in candidates:
        evidence_key = candidate_root_key(candidate)
        if evidence_key is None or evidence_key not in canonical_by_evidence_key:
            continue
        object_id, physical_group_id, canonical_key, canonical = (
            canonical_by_evidence_key[evidence_key]
        )
        is_root = (
            candidate.semantic_name == candidate.semantic_parent
            and candidate.metadata.get("parent_candidate_key") is None
        )
        if is_root:
            if canonical_key not in emitted_roots:
                routed.append(canonical)
                emitted_roots.add(canonical_key)
            continue
        remapped = _remap_candidate_to_canonical_root(
            candidate,
            evidence_root_key=evidence_key,
            canonical_root_key=canonical_key,
            canonical_root=canonical,
        )
        routed.append(
            _tag_scene_candidate(
                remapped,
                object_id=object_id,
                physical_group_id=physical_group_id,
                scene_role=str(canonical.metadata["scene_role"]),
            )
        )

    public_rows = []
    for row in rows:
        root_key = str(row["root_key"])
        assignment = canonical_by_evidence_key.get(root_key)
        public_rows.append(
            {
                "root_key": root_key,
                "origin": str(row["origin"]),
                "semantic_name": str(row["semantic_name"]),
                "physical_group_id": str(row.get("physical_group_id", "")),
                "routing_score": float(row["routing_score"]),
                "group_score": float(row.get("physical_group_score", 0.0)),
                "selected": assignment is not None,
                "selection_role": "scene_evidence" if assignment is not None else None,
                "scene_object_id": assignment[0] if assignment is not None else None,
                **dict(row["metrics"]),
            }
        )
    return RootRoutingResult(
        tuple(routed),
        {
            "algorithm": "hpid-physical-first-scene-router-v1",
            "mode": "scene",
            "root_proposal_count": len(
                [
                    candidate
                    for candidate in candidates
                    if candidate.semantic_name == candidate.semantic_parent
                    and candidate.metadata.get("parent_candidate_key") is None
                ]
            ),
            "physical_group_count": len(physical_group_rows),
            "selected_root_count": len(ordered_groups),
            "selected_evidence_root_count": len(selected_keys),
            "selected_candidate_count": len(routed),
            "rejected_candidate_count": len(candidates) - len(routed),
            "scene_objects": public_groups,
            "rejected_scene_groups": rejected_groups,
            "root_scores": public_rows,
            "ground_truth_used": False,
        },
    )


def route_asset_roots(
    candidates: list[MaskCandidate],
    *,
    image_shape: tuple[int, int],
    image: Image.Image | None = None,
    config: RootRoutingConfig | None = None,
) -> RootRoutingResult:
    """Select a salient asset and its cross-model root groups without labels.

    The router separates scene-level object discovery from part-level ownership.
    In ``primary`` mode, only candidates belonging to the most salient physical
    root enter HPID fusion. Root proposals from independent model stacks are
    retained when their masks match the selected root.
    """

    config = config or RootRoutingConfig()
    root_edge_strength = _edge_strength(image) if image is not None else None
    if config.mode not in {"primary", "scene", "all"}:
        raise ValueError(f"unsupported root routing mode: {config.mode}")
    roots = [
        candidate
        for candidate in candidates
        if candidate.semantic_name == candidate.semantic_parent
        and candidate.metadata.get("root_index") is not None
        and candidate.metadata.get("parent_candidate_key") is None
    ]
    groups: dict[str, list[MaskCandidate]] = {}
    for candidate in candidates:
        key = candidate_root_key(candidate)
        if key is not None:
            groups.setdefault(key, []).append(candidate)

    if config.mode == "all":
        return RootRoutingResult(
            tuple(candidates),
            {
                "algorithm": "hpid-physical-first-root-router-v2",
                "mode": config.mode,
                "root_proposal_count": len(roots),
                "selected_root_count": len(roots),
                "selected_candidate_count": len(candidates),
                "rejected_candidate_count": 0,
                "ground_truth_used": False,
            },
        )

    rows: list[dict[str, object]] = []
    for root in roots:
        key = candidate_root_key(root)
        if key is None:
            continue
        score, metrics = _root_score(
            root,
            groups.get(key, []),
            image_shape,
            config,
            root_edge_strength,
        )
        rows.append(
            {
                "candidate": root,
                "root_key": key,
                "origin": _root_origin(root),
                "semantic_name": root.semantic_name,
                "routing_score": score,
                "metrics": metrics,
            }
        )
    rows = _propagate_isolated_profile_hypotheses(rows)
    if not rows:
        return RootRoutingResult(
            tuple(candidates),
            {
                "algorithm": "hpid-physical-first-root-router-v2",
                "mode": config.mode,
                "root_proposal_count": 0,
                "selected_root_count": 0,
                "selected_candidate_count": len(candidates),
                "rejected_candidate_count": 0,
                "ground_truth_used": False,
            },
        )

    physical_groups = _cluster_physical_hypotheses(
        rows,
        image_shape=image_shape,
        config=config,
    )
    physical_group_rows: list[dict[str, object]] = []
    for physical_group in physical_groups:
        semantic_winner, semantic_ranking, semantic_margin = _semantic_arbitration(
            physical_group
        )
        physical_score = max(
            float(dict(row["metrics"])["physical_salience_score"])
            for row in physical_group
        )
        unique_semantics = len({str(row["semantic_name"]) for row in physical_group})
        unique_origins = len({str(row["origin"]) for row in physical_group})
        consensus_support = float(
            np.clip(
                0.75 * unique_semantics / 4.0 + 0.25 * unique_origins / 2.0,
                0.0,
                1.0,
            )
        )
        proposal_priority = max(
            float(dict(row["metrics"])["global_asset_proposal_priority"])
            for row in physical_group
        )
        group_score = (
            0.86 * physical_score
            + 0.06 * consensus_support
            + 0.08 * proposal_priority
        )
        group_id = str(physical_group[0]["physical_group_id"])
        for row in physical_group:
            row["physical_group_score"] = float(group_score)
            row["physical_consensus_support"] = consensus_support
            row["semantic_winner"] = str(row["root_key"]) == str(
                semantic_winner["root_key"]
            )
        physical_group_rows.append(
            {
                "physical_group_id": group_id,
                "rows": physical_group,
                "winner": semantic_winner,
                "semantic_ranking": semantic_ranking,
                "semantic_margin": float(semantic_margin),
                "physical_score": float(physical_score),
                "consensus_support": consensus_support,
                "global_asset_proposal_priority": proposal_priority,
                "group_score": float(group_score),
                "semantic_names": sorted(
                    {str(row["semantic_name"]) for row in physical_group}
                ),
            }
        )

    if config.mode == "scene":
        return _route_scene_groups(
            candidates,
            rows,
            physical_group_rows,
            image_shape=image_shape,
            config=config,
        )

    point_routing: dict[str, object] = {
        "requested": config.target_point_xy is not None,
        "target_point_xy": (
            list(config.target_point_xy) if config.target_point_xy is not None else None
        ),
        "status": "not_requested",
        "ground_truth_used": False,
    }
    seed_group: dict[str, object]
    salience_tie_candidate_count = 1
    prompt_owned_groups = [
        group
        for group in physical_group_rows
        if any(
            dict(row["candidate"].metadata).get("root_query_mode")
            == "user_asset_prompt"
            for row in group["rows"]
        )
    ]
    prompt_ownership_enforced = bool(
        config.target_point_xy is None and prompt_owned_groups
    )
    prompt_fragment_consensus: dict[str, object] = {
        "algorithm": "direct-prompt-fragment-consensus-v1",
        "candidate_group_count": 0,
        "selected_physical_group_id": None,
        "minimum_support_count": 2,
        "rows": [],
        "ground_truth_used": False,
    }
    if config.target_point_xy is None:
        consensus_group, prompt_fragment_consensus = _select_prompt_consensus_group(
            prompt_owned_groups
        )
        if consensus_group is not None:
            seed_group = consensus_group
            salience_tie_candidate_count = 1
        else:
            seed_group, salience_tie_candidate_count = _select_salient_group(
                prompt_owned_groups if prompt_owned_groups else physical_group_rows
            )
    else:
        point_x, point_y = config.target_point_xy
        height, width = image_shape
        if not (0.0 <= point_x < width and 0.0 <= point_y < height):
            raise ValueError(
                f"target point {(point_x, point_y)} is outside image bounds "
                f"{(width, height)}"
            )
        pixel_x = min(width - 1, max(0, round(point_x)))
        pixel_y = min(height - 1, max(0, round(point_y)))
        diagonal = max(1.0, float(np.hypot(height, width)))
        point_rows: list[tuple[bool, float, float, dict[str, object]]] = []
        for group in physical_group_rows:
            distances: list[float] = []
            contains = False
            for row in group["rows"]:
                candidate = row["candidate"]
                assert isinstance(candidate, MaskCandidate)
                if bool(candidate.mask[pixel_y, pixel_x]):
                    contains = True
                    distances.append(0.0)
                    continue
                distance = cv2.distanceTransform(
                    (~candidate.mask).astype(np.uint8), cv2.DIST_L2, 3
                )
                distances.append(float(distance[pixel_y, pixel_x]) / diagonal)
            minimum_distance = min(distances, default=np.inf)
            point_rows.append(
                (
                    contains,
                    minimum_distance,
                    float(group["group_score"]),
                    group,
                )
            )
        containing = [row for row in point_rows if row[0]]
        if containing:
            seed_group, salience_tie_candidate_count = _select_salient_group(
                [row[3] for row in containing]
            )
            point_routing["status"] = "selected_containing_root"
            point_routing["selected_distance_ratio"] = 0.0
        else:
            nearest = min(point_rows, key=lambda row: (row[1], -row[2]))
            if nearest[1] <= config.maximum_target_point_distance_ratio:
                seed_group = nearest[3]
                point_routing["status"] = "selected_nearest_root"
                point_routing["selected_distance_ratio"] = float(nearest[1])
            else:
                seed_group, salience_tie_candidate_count = _select_salient_group(
                    physical_group_rows
                )
                point_routing["status"] = "no_root_near_point_salience_fallback"
                point_routing["nearest_distance_ratio"] = float(nearest[1])
    seed = seed_group["winner"]
    assert isinstance(seed, dict)
    # Profile evidence may refine geometry inside the selected physical group,
    # but it must not promote a larger contextual host from another group. A
    # common failure is a headset crop being replaced by the person wearing it
    # merely because the person mask contains the headset mask.
    seed = _resolve_profile_geometry(seed, list(seed_group["rows"]))
    seed_candidate = seed["candidate"]
    assert isinstance(seed_candidate, MaskCandidate)
    selected_rows = [seed]
    selection_roles = {str(seed["root_key"]): "primary"}
    attachment_rejections: dict[str, str] = {}
    selected_origins = {str(seed["origin"])}
    height, width = image_shape
    diagonal = max(1.0, float(np.hypot(height, width)))
    for origin in sorted({str(row["origin"]) for row in rows} - selected_origins):
        matches: list[tuple[float, dict[str, object]]] = []
        for row in rows:
            if str(row["origin"]) != origin:
                continue
            candidate = row["candidate"]
            assert isinstance(candidate, MaskCandidate)
            if candidate.semantic_name != seed_candidate.semantic_name:
                continue
            overlap = mask_iou(candidate.mask, seed_candidate.mask)
            containment = _mask_containment(candidate.mask, seed_candidate.mask)
            candidate_x, candidate_y, _, _ = _geometry(candidate.mask)
            seed_x, seed_y, _, _ = _geometry(seed_candidate.mask)
            centroid_ratio = float(
                np.hypot(candidate_x - seed_x, candidate_y - seed_y) / diagonal
            )
            if not (
                overlap >= config.cross_source_match_iou
                or containment >= config.cross_source_match_containment
            ):
                continue
            if centroid_ratio > config.maximum_cross_source_centroid_ratio:
                continue
            affinity = max(overlap, 0.90 * containment) - 0.10 * centroid_ratio
            matches.append((float(affinity), row))
        if matches:
            matched = max(matches, key=lambda item: item[0])[1]
            selected_rows.append(matched)
            selection_roles[str(matched["root_key"])] = "cross_source_match"

    # Physical clustering deliberately refuses large crop-to-object area jumps,
    # while cross-source matching is allowed to bridge them when an independent
    # detector confirms overlap, containment, and centroid agreement. Re-run the
    # profile geometry resolver over only that audited evidence. This recovers a
    # complete bottle from a pump-only crop without reopening the unsafe pool of
    # unrelated same-domain hosts (for example, a person containing a headset).
    geometry_rows_by_key = {
        str(row["root_key"]): row for row in list(seed_group["rows"])
    }
    geometry_rows_by_key.update(
        {str(row["root_key"]): row for row in selected_rows}
    )
    resolved_seed = _resolve_profile_geometry(seed, list(geometry_rows_by_key.values()))
    resolved_seed_key = str(resolved_seed["root_key"])
    if resolved_seed_key != str(seed["root_key"]):
        selection_roles[str(seed["root_key"])] = "profile_geometry_evidence"
        selection_roles[resolved_seed_key] = "primary"
        selected_rows_by_key = {
            str(row["root_key"]): row for row in selected_rows
        }
        selected_rows_by_key[resolved_seed_key] = resolved_seed
        selected_rows = list(selected_rows_by_key.values())
        seed = resolved_seed
        seed_candidate = seed["candidate"]
        assert isinstance(seed_candidate, MaskCandidate)
        resolved_group_id = str(seed.get("physical_group_id", ""))
        seed_group = next(
            (
                group
                for group in physical_group_rows
                if str(group["physical_group_id"]) == resolved_group_id
            ),
            seed_group,
        )

    if config.include_attached_roots:
        seed_area = max(1, int(np.count_nonzero(seed_candidate.mask)))
        seed_distance = cv2.distanceTransform(
            (~seed_candidate.mask).astype(np.uint8), cv2.DIST_L2, 3
        )
        maximum_distance = max(
            2.0,
            min(image_shape) * config.maximum_attached_distance_ratio,
        )
        attached_by_semantic: dict[str, list[tuple[float, dict[str, object]]]] = {}
        for physical_group_row in physical_group_rows:
            if str(physical_group_row["physical_group_id"]) == str(
                seed_group["physical_group_id"]
            ):
                continue
            row = physical_group_row["winner"]
            assert isinstance(row, dict)
            root_key = str(row["root_key"])
            if root_key in selection_roles:
                continue
            candidate = row["candidate"]
            assert isinstance(candidate, MaskCandidate)
            if candidate.semantic_name == seed_candidate.semantic_name:
                continue
            candidate_area = max(1, int(np.count_nonzero(candidate.mask)))
            area_ratio = candidate_area / seed_area
            if area_ratio > config.maximum_attached_area_ratio:
                attachment_rejections[root_key] = "area_ratio"
                continue
            metrics = row["metrics"]
            assert isinstance(metrics, dict)
            if float(metrics["coherence"]) < config.minimum_attached_coherence:
                attachment_rejections[root_key] = "low_coherence"
                continue
            domain_evidence = metrics.get("domain_evidence_score")
            domain_contrast = metrics.get("domain_evidence_contrast")
            child_semantic_count = int(metrics["child_semantic_count"])
            weak_domain = domain_evidence is not None and (
                float(domain_evidence) < config.minimum_attached_domain_evidence
                or float(domain_contrast or 0.0)
                < config.minimum_attached_domain_contrast
            )
            border_like = (
                float(metrics["border_fraction"])
                > config.maximum_weak_attached_border_fraction
                or int(metrics["touched_sides"])
                > config.maximum_weak_attached_touched_sides
            )
            if weak_domain and border_like:
                attachment_rejections[root_key] = "weak_border_domain"
                continue
            if child_semantic_count == 0 and weak_domain:
                attachment_rejections[root_key] = "weak_domain_without_parts"
                continue
            if (
                child_semantic_count == 0
                and border_like
                and (domain_evidence is None or float(domain_evidence) < 0.08)
            ):
                attachment_rejections[root_key] = "border_scene_surface"
                continue
            intersection = int(np.count_nonzero(candidate.mask & seed_candidate.mask))
            seed_overlap = intersection / candidate_area
            if seed_overlap > config.maximum_attached_seed_containment:
                attachment_rejections[root_key] = "contained_alternative"
                continue
            minimum_distance = float(seed_distance[candidate.mask].min(initial=np.inf))
            if (
                seed_overlap < config.minimum_attached_seed_overlap
                and minimum_distance > maximum_distance
            ):
                attachment_rejections[root_key] = "too_distant"
                continue
            attachment_score = (
                seed_overlap
                + 0.20 * float(row["routing_score"])
                - 0.10 * area_ratio
                + 0.08 * max(0.0, 1.0 - minimum_distance / maximum_distance)
            )
            attached_by_semantic.setdefault(candidate.semantic_name, []).append(
                (float(attachment_score), row)
            )
        for matches in attached_by_semantic.values():
            attached = max(matches, key=lambda item: item[0])[1]
            selected_rows.append(attached)
            selection_roles[str(attached["root_key"])] = "attached_root"

    selected_keys = {str(row["root_key"]) for row in selected_rows}
    selected_root_candidates = {
        str(row["root_key"]): row["candidate"] for row in selected_rows
    }
    canonical_root_key = str(seed["root_key"])
    canonical_root = seed["candidate"]
    assert isinstance(canonical_root, MaskCandidate)
    selected_semantic = seed_candidate.semantic_name
    routed: list[MaskCandidate] = []
    for candidate in candidates:
        key = candidate_root_key(candidate)
        if key is not None:
            if key in selected_keys:
                role = selection_roles.get(key)
                if role in {"cross_source_match", "profile_geometry_evidence"}:
                    if not (
                        candidate.semantic_name == candidate.semantic_parent
                        and candidate.metadata.get("parent_candidate_key") is None
                    ):
                        routed.append(
                            _remap_candidate_to_canonical_root(
                                candidate,
                                evidence_root_key=key,
                                canonical_root_key=canonical_root_key,
                                canonical_root=canonical_root,
                            )
                        )
                    continue
                if (
                    candidate.semantic_name == candidate.semantic_parent
                    and candidate.metadata.get("parent_candidate_key") is None
                ):
                    routed.append(canonical_root if role == "primary" else candidate)
                else:
                    root_for_key = selected_root_candidates.get(key, canonical_root)
                    assert isinstance(root_for_key, MaskCandidate)
                    routed.append(
                        _remap_candidate_to_canonical_root(
                            candidate,
                            evidence_root_key=key,
                            canonical_root_key=key,
                            canonical_root=(
                                canonical_root if role == "primary" else root_for_key
                            ),
                        )
                    )
            continue
        if (
            candidate.semantic_parent == selected_semantic
            or candidate.semantic_name == selected_semantic
            or candidate.semantic_name.startswith(f"{selected_semantic}_")
        ):
            routed.append(candidate)

    public_rows = [
        {
            "root_key": str(row["root_key"]),
            "origin": str(row["origin"]),
            "semantic_name": str(row["semantic_name"]),
            "routing_score": float(row["routing_score"]),
            "physical_group_id": str(row.get("physical_group_id", "")),
            "physical_group_score": float(row.get("physical_group_score", 0.0)),
            "physical_consensus_support": float(
                row.get("physical_consensus_support", 0.0)
            ),
            "semantic_arbitration_score": float(
                row.get("semantic_arbitration_score", 0.0)
            ),
            "semantic_category_score": float(row.get("semantic_category_score", 0.0)),
            "geometry_representative_score": float(
                row.get("geometry_representative_score", 0.0)
            ),
            "physical_group_relative_coverage": float(
                row.get("physical_group_relative_coverage", 0.0)
            ),
            "semantic_arbitration_margin": next(
                (
                    float(group["semantic_margin"])
                    for group in physical_group_rows
                    if str(group["physical_group_id"])
                    == str(row.get("physical_group_id", ""))
                ),
                0.0,
            ),
            "domain_evidence_rank": float(row.get("domain_evidence_rank", 0.5)),
            "root_label_specificity": float(row.get("root_label_specificity", 0.0)),
            "part_profile_specificity": float(row.get("part_profile_specificity", 0.0)),
            "selected_part_profile": _profile_name(row["candidate"]),
            "profile_evidence_root_key": row.get("profile_evidence_root_key"),
            "profile_hypothesis_score": float(row.get("profile_hypothesis_score", 0.0)),
            "profile_hypothesis_margin": float(
                row.get("profile_hypothesis_margin", 0.0)
            ),
            "profile_classifier": dict(row["candidate"].metadata).get(
                "profile_classifier"
            ),
            "root_model_label": dict(row["candidate"].metadata).get(
                "root_model_label"
            ),
            "root_query_mode": dict(row["candidate"].metadata).get(
                "root_query_mode"
            ),
            "semantic_winner": bool(row.get("semantic_winner", False)),
            **dict(row["metrics"]),
            "selected": str(row["root_key"]) in selected_keys,
            "selection_role": selection_roles.get(str(row["root_key"])),
            "attachment_rejection_reason": attachment_rejections.get(
                str(row["root_key"])
            ),
        }
        for row in sorted(
            rows,
            key=lambda row: (
                float(row.get("physical_group_score", 0.0)),
                float(row.get("semantic_arbitration_score", 0.0)),
            ),
            reverse=True,
        )
    ]
    public_groups = [
        {
            "physical_group_id": str(group["physical_group_id"]),
            "root_keys": sorted(str(row["root_key"]) for row in group["rows"]),
            "semantic_names": list(group["semantic_names"]),
            "selected_semantic": str(dict(group["winner"])["semantic_name"]),
            "selected_root_key": str(dict(group["winner"])["root_key"]),
            "physical_score": float(group["physical_score"]),
            "consensus_support": float(group["consensus_support"]),
            "group_score": float(group["group_score"]),
            "semantic_margin": float(group["semantic_margin"]),
            "selected_as_primary": (
                str(group["physical_group_id"]) == str(seed_group["physical_group_id"])
            ),
        }
        for group in sorted(
            physical_group_rows,
            key=lambda item: float(item["group_score"]),
            reverse=True,
        )
    ]
    return RootRoutingResult(
        tuple(routed),
        {
            "algorithm": "hpid-physical-first-root-router-v3",
            "mode": config.mode,
            "root_proposal_count": len(roots),
            "selected_root_count": sum(
                candidate.semantic_name == candidate.semantic_parent
                and candidate.metadata.get("parent_candidate_key") is None
                for candidate in routed
            ),
            "selected_evidence_root_count": len(selected_rows),
            "selected_semantic": selected_semantic,
            "selected_physical_group_id": str(seed_group["physical_group_id"]),
            "canonical_root_key": canonical_root_key,
            "salience_tie_candidate_count": salience_tie_candidate_count,
            "selected_semantic_margin": float(seed_group["semantic_margin"]),
            "selected_root_keys": sorted(selected_keys),
            "prompt_owned_group_count": len(prompt_owned_groups),
            "prompt_ownership_enforced": prompt_ownership_enforced,
            "prompt_fragment_consensus": prompt_fragment_consensus,
            "selected_candidate_count": len(routed),
            "rejected_candidate_count": len(candidates) - len(routed),
            "root_scores": public_rows,
            "physical_groups": public_groups,
            "target_point_routing": point_routing,
            "ground_truth_used": False,
        },
    )
