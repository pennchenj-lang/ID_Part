from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .fusion import MaskCandidate
from .prompt_bank import PartPrompt, PromptBank
from .taxonomy import Taxonomy


@dataclass(frozen=True)
class RelationalAppearanceConfig:
    """Controls image-only detail proposals derived from stable anchor parts."""

    minimum_anchor_area_px: int = 20
    minimum_candidate_area_px: int = 10
    parent_support_dilation_ratio: float = 0.006
    response_quantile: float = 0.68
    source_reliability: float = 0.68
    repetitive_detail_minimum_instances: int = 4
    repetitive_detail_maximum_instances: int = 32


@dataclass(frozen=True)
class RelationalCandidateGeneration:
    candidates: tuple[MaskCandidate, ...]
    diagnostics: dict[str, object]


def _mask_box(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("cannot compute a box for an empty mask")
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def _components(mask: np.ndarray, minimum_area: int) -> list[np.ndarray]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    output: list[np.ndarray] = []
    for component_id in range(1, count):
        if int(stats[component_id, cv2.CC_STAT_AREA]) < minimum_area:
            continue
        output.append(labels == component_id)
    output.sort(key=np.count_nonzero, reverse=True)
    return output


def _descendants(domain_name: str, parts: tuple[PartPrompt, ...]) -> dict[str, set[str]]:
    children: dict[str, list[str]] = {}
    for part in parts:
        parent = part.semantic_parent or domain_name
        children.setdefault(parent, []).append(part.semantic_name)

    cache: dict[str, set[str]] = {}

    def collect(name: str) -> set[str]:
        if name in cache:
            return cache[name]
        values = {name}
        for child in children.get(name, []):
            values.update(collect(child))
        cache[name] = values
        return values

    collect(domain_name)
    for part in parts:
        collect(part.semantic_name)
    return cache


def _semantic_union(
    labels: np.ndarray,
    taxonomy: Taxonomy,
    semantic_names: set[str],
) -> np.ndarray:
    name_to_id = {name: index for index, name in enumerate(taxonomy.fine_names)}
    class_ids = [name_to_id[name] for name in semantic_names if name in name_to_id]
    if not class_ids:
        return np.zeros(labels.shape, dtype=bool)
    return np.isin(labels, class_ids)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def _repetitive_component_rows(
    gray: np.ndarray,
    root_mask: np.ndarray,
    existing_mask: np.ndarray,
    *,
    polarity: str,
    quantile: float,
) -> list[dict[str, object]]:
    root_area = int(np.count_nonzero(root_mask))
    _, xs = np.nonzero(root_mask)
    if root_area == 0 or not len(xs):
        return []
    x0, y0, x1, y1 = _mask_box(root_mask)
    root_width = max(1, x1 - x0)
    root_height = max(1, y1 - y0)
    interior_distance = max(2.0, min(root_width, root_height) * 0.012)
    interior = (
        cv2.distanceTransform(root_mask.astype(np.uint8), cv2.DIST_L2, 5)
        >= interior_distance
    )
    values = gray[interior]
    if not len(values):
        return []
    threshold = float(np.quantile(values, quantile))
    if polarity == "dark":
        active = (gray <= threshold) & interior
    elif polarity == "light":
        active = (gray >= threshold) & interior
    else:
        raise ValueError(f"unsupported repetitive-detail polarity: {polarity!r}")
    active &= ~_dilate(existing_mask, 2)
    if min(root_width, root_height) >= 80:
        active = cv2.morphologyEx(
            active.astype(np.uint8),
            cv2.MORPH_OPEN,
            np.ones((2, 2), dtype=np.uint8),
        ).astype(bool)

    minimum_area = max(12, round(root_area * 0.0010))
    maximum_area = max(24, round(root_area * 0.0180))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        active.astype(np.uint8), 8
    )
    rows: list[dict[str, object]] = []
    for component_id in range(1, count):
        x, y, width, height, area = (
            int(value) for value in stats[component_id]
        )
        if not minimum_area <= area <= maximum_area:
            continue
        if (
            width < 3
            or height < 3
            or width > root_width * 0.22
            or height > root_height * 0.16
        ):
            continue
        aspect = width / max(1, height)
        fill_ratio = area / max(1, width * height)
        if not 0.30 <= aspect <= 3.70 or fill_ratio < 0.42:
            continue
        component = labels == component_id
        contours, _ = cv2.findContours(
            component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        perimeter = sum(cv2.arcLength(contour, True) for contour in contours)
        circularity = 4.0 * math.pi * area / max(1.0, perimeter * perimeter)
        if circularity < 0.28:
            continue
        ring = (
            cv2.dilate(component.astype(np.uint8), np.ones((7, 7), np.uint8))
            .astype(bool)
            & interior
            & ~component
        )
        if not ring.any():
            continue
        contrast = abs(
            float(gray[component].mean()) - float(np.median(gray[ring]))
        ) / 255.0
        if contrast < 0.055:
            continue
        rows.append(
            {
                "mask": component,
                "area_px": area,
                "bbox_xyxy": [x, y, x + width, y + height],
                "centroid_xy": [
                    float(centroids[component_id, 0]),
                    float(centroids[component_id, 1]),
                ],
                "contrast": contrast,
                "fill_ratio": fill_ratio,
                "circularity": circularity,
                "polarity": polarity,
                "quantile": quantile,
                "threshold": threshold,
            }
        )
    if len(rows) < 2:
        return rows
    median_area = float(np.median([float(row["area_px"]) for row in rows]))
    return [
        row
        for row in rows
        if median_area * 0.25 <= float(row["area_px"]) <= median_area * 4.0
    ]


def _repetitive_detail_candidates(
    image: Image.Image,
    preliminary_labels: np.ndarray,
    preliminary_taxonomy: Taxonomy,
    roots: Sequence[MaskCandidate],
    config: RelationalAppearanceConfig,
) -> tuple[list[MaskCandidate], list[dict[str, object]]]:
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    existing_buttons = _semantic_union(
        preliminary_labels, preliminary_taxonomy, {"device_button"}
    )
    output: list[MaskCandidate] = []
    diagnostics: list[dict[str, object]] = []
    supported_profiles = {"controls", "game_controller"}
    for root_index, root in enumerate(roots, start=1):
        profile = str(root.metadata.get("selected_part_profile", ""))
        if root.semantic_name != "device" or profile not in supported_profiles:
            continue
        root_mask = root.mask.astype(bool)
        root_area = int(np.count_nonzero(root_mask))
        if root_area < 160:
            continue
        trials: list[tuple[float, list[dict[str, object]]]] = []
        for polarity, quantiles in (
            ("dark", (0.10, 0.14, 0.18, 0.22)),
            ("light", (0.78, 0.82, 0.86, 0.90)),
        ):
            for quantile in quantiles:
                rows = _repetitive_component_rows(
                    gray,
                    root_mask,
                    existing_buttons,
                    polarity=polarity,
                    quantile=quantile,
                )
                if len(rows) < config.repetitive_detail_minimum_instances:
                    continue
                contrasts = [float(row["contrast"]) for row in rows]
                areas = [float(row["area_px"]) for row in rows]
                centers = np.asarray([row["centroid_xy"] for row in rows], dtype=float)
                spread = max(
                    float(np.ptp(centers[:, 0])), float(np.ptp(centers[:, 1]))
                ) / max(1.0, math.sqrt(float(np.median(areas))))
                area_dispersion = float(np.std(areas) / max(1.0, np.mean(areas)))
                excess_penalty = max(0.0, (len(rows) - 48) / 48.0)
                trial_score = (
                    float(np.median(contrasts))
                    + 0.15 * min(1.0, len(rows) / 24.0)
                    + 0.10 * min(1.0, spread / 10.0)
                    - 0.08 * min(1.0, area_dispersion)
                    - 0.25 * excess_penalty
                )
                trials.append((trial_score, rows))
        if not trials:
            diagnostics.append(
                {
                    "root_index": root_index,
                    "profile": profile,
                    "status": "no_repeated_cohort",
                    "accepted_count": 0,
                }
            )
            continue
        trial_score, selected = max(trials, key=lambda item: item[0])
        selected.sort(
            key=lambda row: (
                float(row["centroid_xy"][1]),
                float(row["centroid_xy"][0]),
            )
        )
        selected = selected[: config.repetitive_detail_maximum_instances]
        root_key = str(root.metadata.get("candidate_key", f"device-root:{root_index}"))
        for detail_index, row in enumerate(selected, start=1):
            output.append(
                MaskCandidate(
                    semantic_name="device_button",
                    semantic_parent="device_body",
                    mask=np.asarray(row["mask"], dtype=bool),
                    score=float(np.clip(0.48 + 0.65 * float(row["contrast"]), 0.0, 0.94)),
                    source="hpid-repetitive-physical-detail-v1",
                    prompt="repeated compact control region",
                    source_reliability=0.76,
                    metadata={
                        "source_family": "hpid-repetitive-physical-detail-v1",
                        "candidate_key": f"{root_key}:repeated-detail:{detail_index:02d}",
                        "parent_candidate_key": root_key,
                        "query_parent_semantic": "device_body",
                        "assembly_parent_semantic": "device",
                        "assembly_parent_candidate_key": root_key,
                        "repetitive_physical_detail": True,
                        "selected_part_profile": profile,
                        "maximum_instances": config.repetitive_detail_maximum_instances,
                        "ground_truth_used": False,
                        **{key: value for key, value in row.items() if key != "mask"},
                    },
                )
            )
        diagnostics.append(
            {
                "root_index": root_index,
                "root_key": root_key,
                "profile": profile,
                "status": "accepted",
                "trial_score": trial_score,
                "polarity": selected[0]["polarity"],
                "quantile": selected[0]["quantile"],
                "accepted_count": len(selected),
            }
        )
    return output, diagnostics


def _appearance_response(
    lab: np.ndarray,
    search: np.ndarray,
    polarity: str,
    *,
    baseline_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    lightness = lab[..., 0].astype(np.float32) / 255.0
    values = lightness[baseline_mask if baseline_mask is not None else search]
    if not len(values):
        return np.zeros(search.shape, dtype=np.float32), 0.0
    baseline = float(np.median(values))
    if polarity == "dark":
        response = baseline - lightness
    elif polarity == "light":
        response = lightness - baseline
    else:
        response = np.abs(lightness - baseline)
    response[~search] = 0.0
    return np.maximum(response, 0.0), baseline


def _active_appearance(
    response: np.ndarray,
    search: np.ndarray,
    *,
    minimum_contrast: float,
    quantile: float,
    close_width: int,
) -> tuple[np.ndarray, float]:
    values = response[search]
    positive = values[values > 0]
    if not len(positive):
        return np.zeros(search.shape, dtype=bool), minimum_contrast
    threshold = max(minimum_contrast, float(np.quantile(positive, quantile)))
    active = ((response >= threshold) & search).astype(np.uint8)
    width = max(3, close_width)
    if width % 2 == 0:
        width += 1
    active = cv2.morphologyEx(
        active,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, 3)),
    )
    return active.astype(bool), threshold


def _above_candidate(
    lab: np.ndarray,
    anchor: np.ndarray,
    parent_support: np.ndarray,
    part: PartPrompt,
    config: RelationalAppearanceConfig,
) -> tuple[np.ndarray | None, dict[str, float | list[int]]]:
    _, width = anchor.shape
    x0, y0, x1, y1 = _mask_box(anchor)
    anchor_width = max(1, x1 - x0)
    anchor_height = max(1, y1 - y0)
    scale = part.appearance_search_scale
    margin_x = round(anchor_width * 0.22 * scale)
    search_top = max(0, y0 - round(anchor_height * 0.90 * scale))
    search_bottom = max(search_top + 1, y0 - round(anchor_height * 0.20))
    search_left = max(0, x0 - margin_x)
    search_right = min(width, x1 + margin_x)
    search = np.zeros(anchor.shape, dtype=bool)
    search[search_top:search_bottom, search_left:search_right] = True
    search &= parent_support
    response, baseline = _appearance_response(
        lab, search, part.appearance_polarity
    )
    active, threshold = _active_appearance(
        response,
        search,
        minimum_contrast=part.appearance_minimum_contrast,
        quantile=config.response_quantile,
        close_width=round(anchor_width * 0.10),
    )
    anchor_area = max(1, int(np.count_nonzero(anchor)))
    expected_x = (x0 + x1) / 2.0
    expected_y = y0 - anchor_height * 0.48
    scored: list[tuple[float, np.ndarray]] = []
    for component in _components(active, config.minimum_candidate_area_px):
        area = int(np.count_nonzero(component))
        fraction = area / anchor_area
        if not 0.015 <= fraction <= 0.65:
            continue
        bx0, by0, bx1, by1 = _mask_box(component)
        component_width = max(1, bx1 - bx0)
        component_height = max(1, by1 - by0)
        aspect = component_width / component_height
        if aspect < 1.15:
            continue
        cx = (bx0 + bx1) / 2.0
        cy = (by0 + by1) / 2.0
        dx = abs(cx - expected_x) / anchor_width
        dy = abs(cy - expected_y) / anchor_height
        geometry = float(np.exp(-(1.3 * dx + 0.9 * dy)))
        mean_response = float(response[component].mean())
        contrast_score = min(1.0, mean_response / max(0.01, threshold * 1.6))
        shape_score = min(1.0, aspect / 3.0)
        score = 0.46 * contrast_score + 0.34 * geometry + 0.20 * shape_score
        scored.append((score, component))
    if not scored:
        return None, {
            "baseline_lightness": baseline,
            "activation_threshold": threshold,
            "search_box_xyxy": [
                search_left,
                search_top,
                search_right,
                search_bottom,
            ],
        }
    score, candidate = max(scored, key=lambda item: item[0])
    return candidate, {
        "appearance_score": score,
        "baseline_lightness": baseline,
        "activation_threshold": threshold,
        "search_box_xyxy": [
            search_left,
            search_top,
            search_right,
            search_bottom,
        ],
    }


def _upper_boundary_candidate(
    lab: np.ndarray,
    anchor: np.ndarray,
    parent_support: np.ndarray,
    part: PartPrompt,
    config: RelationalAppearanceConfig,
) -> tuple[np.ndarray | None, dict[str, float | list[int]]]:
    x0, y0, x1, y1 = _mask_box(anchor)
    anchor_width = max(1, x1 - x0)
    anchor_height = max(1, y1 - y0)
    radius = max(2, round(anchor_height * 0.14 * part.appearance_search_scale))
    search = np.zeros(anchor.shape, dtype=bool)
    for x in range(x0, x1):
        ys = np.flatnonzero(anchor[:, x])
        if not len(ys):
            continue
        top = int(ys.min())
        search[max(0, top - radius) : min(anchor.shape[0], top + radius + 1), x] = (
            True
        )
    search = _dilate(search, max(1, round(radius * 0.35))) & parent_support
    baseline_support = (
        _dilate(anchor, max(radius * 3, 4))
        & parent_support
        & ~_dilate(anchor, radius)
    )
    response, baseline = _appearance_response(
        lab,
        search,
        part.appearance_polarity,
        baseline_mask=baseline_support if baseline_support.any() else None,
    )
    active, threshold = _active_appearance(
        response,
        search,
        minimum_contrast=part.appearance_minimum_contrast,
        quantile=max(0.55, config.response_quantile - 0.08),
        close_width=round(anchor_width * 0.08),
    )
    adjacency = _dilate(anchor, radius) & active
    scored: list[tuple[float, np.ndarray]] = []
    anchor_area = max(1, int(np.count_nonzero(anchor)))
    for component in _components(adjacency, config.minimum_candidate_area_px):
        area = int(np.count_nonzero(component))
        fraction = area / anchor_area
        if not 0.01 <= fraction <= 0.70:
            continue
        bx0, by0, bx1, by1 = _mask_box(component)
        aspect = max(1, bx1 - bx0) / max(1, by1 - by0)
        mean_response = float(response[component].mean())
        contrast_score = min(1.0, mean_response / max(0.01, threshold * 1.5))
        width_score = min(1.0, (bx1 - bx0) / max(1, anchor_width * 0.65))
        score = 0.58 * contrast_score + 0.27 * width_score + 0.15 * min(
            1.0, aspect / 2.5
        )
        scored.append((score, component))
    if not scored:
        return None, {
            "baseline_lightness": baseline,
            "activation_threshold": threshold,
            "search_box_xyxy": [x0, max(0, y0 - radius), x1, min(anchor.shape[0], y1)],
        }
    score, candidate = max(scored, key=lambda item: item[0])
    return candidate, {
        "appearance_score": score,
        "baseline_lightness": baseline,
        "activation_threshold": threshold,
        "search_box_xyxy": [x0, max(0, y0 - radius), x1, min(anchor.shape[0], y1)],
    }


def propose_relational_candidates(
    image: Image.Image,
    preliminary_labels: np.ndarray,
    preliminary_taxonomy: Taxonomy,
    prompt_bank: PromptBank,
    *,
    roots: Sequence[MaskCandidate] = (),
    config: RelationalAppearanceConfig | None = None,
) -> RelationalCandidateGeneration:
    """Generate fine-detail masks from anchor geometry and local appearance.

    The function receives only an image and a first-pass prediction. Ground-truth
    labels are deliberately absent from the API.
    """
    active_config = config or RelationalAppearanceConfig()
    if preliminary_labels.shape != (image.height, image.width):
        raise ValueError("preliminary labels must match the source image")
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    output: list[MaskCandidate] = []
    rule_diagnostics: list[dict[str, object]] = []
    image_radius = max(
        1,
        round(
            min(image.height, image.width)
            * active_config.parent_support_dilation_ratio
        ),
    )

    for domain in prompt_bank.domains:
        descendants = _descendants(domain.name, domain.parts)
        for part in domain.parts:
            if part.appearance_anchor is None:
                continue
            anchor_mask = _semantic_union(
                preliminary_labels,
                preliminary_taxonomy,
                {part.appearance_anchor},
            )
            anchors = _components(
                anchor_mask, active_config.minimum_anchor_area_px
            )[: part.maximum_instances]
            parent_semantic = part.semantic_parent or domain.name
            parent_support = _semantic_union(
                preliminary_labels,
                preliminary_taxonomy,
                descendants.get(parent_semantic, {parent_semantic}),
            )
            parent_support = _dilate(parent_support, image_radius)
            accepted = 0
            rejected = 0
            for anchor_index, anchor in enumerate(anchors, start=1):
                if part.appearance_relation == "above":
                    mask, proposal_diagnostics = _above_candidate(
                        lab, anchor, parent_support, part, active_config
                    )
                elif part.appearance_relation == "upper_boundary":
                    mask, proposal_diagnostics = _upper_boundary_candidate(
                        lab, anchor, parent_support, part, active_config
                    )
                else:
                    raise ValueError(
                        f"unsupported appearance relation: {part.appearance_relation!r}"
                    )
                if mask is None:
                    rejected += 1
                    continue
                area = int(np.count_nonzero(mask))
                parent_area = max(1, int(np.count_nonzero(parent_support)))
                parent_fraction = area / parent_area
                if not (
                    part.minimum_parent_fraction
                    <= parent_fraction
                    <= part.maximum_parent_fraction
                ):
                    rejected += 1
                    continue
                appearance_score = float(
                    proposal_diagnostics.get("appearance_score", 0.0)
                )
                score = float(np.clip(0.35 + 0.60 * appearance_score, 0.0, 0.95))
                candidate_key = (
                    f"relational:{domain.name}:{part.semantic_name}:{anchor_index:02d}"
                )
                output.append(
                    MaskCandidate(
                        semantic_name=part.semantic_name,
                        semantic_parent=parent_semantic,
                        mask=mask,
                        score=score,
                        source="hpid-relational-appearance-v1/anchor-refinement",
                        prompt=(
                            f"{part.appearance_relation} "
                            f"{part.appearance_anchor} ({part.appearance_polarity})"
                        ),
                        source_reliability=(
                            active_config.source_reliability * part.priority
                        ),
                        metadata={
                            "source_family": "hpid-relational-appearance-v1",
                            "candidate_key": candidate_key,
                            "parent_candidate_key": None,
                            "query_parent_semantic": parent_semantic,
                            "assembly_parent_semantic": (
                                part.assembly_parent or parent_semantic
                            ),
                            "assembly_parent_candidate_key": None,
                            "relational_appearance": True,
                            "appearance_anchor": part.appearance_anchor,
                            "appearance_relation": part.appearance_relation,
                            "appearance_polarity": part.appearance_polarity,
                            "anchor_index": anchor_index,
                            "anchor_box_xyxy": list(_mask_box(anchor)),
                            "parent_area_fraction": parent_fraction,
                            "ground_truth_used": False,
                            "maximum_instances": part.maximum_instances,
                            **proposal_diagnostics,
                        },
                    )
                )
                accepted += 1
            rule_diagnostics.append(
                {
                    "domain": domain.name,
                    "semantic_name": part.semantic_name,
                    "anchor_semantic": part.appearance_anchor,
                    "relation": part.appearance_relation,
                    "anchor_component_count": len(anchors),
                    "accepted_candidate_count": accepted,
                    "rejected_candidate_count": rejected,
                }
            )
    repetitive_candidates, repetitive_diagnostics = _repetitive_detail_candidates(
        image,
        preliminary_labels,
        preliminary_taxonomy,
        roots,
        active_config,
    )
    output.extend(repetitive_candidates)
    return RelationalCandidateGeneration(
        tuple(output),
        {
            "algorithm": "hpid-relational-appearance-v1",
            "ground_truth_used": False,
            "candidate_count": len(output),
            "rules": rule_diagnostics,
            "repetitive_physical_details": repetitive_diagnostics,
        },
    )
