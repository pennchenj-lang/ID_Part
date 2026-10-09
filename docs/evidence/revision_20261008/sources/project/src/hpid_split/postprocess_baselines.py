from __future__ import annotations

import hashlib
from dataclasses import dataclass

import cv2
import numpy as np

from .fusion import MaskCandidate


@dataclass(frozen=True)
class BaselineInstance:
    identity: str
    semantic_name: str
    semantic_parent: str
    mask: np.ndarray
    confidence: float


@dataclass(frozen=True)
class BaselinePrediction:
    method: str
    instances: tuple[BaselineInstance, ...]
    root_mask: np.ndarray
    diagnostics: dict[str, object]


def _candidate_strength(candidate: MaskCandidate) -> float:
    return float(
        np.clip(
            0.72 * float(candidate.score)
            + 0.28 * float(candidate.source_reliability),
            0.0,
            1.0,
        )
    )


def _mask_digest(mask: np.ndarray) -> str:
    packed = np.packbits(np.asarray(mask, dtype=np.uint8), axis=None)
    return hashlib.sha1(packed.tobytes()).hexdigest()[:10]


def _root_candidate(candidates: list[MaskCandidate]) -> MaskCandidate:
    roots = [
        candidate
        for candidate in candidates
        if candidate.semantic_name == candidate.semantic_parent
    ]
    if not roots:
        roots = list(candidates)
    return max(
        roots,
        key=lambda candidate: (
            _candidate_strength(candidate)
            * np.sqrt(max(1, int(np.count_nonzero(candidate.mask)))),
            candidate.semantic_name,
            candidate.source,
        ),
    )


def _non_root_candidates(
    candidates: list[MaskCandidate], root: MaskCandidate
) -> list[MaskCandidate]:
    return [
        candidate
        for candidate in candidates
        if candidate is not root
        and candidate.semantic_name != candidate.semantic_parent
        and int(np.count_nonzero(candidate.mask)) >= 6
    ]


def _overlap(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    intersection = int(np.count_nonzero(first & second))
    if not intersection:
        return 0.0, 0.0
    first_area = int(np.count_nonzero(first))
    second_area = int(np.count_nonzero(second))
    return (
        intersection / max(1, first_area + second_area - intersection),
        intersection / max(1, min(first_area, second_area)),
    )


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    y, x = np.nonzero(mask)
    if not len(x):
        return 0.0, 0.0
    return float(np.mean(x)), float(np.mean(y))


def _stable_instances(
    rows: list[tuple[str, str, np.ndarray, float]],
    *,
    root_mask: np.ndarray,
) -> tuple[BaselineInstance, ...]:
    foreground_x = np.nonzero(root_mask)[1]
    width = root_mask.shape[1]
    center_x = (
        float(np.median(foreground_x)) if len(foreground_x) else width / 2.0
    )
    dead_zone = max(2.0, width * 0.025)
    grouped: dict[str, list[tuple[str, np.ndarray, float, float, float]]] = {}
    for semantic, parent, mask, confidence in rows:
        binary = np.asarray(mask, dtype=bool)
        if not binary.any():
            continue
        cx, cy = _centroid(binary)
        side = "left" if cx < center_x - dead_zone else (
            "right" if cx > center_x + dead_zone else "center"
        )
        grouped.setdefault(semantic, []).append(
            (parent, binary, float(confidence), cx, cy)
        )

    side_order = {"left": 0, "center": 1, "right": 2}
    output: list[BaselineInstance] = []
    for semantic in sorted(grouped):
        ordered = []
        for parent, mask, confidence, cx, cy in grouped[semantic]:
            side = "left" if cx < center_x - dead_zone else (
                "right" if cx > center_x + dead_zone else "center"
            )
            ordered.append(
                (
                    side_order[side],
                    cy,
                    cx,
                    -int(np.count_nonzero(mask)),
                    _mask_digest(mask),
                    side,
                    parent,
                    mask,
                    confidence,
                )
            )
        ordered.sort(key=lambda row: row[:5])
        counters = {"left": 0, "center": 0, "right": 0}
        for row in ordered:
            side = row[5]
            counters[side] += 1
            output.append(
                BaselineInstance(
                    identity=f"{semantic}/{side}/{counters[side]:02d}",
                    semantic_name=semantic,
                    semantic_parent=row[6],
                    mask=row[7],
                    confidence=row[8],
                )
            )
    return tuple(output)


def _exclusive_rows(
    rows: list[tuple[str, str, np.ndarray, float]],
    *,
    root_semantic: str,
    root_parent: str,
    root_mask: np.ndarray,
    minimum_area: int = 6,
) -> list[tuple[str, str, np.ndarray, float]]:
    if rows:
        stack = np.stack([np.asarray(row[2], dtype=bool) for row in rows])
        strengths = np.asarray([row[3] for row in rows], dtype=np.float32)
        weighted = np.where(stack, strengths[:, None, None], -np.inf)
        winner = np.argmax(weighted, axis=0)
        foreground = stack.any(axis=0) & root_mask
    else:
        stack = np.zeros((0, *root_mask.shape), dtype=bool)
        winner = np.zeros(root_mask.shape, dtype=np.int32)
        foreground = np.zeros(root_mask.shape, dtype=bool)

    output: list[tuple[str, str, np.ndarray, float]] = []
    for index, row in enumerate(rows):
        owned = foreground & (winner == index)
        if int(np.count_nonzero(owned)) >= minimum_area:
            output.append((row[0], row[1], owned, row[3]))
    residual = root_mask & ~foreground
    if int(np.count_nonzero(residual)) >= minimum_area:
        output.append((root_semantic, root_parent, residual, 1.0))
    return output


def raw_proposals(candidates: list[MaskCandidate]) -> BaselinePrediction:
    root = _root_candidate(candidates)
    instances = tuple(
        BaselineInstance(
            identity=f"proposal/{index:04d}",
            semantic_name=candidate.semantic_name,
            semantic_parent=candidate.semantic_parent,
            mask=np.asarray(candidate.mask, dtype=bool),
            confidence=_candidate_strength(candidate),
        )
        for index, candidate in enumerate(candidates, start=1)
        if int(np.count_nonzero(candidate.mask)) >= 6
    )
    return BaselinePrediction(
        method="raw_proposals",
        instances=instances,
        root_mask=np.asarray(root.mask, dtype=bool),
        diagnostics={"input_candidate_count": len(candidates)},
    )


def greedy_nms_ownership(
    candidates: list[MaskCandidate],
    *,
    iou_threshold: float = 0.80,
    containment_threshold: float = 0.92,
) -> BaselinePrediction:
    root = _root_candidate(candidates)
    root_mask = np.asarray(root.mask, dtype=bool)
    ordered = sorted(
        _non_root_candidates(candidates, root),
        key=lambda candidate: (
            -_candidate_strength(candidate),
            candidate.semantic_name,
            candidate.source,
            _mask_digest(candidate.mask),
        ),
    )
    kept: list[MaskCandidate] = []
    for candidate in ordered:
        mask = np.asarray(candidate.mask, dtype=bool) & root_mask
        if not mask.any():
            continue
        if any(
            (
                _overlap(mask, np.asarray(previous.mask, dtype=bool) & root_mask)[0]
                >= iou_threshold
                or _overlap(
                    mask, np.asarray(previous.mask, dtype=bool) & root_mask
                )[1]
                >= containment_threshold
            )
            for previous in kept
        ):
            continue
        kept.append(candidate)
    rows = [
        (
            candidate.semantic_name,
            candidate.semantic_parent,
            np.asarray(candidate.mask, dtype=bool) & root_mask,
            _candidate_strength(candidate),
        )
        for candidate in kept
    ]
    owned = _exclusive_rows(
        rows,
        root_semantic=root.semantic_name,
        root_parent=root.semantic_parent,
        root_mask=root_mask,
    )
    return BaselinePrediction(
        method="greedy_nms",
        instances=_stable_instances(owned, root_mask=root_mask),
        root_mask=root_mask,
        diagnostics={
            "input_candidate_count": len(candidates),
            "kept_non_root_count": len(kept),
            "iou_threshold": iou_threshold,
            "containment_threshold": containment_threshold,
        },
    )


def _candidate_distance(first: MaskCandidate, second: MaskCandidate) -> float:
    first_mask = np.asarray(first.mask, dtype=bool)
    second_mask = np.asarray(second.mask, dtype=bool)
    iou, containment = _overlap(first_mask, second_mask)
    first_area = max(1, int(np.count_nonzero(first_mask)))
    second_area = max(1, int(np.count_nonzero(second_mask)))
    coherence = min(first_area, second_area) / max(first_area, second_area)
    overlap_similarity = max(iou, containment * coherence)
    first_x, first_y = _centroid(first_mask)
    second_x, second_y = _centroid(second_mask)
    diagonal = max(1.0, float(np.hypot(*first_mask.shape)))
    centroid_distance = min(
        1.0,
        float(np.hypot(first_x - second_x, first_y - second_y)) / diagonal,
    )
    area_distance = min(1.0, abs(np.log(first_area / second_area)) / 3.0)
    semantic_distance = float(first.semantic_name != second.semantic_name)
    parent_distance = float(first.semantic_parent != second.semantic_parent)
    return float(
        0.48 * (1.0 - overlap_similarity)
        + 0.20 * centroid_distance
        + 0.14 * area_distance
        + 0.12 * semantic_distance
        + 0.06 * parent_distance
    )


def _dbscan_components(distance: np.ndarray, eps: float) -> list[list[int]]:
    # With min_samples=1, DBSCAN is the connected-components closure of the
    # eps-neighbourhood graph. This small implementation keeps the baseline
    # dependency-free and deterministic.
    count = distance.shape[0]
    seen = np.zeros(count, dtype=bool)
    clusters: list[list[int]] = []
    for start in range(count):
        if seen[start]:
            continue
        seen[start] = True
        stack = [start]
        cluster: list[int] = []
        while stack:
            current = stack.pop()
            cluster.append(current)
            neighbours = np.flatnonzero(distance[current] <= eps)
            for neighbour in neighbours.tolist():
                if not seen[neighbour]:
                    seen[neighbour] = True
                    stack.append(neighbour)
        clusters.append(sorted(cluster))
    return clusters


def dbscan_proposal_fusion(
    candidates: list[MaskCandidate],
    *,
    eps: float = 0.43,
    vote_threshold: float = 0.50,
) -> BaselinePrediction:
    root = _root_candidate(candidates)
    root_mask = np.asarray(root.mask, dtype=bool)
    proposals = sorted(
        _non_root_candidates(candidates, root),
        key=lambda candidate: (
            candidate.semantic_name,
            candidate.semantic_parent,
            candidate.source,
            _mask_digest(candidate.mask),
        ),
    )
    distance = np.zeros((len(proposals), len(proposals)), dtype=np.float32)
    for first in range(len(proposals)):
        for second in range(first + 1, len(proposals)):
            value = _candidate_distance(proposals[first], proposals[second])
            distance[first, second] = value
            distance[second, first] = value
    clusters = _dbscan_components(distance, eps) if proposals else []
    rows: list[tuple[str, str, np.ndarray, float]] = []
    for indexes in clusters:
        members = [proposals[index] for index in indexes]
        strengths = np.asarray(
            [_candidate_strength(candidate) for candidate in members],
            dtype=np.float32,
        )
        semantic_scores: dict[tuple[str, str], float] = {}
        for candidate, strength in zip(members, strengths, strict=True):
            key = (candidate.semantic_name, candidate.semantic_parent)
            semantic_scores[key] = semantic_scores.get(key, 0.0) + float(strength)
        semantic, parent = max(
            semantic_scores,
            key=lambda key: (semantic_scores[key], key[0], key[1]),
        )
        stack = np.stack(
            [np.asarray(candidate.mask, dtype=bool) & root_mask for candidate in members]
        )
        support = np.sum(stack * strengths[:, None, None], axis=0) / max(
            1e-6, float(np.sum(strengths))
        )
        cluster_mask = support >= vote_threshold
        if not cluster_mask.any():
            representative = int(np.argmax(strengths))
            cluster_mask = stack[representative]
        rows.append(
            (
                semantic,
                parent,
                cluster_mask,
                float(np.max(strengths)),
            )
        )
    owned = _exclusive_rows(
        rows,
        root_semantic=root.semantic_name,
        root_parent=root.semantic_parent,
        root_mask=root_mask,
    )
    return BaselinePrediction(
        method="dbscan_fusion",
        instances=_stable_instances(owned, root_mask=root_mask),
        root_mask=root_mask,
        diagnostics={
            "input_candidate_count": len(candidates),
            "cluster_count": len(clusters),
            "eps": eps,
            "min_samples": 1,
            "vote_threshold": vote_threshold,
        },
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=0, keepdims=True)
    exp = np.exp(np.clip(shifted, -30.0, 30.0))
    return exp / np.maximum(1e-8, np.sum(exp, axis=0, keepdims=True))


def _pairwise_message(q: np.ndarray, image: np.ndarray, sigma: float) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    message = np.zeros_like(q, dtype=np.float32)
    normalizer = np.zeros(q.shape[1:], dtype=np.float32)

    horizontal = np.exp(
        -np.sum((rgb[:, 1:] - rgb[:, :-1]) ** 2, axis=2) / (2.0 * sigma**2)
    ).astype(np.float32)
    message[:, :, 1:] += q[:, :, :-1] * horizontal[None, :, :]
    message[:, :, :-1] += q[:, :, 1:] * horizontal[None, :, :]
    normalizer[:, 1:] += horizontal
    normalizer[:, :-1] += horizontal

    vertical = np.exp(
        -np.sum((rgb[1:] - rgb[:-1]) ** 2, axis=2) / (2.0 * sigma**2)
    ).astype(np.float32)
    message[:, 1:, :] += q[:, :-1, :] * vertical[None, :, :]
    message[:, :-1, :] += q[:, 1:, :] * vertical[None, :, :]
    normalizer[1:, :] += vertical
    normalizer[:-1, :] += vertical
    return message / np.maximum(1e-6, normalizer[None, :, :])


def local_pairwise_crf(
    candidates: list[MaskCandidate],
    image: np.ndarray,
    *,
    iterations: int = 4,
    pairwise_weight: float = 1.1,
    color_sigma: float = 0.12,
) -> BaselinePrediction:
    root = _root_candidate(candidates)
    root_mask = np.asarray(root.mask, dtype=bool)
    proposals = _non_root_candidates(candidates, root)
    semantic_rows: dict[tuple[str, str], list[MaskCandidate]] = {}
    for candidate in proposals:
        semantic_rows.setdefault(
            (candidate.semantic_name, candidate.semantic_parent), []
        ).append(candidate)
    labels = [(root.semantic_name, root.semantic_parent), *sorted(semantic_rows)]
    unary = np.full((len(labels), *root_mask.shape), -3.5, dtype=np.float32)
    unary[0, root_mask] = 0.25
    for label_index, key in enumerate(labels[1:], start=1):
        evidence = np.zeros(root_mask.shape, dtype=np.float32)
        for candidate in semantic_rows[key]:
            mask = np.asarray(candidate.mask, dtype=bool) & root_mask
            strength = _candidate_strength(candidate)
            evidence = np.maximum(evidence, mask.astype(np.float32) * strength)
        unary[label_index] = 4.0 * evidence - 1.8
        unary[label_index, ~root_mask] = -8.0
    unary[0, ~root_mask] = -8.0
    q = _softmax(unary)
    for _ in range(iterations):
        message = _pairwise_message(q, image, color_sigma)
        q = _softmax(unary + pairwise_weight * message)
    labels_map = np.argmax(q, axis=0)
    labels_map[~root_mask] = -1

    rows: list[tuple[str, str, np.ndarray, float]] = []
    for label_index, (semantic, parent) in enumerate(labels):
        class_mask = labels_map == label_index
        count, components, stats, _ = cv2.connectedComponentsWithStats(
            class_mask.astype(np.uint8), connectivity=8
        )
        for component in range(1, count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            if area < 6:
                continue
            mask = components == component
            rows.append(
                (
                    semantic,
                    parent,
                    mask,
                    float(np.mean(q[label_index][mask])),
                )
            )
    return BaselinePrediction(
        method="local_pairwise_crf",
        instances=_stable_instances(rows, root_mask=root_mask),
        root_mask=root_mask,
        diagnostics={
            "input_candidate_count": len(candidates),
            "label_count": len(labels),
            "iterations": iterations,
            "pairwise_weight": pairwise_weight,
            "color_sigma": color_sigma,
            "scope": "local four-neighbour Potts CRF; not DenseCRF",
        },
    )


def overlap_excess(instances: tuple[BaselineInstance, ...], root: np.ndarray) -> float:
    if not instances:
        return 0.0
    stack = np.stack([instance.mask for instance in instances]).sum(axis=0)
    return float(np.maximum(stack - 1, 0)[root].sum() / max(1, np.count_nonzero(root)))


def unassigned_fraction(
    instances: tuple[BaselineInstance, ...], root: np.ndarray
) -> float:
    if not instances:
        return 1.0
    union = np.logical_or.reduce([instance.mask for instance in instances])
    return float(np.count_nonzero(root & ~union) / max(1, np.count_nonzero(root)))
