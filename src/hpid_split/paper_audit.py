from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def overlap_excess_fraction(
    masks: Iterable[np.ndarray],
    root_mask: np.ndarray,
) -> float:
    """Return repeated ownership inside the supplied object root.

    A value of zero means every root pixel is claimed by at most one mask. A
    value above one is possible when several masks repeatedly cover the same
    root pixels.
    """

    root = np.asarray(root_mask, dtype=bool)
    selected = [np.asarray(mask, dtype=bool) & root for mask in masks]
    if not selected:
        return 0.0
    cover_count = np.stack(selected, axis=0).sum(axis=0)
    repeated = np.maximum(cover_count - 1, 0).sum()
    return float(repeated / max(1, int(root.sum())))


def unassigned_root_fraction(
    masks: Iterable[np.ndarray],
    root_mask: np.ndarray,
) -> float:
    """Return the fraction of supplied-root pixels with no predicted owner."""

    root = np.asarray(root_mask, dtype=bool)
    selected = [np.asarray(mask, dtype=bool) & root for mask in masks]
    if not selected:
        return 1.0 if root.any() else 0.0
    union = np.stack(selected, axis=0).any(axis=0)
    missing = root & ~union
    return float(missing.sum() / max(1, int(root.sum())))


def paired_bootstrap_interval(
    differences: Iterable[float],
    *,
    seed: int,
    iterations: int,
) -> dict[str, float | bool | int]:
    """Bootstrap a case-paired mean difference without a pixel-level test."""

    values = np.asarray(list(differences), dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("paired differences must be a non-empty vector")
    if iterations <= 0:
        raise ValueError("bootstrap iterations must be positive")
    generator = np.random.default_rng(seed)
    indexes = generator.integers(0, len(values), size=(iterations, len(values)))
    means = values[indexes].mean(axis=1)
    low = float(np.quantile(means, 0.025))
    high = float(np.quantile(means, 0.975))
    return {
        "case_count": len(values),
        "mean_paired_difference": float(values.mean()),
        "ci95_low": low,
        "ci95_high": high,
        "ci95_includes_zero": bool(low <= 0.0 <= high),
    }
