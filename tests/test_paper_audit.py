from __future__ import annotations

import numpy as np
import pytest

from hpid_split.paper_audit import (
    overlap_excess_fraction,
    paired_bootstrap_interval,
    unassigned_root_fraction,
)


def test_identity_pixel_metrics_use_only_the_supplied_root() -> None:
    root = np.zeros((4, 4), dtype=bool)
    root[1:3, 1:3] = True
    first = np.zeros_like(root)
    first[1:3, 1] = True
    second = np.zeros_like(root)
    second[1:3, 1:3] = True
    second[0, 0] = True

    assert overlap_excess_fraction([first, second], root) == pytest.approx(0.5)
    assert unassigned_root_fraction([first, second], root) == 0.0


def test_paired_bootstrap_reports_zero_crossing() -> None:
    result = paired_bootstrap_interval(
        [-1.0, 1.0, -1.0, 1.0],
        seed=7,
        iterations=2_000,
    )

    assert result["mean_paired_difference"] == 0.0
    assert result["ci95_includes_zero"] is True
    assert result["case_count"] == 4
