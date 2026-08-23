from __future__ import annotations

import numpy as np

from scripts.analyze_cross_domain_fusion_ablation import _coverage_metrics


def test_coverage_metrics_distinguish_root_from_part_union() -> None:
    truth = np.ones((4, 4), dtype=bool)
    root = truth.copy()
    editable_part = np.zeros_like(truth)
    editable_part[:2] = True

    root_iou, _, root_recall = _coverage_metrics([root], truth)
    part_iou, _, part_recall = _coverage_metrics([editable_part], truth)

    assert root_iou == 1.0
    assert root_recall == 1.0
    assert part_iou == 0.5
    assert part_recall == 0.5
