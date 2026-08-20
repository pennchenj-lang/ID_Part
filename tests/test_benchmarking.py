import numpy as np

from hpid_split.benchmarking import evaluate_amodal_case, make_edge_occlusion


def test_synthetic_occlusion_is_connected_and_has_requested_scale() -> None:
    full = np.zeros((80, 100), dtype=bool)
    full[15:65, 20:80] = True
    case = make_edge_occlusion(full, target_hidden_fraction=0.30, direction_offset=2)
    assert np.array_equal(case.visible_mask | case.hidden_mask, full)
    assert not np.any(case.visible_mask & case.hidden_mask)
    assert 0.25 <= case.hidden_fraction <= 0.35
    assert np.any(case.occluder_mask & ~full)


def test_amodal_metrics_distinguish_recovery_from_hallucination() -> None:
    full = np.zeros((20, 20), dtype=bool)
    full[4:16, 4:16] = True
    visible = full.copy()
    visible[:, 4:8] = False
    baseline = evaluate_amodal_case(full, visible, visible)
    recovered = evaluate_amodal_case(full, visible, full)
    hallucinated = full.copy()
    hallucinated[2:18, 2:4] = True
    hallucinated_metrics = evaluate_amodal_case(full, visible, hallucinated)
    assert baseline.hidden_recall == 0.0
    assert recovered.completed_iou == 1.0
    assert recovered.hidden_recall == 1.0
    assert hallucinated_metrics.completed_iou < recovered.completed_iou
    assert hallucinated_metrics.false_added_ratio > 0.0
