import numpy as np

from hpid_split.paper_eval import evaluate_part_predictions


def _mask(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    value = np.zeros((12, 12), dtype=bool)
    value[y0:y1, x0:x1] = True
    return value


def test_strict_metrics_penalize_wrong_semantics() -> None:
    first = _mask(1, 1, 5, 5)
    second = _mask(7, 7, 11, 11)
    result = evaluate_part_predictions(
        truth_masks=[first, second],
        truth_semantics=["screen", "button"],
        prediction_masks=[first, second],
        prediction_semantics=["button", "screen"],
        truth_object_mask=first | second,
    )

    assert result["part_f1_at_075"] == 1.0
    assert result["semantic_f1_at_025"] == 0.0
    assert result["object_iou"] == 1.0


def test_unlabeled_oversegmentation_reduces_precision() -> None:
    truth = _mask(1, 1, 5, 5)
    false_positive = _mask(7, 7, 11, 11)
    result = evaluate_part_predictions(
        truth_masks=[truth],
        truth_semantics=["blade"],
        prediction_masks=[truth, false_positive],
        prediction_semantics=[None, None],
        truth_object_mask=truth,
    )

    assert result["part_precision_at_050"] == 0.5
    assert result["part_recall_at_050"] == 1.0
    assert result["semantic_f1_at_050"] == 0.0
    assert result["oversegmentation_ratio"] == 2.0
