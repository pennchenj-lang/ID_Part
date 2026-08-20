import numpy as np

from hpid_split.metrics import evaluate_semantic
from hpid_split.taxonomy import Taxonomy


def test_part_count_error_cannot_cancel_between_semantic_classes() -> None:
    taxonomy = Taxonomy(
        fine_names=("background", "first", "second"),
        parent_names=("background", "foreground"),
        fine_to_parent=(0, 1, 1),
        detail_names=(),
    )
    truth = np.zeros((48, 48), dtype=np.uint8)
    truth[3:11, 3:11] = 1
    truth[4:12, 24:32] = 2
    truth[19:27, 24:32] = 2
    truth[34:42, 24:32] = 2

    prediction = np.zeros_like(truth)
    prediction[3:11, 3:11] = 1
    prediction[19:27, 3:11] = 1
    prediction[34:42, 3:11] = 1
    prediction[4:12, 24:32] = 2

    result = evaluate_semantic(prediction, truth, taxonomy)
    assert result["component_prediction_count"] == 4
    assert result["component_truth_count"] == 4
    assert result["part_count_abs_error"] == 4
