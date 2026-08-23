import cv2
import numpy as np

from hpid_split.cli import _conditional_mask_root_gate


def test_conditional_mask_root_gate_rejects_sparse_compact_open_form() -> None:
    canvas = np.zeros((110, 150), dtype=np.uint8)
    cv2.circle(canvas, (38, 76), 23, 1, -1)
    cv2.circle(canvas, (112, 76), 23, 1, -1)
    cv2.line(canvas, (38, 76), (72, 34), 1, 7)
    cv2.line(canvas, (72, 34), (112, 76), 1, 7)
    cv2.line(canvas, (38, 76), (112, 76), 1, 7)
    cv2.line(canvas, (72, 34), (88, 18), 1, 6)
    mask = canvas.astype(bool)

    row = _conditional_mask_root_gate(mask)

    assert row["accepted"] is False
    assert row["reason"] == "ambiguous_open_form"
    assert row["compactness"] < 0.40


def test_conditional_mask_root_gate_keeps_sparse_elongated_tool() -> None:
    mask = np.zeros((160, 80), dtype=bool)
    mask[12:148, 34:44] = True
    mask[128:150, 20:58] = True

    row = _conditional_mask_root_gate(mask)

    assert row["accepted"] is True
    assert row["reason"] == "root_geometry_supported"


def test_conditional_mask_root_gate_keeps_compact_solid_asset() -> None:
    mask = np.zeros((100, 120), dtype=bool)
    mask[12:90, 10:112] = True

    row = _conditional_mask_root_gate(mask)

    assert row["accepted"] is True
