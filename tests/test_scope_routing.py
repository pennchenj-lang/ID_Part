import cv2
import numpy as np
from PIL import Image

from hpid_split.scope_routing import (
    extract_primary_foreground,
    route_extraction_scope,
)


def test_scene_request_resolves_isolated_asset_to_primary() -> None:
    image = np.full((180, 260, 3), 28, dtype=np.uint8)
    cv2.ellipse(image, (130, 90), (78, 34), 0, 0, 360, (170, 92, 38), -1)
    cv2.rectangle(image, (118, 82), (150, 158), (80, 90, 105), -1)

    result = route_extraction_scope(Image.fromarray(image), "Entire scene")

    assert result.resolved_scope == "Primary asset"
    assert result.diagnostics["status"] == "resolved_single_subject"
    assert result.diagnostics["ground_truth_used"] is False


def test_scene_request_keeps_large_scene_surface() -> None:
    image = np.full((180, 260, 3), 245, dtype=np.uint8)
    points = np.asarray([[8, 55], [130, 12], [252, 55], [190, 175], [65, 175]])
    cv2.fillConvexPoly(image, points, (62, 165, 58))
    for center in ((55, 88), (110, 65), (175, 100), (205, 58)):
        cv2.circle(image, center, 18, (120, 118, 102), -1)

    result = route_extraction_scope(Image.fromarray(image), "Entire scene")

    assert result.resolved_scope == "Entire scene"
    assert result.diagnostics["status"] == "kept_scene_scope"


def test_explicit_primary_scope_is_never_overridden() -> None:
    result = route_extraction_scope(
        Image.new("RGB", (80, 80), "white"), "Primary asset"
    )

    assert result.resolved_scope == "Primary asset"
    assert result.diagnostics["status"] == "kept_explicit_scope"


def test_border_foreground_extracts_one_isolated_object() -> None:
    image = np.full((180, 260, 3), 242, dtype=np.uint8)
    cv2.rectangle(image, (45, 58), (220, 118), (38, 72, 115), -1)
    cv2.rectangle(image, (125, 115), (160, 165), (110, 45, 38), -1)

    result = extract_primary_foreground(Image.fromarray(image))

    assert result.mask is not None
    expected_u8 = np.zeros((180, 260), dtype=np.uint8)
    cv2.rectangle(expected_u8, (45, 58), (220, 118), 1, -1)
    cv2.rectangle(expected_u8, (125, 115), (160, 165), 1, -1)
    expected = expected_u8.astype(bool)
    intersection = np.count_nonzero(result.mask & expected)
    union = np.count_nonzero(result.mask | expected)
    assert intersection / union >= 0.72
    assert result.diagnostics["ground_truth_used"] is False


def test_border_foreground_rejects_nonuniform_border_scene() -> None:
    image = np.zeros((180, 260, 3), dtype=np.uint8)
    image[:90] = (220, 40, 40)
    image[90:] = (40, 180, 80)

    result = extract_primary_foreground(Image.fromarray(image))

    assert result.mask is None
    assert result.diagnostics["status"] == "rejected"
