import numpy as np

from hpid_split.dense_semantic import parent_envelope, select_dense_regions


def test_parent_envelope_fills_child_hole_and_expands_visible_support() -> None:
    mask = np.zeros((64, 64), dtype=bool)
    mask[12:52, 12:52] = True
    mask[26:38, 26:38] = False

    envelope = parent_envelope(mask, 0.05)

    assert envelope[32, 32]
    assert envelope[10, 32]
    assert not envelope[0, 0]


def test_dense_regions_recover_two_separated_relative_peaks() -> None:
    yy, xx = np.mgrid[:96, :128]
    first = np.exp(-((xx - 36) ** 2 + (yy - 48) ** 2) / (2 * 9**2))
    second = 0.92 * np.exp(-((xx - 92) ** 2 + (yy - 44) ** 2) / (2 * 10**2))
    probability = (0.01 + 0.58 * np.maximum(first, second)).astype(np.float32)
    allowed = np.ones(probability.shape, dtype=bool)

    regions = select_dense_regions(
        probability,
        allowed,
        maximum=2,
        minimum_peak_probability=0.08,
        minimum_peak_contrast=0.035,
        activation_quantile=0.88,
        peak_ratio=0.46,
        box_padding_ratio=0.30,
    )

    assert len(regions) == 2
    centers = sorted((box[0] + box[2]) / 2 for box, _, _ in regions)
    assert 30 <= centers[0] <= 42
    assert 86 <= centers[1] <= 98


def test_dense_regions_reject_flat_low_contrast_response() -> None:
    probability = np.full((48, 48), 0.12, dtype=np.float32)
    allowed = np.ones(probability.shape, dtype=bool)

    regions = select_dense_regions(
        probability,
        allowed,
        maximum=2,
        minimum_peak_probability=0.08,
        minimum_peak_contrast=0.035,
        activation_quantile=0.88,
        peak_ratio=0.46,
        box_padding_ratio=0.30,
    )

    assert regions == []
