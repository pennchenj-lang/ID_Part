from pathlib import Path

import numpy as np
from PIL import Image

from hpid_split.export import (
    _algorithm_metadata,
    colorize_part_ids,
    render_source_overlay,
)


def test_foundation_algorithm_metadata_is_reproducible(tmp_path: Path) -> None:
    diagnostics = {
        "prompt_bank": {"sha256": "abc123"},
        "candidate_generations": [
            {
                "models": {
                    "grounding_model": "detector-a",
                    "segmentation_model": "segmenter-a",
                    "dense_semantic_model": "dense-a",
                    "box_threshold": 0.24,
                }
            },
            {
                "models": {
                    "grounding_model": "detector-b",
                    "segmentation_model": "segmenter-a",
                }
            },
        ],
        "fusion": {"ablation": {"consensus": True, "direct_gate": True}},
        "completion": {"backend": "completion-a"},
    }

    metadata = _algorithm_metadata(diagnostics, checkpoint=None)

    assert metadata["name"] == "HPID-Split"
    assert metadata["version"]
    assert metadata["mode"] == "foundation-fusion"
    assert metadata["ground_truth_used"] is False
    assert metadata["prompt_bank_sha256"] == "abc123"
    assert metadata["candidate_models"] == [
        {
            "grounding_model": "detector-a",
            "segmentation_model": "segmenter-a",
            "dense_semantic_model": "dense-a",
        },
        {
            "grounding_model": "detector-b",
            "segmentation_model": "segmenter-a",
        },
    ]
    assert metadata["fusion_config"] == {"consensus": True, "direct_gate": True}
    assert metadata["completion_backend"] == "completion-a"
    assert metadata["checkpoint_sha256"] is None


def test_learned_algorithm_metadata_hashes_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint")

    metadata = _algorithm_metadata(None, checkpoint)

    assert metadata["mode"] == "learned-checkpoint"
    assert metadata["checkpoint_sha256"] == (
        "47320987f9a49d5b00119b960f247a956773f57543982b8bfcb6da5bb3afd9ef"
    )


def test_part_id_preview_uses_distinct_instance_colors() -> None:
    instance_map = np.asarray([[0, 1, 1], [0, 2, 2]], dtype=np.uint16)

    preview = np.asarray(colorize_part_ids(instance_map))

    assert tuple(preview[0, 0]) == (0, 0, 0)
    assert tuple(preview[0, 1]) != tuple(preview[1, 1])


def test_part_id_preview_prioritizes_contrast_between_adjacent_ids() -> None:
    instance_map = np.asarray(
        [
            [1, 1, 2, 2, 3, 3],
            [1, 4, 4, 2, 3, 5],
            [6, 4, 7, 7, 5, 5],
        ],
        dtype=np.uint16,
    )

    first = np.asarray(colorize_part_ids(instance_map))
    second = np.asarray(colorize_part_ids(instance_map))

    assert np.array_equal(first, second)
    colors = {part_id: first[instance_map == part_id][0] for part_id in range(1, 8)}
    assert len({tuple(color) for color in colors.values()}) == 7
    for left, right in ((1, 2), (1, 4), (2, 3), (2, 4), (2, 7), (4, 6)):
        distance = np.linalg.norm(
            colors[left].astype(np.float32) - colors[right].astype(np.float32)
        )
        assert distance >= 70.0


def test_source_overlay_preserves_shape_and_marks_part_boundaries() -> None:
    source = Image.new("RGB", (4, 3), (80, 90, 100))
    instance_map = np.asarray(
        [[0, 1, 1, 2], [0, 1, 1, 2], [0, 1, 1, 2]], dtype=np.uint16
    )

    overlay = np.asarray(render_source_overlay(source, instance_map))

    assert overlay.shape == (3, 4, 3)
    assert tuple(overlay[0, 0]) == (80, 90, 100)
    assert np.any(np.all(overlay == 255, axis=2))
