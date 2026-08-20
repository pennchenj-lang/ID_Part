from __future__ import annotations

import numpy as np

from hpid_split.semantic_refinement import (
    RefinedSemanticMask,
    SemanticRefinementConfig,
    _component_prompts,
    exclusive_semantic_assignment,
)


def test_component_prompts_keep_disconnected_instances_separate() -> None:
    mask = np.zeros((48, 64), dtype=bool)
    mask[8:18, 7:17] = True
    mask[25:40, 42:55] = True
    probability = mask.astype(np.float32)

    prompts = _component_prompts(
        "wheel",
        mask,
        probability,
        config=SemanticRefinementConfig(minimum_component_area=5),
    )

    assert len(prompts) == 2
    assert all(sum(label == 1 for label in prompt.labels) >= 1 for prompt in prompts)
    assert all(len(prompt.points) == len(prompt.labels) == 7 for prompt in prompts)


def test_exclusive_assignment_gives_each_pixel_one_owner() -> None:
    root = np.ones((20, 20), dtype=bool)
    left = np.zeros(root.shape, dtype=bool)
    right = np.zeros(root.shape, dtype=bool)
    left[:, :13] = True
    right[:, 7:] = True
    left_probability = np.full(root.shape, 0.2, dtype=np.float32)
    right_probability = np.full(root.shape, 0.2, dtype=np.float32)
    left_probability[:, :10] = 0.9
    right_probability[:, 10:] = 0.9
    refined = {
        "left": RefinedSemanticMask("left", left, left_probability, True, 1, ()),
        "right": RefinedSemanticMask("right", right, right_probability, True, 1, ()),
    }

    assigned, residual = exclusive_semantic_assignment(
        root, refined, activation_threshold=0.3
    )

    assert not (assigned["left"] & assigned["right"]).any()
    assert (assigned["left"] | assigned["right"] | residual).all()
    assert assigned["left"][:, :8].all()
    assert assigned["right"][:, 12:].all()


def test_exclusive_assignment_leaves_unproposed_root_as_residual() -> None:
    root = np.ones((12, 12), dtype=bool)
    mask = np.zeros(root.shape, dtype=bool)
    mask[3:9, 3:9] = True
    probability = np.full(root.shape, 0.8, dtype=np.float32)
    refined = {"detail": RefinedSemanticMask("detail", mask, probability, False, 1, ())}

    assigned, residual = exclusive_semantic_assignment(
        root, refined, activation_threshold=0.3
    )

    assert np.array_equal(assigned["detail"], mask)
    assert np.array_equal(residual, root & ~mask)
