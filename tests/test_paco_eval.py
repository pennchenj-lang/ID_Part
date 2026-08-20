import json
from pathlib import Path

import numpy as np
from PIL import Image

from hpid_split.paco_eval import evaluate_paco_package


def _save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def _minimal_case(tmp_path: Path) -> tuple[Path, Path]:
    case_dir = tmp_path / "minimal_case"
    package = tmp_path / "minimal_package"
    screen = np.zeros((24, 24), dtype=bool)
    screen[4:20, 4:20] = True
    _save_mask(case_dir / "screen.png", screen)
    _save_mask(case_dir / "object_mask_crop.png", screen)
    case_path = case_dir / "case.json"
    case_path.write_text(
        json.dumps(
            {
                "object_category": "cellular_telephone",
                "parts": [{"part_name": "screen", "mask_crop": "screen.png"}],
            }
        ),
        encoding="utf-8",
    )
    _save_mask(package / "masks" / "screen.png", screen)
    (package / "parts.json").write_text(
        json.dumps(
            [
                {
                    "semantic_name": "device_screen",
                    "mask_visible_path": "masks/screen.png",
                }
            ]
        ),
        encoding="utf-8",
    )
    (package / "inference_diagnostics.json").write_text(
        json.dumps({"root_routing": {"selected_semantic": "device"}}),
        encoding="utf-8",
    )
    return package, case_path


def test_paco_evaluator_matches_parts_without_leaking_truth_into_inference(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case"
    package = tmp_path / "package"
    left = np.zeros((32, 32), dtype=bool)
    left[4:28, 3:15] = True
    right = np.zeros((32, 32), dtype=bool)
    right[4:28, 17:29] = True
    _save_mask(case_dir / "parts_crop" / "screen.png", left)
    _save_mask(case_dir / "parts_crop" / "button.png", right)
    _save_mask(case_dir / "object_mask_crop.png", left | right)
    case = {
        "object_category": "cellular_telephone",
        "parts": [
            {"part_name": "screen", "mask_crop": "parts_crop/screen.png"},
            {"part_name": "button", "mask_crop": "parts_crop/button.png"},
        ],
    }
    case_dir.mkdir(exist_ok=True)
    (case_dir / "case.json").write_text(json.dumps(case), encoding="utf-8")
    _save_mask(package / "masks_visible" / "screen.png", left)
    _save_mask(package / "masks_visible" / "button.png", right)
    package.mkdir(exist_ok=True)
    (package / "parts.json").write_text(
        json.dumps(
            [
                {
                    "semantic_name": "device_screen",
                    "mask_visible_path": "masks_visible/screen.png",
                },
                {
                    "semantic_name": "device_button",
                    "mask_visible_path": "masks_visible/button.png",
                },
            ]
        ),
        encoding="utf-8",
    )
    (package / "inference_diagnostics.json").write_text(
        json.dumps(
            {
                "root_routing": {"selected_semantic": "device"},
                "profile_root_resolution": {
                    "profile_consensus": {
                        "roots": [
                            {"status": "accepted", "selected_profile": "phone"}
                        ]
                    }
                },
                "profile_refinement": {"roots": [{"selected_profile": "phone"}]},
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_paco_package(
        package,
        case_dir / "case.json",
        expected_domain="device",
        expected_profile="phone",
    )

    assert result["domain_correct"] is True
    assert result["profile_correct"] is True
    assert result["object_iou"] == 1.0
    assert result["part_discovery_f1_at_025"] == 1.0
    assert result["semantic_part_recall"] == 1.0
    assert result["inference_uses_ground_truth"] is False


def test_semantic_recall_is_not_lost_to_class_agnostic_assignment(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case"
    package = tmp_path / "package"
    screen = np.zeros((40, 40), dtype=bool)
    screen[5:35, 5:35] = True
    bezel = np.zeros((40, 40), dtype=bool)
    bezel[3:37, 3:37] = True
    bezel[5:35, 5:35] = False
    wrong_bezel = screen.copy()
    predicted_screen = screen.copy()
    _save_mask(case_dir / "screen.png", screen)
    _save_mask(case_dir / "bezel.png", bezel)
    _save_mask(case_dir / "object_mask_crop.png", screen | bezel)
    (case_dir / "case.json").write_text(
        json.dumps(
            {
                "object_category": "phone",
                "parts": [
                    {"part_name": "screen", "mask_crop": "screen.png"},
                    {"part_name": "bezel", "mask_crop": "bezel.png"},
                ],
            }
        ),
        encoding="utf-8",
    )
    _save_mask(package / "masks" / "screen.png", predicted_screen)
    _save_mask(package / "masks" / "bezel.png", wrong_bezel)
    (package / "parts.json").write_text(
        json.dumps(
            [
                {
                    "semantic_name": "device_screen",
                    "mask_visible_path": "masks/screen.png",
                },
                {
                    "semantic_name": "device_bezel",
                    "mask_visible_path": "masks/bezel.png",
                },
            ]
        ),
        encoding="utf-8",
    )
    (package / "inference_diagnostics.json").write_text(
        json.dumps(
            {
                "root_routing": {"selected_semantic": "device"},
                "profile_refinement": {"roots": []},
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_paco_package(
        package,
        case_dir / "case.json",
        expected_domain="device",
    )

    assert result["semantic_part_recall"] == 0.5
    assert sum(row["accepted"] for row in result["semantic_matches"]) == 1
    screen_row = next(
        row for row in result["semantic_matches"] if row["truth_part"] == "screen"
    )
    assert screen_row["best_same_semantic_iou"] == 1.0


def test_semantic_recall_does_not_reuse_one_prediction_for_two_truth_parts(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case"
    package = tmp_path / "package"
    left_button = np.zeros((32, 32), dtype=bool)
    left_button[5:13, 3:11] = True
    right_button = np.zeros((32, 32), dtype=bool)
    right_button[5:13, 21:29] = True
    predicted_button = left_button | right_button
    _save_mask(case_dir / "button_left.png", left_button)
    _save_mask(case_dir / "button_right.png", right_button)
    _save_mask(case_dir / "object_mask_crop.png", predicted_button)
    (case_dir / "case.json").write_text(
        json.dumps(
            {
                "object_category": "remote_control",
                "parts": [
                    {"part_name": "button", "mask_crop": "button_left.png"},
                    {"part_name": "button", "mask_crop": "button_right.png"},
                ],
            }
        ),
        encoding="utf-8",
    )
    _save_mask(package / "masks" / "button.png", predicted_button)
    (package / "parts.json").write_text(
        json.dumps(
            [
                {
                    "semantic_name": "device_button",
                    "mask_visible_path": "masks/button.png",
                }
            ]
        ),
        encoding="utf-8",
    )
    (package / "inference_diagnostics.json").write_text(
        json.dumps(
            {
                "root_routing": {"selected_semantic": "device"},
                "profile_refinement": {"roots": []},
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_paco_package(
        package,
        case_dir / "case.json",
        expected_domain="device",
    )

    assert result["semantic_part_recall"] == 0.5


def test_semantic_union_reports_split_instances_without_hiding_strict_failure(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case"
    package = tmp_path / "package"
    buttons = []
    for index, x0 in enumerate((2, 9, 16, 23, 30), start=1):
        mask = np.zeros((36, 36), dtype=bool)
        mask[10:18, x0 : x0 + 4] = True
        buttons.append(mask)
        _save_mask(package / "masks" / f"button_{index}.png", mask)
    truth_union = np.logical_or.reduce(buttons)
    _save_mask(case_dir / "buttons.png", truth_union)
    _save_mask(case_dir / "object_mask_crop.png", truth_union)
    (case_dir / "case.json").write_text(
        json.dumps(
            {
                "object_category": "remote_control",
                "parts": [
                    {"part_name": "button", "mask_crop": "buttons.png"}
                ],
            }
        ),
        encoding="utf-8",
    )
    (package / "parts.json").write_text(
        json.dumps(
            [
                {
                    "semantic_name": "device_button",
                    "mask_visible_path": f"masks/button_{index}.png",
                }
                for index in range(1, 6)
            ]
        ),
        encoding="utf-8",
    )
    (package / "inference_diagnostics.json").write_text(
        json.dumps({"root_routing": {"selected_semantic": "device"}}),
        encoding="utf-8",
    )

    result = evaluate_paco_package(
        package,
        case_dir / "case.json",
        expected_domain="device",
    )

    assert result["semantic_part_recall"] == 0.0
    assert result["semantic_union_recall"] == 1.0
    assert result["mean_semantic_union_iou"] == 1.0
    assert result["semantic_union_matches"][0]["predicted_instance_count"] == 5


def test_profile_accuracy_uses_consensus_not_refinement_rows(tmp_path: Path) -> None:
    package, case = _minimal_case(tmp_path)
    diagnostics_path = package / "inference_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["profile_root_resolution"] = {
        "profile_consensus": {
            "roots": [
                {
                    "status": "unresolved",
                    "selected_profile": None,
                }
            ]
        }
    }
    diagnostics["profile_refinement"] = {
        "roots": [{"selected_profile": "phone"}]
    }
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")

    result = evaluate_paco_package(
        package,
        case,
        expected_domain="device",
        expected_profile="phone",
    )

    assert result["selected_profiles"] == []
    assert result["profile_correct"] is False


def test_profile_accuracy_reads_explicit_profile_lock(tmp_path: Path) -> None:
    package, case = _minimal_case(tmp_path)
    diagnostics_path = package / "inference_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    diagnostics["profile_root_resolution"] = {
        "algorithm": "user-asset-prompt-profile-lock-v1",
        "selected_profiles": ["phone"],
        "ground_truth_used": False,
    }
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")

    result = evaluate_paco_package(
        package,
        case,
        expected_domain="device",
        expected_profile="phone",
    )

    assert result["selected_profiles"] == ["phone"]
    assert result["profile_correct"] is True


def test_category_specific_truth_aliases_are_evaluated_canonically(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "pan_case"
    package = tmp_path / "pan_package"
    pan_body = np.zeros((32, 32), dtype=bool)
    pan_body[5:27, 4:28] = True
    _save_mask(case_dir / "bottom.png", pan_body)
    _save_mask(case_dir / "object_mask_crop.png", pan_body)
    (case_dir / "case.json").write_text(
        json.dumps(
            {
                "object_category": "pan_for_cooking",
                "parts": [
                    {"part_name": "bottom", "mask_crop": "bottom.png"},
                ],
            }
        ),
        encoding="utf-8",
    )
    _save_mask(package / "masks" / "pan_body.png", pan_body)
    (package / "parts.json").write_text(
        json.dumps(
            [
                {
                    "semantic_name": "tool_prop_pan_body",
                    "mask_visible_path": "masks/pan_body.png",
                }
            ]
        ),
        encoding="utf-8",
    )
    (package / "inference_diagnostics.json").write_text(
        json.dumps({"root_routing": {"selected_semantic": "tool_prop"}}),
        encoding="utf-8",
    )

    result = evaluate_paco_package(
        package,
        case_dir / "case.json",
        expected_domain="tool_prop",
    )

    assert result["semantic_part_recall"] == 1.0
    assert result["matches"][0]["truth_part"] == "pan_body"
