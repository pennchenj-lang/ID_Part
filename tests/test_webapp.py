from pathlib import Path

import pytest

from hpid_split.webapp import (
    _build_auto_command,
    _default_asset_router_index,
    _default_retrieval_index,
    _default_vlm_4bit,
    _default_vlm_model,
    _launch_allowed_paths,
    _output_root,
    _runtime_timeout_seconds,
)


def test_web_command_uses_bounded_ensemble_without_ground_truth(
    tmp_path: Path,
) -> None:
    completion = tmp_path / "completion.json"
    completion.write_text("{}", encoding="utf-8")
    command = _build_auto_command(
        tmp_path / "input.png",
        tmp_path / "output",
        domain="character",
        quality="Ensemble",
        complete_hidden_regions=True,
        prompt_bank=tmp_path / "prompts.json",
        completion_config=completion,
    )
    joined = " ".join(command)

    assert "IDEA-Research/grounding-dino-tiny" in command
    assert "IDEA-Research/grounding-dino-base" not in command
    assert "--completion-config" in command
    assert "--semantic-part-multimask" in command
    assert "--proposal-first-fast" not in command
    assert "--vlm-parts" not in command
    assert "--domains character" in joined
    assert "--root-mode primary" in joined
    assert "ground_truth" not in joined
    assert "reference_mask" not in joined


def test_web_command_rejects_unknown_quality(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown quality"):
        _build_auto_command(
            tmp_path / "input.png",
            tmp_path / "output",
            domain="auto",
            quality="Unsupported",
            complete_hidden_regions=False,
        )


def test_web_command_canonicalizes_all_scene_objects(tmp_path: Path) -> None:
    command = _build_auto_command(
        tmp_path / "input.png",
        tmp_path / "output",
        domain="auto",
        scope="Entire scene",
        quality="Fast",
        complete_hidden_regions=False,
    )

    assert "--root-mode" in command
    assert command[command.index("--root-mode") + 1] == "scene"
    assert command[command.index("--grounding-model") + 1] == (
        "IDEA-Research/grounding-dino-tiny"
    )
    assert command[command.index("--maximum-roots-per-domain") + 1] == "12"
    assert command[command.index("--maximum-total-roots") + 1] == "48"
    assert "--no-scene-profile-root-queries" in command
    assert "--no-isolated-profile-resolution" in command
    assert "--no-profile-refinement" in command
    assert "--adaptive-profile-refinement" not in command
    assert command[command.index("--visual-points-per-crop") + 1] == "12"
    assert "--dense-semantic-fallback" not in command
    assert command[command.index("--maximum-grabcut-candidates") + 1] == "8"
    assert "--no-relational-appearance" in command
    assert "--no-ontology-scene-consensus" in command


def test_web_scene_ensemble_is_bounded_by_default(tmp_path: Path) -> None:
    vlm_model = tmp_path / "vlm"
    vlm_model.mkdir()
    command = _build_auto_command(
        tmp_path / "input.png",
        tmp_path / "output",
        domain="auto",
        scope="Entire scene",
        quality="Ensemble",
        complete_hidden_regions=False,
        vlm_model=vlm_model,
    )

    assert command[command.index("--grounding-model") + 1] == (
        "IDEA-Research/grounding-dino-tiny"
    )
    assert "--additional-grounding-model" not in command
    assert command[command.index("--maximum-roots-per-domain") + 1] == "16"
    assert command[command.index("--maximum-total-roots") + 1] == "64"
    assert "--vlm-parts" not in command
    assert "--no-scene-profile-root-queries" in command
    assert "--no-isolated-profile-resolution" in command
    assert "--no-profile-refinement" in command
    assert "--proposal-first-fast" in command
    assert "--adaptive-profile-refinement" in command
    assert "--no-ontology-scene-consensus" not in command


def test_web_ensemble_enables_grounded_dynamic_inventory_when_vlm_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HPID_ENABLE_HEAVY_ENSEMBLE", "1")
    vlm_model = tmp_path / "vlm"
    vlm_model.mkdir()

    command = _build_auto_command(
        tmp_path / "input.png",
        tmp_path / "output",
        domain="auto",
        quality="Ensemble",
        complete_hidden_regions=False,
        vlm_model=vlm_model,
    )

    assert command[command.index("--vlm-model") + 1] == str(vlm_model)
    assert "--vlm-parts" in command
    assert "--vlm-dynamic-inventory" in command
    assert command[command.index("--vlm-maximum-total-queries") + 1] == "12"
    assert command[command.index("--vlm-maximum-root-audits") + 1] == "4"
    assert command[command.index("--vlm-maximum-semantic-audits") + 1] == "3"
    assert command[command.index("--vlm-maximum-physicality-audits") + 1] == "0"


def test_web_primary_ensemble_spends_vlm_budget_on_part_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HPID_ENABLE_HEAVY_ENSEMBLE", "1")
    vlm_model = tmp_path / "vlm"
    vlm_model.mkdir()

    command = _build_auto_command(
        tmp_path / "input.png",
        tmp_path / "output",
        domain="auto",
        scope="Primary asset",
        quality="Ensemble",
        complete_hidden_regions=False,
        vlm_model=vlm_model,
    )

    assert command[command.index("--vlm-maximum-physicality-audits") + 1] == "0"


def test_default_vlm_model_discovers_model_beside_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    model = runtime / "models" / "qwen3-vl-4b-instruct"
    model.mkdir(parents=True)
    (model / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.delenv("HPID_VLM_MODEL", raising=False)
    monkeypatch.delenv("HPID_RUNTIME_ROOT", raising=False)
    monkeypatch.setenv("HPID_HOME", str(tmp_path / "empty-home"))
    monkeypatch.setattr("hpid_split.webapp.sys.prefix", str(runtime / ".venv"))

    assert _default_vlm_model() == model


def test_default_asset_router_discovers_index_beside_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    router = runtime / "models" / "hpid_siglip2_asset_router_v1"
    router.mkdir(parents=True)
    (router / "index.json").write_text("{}", encoding="utf-8")
    monkeypatch.delenv("HPID_ASSET_ROUTER_INDEX", raising=False)
    monkeypatch.delenv("HPID_RUNTIME_ROOT", raising=False)
    monkeypatch.setenv("HPID_HOME", str(tmp_path / "empty-home"))
    monkeypatch.setattr("hpid_split.webapp.sys.prefix", str(runtime / ".venv"))

    assert _default_asset_router_index() == router


def test_default_retrieval_discovers_index_beside_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime"
    index = runtime / "models" / "hpid_paco_prototypes_v2" / "index.json"
    index.parent.mkdir(parents=True)
    index.write_text("{}", encoding="utf-8")
    monkeypatch.delenv("HPID_RETRIEVAL_INDEX", raising=False)
    monkeypatch.delenv("HPID_RUNTIME_ROOT", raising=False)
    monkeypatch.setenv("HPID_HOME", str(tmp_path / "empty-home"))
    monkeypatch.setattr("hpid_split.webapp.sys.prefix", str(runtime / ".venv"))

    assert _default_retrieval_index() == index


def test_default_vlm_model_prefers_explicit_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "custom-vlm"
    model.mkdir()
    monkeypatch.setenv("HPID_VLM_MODEL", str(model))

    assert _default_vlm_model() == model


def test_default_vlm_4bit_tracks_local_model_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("HPID_VLM_4BIT", raising=False)

    assert _default_vlm_4bit(tmp_path / "qwen3-vl-4b-instruct") is True
    assert _default_vlm_4bit(tmp_path / "qwen3-vl-2b-instruct") is False


def test_default_vlm_4bit_honors_explicit_disable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HPID_VLM_4BIT", "0")

    assert _default_vlm_4bit(tmp_path / "qwen3-vl-4b-instruct") is False


def test_web_command_exposes_prompt_guided_mode(tmp_path: Path) -> None:
    command = _build_auto_command(
        tmp_path / "input.png",
        tmp_path / "output",
        domain="tool_prop",
        quality="Fast",
        complete_hidden_regions=False,
        decomposition_mode="Prompt-guided",
        part_prompts="stock, magazine, trigger",
    )

    assert command[command.index("--decomposition-mode") + 1] == "prompt-guided"
    assert command[command.index("--part-prompts") + 1] == ("stock, magazine, trigger")
    assert command[command.index("--guided-backend") + 1] == "auto"


def test_web_prompt_guided_ensemble_omits_automatic_only_vlm(tmp_path: Path) -> None:
    vlm_model = tmp_path / "qwen3-vl-4b-instruct"
    vlm_model.mkdir()
    command = _build_auto_command(
        tmp_path / "input.png",
        tmp_path / "output",
        domain="tool_prop",
        quality="Ensemble",
        complete_hidden_regions=False,
        decomposition_mode="Prompt-guided",
        part_prompts="stock, magazine, trigger",
        vlm_model=vlm_model,
    )

    assert "--additional-grounding-model" not in command
    assert "--semantic-part-multimask" in command
    assert "--proposal-first-fast" not in command
    assert "--vlm-parts" not in command
    assert "--vlm-model" not in command


def test_web_command_can_select_an_asset_before_automatic_parts(
    tmp_path: Path,
) -> None:
    command = _build_auto_command(
        tmp_path / "input.png",
        tmp_path / "output",
        domain="container",
        quality="Fast",
        complete_hidden_regions=False,
        asset_prompt="serving tray",
    )

    assert command[command.index("--asset-prompt") + 1] == "serving tray"
    assert command[command.index("--decomposition-mode") + 1] == "automatic"


def test_web_command_can_select_one_prompted_instance_by_image_point(
    tmp_path: Path,
) -> None:
    command = _build_auto_command(
        tmp_path / "input.png",
        tmp_path / "output",
        domain="furniture",
        quality="Fast",
        complete_hidden_regions=False,
        asset_prompt="chair",
        target_point="42.5,87",
    )

    index = command.index("--target-point")
    assert command[index + 1 : index + 3] == ["42.5", "87.0"]


def test_web_heavy_ensemble_automatic_mode_loads_existing_retrieval_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HPID_ENABLE_HEAVY_ENSEMBLE", "1")
    index = tmp_path / "index.json"
    index.write_text("{}", encoding="utf-8")

    automatic = _build_auto_command(
        tmp_path / "input.png",
        tmp_path / "automatic",
        domain="auto",
        quality="Ensemble",
        complete_hidden_regions=False,
        retrieval_index=index,
    )
    guided = _build_auto_command(
        tmp_path / "input.png",
        tmp_path / "guided",
        domain="auto",
        quality="Ensemble",
        complete_hidden_regions=False,
        decomposition_mode="Prompt-guided",
        part_prompts="stock",
        retrieval_index=index,
    )

    assert automatic[automatic.index("--retrieval-index") + 1] == str(index)
    assert "--retrieval-index" not in guided


def test_web_fast_automatic_reuses_proposals_without_prototype_queries(
    tmp_path: Path,
) -> None:
    index = tmp_path / "index.json"
    index.write_text("{}", encoding="utf-8")

    automatic = _build_auto_command(
        tmp_path / "input.png",
        tmp_path / "automatic",
        domain="auto",
        quality="Fast",
        complete_hidden_regions=False,
        retrieval_index=index,
    )
    guided = _build_auto_command(
        tmp_path / "input.png",
        tmp_path / "guided",
        domain="auto",
        quality="Fast",
        complete_hidden_regions=False,
        decomposition_mode="Prompt-guided",
        part_prompts="stock",
        retrieval_index=index,
    )

    assert "--proposal-first-fast" in automatic
    assert "--retrieval-index" not in automatic
    assert "--proposal-first-fast" not in guided


def test_web_automatic_mode_exposes_asset_router_without_retrieval(
    tmp_path: Path,
) -> None:
    router = tmp_path / "router"
    router.mkdir()
    missing_retrieval = tmp_path / "missing-index.json"

    command = _build_auto_command(
        tmp_path / "input.png",
        tmp_path / "automatic",
        domain="auto",
        scope="Entire scene",
        quality="Fast",
        complete_hidden_regions=False,
        retrieval_index=missing_retrieval,
        asset_router_index=router,
    )

    assert command[command.index("--asset-router-index") + 1] == str(router)
    assert "--retrieval-index" not in command


def test_web_scene_mode_skips_object_prototype_retrieval(tmp_path: Path) -> None:
    index = tmp_path / "index.json"
    index.write_text("{}", encoding="utf-8")

    command = _build_auto_command(
        tmp_path / "input.png",
        tmp_path / "scene",
        domain="auto",
        scope="Entire scene",
        quality="Fast",
        complete_hidden_regions=False,
        retrieval_index=index,
    )

    assert "--root-mode" in command
    assert command[command.index("--root-mode") + 1] == "scene"
    assert "--retrieval-index" not in command


def test_web_launch_allows_only_configured_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured = tmp_path / "outside-working-directory" / "runs"
    monkeypatch.setenv("HPID_OUTPUT_ROOT", str(configured))

    resolved = _output_root()

    assert resolved == configured.resolve()
    assert resolved.is_dir()
    assert _launch_allowed_paths() == [str(resolved)]


def test_web_runtime_timeout_is_bounded_by_mode() -> None:
    assert (
        _runtime_timeout_seconds(
            quality="Fast", scope="Primary asset", complete_hidden_regions=False
        )
        == 120
    )
    assert (
        _runtime_timeout_seconds(
            quality="Fast", scope="Entire scene", complete_hidden_regions=False
        )
        == 150
    )
    assert (
        _runtime_timeout_seconds(
            quality="Ensemble", scope="Entire scene", complete_hidden_regions=True
        )
        == 300
    )
