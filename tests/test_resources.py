import hashlib
import json
from pathlib import Path

from hpid_split.cli import build_parser
from hpid_split.prompt_bank import PromptBank
from hpid_split.resource_paths import DEFAULT_PROMPT_BANK
from hpid_split.setup_completion import setup_completion_backend


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bundled_prompt_bank_matches_repository_copy() -> None:
    repository_copy = (
        Path(__file__).resolve().parents[1] / "configs" / "general_asset_prompts.json"
    )

    assert DEFAULT_PROMPT_BANK.is_file()
    assert _sha256(DEFAULT_PROMPT_BANK) == _sha256(repository_copy)
    repository_extension = repository_copy.with_name("paco_modern_extensions.json")
    bundled_extension = DEFAULT_PROMPT_BANK.with_name("paco_modern_extensions.json")
    assert bundled_extension.is_file()
    assert _sha256(bundled_extension) == _sha256(repository_extension)
    repository_game_extension = repository_copy.with_name("game_asset_extensions.json")
    bundled_game_extension = DEFAULT_PROMPT_BANK.with_name("game_asset_extensions.json")
    assert bundled_game_extension.is_file()
    assert _sha256(bundled_game_extension) == _sha256(repository_game_extension)


def test_auto_parser_defaults_to_bundled_prompt_bank() -> None:
    args = build_parser().parse_args(
        ["auto", "--image", "input.png", "--output", "output"]
    )

    assert args.prompt_bank == DEFAULT_PROMPT_BANK


def test_bundled_game_profiles_expose_scene_root_queries() -> None:
    bank = PromptBank.from_json(DEFAULT_PROMPT_BANK)
    domains = {domain.name: domain for domain in bank.domains}
    expected_profiles = {
        "tool_prop": {
            "melee_weapon",
            "bow_weapon",
            "shield",
            "throwable_weapon",
        },
        "container": {"game_loot_container"},
        "daily_object": {"armor"},
        "structure": {"game_structure", "bridge", "gate"},
        "natural_object": {
            "tree",
            "log_or_stump",
            "bush",
            "rock",
            "mushroom",
            "crystal",
        },
    }

    for domain_name, profile_names in expected_profiles.items():
        profiles = {
            profile.name: profile for profile in domains[domain_name].part_profiles
        }
        for profile_name in profile_names:
            assert profiles[profile_name].scene_root_query_groups


def test_setup_completion_writes_portable_local_config(tmp_path: Path) -> None:
    package_root = tmp_path / "lama"
    model_cache = tmp_path / "models"
    config_path = tmp_path / "completion.json"

    written = setup_completion_backend(
        package_root=package_root,
        model_cache=model_cache,
        config_path=config_path,
        skip_install=True,
    )
    payload = json.loads(written.read_text(encoding="utf-8"))

    assert written == config_path.resolve()
    assert Path(payload["package_root"]) == package_root.resolve()
    assert Path(payload["model_cache"]) == model_cache.resolve()
    assert payload["kind"] == "target-package-lama-sam2"
