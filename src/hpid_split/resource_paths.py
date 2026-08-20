from __future__ import annotations

import os
from pathlib import Path

PACKAGE_RESOURCES = Path(__file__).resolve().parent / "resources"
DEFAULT_PROMPT_BANK = PACKAGE_RESOURCES / "general_asset_prompts.json"


def user_hpid_home() -> Path:
    return Path(
        os.environ.get("HPID_HOME", str(Path.home() / ".hpid-split"))
    ).expanduser()


def user_completion_config() -> Path:
    return user_hpid_home() / "lama_sam2_evidence.local.json"
