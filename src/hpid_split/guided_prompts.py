from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GuidedPromptSpec:
    """One user-named part and the phrases used to locate it."""

    label: str
    slug: str
    phrases: tuple[str, ...]
    maximum_instances: int = 8


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if normalized:
        return normalized[:64]
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    return f"part_{digest}"


def parse_guided_prompts(text: str, *, maximum_prompts: int = 64) -> tuple[GuidedPromptSpec, ...]:
    """Parse comma/newline prompts, with optional ``label = phrase | alias`` syntax."""

    raw_items = re.split(r"[\n,;，；]+", text)
    specs: list[GuidedPromptSpec] = []
    seen_phrases: set[tuple[str, ...]] = set()
    slug_counts: dict[str, int] = {}
    for raw_item in raw_items:
        item = raw_item.strip()
        if not item:
            continue
        if len(item) > 300:
            raise ValueError("each guided part prompt must be at most 300 characters")
        if "=" in item:
            raw_label, raw_phrases = item.split("=", maxsplit=1)
            label = raw_label.strip()
            phrases = tuple(
                phrase.strip()
                for phrase in raw_phrases.split("|")
                if phrase.strip()
            )
        else:
            phrases = tuple(
                phrase.strip() for phrase in item.split("|") if phrase.strip()
            )
            label = phrases[0] if phrases else ""
        if not label or not phrases:
            raise ValueError(f"invalid guided part prompt: {item!r}")
        normalized_phrases = tuple(
            dict.fromkeys(re.sub(r"\s+", " ", phrase).strip() for phrase in phrases)
        )
        phrase_key = tuple(phrase.casefold() for phrase in normalized_phrases)
        if phrase_key in seen_phrases:
            continue
        seen_phrases.add(phrase_key)
        base_slug = _slug(label)
        slug_counts[base_slug] = slug_counts.get(base_slug, 0) + 1
        suffix = slug_counts[base_slug]
        slug = base_slug if suffix == 1 else f"{base_slug}_{suffix:02d}"
        specs.append(
            GuidedPromptSpec(
                label=label,
                slug=slug,
                phrases=normalized_phrases,
            )
        )
        if len(specs) > maximum_prompts:
            raise ValueError(f"at most {maximum_prompts} guided part prompts are allowed")
    return tuple(specs)
