from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _merge_extension(
    payload: dict[str, object],
    extension: dict[str, object],
) -> dict[str, object]:
    """Merge an auditable domain extension without mutating either input."""

    merged = deepcopy(payload)
    raw_domains = merged.get("domains")
    if not isinstance(raw_domains, list):
        raise TypeError("prompt-bank base must contain a domains list")
    domain_by_name = {
        str(row.get("name")): row
        for row in raw_domains
        if isinstance(row, dict) and row.get("name") is not None
    }
    raw_new_domains = extension.get("new_domains", [])
    if not isinstance(raw_new_domains, list):
        raise TypeError("prompt-bank extension new_domains must be a list")
    for raw_domain in raw_new_domains:
        if not isinstance(raw_domain, dict):
            raise TypeError("every extension domain must be an object")
        domain_name = str(raw_domain.get("name", ""))
        if not domain_name:
            raise ValueError("extension domain must have a name")
        if domain_name in domain_by_name:
            raise ValueError(f"extension duplicates domain {domain_name!r}")
        copied = deepcopy(raw_domain)
        raw_domains.append(copied)
        domain_by_name[domain_name] = copied
    raw_extensions = extension.get("domain_extensions")
    if not isinstance(raw_extensions, list):
        raise TypeError("prompt-bank extension must contain domain_extensions")
    for raw_extension in raw_extensions:
        if not isinstance(raw_extension, dict):
            raise TypeError("every domain extension must be an object")
        domain_name = str(raw_extension.get("name", ""))
        domain = domain_by_name.get(domain_name)
        if domain is None:
            raise ValueError(f"extension references unknown domain {domain_name!r}")
        parts = domain.setdefault("parts", [])
        if not isinstance(parts, list):
            raise TypeError(f"domain {domain_name!r} parts must be a list")
        existing_parts = {
            str(row.get("semantic_name")) for row in parts if isinstance(row, dict)
        }
        for raw_part in raw_extension.get("parts", []):
            if not isinstance(raw_part, dict):
                raise TypeError("extension parts must be objects")
            semantic_name = str(raw_part.get("semantic_name", ""))
            if semantic_name in existing_parts:
                raise ValueError(
                    f"extension duplicates part {semantic_name!r} in {domain_name!r}"
                )
            parts.append(deepcopy(raw_part))
            existing_parts.add(semantic_name)

        for field in (
            "root_prompts",
            "generic_root_prompts",
            "default_part_semantics",
        ):
            additions = raw_extension.get(field, [])
            if not isinstance(additions, list):
                raise TypeError(f"extension {field} must be a list")
            values = domain.setdefault(field, [])
            if not isinstance(values, list):
                raise TypeError(f"domain {domain_name!r} {field} must be a list")
            for addition in additions:
                value = str(addition)
                if field == "default_part_semantics" and value not in existing_parts:
                    raise ValueError(
                        f"extension default part {value!r} is unknown in "
                        f"{domain_name!r}"
                    )
                if value not in values:
                    values.append(value)

        profile_additions = raw_extension.get("profile_parts", {})
        if not isinstance(profile_additions, dict):
            raise TypeError("profile_parts must be an object")
        profiles = domain.setdefault("part_profiles", [])
        if not isinstance(profiles, list):
            raise TypeError(f"domain {domain_name!r} part_profiles must be a list")
        profile_by_name = {
            str(row.get("name")): row
            for row in profiles
            if isinstance(row, dict) and row.get("name") is not None
        }
        new_profiles = raw_extension.get("part_profiles", [])
        if not isinstance(new_profiles, list):
            raise TypeError("extension part_profiles must be a list")
        for raw_profile in new_profiles:
            if not isinstance(raw_profile, dict):
                raise TypeError("every extension part profile must be an object")
            profile_name = str(raw_profile.get("name", ""))
            if not profile_name:
                raise ValueError("extension part profile must have a name")
            if profile_name in profile_by_name:
                raise ValueError(
                    f"extension duplicates profile {profile_name!r} in {domain_name!r}"
                )
            referenced = [str(value) for value in raw_profile.get("parts", [])]
            for subtype in raw_profile.get("subtypes", []):
                if not isinstance(subtype, dict):
                    raise TypeError("extension profile subtypes must be objects")
                referenced.extend(str(value) for value in subtype.get("parts", []))
            unknown = sorted(set(referenced) - existing_parts)
            if unknown:
                raise ValueError(
                    f"extension profile {profile_name!r} references unknown "
                    f"parts {unknown!r}"
                )
            copied = deepcopy(raw_profile)
            profiles.append(copied)
            profile_by_name[profile_name] = copied
        for profile_name, additions in profile_additions.items():
            profile = profile_by_name.get(str(profile_name))
            if profile is None:
                raise ValueError(
                    f"extension references unknown profile {profile_name!r} "
                    f"in {domain_name!r}"
                )
            profile_parts = profile.setdefault("parts", [])
            if not isinstance(profile_parts, list) or not isinstance(additions, list):
                raise TypeError("profile part additions must be lists")
            for semantic_name in additions:
                value = str(semantic_name)
                if value not in existing_parts:
                    raise ValueError(
                        f"extension profile {profile_name!r} references unknown "
                        f"part {value!r}"
                    )
                if value not in profile_parts:
                    profile_parts.append(value)

        subtype_additions = raw_extension.get("subtype_parts", {})
        if not isinstance(subtype_additions, dict):
            raise TypeError("subtype_parts must be an object")
        for profile_name, additions_by_subtype in subtype_additions.items():
            profile = profile_by_name.get(str(profile_name))
            if profile is None:
                raise ValueError(
                    f"extension references unknown profile {profile_name!r} "
                    f"in {domain_name!r}"
                )
            if not isinstance(additions_by_subtype, dict):
                raise TypeError("subtype part additions must be an object")
            subtypes = profile.get("subtypes", [])
            if not isinstance(subtypes, list):
                raise TypeError("profile subtypes must be a list")
            subtype_by_name = {
                str(row.get("name")): row
                for row in subtypes
                if isinstance(row, dict) and row.get("name") is not None
            }
            profile_parts = profile.setdefault("parts", [])
            if not isinstance(profile_parts, list):
                raise TypeError("profile parts must be a list")
            for subtype_name, additions in additions_by_subtype.items():
                subtype = subtype_by_name.get(str(subtype_name))
                if subtype is None:
                    raise ValueError(
                        f"extension references unknown subtype {subtype_name!r} "
                        f"in profile {profile_name!r}"
                    )
                subtype_parts = subtype.setdefault("parts", [])
                if not isinstance(subtype_parts, list) or not isinstance(
                    additions, list
                ):
                    raise TypeError("subtype part additions must be lists")
                for semantic_name in additions:
                    value = str(semantic_name)
                    if value not in existing_parts:
                        raise ValueError(
                            f"extension subtype {subtype_name!r} references "
                            f"unknown part {value!r}"
                        )
                    if value not in profile_parts:
                        profile_parts.append(value)
                    if value not in subtype_parts:
                        subtype_parts.append(value)
    return merged


@dataclass(frozen=True)
class PartPrompt:
    semantic_name: str
    prompts: tuple[str, ...]
    dense_prompts: tuple[str, ...] = ()
    semantic_parent: str | None = None
    query_parent: str | None = None
    fallback_query_parent: str | None = None
    fallback_if_coverage_below: float = 0.25
    assembly_parent: str | None = None
    spatial_anchor: str | None = None
    spatial_relation: str | None = None
    spatial_tolerance: float = 0.05
    topology_anchor: str | None = None
    topology_relation: str | None = None
    topology_scale: float = 0.24
    aliases: tuple[str, ...] = ()
    planner_description: str = ""
    planner_exclusions: tuple[str, ...] = ()
    minimum_parent_fraction: float = 0.0001
    maximum_parent_fraction: float = 0.85
    fallback_maximum_parent_fraction: float | None = None
    minimum_parent_containment: float | None = None
    maximum_instances: int = 4
    detail: bool = False
    dense_fallback: bool = False
    appearance_anchor: str | None = None
    appearance_relation: str | None = None
    appearance_polarity: str = "dark"
    appearance_search_scale: float = 1.0
    appearance_minimum_contrast: float = 0.045
    axis_position: float | None = None
    axis_tolerance: float = 0.4
    priority: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_parent_fraction < self.maximum_parent_fraction:
            raise ValueError("invalid parent-area interval for part prompt")
        if (
            self.fallback_maximum_parent_fraction is not None
            and self.fallback_maximum_parent_fraction <= self.minimum_parent_fraction
        ):
            raise ValueError("invalid fallback parent-area maximum")
        if self.maximum_instances < 1:
            raise ValueError("maximum_instances must be positive")
        if self.priority <= 0.0:
            raise ValueError("part priority must be positive")
        if self.minimum_parent_containment is not None and not (
            0.0 <= self.minimum_parent_containment <= 1.0
        ):
            raise ValueError("minimum_parent_containment must be in [0, 1]")
        valid_relations = {None, "above", "below", "left_of", "right_of", "overlap"}
        if self.spatial_relation not in valid_relations:
            raise ValueError(f"invalid spatial relation: {self.spatial_relation!r}")
        if (self.spatial_anchor is None) != (self.spatial_relation is None):
            raise ValueError("spatial_anchor and spatial_relation must be set together")
        if self.spatial_tolerance < 0.0:
            raise ValueError("spatial_tolerance must be non-negative")
        if (self.topology_anchor is None) != (self.topology_relation is None):
            raise ValueError(
                "topology_anchor and topology_relation must be set together"
            )
        if self.topology_relation not in {None, "surround", "terminal_complement"}:
            raise ValueError(f"invalid topology relation: {self.topology_relation!r}")
        if self.topology_scale <= 0.0:
            raise ValueError("topology_scale must be positive")
        if not 0.0 <= self.fallback_if_coverage_below <= 1.0:
            raise ValueError("fallback_if_coverage_below must be in [0, 1]")
        if (self.appearance_anchor is None) != (self.appearance_relation is None):
            raise ValueError(
                "appearance_anchor and appearance_relation must be set together"
            )
        if self.appearance_relation not in {None, "above", "upper_boundary"}:
            raise ValueError(
                f"invalid appearance relation: {self.appearance_relation!r}"
            )
        if self.appearance_polarity not in {"dark", "light", "contrast"}:
            raise ValueError(
                f"invalid appearance polarity: {self.appearance_polarity!r}"
            )
        if self.appearance_search_scale <= 0.0:
            raise ValueError("appearance_search_scale must be positive")
        if not 0.0 <= self.appearance_minimum_contrast <= 1.0:
            raise ValueError("appearance_minimum_contrast must be in [0, 1]")
        if self.axis_position is not None and not -1.0 <= self.axis_position <= 1.0:
            raise ValueError("axis_position must be in [-1, 1]")
        if not 0.0 < self.axis_tolerance <= 2.0:
            raise ValueError("axis_tolerance must be in (0, 2]")

    @property
    def phrases(self) -> tuple[str, ...]:
        return (*self.prompts, *self.aliases, self.semantic_name.replace("_", " "))

    @property
    def dense_phrases(self) -> tuple[str, ...]:
        return self.dense_prompts or self.prompts[:1]


@dataclass(frozen=True)
class PartProfileOverride:
    """Category-conditioned geometry for one shared semantic part label."""

    semantic_name: str
    minimum_parent_fraction: float | None = None
    maximum_parent_fraction: float | None = None
    minimum_parent_containment: float | None = None
    maximum_instances: int | None = None
    axis_position: float | None = None
    axis_tolerance: float | None = None
    topology_anchor: str | None = None
    topology_relation: str | None = None
    topology_scale: float | None = None
    detail: bool | None = None
    priority: float | None = None

    @classmethod
    def from_dict(
        cls, semantic_name: str, payload: object
    ) -> PartProfileOverride:
        if not isinstance(payload, dict):
            raise TypeError("profile part override must be an object")

        def optional_float(name: str) -> float | None:
            value = payload.get(name)
            return float(value) if value is not None else None

        maximum_instances = payload.get("maximum_instances")
        return cls(
            semantic_name=semantic_name,
            minimum_parent_fraction=optional_float("minimum_parent_fraction"),
            maximum_parent_fraction=optional_float("maximum_parent_fraction"),
            minimum_parent_containment=optional_float("minimum_parent_containment"),
            maximum_instances=(
                int(maximum_instances) if maximum_instances is not None else None
            ),
            axis_position=optional_float("axis_position"),
            axis_tolerance=optional_float("axis_tolerance"),
            topology_anchor=(
                str(payload["topology_anchor"])
                if payload.get("topology_anchor") is not None
                else None
            ),
            topology_relation=(
                str(payload["topology_relation"])
                if payload.get("topology_relation") is not None
                else None
            ),
            topology_scale=optional_float("topology_scale"),
            detail=(
                bool(payload["detail"])
                if payload.get("detail") is not None
                else None
            ),
            priority=optional_float("priority"),
        )

    def apply(self, part: PartPrompt) -> PartPrompt:
        if part.semantic_name != self.semantic_name:
            raise ValueError(
                f"cannot apply override for {self.semantic_name!r} to "
                f"{part.semantic_name!r}"
            )
        updates = {
            name: value
            for name, value in (
                ("minimum_parent_fraction", self.minimum_parent_fraction),
                ("maximum_parent_fraction", self.maximum_parent_fraction),
                ("minimum_parent_containment", self.minimum_parent_containment),
                ("maximum_instances", self.maximum_instances),
                ("axis_position", self.axis_position),
                ("axis_tolerance", self.axis_tolerance),
                ("topology_anchor", self.topology_anchor),
                ("topology_relation", self.topology_relation),
                ("topology_scale", self.topology_scale),
                ("detail", self.detail),
                ("priority", self.priority),
            )
            if value is not None
        }
        return replace(part, **updates)


@dataclass(frozen=True)
class PartSubtype:
    """One object kind within a profile that shares a semantic namespace."""

    name: str
    root_hints: tuple[str, ...]
    part_semantics: tuple[str, ...]


@dataclass(frozen=True)
class PartProfile:
    """A category-specific subset of one broad asset-domain ontology."""

    name: str
    root_hints: tuple[str, ...]
    part_semantics: tuple[str, ...]
    root_query_groups: tuple[tuple[str, ...], ...] = ()
    part_subtypes: tuple[PartSubtype, ...] = ()
    classifier_prompt: str = ""
    scene_root_query_groups: tuple[tuple[str, ...], ...] = ()
    part_overrides: tuple[PartProfileOverride, ...] = ()
    confusion_groups: tuple[tuple[str, ...], ...] = ()
    requires_grounded_refinement: bool = False

    @staticmethod
    def _match_phrases(model_label: str, phrases: tuple[str, ...]) -> float:
        label = _normalize(model_label)
        if not label:
            return 0.0
        label_tokens = set(label.split())
        score = 0.0
        for hint in phrases:
            normalized = _normalize(hint)
            if not normalized:
                continue
            if label == normalized:
                candidate = 1.0
            elif normalized in label or label in normalized:
                candidate = 0.88
            else:
                tokens = set(normalized.split())
                candidate = len(label_tokens & tokens) / max(
                    1, len(label_tokens | tokens)
                )
            score = max(score, candidate)
        return score

    def match_score(self, model_label: str) -> float:
        return self._match_phrases(model_label, self.root_hints)

    def query_hints(self, model_label: str) -> tuple[str, ...]:
        """Return aliases for the one object kind named by a target prompt."""

        subtype, score = self.resolve_subtype(model_label)
        if subtype is not None and score > 0.0:
            return subtype.root_hints
        if not self.root_query_groups:
            return self.root_hints
        ranked = [
            (self._match_phrases(model_label, group), -index, group)
            for index, group in enumerate(self.root_query_groups)
        ]
        score, _, group = max(ranked, key=lambda item: (item[0], item[1]))
        return group if score > 0.0 else self.root_hints

    def resolve_subtype(self, model_label: str) -> tuple[PartSubtype | None, float]:
        """Resolve a concrete object kind without crossing profile boundaries."""

        if not self.part_subtypes:
            return None, 0.0
        ranked = [
            (self._match_phrases(model_label, subtype.root_hints), -index, subtype)
            for index, subtype in enumerate(self.part_subtypes)
        ]
        score, _, subtype = max(ranked, key=lambda item: (item[0], item[1]))
        return (subtype if score > 0.0 else None), float(score)

    def part_semantics_for(
        self, model_label: str
    ) -> tuple[tuple[str, ...], PartSubtype | None, float]:
        """Return the narrowest valid inventory for the named object kind."""

        subtype, score = self.resolve_subtype(model_label)
        if subtype is None:
            return self.part_semantics, None, score
        return subtype.part_semantics, subtype, score

    def apply_overrides(
        self, parts: tuple[PartPrompt, ...]
    ) -> tuple[PartPrompt, ...]:
        by_name = {override.semantic_name: override for override in self.part_overrides}
        return tuple(
            by_name[part.semantic_name].apply(part)
            if part.semantic_name in by_name
            else part
            for part in parts
        )

    def confusion_group_for(self, semantic_name: str) -> tuple[str, ...]:
        return next(
            (group for group in self.confusion_groups if semantic_name in group),
            (),
        )


@dataclass(frozen=True)
class DomainPrompt:
    name: str
    root_prompts: tuple[str, ...]
    parts: tuple[PartPrompt, ...]
    classifier_prompt: str = ""
    generic_root_prompts: tuple[str, ...] = ()
    default_part_semantics: tuple[str, ...] = ()
    part_profiles: tuple[PartProfile, ...] = ()

    def root_label_specificity(self, model_label: str) -> float:
        """Measure how specifically a detector label names this asset domain."""

        label = _normalize(model_label)
        if not label:
            return 0.0
        label_tokens = set(label.split())
        score = 0.0
        generic = {_normalize(value) for value in self.generic_root_prompts}
        for phrase in self.root_prompts:
            normalized = _normalize(phrase)
            if not normalized:
                continue
            if label == normalized:
                candidate = 1.0
            elif normalized in label or label in normalized:
                candidate = 0.88
            else:
                tokens = set(normalized.split())
                candidate = len(label_tokens & tokens) / max(
                    1, len(label_tokens | tokens)
                )
            if normalized in generic:
                candidate *= 0.35
            score = max(score, candidate)
        return score

    def match_part(
        self,
        model_label: str,
        parts: tuple[PartPrompt, ...] | list[PartPrompt] | None = None,
    ) -> PartPrompt | None:
        label = _normalize(model_label)
        if not label:
            return None
        label_tokens = set(label.split())
        scores: list[tuple[float, PartPrompt]] = []
        for part in self.parts if parts is None else parts:
            part_score = 0.0
            for phrase in part.phrases:
                normalized = _normalize(phrase)
                if not normalized:
                    continue
                if label == normalized:
                    score = 1.0
                elif normalized in label or label in normalized:
                    score = 0.86
                else:
                    tokens = set(normalized.split())
                    score = len(label_tokens & tokens) / max(
                        1, len(label_tokens | tokens)
                    )
                part_score = max(part_score, score)
            scores.append((part_score, part))
        scores.sort(key=lambda item: item[0], reverse=True)
        if not scores or scores[0][0] < 0.34:
            return None
        if (
            len(scores) > 1
            and scores[0][0] < 0.98
            and scores[0][0] - scores[1][0] < 0.10
        ):
            return None
        return scores[0][1]

    def select_parts(
        self,
        model_label: str,
        *,
        profile_hint: str | None = None,
        profile_hint_source: str | None = None,
    ) -> tuple[tuple[PartPrompt, ...], str | None, dict[str, object]]:
        """Select a conservative part inventory from the detected root label."""

        if not self.part_profiles:
            return (
                self.parts,
                None,
                {
                    "algorithm": "hpid-domain-part-profile-v1",
                    "selected_profile": None,
                    "selection_reason": "domain_has_no_profiles",
                    "root_label": model_label,
                },
            )
        if profile_hint is not None:
            matching_profile = next(
                (
                    profile
                    for profile in self.part_profiles
                    if profile.name == profile_hint
                ),
                None,
            )
            if matching_profile is None:
                raise ValueError(
                    f"domain {self.name!r} has no part profile {profile_hint!r}"
                )
            best_profile = matching_profile
            best_score = 1.0
            second_score = 0.0
            accepted = True
        else:
            ranked = sorted(
                (
                    (profile.match_score(model_label), profile)
                    for profile in self.part_profiles
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            best_score, best_profile = ranked[0]
            second_score = ranked[1][0] if len(ranked) > 1 else 0.0
            accepted = best_score >= 0.50 and (
                best_score >= 0.88 or best_score - second_score >= 0.18
            )
        selected_semantics: set[str]
        selected_subtype: PartSubtype | None = None
        subtype_score = 0.0
        if accepted:
            profile_semantics, selected_subtype, subtype_score = (
                best_profile.part_semantics_for(model_label)
            )
            selected_semantics = set(profile_semantics)
            selected_semantics.update(
                semantic
                for semantic in self.default_part_semantics
                if semantic.endswith("_body")
            )
        elif self.default_part_semantics:
            selected_semantics = set(self.default_part_semantics)
        else:
            return (
                self.parts,
                None,
                {
                    "algorithm": "hpid-domain-part-profile-v1",
                    "selected_profile": None,
                    "selection_reason": "ambiguous_profile_all_parts_fallback",
                    "root_label": model_label,
                    "best_profile": best_profile.name,
                    "best_score": best_score,
                    "second_score": second_score,
                },
            )

        by_name = {part.semantic_name: part for part in self.parts}
        pending = list(selected_semantics)
        while pending:
            semantic = pending.pop()
            part = by_name.get(semantic)
            if part is None:
                continue
            for parent in (
                part.semantic_parent,
                part.query_parent,
                part.fallback_query_parent,
                part.assembly_parent,
            ):
                if (
                    parent is not None
                    and parent != self.name
                    and parent not in selected_semantics
                ):
                    selected_semantics.add(parent)
                    pending.append(parent)
        selected = tuple(
            part for part in self.parts if part.semantic_name in selected_semantics
        )
        if accepted:
            selected = best_profile.apply_overrides(selected)
        return (
            selected,
            (best_profile.name if accepted else None),
            {
                "algorithm": "hpid-domain-part-profile-v1",
                "selected_profile": best_profile.name if accepted else None,
                "selection_reason": (
                    (
                        profile_hint_source or "explicit_profile_hint"
                        if profile_hint is not None
                        else "root_label_match"
                    )
                    if accepted
                    else "default_inventory_fallback"
                ),
                "root_label": model_label,
                "profile_hint": profile_hint,
                "best_profile": best_profile.name,
                "best_score": best_score,
                "second_score": second_score,
                "selected_subtype": (
                    selected_subtype.name if selected_subtype is not None else None
                ),
                "subtype_score": subtype_score,
                "subtype_root_hints": (
                    list(selected_subtype.root_hints)
                    if selected_subtype is not None
                    else []
                ),
                "selected_part_semantics": [part.semantic_name for part in selected],
                "profile_part_overrides": [
                    override.semantic_name for override in best_profile.part_overrides
                ]
                if accepted
                else [],
            },
        )

    def profile_specificity(self, model_label: str) -> float:
        if not self.part_profiles:
            return 0.0
        return max(profile.match_score(model_label) for profile in self.part_profiles)


@dataclass(frozen=True)
class PromptBank:
    domains: tuple[DomainPrompt, ...]

    def __post_init__(self) -> None:
        domain_names = [domain.name for domain in self.domains]
        if len(domain_names) != len(set(domain_names)):
            raise ValueError("prompt-bank domain names must be unique")
        owners: dict[str, str] = {}
        for domain in self.domains:
            if not domain.root_prompts:
                raise ValueError(f"domain {domain.name!r} has no root prompts")
            unknown_generic_roots = set(domain.generic_root_prompts) - set(
                domain.root_prompts
            )
            if unknown_generic_roots:
                raise ValueError(
                    f"domain {domain.name!r} has generic root prompts outside "
                    f"root_prompts: {sorted(unknown_generic_roots)}"
                )
            part_names = {part.semantic_name for part in domain.parts}
            unknown_defaults = set(domain.default_part_semantics) - part_names
            if unknown_defaults:
                raise ValueError(
                    f"domain {domain.name!r} default inventory references unknown "
                    f"parts: {sorted(unknown_defaults)}"
                )
            profile_names: set[str] = set()
            for profile in domain.part_profiles:
                if profile.name in profile_names:
                    raise ValueError(
                        f"domain {domain.name!r} has duplicate part profile "
                        f"{profile.name!r}"
                    )
                profile_names.add(profile.name)
                if not profile.root_hints:
                    raise ValueError(f"part profile {profile.name!r} has no root hints")
                for group in (
                    *profile.root_query_groups,
                    *profile.scene_root_query_groups,
                ):
                    if not group:
                        raise ValueError(
                            f"part profile {profile.name!r} has an empty root query group"
                        )
                    unknown_hints = set(group) - set(profile.root_hints)
                    if unknown_hints:
                        raise ValueError(
                            f"part profile {profile.name!r} root query group "
                            f"references unknown hints: {sorted(unknown_hints)}"
                        )
                subtype_names: set[str] = set()
                for subtype in profile.part_subtypes:
                    if subtype.name in subtype_names:
                        raise ValueError(
                            f"part profile {profile.name!r} has duplicate subtype "
                            f"{subtype.name!r}"
                        )
                    subtype_names.add(subtype.name)
                    if not subtype.root_hints:
                        raise ValueError(
                            f"part subtype {subtype.name!r} has no root hints"
                        )
                    unknown_hints = set(subtype.root_hints) - set(profile.root_hints)
                    if unknown_hints:
                        raise ValueError(
                            f"part subtype {subtype.name!r} references unknown root "
                            f"hints: {sorted(unknown_hints)}"
                        )
                    unknown_parts = set(subtype.part_semantics) - set(
                        profile.part_semantics
                    )
                    if unknown_parts:
                        raise ValueError(
                            f"part subtype {subtype.name!r} references parts outside "
                            f"profile {profile.name!r}: {sorted(unknown_parts)}"
                        )
                unknown = set(profile.part_semantics) - part_names
                if unknown:
                    raise ValueError(
                        f"part profile {profile.name!r} references unknown parts: "
                        f"{sorted(unknown)}"
                    )
                override_names = [
                    override.semantic_name for override in profile.part_overrides
                ]
                if len(override_names) != len(set(override_names)):
                    raise ValueError(
                        f"part profile {profile.name!r} has duplicate part overrides"
                    )
                unknown_overrides = set(override_names) - set(profile.part_semantics)
                if unknown_overrides:
                    raise ValueError(
                        f"part profile {profile.name!r} overrides parts outside the "
                        f"profile: {sorted(unknown_overrides)}"
                    )
                base_by_name = {
                    part.semantic_name: part for part in domain.parts
                }
                for override in profile.part_overrides:
                    effective = override.apply(base_by_name[override.semantic_name])
                    if (
                        effective.topology_anchor is not None
                        and effective.topology_anchor not in profile.part_semantics
                    ):
                        raise ValueError(
                            f"part profile {profile.name!r} override for "
                            f"{override.semantic_name!r} references topology "
                            f"anchor outside the profile: "
                            f"{effective.topology_anchor!r}"
                        )
                grouped_semantics: set[str] = set()
                for group in profile.confusion_groups:
                    if len(group) < 2:
                        raise ValueError(
                            f"part profile {profile.name!r} confusion groups need "
                            "at least two semantics"
                        )
                    unknown_group = set(group) - set(profile.part_semantics)
                    if unknown_group:
                        raise ValueError(
                            f"part profile {profile.name!r} confusion group "
                            f"references parts outside the profile: "
                            f"{sorted(unknown_group)}"
                        )
                    repeated = grouped_semantics & set(group)
                    if repeated:
                        raise ValueError(
                            f"part profile {profile.name!r} repeats semantics across "
                            f"confusion groups: {sorted(repeated)}"
                        )
                    grouped_semantics.update(group)
            parent_by_name = {
                part.semantic_name: part.semantic_parent or domain.name
                for part in domain.parts
            }
            query_parent_by_name = {
                part.semantic_name: (
                    part.query_parent or part.semantic_parent or domain.name
                )
                for part in domain.parts
            }
            for part in domain.parts:
                owner = owners.setdefault(part.semantic_name, domain.name)
                if owner != domain.name:
                    raise ValueError(
                        f"semantic name {part.semantic_name!r} is shared by "
                        f"{owner!r} and {domain.name!r}; use namespaced names"
                    )
                parent = parent_by_name[part.semantic_name]
                if parent != domain.name and parent not in part_names:
                    raise ValueError(
                        f"part {part.semantic_name!r} references unknown parent "
                        f"{parent!r}"
                    )
                query_parent = query_parent_by_name[part.semantic_name]
                if query_parent != domain.name and query_parent not in part_names:
                    raise ValueError(
                        f"part {part.semantic_name!r} references unknown query "
                        f"parent {query_parent!r}"
                    )
                if (
                    part.fallback_query_parent is not None
                    and part.fallback_query_parent != domain.name
                    and part.fallback_query_parent not in part_names
                ):
                    raise ValueError(
                        f"part {part.semantic_name!r} references unknown fallback "
                        f"query parent {part.fallback_query_parent!r}"
                    )
                assembly_parent = part.assembly_parent or parent
                if assembly_parent != domain.name and assembly_parent not in part_names:
                    raise ValueError(
                        f"part {part.semantic_name!r} references unknown assembly "
                        f"parent {assembly_parent!r}"
                    )
                if (
                    part.spatial_anchor is not None
                    and part.spatial_anchor != domain.name
                    and part.spatial_anchor not in part_names
                ):
                    raise ValueError(
                        f"part {part.semantic_name!r} references unknown spatial "
                        f"anchor {part.spatial_anchor!r}"
                    )
                if (
                    part.topology_anchor is not None
                    and part.topology_anchor != domain.name
                    and part.topology_anchor not in part_names
                ):
                    raise ValueError(
                        f"part {part.semantic_name!r} references unknown topology "
                        f"anchor {part.topology_anchor!r}"
                    )
                if (
                    part.appearance_anchor is not None
                    and part.appearance_anchor != domain.name
                    and part.appearance_anchor not in part_names
                ):
                    raise ValueError(
                        f"part {part.semantic_name!r} references unknown appearance "
                        f"anchor {part.appearance_anchor!r}"
                    )
                if part.appearance_anchor == part.semantic_name:
                    raise ValueError("a part cannot use itself as an appearance anchor")

            def visit(
                name: str,
                active: frozenset[str] = frozenset(),
                *,
                domain_name: str = domain.name,
                parent_map: dict[str, str] = parent_by_name,
            ) -> None:
                if name in active:
                    raise ValueError(
                        f"domain {domain_name!r} contains a semantic-parent cycle"
                    )
                parent = parent_map.get(name, domain_name)
                if parent != domain_name:
                    visit(
                        parent,
                        active | {name},
                        domain_name=domain_name,
                        parent_map=parent_map,
                    )

            for part_name in part_names:
                visit(part_name)

            def visit_query(
                name: str,
                active: frozenset[str] = frozenset(),
                *,
                domain_name: str = domain.name,
                parent_map: dict[str, str] = query_parent_by_name,
            ) -> None:
                if name in active:
                    raise ValueError(
                        f"domain {domain_name!r} contains a query-parent cycle"
                    )
                parent = parent_map.get(name, domain_name)
                if parent != domain_name:
                    visit_query(
                        parent,
                        active | {name},
                        domain_name=domain_name,
                        parent_map=parent_map,
                    )

            for part_name in part_names:
                visit_query(part_name)

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> PromptBank:
        domains: list[DomainPrompt] = []
        raw_domains = payload.get("domains")
        if not isinstance(raw_domains, list):
            raise TypeError("prompt bank must contain a domains list")
        for raw_domain in raw_domains:
            if not isinstance(raw_domain, dict):
                raise TypeError("each prompt-bank domain must be an object")
            parts = []
            for raw_part in raw_domain.get("parts", []):
                if not isinstance(raw_part, dict):
                    raise TypeError("each part prompt must be an object")
                parts.append(
                    PartPrompt(
                        semantic_name=str(raw_part["semantic_name"]),
                        prompts=tuple(str(value) for value in raw_part["prompts"]),
                        dense_prompts=tuple(
                            str(value) for value in raw_part.get("dense_prompts", ())
                        ),
                        semantic_parent=(
                            str(raw_part["semantic_parent"])
                            if raw_part.get("semantic_parent") is not None
                            else None
                        ),
                        query_parent=(
                            str(raw_part["query_parent"])
                            if raw_part.get("query_parent") is not None
                            else None
                        ),
                        fallback_query_parent=(
                            str(raw_part["fallback_query_parent"])
                            if raw_part.get("fallback_query_parent") is not None
                            else None
                        ),
                        fallback_if_coverage_below=float(
                            raw_part.get("fallback_if_coverage_below", 0.25)
                        ),
                        assembly_parent=(
                            str(raw_part["assembly_parent"])
                            if raw_part.get("assembly_parent") is not None
                            else None
                        ),
                        spatial_anchor=(
                            str(raw_part["spatial_anchor"])
                            if raw_part.get("spatial_anchor") is not None
                            else None
                        ),
                        spatial_relation=(
                            str(raw_part["spatial_relation"])
                            if raw_part.get("spatial_relation") is not None
                            else None
                        ),
                        spatial_tolerance=float(
                            raw_part.get("spatial_tolerance", 0.05)
                        ),
                        topology_anchor=(
                            str(raw_part["topology_anchor"])
                            if raw_part.get("topology_anchor") is not None
                            else None
                        ),
                        topology_relation=(
                            str(raw_part["topology_relation"])
                            if raw_part.get("topology_relation") is not None
                            else None
                        ),
                        topology_scale=float(raw_part.get("topology_scale", 0.24)),
                        aliases=tuple(
                            str(value) for value in raw_part.get("aliases", [])
                        ),
                        planner_description=str(
                            raw_part.get("planner_description", "")
                        ).strip(),
                        planner_exclusions=tuple(
                            str(value)
                            for value in raw_part.get("planner_exclusions", ())
                        ),
                        minimum_parent_fraction=float(
                            raw_part.get("minimum_parent_fraction", 0.0001)
                        ),
                        maximum_parent_fraction=float(
                            raw_part.get("maximum_parent_fraction", 0.85)
                        ),
                        fallback_maximum_parent_fraction=(
                            float(raw_part["fallback_maximum_parent_fraction"])
                            if raw_part.get("fallback_maximum_parent_fraction")
                            is not None
                            else None
                        ),
                        minimum_parent_containment=(
                            float(raw_part["minimum_parent_containment"])
                            if raw_part.get("minimum_parent_containment") is not None
                            else None
                        ),
                        maximum_instances=int(raw_part.get("maximum_instances", 4)),
                        detail=bool(raw_part.get("detail", False)),
                        dense_fallback=bool(raw_part.get("dense_fallback", False)),
                        appearance_anchor=(
                            str(raw_part["appearance_anchor"])
                            if raw_part.get("appearance_anchor") is not None
                            else None
                        ),
                        appearance_relation=(
                            str(raw_part["appearance_relation"])
                            if raw_part.get("appearance_relation") is not None
                            else None
                        ),
                        appearance_polarity=str(
                            raw_part.get("appearance_polarity", "dark")
                        ),
                        appearance_search_scale=float(
                            raw_part.get("appearance_search_scale", 1.0)
                        ),
                        appearance_minimum_contrast=float(
                            raw_part.get("appearance_minimum_contrast", 0.045)
                        ),
                        axis_position=(
                            float(raw_part["axis_position"])
                            if raw_part.get("axis_position") is not None
                            else None
                        ),
                        axis_tolerance=float(raw_part.get("axis_tolerance", 0.4)),
                        priority=float(raw_part.get("priority", 1.0)),
                    )
                )
            domains.append(
                DomainPrompt(
                    name=str(raw_domain["name"]),
                    root_prompts=tuple(
                        str(value) for value in raw_domain["root_prompts"]
                    ),
                    parts=tuple(parts),
                    classifier_prompt=str(
                        raw_domain.get("classifier_prompt", "")
                    ).strip(),
                    generic_root_prompts=tuple(
                        str(value)
                        for value in raw_domain.get("generic_root_prompts", ())
                    ),
                    default_part_semantics=tuple(
                        str(value)
                        for value in raw_domain.get("default_part_semantics", ())
                    ),
                    part_profiles=tuple(
                        PartProfile(
                            name=str(raw_profile["name"]),
                            root_hints=tuple(
                                str(value)
                                for value in raw_profile.get("root_hints", ())
                            ),
                            part_semantics=tuple(
                                str(value) for value in raw_profile.get("parts", ())
                            ),
                            classifier_prompt=str(
                                raw_profile.get("classifier_prompt", "")
                            ).strip(),
                            root_query_groups=tuple(
                                tuple(str(value) for value in group)
                                for group in raw_profile.get("root_query_groups", ())
                                if isinstance(group, list)
                            ),
                            scene_root_query_groups=tuple(
                                tuple(str(value) for value in group)
                                for group in raw_profile.get(
                                    "scene_root_query_groups", ()
                                )
                                if isinstance(group, list)
                            ),
                            part_overrides=tuple(
                                PartProfileOverride.from_dict(
                                    str(semantic_name), override
                                )
                                for semantic_name, override in raw_profile.get(
                                    "part_overrides", {}
                                ).items()
                            ),
                            confusion_groups=tuple(
                                tuple(str(value) for value in group)
                                for group in raw_profile.get(
                                    "confusion_groups", ()
                                )
                                if isinstance(group, list)
                            ),
                            requires_grounded_refinement=bool(
                                raw_profile.get(
                                    "requires_grounded_refinement", False
                                )
                            ),
                            part_subtypes=tuple(
                                PartSubtype(
                                    name=str(raw_subtype["name"]),
                                    root_hints=tuple(
                                        str(value)
                                        for value in raw_subtype.get("root_hints", ())
                                    ),
                                    part_semantics=tuple(
                                        str(value)
                                        for value in raw_subtype.get("parts", ())
                                    ),
                                )
                                for raw_subtype in raw_profile.get("subtypes", ())
                                if isinstance(raw_subtype, dict)
                            ),
                        )
                        for raw_profile in raw_domain.get("part_profiles", ())
                        if isinstance(raw_profile, dict)
                    ),
                )
            )
        return cls(tuple(domains))

    @classmethod
    def from_json(cls, path: Path) -> PromptBank:
        path = path.resolve()
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        includes = payload.pop("include", [])
        if not isinstance(includes, list):
            raise TypeError("prompt-bank include must be a list")
        seen = {path}
        for raw_include in includes:
            include_path = (path.parent / str(raw_include)).resolve()
            if include_path in seen:
                raise ValueError(f"duplicate prompt-bank include: {include_path}")
            seen.add(include_path)
            extension = json.loads(include_path.read_text(encoding="utf-8-sig"))
            payload = _merge_extension(payload, extension)
        return cls.from_dict(payload)
