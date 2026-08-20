from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Taxonomy:
    """A fine-to-parent taxonomy used by both training and inference."""

    fine_names: tuple[str, ...]
    parent_names: tuple[str, ...]
    fine_to_parent: tuple[int, ...]
    detail_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.fine_names or self.fine_names[0] != "background":
            raise ValueError("fine_names[0] must be background")
        if not self.parent_names or self.parent_names[0] != "background":
            raise ValueError("parent_names[0] must be background")
        if len(self.fine_names) != len(self.fine_to_parent):
            raise ValueError("fine_to_parent must contain one entry per fine class")
        if any(
            value < 0 or value >= len(self.parent_names)
            for value in self.fine_to_parent
        ):
            raise ValueError("fine_to_parent contains an invalid parent index")

    @property
    def num_fine_classes(self) -> int:
        return len(self.fine_names)

    @property
    def num_parent_classes(self) -> int:
        return len(self.parent_names)

    @property
    def detail_ids(self) -> tuple[int, ...]:
        lookup = {name: index for index, name in enumerate(self.fine_names)}
        return tuple(lookup[name] for name in self.detail_names)

    @property
    def mapping_array(self) -> np.ndarray:
        return np.asarray(self.fine_to_parent, dtype=np.int64)

    def parent_target(self, fine_target: np.ndarray) -> np.ndarray:
        if (
            fine_target.min(initial=0) < 0
            or fine_target.max(initial=0) >= self.num_fine_classes
        ):
            raise ValueError("fine target contains an invalid class")
        return self.mapping_array[fine_target]

    def to_dict(self) -> dict[str, Any]:
        return {
            "fine_names": list(self.fine_names),
            "parent_names": list(self.parent_names),
            "fine_to_parent": list(self.fine_to_parent),
            "detail_names": list(self.detail_names),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Taxonomy:
        return cls(
            fine_names=tuple(str(value) for value in payload["fine_names"]),
            parent_names=tuple(str(value) for value in payload["parent_names"]),
            fine_to_parent=tuple(int(value) for value in payload["fine_to_parent"]),
            detail_names=tuple(str(value) for value in payload.get("detail_names", ())),
        )

    @classmethod
    def from_json(cls, path: Path) -> Taxonomy:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def child_ids(
        self, parent_id: int, *, include_parent_fallback: bool = True
    ) -> tuple[int, ...]:
        children = [
            fine_id
            for fine_id, mapped_parent in enumerate(self.fine_to_parent)
            if fine_id != 0 and mapped_parent == parent_id
        ]
        if (
            not include_parent_fallback
            and self.parent_names[parent_id] in self.fine_names
        ):
            fallback_id = self.fine_names.index(self.parent_names[parent_id])
            children = [value for value in children if value != fallback_id]
        return tuple(children)


def default_character_taxonomy() -> Taxonomy:
    fine_names = (
        "background",
        "skin",
        "eyes",
        "eyebrow",
        "eyelash",
        "hair",
        "front_hair",
        "back_hair",
        "side_hair",
        "upper_cloth",
        "collar",
        "torso_cloth",
        "sleeve",
        "cuff",
        "hem",
        "inner_cloth",
        "lower_cloth",
        "shoes",
        "shoe_upper",
        "shoe_sole",
        "shoe_tongue",
        "shoelace",
        "heel",
        "sock",
        "accessory",
    )
    parent_names = (
        "background",
        "skin",
        "hair",
        "upper_cloth",
        "lower_cloth",
        "shoes",
        "accessory",
    )
    parent_by_name = {
        "background": "background",
        "skin": "skin",
        "eyes": "skin",
        "eyebrow": "skin",
        "eyelash": "skin",
        "hair": "hair",
        "front_hair": "hair",
        "back_hair": "hair",
        "side_hair": "hair",
        "upper_cloth": "upper_cloth",
        "collar": "upper_cloth",
        "torso_cloth": "upper_cloth",
        "sleeve": "upper_cloth",
        "cuff": "upper_cloth",
        "hem": "upper_cloth",
        "inner_cloth": "upper_cloth",
        "lower_cloth": "lower_cloth",
        "shoes": "shoes",
        "shoe_upper": "shoes",
        "shoe_sole": "shoes",
        "shoe_tongue": "shoes",
        "shoelace": "shoes",
        "heel": "shoes",
        "sock": "shoes",
        "accessory": "accessory",
    }
    parent_lookup = {name: index for index, name in enumerate(parent_names)}
    detail_names = (
        "eyes",
        "eyebrow",
        "eyelash",
        "front_hair",
        "side_hair",
        "collar",
        "cuff",
        "hem",
        "inner_cloth",
        "shoe_upper",
        "shoe_sole",
        "shoe_tongue",
        "shoelace",
        "heel",
        "sock",
        "accessory",
    )
    return Taxonomy(
        fine_names=fine_names,
        parent_names=parent_names,
        fine_to_parent=tuple(
            parent_lookup[parent_by_name[name]] for name in fine_names
        ),
        detail_names=detail_names,
    )
