from pathlib import Path

import numpy as np

from hpid_split.taxonomy import default_character_taxonomy


def test_parent_mapping_is_total() -> None:
    taxonomy = default_character_taxonomy()
    labels = np.arange(taxonomy.num_fine_classes, dtype=np.uint8)
    parents = taxonomy.parent_target(labels)
    assert parents.shape == labels.shape
    assert parents[0] == 0
    assert parents.max() < taxonomy.num_parent_classes


def test_taxonomy_json_round_trip(tmp_path: Path) -> None:
    taxonomy = default_character_taxonomy()
    path = tmp_path / "taxonomy.json"
    taxonomy.to_json(path)
    restored = type(taxonomy).from_json(path)
    assert restored == taxonomy
