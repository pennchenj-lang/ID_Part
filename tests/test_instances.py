import numpy as np

from hpid_split.instances import semantic_to_part_ids
from hpid_split.taxonomy import default_character_taxonomy


def test_part_ids_are_deterministic_and_side_aware() -> None:
    taxonomy = default_character_taxonomy()
    eyes = taxonomy.fine_names.index("eyes")
    labels = np.zeros((32, 48), dtype=np.uint8)
    labels[8:14, 7:15] = eyes
    labels[8:14, 33:41] = eyes
    first_map, first = semantic_to_part_ids(labels, taxonomy, minimum_area=4)
    second_map, second = semantic_to_part_ids(labels.copy(), taxonomy, minimum_area=4)
    assert np.array_equal(first_map, second_map)
    assert [item.part_id for item in first] == [item.part_id for item in second]
    assert [item.side for item in first] == ["left", "right"]
