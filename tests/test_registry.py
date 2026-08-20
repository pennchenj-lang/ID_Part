import numpy as np

from hpid_split.instances import PartInstance
from hpid_split.registry import preserve_part_ids


def _record(part_id: str, index: int, x: float) -> PartInstance:
    return PartInstance(
        part_id, "wheel", "vehicle", index, "center", (0, 0, 4, 4), (x, 2.0), 16
    )


def test_registry_reuses_id_after_small_mask_change() -> None:
    previous_map = np.zeros((8, 12), dtype=np.uint16)
    previous_map[2:6, 2:6] = 1
    current_map = np.zeros_like(previous_map)
    current_map[2:6, 3:7] = 1
    updated = preserve_part_ids(
        current_map,
        [_record("vehicle/wheel/left/01", 1, 4.5)],
        previous_map,
        [_record("asset-wheel-A", 1, 3.5)],
    )
    assert updated[0].part_id == "asset-wheel-A"
