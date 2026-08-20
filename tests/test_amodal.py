import numpy as np
from PIL import Image

from hpid_split.amodal import complete_instances
from hpid_split.instances import PartInstance


def test_amodal_completion_never_changes_visible_pixels() -> None:
    image = Image.new("RGB", (32, 24), "white")
    instance_map = np.zeros((24, 32), dtype=np.uint16)
    instance_map[7:18, 4:14] = 1
    instance_map[7:18, 14:20] = 2
    record = PartInstance(
        "asset/body/left/01",
        "body",
        "asset",
        1,
        "left",
        (4, 7, 14, 18),
        (8.5, 12.0),
        110,
    )
    completed, _ = complete_instances(image, instance_map, [record])
    assert np.all(completed[0].full_mask[instance_map == 1])
    assert completed[0].added_area_px >= 0
    assert 0.0 <= completed[0].completion_confidence <= 1.0
