import numpy as np
import pytest
from PIL import Image

from hpid_split.instances import PartInstance
from hpid_split.restoration import (
    BackendProvenance,
    CompletionOutput,
    CompletionRequest,
    DiffusersInpaintSamBackend,
    _resolve_config_path,
    complete_and_export_parts,
    visible_lock_compose,
)


def test_completion_config_paths_are_relative_to_config(tmp_path) -> None:
    config_path = tmp_path / "configs" / "completion.json"
    config_path.parent.mkdir()

    resolved = _resolve_config_path("../runtime/lama", config_path)

    assert resolved == (tmp_path / "runtime" / "lama").resolve()


def test_completion_config_rejects_undefined_environment_variable(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.delenv("HPID_MISSING_TEST_ROOT", raising=False)

    with pytest.raises(ValueError, match="undefined variable"):
        _resolve_config_path("${HPID_MISSING_TEST_ROOT}/lama", tmp_path / "config.json")


def test_visible_lock_uses_generation_only_in_hidden_region() -> None:
    source = Image.new("RGBA", (12, 10), (20, 80, 160, 255))
    visible = np.zeros((10, 12), dtype=bool)
    visible[3:7, 4:8] = True
    full = np.zeros_like(visible)
    full[2:8, 3:9] = True
    generated = np.zeros((10, 12, 4), dtype=np.uint8)
    generated[:] = (220, 30, 40, 255)
    output = visible_lock_compose(source, generated, visible, full)
    assert np.all(output[visible] == np.array([20, 80, 160, 255]))
    assert np.all(output[full & ~visible] == np.array([220, 30, 40, 255]))
    assert np.all(output[~full, 3] == 0)


def test_completion_budget_skips_unselected_part_ids(tmp_path) -> None:
    class SelectiveBackend:
        provenance = BackendProvenance(
            "test", "1", "", "", "test", is_hpid_split_method=True
        )

        def __init__(self) -> None:
            self.calls: list[int] = []

        def select_target_indices(self, instance_map, records):
            return frozenset({records[0].instance_index})

        def complete(self, request):
            self.calls.append(request.target.instance_index)
            visible = request.visible_mask
            return CompletionOutput(
                visible,
                np.asarray(request.image.convert("RGBA")),
                1.0,
                self.provenance,
                {
                    "status": "completed",
                    "accepted_occluder_instance_indices": [],
                },
            )

    image = Image.new("RGB", (20, 12), (80, 120, 160))
    instance_map = np.zeros((12, 20), dtype=np.uint16)
    instance_map[2:10, 1:9] = 1
    instance_map[2:10, 11:19] = 2
    records = [
        PartInstance(
            "asset/first/left/01",
            "first",
            "asset",
            1,
            "left",
            (1, 2, 9, 10),
            (5.0, 6.0),
            64,
        ),
        PartInstance(
            "asset/second/right/01",
            "second",
            "asset",
            2,
            "right",
            (11, 2, 19, 10),
            (15.0, 6.0),
            64,
        ),
    ]
    backend = SelectiveBackend()
    result = complete_and_export_parts(image, instance_map, records, tmp_path, backend)
    assert backend.calls == [1]
    assert result[2]["added_area_px"] == 0
    assert (
        result[2]["completion_metadata"]["status"]
        == "not_selected_by_completion_budget"
    )


def test_diffusion_backend_reports_its_actual_appearance_source(tmp_path) -> None:
    class Refiner:
        provenance = BackendProvenance("test mask refiner", "1", "", "", "test", False)

    image = Image.new("RGB", (12, 12), (80, 120, 160))
    instance_map = np.zeros((12, 12), dtype=np.uint16)
    instance_map[3:9, 3:9] = 1
    target = PartInstance(
        "asset/panel/center/01",
        "panel",
        "asset",
        1,
        "center",
        (3, 3, 9, 9),
        (6.0, 6.0),
        36,
    )
    backend = DiffusersInpaintSamBackend(
        "test/diffusion-inpaint",
        tmp_path / "cache",
        Refiner(),
        device="cpu",
        variant=None,
    )
    result = backend.complete(CompletionRequest(image, instance_map, target, (target,)))
    assert (
        result.metadata["hidden_appearance_source"]
        == "Stable Diffusion v1.5 inpainting"
    )
    assert "Stable Diffusion v1.5 inpainting" in result.metadata["amodal_shape_source"]
