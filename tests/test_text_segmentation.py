import json

import numpy as np
import torch
from PIL import Image

from hpid_split.fusion import MaskCandidate
from hpid_split.guided_prompts import parse_guided_prompts
from hpid_split.text_segmentation import Sam3TextPartGenerator


class _Batch(dict):
    def __init__(self, *, image: bool) -> None:
        super().__init__()
        if image:
            self["pixel_values"] = torch.zeros((1, 3, 16, 16))
            self["original_sizes"] = torch.tensor([[100, 100]])
        else:
            self["input_ids"] = torch.ones((1, 2), dtype=torch.long)

    def to(self, *args, **kwargs):
        del args, kwargs
        return self

    @property
    def pixel_values(self):
        return self["pixel_values"]


class _Processor:
    def __call__(self, *, images=None, text=None, return_tensors=None):
        del return_tensors
        return _Batch(image=images is not None and text is None)

    def post_process_instance_segmentation(self, *args, **kwargs):
        del args, kwargs
        mask = torch.zeros((1, 100, 100), dtype=torch.bool)
        mask[:, 25:65, 55:85] = True
        return [{"masks": mask, "scores": torch.tensor([0.91])}]


class _Model:
    def eval(self):
        return self

    def to(self, device):
        del device
        return self

    def get_vision_features(self, **kwargs):
        del kwargs
        return torch.zeros((1, 4, 4))

    def __call__(self, **kwargs):
        del kwargs
        return object()


def _root() -> MaskCandidate:
    mask = np.zeros((100, 100), dtype=bool)
    mask[10:90, 10:90] = True
    return MaskCandidate(
        semantic_name="tool_prop",
        semantic_parent="tool_prop",
        mask=mask,
        score=0.9,
        source="test/root",
        metadata={
            "root_origin": "test-origin",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
        },
    )


def test_sam3_text_masks_become_auditable_guided_hpid_candidates() -> None:
    generator = Sam3TextPartGenerator(
        device="cpu",
        processor=_Processor(),
        model=_Model(),
    )
    result = generator.generate(
        Image.new("RGB", (100, 100), "white"),
        [_root()],
        parse_guided_prompts("stock=buttstock|rear stock"),
    )

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.semantic_name == "tool_prop_guided_stock"
    assert candidate.semantic_parent == "tool_prop"
    assert candidate.metadata["guided_backend"] == "sam3-text"
    assert candidate.metadata["ground_truth_used"] is False
    json.dumps(result.diagnostics)
