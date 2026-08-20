import json
from types import MethodType

import numpy as np
from PIL import Image

from hpid_split.foundation import (
    Detection,
    FoundationCandidateGenerator,
    FoundationConfig,
    SegmentProposal,
)
from hpid_split.fusion import MaskCandidate
from hpid_split.guided_prompts import parse_guided_prompts


def _root(mask: np.ndarray) -> MaskCandidate:
    return MaskCandidate(
        semantic_name="tool_prop",
        semantic_parent="tool_prop",
        mask=mask,
        score=0.9,
        source="test/root",
        metadata={
            "root_origin": "test-root",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
        },
    )


def test_guided_prompt_parser_supports_aliases_and_unicode() -> None:
    specs = parse_guided_prompts(
        "stock = buttstock | rear stock, magazine\n扳机 = trigger"
    )

    assert [spec.label for spec in specs] == ["stock", "magazine", "扳机"]
    assert specs[0].phrases == ("buttstock", "rear stock")
    assert specs[2].slug.startswith("part_")
    assert len({spec.slug for spec in specs}) == 3


def test_guided_generation_is_root_bounded_and_auditable() -> None:
    generator = FoundationCandidateGenerator.__new__(FoundationCandidateGenerator)
    generator.config = FoundationConfig(crop_padding=0.0)
    generator.device = "cpu"

    def ground(self, image, phrases):
        del self, image, phrases
        return [Detection("buttstock", 0.91, (45, 20, 75, 70))]

    def segment(self, image, detections):
        del self, detections
        mask = np.zeros((image.height, image.width), dtype=bool)
        mask[20:70, 45:75] = True
        return [SegmentProposal(mask, 0.88)]

    generator._ground = MethodType(ground, generator)
    generator._segment_boxes = MethodType(segment, generator)

    root_mask = np.zeros((100, 100), dtype=bool)
    root_mask[10:90, 10:90] = True
    result = generator.generate_guided_parts(
        Image.new("RGB", (100, 100), "white"),
        [_root(root_mask)],
        parse_guided_prompts("stock = buttstock | rear stock"),
    )

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.semantic_name == "tool_prop_guided_stock"
    assert candidate.semantic_parent == "tool_prop"
    assert candidate.metadata["ground_truth_used"] is False
    assert not np.any(candidate.mask & ~root_mask)
    json.dumps(result.diagnostics)


def test_guided_generation_rejects_cross_model_semantic_disagreement() -> None:
    generator = FoundationCandidateGenerator.__new__(FoundationCandidateGenerator)
    generator.config = FoundationConfig(crop_padding=0.0)
    generator.device = "cpu"

    def ground(self, image, phrases):
        del self, image, phrases
        return [Detection("butt plate", 0.90, (20, 20, 70, 70))]

    def segment(self, image, detections):
        del self, detections
        mask = np.zeros((image.height, image.width), dtype=bool)
        mask[20:70, 20:70] = True
        return [SegmentProposal(mask, 0.92)]

    class RejectingDenseProposer:
        @staticmethod
        def score_regions(image, queries, *, top_fraction):
            del image, top_fraction
            return {
                key: {
                    "prompt": prompt,
                    "top_mean": 0.01,
                    "median": 0.009,
                    "contrast": 0.001,
                }
                for key, prompt, _ in queries
            }

    generator._ground = MethodType(ground, generator)
    generator._segment_boxes = MethodType(segment, generator)
    generator.dense_proposer = RejectingDenseProposer()
    root_mask = np.zeros((100, 100), dtype=bool)
    root_mask[10:90, 10:90] = True

    result = generator.generate_guided_parts(
        Image.new("RGB", (100, 100), "white"),
        [_root(root_mask)],
        parse_guided_prompts("butt_plate=butt plate"),
    )

    assert result.candidates == ()
    assert result.diagnostics["dense_semantic_gate_rejection_count"] == 1
    assert result.diagnostics["dense_semantic_gate_rows"][0]["accepted"] is False
