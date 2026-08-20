import numpy as np
from PIL import Image

from hpid_split.foundation import SegmentProposal
from hpid_split.vlm_roots import (
    VlmRootConfig,
    VlmRootProposer,
    build_root_localization_prompt,
    parse_root_plan,
)


def test_root_prompt_rejects_context_hosts_and_requests_detached_components() -> None:
    prompt = build_root_localization_prompt("headphones")

    assert "do not label the occluding person" in prompt
    assert "cable attached to headphones" in prompt
    assert "merely contains the target" in prompt


def test_parse_root_plan_clips_filters_and_suppresses_duplicate_boxes() -> None:
    response = """
    {"target":"crate","instances":[
      {"bbox_2d":[-20,100,600,900],"confidence":0.93,"visible":true},
      {"bbox_2d":[0,105,605,905],"confidence":0.88,"visible":true},
      {"bbox_2d":[700,100,710,120],"confidence":0.99,"visible":true},
      {"bbox_2d":[650,200,950,800],"confidence":0.20,"visible":true},
      {"bbox_2d":[620,200,980,850],"confidence":0.80,"visible":false}
    ]}
    """

    roots, diagnostics = parse_root_plan(
        response,
        image_size=(200, 100),
        config=VlmRootConfig(minimum_box_side_px=3),
    )

    assert [root.box_xyxy for root in roots] == [(0, 10, 120, 90)]
    assert diagnostics["rejection_counts"] == {
        "not_visible": 1,
        "invalid_box": 0,
        "low_confidence": 1,
        "too_small": 1,
        "duplicate": 1,
    }


def test_vlm_root_proposer_converts_verified_box_to_root_candidate() -> None:
    class Planner:
        backend_id = "fake-vlm"

        def generate_response(self, image: Image.Image, prompt: str) -> str:
            del image, prompt
            return (
                '{"target":"monitor","instances":['
                '{"bbox_2d":[100,200,900,800],"confidence":0.81,'
                '"visible":true}]}'
            )

    def segment_boxes(image, detections):
        mask = np.zeros((image.height, image.width), dtype=bool)
        x0, y0, x1, y1 = detections[0].box_xyxy
        mask[y0:y1, x0:x1] = True
        return [SegmentProposal(mask, 0.9, {"selected_index": 1})]

    result = VlmRootProposer(Planner(), segment_boxes).generate(
        Image.new("RGB", (100, 80)),
        target_label="monitor",
        semantic_name="device",
        selected_profile="display",
    )

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.semantic_name == candidate.semantic_parent == "device"
    assert candidate.metadata["root_query_mode"] == "vlm_target_localization"
    assert candidate.metadata["selected_part_profile"] == "display"
    assert candidate.metadata["sam_multimask_selection"]["selected_index"] == 1
    assert result.diagnostics["ground_truth_used"] is False
