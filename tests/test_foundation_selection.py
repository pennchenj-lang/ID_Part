import numpy as np
import torch
from PIL import Image

from hpid_split.foundation import (
    AutomaticAssetQuery,
    Detection,
    FoundationCandidateGenerator,
    FoundationConfig,
    SegmentProposal,
    _profile_confusion_reassignment,
    _remove_ambiguous_guided_candidates,
    _select_child_detections,
    _select_root_detections,
    _select_semantic_multimask_index,
    _terminal_complement_mask,
)
from hpid_split.fusion import MaskCandidate
from hpid_split.prompt_bank import (
    DomainPrompt,
    PartProfile,
    PartProfileOverride,
    PartPrompt,
    PromptBank,
)


def _mask(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((100, 100), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _part(name: str, *, maximum_instances: int = 4) -> PartPrompt:
    return PartPrompt(
        semantic_name=name,
        prompts=(name,),
        maximum_instances=maximum_instances,
    )


def test_semantic_multimask_selection_can_prefer_a_cleaner_near_tie() -> None:
    selected, diagnostics = _select_semantic_multimask_index(
        np.asarray([0.8774, 0.7544, 0.8716], dtype=np.float32),
        np.asarray([11877, 2384, 8545], dtype=np.int64),
        [
            {"probability": 0.4583, "rank": 1},
            {"probability": 0.2485, "rank": 1},
            {"probability": 0.4758, "rank": 1},
        ],
    )

    assert selected == 2
    assert diagnostics["baseline_index"] == 0
    assert diagnostics["selection_changed"] is True
    assert diagnostics["ground_truth_used"] is False


def test_terminal_complement_recovers_ordered_endpoint_remainder() -> None:
    root = np.zeros((80, 120), dtype=bool)
    root[30:50, 10:110] = True
    blade = np.zeros_like(root)
    blade[30:50, 35:110] = True

    complement, diagnostics = _terminal_complement_mask(
        root,
        blade,
        anchor_position=-0.55,
        target_position=0.85,
    )

    expected = np.zeros_like(root)
    expected[30:50, 10:35] = True
    assert np.array_equal(complement, expected)
    assert diagnostics["component_count"] == 1


def test_semantic_multimask_selection_rejects_a_tiny_semantic_fragment() -> None:
    selected, diagnostics = _select_semantic_multimask_index(
        np.asarray([0.88, 0.84], dtype=np.float32),
        np.asarray([10000, 1200], dtype=np.int64),
        [
            {"probability": 0.35, "rank": 1},
            {"probability": 0.90, "rank": 1},
        ],
    )

    assert selected == 0
    assert diagnostics["candidate_eligible"] == [True, False]


def test_ground_lazily_reactivates_an_offloaded_detector() -> None:
    class FakeInputs(dict[str, torch.Tensor]):
        def __init__(self) -> None:
            super().__init__(
                pixel_values=torch.zeros((1, 3, 4, 4)),
                input_ids=torch.ones((1, 2), dtype=torch.long),
            )
            self.moved_to: str | None = None

        @property
        def input_ids(self) -> torch.Tensor:
            return self["input_ids"]

        def to(self, device: str) -> "FakeInputs":
            self.moved_to = device
            return self

    class FakeProcessor:
        def __init__(self) -> None:
            self.inputs: FakeInputs | None = None

        def __call__(self, **_: object) -> FakeInputs:
            self.inputs = FakeInputs()
            return self.inputs

        def post_process_grounded_object_detection(
            self, *_: object, **__: object
        ) -> list[dict[str, object]]:
            return [
                {
                    "boxes": torch.tensor([[1.0, 1.0, 3.0, 3.0]]),
                    "scores": torch.tensor([0.9]),
                    "text_labels": ["part"],
                }
            ]

    class FakeModel:
        def __init__(self) -> None:
            self.devices: list[str] = []
            self.active = False

        def to(self, device: str) -> "FakeModel":
            self.devices.append(device)
            self.active = True
            return self

        def eval(self) -> "FakeModel":
            return self

        def __call__(self, **_: object) -> object:
            assert self.active
            return object()

    generator = FoundationCandidateGenerator.__new__(FoundationCandidateGenerator)
    generator.device = "cpu"
    generator.config = FoundationConfig()
    generator.grounding_processor = FakeProcessor()
    generator.grounding_model = FakeModel()
    generator.dense_proposer = None
    generator._grounding_model_active = False
    generator._grounding_output_mismatch_count = 0

    detections = generator._ground(Image.new("RGB", (4, 4)), ["part"])

    assert generator.grounding_model.devices == ["cpu"]
    assert generator.grounding_processor.inputs is not None
    assert generator.grounding_processor.inputs.moved_to == "cpu"
    assert len(detections) == 1
    assert detections[0].label == "part"
    assert np.isclose(detections[0].score, 0.9)
    assert detections[0].box_xyxy == (1, 1, 3, 3)


def test_dense_fallback_is_controlled_by_part_opt_in_not_part_size() -> None:
    config = FoundationConfig()

    assert config.dense_require_opt_in is True
    assert config.dense_detail_only is False


def test_ambiguity_filter_preserves_legitimate_nested_part_masks() -> None:
    root_metadata = {"root_origin": "test", "root_index": 1}
    panel = MaskCandidate(
        "device_base_panel",
        "device",
        _mask(10, 90, 10, 90),
        0.60,
        "test",
        metadata=root_metadata,
    )
    keyboard = MaskCandidate(
        "device_keyboard",
        "device_body",
        _mask(45, 75, 20, 80),
        0.58,
        "test",
        metadata=root_metadata,
    )

    retained, rejected = _remove_ambiguous_guided_candidates([panel, keyboard])

    assert retained == [panel, keyboard]
    assert rejected == 0


def test_ambiguity_filter_rejects_same_scale_competing_semantics() -> None:
    metadata = {"root_origin": "test", "root_index": 1}
    mask = _mask(20, 80, 20, 80)
    first = MaskCandidate(
        "tool_prop_stock", "tool_prop", mask, 0.60, "test", metadata=metadata
    )
    second = MaskCandidate(
        "tool_prop_barrel", "tool_prop", mask, 0.57, "test", metadata=metadata
    )

    retained, rejected = _remove_ambiguous_guided_candidates([first, second])

    assert retained == []
    assert rejected == 2


def test_semantic_quota_prevents_one_class_from_consuming_budget() -> None:
    arm = _part("arm")
    eye = _part("eye")
    mapped = [
        (Detection("arm", 0.95, (0, 0, 10, 10)), arm),
        (Detection("arm", 0.90, (12, 0, 22, 10)), arm),
        (Detection("eye", 0.45, (5, 15, 9, 19)), eye),
    ]
    selected, diagnostics = _select_child_detections(
        mapped,
        limit=2,
        nms_iou=0.76,
        use_semantic_quota=True,
    )
    assert {part.semantic_name for _, part in selected} == {"arm", "eye"}
    assert diagnostics["selected_semantic_count"] == 2


def test_root_selection_preserves_distinct_prompt_hypotheses() -> None:
    detections = [
        Detection("handheld object", 0.91, (0, 0, 20, 20)),
        Detection("handheld object", 0.82, (25, 0, 45, 20)),
        Detection("staff", 0.80, (5, 5, 70, 90)),
        Detection("weapon spear", 0.74, (40, 10, 95, 45)),
    ]

    selected = _select_root_detections(detections, threshold=0.70, limit=3)

    assert [item.label for item in selected] == [
        "handheld object",
        "staff",
        "weapon spear",
    ]


def test_scene_root_discovery_runs_declared_profile_query_groups() -> None:
    class SceneGenerator(FoundationCandidateGenerator):
        def __init__(self) -> None:
            self.config = FoundationConfig(
                use_scene_profile_root_queries=True,
                maximum_roots_per_domain=8,
                maximum_total_roots=8,
            )
            self.prompt_bank = PromptBank(
                (
                    DomainPrompt(
                        name="natural_object",
                        root_prompts=("natural object",),
                        parts=(),
                        part_profiles=(
                            PartProfile(
                                "tree",
                                ("tree", "pine tree"),
                                (),
                                scene_root_query_groups=(("tree", "pine tree"),),
                            ),
                        ),
                    ),
                )
            )
            self.queries: list[tuple[str, ...]] = []
            self._asset_prompt_diagnostics = None

        def _ground(self, image: Image.Image, phrases: list[str]) -> list[Detection]:
            self.queries.append(tuple(phrases))
            if "pine tree" in phrases:
                return [
                    Detection("pine tree", 0.82, (4, 4, 20, 28)),
                    Detection("pine tree", 0.79, (28, 4, 44, 28)),
                ]
            return []

        def _segment_boxes(
            self, image: Image.Image, detections: list[Detection]
        ) -> list[SegmentProposal]:
            output = []
            for detection in detections:
                mask = np.zeros((image.height, image.width), dtype=bool)
                x0, y0, x1, y1 = detection.box_xyxy
                mask[y0:y1, x0:x1] = True
                output.append(SegmentProposal(mask, 0.9))
            return output

    generator = SceneGenerator()
    roots = generator._root_candidates(Image.new("RGB", (48, 32)))

    assert ("tree", "pine tree") in generator.queries
    assert len(roots) == 2


def test_full_image_asset_proposal_adds_an_isolated_root_query() -> None:
    class ProposalGenerator(FoundationCandidateGenerator):
        def __init__(self) -> None:
            self.config = FoundationConfig(
                automatic_asset_queries=(
                    AutomaticAssetQuery(
                        label="guitar",
                        domain="tool_prop",
                        profile="guitar",
                        score=0.27,
                    ),
                ),
                maximum_roots_per_domain=4,
                maximum_total_roots=4,
            )
            self.prompt_bank = PromptBank(
                (
                    DomainPrompt(
                        name="tool_prop",
                        root_prompts=("generic tool",),
                        parts=(),
                        part_profiles=(PartProfile("guitar", ("guitar",), ()),),
                    ),
                )
            )
            self.queries: list[tuple[str, ...]] = []
            self._asset_prompt_diagnostics = None
            self._automatic_asset_diagnostics = None

        def _ground(self, image: Image.Image, phrases: list[str]) -> list[Detection]:
            self.queries.append(tuple(phrases))
            return (
                [Detection("guitar", 0.81, (4, 6, 44, 28))]
                if phrases == ["guitar"]
                else []
            )

        def _segment_boxes(
            self, image: Image.Image, detections: list[Detection]
        ) -> list[SegmentProposal]:
            output = []
            for detection in detections:
                mask = np.zeros((image.height, image.width), dtype=bool)
                x0, y0, x1, y1 = detection.box_xyxy
                mask[y0:y1, x0:x1] = True
                output.append(SegmentProposal(mask, 0.92))
            return output

    generator = ProposalGenerator()
    roots = generator._root_candidates(Image.new("RGB", (48, 32)))

    assert ("guitar",) in generator.queries
    assert len(roots) == 1
    assert roots[0].query_mode == "global_asset_proposal"
    assert roots[0].domain.name == "tool_prop"
    assert roots[0].profile_hint == "guitar"
    assert generator._automatic_asset_diagnostics["accepted_root_count"] == 1


def test_root_selection_deduplicates_synonyms_before_spending_budget() -> None:
    detections = [
        Detection("watch", 0.92, (0, 0, 100, 100)),
        Detection("wristwatch", 0.88, (0, 0, 100, 100)),
        Detection("smartwatch", 0.84, (1, 0, 100, 100)),
        Detection("analog watch", 0.80, (0, 1, 100, 100)),
        Detection("wristwatch", 0.71, (18, 8, 82, 84)),
    ]

    selected = _select_root_detections(detections, threshold=0.70, limit=4)

    assert [item.box_xyxy for item in selected] == [
        (0, 0, 100, 100),
        (18, 8, 82, 84),
    ]


def test_root_domain_evidence_compares_categories_in_one_embedding_space() -> None:
    class FakeDenseProposer:
        def rank_regions_labels(
            self,
            image: Image.Image,
            regions: list[tuple[str, np.ndarray]],
            labels: list[tuple[str, str]],
            *,
            masked_weight: float,
        ) -> dict[str, dict[str, dict[str, object]]]:
            assert masked_weight == 0.90
            assert {label for label, _ in labels} == {"character", "tool_prop"}
            return {
                key: {
                    "character": {
                        "prompt": "person",
                        "full_similarity": 0.1,
                        "masked_similarity": 0.1,
                        "combined_similarity": 0.1,
                        "probability": 0.2,
                        "rank": 2,
                    },
                    "tool_prop": {
                        "prompt": "weapon",
                        "full_similarity": 0.3,
                        "masked_similarity": 0.3,
                        "combined_similarity": 0.3,
                        "probability": 0.8,
                        "rank": 1,
                    },
                }
                for key, _ in regions
            }

    mask = np.zeros((40, 60), dtype=bool)
    mask[5:35, 4:56] = True
    candidates = [
        MaskCandidate(
            "character",
            "character",
            mask,
            0.9,
            "test/root",
            metadata={
                "root_origin": "test",
                "root_index": 1,
                "candidate_key": "root:1",
                "parent_candidate_key": None,
            },
        ),
        MaskCandidate(
            "tool_prop",
            "tool_prop",
            mask,
            0.6,
            "test/root",
            metadata={
                "root_origin": "test",
                "root_index": 2,
                "candidate_key": "root:2",
                "parent_candidate_key": None,
            },
        ),
    ]
    generator = object.__new__(FoundationCandidateGenerator)
    generator.dense_proposer = FakeDenseProposer()
    generator.prompt_bank = PromptBank(
        (
            DomainPrompt(
                "character",
                ("person",),
                (),
                classifier_prompt="person",
            ),
            DomainPrompt(
                "tool_prop",
                ("weapon",),
                (),
                classifier_prompt="weapon",
            ),
        )
    )

    enriched, diagnostics = generator.attach_root_domain_evidence(
        Image.new("RGB", (60, 40)), candidates
    )

    assert enriched[0].metadata["domain_evidence_score"] == 0.2
    assert enriched[0].metadata["domain_evidence_rank"] == 2
    assert enriched[1].metadata["domain_evidence_score"] == 0.8
    assert np.isclose(enriched[1].metadata["domain_evidence_contrast"], 0.2)
    assert diagnostics["algorithm"] == "clipseg-embedding-root-arbitration-v3"


def test_child_selection_removes_duplicate_boxes_and_honors_instance_cap() -> None:
    wheel = _part("wheel", maximum_instances=2)
    mapped = [
        (Detection("wheel", 0.95, (0, 0, 10, 10)), wheel),
        (Detection("tire", 0.90, (0, 0, 10, 10)), wheel),
        (Detection("wheel", 0.85, (20, 0, 30, 10)), wheel),
        (Detection("wheel", 0.80, (40, 0, 50, 10)), wheel),
    ]
    selected, diagnostics = _select_child_detections(
        mapped,
        limit=8,
        nms_iou=0.76,
        use_semantic_quota=True,
    )
    assert len(selected) == 2
    assert {item[0].box_xyxy for item in selected} == {
        (0, 0, 10, 10),
        (20, 0, 30, 10),
    }
    assert diagnostics["post_nms_detection_count"] == 2


def test_hierarchy_generator_queries_children_inside_detected_parent() -> None:
    class FakeGenerator(FoundationCandidateGenerator):
        def __init__(self) -> None:
            self.config = FoundationConfig(crop_padding=0.0)

        def _ground(self, image: Image.Image, phrases: list[str]) -> list[Detection]:
            if "asset body" in phrases:
                return [Detection("asset body", 0.92, (4, 4, 52, 52))]
            if "button" in phrases:
                return [Detection("button", 0.84, (10, 10, 18, 18))]
            return []

        def _segment_boxes(
            self, image: Image.Image, detections: list[Detection]
        ) -> list[SegmentProposal]:
            masks = []
            for detection in detections:
                mask = np.zeros((image.height, image.width), dtype=bool)
                x0, y0, x1, y1 = detection.box_xyxy
                mask[y0:y1, x0:x1] = True
                masks.append(SegmentProposal(mask, 0.9))
            return masks

    domain = DomainPrompt(
        name="asset",
        root_prompts=("object",),
        parts=(
            PartPrompt(
                semantic_name="asset_body",
                prompts=("asset body",),
                maximum_parent_fraction=0.90,
                maximum_instances=1,
            ),
            PartPrompt(
                semantic_name="asset_button",
                semantic_parent="asset",
                query_parent="asset_body",
                assembly_parent="asset_body",
                prompts=("button",),
                maximum_parent_fraction=0.20,
                maximum_instances=1,
            ),
        ),
    )
    root_mask = np.zeros((64, 64), dtype=bool)
    root_mask[4:60, 4:60] = True
    candidates, diagnostics = FakeGenerator()._child_candidates(
        Image.new("RGB", (64, 64)),
        domain,
        root_index=1,
        root_box=(4, 4, 60, 60),
        root_mask=root_mask,
    )
    assert [candidate.semantic_name for candidate in candidates] == [
        "asset_body",
        "asset_button",
    ]
    assert candidates[1].semantic_parent == "asset"
    assert (
        candidates[1].metadata["parent_candidate_key"]
        == candidates[0].metadata["candidate_key"]
    )
    assert candidates[1].metadata["assembly_parent_semantic"] == "asset_body"
    assert (
        candidates[1].metadata["assembly_parent_candidate_key"]
        == candidates[0].metadata["candidate_key"]
    )
    assert diagnostics["hierarchy_call_count"] == 2


def test_hierarchy_generator_queries_only_selected_part_profile() -> None:
    class ProfileGenerator(FoundationCandidateGenerator):
        def __init__(self) -> None:
            self.config = FoundationConfig(crop_padding=0.0)
            self.queried: list[tuple[str, ...]] = []

        def _ground(self, image: Image.Image, phrases: list[str]) -> list[Detection]:
            self.queried.append(tuple(phrases))
            if "screen" in phrases:
                return [Detection("screen", 0.9, (4, 4, 28, 28))]
            return []

        def _segment_boxes(
            self, image: Image.Image, detections: list[Detection]
        ) -> list[SegmentProposal]:
            output = []
            for detection in detections:
                mask = np.zeros((image.height, image.width), dtype=bool)
                x0, y0, x1, y1 = detection.box_xyxy
                mask[y0:y1, x0:x1] = True
                output.append(SegmentProposal(mask, 0.9))
            return output

    domain = DomainPrompt(
        name="device",
        root_prompts=("device",),
        parts=(
            PartPrompt("device_body", ("device body",)),
            PartPrompt("device_screen", ("screen",)),
            PartPrompt("device_keyboard", ("keyboard",)),
        ),
        default_part_semantics=("device_body",),
        part_profiles=(
            PartProfile("phone", ("phone",), ("device_screen",)),
            PartProfile("laptop", ("laptop",), ("device_keyboard",)),
        ),
    )
    generator = ProfileGenerator()
    root_mask = np.ones((32, 32), dtype=bool)

    candidates, diagnostics = generator._child_candidates(
        Image.new("RGB", (32, 32)),
        domain,
        root_index=1,
        root_box=(0, 0, 32, 32),
        root_mask=root_mask,
        root_label="cell phone",
    )

    assert diagnostics["selected_part_profile"] == "phone"
    assert all("keyboard" not in phrases for phrases in generator.queried)
    assert any(candidate.semantic_name == "device_screen" for candidate in candidates)


def test_routed_profile_refinement_uses_isolated_canonical_queries() -> None:
    class ProfileRefinementGenerator(FoundationCandidateGenerator):
        def __init__(self) -> None:
            self.config = FoundationConfig(crop_padding=0.0)
            self.device = "cpu"
            self.dense_proposer = None
            self.queried: list[tuple[str, ...]] = []

        def _ground(self, image: Image.Image, phrases: list[str]) -> list[Detection]:
            self.queried.append(tuple(phrases))
            if "screen" in phrases:
                return [Detection("screen", 0.92, (8, 8, 56, 44))]
            if "button" in phrases:
                return [Detection("button", 0.84, (26, 48, 38, 56))]
            return []

        def _segment_boxes(
            self, image: Image.Image, detections: list[Detection]
        ) -> list[SegmentProposal]:
            output = []
            for detection in detections:
                mask = np.zeros((image.height, image.width), dtype=bool)
                x0, y0, x1, y1 = detection.box_xyxy
                mask[y0:y1, x0:x1] = True
                output.append(SegmentProposal(mask, 0.9))
            return output

    domain = DomainPrompt(
        name="device",
        root_prompts=("cell phone",),
        parts=(
            PartPrompt("device_body", ("device body",)),
            PartPrompt(
                "device_screen",
                ("screen",),
                semantic_parent="device",
                maximum_instances=1,
            ),
            PartPrompt(
                "device_button",
                ("button",),
                semantic_parent="device",
                maximum_parent_fraction=0.12,
                maximum_instances=2,
            ),
            PartPrompt(
                "device_keyboard",
                ("keyboard",),
                semantic_parent="device",
            ),
        ),
        default_part_semantics=("device_body",),
        part_profiles=(
            PartProfile(
                "phone",
                ("cell phone",),
                ("device_screen", "device_button"),
            ),
            PartProfile("laptop", ("laptop",), ("device_keyboard",)),
        ),
    )
    root_mask = np.ones((64, 64), dtype=bool)
    root = MaskCandidate(
        semantic_name="device",
        semantic_parent="device",
        mask=root_mask,
        score=0.9,
        source="test/root",
        prompt="cell phone",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
            "selected_part_profile": "phone",
            "profile_resolution_status": "accepted",
        },
    )
    generator = ProfileRefinementGenerator()

    result = generator.refine_profile_parts(
        Image.new("RGB", (64, 64)), [root], {"device": domain}
    )

    assert {candidate.semantic_name for candidate in result.candidates} == {
        "device_screen",
        "device_button",
    }
    assert all("keyboard" not in query for query in generator.queried)
    assert all(len(query) <= 2 for query in generator.queried)
    assert result.diagnostics["algorithm"] == "hpid-routed-profile-refinement-v1"
    assert result.diagnostics["ground_truth_used"] is False


def test_profile_refinement_keeps_alternative_boxes_until_geometry_validation() -> (
    None
):
    class HypothesisGenerator(FoundationCandidateGenerator):
        def __init__(self) -> None:
            self.config = FoundationConfig(
                crop_padding=0.0,
                part_box_padding_ratio=0.0,
                part_detection_hypotheses_per_instance=3,
            )
            self.device = "cpu"
            self.dense_proposer = None

        def _ground(self, image: Image.Image, phrases: list[str]) -> list[Detection]:
            del image, phrases
            return [
                Detection("blade", 0.92, (2, 2, 98, 98)),
                Detection("blade", 0.74, (55, 35, 95, 65)),
            ]

        def _segment_boxes(
            self, image: Image.Image, detections: list[Detection]
        ) -> list[SegmentProposal]:
            output = []
            for detection in detections:
                mask = np.zeros((image.height, image.width), dtype=bool)
                x0, y0, x1, y1 = detection.box_xyxy
                mask[y0:y1, x0:x1] = True
                output.append(SegmentProposal(mask, 0.9))
            return output

    blade = PartPrompt(
        "tool_prop_blade",
        ("blade",),
        semantic_parent="tool_prop",
        maximum_parent_fraction=0.45,
        maximum_instances=1,
    )
    domain = DomainPrompt(
        name="tool_prop",
        root_prompts=("knife",),
        parts=(blade,),
        part_profiles=(
            PartProfile("knife", ("knife",), ("tool_prop_blade",)),
        ),
    )
    root = MaskCandidate(
        semantic_name="tool_prop",
        semantic_parent="tool_prop",
        mask=np.ones((100, 100), dtype=bool),
        score=0.9,
        source="test/root",
        prompt="knife",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
            "selected_part_profile": "knife",
            "profile_resolution_status": "accepted",
        },
    )

    result = HypothesisGenerator().refine_profile_parts(
        Image.new("RGB", (100, 100)),
        [root],
        {"tool_prop": domain},
    )

    assert len(result.candidates) == 1
    assert result.candidates[0].semantic_name == "tool_prop_blade"
    assert result.candidates[0].metadata["box_xyxy_local"] == [55, 35, 95, 65]
    row = result.diagnostics["roots"][0]
    assert row["detected_by_semantic"]["tool_prop_blade"] == 2
    assert row["geometry_rejection_count"] == 1


def test_profile_refinement_reassigns_only_unique_geometry_compatible_role() -> None:
    class PanelGenerator(FoundationCandidateGenerator):
        def __init__(self) -> None:
            self.config = FoundationConfig(
                crop_padding=0.0,
                part_box_padding_ratio=0.0,
            )
            self.device = "cpu"
            self.dense_proposer = None

        def _ground(self, image: Image.Image, phrases: list[str]) -> list[Detection]:
            if "screen" in phrases:
                return [Detection("screen", 0.80, (10, 15, 90, 85))]
            return []

        def _segment_boxes(
            self, image: Image.Image, detections: list[Detection]
        ) -> list[SegmentProposal]:
            output = []
            for detection in detections:
                mask = np.zeros((image.height, image.width), dtype=bool)
                x0, y0, x1, y1 = detection.box_xyxy
                mask[y0:y1, x0:x1] = True
                output.append(SegmentProposal(mask, 0.95))
            return output

    parts = (
        PartPrompt("device_door", ("door",), semantic_parent="device"),
        PartPrompt("device_control_panel", ("control panel",), semantic_parent="device"),
        PartPrompt("device_screen", ("screen",), semantic_parent="device"),
    )
    profile = PartProfile(
        "microwave",
        ("microwave",),
        tuple(part.semantic_name for part in parts),
        part_overrides=(
            PartProfileOverride(
                "device_door",
                minimum_parent_fraction=0.20,
                maximum_parent_fraction=0.80,
            ),
            PartProfileOverride(
                "device_control_panel",
                minimum_parent_fraction=0.01,
                maximum_parent_fraction=0.18,
            ),
            PartProfileOverride(
                "device_screen",
                minimum_parent_fraction=0.001,
                maximum_parent_fraction=0.10,
            ),
        ),
        confusion_groups=((
            "device_door",
            "device_control_panel",
            "device_screen",
        ),),
    )
    domain = DomainPrompt(
        name="device",
        root_prompts=("microwave",),
        parts=parts,
        part_profiles=(profile,),
    )
    root = MaskCandidate(
        "device",
        "device",
        np.ones((100, 100), dtype=bool),
        0.9,
        "test/root",
        prompt="microwave",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
            "selected_part_profile": "microwave",
            "profile_resolution_status": "accepted",
        },
    )

    result = PanelGenerator().refine_profile_parts(
        Image.new("RGB", (100, 100)), [root], {"device": domain}
    )

    assert [candidate.semantic_name for candidate in result.candidates] == [
        "device_door"
    ]
    candidate = result.candidates[0]
    assert candidate.metadata["detector_semantic_name"] == "device_screen"
    assert candidate.metadata["profile_semantic_reassignment"] is True
    row = result.diagnostics["roots"][0]
    assert row["profile_semantic_reassignment_count"] == 1
    assert row["profile_semantic_reassignment_rows"][0]["ground_truth_used"] is False


def test_profile_confusion_reassignment_stays_unresolved_with_two_valid_roles() -> None:
    source = PartPrompt(
        "device_screen",
        ("screen",),
        maximum_parent_fraction=0.10,
    )
    door = PartPrompt(
        "device_door",
        ("door",),
        minimum_parent_fraction=0.20,
        maximum_parent_fraction=0.80,
    )
    panel = PartPrompt(
        "device_control_panel",
        ("panel",),
        minimum_parent_fraction=0.20,
        maximum_parent_fraction=0.70,
    )
    profile = PartProfile(
        "appliance",
        ("appliance",),
        ("device_screen", "device_door", "device_control_panel"),
        confusion_groups=((
            "device_screen",
            "device_door",
            "device_control_panel",
        ),),
    )

    assert (
        _profile_confusion_reassignment(
            source,
            (source, door, panel),
            profile,
            root_area_fraction=0.50,
            root_containment=1.0,
            default_minimum_containment=0.48,
        )
        is None
    )


def test_profile_refinement_confines_sam_object_leakage_to_part_box() -> None:
    class LeakySamGenerator(FoundationCandidateGenerator):
        def __init__(self) -> None:
            self.config = FoundationConfig(
                crop_padding=0.0,
                part_box_padding_ratio=0.0,
            )
            self.device = "cpu"
            self.dense_proposer = None

        def _ground(self, image: Image.Image, phrases: list[str]) -> list[Detection]:
            return [Detection("screen", 0.92, (20, 20, 40, 36))]

        def _segment_boxes(
            self, image: Image.Image, detections: list[Detection]
        ) -> list[SegmentProposal]:
            return [
                SegmentProposal(
                    np.ones((image.height, image.width), dtype=bool),
                    0.90,
                )
                for _ in detections
            ]

    domain = DomainPrompt(
        name="device",
        root_prompts=("phone",),
        parts=(
            PartPrompt(
                "device_screen",
                ("screen",),
                semantic_parent="device",
                maximum_parent_fraction=0.25,
            ),
        ),
        part_profiles=(
            PartProfile("phone", ("phone",), ("device_screen",)),
        ),
    )
    root = MaskCandidate(
        "device",
        "device",
        np.ones((64, 64), dtype=bool),
        0.9,
        "test/root",
        prompt="phone",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
            "selected_part_profile": "phone",
            "profile_resolution_status": "accepted",
        },
    )

    result = LeakySamGenerator().refine_profile_parts(
        Image.new("RGB", (64, 64)), [root], {"device": domain}
    )

    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.semantic_name == "device_screen"
    assert int(np.count_nonzero(candidate.mask)) == 24 * 20
    assert candidate.metadata["sam_unconstrained_area_px"] == 64 * 64
    assert candidate.metadata["sam_box_constrained_area_px"] == 24 * 20
    assert candidate.metadata["detector_box_envelope_applied"] is True


def test_profile_refinement_does_not_reopen_unresolved_consensus() -> None:
    domain = DomainPrompt(
        name="device",
        root_prompts=("phone", "tablet"),
        parts=(
            PartPrompt("device_body", ("body",)),
            PartPrompt("device_screen", ("screen",)),
        ),
        part_profiles=(
            PartProfile("phone", ("phone",), ("device_screen",)),
            PartProfile("tablet", ("tablet",), ("device_screen",)),
        ),
    )
    root = MaskCandidate(
        "device",
        "device",
        np.ones((32, 32), dtype=bool),
        0.8,
        "test/root",
        prompt="tablet",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
            "profile_resolution_status": "unresolved",
        },
    )

    result = FoundationCandidateGenerator.refine_profile_parts(
        object(),
        Image.new("RGB", (32, 32)),
        [root],
        {"device": domain},
    )

    assert result.candidates == ()
    assert result.diagnostics["roots"][0]["skipped_reason"] == (
        "profile_consensus_unresolved"
    )


def test_isolated_profile_root_queries_use_classifier_budget() -> None:
    class Ranker:
        def rank_region_labels(
            self,
            image: Image.Image,
            mask: np.ndarray,
            labels: list[tuple[str, str]],
        ) -> dict[str, dict[str, object]]:
            order = {"gamma": 1, "beta": 2, "alpha": 3}
            return {
                name: {
                    "rank": order[name],
                    "probability": 0.6 / order[name],
                    "combined_similarity": 0.4 - 0.02 * order[name],
                }
                for name, _ in labels
            }

    class ProfileRootGenerator(FoundationCandidateGenerator):
        def __init__(self) -> None:
            self.config = FoundationConfig(maximum_profile_queries_per_domain=2)
            self.device = "cpu"
            self.dense_proposer = Ranker()
            self.queried: list[tuple[str, ...]] = []

        def _root_origin(self) -> str:
            return "test"

        def _grounded_source(self, stage: str) -> str:
            return f"test/{stage}"

        def _ground(self, image: Image.Image, phrases: list[str]) -> list[Detection]:
            self.queried.append(tuple(phrases))
            return []

        def _segment_boxes(
            self, image: Image.Image, detections: list[Detection]
        ) -> list[SegmentProposal]:
            return []

    domain = DomainPrompt(
        name="device",
        root_prompts=("device",),
        parts=(PartPrompt("device_body", ("body",)),),
        part_profiles=(
            PartProfile("alpha", ("alpha object",), ()),
            PartProfile("beta", ("beta object",), ()),
            PartProfile("gamma", ("gamma object",), ()),
        ),
    )
    root = MaskCandidate(
        "device",
        "device",
        np.ones((32, 32), dtype=bool),
        0.8,
        "test/root",
        prompt="device",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
        },
    )
    generator = ProfileRootGenerator()

    result = generator.generate_isolated_profile_roots(
        Image.new("RGB", (32, 32)), [root], {"device": domain}
    )

    assert generator.queried == [("beta object",), ("gamma object",)]
    assert result.diagnostics["profile_query_count"] == 2
    selection = result.diagnostics["profiles"][-1]
    assert selection["queried_profiles"] == ["beta", "gamma"]
    assert selection["skipped_profiles"] == ["alpha"]


def test_routed_profile_refinement_derives_surround_topology() -> None:
    class TopologyGenerator(FoundationCandidateGenerator):
        def __init__(self) -> None:
            self.config = FoundationConfig(crop_padding=0.0)
            self.device = "cpu"
            self.dense_proposer = None

        def _ground(self, image: Image.Image, phrases: list[str]) -> list[Detection]:
            if "screen" in phrases:
                return [Detection("screen", 0.92, (20, 20, 60, 60))]
            if "bezel" in phrases:
                raise AssertionError("topology parts must not use direct grounding")
            return []

        def _segment_boxes(
            self, image: Image.Image, detections: list[Detection]
        ) -> list[SegmentProposal]:
            masks = []
            for detection in detections:
                mask = np.zeros((image.height, image.width), dtype=bool)
                x0, y0, x1, y1 = detection.box_xyxy
                mask[y0:y1, x0:x1] = True
                masks.append(SegmentProposal(mask, 0.9))
            return masks

    domain = DomainPrompt(
        name="device",
        root_prompts=("phone",),
        parts=(
            PartPrompt("device_body", ("device body",)),
            PartPrompt(
                "device_screen",
                ("screen",),
                semantic_parent="device",
            ),
            PartPrompt(
                "device_bezel",
                ("bezel",),
                semantic_parent="device",
                topology_anchor="device_screen",
                topology_relation="surround",
                topology_scale=0.20,
            ),
        ),
        default_part_semantics=("device_body",),
        part_profiles=(
            PartProfile("phone", ("phone",), ("device_screen", "device_bezel")),
        ),
    )
    root_mask = np.zeros((80, 80), dtype=bool)
    root_mask[8:72, 8:72] = True
    root = MaskCandidate(
        "device",
        "device",
        root_mask,
        0.9,
        "test/root",
        prompt="phone",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
        },
    )

    result = TopologyGenerator().refine_profile_parts(
        Image.new("RGB", (80, 80)), [root], {"device": domain}
    )

    by_name = {candidate.semantic_name: candidate for candidate in result.candidates}
    assert set(by_name) == {"device_screen", "device_bezel"}
    assert not np.any(by_name["device_screen"].mask & by_name["device_bezel"].mask)
    assert by_name["device_bezel"].metadata["topology_relation"] == "surround"
    assert result.diagnostics["roots"][0]["topology_candidate_count"] == 1


def test_spatial_relation_rejects_wrong_sibling_region() -> None:
    class SpatialGenerator(FoundationCandidateGenerator):
        def __init__(self) -> None:
            self.config = FoundationConfig(crop_padding=0.0)

        def _ground(self, image: Image.Image, phrases: list[str]) -> list[Detection]:
            return [
                Detection("head", 0.90, (20, 5, 40, 20)),
                Detection("shirt", 0.88, (18, 5, 42, 20)),
                Detection("shirt", 0.82, (16, 32, 44, 52)),
            ]

        def _segment_boxes(
            self, image: Image.Image, detections: list[Detection]
        ) -> list[SegmentProposal]:
            output = []
            for detection in detections:
                mask = np.zeros((image.height, image.width), dtype=bool)
                x0, y0, x1, y1 = detection.box_xyxy
                mask[y0:y1, x0:x1] = True
                output.append(SegmentProposal(mask, 0.9))
            return output

    domain = DomainPrompt(
        name="asset",
        root_prompts=("object",),
        parts=(
            PartPrompt(
                semantic_name="asset_head",
                prompts=("head",),
                maximum_instances=1,
            ),
            PartPrompt(
                semantic_name="asset_shirt",
                prompts=("shirt",),
                maximum_instances=2,
                spatial_anchor="asset_head",
                spatial_relation="below",
            ),
        ),
    )
    root_mask = np.ones((64, 64), dtype=bool)
    candidates, diagnostics = SpatialGenerator()._child_candidates(
        Image.new("RGB", (64, 64)),
        domain,
        root_index=1,
        root_box=(0, 0, 64, 64),
        root_mask=root_mask,
    )
    shirts = [item for item in candidates if item.semantic_name == "asset_shirt"]
    assert len(shirts) == 1
    ys, _ = np.nonzero(shirts[0].mask)
    assert int(ys.min()) == 32
    assert diagnostics["hierarchy_calls"][0]["spatial_rejected_count"] == 1


def test_adaptive_fallback_queries_parent_roi_when_primary_coverage_is_low() -> None:
    class FallbackGenerator(FoundationCandidateGenerator):
        def __init__(self) -> None:
            self.config = FoundationConfig(crop_padding=0.0)
            self.ground_calls: list[tuple[str, ...]] = []

        def _ground(self, image: Image.Image, phrases: list[str]) -> list[Detection]:
            self.ground_calls.append(tuple(phrases))
            if "torso" in phrases and "shirt" in phrases:
                return [
                    Detection("torso", 0.95, (16, 8, 48, 56)),
                    Detection("shirt", 0.90, (36, 20, 44, 36)),
                ]
            if "shirt" in phrases:
                return [Detection("shirt", 0.88, (0, 0, image.width, image.height))]
            return []

        def _segment_boxes(
            self, image: Image.Image, detections: list[Detection]
        ) -> list[SegmentProposal]:
            output = []
            for detection in detections:
                mask = np.zeros((image.height, image.width), dtype=bool)
                x0, y0, x1, y1 = detection.box_xyxy
                mask[y0:y1, x0:x1] = True
                output.append(SegmentProposal(mask, 0.9))
            return output

    domain = DomainPrompt(
        name="asset",
        root_prompts=("object",),
        parts=(
            PartPrompt(
                semantic_name="asset_torso",
                prompts=("torso",),
                maximum_parent_fraction=0.50,
                maximum_instances=1,
            ),
            PartPrompt(
                semantic_name="asset_shirt",
                semantic_parent="asset",
                fallback_query_parent="asset_torso",
                fallback_if_coverage_below=0.35,
                assembly_parent="asset_torso",
                prompts=("shirt",),
                maximum_parent_fraction=0.50,
                fallback_maximum_parent_fraction=1.10,
                maximum_instances=1,
            ),
        ),
    )
    generator = FallbackGenerator()
    candidates, diagnostics = generator._child_candidates(
        Image.new("RGB", (64, 64)),
        domain,
        root_index=1,
        root_box=(0, 0, 64, 64),
        root_mask=np.ones((64, 64), dtype=bool),
    )

    shirts = [item for item in candidates if item.semantic_name == "asset_shirt"]
    assert len(shirts) == 2
    fallback = next(item for item in shirts if item.metadata["fallback_query"])
    torso = next(item for item in candidates if item.semantic_name == "asset_torso")
    assert fallback.metadata["query_parent_semantic"] == "asset_torso"
    assert fallback.metadata["maximum_parent_fraction_applied"] == 1.10
    assert fallback.metadata["parent_candidate_key"] == torso.metadata["candidate_key"]
    assert (
        fallback.metadata["assembly_parent_candidate_key"]
        == torso.metadata["candidate_key"]
    )
    assert (
        sum("shirt" in call and "torso" not in call for call in generator.ground_calls)
        == 1
    )
    assert any(
        call["fallback_query_semantics"] == ["asset_shirt"]
        for call in diagnostics["hierarchy_calls"]
    )


def test_adaptive_fallback_skips_parent_roi_when_primary_coverage_is_sufficient() -> (
    None
):
    class CoveredGenerator(FoundationCandidateGenerator):
        def __init__(self) -> None:
            self.config = FoundationConfig(crop_padding=0.0)
            self.ground_calls: list[tuple[str, ...]] = []

        def _ground(self, image: Image.Image, phrases: list[str]) -> list[Detection]:
            self.ground_calls.append(tuple(phrases))
            if "torso" in phrases and "shirt" in phrases:
                return [
                    Detection("torso", 0.95, (16, 8, 48, 56)),
                    Detection("shirt", 0.90, (16, 8, 48, 56)),
                ]
            if "shirt" in phrases:
                raise AssertionError("fallback query should have been skipped")
            return []

        def _segment_boxes(
            self, image: Image.Image, detections: list[Detection]
        ) -> list[SegmentProposal]:
            output = []
            for detection in detections:
                mask = np.zeros((image.height, image.width), dtype=bool)
                x0, y0, x1, y1 = detection.box_xyxy
                mask[y0:y1, x0:x1] = True
                output.append(SegmentProposal(mask, 0.9))
            return output

    domain = DomainPrompt(
        name="asset",
        root_prompts=("object",),
        parts=(
            PartPrompt(
                semantic_name="asset_torso",
                prompts=("torso",),
                maximum_parent_fraction=0.50,
                maximum_instances=1,
            ),
            PartPrompt(
                semantic_name="asset_shirt",
                fallback_query_parent="asset_torso",
                fallback_if_coverage_below=0.35,
                prompts=("shirt",),
                maximum_parent_fraction=1.10,
                maximum_instances=1,
            ),
        ),
    )
    generator = CoveredGenerator()
    candidates, _ = generator._child_candidates(
        Image.new("RGB", (64, 64)),
        domain,
        root_index=1,
        root_box=(0, 0, 64, 64),
        root_mask=np.ones((64, 64), dtype=bool),
    )

    shirts = [item for item in candidates if item.semantic_name == "asset_shirt"]
    assert len(shirts) == 1
    assert not shirts[0].metadata["fallback_query"]
    assert not any(
        "shirt" in call and "torso" not in call for call in generator.ground_calls
    )
