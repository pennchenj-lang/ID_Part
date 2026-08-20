import numpy as np
from PIL import Image

from hpid_split.fusion import MaskCandidate
from hpid_split.physical_region_audit import (
    PhysicalRegionAuditConfig,
    PhysicalRegionAuditor,
    make_physical_region_contact_sheet,
    parse_physical_region_audit_response,
)
from hpid_split.prompt_bank import DomainPrompt, PartProfile, PartPrompt


def _mask(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    mask = np.zeros((80, 80), dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _root() -> MaskCandidate:
    return MaskCandidate(
        "device",
        "device",
        _mask(5, 75, 5, 75),
        0.95,
        "test/root",
        prompt="phone",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
            "selected_part_profile": "phone",
        },
    )


def _candidate(key: str, mask: np.ndarray) -> MaskCandidate:
    return MaskCandidate(
        "device_visual_detail_01",
        "device",
        mask,
        0.9,
        "sam2/test",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": key,
            "generic_visual_region": True,
            "visual_region_kind": "detail",
            "root_area_fraction": float(mask.sum() / _root().mask.sum()),
        },
    )


def _domain() -> DomainPrompt:
    return DomainPrompt(
        "device",
        ("phone",),
        (PartPrompt("device_screen", ("screen",)),),
        part_profiles=(PartProfile("phone", ("phone",), ("device_screen",)),),
    )


class _Planner:
    backend_id = "test-vlm"

    def __init__(self) -> None:
        self.calls = 0

    def generate_response(self, image: Image.Image, prompt: str) -> str:
        self.calls += 1
        assert "TARGET 1" not in prompt
        assert image.width > image.height
        return (
            '{"objects":['
            '{"id":1,"label":"physical_component","certainty":"medium"},'
            '{"id":2,"label":"surface_detail","certainty":"high"}'
            "]}"
        )


def test_physicality_parser_rejects_out_of_range_ids() -> None:
    parsed, diagnostics = parse_physical_region_audit_response(
        '{"objects":['
        '{"id":1,"label":"physical_component","certainty":"high"},'
        '{"id":9,"label":"surface_detail","certainty":"high"}'
        "]}",
        expected_count=2,
    )

    assert set(parsed) == {1}
    assert diagnostics["invalid_count"] == 1
    assert diagnostics["missing_count"] == 1


def test_physicality_auditor_annotates_regions_without_assigning_ids() -> None:
    root = _root()
    first = _candidate("region:1", _mask(20, 30, 20, 30))
    second = _candidate("region:2", _mask(40, 50, 40, 50))
    planner = _Planner()
    auditor = PhysicalRegionAuditor(
        planner,
        config=PhysicalRegionAuditConfig(maximum_queries=1, batch_size=2),
    )

    result = auditor.audit(
        Image.new("RGB", (80, 80), "white"),
        [root],
        [root, first, second],
        {"device": _domain()},
    )

    assert planner.calls == 1
    assert result.candidates[1].semantic_name == first.semantic_name
    assert (
        result.candidates[1].metadata["vlm_physicality_audit"]["decision"]
        == "physical_supported"
    )
    assert (
        result.candidates[2].metadata["vlm_physicality_audit"]["decision"]
        == "nonphysical_supported"
    )
    assert result.diagnostics["audited_candidate_count"] == 2


def test_physicality_auditor_skips_open_domains() -> None:
    root = _root()
    character_root = MaskCandidate(
        "character",
        "character",
        root.mask,
        root.score,
        root.source,
        metadata={**root.metadata, "selected_part_profile": "character"},
    )
    candidate = _candidate("region:1", _mask(20, 30, 20, 30))
    candidate = MaskCandidate(
        candidate.semantic_name,
        "character",
        candidate.mask,
        candidate.score,
        candidate.source,
        metadata=candidate.metadata,
    )
    planner = _Planner()

    result = PhysicalRegionAuditor(planner).audit(
        Image.new("RGB", (80, 80), "white"),
        [character_root],
        [character_root, candidate],
        {},
    )

    assert planner.calls == 0
    assert result.diagnostics["eligible_candidate_count"] == 0


def test_physicality_auditor_includes_named_visual_region() -> None:
    root = _root()
    candidate = _candidate("region:1", _mask(15, 65, 15, 65))
    candidate = MaskCandidate(
        "device_screen",
        "device",
        candidate.mask,
        candidate.score,
        candidate.source,
        metadata={
            **candidate.metadata,
            "generic_visual_region": False,
            "visual_region": True,
        },
    )
    planner = _Planner()

    result = PhysicalRegionAuditor(
        planner,
        config=PhysicalRegionAuditConfig(maximum_queries=1, batch_size=1),
    ).audit(
        Image.new("RGB", (80, 80), "white"),
        [root],
        [root, candidate],
        {"device": _domain()},
    )

    assert planner.calls == 1
    assert result.diagnostics["eligible_candidate_count"] == 1
    assert result.candidates[1].metadata["vlm_physicality_audit"]["decision"] == (
        "physical_supported"
    )


def test_physicality_contact_sheet_contains_numbered_tiles() -> None:
    root = _root()
    candidate = _candidate("region:1", _mask(20, 30, 20, 30))

    sheet = make_physical_region_contact_sheet(
        Image.new("RGB", (80, 80), "white"),
        [(root, candidate)],
    )

    assert sheet.size == (384, 220)
