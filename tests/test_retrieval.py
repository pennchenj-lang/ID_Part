from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from hpid_split.fusion import MaskCandidate
from hpid_split.retrieval import (
    REFERENCE_FORMAT,
    PrototypeIndex,
    PrototypeRetriever,
    RetrievalConfig,
    apply_retrieval_domain_corrections,
    build_retrieval_index,
)
from hpid_split.taxonomy import Taxonomy


class _MeanColorEncoder:
    model_name = "test/mean-color"

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        rows = []
        for image in images:
            rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
            rows.append(
                [
                    *rgb.mean(axis=(0, 1)).tolist(),
                    float(rgb.std()),
                    1.0,
                ]
            )
        return np.asarray(rows, dtype=np.float32)


def _write_reference(
    root: Path,
    asset_id: str,
    colors: tuple[tuple[int, int, int], tuple[int, int, int]],
    class_ids: tuple[int, int],
) -> tuple[Path, Path]:
    image = np.full((48, 48, 3), 245, dtype=np.uint8)
    labels = np.zeros((48, 48), dtype=np.uint8)
    image[8:40, 5:32] = colors[0]
    labels[8:40, 5:32] = class_ids[0]
    image[18:42, 32:43] = colors[1]
    labels[18:42, 32:43] = class_ids[1]
    image_path = root / f"{asset_id}.png"
    label_path = root / f"{asset_id}_labels.png"
    Image.fromarray(image, mode="RGB").save(image_path)
    Image.fromarray(labels, mode="L").save(label_path)
    return image_path, label_path


def _build_test_index(tmp_path: Path) -> tuple[PrototypeIndex, Path]:
    taxonomy = Taxonomy(
        fine_names=(
            "background",
            "stock",
            "magazine",
            "seat",
            "leg",
        ),
        parent_names=("background", "tool_prop", "furniture"),
        fine_to_parent=(0, 1, 1, 2, 2),
        detail_names=("magazine", "leg"),
    )
    taxonomy_path = tmp_path / "taxonomy.json"
    taxonomy.to_json(taxonomy_path)
    entries = []
    for index, colors in enumerate(
        (
            ((205, 45, 35), (35, 165, 60)),
            ((190, 55, 40), (45, 155, 65)),
        ),
        start=1,
    ):
        image, labels = _write_reference(tmp_path, f"rifle_{index}", colors, (1, 2))
        entries.append(
            {
                "asset_id": f"rifle_{index}",
                "asset_label": "rifle",
                "asset_domain": "tool_prop",
                "asset_profile": "weapon",
                "reviewed": True,
                "image": image.name,
                "label_map": labels.name,
                "taxonomy": taxonomy_path.name,
                "part_aliases": {"stock": ["buttstock"]},
            }
        )
    for index, colors in enumerate(
        (
            ((35, 65, 205), (225, 190, 35)),
            ((45, 75, 190), (215, 180, 45)),
        ),
        start=1,
    ):
        image, labels = _write_reference(tmp_path, f"chair_{index}", colors, (3, 4))
        entries.append(
            {
                "asset_id": f"chair_{index}",
                "asset_label": "chair",
                "asset_domain": "furniture",
                "asset_profile": "seat",
                "reviewed": True,
                "image": image.name,
                "label_map": labels.name,
                "taxonomy": taxonomy_path.name,
            }
        )
    reference_path = tmp_path / "references.json"
    reference_path.write_text(
        json.dumps({"format": REFERENCE_FORMAT, "entries": entries}),
        encoding="utf-8",
    )
    output = tmp_path / "index"
    build_retrieval_index(
        reference_path,
        output,
        _MeanColorEncoder(),
        metric_epochs=8,
        metric_device="cpu",
    )
    return PrototypeIndex.load(output), reference_path


def _root(
    image_path: Path, semantic: str = "device"
) -> tuple[Image.Image, MaskCandidate]:
    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image)
    mask = np.any(rgb < 235, axis=2)
    return image, MaskCandidate(
        semantic_name=semantic,
        semantic_parent=semantic,
        mask=mask,
        score=0.8,
        source="test/root",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1",
            "parent_candidate_key": None,
        },
    )


def test_index_requires_explicitly_reviewed_references(tmp_path: Path) -> None:
    manifest = tmp_path / "references.json"
    manifest.write_text(
        json.dumps(
            {
                "format": REFERENCE_FORMAT,
                "entries": [
                    {
                        "asset_id": "prediction",
                        "asset_label": "unknown",
                        "asset_domain": "device",
                        "reviewed": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not reviewed"):
        build_retrieval_index(manifest, tmp_path / "index", _MeanColorEncoder())


def test_index_builds_supervised_prototypes_from_materialized_paco_case(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "paco_case"
    case_dir.mkdir()
    image = np.full((40, 48, 3), 235, dtype=np.uint8)
    image[5:35, 5:43] = (45, 85, 155)
    root = np.zeros((40, 48), dtype=np.uint8)
    root[5:35, 5:43] = 255
    screen = np.zeros_like(root)
    screen[8:29, 8:39] = 255
    button = np.zeros_like(root)
    button[30:34, 35:40] = 255
    Image.fromarray(image, mode="RGB").save(case_dir / "source_crop.png")
    Image.fromarray(root, mode="L").save(case_dir / "object_mask_crop.png")
    (case_dir / "parts_crop").mkdir()
    Image.fromarray(screen, mode="L").save(case_dir / "parts_crop/screen.png")
    Image.fromarray(button, mode="L").save(case_dir / "parts_crop/button.png")
    (case_dir / "case.json").write_text(
        json.dumps(
            {
                "format": "HPID PACO benchmark case",
                "object_category": "cellular_telephone",
                "parts": [
                    {
                        "annotation_id": 11,
                        "part_name": "screen",
                        "mask_crop": "parts_crop/screen.png",
                    },
                    {
                        "annotation_id": 12,
                        "part_name": "left_button",
                        "mask_crop": "parts_crop/button.png",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "references.json"
    manifest.write_text(
        json.dumps(
            {
                "format": REFERENCE_FORMAT,
                "entries": [
                    {
                        "asset_id": "paco-phone-1",
                        "asset_label": "cellular telephone",
                        "asset_domain": "device",
                        "reviewed": True,
                        "paco_case": str(case_dir / "case.json"),
                        "part_name_mapping": {
                            "screen": "device_screen",
                            "left_button": "device_button",
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    built = build_retrieval_index(
        manifest,
        tmp_path / "index",
        _MeanColorEncoder(),
        metric_epochs=0,
    )

    assert built["training"]["ground_truth_used_for_index_building"] is True
    assert built["training"]["ground_truth_used_during_query_inference"] is False
    assert built["assets"][0]["source_kind"] == ("public_human_annotated_paco_case")
    assert {row["semantic_name"] for row in built["parts"]} == {
        "device_screen",
        "device_button",
    }


def test_canonical_prompt_bank_parent_overrides_source_taxonomy_parent(
    tmp_path: Path,
) -> None:
    taxonomy = Taxonomy(
        fine_names=("background", "skin"),
        parent_names=("background", "skin"),
        fine_to_parent=(0, 1),
        detail_names=(),
    )
    taxonomy_path = tmp_path / "taxonomy.json"
    taxonomy.to_json(taxonomy_path)
    image_path, labels_path = _write_reference(
        tmp_path, "character", ((190, 80, 70), (190, 80, 70)), (1, 1)
    )
    prompt_bank = tmp_path / "prompts.json"
    prompt_bank.write_text(
        json.dumps(
            {
                "domains": [
                    {
                        "name": "character",
                        "root_prompts": ["person"],
                        "parts": [
                            {
                                "semantic_name": "character_head",
                                "prompts": ["head"],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "references.json"
    manifest.write_text(
        json.dumps(
            {
                "format": REFERENCE_FORMAT,
                "prompt_bank": prompt_bank.name,
                "part_name_mapping": {"skin": "character_head"},
                "entries": [
                    {
                        "asset_id": "character",
                        "asset_label": "character",
                        "asset_domain": "character",
                        "reviewed": True,
                        "image": image_path.name,
                        "label_map": labels_path.name,
                        "taxonomy": taxonomy_path.name,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    built = build_retrieval_index(
        manifest,
        tmp_path / "index",
        _MeanColorEncoder(),
        metric_epochs=0,
    )

    assert built["parts"][0]["semantic_name"] == "character_head"
    assert built["parts"][0]["semantic_parent"] == "character"


def test_source_label_can_be_explicitly_excluded_from_learning(
    tmp_path: Path,
) -> None:
    taxonomy = Taxonomy(
        fine_names=("background", "skin", "hair"),
        parent_names=("background", "character"),
        fine_to_parent=(0, 1, 1),
        detail_names=(),
    )
    taxonomy_path = tmp_path / "taxonomy.json"
    taxonomy.to_json(taxonomy_path)
    image_path, labels_path = _write_reference(
        tmp_path, "character", ((190, 80, 70), (45, 35, 30)), (1, 2)
    )
    manifest = tmp_path / "references.json"
    manifest.write_text(
        json.dumps(
            {
                "format": REFERENCE_FORMAT,
                "exclude_parts": ["skin"],
                "entries": [
                    {
                        "asset_id": "character",
                        "asset_label": "character",
                        "asset_domain": "character",
                        "reviewed": True,
                        "image": image_path.name,
                        "label_map": labels_path.name,
                        "taxonomy": taxonomy_path.name,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    built = build_retrieval_index(
        manifest,
        tmp_path / "index",
        _MeanColorEncoder(),
        metric_epochs=0,
    )

    assert {row["source_semantic_name"] for row in built["parts"]} == {"hair"}


def test_retrieval_learns_object_inventory_and_aliases(tmp_path: Path) -> None:
    index, _ = _build_test_index(tmp_path)
    image, root = _root(tmp_path / "rifle_1.png")
    retriever = PrototypeRetriever(
        index,
        _MeanColorEncoder(),
        config=RetrievalConfig(
            minimum_asset_similarity=0.50,
            domain_relabel_similarity=0.50,
        ),
    )

    result = retriever.query(image, [root])

    plan = result.plans[0]
    assert plan.accepted
    assert plan.asset_label == "rifle"
    assert plan.asset_domain == "tool_prop"
    assert plan.supporting_asset_count == 2
    assert {prior.output_semantic_name for prior in plan.part_priors} == {
        "tool_prop_stock",
        "tool_prop_magazine",
    }
    stock = next(prior for prior in plan.part_priors if prior.semantic_name == "stock")
    assert "buttstock" in stock.phrases
    assert stock.guided_spec().maximum_instances == stock.maximum_instances
    assert index.assets[0]["asset_profile"] == "weapon"


def test_retrieval_stays_within_resolved_root_domain_without_strong_override(
    tmp_path: Path,
) -> None:
    index, _ = _build_test_index(tmp_path)
    image, root = _root(tmp_path / "rifle_1.png", semantic="furniture")
    retriever = PrototypeRetriever(
        index,
        _MeanColorEncoder(),
        config=RetrievalConfig(
            minimum_asset_similarity=-1.0,
            domain_relabel_similarity=1.01,
        ),
    )

    plan = retriever.query(image, [root]).plans[0]

    assert plan.accepted
    assert plan.asset_label == "chair"
    assert plan.asset_domain == "furniture"
    assert all(row["asset_domain"] == "furniture" for row in plan.nearest_assets)


def test_explicit_asset_hint_constrains_retrieval_category(tmp_path: Path) -> None:
    index, _ = _build_test_index(tmp_path)
    image, root = _root(tmp_path / "rifle_1.png", semantic="furniture")
    retriever = PrototypeRetriever(
        index,
        _MeanColorEncoder(),
        config=RetrievalConfig(minimum_prompted_asset_similarity=-1.0),
    )

    plan = retriever.query(image, [root], asset_hint="red rifle").plans[0]

    assert plan.accepted
    assert plan.asset_label == "rifle"
    assert plan.asset_domain == "tool_prop"
    assert all(row["asset_label"] == "rifle" for row in plan.nearest_assets)


def test_unknown_asset_hint_does_not_force_unrelated_prototype(
    tmp_path: Path,
) -> None:
    index, _ = _build_test_index(tmp_path)
    image, root = _root(tmp_path / "rifle_1.png", semantic="tool_prop")
    retriever = PrototypeRetriever(index, _MeanColorEncoder())

    plan = retriever.query(image, [root], asset_hint="espresso machine").plans[0]

    assert not plan.accepted
    assert plan.reason == "asset_hint_not_indexed"


def test_ambiguous_automatic_route_builds_a_conservative_candidate_inventory(
    tmp_path: Path,
) -> None:
    index, _ = _build_test_index(tmp_path)
    image, root = _root(tmp_path / "rifle_1.png", semantic="device")
    retriever = PrototypeRetriever(
        index,
        _MeanColorEncoder(),
        config=RetrievalConfig(minimum_prompted_asset_similarity=-1.0),
    )

    plan = retriever.query(
        image,
        [root],
        asset_candidates_by_root={"test::1": ("rifle", "chair")},
    ).plans[0]

    assert plan.accepted
    assert plan.reason == "accepted_ambiguous_candidate_inventory"
    assert plan.asset_label is None
    assert plan.supporting_asset_count == 4
    assert {prior.output_semantic_name for prior in plan.part_priors} == {
        "tool_prop_stock",
        "tool_prop_magazine",
        "tool_prop_seat",
        "tool_prop_leg",
    }


def test_root_domain_filters_an_ambiguous_route_before_inventory_fusion(
    tmp_path: Path,
) -> None:
    index, _ = _build_test_index(tmp_path)
    image, root = _root(tmp_path / "rifle_1.png", semantic="tool_prop")
    retriever = PrototypeRetriever(
        index,
        _MeanColorEncoder(),
        config=RetrievalConfig(minimum_prompted_asset_similarity=-1.0),
    )

    plan = retriever.query(
        image,
        [root],
        asset_candidates_by_root={"test::1": ("rifle", "chair")},
    ).plans[0]

    assert plan.accepted
    assert plan.asset_label == "rifle"
    assert {prior.output_semantic_name for prior in plan.part_priors} == {
        "tool_prop_stock",
        "tool_prop_magazine",
    }


def test_resolved_profile_rejects_unrelated_candidate_inventory(
    tmp_path: Path,
) -> None:
    index, _ = _build_test_index(tmp_path)
    image, root = _root(tmp_path / "rifle_1.png", semantic="tool_prop")
    root = MaskCandidate(
        root.semantic_name,
        root.semantic_parent,
        root.mask,
        root.score,
        root.source,
        root.prompt,
        root.source_reliability,
        {
            **root.metadata,
            "selected_part_profile": "unseen_game_firearm",
            "profile_resolution_status": "accepted",
        },
    )
    retriever = PrototypeRetriever(
        index,
        _MeanColorEncoder(),
        config=RetrievalConfig(minimum_prompted_asset_similarity=-1.0),
    )

    plan = retriever.query(
        image,
        [root],
        asset_candidates_by_root={"test::1": ("chair", "rifle")},
    ).plans[0]

    assert not plan.accepted
    assert plan.reason == "resolved_profile_not_indexed"
    assert not plan.part_priors


def test_retrieval_can_correct_a_supported_wrong_root_domain(tmp_path: Path) -> None:
    index, _ = _build_test_index(tmp_path)
    image, root = _root(tmp_path / "rifle_1.png")
    child = MaskCandidate(
        semantic_name="device_button",
        semantic_parent="device",
        mask=root.mask.copy(),
        score=0.4,
        source="test/child",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "parent_candidate_key": "root:1",
        },
    )
    config = RetrievalConfig(
        minimum_asset_similarity=0.50,
        domain_relabel_similarity=0.50,
        domain_relabel_minimum_support=2,
        allow_domain_relabel=True,
    )
    result = PrototypeRetriever(index, _MeanColorEncoder(), config=config).query(
        image, [root]
    )

    corrected, diagnostics = apply_retrieval_domain_corrections(
        [root, child], result, config=config
    )

    assert diagnostics["correction_count"] == 1
    assert len(corrected) == 1
    assert corrected[0].semantic_name == "tool_prop"
    assert corrected[0].metadata["retrieval_previous_domain"] == "device"


def test_retrieved_candidate_inherits_reviewed_semantic_parent(tmp_path: Path) -> None:
    index, _ = _build_test_index(tmp_path)
    image, root = _root(tmp_path / "rifle_1.png", semantic="tool_prop")
    config = RetrievalConfig(minimum_asset_similarity=0.50)
    retriever = PrototypeRetriever(index, _MeanColorEncoder(), config=config)
    plan = retriever.query(image, [root]).plans[0]
    stock = next(prior for prior in plan.part_priors if prior.semantic_name == "stock")
    stock_mask = np.zeros(root.mask.shape, dtype=bool)
    stock_mask[8:40, 5:32] = True
    candidate = MaskCandidate(
        semantic_name=f"tool_prop_guided_{stock.slug}",
        semantic_parent="tool_prop",
        mask=stock_mask,
        score=0.8,
        source="test/guided-part",
        metadata={"guided_prompt_slug": stock.slug},
    )

    reranked, diagnostics = retriever.rerank_candidates(
        image, root, [candidate], plan.part_priors
    )

    assert diagnostics["accepted_count"] == 1
    assert reranked[0].semantic_name == "tool_prop_stock"
    assert reranked[0].semantic_parent == "tool_prop"
    assert reranked[0].metadata["retrieval_prior"] is True


def test_retrieved_candidate_rejects_implausible_reviewed_geometry(
    tmp_path: Path,
) -> None:
    index, _ = _build_test_index(tmp_path)
    image, root = _root(tmp_path / "rifle_1.png", semantic="tool_prop")
    config = RetrievalConfig(
        minimum_asset_similarity=0.50,
        minimum_geometry_compatibility=0.20,
    )
    retriever = PrototypeRetriever(index, _MeanColorEncoder(), config=config)
    plan = retriever.query(image, [root]).plans[0]
    stock = next(prior for prior in plan.part_priors if prior.semantic_name == "stock")
    whole_root = MaskCandidate(
        semantic_name=f"tool_prop_guided_{stock.slug}",
        semantic_parent="tool_prop",
        mask=root.mask.copy(),
        score=0.95,
        source="test/guided-part",
        metadata={"guided_prompt_slug": stock.slug},
    )

    reranked, diagnostics = retriever.rerank_candidates(
        image, root, [whole_root], plan.part_priors
    )

    assert not reranked
    assert diagnostics["rejected_count"] == 1
    assert diagnostics["rows"][0]["geometry_compatibility"] < 0.20


def test_retrieval_is_a_fallback_when_semantic_already_exists(
    tmp_path: Path,
) -> None:
    index, _ = _build_test_index(tmp_path)
    image, root = _root(tmp_path / "rifle_1.png", semantic="tool_prop")
    retriever = PrototypeRetriever(
        index,
        _MeanColorEncoder(),
        config=RetrievalConfig(
            minimum_asset_similarity=0.50,
            minimum_raw_part_similarity=0.0,
            minimum_geometry_compatibility=0.0,
        ),
    )
    plan = retriever.query(image, [root]).plans[0]
    stock = next(prior for prior in plan.part_priors if prior.semantic_name == "stock")
    stock_mask = np.zeros(root.mask.shape, dtype=bool)
    stock_mask[8:40, 5:32] = True
    existing = MaskCandidate(
        semantic_name="tool_prop_stock",
        semantic_parent="tool_prop",
        mask=stock_mask,
        score=0.8,
        source="test/existing-part",
        metadata={"root_origin": "test", "root_index": 1},
    )
    retrieved = MaskCandidate(
        semantic_name=f"tool_prop_guided_{stock.slug}",
        semantic_parent="tool_prop",
        mask=stock_mask.copy(),
        score=0.9,
        source="test/guided-part",
        metadata={"guided_prompt_slug": stock.slug},
    )

    reranked, diagnostics = retriever.rerank_candidates(
        image,
        root,
        [retrieved],
        plan.part_priors,
        existing_candidates=[existing],
    )

    assert not reranked
    assert diagnostics["rows"][0]["rejection_reason"] == ("existing_semantic_duplicate")


def test_reviewed_prototype_can_label_compatible_generic_visual_region(
    tmp_path: Path,
) -> None:
    index, _ = _build_test_index(tmp_path)
    image, root = _root(tmp_path / "rifle_1.png", semantic="tool_prop")
    config = RetrievalConfig(
        minimum_asset_similarity=0.50,
        minimum_raw_part_similarity=0.50,
        minimum_geometry_compatibility=0.12,
        prototype_label_margin=0.0,
    )
    retriever = PrototypeRetriever(index, _MeanColorEncoder(), config=config)
    result = retriever.query(image, [root])
    stock_mask = np.zeros(root.mask.shape, dtype=bool)
    stock_mask[8:40, 5:32] = True
    visual = MaskCandidate(
        semantic_name="tool_prop_visual_panel_01",
        semantic_parent="tool_prop",
        mask=stock_mask,
        score=0.88,
        source="test/point-grid",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1/visual-region:01",
            "generic_visual_region": True,
        },
    )

    labelled, diagnostics = retriever.label_visual_candidates(
        image, [root], [visual], result.plans
    )

    assert diagnostics["labelled_count"] == 1
    assert labelled[0].semantic_name == "tool_prop_stock"
    assert labelled[0].metadata["retrieval_region_label"] is True
    assert labelled[0].metadata["generic_visual_region"] is False


def test_visual_region_stays_generic_when_prototype_geometry_is_wrong(
    tmp_path: Path,
) -> None:
    index, _ = _build_test_index(tmp_path)
    image, root = _root(tmp_path / "rifle_1.png", semantic="tool_prop")
    config = RetrievalConfig(
        minimum_asset_similarity=0.50,
        minimum_geometry_compatibility=0.80,
        prototype_label_margin=0.0,
    )
    retriever = PrototypeRetriever(index, _MeanColorEncoder(), config=config)
    result = retriever.query(image, [root])
    tiny = np.zeros(root.mask.shape, dtype=bool)
    tiny[2:5, 42:45] = True
    visual = MaskCandidate(
        semantic_name="tool_prop_visual_detail_01",
        semantic_parent="tool_prop",
        mask=tiny,
        score=0.9,
        source="test/point-grid",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1/visual-region:01",
            "generic_visual_region": True,
        },
    )

    labelled, diagnostics = retriever.label_visual_candidates(
        image, [root], [visual], result.plans
    )

    assert diagnostics["labelled_count"] == 0
    assert labelled[0].semantic_name == "tool_prop_visual_detail_01"
    assert labelled[0].metadata["generic_visual_region"] is True


def test_visual_prototype_does_not_replace_existing_semantic_part(
    tmp_path: Path,
) -> None:
    index, _ = _build_test_index(tmp_path)
    image, root = _root(tmp_path / "rifle_1.png", semantic="tool_prop")
    retriever = PrototypeRetriever(
        index,
        _MeanColorEncoder(),
        config=RetrievalConfig(
            minimum_asset_similarity=0.50,
            minimum_raw_part_similarity=0.0,
            minimum_geometry_compatibility=0.0,
            prototype_label_margin=0.0,
        ),
    )
    result = retriever.query(image, [root])
    stock_mask = np.zeros(root.mask.shape, dtype=bool)
    stock_mask[8:40, 5:32] = True
    existing = MaskCandidate(
        semantic_name="tool_prop_stock",
        semantic_parent="tool_prop",
        mask=stock_mask,
        score=0.8,
        source="test/existing-part",
        metadata={"root_origin": "test", "root_index": 1},
    )
    visual = MaskCandidate(
        semantic_name="tool_prop_visual_panel_01",
        semantic_parent="tool_prop",
        mask=stock_mask.copy(),
        score=0.9,
        source="test/point-grid",
        metadata={
            "root_origin": "test",
            "root_index": 1,
            "candidate_key": "root:1/visual-region:01",
            "generic_visual_region": True,
        },
    )

    labelled, diagnostics = retriever.label_visual_candidates(
        image,
        [root],
        [visual],
        result.plans,
        existing_candidates=[existing],
    )

    assert diagnostics["labelled_count"] == 0
    assert labelled[0].semantic_name == "tool_prop_visual_panel_01"
    assert diagnostics["rows"][0]["rejection_reason"] == ("existing_semantic_duplicate")


def test_object_level_assignment_does_not_duplicate_one_semantic_part(
    tmp_path: Path,
) -> None:
    index, _ = _build_test_index(tmp_path)
    image, root = _root(tmp_path / "rifle_1.png", semantic="tool_prop")
    retriever = PrototypeRetriever(
        index,
        _MeanColorEncoder(),
        config=RetrievalConfig(
            minimum_asset_similarity=0.50,
            minimum_raw_part_similarity=0.50,
            minimum_geometry_compatibility=0.10,
            prototype_label_margin=0.0,
        ),
    )
    result = retriever.query(image, [root])
    first_mask = np.zeros(root.mask.shape, dtype=bool)
    first_mask[8:40, 5:32] = True
    second_mask = first_mask.copy()
    second_mask[8:12, 5:10] = False
    visual_candidates = [
        MaskCandidate(
            semantic_name=f"tool_prop_visual_panel_{index:02d}",
            semantic_parent="tool_prop",
            mask=mask,
            score=0.90 - index * 0.01,
            source="test/point-grid",
            metadata={
                "root_origin": "test",
                "root_index": 1,
                "candidate_key": f"root:1/visual-region:{index:02d}",
                "generic_visual_region": True,
            },
        )
        for index, mask in enumerate((first_mask, second_mask), start=1)
    ]

    labelled, diagnostics = retriever.label_visual_candidates(
        image, [root], visual_candidates, result.plans
    )

    semantic_names = [candidate.semantic_name for candidate in labelled]
    assert semantic_names.count("tool_prop_stock") == 1
    assert diagnostics["semantic_instance_counts"]["tool_prop_stock"] == 1
    assert any(
        row["rejection_reason"] in {"instance_limit", "duplicate_semantic_region"}
        for row in diagnostics["rows"]
        if not row["accepted"]
    )


def test_open_set_rejection_does_not_force_unseen_object(tmp_path: Path) -> None:
    index, _ = _build_test_index(tmp_path)
    image = Image.new("RGB", (48, 48), (120, 120, 120))
    mask = np.ones((48, 48), dtype=bool)
    root = MaskCandidate("device", "device", mask, 0.7, "test/root")
    retriever = PrototypeRetriever(
        index,
        _MeanColorEncoder(),
        config=RetrievalConfig(minimum_asset_similarity=0.9999),
    )

    plan = retriever.query(image, [root]).plans[0]

    assert not plan.accepted
    assert plan.reason == "open_set_similarity_rejection"
    assert not plan.part_priors
