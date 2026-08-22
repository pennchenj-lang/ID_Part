from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from hpid_split.appearance_graph import AppearanceGraphConfig
from hpid_split.appearance_proposals import AppearanceProposalConfig
from hpid_split.asset_routing import AssetRouterConfig, ProfileTextRouterConfig
from hpid_split.ensemble_gate import EnsembleEvidenceGateConfig
from hpid_split.foundation import FoundationConfig
from hpid_split.fusion import FusionConfig
from hpid_split.mask_refinement import MaskRefinementConfig
from hpid_split.ontology_routing import OntologyRoutingConfig
from hpid_split.physical_region_audit import PhysicalRegionAuditConfig
from hpid_split.profile_resolution import ProfileResolutionConfig
from hpid_split.proposal_first import ProposalFirstConfig
from hpid_split.relational import RelationalAppearanceConfig
from hpid_split.retrieval import RetrievalConfig
from hpid_split.root_cleanup import RootCleanupConfig
from hpid_split.root_geometry import RootGeometryConfig
from hpid_split.root_routing import RootRoutingConfig
from hpid_split.scene_instances import SceneInstanceConfig
from hpid_split.semantic_candidate_audit import SemanticCandidateAuditConfig
from hpid_split.shape_proposals import ShapeProposalConfig
from hpid_split.structural_fusion import StructuralFusionConfig
from hpid_split.visual_regions import VisualRegionConfig
from hpid_split.visual_semantics import (
    AxisConsistencyConfig,
    PhysicalRegionGateConfig,
    VisualSemanticConfig,
)

CONFIGS = (
    FoundationConfig(),
    ProposalFirstConfig(),
    AssetRouterConfig(),
    ProfileTextRouterConfig(),
    RetrievalConfig(),
    AppearanceProposalConfig(),
    ShapeProposalConfig(),
    VisualRegionConfig(),
    VisualSemanticConfig(),
    AxisConsistencyConfig(),
    PhysicalRegionGateConfig(),
    AppearanceGraphConfig(),
    EnsembleEvidenceGateConfig(),
    StructuralFusionConfig(),
    RelationalAppearanceConfig(),
    RootCleanupConfig(),
    RootGeometryConfig(),
    RootRoutingConfig(),
    SceneInstanceConfig(),
    MaskRefinementConfig(),
    PhysicalRegionAuditConfig(),
    SemanticCandidateAuditConfig(),
    ProfileResolutionConfig(),
    OntologyRoutingConfig(),
    FusionConfig(),
)

EXECUTION_SETTINGS = {
    "routing_condition": "automatic",
    "root_mode": "primary",
    "decomposition_mode": "automatic",
    "visual_crop_layers": 1,
    "visual_points_per_crop": 18,
    "dense_semantic_fallback": True,
    "profile_refinement": True,
    "florence_parts": False,
    "additional_grounding_models": [],
    "device": "cuda",
    "precision_mode": "model defaults; no explicit autocast override",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _value(value: object) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _git_commit(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export the frozen HPID-Split parameter and inventory registry."
    )
    parser.add_argument("--prompt-bank", type=Path, required=True)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    global_rows: list[dict[str, object]] = []
    for config in CONFIGS:
        class_name = type(config).__name__
        module_name = type(config).__module__
        for parameter, value in asdict(config).items():
            global_rows.append(
                {
                    "module": module_name,
                    "config_class": class_name,
                    "parameter": parameter,
                    "value": _value(value),
                    "category_specific": False,
                    "value_type": type(value).__name__,
                    "origin": "release_code_default",
                    "frozen_before_test": True,
                }
            )
    global_rows.extend(
        {
            "module": "scripts.run_paco_benchmark",
            "config_class": "ExecutionSetting",
            "parameter": parameter,
            "value": _value(value),
            "category_specific": False,
            "value_type": type(value).__name__,
            "origin": "evaluation_command",
            "frozen_before_test": True,
        }
        for parameter, value in EXECUTION_SETTINGS.items()
    )

    prompt_bank = json.loads(args.prompt_bank.read_text(encoding="utf-8-sig"))
    inventory_rows: list[dict[str, object]] = []
    for domain in prompt_bank.get("domains", []):
        domain_name = str(domain["name"])
        for part in domain.get("parts", []):
            semantic_name = str(part["semantic_name"])
            for parameter, value in part.items():
                if parameter == "semantic_name":
                    continue
                inventory_rows.append(
                    {
                        "domain": domain_name,
                        "profile": "",
                        "semantic_name": semantic_name,
                        "rule_scope": "part_inventory",
                        "parameter": parameter,
                        "value": _value(value),
                        "category_specific": True,
                        "origin": "manually_specified_release_inventory",
                        "frozen_before_test": True,
                    }
                )
        for profile in domain.get("part_profiles", []):
            profile_name = str(profile["name"])
            for parameter, value in profile.items():
                if parameter == "name":
                    continue
                inventory_rows.append(
                    {
                        "domain": domain_name,
                        "profile": profile_name,
                        "semantic_name": "",
                        "rule_scope": "object_profile",
                        "parameter": parameter,
                        "value": _value(value),
                        "category_specific": True,
                        "origin": "manually_specified_release_inventory",
                        "frozen_before_test": True,
                    }
                )

    args.output.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output / "global_parameter_registry.csv", global_rows)
    _write_csv(args.output / "inventory_rule_registry.csv", inventory_rows)
    report = {
        "format": "HPID-Split frozen parameter registry",
        "format_version": "1.0.0",
        "global_parameter_count": len(global_rows),
        "inventory_rule_count": len(inventory_rows),
        "prompt_bank": str(args.prompt_bank.resolve()),
        "prompt_bank_sha256": _sha256(args.prompt_bank),
        "development_manifest_sha256": _sha256(args.development_manifest),
        "test_manifest_sha256": _sha256(args.test_manifest),
        "git_commit_at_export": _git_commit(Path.cwd()),
        "benchmark_isolation": (
            "Release defaults and the manually specified inventory were frozen "
            "before the independent test manifest was evaluated. Inventory "
            "development could use public PACO train/development taxonomy and "
            "the old development cases; this is disclosed rather than treated "
            "as zero-shot taxonomy discovery. Independent-test part labels and "
            "masks are never read by inference and are opened only for scoring."
        ),
        "parameter_origin_note": (
            "Values are hand-engineered release constants or manual inventory "
            "rules, not estimates fitted on the independent test labels."
        ),
    }
    (args.output / "parameter_registry_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
