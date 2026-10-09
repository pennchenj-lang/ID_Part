"""Audit stored proposal-score provenance without rerunning upstream models."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from portable_paths import (
    BUNDLE_ROOT,
    DATA_ROOT,
    OUTPUT_DIR,
    PROJECT_ROOT,
    SOURCE_ROOT,
    resolve_data,
)

ROOT = BUNDLE_ROOT
PROJECT = PROJECT_ROOT
RUNTIME = DATA_ROOT
SNAPSHOT = SOURCE_ROOT
SRC = SNAPSHOT / "src/hpid_split"
BENCH = RUNTIME / "experiments/paper_v031/test226_hpid"
OUT = OUTPUT_DIR
sys.dont_write_bytecode = True
sys.path.insert(0, str(SNAPSHOT / "src"))
sys.path.insert(0, str(PROJECT / "hpid_split/scripts"))
import cv2
import numpy as np
from analyze_cross_domain_fusion_ablation import _load_candidates
from hpid_split.fusion import (
    FusionConfig,
    _is_broad_scene_layer,
    _source_agreement_factors,
    _source_family,
)
from hpid_split.prompt_bank import PromptBank

cv2.setNumThreads(1)


def read(path):
    return resolve_data(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader(); writer.writerows(rows)


RULES = [
    ("grounding_score", "foundation.py", 2236, 2288,
     "s0 is Grounding DINO processor result['scores']; detection.score is used at root/guided/profile construction. Local installed processor computes max_token(sigmoid(logit)), not a calibrated mask probability."),
    ("root_candidate", "foundation.py", 3438, 3446,
     "s=detection.score; r=.80+.20*sam_quality. Root-routing selection can take maxima across detector-derived roots and retain geometry metadata from one root."),
    ("isolated_root", "foundation.py", 996, 1004,
     "s=detection.score; r=.80+.20*segmentation.quality."),
    ("profile_root_resolution", "profile_resolution.py", 513, 545,
     "Resolved root keeps geometry sam_quality metadata, uses s=max(broad_root.score,evidence.score), and r=max(broad_root.r,evidence.r,geometry.r). Consequently the retained single sam_quality field need not reconstruct final r."),
    ("guided_candidate", "foundation.py", 1402, 1411,
     "s=detection.score; r=.86*(.55+.45*segmentation.quality)."),
    ("retrieved_part", "retrieval.py", 1603, 1619,
     "s=s0*(.58+.27*clip((visual_similarity+1)/2,0,1)+.15*geometry_score); r=r0*(.72+.28*retrieval_score)."),
    ("profile_refine", "foundation.py", 1754, 1767,
     "s=detection.score; r=.88*(.55+.45*segmentation.quality)*part.priority; positive profile priority need not be <=1."),
    ("clipseg_activation", "dense_semantic.py", 274, 296,
     "CLIPSeg logits are interpolated and passed through sigmoid."),
    ("dense_region_score", "dense_semantic.py", 209, 244,
     "s0 is max Gaussian-smoothed CLIPSeg sigmoid activation in each retained region, not the direct-mask composite score in the other routine."),
    ("profile_dense", "foundation.py", 1984, 2000,
     "s=region.score; r=dense_source_reliability*(.40+.60*s)*(.55+.45*SAM_quality)*part.priority. Default dense_source_reliability=.74. Metadata family retains grounded/profile-dense although actual source is CLIPSeg-SAM2."),
    ("sam2_pipeline", "visual_regions.py", 866, 910,
     "VisualMaskProposal.score=result['scores'] from mask-generation pipeline, which returns model iou_scores. It is a model mask-quality estimate."),
    ("sam2_multiview", "visual_regions.py", 218, 240,
     "Select representative by .60*s+.24*best_view_IoU+.16*boundary_alignment; multiview s=min(1,s*(.96+.08*best_view_IoU)); isolated crop single view scales s by config.isolated_crop_score_scale."),
    ("visual_to_candidate", "visual_regions.py", 620, 644,
     "s=clip(proposal.score,0,1); r=.62 if generic else .68. Metadata field sam_quality stores proposal.score for ALL visual proposals including appearance-contour heuristics."),
    ("contour_score", "appearance_proposals.py", 274, 298,
     "s0=clip(.43+.16*alignment+.14*closure+.08*chroma+.06*texture_delta+.13*shape_support,0,.95)."),
    ("appearance_utility", "appearance_graph.py", 191, 208,
     "u=clip(.31*q+.18*boundary_alignment+.15*boundary_closure+.13*chroma+.08*texture+.08*multiview+.10*geometry+.07*named_semantic-.18*shading_penalty,0,1). q is stored proposal score, not universally raw SAM quality."),
    ("appearance_score", "appearance_graph.py", 443, 456,
     "Every accepted visual/contour candidate score becomes clip(.62*s0+.38*u,0,1)."),
    ("cross_source_max", "appearance_graph.py", 368, 386,
     "Cross-source duplicate confirmation may replace incumbent score by max(incumbent.score,other.score); incumbent metadata alone may not preserve the other precursor score."),
    ("prototype_label", "retrieval.py", 1869, 1880,
     "Prototype labelling leaves s unchanged and scales r by .72+.28*combined prototype label score."),
    ("semantic_rerank", "visual_semantics.py", 2005, 2021,
     "s=clip(.65*s0+.35*label_rank_probability,0,1); r=max(r0,.66 default). Label-rank softmax is inventory-relative, not correctness calibration."),
    ("label_rank_probability", "dense_semantic.py", 568, 592,
     "CLIPSeg image/text features: combined similarity is weighted full/masked similarity; exp((combined-max(combined))/.035) is normalized across the queried inventory. This p is only a relative label-ranking weight."),
    ("inventory_cluster", "visual_semantics.py", 2166, 2186,
     "s=clip(.68*s0+.32*label_rank_probability,0,1); r=max(r0,.61 default)."),
    ("within_root_consensus", "visual_semantics.py", 2450, 2469,
     "s=clip(.68*s0+.32*label_rank_probability,0,1); r=max(r0,.61 default)."),
    ("structural_visual", "structural_fusion.py", 676, 689,
     "Existing visual candidate s=clip(.68*s0+.24,0,1); r=max(r0,.74 default); source family remains the original visual family."),
    ("structural_partition", "structural_fusion.py", 418, 430,
     "Generated silhouette-axial candidate s=clip(.78+.16*partition_score,0,.95), r=.74 default; family hpid-structural-fusion-v2."),
    ("partition_score", "structural_fusion.py", 373, 378,
     "partition_score=.45*min(width_ratio/4,1)+.25*max(0,1-neck_ratio)+.20*min(local_contrast,1)+.10*min(narrow_fraction/.35,1)."),
    ("structural_residual", "structural_fusion.py", 1465, 1476,
     "Generated axial residual s=clip(.62+.28*coverage,0,1), r=.74 default; family hpid-structural-fusion-v1. Other functions under this family exist but are absent from these 2 archived rows."),
    ("topology", "foundation.py", 2109, 2121,
     "s=clip(.94*anchor.score,0,1), r=.82*part.priority. Actual source says hpid-topology-v2 while recorded source_family is hpid-topology-v1."),
    ("repetitive", "relational.py", 283, 294,
     "s=clip(.48+.65*local_contrast,0,.94), r=.76."),
    ("root_geometry", "root_geometry.py", 208, 216,
     "Terminal relabelling can raise r=max(r,.82) while preserving score/source; similarly handle refinement at lines384-390."),
    ("source_family", "fusion.py", 153, 171,
     "Prefer explicit metadata source_family, else remove last source suffix; normalize DINO tiny/base names. These identifiers are not evidence of statistical independence."),
    ("agreement", "fusion.py", 471, 511,
     "a=lambda+(1-lambda)*min(1,best_cross_family_overlap/.55); single-family a=1. Default lambda=1 makes a=1 for every candidate in this release."),
    ("fusion_weight", "fusion.py", 1519, 1540,
     "t=clip(s,0,1); w0=clip((.45+.55*t)*r*a,.01,.995); if broad scene-layer predicate then w=clip(.72*w0,.01,.995), else w=w0."),
    ("evidence_pooling", "fusion.py", 1542, 1557,
     "Multiply soft mask membership by w. Max within class/family, then noisy-OR across recorded families (or max when consensus disabled). No empirical probability calibration."),
]


def main():
    benchmark = read(BENCH / "benchmark_summary.json")
    bank = PromptBank.from_json(Path(benchmark["prompt_bank"]))
    domain_for_part = {part.semantic_name: d for d in bank.domains for part in d.parts}
    config = FusionConfig()
    rows, input_hashes = [], []
    for raw in benchmark["cases"]:
        if raw["return_code"] != 0:
            continue
        cid = raw["case_id"]
        path = BENCH / cid
        candidates = _load_candidates(path)
        agreements, _ = _source_agreement_factors(candidates, config)
        by_key = {str(c.metadata.get("candidate_key")): c for c in candidates}
        input_hashes.append({"case_id": cid, "candidates_json_sha256": sha(path / "candidates.json")})
        for i, (c, a) in enumerate(zip(candidates, agreements)):
            m, family = c.metadata, _source_family(c)
            expected_score, expected_r = None, None
            reconstruction, reliability_basis = "Stored score; precursor logits not archived in candidates.json", ""
            priority = None
            domain = domain_for_part.get(c.semantic_name)
            if domain is not None:
                parts = domain.parts
                profile = next((p for p in domain.part_profiles if p.name == m.get("selected_part_profile")), None)
                if profile is not None:
                    parts = profile.apply_overrides(parts)
                priority = next(p.priority for p in parts if p.semantic_name == c.semantic_name)
            if m.get("visual_region"):
                base = float(np.clip(float(m["sam_quality"]), 0, 1))
                expected_score = float(np.clip(.62*base + .38*float(m["appearance_graph_evidence"]["utility"]), 0, 1))
                expected_r = .68 if "semantic_support_candidate_key" in m else .62
                reconstruction = "visual candidate from stored precursor score and appearance utility"
                reliability_basis = "original matched/generic visual candidate; downstream transformations applied"
                if c.source.endswith("/semantic-rerank"):
                    expected_score = float(np.clip(.65*expected_score + .35*float(m["semantic_rerank_probability"]), 0, 1))
                    expected_r = max(expected_r, .66)
                    reconstruction += "; semantic rerank"
                elif c.source.endswith(("/within-root-repetition-consensus", "/semantic-inventory-cluster")):
                    expected_score = float(np.clip(.68*expected_score + .32*float(m["semantic_rerank_probability"]), 0, 1))
                    expected_r = max(expected_r, .61)
                    reconstruction += "; within-root/inventory consensus"
                elif c.source.endswith("/structural-fusion"):
                    expected_score = float(np.clip(.68*expected_score + .24, 0, 1))
                    expected_r = max(expected_r, .74)
                    reconstruction += "; structural relabel"
                elif c.source.endswith("/prototype-labelled-region"):
                    expected_r *= .72 + .28*float(m["retrieval_label_score"])
                    reconstruction += "; prototype label leaves score unchanged"
                if m.get("cross_source_confirmed"):
                    expected_score = None
                    reconstruction = "Partial chain only: cross-source max can inherit unarchived other proposal score"
            elif m.get("repetitive_physical_detail"):
                expected_score = float(np.clip(.48+.65*float(m["contrast"]), 0, .94))
                expected_r = .76
                reconstruction = "repetitive local contrast formula"
            elif family == "hpid-structural-fusion-v1" and "structural_coverage" in m:
                expected_score = float(np.clip(.62+.28*float(m["structural_coverage"]), 0, 1))
                expected_r = .74
                reconstruction = "axial residual coverage formula"
            elif family == "hpid-structural-fusion-v2":
                expected_score = float(np.clip(.78+.16*float(m["structural_partition_score"]), 0, .95))
                expected_r = .74
                reconstruction = "silhouette axial partition formula"
            elif m.get("topology_refinement"):
                anchor = by_key.get(str(m["topology_anchor_candidate_key"]))
                if anchor is not None:
                    expected_score = float(np.clip(.94*anchor.score, 0, 1))
                    reconstruction = "topology formula using same archived anchor key"
                else:
                    reconstruction = "Topology anchor not retained in exported pool; score precursor unavailable"
                expected_r = .82*priority if priority is not None else None
            elif m.get("profile_dense_supplement"):
                expected_score = float(m["dense_score"])
                expected_r = .74*(.40+.60*expected_score)*(.55+.45*float(m["sam_quality"]))*priority
                reconstruction = "stored dense-region score identity"
            elif m.get("profile_refinement"):
                expected_r = .88*(.55+.45*float(m["sam_quality"]))*priority
                reliability_basis = "profile priority + SAM quality"
            elif c.semantic_name == c.semantic_parent:
                expected_r = .80+.20*float(m["sam_quality"])
                reliability_basis = "root quality rule; later root-routing maxima can retain different precursor metadata"
            elif m.get("retrieval_prior"):
                expected_r = .86*(.55+.45*float(m["sam_quality"]))* (.72+.28*float(m["retrieval_score"]))
                reliability_basis = "guided SAM quality times retrieval reliability scale"
            if m.get("root_geometry_support") and ("handle_ring_refined" in m or m.get("structural_fusion_algorithm") == "hinged-part-graph-v1"):
                if expected_r is not None:
                    expected_r = max(expected_r, .82)
                reliability_basis += "; root geometry support floor"
            weight0 = float(np.clip((.45+.55*np.clip(c.score, 0, 1))*c.source_reliability*a, .01, .995))
            broad = _is_broad_scene_layer(c)
            weight = float(np.clip(weight0*config.scene_layer_fallback_weight, .01, .995)) if broad else weight0
            rows.append({
                "case_id": cid, "candidate_index": i+1, "candidate_key": m.get("candidate_key"),
                "semantic_name": c.semantic_name, "source": c.source, "family": family,
                "score": c.score, "source_reliability": c.source_reliability, "agreement": a,
                "clipped_score": float(np.clip(c.score, 0, 1)), "weight_before_scene_factor": weight0,
                "broad_scene_layer": broad, "effective_weight": weight, "profile_priority": priority,
                "score_reconstruction": reconstruction, "reconstructed_score": expected_score,
                "score_abs_error": abs(expected_score-c.score) if expected_score is not None else None,
                "reliability_basis": reliability_basis, "reconstructed_reliability": expected_r,
                "reliability_abs_error": abs(expected_r-c.source_reliability) if expected_r is not None else None,
                "metadata_semantic_reranked": bool(m.get("semantic_reranked")),
                "metadata_root_geometry_support": bool(m.get("root_geometry_support")),
                "metadata_cross_source_confirmed": bool(m.get("cross_source_confirmed")),
                "profile_hint_source": m.get("profile_hint_source", ""),
                "root_resolution_note": "Final r may inherit another root via max; a simple single-q discrepancy is not evidence of incorrect frozen r" if c.semantic_name == c.semantic_parent and m.get("profile_hint_source") == "isolated_profile_consensus" else "",
            })
    summaries = []
    for family in sorted({r["family"] for r in rows}):
        selected = [r for r in rows if r["family"] == family]
        summary = {"family": family, "n": len(selected), "case_count": len({r["case_id"] for r in selected}),
                   "source_counts": json.dumps(dict(Counter(r["source"] for r in selected))),
                   "score_reconstructed_n": sum(r["score_abs_error"] is not None for r in selected),
                   "score_reconstruction_mismatches": sum(r["score_abs_error"] is not None and r["score_abs_error"] > 1e-10 for r in selected),
                   "reliability_reconstructed_n": sum(r["reliability_abs_error"] is not None for r in selected),
                   "reliability_reconstruction_mismatches": sum(r["reliability_abs_error"] is not None and r["reliability_abs_error"] > 1e-10 for r in selected)}
        for key in ["score", "source_reliability", "agreement", "weight_before_scene_factor", "effective_weight"]:
            summary[key+"_min"] = min(r[key] for r in selected)
            summary[key+"_max"] = max(r[key] for r in selected)
        summaries.append(summary)
    rules = [{"rule": label, "source_file": filename, "start_line": first, "end_line": last,
              "explanation": explanation, "sha256": sha(SRC / filename),
              "code_excerpt": "\n".join((SRC / filename).read_text(encoding="utf-8").splitlines()[first-1:last])}
             for label, filename, first, last, explanation in RULES]
    source_paths = sorted({SRC / filename for _, filename, *_ in RULES} | {SRC / "cli.py", SRC / "prompt_bank.py", SRC / "asset_routing.py", SRC / "root_routing.py"})
    config_paths = [Path(benchmark["prompt_bank"])]
    config_paths += [config_paths[0].parent / v for v in read(config_paths[0]).get("include", [])]
    library_paths = [RUNTIME / ".venv/Lib/site-packages/transformers/models/grounding_dino/processing_grounding_dino.py",
                     RUNTIME / ".venv/Lib/site-packages/transformers/pipelines/mask_generation.py"]
    config_equivalence = []
    for path in config_paths:
        frozen = SNAPSHOT / "configs" / path.name
        config_equivalence.append({"name": path.name, "used_sha256": sha(path), "snapshot_sha256": sha(frozen),
                                   "json_semantically_equal": read(path) == read(frozen),
                                   "newline_normalized_text_equal": path.read_text(encoding="utf-8-sig") == frozen.read_text(encoding="utf-8-sig")})
    with (OUT / "source_reliability_candidates.csv").open(encoding="utf-8-sig", newline="") as f:
        previous = list(csv.DictReader(f))
    old_values = Counter((r["case_id"], r["family"], float(r["score"]), float(r["source_reliability"]), float(r["agreement"]), float(r["weight"])) for r in previous)
    new_values = Counter((r["case_id"], r["family"], r["score"], r["source_reliability"], r["agreement"], r["effective_weight"]) for r in rows)
    assert old_values == new_values, "Mismatch with previous fixed-candidate weight audit"
    write_csv(OUT / "score_candidate_provenance.csv", rows)
    write_csv(OUT / "score_family_ranges.csv", summaries)
    write_csv(OUT / "score_input_hashes.csv", input_hashes)
    write_csv(OUT / "score_reconstruction_exceptions.csv", [r for r in rows if (r["score_abs_error"] is not None and r["score_abs_error"] > 1e-10) or (r["reliability_abs_error"] is not None and r["reliability_abs_error"] > 1e-10)])
    dump(OUT / "score_source_rules.json", rules)
    protocol = {
        "n_cases": len(input_hashes), "n_candidates": len(rows), "n_recorded_families": len(summaries),
        "fusion_config": asdict(config), "no_model_generation_or_retraining": True,
        "source_hashes": [{"path": str(path), "sha256": sha(path)} for path in source_paths + config_paths + library_paths],
        "config_equivalence_to_snapshot": config_equivalence,
        "prior_fixed_candidate_weights_exact_multiset_match": old_values == new_values,
        "observed_global_ranges": {k: {"min": min(r[k] for r in rows), "max": max(r[k] for r in rows)} for k in ["score", "source_reliability", "agreement", "effective_weight"]},
        "prompt_bank_matches_benchmark_sha256": sha(config_paths[0]) == benchmark["prompt_bank_sha256"],
        "frozen_fusion_matches_handoff_sha256": sha(SRC / "fusion.py") == "d9bb3f738b262109d88587dae5b71796d312a461790d66fe7dfb689f1c372bb5",
        "script_sha256": sha(__file__), "reliability_gt_one_n": sum(r["source_reliability"] > 1 for r in rows),
        "score_outside_unit_interval_n": sum(not 0 <= r["score"] <= 1 for r in rows),
        "broad_scene_layer_n": sum(r["broad_scene_layer"] for r in rows),
        "all_agreements_equal_one": all(r["agreement"] == 1 for r in rows),
        "score_reconstructed_n": sum(r["score_abs_error"] is not None for r in rows),
        "score_reconstruction_mismatches": sum(r["score_abs_error"] is not None and r["score_abs_error"] > 1e-10 for r in rows),
        "reliability_reconstructed_n": sum(r["reliability_abs_error"] is not None for r in rows),
        "reliability_reconstruction_mismatches": sum(r["reliability_abs_error"] is not None and r["reliability_abs_error"] > 1e-10 for r in rows),
        "root_composite_simple_formula_exception_n": sum(r["reliability_abs_error"] is not None and r["reliability_abs_error"] > 1e-10 and bool(r["root_resolution_note"]) for r in rows),
        "boundaries": [
            "s is final stored proposal score, not universally detector score. Formula and implementation are heuristic score shaping; no reliability diagram, ECE, or learned correctness calibration was performed.",
            "Producer formulas are source-audited. Numerical checks reconstruct only fields with retained precursor metadata; raw detector/CLIPSeg/SAM tensors are not rerun. Observed final scores are authoritative.",
            "For 7 profile-resolved roots, r is a maximum across broad/evidence/geometry roots, but retained sam_quality belongs only to the selected geometry. The simple .8+.2*q check differs; archived r remains authoritative, and exact reconstruction of discarded precursor r values is not claimed.",
            "Visual metadata sam_quality stores pre-appearance proposal.score, including contour heuristic scores. Model-origin SAM quality can already include multiview/crop scaling.",
            "Family labels are grouping keys with explicit overrides. Correlation persists between profile-refine, profile-dense and other shared models; noisy-OR does not establish calibrated independent evidence.",
            "Frozen inputs/configuration and source hashes are preserved. This audit does not independently establish the historical date at which every upstream dependency/checkpoint was frozen.",
            "No reference masks enter score checks. Reconstruction exceptions remain visible and must be explained, never silently excluded.",
        ],
    }
    dump(OUT / "score_protocol.json", protocol)
    print(json.dumps({k:v for k,v in protocol.items() if k not in ["source_hashes", "fusion_config", "boundaries"]}, indent=2))


if __name__ == "__main__":
    main()
