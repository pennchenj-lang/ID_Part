"""Read-only execution trace of frozen fine-ID fusion, plus shared-input audit.

Run with the frozen algorithm Python. No source file is edited and no label is
fed to inference. Stage masks are transient; all reference tracks and aggregate
diagnostics are exported. --limit uses a separate prefix and cannot replace the
complete run. Read stage_protocol.json before interpreting intermediate rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import pairwise
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
BENCH = RUNTIME / "experiments/paper_v031/test226_hpid"
FRONT = RUNTIME / "experiments/paper_v031_identity_frontend_20260828/01_same_candidate"
FUSION = SNAPSHOT / "src/hpid_split/fusion.py"
BASELINE = PROJECT / "hpid_split/src/hpid_split/postprocess_baselines.py"
sys.dont_write_bytecode = True
sys.path.insert(0, str(SNAPSHOT / "src"))
sys.path.insert(0, str(PROJECT / "hpid_split/scripts"))

import cv2
import numpy as np
from analyze_cross_domain_fusion_ablation import _load_candidates
from hpid_split import fusion
from hpid_split.paco_eval import _normalize
from hpid_split.paper_eval import _hungarian, _semantic_hungarian
from PIL import Image

# The older algorithm snapshot predates the frozen comparison baseline. Load
# that baseline's unchanged workspace file explicitly and record its hash.
spec = importlib.util.spec_from_file_location("hpid_split.postprocess_baselines", BASELINE)
baseline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = baseline
spec.loader.exec_module(baseline)
from evaluate_same_candidate_postprocessing import _evaluate, _hpid_prediction

cv2.setNumThreads(1)
SEED = 20261008
STAGES = [
    "input_pool", "after_same_source_dedup", "after_instance_cap",
    "after_hierarchy_duplicate_filter", "diagnostic_consensus_argmax",
    "diagnostic_parent_support_argmax", "diagnostic_parent_residual_argmax",
    "diagnostic_specificity_argmax", "actual_semantic_argmax",
    "after_direct_detail_gate", "after_cleanup", "after_root_restoration",
    "identity_supported_before_remainder", "identity_plus_eligible_remainder",
    "after_remainder_attachment_before_filter", "final_hpid", "final_dbscan",
]


def read(path):
    return resolve_data(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mask_sha(mask):
    return hashlib.sha256(np.asarray(mask, dtype=np.uint8).tobytes()).hexdigest()


def csv_read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def csv_write(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def source_line(prefix):
    hits = [i for i, line in enumerate(FUSION.read_text(encoding="utf-8").splitlines(), 1)
            if line.startswith(prefix)]
    if len(hits) != 1:
        raise RuntimeError((prefix, hits))
    return hits[0]


POINTS = {
    source_line("    accepted, capped_groups_dropped"): "after_same_source_dedup",
    source_line("    accepted, hierarchical_duplicates_suppressed"): "after_instance_cap",
    source_line("    taxonomy = taxonomy or"): "after_hierarchy_duplicate_filter",
    source_line("    if config.use_parent_support:"): "diagnostic_consensus_argmax",
    source_line("    if config.use_parent_residual:"): "diagnostic_parent_support_argmax",
    source_line("    specificity_suppression_count = 0"): "diagnostic_parent_residual_argmax",
    source_line("    if config.detail_bonus:"): "diagnostic_specificity_argmax",
    source_line("    if config.use_direct_gate:"): "actual_semantic_argmax",
    source_line("    if config.use_boundary_ownership:"): "after_direct_detail_gate",
    source_line("    lost_root_pixels_before_conservation ="): "after_cleanup",
    source_line("    if taxonomy.num_fine_classes <="): "after_root_restoration",
    source_line("        remainder = class_mask & ~claimed"): "identity_supported_before_remainder",
    source_line("        detached_components: list[np.ndarray] = []"): "identity_plus_eligible_remainder",
    source_line("        for assignment in assignments:"): "after_remainder_attachment_before_filter",
}


def candidate_rows(candidates):
    return [(c.semantic_name, c.semantic_parent, c.mask.copy(),
             str(c.metadata.get("candidate_key", f"candidate_{i}")))
            for i, c in enumerate(candidates)]


def label_rows(labels, taxonomy):
    return [(taxonomy.fine_names[i], taxonomy.parent_names[taxonomy.fine_to_parent[i]],
             labels == i, "semantic:" + taxonomy.fine_names[i])
            for i in range(1, taxonomy.num_fine_classes) if np.any(labels == i)]


class ReadOnlyTrace:
    def __init__(self, candidates):
        self.stages = {"input_pool": candidate_rows(candidates)}
        self.per_class = defaultdict(dict)
        self.code = {fusion.fuse_candidates.__code__, fusion._hierarchical_part_ids.__code__}
        self.after_remainder_seen = set()

    def global_trace(self, frame, event, arg):
        return self.local_trace if event == "call" and frame.f_code in self.code else None

    def local_trace(self, frame, event, arg):
        if event != "line" or frame.f_lineno not in POINTS:
            return self.local_trace
        stage = POINTS[frame.f_lineno]
        local = frame.f_locals
        if stage.startswith("after_same") or stage in {"after_instance_cap", "after_hierarchy_duplicate_filter"}:
            self.stages[stage] = candidate_rows(local["accepted"])
        elif stage.startswith("diagnostic_"):
            # Preserve real evidence values; evaluate only a COPY with the same
            # final background convention. These argmaxes are readouts, not
            # actual emitted maps and not separate algorithms or causal effects.
            evidence = local["evidence"].copy()
            support = local["parent_support"]
            foreground = support[1:].max(axis=0) if len(support) > 1 else 0.0
            evidence[0] = np.clip(1.0 - foreground, 0.04, 0.98)
            evidence[0, local["root_union"]] = np.minimum(evidence[0, local["root_union"]], .01)
            self.stages[stage] = label_rows(evidence.argmax(axis=0), local["taxonomy"])
        elif stage in {"actual_semantic_argmax", "after_direct_detail_gate", "after_cleanup", "after_root_restoration"}:
            self.stages[stage] = label_rows(local["labels"], local["taxonomy"])
        else:
            name, parent = local["semantic_name"], local["semantic_parent"]
            if stage == "after_remainder_attachment_before_filter":
                if name in self.after_remainder_seen:
                    return self.local_trace
                self.after_remainder_seen.add(name)
            rows = [(name, parent, a.visible_mask.copy(),
                     str(a.representative.metadata.get("candidate_key", f"assignment_{i}")))
                    for i, a in enumerate(local["assignments"])]
            if stage == "identity_plus_eligible_remainder":
                components = local["component_masks"]
            elif stage == "after_remainder_attachment_before_filter":
                components = local["detached_components"]
            else:
                components = []
            rows.extend((name, parent, mask.copy(), f"remainder:{i}") for i, mask in enumerate(components))
            self.per_class[stage][name] = rows
        return self.local_trace

    def finish(self):
        for stage, classes in self.per_class.items():
            self.stages[stage] = [r for rows in classes.values() for r in rows]
        return self.stages


def normalized(name, domain, category):
    return "body" if name == domain else _normalize(name.removeprefix(domain + "_"), domain, object_category=category)


def summarize_stage(raw_rows, truth_masks, truth_names, domain, category):
    rows = [row for row in raw_rows if row[0] != domain or "body" in truth_names]
    names = [normalized(row[0], domain, category) for row in rows]
    masks = [row[2] for row in rows]
    matches = {i: (j, x) for i, j, x in _hungarian(truth_masks, masks)}
    sem_matches = {i: (j, x) for i, j, x in _semantic_hungarian(truth_masks, truth_names, masks, names)}
    counts = np.zeros(truth_masks[0].shape, dtype=np.uint16)
    for m in masks:
        counts += m
    tracks = []
    for ref_index, (truth, ref_name) in enumerate(zip(truth_masks, truth_names)):
        areas = int(truth.sum())
        intersections = [int((truth & m).sum()) for m in masks]
        ious = [inter / max(1, areas + int(m.sum()) - inter) for inter, m in zip(intersections, masks)]
        j = int(np.argmax(ious)) if ious else None
        same = [k for k, name in enumerate(names) if name == ref_name]
        sj = max(same, key=lambda k: ious[k]) if same else None
        sem_union = np.logical_or.reduce([masks[k] for k in same]) if same else np.zeros(truth.shape, bool)
        tracks.append({
            "reference_index": ref_index, "reference_semantic": ref_name, "reference_pixels": areas,
            "prediction_count": len(rows), "best_iou": ious[j] if j is not None else 0.0,
            "best_identity": rows[j][3] if j is not None else "",
            "best_semantic": names[j] if j is not None else "",
            "best_semantic_iou": ious[sj] if sj is not None else 0.0,
            "best_semantic_identity": rows[sj][3] if sj is not None else "",
            "semantic_union_recall": int((sem_union & truth).sum()) / max(1, areas),
            "semantic_union_precision": int((sem_union & truth).sum()) / max(1, int(sem_union.sum())),
            "any_union_recall": int(((counts > 0) & truth).sum()) / max(1, areas),
            "hungarian_iou": matches.get(ref_index, (None, 0.0))[1],
            "semantic_hungarian_iou": sem_matches.get(ref_index, (None, 0.0))[1],
        })
    k = sum(x["hungarian_iou"] >= .25 for x in tracks)
    sk = sum(x["semantic_hungarian_iou"] >= .25 for x in tracks)
    metrics = {
        "prediction_count": len(rows), "reference_count": len(tracks), "matches_at_025": k,
        "semantic_matches_at_025": sk,
        "diagnostic_f1_at_025": 2 * k / max(1, len(rows) + len(tracks)),
        "diagnostic_semantic_f1_at_025": 2 * sk / max(1, len(rows) + len(tracks)),
        "overlap_excess_pixels": int(np.maximum(counts.astype(int) - 1, 0).sum()),
        **{metric: float(np.mean([r[metric] for r in tracks])) for metric in
           ["best_iou", "best_semantic_iou", "semantic_union_recall", "any_union_recall"]},
    }
    return tracks, metrics


def worker(raw, manifest, archived):
    cid = raw["case_id"]
    package = BENCH / cid
    candidates = _load_candidates(package)
    source_before = [mask_sha(c.mask) for c in candidates]
    audit = ReadOnlyTrace(candidates)
    try:
        sys.settrace(audit.global_trace)
        result = fusion.fuse_candidates(candidates, config=fusion.FusionConfig())
    finally:
        sys.settrace(None)
    stages = audit.finish()
    if not np.array_equal(result.instance_map, np.asarray(Image.open(package / "part_id_map.tiff"))):
        raise RuntimeError(f"Frozen map mismatch: {cid}")
    assert source_before == [mask_sha(c.mask) for c in candidates], f"Input mutation: {cid}"
    stages["final_hpid"] = [(r.semantic_name, r.semantic_parent,
                             result.instance_map == r.instance_index, r.part_id) for r in result.instances]
    dbscan = baseline.dbscan_proposal_fusion(candidates)
    stages["final_dbscan"] = [(r.semantic_name, r.semantic_parent, r.mask, r.identity) for r in dbscan.instances]
    assert source_before == [mask_sha(c.mask) for c in candidates], f"DBSCAN input mutation: {cid}"
    if set(stages) != set(STAGES):
        raise RuntimeError((cid, set(STAGES) - set(stages)))
    case_path = Path(manifest["case_path"])
    case = read(case_path)
    domain, category = raw["expected_domain"], case["object_category"]
    truth_masks = [np.asarray(Image.open(case_path.parent / row["mask_crop"]).convert("L")) >= 128 for row in case["parts"]]
    truth_names = [_normalize(r["part_name"], domain, object_category=category) for r in case["parts"]]
    mismatch_max = 0.0
    # Verify all archived numeric evaluation endpoints, except timing and
    # ancillary flags, using the same evaluator and same shared object inputs.
    for name, pred in [("dbscan_fusion", dbscan), ("hpid_split_a3", _hpid_prediction(package, dbscan.root_mask))]:
        metrics = _evaluate(pred, case=case, case_dir=case_path.parent, expected_domain=domain)
        for key, value in metrics.items():
            if key in archived[name]:
                diff = abs(value - float(archived[name][key]))
                mismatch_max = max(mismatch_max, diff)
                if diff > 1e-10:
                    raise RuntimeError((cid, name, key, value, archived[name][key]))
        if int(archived[name]["candidate_count"]) != len(candidates):
            raise RuntimeError((cid, name, "candidate_count"))
    tracks, metrics = [], []
    for stage in STAGES:
        rows, metric = summarize_stage(stages[stage], truth_masks, truth_names, domain, category)
        common = {"case_id": cid, "domain": domain, "category": category, "stage": stage}
        tracks.extend({**common, **r} for r in rows)
        metrics.append({**common, **metric})
    shared = {"case_id": cid, "candidate_count": len(candidates), "reference_count": len(truth_masks),
              "exact_hpid_map_reproduction": True, "archived_numeric_metric_max_abs_difference": mismatch_max,
              "input_masks_unchanged": True, "candidates_json_sha256": sha(package / "candidates.json"),
              "ordered_mask_digests_sha256": hashlib.sha256("".join(source_before).encode()).hexdigest(),
              "final_map_sha256": mask_sha(result.instance_map),
              "accepted_count_after_refilter": len(result.accepted_candidates),
              "remainder_components_attached": result.diagnostics.get("remainder_components_attached", 0),
              "identity_visibility_slivers_dropped": result.diagnostics.get("identity_visibility_slivers_dropped", 0)}
    return tracks, metrics, shared


def interval(values, categories):
    values = np.asarray(values, float)
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(np.asarray(categories) == category) for category in sorted(set(categories))]
    draws = [values[np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])].mean()
             for _ in range(10000)]
    return {"mean": float(values.mean()), "low": float(np.quantile(draws, .025)), "high": float(np.quantile(draws, .975))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    out = OUTPUT_DIR
    out.mkdir(exist_ok=True)
    prefix = "stage_trial" if args.limit else "stage"
    bench = read(BENCH / "benchmark_summary.json")
    manifest = {r["case_id"]: r for r in read(bench["source_manifest"])["cases"]}
    rows = csv_read(FRONT / "same_candidate_postprocessing_cases.csv")
    archived = defaultdict(dict)
    for row in rows:
        archived[row["case_id"]][row["method"]] = row
    cases = [r for r in bench["cases"] if r["return_code"] == 0]
    if args.limit:
        cases = cases[:args.limit]
    protocol = {
        "n": len(cases), "seed": SEED, "stage_order": STAGES,
        "fusion_source": str(FUSION), "fusion_sha256": sha(FUSION),
        "baseline_source": str(BASELINE), "baseline_sha256": sha(BASELINE),
        "evaluator_sha256": sha(PROJECT / "hpid_split/scripts/evaluate_same_candidate_postprocessing.py"),
        "source_manifest_sha256": sha(bench["source_manifest"]),
        "script_sha256": sha(__file__), "trace_line_numbers": POINTS,
        "inference_modification": "None. sys.settrace reads frame locals, makes copies, never writes algorithm state; exact released map asserted.",
        "shared_inputs": "The identical loaded exported candidate objects are passed to frozen HPID and workspace DBSCAN. Masks are hashed before and after both; archived candidate counts and all numeric evaluator endpoints are asserted.",
        "candidate_boundary": "Exported candidates.json is the archived accepted fine-fusion pool, before later Group verification. It is not the complete pre-proposal-rejection universe.",
        "diagnostic_argmax": "Before actual argmax, frozen intermediate evidence is copied and given the same final background convention; these provisional readouts isolate scoring transitions but are not emitted masks, counterfactual endpoint results, or independent algorithms.",
        "identity_and_remainder": "Per-class actual supported assignment masks, plus eligible connected remainder components, then actual attachment before small-area/visibility-sliver final filtering. These are actual run states, aggregated after tracing all classes.",
        "reference_tracks": "Every reference part tracked through every stage. Best IoU is descriptive oracle support and may choose a different identity across stages; Hungarian matching also reported. Same-semantic union recall has reference pixels as denominator. Reference parts/pixels are not independent samples.",
        "uncertainty": "Case-macro means; categories sampled with replacement retaining every case in selected category, 10000 draws, seed 20261008; pointwise exploratory 95% intervals, no superiority tests or multiplicity-adjusted claim.",
        "scope": "226 existing test cases, post-review diagnostic reuse. No rerun of upstream proposal generation. No post-Group spatial attribution. Stage deltas are sequential state changes, not additive causal explanations for DBSCAN-versus-HPID gap.",
    }
    dump(out / f"{prefix}_protocol.json", protocol)
    tracks, metrics, shared = [], [], []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(worker, r, manifest[r["case_id"]], archived[r["case_id"]]): r["case_id"] for r in cases}
        for n, job in enumerate(as_completed(jobs), 1):
            a, b, c = job.result()
            tracks.extend(a); metrics.extend(b); shared.append(c)
            print(f"stage {n}/{len(cases)} {jobs[job]}", flush=True)
    tracks.sort(key=lambda r: (r["case_id"], r["reference_index"], STAGES.index(r["stage"])))
    metrics.sort(key=lambda r: (r["case_id"], STAGES.index(r["stage"])))
    shared.sort(key=lambda r: r["case_id"])
    csv_write(out / f"{prefix}_reference_tracks.csv", tracks)
    csv_write(out / f"{prefix}_case_metrics.csv", metrics)
    csv_write(out / f"{prefix}_shared_input_audit.csv", shared)
    summaries, transitions = [], []
    metric_names = ["diagnostic_f1_at_025", "diagnostic_semantic_f1_at_025", "best_iou", "best_semantic_iou", "semantic_union_recall", "any_union_recall"]
    indexed = {(r["case_id"], r["stage"]): r for r in metrics}
    ids = sorted(r["case_id"] for r in shared)
    categories = [indexed[cid, "input_pool"]["category"] for cid in ids]
    for stage in STAGES:
        summary = {"stage": stage, "n": len(ids), "total_matches_at_025": sum(indexed[cid, stage]["matches_at_025"] for cid in ids),
                   "total_prediction_count": sum(indexed[cid, stage]["prediction_count"] for cid in ids)}
        for metric in metric_names:
            for key, value in interval([indexed[cid, stage][metric] for cid in ids], categories).items():
                summary[f"{metric}_{key}"] = value
        summaries.append(summary)
    track_index = {(r["case_id"], r["reference_index"], r["stage"]): r for r in tracks}
    pairs = list(pairwise(STAGES[:-1])) + [("final_hpid", "final_dbscan")]
    for before, after in pairs:
        transition = {"before": before, "after": after, "case_count": len(ids)}
        for metric in metric_names:
            deltas = [indexed[cid, after][metric] - indexed[cid, before][metric] for cid in ids]
            for key, value in interval(deltas, categories).items():
                transition[f"delta_{metric}_{key}"] = value
        for metric in ["best_iou", "best_semantic_iou", "semantic_union_recall", "any_union_recall"]:
            deltas = [track_index[r["case_id"], r["reference_index"], after][metric] - r[metric]
                      for r in tracks if r["stage"] == before]
            transition[metric + "_references_decreased"] = sum(x < -1e-10 for x in deltas)
            transition[metric + "_references_increased"] = sum(x > 1e-10 for x in deltas)
        transitions.append(transition)
    csv_write(out / f"{prefix}_summary.csv", summaries)
    csv_write(out / f"{prefix}_transitions.csv", transitions)
    dump(out / f"{prefix}_audit_summary.json", {
        "cases": len(shared), "total_candidates": sum(r["candidate_count"] for r in shared),
        "total_references": sum(r["reference_count"] for r in shared), "category_clusters": len(set(categories)),
        "exact_hpid_maps": sum(r["exact_hpid_map_reproduction"] for r in shared),
        "maximum_archived_evaluator_discrepancy": max(r["archived_numeric_metric_max_abs_difference"] for r in shared),
        "candidate_masks_unchanged_all_cases": all(r["input_masks_unchanged"] for r in shared),
        "candidates_removed_on_refilter": sum(r["candidate_count"] - r["accepted_count_after_refilter"] for r in shared),
        "remainder_components_attached": sum(r["remainder_components_attached"] for r in shared),
        "visibility_slivers_dropped": sum(r["identity_visibility_slivers_dropped"] for r in shared),
        "all_17_stages_present_each_case": len(metrics) == len(shared) * len(STAGES),
    })
    print(json.dumps({"cases": len(shared), "tracks": len(tracks), "prefix": prefix}), flush=True)


if __name__ == "__main__":
    main()
