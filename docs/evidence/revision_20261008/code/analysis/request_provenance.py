"""Replay the same frozen 37 requests and record observed candidate/pixel paths.

No proposal-generation rerun, retuning, causal deletion intervention, or new cases.
Wrappers call unchanged implementations; exact stored endpoint checks are mandatory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
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
HOLDOUT = RUNTIME / "experiments/paper_v031_untouched_group_holdout_42_20260825_r5"
FRONT = RUNTIME / "experiments/paper_v031_identity_frontend_20260828"
FOLDER = FRONT / "03_completion_group_frontend"
MANIFEST = FRONT / "02_completion_frontend/completion_target_manifest.json"
sys.dont_write_bytecode = True
sys.path.insert(0, str(SNAPSHOT / "src"))
import cv2
import hpid_split
import numpy as np
from PIL import Image, ImageDraw

# Baselines were added after the holdout algorithm snapshot. Load that one missing
# module from the archived workspace, while fusion/groups remain the frozen code.
hpid_split.__path__.append(str(PROJECT / "hpid_split/src/hpid_split"))
import hpid_split.physical_groups as pg
import hpid_split.postprocess_baselines as baseline
from hpid_split import fusion
from hpid_split.export import load_previous_package
from hpid_split.paco_eval import _normalize

cv2.setNumThreads(1)


def read_json(path):
    return resolve_data(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_mask(path):
    return np.asarray(Image.open(path).convert("L")) >= 128


def area(mask):
    return int(np.count_nonzero(mask))


def iou(a, b):
    return area(a & b) / max(1, area(a | b))


def spatial(mask, truth):
    tp = area(mask & truth)
    return {"area_px": area(mask), "target_intersection_px": tp,
            "target_iou": iou(mask, truth), "target_precision": tp / max(1, area(mask)),
            "target_recall": tp / max(1, area(truth))}


def normal(name, target):
    domain = target["expected_domain"]
    return "body" if name == domain else _normalize(name.removeprefix(domain + "_"), domain,
                                                   object_category=target["object_category"])


def token(target):
    return _normalize(target["target_part_name"], target["expected_domain"],
                      object_category=target["object_category"])


def instances_from_map(label_map, records, group=False):
    return tuple(baseline.BaselineInstance(
        identity=r.group_id if group else r.part_id, semantic_name=r.semantic_name,
        semantic_parent=r.semantic_name if group else r.semantic_parent,
        mask=label_map == (r.group_index if group else r.instance_index), confidence=1.0,
    ) for r in records)


def state(instances, target, truth):
    matches = [r for r in instances if normal(r.semantic_name, target) == token(target)]
    selected = max(matches, key=lambda r: (float(r.confidence), area(r.mask), r.identity)) if matches else None
    selected_iou = iou(selected.mask, truth) if selected else 0.0
    best_iou = max((iou(r.mask, truth) for r in matches), default=0.0)
    label = ("unresolved" if not matches else "ambiguous" if len(matches) > 1 else
             "unique_correct" if selected_iou >= .25 else "unique_wrong")
    return {"state": label, "query_match_count": len(matches), "selected_identity": selected.identity if selected else "",
            "selected_semantic": selected.semantic_name if selected else "", "selected_part_iou": selected_iou,
            "best_review_match_iou": best_iou, "has_correct_option_025": best_iou >= .25,
            "unique_and_correct_025": int(len(matches) == 1 and selected_iou >= .25),
            "unique_and_correct_050": int(len(matches) == 1 and selected_iou >= .5),
            "wrong_unique_025": int(len(matches) == 1 and selected_iou < .25),
            "wrong_unique_050": int(len(matches) == 1 and selected_iou < .5),
            "query_ambiguous": int(len(matches) > 1), "query_unresolved": int(not matches),
            "correct_in_review_set_025": int(len(matches) > 1 and best_iou >= .25),
            "correct_in_review_set_050": int(len(matches) > 1 and best_iou >= .5)}, matches, selected


def wilson(k, n):
    if not n:
        return {"k": k, "n": n, "rate": None, "low": None, "high": None}
    z = 1.959963984540054
    p = k / n
    center = (p + z*z/(2*n)) / (1+z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return {"k": k, "n": n, "rate": p, "low": max(0.0, center-half), "high": min(1.0, center+half)}


class Observation:
    def __init__(self, candidates, target, truth):
        self.candidates = candidates
        self.lookup = {id(c): i for i, c in enumerate(candidates, 1)}
        self.target = target
        self.truth = truth
        self.retention = {}
        self.stages = []
        self.group_stages = []
        self.db_rows = []
        self.clusters = []
        self.originals = []

    def patch(self, module, name, wrapper):
        old = getattr(module, name)
        self.originals.append((module, name, old))
        setattr(module, name, wrapper(old))

    def restore(self):
        for module, name, old in reversed(self.originals):
            setattr(module, name, old)

    def record_stage(self, name, instances):
        row, _, _ = state(instances, self.target, self.truth)
        self.stages.append({"stage_order": len(self.stages), "stage": name, **row})

    def patch_fusion(self):
        for name in ("_deduplicate", "_filter_candidates_by_instance_caps", "_suppress_hierarchical_duplicates"):
            def factory(old, stage=name):
                def wrapped(*args, **kwargs):
                    result = old(*args, **kwargs)
                    self.retention[stage] = {self.lookup[id(c)] for c in result[0]}
                    return result
                return wrapped
            self.patch(fusion, name, factory)

        def labels_factory(old):
            def wrapped(labels, taxonomy, *args, **kwargs):
                def record(name, labels):
                    instances = tuple(baseline.BaselineInstance(str(i), semantic, semantic,
                                      labels == i, 1.0) for i, semantic in enumerate(taxonomy.fine_names) if i)
                    self.record_stage(name, instances)
                record("fine_semantic_before_label_cleanup", labels)
                result = old(labels, taxonomy, *args, **kwargs)
                record("fine_semantic_after_label_cleanup", result)
                return result
            return wrapped
        self.patch(fusion, "_clean_labels", labels_factory)

    def patch_groups(self):
        names = ("_regularize_character_surface_boundaries", "_regularize_stem_base_boundaries",
                 "_clean_group_satellites", "_split_repeated_group_components")
        for name in names:
            def factory(old, stage=name):
                def wrapped(group_map, groups, *args, **kwargs):
                    before = group_map.copy()
                    records_before = tuple(groups)
                    self.record_stage("group_before" + stage, instances_from_map(before, records_before, True))
                    self.group_stages.append(("before" + stage, before, records_before))
                    result = old(group_map, groups, *args, **kwargs)
                    after = result[0].copy()
                    records_after = tuple(result[1])
                    self.record_stage("group_after" + stage, instances_from_map(after, records_after, True))
                    self.group_stages.append(("after" + stage, after, records_after))
                    return result
                return wrapped
            self.patch(pg, name, factory)

    def patch_dbscan(self):
        def component_factory(old):
            def wrapped(*args, **kwargs):
                result = old(*args, **kwargs)
                self.clusters = result
                return result
            return wrapped
        self.patch(baseline, "_dbscan_components", component_factory)

        def exclusive_factory(old):
            def wrapped(rows, **kwargs):
                self.db_rows = [(a, b, mask.copy(), conf) for a, b, mask, conf in rows]
                return old(rows, **kwargs)
            return wrapped
        self.patch(baseline, "_exclusive_rows", exclusive_factory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.limit and args.output == BUNDLE_ROOT / "evidence":
        raise ValueError("Limited probe must use a separate --output directory.")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(MANIFEST)
    targets = manifest["selected_cases"][:args.limit or None]
    stored = {(r["case_id"], r["method"]): r for r in read_csv(FOLDER / "routing_correctness_cases.csv")}
    endpoints = []; candidates_out = []; flows = []; stages = []; six = []; nine = []
    instance_rows = []; details = []; assertions = []; source_files = []; montage = []
    for number, target in enumerate(targets, 1):
        cid = target["case_id"]
        package = HOLDOUT / "02_blind_inference" / cid
        reference = HOLDOUT / "01_sealed_references" / cid
        truth_path = HOLDOUT / "01_sealed_references" / target["target_mask_relative_path"]
        truth = load_mask(truth_path)
        assert sha(truth_path) == target["target_mask_sha256"], cid
        assert np.array_equal(truth, load_mask(FOLDER / "cases" / cid / "truth_part.png")), cid
        case = read_json(reference / "case.json")
        assert case["object_category"] == target["object_category"], cid
        candidate_json = read_json(package / "candidates.json")
        candidates = [fusion.MaskCandidate(semantic_name=r["semantic_name"], semantic_parent=r["semantic_parent"],
                       mask=load_mask(package / r["mask_path"]), score=r["score"], source=r["source"],
                       prompt=r.get("prompt", ""), source_reliability=r.get("source_reliability", 1.0),
                       metadata=r.get("metadata", {})) for r in candidate_json]
        observer = Observation(candidates, target, truth)
        raw = baseline.raw_proposals(candidates)
        observer.patch_fusion()
        try:
            fine = fusion.fuse_candidates(candidates)
        finally:
            observer.restore(); observer.originals = []
        fine_map, fine_records = load_previous_package(package)
        assert np.array_equal(fine.instance_map, fine_map), cid + " fine map"
        assert [(r.part_id, r.semantic_name) for r in fine.instances] == [(r.part_id, r.semantic_name) for r in fine_records], cid
        fine_instances = instances_from_map(fine_map, fine_records)
        observer.record_stage("fine_final_instances", fine_instances)
        observer.patch_groups()
        try:
            grouped = pg.build_physical_groups(fine_map, fine_records, candidates=candidates,
                                               image=Image.open(package / "source.png").convert("RGB"))
        finally:
            observer.restore(); observer.originals = []
        assert np.array_equal(grouped.group_map, np.asarray(Image.open(package / "group_id_map.tiff"))), cid + " group map"
        exported_groups = read_json(package / "groups.json")
        assert [(r.group_id, r.semantic_name) for r in grouped.groups] == [(r["group_id"], r["semantic_name"]) for r in exported_groups], cid
        group_instances = instances_from_map(grouped.group_map, grouped.groups, True)
        observer.record_stage("group_final_instances", group_instances)
        observer.patch_dbscan()
        try:
            dbscan = baseline.dbscan_proposal_fusion(candidates)
        finally:
            observer.restore(); observer.originals = []
        common = {"case_id": cid, "domain": target["expected_domain"], "category": target["object_category"],
                  "target": target["target_part_name"], "normalized_target": token(target)}
        predictions = {"raw_proposals": raw.instances, "dbscan_fusion": dbscan.instances,
                       "hpid_split_group_ids": group_instances}
        states = {}; matches = {}; selected = {}
        for method, instances in predictions.items():
            out, matches[method], selected[method] = state(instances, target, truth)
            states[method] = out
            old = stored[(cid, method)]
            for key in ("query_match_count", "selected_part_iou", "best_review_match_iou", "unique_and_correct_025",
                        "wrong_unique_025", "query_ambiguous", "query_unresolved", "correct_in_review_set_025",
                        "unique_and_correct_050", "wrong_unique_050", "correct_in_review_set_050"):
                assert abs(float(out[key]) - float(old[key])) < 1e-12, (cid, method, key, out[key], old[key])
            selected_mask = selected[method].mask if selected[method] else np.zeros(truth.shape, dtype=bool)
            assert np.array_equal(selected_mask, load_mask(FOLDER / "cases" / cid / method / "selected_part.png")), (cid, method)
            endpoints.append({**common, "method": method, **out})
            for instance in instances:
                instance_rows.append({**common, "method": method, "identity": instance.identity,
                                      "semantic": instance.semantic_name, "normalized_semantic": normal(instance.semantic_name, target),
                                      "matches_target": normal(instance.semantic_name, target) == token(target),
                                      **spatial(instance.mask, truth)})
        stages.extend({**common, **r} for r in observer.stages)
        root = baseline._root_candidate(candidates)
        proposals = sorted(baseline._non_root_candidates(candidates, root),
                           key=lambda c: (c.semantic_name, c.semantic_parent, c.source, baseline._mask_digest(c.mask)))
        cluster_of = {id(proposals[index]): ci for ci, cluster in enumerate(observer.clusters) for index in cluster}
        profile = pg._selected_profile(tuple(candidates))
        verified, rejected, gate_rows = pg._profile_candidate_verification(tuple(candidates), profile, fine_map > 0)
        gate_lookup = {r["candidate_key"]: r for r in gate_rows}
        correct_raw = []
        for index, candidate in enumerate(candidates, 1):
            is_match = normal(candidate.semantic_name, target) == token(target)
            correct = is_match and area(candidate.mask) >= 6 and iou(candidate.mask, truth) >= .25
            if correct:
                correct_raw.append(index)
            gate = gate_lookup.get(candidate.metadata.get("candidate_key"))
            cluster_index = cluster_of.get(id(candidate))
            cdata = {**common, "candidate_index": index, "candidate_key": candidate.metadata.get("candidate_key"),
                     "semantic": candidate.semantic_name, "semantic_parent": candidate.semantic_parent,
                     "source": candidate.source, "mask_sha256": sha(package / candidate_json[index-1]["mask_path"]),
                     "matches_target": is_match, "correct_raw_025": correct, **spatial(candidate.mask, truth),
                     "dedup_retained": index in observer.retention["_deduplicate"],
                     "instance_cap_retained": index in observer.retention["_filter_candidates_by_instance_caps"],
                     "hierarchical_suppression_retained": index in observer.retention["_suppress_hierarchical_duplicates"],
                     "group_profile": profile, "nominated_to_profile_verification": gate is not None,
                     "group_verification_accepted": gate["accepted"] if gate else None,
                     "semantic_verified_by_any_candidate": candidate.semantic_name in verified,
                     "semantic_rejected_all_candidates": candidate.semantic_name in rejected,
                     "dbscan_cluster": cluster_index}
            if cluster_index is not None:
                semantic, _parent, cluster_mask, _confidence = observer.db_rows[cluster_index]
                cdata.update({"dbscan_cluster_size": len(observer.clusters[cluster_index]),
                              "dbscan_cluster_semantic": semantic,
                              "dbscan_cluster_semantic_matches_target": normal(semantic, target) == token(target),
                              "dbscan_cluster_pre_ownership_iou": iou(cluster_mask, truth),
                              "dbscan_cluster_keeps_candidate_tp_px": area(cluster_mask & candidate.mask & truth)})
            candidates_out.append(cdata)
            if not is_match:
                continue
            # Pixel-set lineage is exact, but is not a causal attribution of
            # a fused pixel to this individual (possibly overlapping) input.
            tp = candidate.mask & truth
            for stage, instances in [("fine_final", fine_instances), ("group_final", group_instances),
                                     ("dbscan_final", dbscan.instances)]:
                union = np.zeros(truth.shape, dtype=bool)
                for instance in instances:
                    overlap = area(candidate.mask & instance.mask)
                    if overlap:
                        union |= instance.mask
                        flows.append({**common, "candidate_index": index, "candidate_key": candidate.metadata.get("candidate_key"),
                                      "correct_raw_025": correct, "candidate_iou": iou(candidate.mask, truth),
                                      "candidate_tp_px": area(tp), "stage": stage, "owner_identity": instance.identity,
                                      "owner_semantic": instance.semantic_name, "owner_matches_target": normal(instance.semantic_name, target) == token(target),
                                      "owner_iou": iou(instance.mask, truth), "candidate_pixels_to_owner": overlap,
                                      "candidate_tp_pixels_to_owner": area(tp & instance.mask)})
                flows.append({**common, "candidate_index": index, "candidate_key": candidate.metadata.get("candidate_key"),
                              "correct_raw_025": correct, "candidate_iou": iou(candidate.mask, truth), "candidate_tp_px": area(tp),
                              "stage": stage, "owner_identity": "background_or_unassigned", "owner_semantic": "background",
                              "owner_matches_target": False, "owner_iou": 0.0, "candidate_pixels_to_owner": area(candidate.mask & ~union),
                              "candidate_tp_pixels_to_owner": area(tp & ~union)})
                # Each final method is exclusive; these rows must partition
                # every candidate pixel and its true-positive support exactly.
                selected_flows = [f for f in flows if f["case_id"] == cid and f["candidate_index"] == index and f["stage"] == stage]
                assert sum(f["candidate_pixels_to_owner"] for f in selected_flows) == area(candidate.mask), (cid, index, stage)
                assert sum(f["candidate_tp_pixels_to_owner"] for f in selected_flows) == area(tp), (cid, index, stage)
        fine_state, _, _ = state(fine_instances, target, truth)
        summary_case = {**common, "raw_state": states["raw_proposals"]["state"],
                        "raw_correct_candidate_indexes": correct_raw, "raw_best_iou": states["raw_proposals"]["best_review_match_iou"],
                        "fine_state": fine_state["state"], "fine_best_iou": fine_state["best_review_match_iou"],
                        "dbscan_state": states["dbscan_fusion"]["state"], "dbscan_best_iou": states["dbscan_fusion"]["best_review_match_iou"],
                        "hpid_state": states["hpid_split_group_ids"]["state"], "hpid_best_iou": states["hpid_split_group_ids"]["best_review_match_iou"],
                        "root_target_recall": area(root.mask & truth)/max(1, area(truth)),
                        "profile_verified_semantics": sorted(verified), "profile_rejected_semantics": sorted(rejected)}
        details.append({**summary_case, "gate_rows": gate_rows, "group_diagnostics": grouped.diagnostics})
        if summary_case["raw_state"] == "ambiguous" and correct_raw:
            nine.append(summary_case)
        if states["hpid_split_group_ids"]["state"] == "unique_wrong":
            chosen = selected["hpid_split_group_ids"]
            ref_overlaps = []
            for part in case["parts"]:
                pmask = load_mask(reference / part["mask_crop"])
                ref_overlaps.append({"part": part["part_name"], "normalized": _normalize(part["part_name"], target["expected_domain"],
                                    object_category=target["object_category"]), **spatial(chosen.mask, pmask)})
            best_ref = max(ref_overlaps, key=lambda r: r["target_iou"])
            wrong_reference_support = best_ref["normalized"] != token(target) and best_ref["target_iou"] >= .25
            six.append({**summary_case, "selected_identity": chosen.identity, "selected_semantic": chosen.semantic_name,
                        "selected_normalized_semantic": normal(chosen.semantic_name, target),
                        "name_matches_target": normal(chosen.semantic_name, target) == token(target),
                        "failure_definition": "name_matches_but_target_IoU_below_0.25",
                        **spatial(chosen.mask, truth), "best_overlapping_reference": best_ref["part"],
                        "best_overlapping_reference_iou": best_ref["target_iou"],
                        "wrong_reference_has_IoU_at_least_025": wrong_reference_support,
                        "diagnostic_class": "semantic_label_with_support_on_different_reference_part" if wrong_reference_support else "semantic_match_with_deficient_spatial_support",
                        "group_evidence": next(g.evidence for g in grouped.groups if g.group_id == chosen.identity)})
            details[-1]["selected_group_reference_overlaps"] = ref_overlaps
            source = Image.open(package / "source.png").convert("RGB")
            def panel(mask, color, source=source):
                arr = np.asarray(source).copy()
                arr[mask] = (.45*arr[mask] + .55*np.array(color)).astype(np.uint8)
                img = Image.fromarray(arr); img.thumbnail((300, 250)); return img
            canvas = Image.new("RGB", (900, 285), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((5, 3), cid + " target=" + target["target_part_name"], fill="black")
            for j, (mask, label, color) in enumerate([(truth, "Reference", (0,180,40)),
                        (chosen.mask, f"HPID IoU {iou(chosen.mask,truth):.4f}", (230,30,30)),
                        (max(matches["raw_proposals"], key=lambda r:iou(r.mask,truth)).mask if matches["raw_proposals"] else truth*False,
                         f"Best raw IoU {summary_case['raw_best_iou']:.4f}", (20,80,240))]):
                img = panel(mask, color); canvas.paste(img, (300*j, 33)); draw.text((300*j+5, 18), label, fill="black")
            montage.append(canvas)
        assertions.append({"case_id": cid, "candidate_count": len(candidates), "fine_map_exact": True, "fine_labels_exact": True,
                           "group_map_exact": True, "group_labels_exact": True, "reference_hash_exact": True,
                           "routing_metrics_exact_methods": 3, "selected_masks_exact_methods": 3})
        for path in [package/"candidates.json", package/"part_id_map.tiff", package/"groups.json", package/"group_id_map.tiff", truth_path]:
            source_files.append({"case_id": cid, "path": str(path), "sha256": sha(path)})
        print(f"request provenance {number}/{len(targets)} {cid}", flush=True)

    intervals = []
    for level in ("all", "domain", "category"):
        groups = ["all"] if level == "all" else sorted({r[level] for r in endpoints})
        for group in groups:
            for method in ("raw_proposals", "dbscan_fusion", "hpid_split_group_ids"):
                subset = [r for r in endpoints if r["method"] == method and (level == "all" or r[level] == group)]
                for label in ("unique_correct", "unique_wrong", "ambiguous", "unresolved"):
                    intervals.append({"level": level, "stratum": group, "method": method, "state": label,
                                      **wilson(sum(r["state"] == label for r in subset), len(subset))})
                amb = [r for r in subset if r["state"] == "ambiguous"]
                intervals.append({"level": level, "stratum": group, "method": method, "state": "correct_option_given_ambiguous",
                                  **wilson(sum(r["has_correct_option_025"] for r in amb), len(amb))})
    outcomes = []
    for candidate in [r for r in candidates_out if r["correct_raw_025"]]:
        outcome = {k: candidate[k] for k in ("case_id", "domain", "category", "target", "candidate_index", "candidate_key", "semantic", "target_iou", "target_intersection_px")}
        for stage in ("fine_final", "group_final", "dbscan_final"):
            sub = [r for r in flows if r["case_id"] == candidate["case_id"] and r["candidate_index"] == candidate["candidate_index"] and r["stage"] == stage]
            outcome[stage + "_tp_retained_target"] = sum(r["candidate_tp_pixels_to_owner"] for r in sub if r["owner_matches_target"])
            outcome[stage + "_tp_other_semantics"] = sum(r["candidate_tp_pixels_to_owner"] for r in sub if not r["owner_matches_target"] and r["owner_semantic"] != "background")
            outcome[stage + "_tp_background"] = sum(r["candidate_tp_pixels_to_owner"] for r in sub if r["owner_semantic"] == "background")
            best = max(sub, key=lambda r:r["candidate_tp_pixels_to_owner"])
            outcome[stage + "_largest_tp_owner"] = best["owner_semantic"]
        outcomes.append(outcome)
    case_rows = [{k:v for k,v in r.items() if k not in ("gate_rows", "group_diagnostics", "selected_group_reference_overlaps")} for r in details]
    files = {"request_endpoints.csv": endpoints, "request_candidate_trace.csv": candidates_out,
             "request_candidate_pixel_flows.csv": flows, "request_stage_trace.csv": stages,
             "request_instances.csv": instance_rows, "request_six_wrong_cases.csv": six,
             "request_nine_correct_ambiguous.csv": nine, "request_intervals.csv": intervals,
             "request_assertions.csv": assertions, "request_source_hashes.csv": source_files,
             "request_correct_candidate_outcomes.csv": outcomes, "request_cases.csv": case_rows}
    for name, rows in files.items():
        write_csv(args.output / name, rows)
    write_json(args.output / "request_case_details.json", details)
    relevant = [r for r in candidates_out if r["correct_raw_025"]]
    protocol = {"case_count": len(targets), "independent_new_cases": 0, "same_frozen_holdout_requests": True,
                "target_manifest_sha256": sha(MANIFEST), "endpoint_source_sha256": sha(FOLDER/"routing_correctness_cases.csv"),
                "exact_assertion_count": len(assertions), "routing_method_case_reproductions": len(endpoints),
                "code_sha256": {str(p): sha(p) for p in [Path(fusion.__file__), Path(pg.__file__), Path(baseline.__file__),
                    SNAPSHOT/"src/hpid_split/paco_semantics.py", PROJECT/"hpid_split/scripts/audit_completion_routing.py", Path(__file__)]},
                "interval": "pointwise 95% Wilson; descriptive fixed-case proportions; no adjusted multiplicity or population-generalization claim",
                "correct_raw_candidate_count": len(relevant),
                "correct_raw_case_count": len({r["case_id"] for r in relevant}),
                "correct_raw_retained_each_fine_filter": {key: sum(bool(r[key]) for r in relevant) for key in
                    ("dedup_retained", "instance_cap_retained", "hierarchical_suppression_retained")},
                "nine_original_correct_ambiguous_transitions": dict(Counter(r["hpid_state"] for r in nine)),
                "six_failure_classes": dict(Counter(r["diagnostic_class"] for r in six)),
                "six_name_matches": sum(r["name_matches_target"] for r in six),
                "correct_raw_candidates_nominated_group_gates": sum(r["nominated_to_profile_verification"] for r in relevant),
                "correct_raw_candidates_accepted_group_gates": sum(r["group_verification_accepted"] is True for r in relevant),
                "correct_raw_candidates_rejected_group_gates": sum(r["group_verification_accepted"] is False for r in relevant),
                "correct_raw_cases_losing_correct_hpid_option": [r["case_id"] for r in case_rows if r["raw_correct_candidate_indexes"] and r["hpid_best_iou"] < .25],
                "limitations": [
                    "The available Raw pool is the exported pool after original fine-fusion input filtering; rejected proposal-generation masks are unavailable.",
                    "Candidate pixel-flow rows exactly partition an input candidate's support at later saved/replayed outputs. Overlapping candidates prevent unique causal attribution to an individual source.",
                    "Fine semantic stages use one union per class; final Part/Group and routing use instances. These stage match counts are not interchangeable.",
                    "Group naming can reconstruct physical partitions rather than merely merge candidate masks; membership IDs can be rebound after spatial regularization.",
                    "A name-matched output covering another annotated part is a spatial semantic-assignment warning, not independently adjudicated semantic misclassification.",
                    "Category strata are very small; Wilson intervals describe observed counts and do not establish generalization to new categories."]}
    write_json(args.output / "request_protocol.json", protocol)
    report = {
        "scope": "R1-6 and R2-6 post-review provenance diagnostic on the unchanged 37 frozen requests; zero new independent cases.",
        "evidence_files": {
            "request_assertions.csv": "37 exact fine-map/label and Group-map/label checks, 111 routing checks and 111 selected-mask checks.",
            "request_cases.csv": "Same-request Raw, fine, DBSCAN and Group endpoint linkage for every request.",
            "request_candidate_trace.csv": "Every exported candidate: semantic normalization, mask hash, fine-filter retention, Group-gate eligibility/decision and DBSCAN cluster.",
            "request_candidate_pixel_flows.csv": "Exact later ownership of each target-matched input candidate's pixels and true-positive pixels; conserved partition assertions.",
            "request_correct_candidate_outcomes.csv": "Compact ownership destinations of the 23 Raw-correct candidates.",
            "request_stage_trace.csv": "Observed target support before/after fine semantic cleanup, fine-instance assembly and actual Group spatial postprocessing.",
            "request_nine_correct_ambiguous.csv": "The same original nine ambiguous requests with a correct option, without changing denominators.",
            "request_six_wrong_cases.csv": "Name matching, target precision/recall, root coverage and strongest annotated reference overlap for all six unique failures.",
            "request_intervals.csv": "Counts/denominators and pointwise Wilson intervals, overall/by domain/by actual category.",
            "request_case_details.json": "Full replayed Group diagnostics and six-case comparison against every reference part.",
            "request_visual_audit_LOCAL_ONLY.png": "Local-only source-image visual audit; do not publish this asset without separate image-license review."
        },
        "suggested_main_text": [
            "On the same 37 controlled requests, exact replay reproduced all stored fine-Part and Group maps and all Raw/DBSCAN/HPID routing endpoints. The 23 Raw candidates reaching the IoU 0.25 criterion belonged to 18 requests and remained present through replayed fine-input filtering. Candidate retention did not guarantee retained target support: the per-request trace separates consolidation of semantic choices from reassignment of their pixels.",
            "Of the nine Raw-ambiguous requests containing a correct option, five became uniquely correct, two became uniquely wrong, and two remained ambiguous, with a correct option retained in only one of those two. In the box, second-car and scissors cases, target support was already below the criterion in the fine-Part output before Group construction. Thus the reduced endpoint ambiguity cannot be interpreted as unconditional removal of incorrect alternatives.",
            "All six uniquely wrong HPID outputs matched the requested normalized name but failed the target-mask IoU criterion. Two showed strong support on a differently annotated part (box side on lid, IoU 0.9302; knife blade on handle, IoU 0.9294). The remaining four lacked sufficient target support; for the second car, the selected windshield mask lay entirely inside the reference windshield but covered only 21.71% of it. A unique route is consequently neither a confidence estimate nor evidence of anatomical correctness."
        ],
        "precise_failures": [
            "box__u01 candidate 2: 6437 true-positive source pixels; final Group allocations are bottom 6157, side 270 and label 10. Candidate accepted by Group verification; fine target IoU 0.006989, final 0.006949. DBSCAN places this candidate in a four-member cluster named bottom before ownership (target IoU 0.624237), losing semantic side routing despite retained geometry.",
            "car_automobile__u02 candidate 2: 659 true-positive pixels; final windshield 203, window 348, roof 83, body 7, background 18. Correct candidate accepted by Group verification; fine target IoU 0.220321, final 0.217112. DBSCAN two-member cluster is named window (target IoU 0.523560), so windshield request becomes unresolved.",
            "scissors__u01 candidate 7: 3023 true-positive pixels; final handle 1407, body 1588, blade 28. Its Group verification rejects this candidate, but another handle candidate verifies the semantic name. Correct support is already lost before Group construction (fine best IoU 0.141906), and the final best IoU is unchanged. This rejection is not evidence that it caused the observed loss.",
            "shoe__u01 candidate 3: 2164 true-positive pixels; fine upper receives 237, final Group upper 1040, body 1106 and eyelet 18. Group attachment improves IoU from 0.093417 to 0.206917 but remains below threshold. DBSCAN remains correct (0.316625).",
            "telephone__u01 and bottle__u01 already have no correct Raw target option. Bottle's automatic root covers only 0.4783% of the reference base; its selected Group has zero target intersection. Root replacement was not performed in this trace and is not inferred as a demonstrated repair.",
            "screwdriver__u01 is actually the reference category knife. Its Raw blade candidate already targets the handle; fine blade is absent, while Group knife structural partition creates a blade-named region overlapping the reference handle. This is a persistent spatial-semantic mismatch, not novel-taxonomy generalization."
        ],
        "claim_boundaries": protocol["limitations"],
        "verification": protocol,
    }
    write_json(args.output / "request_report.json", report)
    if montage:
        total = Image.new("RGB", (900, 285*len(montage)), "white")
        for i, canvas in enumerate(montage):
            total.paste(canvas, (0, i*285))
        # Local visual QA only: licensed source images are not public evidence.
        visual_path = args.output / "request_visual_audit_LOCAL_ONLY.png"
        total.save(visual_path)
    print(json.dumps(protocol, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
