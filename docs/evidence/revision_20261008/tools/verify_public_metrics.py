"""Independently audit published numeric evidence; never read image pixels.

Usage: python verify_public_metrics.py BUNDLE_OR_EVIDENCE --report report.json
Requires NumPy only for independent bootstrap quantiles. No HPID imports, model
weights, source photographs, reference masks, or mutable inference are used.
This verifies arithmetic and internal agreement, not image-ground-truth truthfulness.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import re
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def boolean(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "1.0")


def numeric(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite numeric value {value!r}")
    return result


def average(rows, field):
    return math.fsum(numeric(r[field]) for r in rows) / len(rows)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


class Audit:
    def __init__(self, evidence):
        self.evidence = evidence
        self.checks = Counter()
        self.failures = []
        self.sections = {}
        self.ci_jobs = defaultdict(list)
        self.inputs = {}

    def record(self, ok, label, detail=None):
        self.checks[label.split("/")[0]] += 1
        if not ok:
            self.failures.append({"check": label, "detail": detail})

    def eq(self, actual, expected, label):
        self.record(actual == expected, label, {"actual": actual, "expected": expected})

    def near(self, actual, expected, label, tolerance=2e-10):
        actual, expected = numeric(actual), numeric(expected)
        self.record(abs(actual - expected) <= tolerance * max(1.0, abs(expected)), label,
                    {"actual": actual, "expected": expected, "abs_difference": abs(actual-expected)})

    def read(self, name):
        path = self.evidence / name
        self.inputs[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return load_csv(path) if path.suffix == ".csv" else load_json(path)

    def unique(self, rows, fields, label):
        index = {tuple(r[k] for k in fields): r for r in rows}
        self.eq(len(index), len(rows), label)
        return index

    def ci(self, label, values, low, high, categories=None, seed=20261008, draws=10000):
        # Group all interval jobs sharing the sampling design. Multinomial
        # count vectors independently reconstruct resampling means by matrix
        # arithmetic instead of copying any experiment's bootstrap function.
        key = (len(values), tuple(categories) if categories else (), seed, draws)
        self.ci_jobs[key].append((label, np.asarray(values, dtype=float), numeric(low), numeric(high)))

    def flush_ci(self):
        total = 0
        for (n, categories, seed, draws), jobs in self.ci_jobs.items():
            if categories:
                units = sorted(set(categories))
                columns = {u:i for i,u in enumerate(units)}
                membership = np.zeros((len(units), n), dtype=np.float64)
                for i, category in enumerate(categories):
                    membership[columns[category], i] = 1
            else:
                units = list(range(n))
                membership = np.eye(n, dtype=np.float64)
            picks = np.random.default_rng(seed).integers(0, len(units), size=(draws, len(units)))
            counts = np.zeros((draws, len(units)), dtype=np.float64)
            np.add.at(counts, (np.repeat(np.arange(draws), len(units)), picks.ravel()), 1)
            case_weights = counts @ membership
            denominators = case_weights.sum(axis=1)
            for start in range(0, len(jobs), 80):
                block = jobs[start:start+80]
                values = np.stack([job[1] for job in block], axis=1)
                distribution = (case_weights @ values) / denominators[:, None]
                quantiles = np.quantile(distribution, [.025, .975], axis=0)
                for j, (label, values, low, high) in enumerate(block):
                    self.near(quantiles[0,j], low, label + "/bootstrap_low")
                    self.near(quantiles[1,j], high, label + "/bootstrap_high")
                    total += 1
        return total

    def section(self, name, function):
        before = len(self.failures)
        try:
            information = function()
        except Exception as error:
            self.failures.append({"check": name + "/exception", "detail": str(error)})
            information = {}
        self.sections[name] = {"status": "PASS" if len(self.failures) == before else "FAIL", **(information or {})}

    def sensitivity(self):
        rows = self.read("sensitivity_cases.csv")
        summaries = self.read("sensitivity_summary.csv")
        protocol = self.read("sensitivity_protocol.json")
        lookup = self.unique(rows, ("case_id", "variant"), "sensitivity/unique_case_variant")
        case_ids = sorted({r["case_id"] for r in rows})
        self.eq(len(case_ids), 226, "sensitivity/cases")
        variants = set(protocol["variants"])
        self.eq(len(variants), 50, "sensitivity/settings")
        self.eq(len(rows), len(case_ids)*len(variants), "sensitivity/complete_cross_product")
        self.eq({r["variant"] for r in summaries}, variants, "sensitivity/summary_variants")
        metrics = [k for k in rows[0] if k not in ("case_id", "domain", "category", "variant", "pixel_change_fraction", "exact_release_map")]
        for row in rows:
            changed = numeric(row["pixel_change_fraction"])
            self.record(0 <= changed <= 1, "sensitivity/changed_fraction_range")
            self.eq(boolean(row["exact_release_map"]), changed == 0, "sensitivity/exact_flag_agreement")
        for summary in summaries:
            v = summary["variant"]
            sub = [lookup[(cid, v)] for cid in case_ids]
            self.eq(int(summary["n"]), len(sub), "sensitivity/summary_n")
            self.eq(int(summary["changed_cases"]), sum(not boolean(r["exact_release_map"]) for r in sub), "sensitivity/changed_count")
            for metric in metrics:
                values = [numeric(r[metric]) for r in sub]
                differences = [numeric(r[metric])-numeric(lookup[(r["case_id"], "release")][metric]) for r in sub]
                self.near(statistics.fmean(values), summary[metric+"_mean"], "sensitivity/"+v+"/"+metric+"/mean")
                self.near(statistics.fmean(differences), summary["delta_"+metric+"_mean"], "sensitivity/"+v+"/"+metric+"/delta")
                self.ci("sensitivity/"+v+"/"+metric, values, summary[metric+"_low"], summary[metric+"_high"])
                self.ci("sensitivity/"+v+"/delta_"+metric, differences, summary["delta_"+metric+"_low"], summary["delta_"+metric+"_high"])
        self.sensitivity_release = {cid:lookup[(cid,"release")] for cid in case_ids}
        return {"cases": len(case_ids), "settings": len(variants), "rows": len(rows), "mean_and_paired_delta_checks": len(summaries)*len(metrics)*2}

    def roots(self):
        rows = self.read("holdout_root_cases.csv")
        summary = self.read("holdout_root_summary.json")
        lookup = self.unique(rows, ("case_id", "condition"), "roots/unique_case_condition")
        case_ids = list(dict.fromkeys(r["case_id"] for r in rows))
        self.eq(len(case_ids), 42, "roots/case_count")
        self.eq(len(rows), 84, "roots/paired_rows")
        for condition in ("automatic_root", "reference_root"):
            sub = [lookup[(cid, condition)] for cid in case_ids]
            for metric, result in summary[condition].items():
                values = [numeric(r[metric]) for r in sub]
                self.near(statistics.fmean(values), result["mean"], "roots/"+condition+"/"+metric)
                self.ci("roots/"+condition+"/"+metric, values, result["low"], result["high"])
        for metric, result in summary["reference_minus_automatic"].items():
            dif = [numeric(lookup[(cid,"reference_root")][metric])-numeric(lookup[(cid,"automatic_root")][metric]) for cid in case_ids]
            self.near(statistics.fmean(dif), result["mean"], "roots/paired_delta/"+metric)
            self.ci("roots/paired_delta/"+metric, dif, result["low"], result["high"])
        automatic = [lookup[(cid,"automatic_root")] for cid in case_ids]
        audit = summary["audit"]
        for out, field in [("exact_fine_map_reproductions", "original_map_reproduced"),
                           ("root_identical_to_reference", "root_equals_reference"),
                           ("reference_in_commands", "reference_passed_to_original_command"),
                           ("category_prompts", "category_prompt_passed")]:
            self.eq(audit[out], sum(boolean(r[field]) for r in automatic), "roots/"+out)
        self.root_ids = set(case_ids)
        return {"cases": 42, "paired_diagnostic_only": True}

    def stages(self):
        rows = self.read("stage_case_metrics.csv")
        tracks = self.read("stage_reference_tracks.csv")
        shared = self.read("stage_shared_input_audit.csv")
        protocol = self.read("stage_protocol.json")
        summaries = self.read("stage_summary.csv")
        transitions = self.read("stage_transitions.csv")
        audit = self.read("stage_audit_summary.json")
        lookup = self.unique(rows, ("case_id", "stage"), "stages/unique_case_stage")
        track_lookup = self.unique(tracks, ("case_id", "reference_index", "stage"), "stages/unique_reference_stage")
        case_ids = sorted(r["case_id"] for r in shared)
        order = protocol["stage_order"]
        self.eq(len(case_ids), 226, "stages/cases")
        self.eq(len(rows), 226*17, "stages/case_cross_product")
        self.eq(len(tracks), 990*17, "stages/reference_cross_product")
        self.eq(set(r["stage"] for r in summaries), set(order), "stages/summary_stages")
        categories = [lookup[(cid, order[0])]["category"] for cid in case_ids]
        tracks_by_case_stage = defaultdict(list)
        for row in tracks:
            tracks_by_case_stage[(row["case_id"], row["stage"])].append(row)
            for metric in ("best_iou", "best_semantic_iou", "semantic_union_recall", "semantic_union_precision", "any_union_recall", "hungarian_iou", "semantic_hungarian_iou"):
                self.record(0 <= numeric(row[metric]) <= 1, "stages/reference_metric_range")
            self.record(numeric(row["best_iou"]) + 1e-8 >= numeric(row["best_semantic_iou"]), "stages/best_semantic_subset")
        metrics = [k[:-5] for k in summaries[0] if k.endswith("_mean")]
        for row in rows:
            sub = tracks_by_case_stage[(row["case_id"], row["stage"])]
            ntruth, npred = int(row["reference_count"]), int(row["prediction_count"])
            self.eq(len(sub), ntruth, "stages/reference_denominator")
            m = sum(numeric(r["hungarian_iou"]) >= .25 for r in sub)
            sm = sum(numeric(r["semantic_hungarian_iou"]) >= .25 for r in sub)
            self.eq(int(row["matches_at_025"]), m, "stages/hungarian_match_count")
            self.eq(int(row["semantic_matches_at_025"]), sm, "stages/semantic_match_count")
            self.near(row["diagnostic_f1_at_025"], 2*m/max(1,ntruth+npred), "stages/f1_from_counts")
            self.near(row["diagnostic_semantic_f1_at_025"], 2*sm/max(1,ntruth+npred), "stages/semantic_f1_from_counts")
            for metric in ("best_iou", "best_semantic_iou", "semantic_union_recall", "any_union_recall"):
                self.near(row[metric], average(sub,metric), "stages/case_mean_from_references")
        for summary in summaries:
            stage = summary["stage"]
            sub = [lookup[(cid,stage)] for cid in case_ids]
            self.eq(int(summary["n"]), len(sub), "stages/summary_n")
            self.eq(int(summary["total_matches_at_025"]), sum(int(r["matches_at_025"]) for r in sub), "stages/total_matches")
            self.eq(int(summary["total_prediction_count"]), sum(int(r["prediction_count"]) for r in sub), "stages/total_predictions")
            for metric in metrics:
                values = [numeric(r[metric]) for r in sub]
                self.near(summary[metric+"_mean"], statistics.fmean(values), "stages/"+stage+"/"+metric)
                self.ci("stages/"+stage+"/"+metric, values, summary[metric+"_low"], summary[metric+"_high"], categories)
        for transition in transitions:
            before, after = transition["before"], transition["after"]
            for metric in metrics:
                dif = [numeric(lookup[(cid,after)][metric])-numeric(lookup[(cid,before)][metric]) for cid in case_ids]
                field = "delta_"+metric
                self.near(transition[field+"_mean"], statistics.fmean(dif), "stages/transition/"+field)
                self.ci("stages/transition/"+before+"/"+after+"/"+field, dif, transition[field+"_low"], transition[field+"_high"], categories)
            for metric in ("best_iou", "best_semantic_iou", "semantic_union_recall", "any_union_recall"):
                dif = [numeric(track_lookup[(r["case_id"],r["reference_index"],after)][metric])-numeric(r[metric]) for r in tracks if r["stage"] == before]
                self.eq(int(transition[metric+"_references_decreased"]), sum(d < -1e-10 for d in dif), "stages/decreased_references")
                self.eq(int(transition[metric+"_references_increased"]), sum(d > 1e-10 for d in dif), "stages/increased_references")
        for field, source in [("total_candidates","candidate_count"), ("total_references","reference_count"),
                              ("remainder_components_attached","remainder_components_attached"), ("visibility_slivers_dropped","identity_visibility_slivers_dropped")]:
            self.eq(audit[field], sum(int(r[source]) for r in shared), "stages/audit_"+field)
        self.eq(audit["exact_hpid_maps"], sum(boolean(r["exact_hpid_map_reproduction"]) for r in shared), "stages/exact_maps_from_flags")
        self.eq(audit["category_clusters"], len(set(categories)), "stages/categories")
        for cid in case_ids:
            self.near(lookup[(cid,"final_hpid")]["diagnostic_f1_at_025"], self.sensitivity_release[cid]["part_f1_at_025"], "stages/cross_sensitivity_f1")
            self.near(lookup[(cid,"final_hpid")]["prediction_count"], self.sensitivity_release[cid]["predicted_part_count"], "stages/cross_sensitivity_parts")
        db = self.read("dbscan_diagnosis.json")
        differences = [numeric(lookup[(cid,"final_hpid")]["diagnostic_f1_at_025"])-numeric(lookup[(cid,"final_dbscan")]["diagnostic_f1_at_025"]) for cid in case_ids]
        self.eq(db["case_comparison"], {"hpid_higher":sum(v>1e-10 for v in differences), "dbscan_higher":sum(v< -1e-10 for v in differences), "equal":sum(abs(v)<=1e-10 for v in differences)}, "stages/dbscan_comparison")
        self.near(statistics.fmean(differences), db["category_cluster_bootstrap"]["delta"], "stages/dbscan_delta")
        self.ci("stages/dbscan_delta", differences, db["category_cluster_bootstrap"]["low"], db["category_cluster_bootstrap"]["high"], categories)
        return {"cases":226, "stages":len(order), "reference_tracks":len(tracks), "candidate_total":audit["total_candidates"]}

    def requests(self):
        endpoints = self.read("request_endpoints.csv")
        instances = self.read("request_instances.csv")
        cases = self.read("request_cases.csv")
        candidates = self.read("request_candidate_trace.csv")
        flows = self.read("request_candidate_pixel_flows.csv")
        outcomes = self.read("request_correct_candidate_outcomes.csv")
        six = self.read("request_six_wrong_cases.csv")
        nine = self.read("request_nine_correct_ambiguous.csv")
        protocol = self.read("request_protocol.json")
        lookup = self.unique(endpoints, ("case_id","method"), "requests/unique_case_method")
        case_lookup = self.unique(cases, ("case_id",), "requests/unique_case")
        candidate_lookup = self.unique(candidates, ("case_id","candidate_index"), "requests/unique_candidate")
        self.eq(len(cases), 37, "requests/cases")
        self.eq(len(endpoints), 111, "requests/rows")
        self.record({r["case_id"] for r in cases} <= self.root_ids, "requests/reused_holdout_subset")
        by_case_method = defaultdict(list)
        for instance in instances:
            by_case_method[(instance["case_id"],instance["method"])].append(instance)
            self.eq(boolean(instance["matches_target"]), instance["normalized_semantic"] == instance["normalized_target"], "requests/normalized_match")
            self.near(instance["target_precision"], int(instance["target_intersection_px"])/max(1,int(instance["area_px"])), "requests/precision_arithmetic")
            self.record(0 <= numeric(instance["target_iou"]) <= 1, "requests/iou_range")
        for row in endpoints:
            matched = [r for r in by_case_method[(row["case_id"],row["method"])] if boolean(r["matches_target"])]
            selected = next((r for r in matched if r["identity"] == row["selected_identity"]), None)
            self.eq(int(row["query_match_count"]), len(matched), "requests/match_count_from_instances")
            actual_selected = numeric(selected["target_iou"]) if selected else 0.0
            best = max([numeric(r["target_iou"]) for r in matched], default=0.0)
            self.near(row["selected_part_iou"], actual_selected, "requests/selected_iou_from_instances")
            self.near(row["best_review_match_iou"], best, "requests/best_iou_from_instances")
            expected = "unresolved" if not matched else "ambiguous" if len(matched)>1 else "unique_correct" if actual_selected >= .25 else "unique_wrong"
            self.eq(row["state"], expected, "requests/state_from_instances")
            for threshold, suffix in ((.25,"025"),(.5,"050")):
                for field, answer in [("unique_and_correct_"+suffix, len(matched)==1 and actual_selected>=threshold),
                                      ("wrong_unique_"+suffix,len(matched)==1 and actual_selected<threshold),
                                      ("correct_in_review_set_"+suffix,len(matched)>1 and best>=threshold)]:
                    self.eq(boolean(row[field]), answer, "requests/threshold_flags")
        good = [r for r in candidates if boolean(r["correct_raw_025"])]
        self.eq(len(good), protocol["correct_raw_candidate_count"], "requests/correct_candidate_count")
        self.eq(len({r["case_id"] for r in good}), protocol["correct_raw_case_count"], "requests/correct_case_count")
        for row in candidates:
            correct = boolean(row["matches_target"]) and int(row["area_px"]) >= 6 and numeric(row["target_iou"]) >= .25
            self.eq(boolean(row["correct_raw_025"]), correct, "requests/correct_candidate_definition")
        for flag in ("dedup_retained", "instance_cap_retained", "hierarchical_suppression_retained"):
            self.eq(protocol["correct_raw_retained_each_fine_filter"][flag], sum(boolean(r[flag]) for r in good), "requests/retention_count")
        grouped_flows = defaultdict(list)
        for row in flows:
            grouped_flows[(row["case_id"],row["candidate_index"],row["stage"])].append(row)
        for (cid,index,stage), sub in grouped_flows.items():
            c = candidate_lookup[(cid,index)]
            self.eq(sum(int(r["candidate_pixels_to_owner"]) for r in sub), int(c["area_px"]), "requests/complete_pixel_partition")
            self.eq(sum(int(r["candidate_tp_pixels_to_owner"]) for r in sub), int(c["target_intersection_px"]), "requests/complete_true_positive_partition")
        for row in outcomes:
            for stage in ("fine_final","group_final","dbscan_final"):
                sub = grouped_flows[(row["case_id"],row["candidate_index"],stage)]
                self.eq(int(row[stage+"_tp_retained_target"]), sum(int(r["candidate_tp_pixels_to_owner"]) for r in sub if boolean(r["owner_matches_target"])), "requests/compact_target_tp_agreement")
                self.eq(int(row[stage+"_tp_background"]), sum(int(r["candidate_tp_pixels_to_owner"]) for r in sub if r["owner_semantic"]=="background"), "requests/compact_background_tp_agreement")
        bad_ids = {r["case_id"] for r in endpoints if r["method"]=="hpid_split_group_ids" and r["state"]=="unique_wrong"}
        self.eq(bad_ids, {r["case_id"] for r in six}, "requests/six_failure_selection")
        self.eq(len(six), 6, "requests/six_count")
        self.eq(Counter(r["diagnostic_class"] for r in six), Counter(protocol["six_failure_classes"]), "requests/six_classes")
        for row in six:
            self.eq(boolean(row["name_matches_target"]), row["selected_normalized_semantic"]==row["normalized_target"], "requests/six_name_matching")
            self.record(numeric(row["target_iou"])<.25, "requests/six_IoU_failure")
        raw_nine = {r["case_id"] for r in endpoints if r["method"]=="raw_proposals" and r["state"]=="ambiguous" and numeric(r["best_review_match_iou"])>=.25}
        self.eq(raw_nine, {r["case_id"] for r in nine}, "requests/original_nine_membership")
        self.eq(Counter(r["hpid_state"] for r in nine), Counter(protocol["nine_original_correct_ambiguous_transitions"]), "requests/nine_transitions")
        self.endpoint_lookup = lookup
        self.endpoints = endpoints
        return {"requests":37,"method_cases":111,"correct_raw_candidates":len(good),"unique_failures":len(six),"original_correct_ambiguous":len(nine)}

    def routing(self):
        def interval(k,n):
            if n==0:
                return (None,None,None)
            # Solve the Wilson score inequality as a quadratic in p.
            z2 = statistics.NormalDist().inv_cdf(.975)**2
            a, b, c = n+z2, -(2*k+z2), k*k/n
            d = math.sqrt(max(0,b*b-4*a*c))
            return k/n, max(0,(-b-d)/(2*a)), min(1,(-b+d)/(2*a))
        rows = self.read("request_intervals.csv")
        for row in rows:
            subset = [r for r in self.endpoints if r["method"]==row["method"] and (row["level"]=="all" or r[row["level"]]==row["stratum"])]
            if row["state"]=="correct_option_given_ambiguous":
                subset = [r for r in subset if r["state"]=="ambiguous"]
                k = sum(numeric(r["best_review_match_iou"])>=.25 for r in subset)
            else:
                k = sum(r["state"]==row["state"] for r in subset)
            n = len(subset)
            self.eq(int(row["k"]), k, "routing/count_from_same_requests")
            self.eq(int(row["n"]), n, "routing/denominator_from_same_requests")
            for field, value in zip(("rate","low","high"), interval(k,n)):
                if value is None:
                    self.record(row[field] in ("",None), "routing/empty_interval")
                else:
                    self.near(row[field], value, "routing/wilson_"+field)
        older = self.read("routing_intervals.csv")
        for row in older:
            k,n = int(row["k"]),int(row["n"])
            for field,value in zip(("rate","low","high"),interval(k,n)):
                if value is not None:
                    self.near(row[field],value,"routing/archived_wilson")
            subset = [r for r in self.endpoints if r["method"]==row["method"]]
            if subset:
                if row["endpoint"].startswith("correct_in_review"):
                    subset = [r for r in subset if r["state"]=="ambiguous"]
                self.eq(n,len(subset),"routing/old_new_denominator")
                self.eq(k,sum(boolean(r[row["endpoint"]]) for r in subset),"routing/old_new_count")
        transitions = self.read("routing_transitions.csv")
        mapping = {"unique_correct":"unique_and_correct_025","unique_wrong":"wrong_unique_025","ambiguous":"query_ambiguous","unresolved":"query_unresolved"}
        for row in transitions:
            for prefix, method in (("raw","raw_proposals"),("hpid","hpid_split_group_ids")):
                expected = self.endpoint_lookup[(row["case_id"],method)]
                self.eq(row[prefix+"_state"],mapping[expected["state"]],"routing/transition_same_request")
                self.near(row[prefix+"_best_iou"],expected["best_review_match_iou"],"routing/transition_best_iou")
        synthesis = self.read("synthesis_diagnostics.csv")
        sa = self.read("routing_audit.json")
        self.eq(len(synthesis),37,"routing/synthesis_same_37")
        for row in synthesis:
            self.near(row["delta_request_recall"],numeric(row["hpid_request_recall"])-numeric(row["raw_request_recall"]),"routing/synthesis_recall_delta")
        for result in sa["synthesis_by_recall_change"]:
            sign = result["recall_change_sign"]
            sub = [r for r in synthesis if (1 if numeric(r["delta_request_recall"])>0 else -1 if numeric(r["delta_request_recall"])<0 else 0)==sign]
            self.eq(len(sub),result["n"],"routing/synthesis_sign_n")
            self.near(average(sub,"delta_psnr"),result["mean_psnr_change"],"routing/synthesis_sign_mean")
        for field, metric in (("mean_psnr_delta","delta_psnr"),("mean_request_recall_delta","delta_request_recall")):
            values = [numeric(r[metric]) for r in synthesis]
            self.near(statistics.fmean(values),sa[field]["mean"],"routing/synthesis_mean")
            self.ci("routing/"+field,values,sa[field]["low"],sa[field]["high"])
        return {"request_interval_rows":len(rows),"earlier_interval_rows":len(older),"synthesis_cases":len(synthesis)}

    def gates(self):
        p = "gate_ablation/"
        cases = self.read(p+"group_gate_cases.csv")
        summaries = self.read(p+"group_gate_summary.csv")
        pool = self.read(p+"group_gate_pool.csv")
        complete = self.read(p+"group_gate_complete_exported_pool.csv")
        calls = self.read(p+"group_gate_actual_calls.csv")
        protocol = self.read(p+"group_gate_protocol.json")
        validation = self.read(p+"gate_validation_checks.json")
        repeat = self.read(p+"gate_repeat_pixel_audit.json")
        lookup = self.unique(cases,("split","case_id","variant"),"gates/unique_case_variant")
        whole = self.unique(complete,("split","case_id","candidate_key"),"gates/unique_complete_candidate")
        nominated = self.unique(pool,("split","case_id","candidate_key"),"gates/unique_nominated_candidate")
        variants = protocol["variants"]
        self.eq(len(cases),268*8,"gates/complete_cross_product")
        self.eq(len(complete),2400,"gates/complete_candidates")
        self.eq(len(pool),1280,"gates/nominated_candidates")
        self.eq(len(calls),validation["actual_gate_invocations"],"gates/call_count")
        pool_by_case = defaultdict(list)
        for row in pool:
            key = (row["split"],row["case_id"],row["candidate_key"])
            self.record(key in whole,"gates/nominations_belong_to_complete_pool")
            self.eq(row["mask_sha256"],whole[key]["mask_sha256"],"gates/nominated_mask_hash_identity")
            self.record(boolean(whole[key]["profile_gate_nomination"]),"gates/nomination_flag")
            pool_by_case[key[:2]].append(row)
            self.eq(boolean(row["accepted"]),all(boolean(row[f]) for f in ("semantic_pass","structure_raw_pass","appearance_raw_pass")),"gates/release_candidate_predicate")
        contexts = defaultdict(set)
        for row in calls:
            bits = [boolean(row[field]) for field in ("semantic_raw_pass","structure_raw_pass","appearance_raw_pass")]
            expected = all((not enabled) or bit for enabled,bit in zip(variants[row["variant"]],bits))
            self.eq(boolean(row["intervention_accepted"]),expected,"gates/actual_call_predicate")
            self.eq(boolean(row["release_accepted"]),all(bits),"gates/release_call_predicate")
            contexts[(row["split"],row["case_id"],row["candidate_key"],row["root_context"])].add(tuple(bits))
        self.record(all(len(values)==1 for values in contexts.values()),"gates/raw_predicates_invariant")
        self.eq(len(contexts),validation["unique_candidate_contexts"],"gates/context_count")
        for row in cases:
            split,cid,v = row["split"],row["case_id"],row["variant"]
            baseline_row = lookup[(split,cid,"STA")]
            eligible = pool_by_case[(split,cid)]
            retain = sum(all((not enabled) or boolean(candidate[field]) for enabled,field in zip(variants[v],("semantic_pass","structure_raw_pass","appearance_raw_pass"))) for candidate in eligible)
            self.eq(int(row["retained_candidates"]),retain,"gates/retained_from_raw_predicates")
            self.eq(int(row["pre_group_gate_candidates"]),len(eligible),"gates/pool_count")
            self.eq(row["fixed_input_sha256"],baseline_row["fixed_input_sha256"],"gates/same_case_inputs")
            self.record(boolean(row["fixed_inputs_unchanged"]),"gates/unchanged_input_flag")
            self.eq(boolean(row["group_map_changed"]),row["group_map_sha256"]!=baseline_row["group_map_sha256"],"gates/hash_change_agreement")
            precision,recall = numeric(row["candidate_precision"]),numeric(row["candidate_recall"])
            self.near(row["candidate_f1"],2*precision*recall/(precision+recall) if precision+recall else 0,"gates/candidate_f1_arithmetic")
            if not eligible:
                self.eq(retain,0,"gates/zero_nomination_retained")
                for metric in ("candidate_precision","candidate_recall","candidate_f1"):
                    self.near(row[metric],0,"gates/zero_nomination_metric")
            if v=="STA":
                self.record(boolean(row["release_group_map_reproduced"]),"gates/baseline_exact_flag")
        for summary in summaries:
            split,variant = summary["split"],summary["variant"]
            sub = sorted((r for r in cases if r["split"]==split and r["variant"]==variant),key=lambda r:r["case_id"])
            expected_n = 42 if split=="holdout42" else 226
            self.eq(len(sub),expected_n,"gates/split_n")
            totals = {"n":len(sub),"pre_gate_pool_total":sum(int(r["pre_group_gate_candidates"]) for r in sub),
                      "complete_exported_pool_total":sum(int(r["complete_exported_candidates"]) for r in sub),
                      "retained_total":sum(int(r["retained_candidates"]) for r in sub),
                      "cases_with_profile_nominations":sum(int(r["pre_group_gate_candidates"])>0 for r in sub),
                      "cases_without_profile_nominations":sum(int(r["pre_group_gate_candidates"])==0 for r in sub),
                      "changed_group_maps":sum(boolean(r["group_map_changed"]) for r in sub),
                      "changed_semantic_partitions":sum(boolean(r["semantic_partition_changed"]) for r in sub)}
            for key,value in totals.items():
                self.eq(int(summary[key]),value,"gates/summary_"+key)
            for metric in [k[:-5] for k in summary if k.endswith("_mean") and not k.startswith("delta_")]:
                values = [numeric(r[metric]) for r in sub]
                dif = [numeric(r[metric])-numeric(lookup[(split,r["case_id"],"STA")][metric]) for r in sub]
                self.near(summary[metric+"_mean"],statistics.fmean(values),"gates/summary_mean")
                self.near(summary["delta_"+metric+"_mean"],statistics.fmean(dif),"gates/summary_delta")
                self.ci("gates/"+split+"/"+variant+"/"+metric,values,summary[metric+"_low"],summary[metric+"_high"])
                self.ci("gates/"+split+"/"+variant+"/delta_"+metric,dif,summary["delta_"+metric+"_low"],summary["delta_"+metric+"_high"])
        nonrepeatable = any(int(r["pixels_differ_across_fixed_seed_runs"])>0 for r in repeat)
        self.eq(validation["counterfactual_maps_bitwise_repeatable"],not nonrepeatable,"gates/disclose_repeat_limit")
        self.record(all(int(r["pixels_differ_across_fixed_seed_runs"])==0 for r in repeat if r["variant"]=="STA"),"gates/repeated_release_flags")
        return {"cases":268,"variants":8,"rows":len(cases),"raw_predicate_contexts":len(contexts),"documented_counterfactual_pixel_repeat_limit":nonrepeatable}

    def scores(self):
        rows = self.read("score_candidate_provenance.csv")
        families = self.read("score_family_ranges.csv")
        self.eq(len(rows),1951,"scores/candidate_count")
        for row in rows:
            s,r,a = numeric(row["score"]),numeric(row["source_reliability"]),numeric(row["agreement"])
            clipped = min(1,max(0,s))
            before = min(.995,max(.01,(.45+.55*clipped)*r*a))
            effective = min(.995,max(.01,before*.72)) if boolean(row["broad_scene_layer"]) else before
            self.near(row["clipped_score"],clipped,"scores/score_clip")
            self.near(row["weight_before_scene_factor"],before,"scores/heuristic_weight")
            self.near(row["effective_weight"],effective,"scores/effective_weight")
        for family in families:
            sub = [r for r in rows if r["family"]==family["family"]]
            self.eq(int(family["n"]),len(sub),"scores/family_n")
            self.eq(int(family["case_count"]),len({r["case_id"] for r in sub}),"scores/family_cases")
            for field in ("score","source_reliability","agreement","weight_before_scene_factor","effective_weight"):
                self.near(family[field+"_min"],min(numeric(r[field]) for r in sub),"scores/family_min")
                self.near(family[field+"_max"],max(numeric(r[field]) for r in sub),"scores/family_max")
        return {"candidates":len(rows),"families":len(families)}


def privacy_audit(bundle):
    """Inspect names/text and decode each allowed generated mask or ID map."""
    import io
    from PIL import Image
    findings = []
    counts = Counter()
    hard_names = re.compile(r"(?:LOCAL_ONLY|qa_content_|ASTRA_HANDOFF|reviewer_tracker|author.?portrait|decision.?letter|submitted_baseline|(?:^|/)\.env(?:$|[./]))",re.I)
    forbidden_image = re.compile(r"(?:^|/)(?:source|source_overlay|truth_part|controlled_defect|corrupted|completed|object_mask_crop|reference[^/]*|portrait)[^/]*\.(?:png|jpe?g|tiff?|webp)$",re.I)
    raw_path = re.compile(r"(?:[A-Z]:(?:\\+|/)(?:Users|HPID_Gaussian|Codex)|/\x55sers/[^/]+/)",re.I)
    secret = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")
    allowed_image = re.compile(r"(?:candidate_masks/[^/]+\.png|(?:part|group)_id_map\.tiff?|masks_visible/[^/]+\.png|/(?:raw_proposals|dbscan_fusion|hpid_split_group_ids)/selected_part\.png)$",re.I)
    images = re.compile(r"\.(?:png|jpe?g|tiff?|webp|gif)$",re.I)
    text_exts = {".csv",".json",".txt",".md",".py",".toml",".yaml",".yml",".ps1",".log",".sh"}
    def inspect(name, data=None):
        name = name.replace("\\","/")
        counts["files_and_zip_members"] += 1
        if hard_names.search(name):
            findings.append({"path":name,"issue":"excluded_private_or_local_only_name"})
        if forbidden_image.search(name):
            findings.append({"path":name,"issue":"source_or_reference_image_name"})
        elif images.search(name):
            counts["prediction_mask_image_members"] += 1
            if not allowed_image.search(name):
                findings.append({"path":name,"issue":"unreviewed_image_member"})
            elif data is None:
                findings.append({"path":name,"issue":"prediction_image_pixels_not_checked"})
            else:
                try:
                    with Image.open(io.BytesIO(data)) as im:
                        if Path(name).suffix.lower()==".png":
                            valid=im.mode in {"1","L"} and set(im.get_flattened_data()) <= {0,1,255}
                        else:
                            valid=im.mode in {"I","I;16","I;16L","I;16B"} and im.getextrema()[0]>=0
                        if not valid:findings.append({"path":name,"issue":"prediction_mask_pixel_mode_or_values_invalid","mode":im.mode})
                        else:counts["prediction_masks_pixel_verified"] += 1
                except Exception as exc:
                    findings.append({"path":name,"issue":"prediction_image_decode_failed","error":str(exc)})
        if data is not None and Path(name).suffix.lower() in text_exts:
            counts["text_files_scanned"] += 1
            text = data.decode("utf-8-sig",errors="replace")
            if raw_path.search(text):
                findings.append({"path":name,"issue":"unsanitized_absolute_local_path"})
            if secret.search(text):
                findings.append({"path":name,"issue":"credential_pattern_detected"})
    for path in sorted(bundle.rglob("*")):
        if not path.is_file() or ".git" in path.parts:
            continue
        rel = path.relative_to(bundle).as_posix()
        if path.suffix.lower()==".zip":
            inspect(rel)
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    if member.is_dir():
                        continue
                    member_name = rel+"!"+member.filename
                    data = archive.read(member) if Path(member.filename).suffix.lower() in text_exts or images.search(member.filename) else None
                    inspect(member_name,data)
        else:
            data = path.read_bytes() if path.suffix.lower() in text_exts or images.search(path.name) else None
            inspect(rel,data)
    return {"status":"PASS" if not findings else "FAIL", "counts":dict(counts),"findings":findings,
            "scope":"Filename/text/ZIP review and pixel decoding of every allowed prediction image. PNGs must be single-channel binary masks; TIFFs must be nonnegative integer ID maps. Source/reference imagery is excluded. This audit does not establish third-party image licenses."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path,nargs="?",default=Path.cwd())
    parser.add_argument("--report",type=Path)
    parser.add_argument("--privacy",action="store_true",help="Also scan names/text and decode prediction-mask pixels in a staged public bundle; requires Pillow")
    args = parser.parse_args()
    evidence = args.root/"evidence" if (args.root/"evidence").is_dir() else args.root
    audit = Audit(evidence)
    for name in ("sensitivity","roots","stages","requests","routing","gates","scores"):
        audit.section(name,getattr(audit,name))
        print(name+": "+audit.sections[name]["status"],flush=True)
    audit.section("bootstrap_quantiles",lambda:{"independently_recomputed_intervals":audit.flush_ci()})
    privacy = privacy_audit(args.root) if args.privacy else {"status":"NOT_REQUESTED"}
    report = {"status":"PASS" if not audit.failures and privacy["status"]!="FAIL" else "FAIL",
              "checks":sum(audit.checks.values()),"checks_by_family":dict(audit.checks),"sections":audit.sections,
              "failure_count":len(audit.failures),"failures":audit.failures,
              "privacy":privacy,"input_file_sha256":audit.inputs,
              "numpy_version":np.__version__,"python_version":sys.version.split()[0],
              "scope":"Independent metric recomputation without algorithm-module imports or models; the optional privacy audit also decodes generated mask images.",
              "limits":["Exact-map flags are cross-file declarations here; pixel equality itself requires the archived masks and original reproduction audits.",
                         "The verifier checks table arithmetic and consistency, not correctness of external annotations or scientific population generalization.",
                         "Request selected-identity scores are not re-ranked because per-instance confidence was not exported in request_instances.csv; selected IDs and their IoUs are cross-checked.",
                         "Privacy checks establish file exclusion and generated-mask pixel types; they do not establish redistribution licenses for separately supplied images or annotations."]}
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=lambda x:sorted(x) if isinstance(x,set) else str(x)),encoding="utf-8")
    print(json.dumps({"status":report["status"],"checks":report["checks"],"failure_count":len(audit.failures),"privacy":privacy["status"],
                      "bootstrap":audit.sections["bootstrap_quantiles"],"first_failures":audit.failures[:8]},ensure_ascii=False,indent=2,default=str))
    return 0 if report["status"]=="PASS" else 1


if __name__=="__main__":
    raise SystemExit(main())
