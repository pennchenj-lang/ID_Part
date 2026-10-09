from __future__ import annotations
from portable_paths import BUNDLE_ROOT, DATA_ROOT, SOURCE_ROOT, PROJECT_ROOT, OUTPUT_DIR, resolve_data

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

ROOT = BUNDLE_ROOT
PROJECT = PROJECT_ROOT
RUNTIME = DATA_ROOT
SNAPSHOT = SOURCE_ROOT
BENCH = RUNTIME / "experiments/paper_v031/test226_hpid"
HOLDOUT = RUNTIME / "experiments/paper_v031_untouched_group_holdout_42_20260825_r5"
FRONT = RUNTIME / "experiments/paper_v031_identity_frontend_20260828"
OUT = OUTPUT_DIR
SEED = 20261008
sys.dont_write_bytecode = True
sys.path.insert(0, str(SNAPSHOT / "src"))
sys.path.insert(0, str(PROJECT / "hpid_split/scripts"))

import cv2
import numpy as np
from PIL import Image
from analyze_cross_domain_fusion_ablation import _load_candidates, _evaluate_result
from hpid_split.fusion import FusionConfig, fuse_candidates, _source_family, _source_agreement_factors

cv2.setNumThreads(1)
METRICS = ["part_f1_at_025", "part_recall_at_025", "part_f1_at_050", "part_f1_at_075", "semantic_f1_at_025", "root_foreground_iou", "mean_matched_boundary_f1_at_025", "predicted_part_count"]


def read_json(path):
    return resolve_data(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bootstrap(values, seed=SEED):
    a = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), (10000, len(a)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean": float(a.mean()), "low": float(lo), "high": float(hi)}


def wilson(k, n):
    if n == 0:
        return {"k": k, "n": n, "rate": None, "low": None, "high": None}
    z = 1.959963984540054
    p = k / n
    center = (p + z*z/(2*n)) / (1+z*z/n)
    half = z * math.sqrt(p*(1-p)/n+z*z/(4*n*n)) / (1+z*z/n)
    return {"k": k, "n": n, "rate": p, "low": max(0, center-half), "high": min(1, center+half)}


def variants():
    out = {"release": {}}
    for name, values in {
        "detail_bonus": [0, 0.04, 0.064, 0.096, 0.12],
        "orphan_support": [0, 0.06, 0.096, 0.144, 0.18],
        "hierarchy_strength": [0, 0.325, 0.52, 0.78, 0.975],
        "conflict_penalty": [0, 0.36, 0.576, 0.864, 0.95],
        "direct_gate_threshold": [0.304, 0.456],
        "direct_gate_margin": [0.656, 0.984],
    }.items():
        for v in values:
            out[f"{name}={v:g}"] = ({"specificity_host_suppression": 1-v} if name == "conflict_penalty" else {name: v})
    for bits in itertools.product([0.8, 1.2], repeat=4):
        out["joint_" + "_".join(map(str, bits))] = {
            "detail_bonus": 0.08*bits[0], "orphan_support": 0.12*bits[1],
            "hierarchy_strength": 0.65*bits[2], "specificity_host_suppression": 1-0.72*bits[3],
        }
    for flag in ["use_parent_support", "use_direct_gate", "use_specificity_ownership", "use_hierarchical_duplicate_suppression", "use_remainder_attachment", "use_consensus"]:
        out["disable_"+flag] = {flag: False}
    out["source_reliability_uniform"] = {"_reliability": "uniform"}
    out["source_reliability_0.8x"] = {"_reliability": 0.8}
    out["source_reliability_1.2x"] = {"_reliability": 1.2}
    return out


def case_worker(raw, case_path, configurations):
    package = BENCH / raw["case_id"]
    candidates = _load_candidates(package)
    stored = np.asarray(Image.open(package / "part_id_map.tiff"))
    base = fuse_candidates(candidates, config=FusionConfig())
    exact = bool(np.array_equal(base.instance_map, stored))
    if not exact:
        raise RuntimeError(f"Frozen release fails exact reproduction: {raw['case_id']}")
    case = read_json(case_path)
    baseline = _evaluate_result(result=base, case=case, case_dir=Path(case_path).parent, expected_domain=raw["expected_domain"])
    rows = []
    cache = {hashlib.sha256(base.instance_map.tobytes()).hexdigest(): baseline}
    for name, spec in configurations.items():
        spec = dict(spec)
        reliability = spec.pop("_reliability", None)
        inputs = candidates if reliability is None else [replace(c, source_reliability=1.0 if reliability == "uniform" else float(np.clip(c.source_reliability*reliability,0,1))) for c in candidates]
        pred = base if name == "release" else fuse_candidates(inputs, config=replace(FusionConfig(), **spec))
        digest = hashlib.sha256(pred.instance_map.tobytes()).hexdigest()
        # Equal instance arrays can have different labels after node suppression.
        key = digest + json.dumps([(x.part_id, x.semantic_name) for x in pred.instances])
        metrics = baseline if name == "release" else cache.get(key)
        if metrics is None:
            metrics = _evaluate_result(result=pred, case=case, case_dir=Path(case_path).parent, expected_domain=raw["expected_domain"])
            cache[key] = metrics
        rows.append({"case_id":raw["case_id"], "domain":raw["expected_domain"], "category":raw["object_category"], "variant":name, "pixel_change_fraction":float(np.mean(pred.instance_map != stored)), "exact_release_map":bool(np.array_equal(pred.instance_map,stored)), **{k:metrics[k] for k in METRICS}})
    families = []
    agreements,_ = _source_agreement_factors(list(base.accepted_candidates),FusionConfig())
    for c,a in zip(base.accepted_candidates,agreements):
        families.append({"case_id":raw["case_id"], "family":_source_family(c), "source_reliability":c.source_reliability, "score":c.score,"agreement":a,"weight":float(np.clip((.45+.55*np.clip(c.score,0,1))*c.source_reliability*a,.01,.995))})
    return rows, families


def run_sensitivity(workers, limit):
    bench = read_json(BENCH / "benchmark_summary.json")
    manifest = read_json(bench["source_manifest"])
    lookup = {r["case_id"]:r for r in manifest["cases"]}
    cases = [r for r in bench["cases"] if r["return_code"] == 0]
    if limit:
        cases=cases[:limit]
    specs=variants()
    write_json(OUT/"sensitivity_protocol.json", {"design":"Post-review fixed-candidate sensitivity, no retuning or replacement of release", "seed":SEED,"cases":len(cases),"variants":specs,"base_config":asdict(FusionConfig()),"code_sha256":sha(SNAPSHOT/"src/hpid_split/fusion.py"),"simultaneous_ci":False,"original_serial_candidate_gates_rerun":False})
    rows=[]; weights=[]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures={ex.submit(case_worker,r,lookup[r["case_id"]]["case_path"],specs):r["case_id"] for r in cases}
        for n,f in enumerate(as_completed(futures),1):
            a,b=f.result();rows.extend(a);weights.extend(b)
            print(f"sensitivity {n}/{len(cases)} {futures[f]}",flush=True)
    rows.sort(key=lambda r:(r["case_id"],r["variant"]))
    write_csv(OUT/"sensitivity_cases.csv",rows)
    write_csv(OUT/"source_reliability_candidates.csv",weights)
    base={r["case_id"]:r for r in rows if r["variant"]=="release"}
    summary=[]
    for variant in specs:
        selected=[r for r in rows if r["variant"]==variant]
        s={"variant":variant,"n":len(selected),"changed_cases":sum(not r["exact_release_map"] for r in selected)}
        for k in METRICS:
            a=bootstrap([r[k] for r in selected]);b=bootstrap([r[k]-base[r["case_id"]][k] for r in selected])
            for label,v in a.items():s[f"{k}_{label}"]=v
            for label,v in b.items():s[f"delta_{k}_{label}"]=v
        summary.append(s)
    write_csv(OUT/"sensitivity_summary.csv",summary)
    family_summary=[]
    for family in sorted({r["family"] for r in weights}):
        a=[r for r in weights if r["family"]==family]
        family_summary.append({"family":family,"n":len(a),"weight_min":min(r["weight"] for r in a),"weight_max":max(r["weight"] for r in a),"reliability_min":min(r["source_reliability"] for r in a),"reliability_max":max(r["source_reliability"] for r in a),"agreement_values":str(sorted({r["agreement"] for r in a}))})
    write_csv(OUT/"source_reliability_summary.csv",family_summary)
    print(json.dumps({"cases":len(cases),"variants":len(specs),"exact_reproductions":len(base)}),flush=True)


def routing_audit():
    folder=FRONT/"03_completion_group_frontend"
    rows=read_csv(folder/"routing_correctness_cases.csv")
    completion=read_csv(folder/"completion_frontend_cases.csv")
    info={r["case_id"]:r for r in completion}
    for r in rows:
        r["domain"]=info[r["case_id"]]["expected_domain"]
        r["category"]=info[r["case_id"]]["object_category"]
    endpoints=["unique_and_correct_025","wrong_unique_025","query_ambiguous","query_unresolved","unique_and_correct_050","wrong_unique_050","correct_in_review_set_025","correct_in_review_set_050"]
    report=[]
    for method in sorted({r["method"] for r in rows}):
        selected=[r for r in rows if r["method"]==method]
        for e in endpoints:
            sub=[r for r in selected if float(r["query_ambiguous"])==1] if e.startswith("correct_in_review") else selected
            report.append({"method":method,"endpoint":e,**wilson(sum(float(r[e])==1 for r in sub),len(sub))})
    write_csv(OUT/"routing_intervals.csv",report)
    strata=[]
    for level in ["domain","category"]:
        for group in sorted({r[level] for r in rows}):
            for method in ["raw_proposals","dbscan_fusion","hpid_split_group_ids"]:
                selected=[r for r in rows if r[level]==group and r["method"]==method]
                for e in endpoints[:4]:
                    strata.append({"level":level,"stratum":group,"method":method,"endpoint":e,**wilson(sum(float(r[e])==1 for r in selected),len(selected))})
    write_csv(OUT/"routing_stratified.csv",strata)
    methods={m:{r["case_id"]:r for r in rows if r["method"]==m} for m in {r["method"] for r in rows}}
    def state(r):
        for name in endpoints[:4]:
            if float(r[name])==1:return name
        raise ValueError(r)
    transitions=[]
    for cid,raw in methods["raw_proposals"].items():
        h=methods["hpid_split_group_ids"][cid]
        transitions.append({"case_id":cid,"domain":h["domain"],"category":h["category"],"target":h["target_part_name"],"raw_state":state(raw),"hpid_state":state(h),"raw_review_correct":raw["correct_in_review_set_025"],"hpid_review_correct":h["correct_in_review_set_025"],"raw_best_iou":raw["best_review_match_iou"],"hpid_best_iou":h["best_review_match_iou"],"raw_selected_iou":raw["selected_part_iou"],"hpid_selected_iou":h["selected_part_iou"]})
    write_csv(OUT/"routing_transitions.csv",transitions)
    failures=[r for r in transitions if r["hpid_state"]=="wrong_unique_025"]
    write_csv(OUT/"wrong_unique_cases.csv",failures)
    paired_completion={m:{r["case_id"]:r for r in completion if r["method"]==m} for m in {r["method"] for r in completion}}
    hmethod=next(m for m in paired_completion if m.startswith("hpid"))
    synth=[]
    for cid,h in paired_completion[hmethod].items():
        raw=paired_completion["raw_proposals"][cid]
        synth.append({"case_id":cid,"domain":h["expected_domain"],"delta_psnr":float(h["hidden_region_psnr"])-float(raw["hidden_region_psnr"]),"delta_request_recall":float(h["completion_request_recall"])-float(raw["completion_request_recall"]),"raw_request_recall":float(raw["completion_request_recall"]),"hpid_request_recall":float(h["completion_request_recall"])})
    write_csv(OUT/"synthesis_diagnostics.csv",synth)
    bysign=[]
    for sign in [-1,0,1]:
        sub=[r for r in synth if np.sign(r["delta_request_recall"])==sign]
        bysign.append({"recall_change_sign":sign,"n":len(sub),"mean_psnr_change":float(np.mean([r["delta_psnr"] for r in sub])) if sub else None})
    write_json(OUT/"routing_audit.json",{"n":37,"interval_method":"95% Wilson score, pointwise, no multiplicity-adjusted superiority claims","routing_source_sha256":sha(folder/"routing_correctness_cases.csv"),"wrong_unique":failures,"raw_correct_ambiguous_transitions":dict(Counter(r["hpid_state"] for r in transitions if r["raw_state"]=="query_ambiguous" and float(r["raw_review_correct"])==1)),"synthesis_by_recall_change":bysign,"mean_psnr_delta":bootstrap([r["delta_psnr"] for r in synth]),"mean_request_recall_delta":bootstrap([r["delta_request_recall"] for r in synth])})
    print('routing complete',flush=True)


def holdout_audit():
    inputs=read_json(HOLDOUT/"01_blind_input_manifest.json")
    refs={r["case_id"]:r for r in read_json(HOLDOUT/"01_sealed_reference_manifest.json")["cases"]}
    commands={r["case_id"]:r for r in read_json(HOLDOUT/"02_blind_inference_manifest.json")["cases"]}
    rows=[]
    for i,raw in enumerate(inputs["cases"],1):
        cid=raw["case_id"];p=HOLDOUT/"02_blind_inference"/cid
        c=_load_candidates(p)
        roots=[x for x in c if x.semantic_name==x.semantic_parent]
        if len(roots)!=1:raise RuntimeError((cid,len(roots)))
        root=roots[0]
        auto=fuse_candidates(c)
        refpath=Path(refs[cid]["case_path"])
        truth=np.asarray(Image.open(refpath.parent/"object_mask_crop.png").convert("L"))>=128
        oracle=fuse_candidates([replace(x,mask=truth) if x is root else x for x in c])
        case=read_json(refpath)
        common={"case_id":cid,"domain":raw["expected_domain"],"category":raw["object_category"],"automatic_root_iou":float(np.sum(root.mask&truth)/max(1,np.sum(root.mask|truth))),"root_equals_reference":bool(np.array_equal(root.mask,truth)),"root_source":root.source,"original_map_reproduced":bool(np.array_equal(auto.instance_map,np.asarray(Image.open(p/"part_id_map.tiff")))),"reference_passed_to_original_command":"01_sealed_references" in " ".join(commands[cid]["command"]),"category_prompt_passed":"--asset-prompt" in commands[cid]["command"]}
        for name,pred in [("automatic_root",auto),("reference_root",oracle)]:
            metrics=_evaluate_result(result=pred,case=case,case_dir=refpath.parent,expected_domain=raw["expected_domain"])
            rows.append({**common,"condition":name,**{k:metrics[k] for k in METRICS}})
        print(f'holdout root {i}/42 {cid}',flush=True)
    write_csv(OUT/"holdout_root_cases.csv",rows)
    summary={}
    for condition in ["automatic_root","reference_root"]:
        sub=[r for r in rows if r["condition"]==condition]
        summary[condition]={k:bootstrap([r[k] for r in sub]) for k in METRICS+["automatic_root_iou"]}
    a={r["case_id"]:r for r in rows if r["condition"]=="automatic_root"}
    b={r["case_id"]:r for r in rows if r["condition"]=="reference_root"}
    summary["reference_minus_automatic"]={k:bootstrap([b[c][k]-a[c][k] for c in a]) for k in METRICS}
    summary["audit"]={"n":len(a),"root_identical_to_reference":sum(r["root_equals_reference"] for r in a.values()),"exact_fine_map_reproductions":sum(r["original_map_reproduced"] for r in a.values()),"reference_in_commands":sum(r["reference_passed_to_original_command"] for r in a.values()),"category_prompts":sum(r["category_prompt_passed"] for r in a.values()),"scope":"Post-review root replacement diagnostic; fixed candidates from automatic inference on ground-truth bbox crops. Reports fine Part IDs, not re-generated Group-ID metrics. Original holdout was single-pass; this audit is subsequent analysis."}
    write_json(OUT/"holdout_root_summary.json",summary)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["routing","sensitivity","holdout"])
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--limit",type=int,default=0)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.mode=="routing":routing_audit()
    elif args.mode=="holdout":holdout_audit()
    else:run_sensitivity(args.workers,args.limit)
