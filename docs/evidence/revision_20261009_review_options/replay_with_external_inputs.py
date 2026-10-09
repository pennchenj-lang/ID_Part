"""Fixed post-hoc, full-cohort review-option audit; never retunes predictors.

Run full replay with the frozen runtime; --recompute-only regenerates the
statistics from the saved per-case source table with NumPy alone.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import numpy as np

HERE = Path(os.environ.get("HPID_REPLAY_OUTPUT", "replayed_review_outputs")).resolve()
HERE.mkdir(parents=True, exist_ok=True)
PACKAGE = Path(__file__).resolve().parent
RUNTIME = Path(os.environ.get("HPID_RUNTIME_ROOT", "external_inputs/runtime"))
PROJECT = Path(os.environ.get("HPID_PROJECT_ROOT", "external_inputs/project"))
HOLDOUT = RUNTIME / "experiments/paper_v031_untouched_group_holdout_42_20260825_r5"
FRONT = RUNTIME / "experiments/paper_v031_identity_frontend_20260828"
MANIFEST = FRONT / "02_completion_frontend/completion_target_manifest.json"
FOLDER = FRONT / "03_completion_group_frontend"
SNAPSHOT = RUNTIME / "code_snapshots/hpid_split_be54300_holdout"
METHODS = ("raw_proposals", "greedy_nms", "dbscan_fusion", "local_pairwise_crf", "hpid_split_group_ids")
HPID = METHODS[-1]
SEED = 20261009
N_BOOT = 10000
METRICS = ["review_option_count", "query_unique", "query_ambiguous", "query_unresolved"] + [
    f"{metric}_{threshold}" for threshold in ("025", "050") for metric in
    ("unique_and_correct", "wrong_unique", "semantic_correct_any_match",
     "geometry_correct_any_exposed", "geometry_correct_any_exported")]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def replay():
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(SNAPSHOT / "src"))
    sys.path.insert(0, str(PROJECT / "hpid_split"))
    import hpid_split
    hpid_split.__path__.append(str(PROJECT / "hpid_split/src/hpid_split"))
    from scripts.run_semantic_completion_frontend_benchmark import (
        _predictions, _normalize, _normalized_name, _mask_iou,
    )
    from PIL import Image
    import cv2
    cv2.setNumThreads(1)
    manifest = read_json(MANIFEST)
    targets = sorted(manifest["selected_cases"], key=lambda r:r["case_id"])
    assert len(targets) == 37 and len({r["case_id"] for r in targets}) == 37
    original = {(r["case_id"],r["method"]):r for r in read_csv(FOLDER/"routing_correctness_cases.csv")}
    original_front = {(r["case_id"],r["method"]):r for r in read_csv(FOLDER/"completion_frontend_cases.csv")}
    rows, instances, units, assertions, hashes = [], [], [], [], []
    for i,target in enumerate(targets):
        cid=target["case_id"]
        ref=HOLDOUT/"01_sealed_references"
        c=read_json(ref/target["case_relative_path"])
        truth_path=ref/target["target_mask_relative_path"]
        assert sha(truth_path)==target["target_mask_sha256"]
        truth=np.asarray(Image.open(truth_path).convert("L"))>=128
        package=HOLDOUT/"02_blind_inference"/cid
        units.append({"case_id":cid,"image_id":c["image_id"],"source_image_sha256":c["source_image_sha256"],
                      "object_annotation_id":c["object_annotation_id"],"object_category":target["object_category"],
                      "source_crop_sha256":sha(package/"source.png")})
        paths=[package/f for f in ("source.png","candidates.json","group_id_map.tiff","groups.json")]
        paths += [ref/target["case_relative_path"],truth_path]
        paths += [package/r["mask_path"] for r in read_json(package/"candidates.json")]
        for path in paths:
            hashes.append({"case_id":cid,"source_role":str(path.relative_to(HOLDOUT)),"sha256":sha(path)})
        predictions=_predictions(package)
        normalized_target=_normalize(target["target_part_name"],target["expected_domain"],object_category=target["object_category"])
        for method in METHODS:
            pred=predictions[method]
            match=[]
            scored=[]
            for ins in pred.instances:
                normal=_normalized_name(ins,expected_domain=target["expected_domain"],object_category=target["object_category"])
                score=_mask_iou(ins.mask,truth)
                scored.append((ins,normal,score))
                if normal==normalized_target:
                    match.append((ins,normal,score))
            exposed=match if match else scored
            selected=max(match,key=lambda r:(float(r[0].confidence),int(np.count_nonzero(r[0].mask)),r[0].identity)) if match else None
            best_match=max((r[2] for r in match),default=0.)
            best_exposed=max((r[2] for r in exposed),default=0.)
            best_exported=max((r[2] for r in scored),default=0.)
            unique=len(match)==1
            selected_iou=selected[2] if selected else 0.
            row={"case_id":cid,"method":method,"image_id":c["image_id"],"target":target["target_part_name"],
                 "normalized_target":normalized_target,"query_match_count":len(match),"all_exported_count":len(scored),
                 "review_option_count":len(exposed),"query_unique":int(unique),"query_ambiguous":int(len(match)>1),
                 "query_unresolved":int(not match),"query_state":"unique" if unique else "ambiguous" if match else "unresolved",
                 "selected_identity":selected[0].identity if selected else "","selected_part_iou":selected_iou,
                 "best_review_match_iou":best_match,"best_exposed_iou":best_exposed,"best_exported_iou":best_exported}
            for suffix,threshold in (("025",.25),("050",.50)):
                row.update({f"unique_and_correct_{suffix}":int(unique and selected_iou>=threshold),
                            f"wrong_unique_{suffix}":int(unique and selected_iou<threshold),
                            f"semantic_correct_any_match_{suffix}":int(best_match>=threshold),
                            f"geometry_correct_any_exposed_{suffix}":int(best_exposed>=threshold),
                            f"geometry_correct_any_exported_{suffix}":int(best_exported>=threshold),
                            f"correct_in_review_set_{suffix}":int(len(match)>1 and best_match>=threshold)})
            old=original[cid,method]
            for key in old:
                if key in ("case_id","method","target_part_name"):
                    continue
                assert key in row, key
                assert abs(float(row[key])-float(old[key]))<1e-12,(cid,method,key,row[key],old[key])
            old_front=original_front[cid,method]
            assert row["selected_identity"]==old_front["selected_identity"],(cid,method,"selected_identity")
            assert row["review_option_count"]==int(old_front["review_option_count"])
            assert abs(selected_iou-float(old_front["selected_part_iou"]))<1e-12
            assertions.append({"case_id":cid,"method":method,"all_old_routing_metrics_exact_tolerance":1e-12,
                               "selected_identity_exact":True,"review_option_rule_exact":True})
            for ins,normal,score in scored:
                instances.append({"case_id":cid,"method":method,"identity":ins.identity,"semantic":ins.semantic_name,
                    "normalized_semantic":normal,"matches_target":int(normal==normalized_target),
                    "actual_exposed":int(not match or normal==normalized_target),"confidence":float(ins.confidence),
                    "area_px":int(np.count_nonzero(ins.mask)),"target_iou":score,
                    "mask_sha256":hashlib.sha256(np.ascontiguousarray(ins.mask,dtype=np.uint8).tobytes()).hexdigest()})
            rows.append(row)
        print(f"[{i+1}/37] {cid} reproduced all five methods",flush=True)
    assert len({r["image_id"] for r in units})==37
    assert len({r["source_image_sha256"] for r in units})==37
    assert len({r["source_crop_sha256"] for r in units})==37
    write_csv(HERE/"review_cases.csv",rows)
    write_csv(HERE/"review_instances.csv",instances)
    write_csv(HERE/"source_units.csv",units)
    write_csv(HERE/"reproduction_assertions.csv",assertions)
    write_csv(HERE/"source_hashes.csv",hashes)
    codepaths=[Path(__file__),PROJECT/"hpid_split/scripts/run_semantic_completion_frontend_benchmark.py",
               PROJECT/"hpid_split/scripts/audit_completion_routing.py",PROJECT/"hpid_split/src/hpid_split/postprocess_baselines.py",
               SNAPSHOT/"src/hpid_split/paco_eval.py",SNAPSHOT/"src/hpid_split/paco_semantics.py"]
    write_json(HERE/"replay_provenance.json",{"cases":37,"methods":5,"reproduced_case_methods":len(assertions),
              "unique_image_ids":37,"unique_source_image_sha256":37,"unique_source_crop_sha256":37,
              "target_manifest_sha256":sha(MANIFEST),"original_routing_csv_sha256":sha(FOLDER/"routing_correctness_cases.csv"),
              "protocol_sha256":sha(PACKAGE/"protocol.json"),
              "code_sha256":{p.name:sha(p) for p in codepaths},"python":sys.version,"numpy":np.__version__})

def interval(values,adjust=False):
    x=np.asarray(values,dtype=float)
    if not len(x):
        return {"n":0,"mean":None,"ci95_low":None,"ci95_high":None}
    indices=np.random.default_rng(SEED).integers(0,len(x),size=(N_BOOT,len(x)))
    b=x[indices].mean(axis=1)
    out={"n":len(x),"mean":float(x.mean()),"ci95_low":float(np.quantile(b,.025)),"ci95_high":float(np.quantile(b,.975))}
    if adjust:
        out.update({"ci98_75_low":float(np.quantile(b,.00625)),"ci98_75_high":float(np.quantile(b,.99375))})
    return out

def summarize():
    rows=read_csv(HERE/"review_cases.csv")
    indexed={(r["case_id"],r["method"]):r for r in rows}
    case_ids=sorted({r["case_id"] for r in rows})
    assert len(case_ids)==37 and len(rows)==185 and len(indexed)==185
    sums=[]
    contrasts=[]
    aux=[]
    transitions=[]
    for method in METHODS:
        for metric in METRICS:
            values=[float(indexed[cid,method][metric]) for cid in case_ids]
            sums.append({"method":method,"metric":metric,"sum":sum(values),**interval(values)})
    for method in METHODS[:-1]:
        for metric in METRICS:
            x=[float(indexed[cid,HPID][metric]) for cid in case_ids]
            y=[float(indexed[cid,method][metric]) for cid in case_ids]
            diffs=np.asarray(x)-np.asarray(y)
            contrast={"comparator":method,"metric":metric,"hpid_sum":sum(x),"comparator_sum":sum(y),
                "hpid_mean":sum(x)/37,"comparator_mean":sum(y)/37,"hpid_only_or_higher":sum(diffs>0),
                "comparator_only_or_higher":sum(diffs<0),"equal":sum(diffs==0),
                "relative_reduction":(sum(y)-sum(x))/sum(y) if metric=="review_option_count" and sum(y)>0 else "",
                **interval(diffs,adjust=metric=="review_option_count")}
            if metric!="review_option_count":
                positive=int(sum(diffs>0))
                negative=int(sum(diffs<0))
                discordant=positive+negative
                exact=min(1.,2*sum(math.comb(discordant,k) for k in range(min(positive,negative)+1))/2**discordant) if discordant else 1.
                contrast.update({"mcnemar_discordant_n":discordant,"mcnemar_exact_two_sided_p":exact})
            contrasts.append(contrast)
            if metric!="review_option_count":
                for cid,a,b,d in zip(case_ids,x,y,diffs):
                    if d:
                        transitions.append({"comparator":method,"metric":metric,"case_id":cid,"hpid":a,"comparator_value":b,"difference":d})
        predicates={"same_query_state":lambda a,b:a["query_state"]==b["query_state"]}
        for threshold in ("025","050"):
            for metric in ("semantic_correct_any_match","geometry_correct_any_exposed"):
                key=f"{metric}_{threshold}"
                predicates[f"both_{key}"]=lambda a,b,k=key:float(a[k])==1 and float(b[k])==1
        for name,predicate in predicates.items():
            cids=[cid for cid in case_ids if predicate(indexed[cid,HPID],indexed[cid,method])]
            x=[float(indexed[cid,HPID]["review_option_count"]) for cid in cids]
            y=[float(indexed[cid,method]["review_option_count"]) for cid in cids]
            aux.append({"comparator":method,"subset":name,"case_ids":";".join(cids),
                        "hpid_mean":float(np.mean(x)) if x else None,"comparator_mean":float(np.mean(y)) if y else None,
                        **interval(np.asarray(x)-np.asarray(y))})
    write_csv(HERE/"method_summary.csv",sums)
    write_csv(HERE/"paired_contrasts.csv",contrasts)
    write_csv(HERE/"auxiliary_subsets.csv",aux)
    write_csv(HERE/"discordant_cases.csv",transitions)
    table_methods=[]
    for method in METHODS:
        selected=[indexed[cid,method] for cid in case_ids]
        table_methods.append({"method":method,"n":37,
            "mean_review_options":sum(float(r['review_option_count']) for r in selected)/37,
            "total_review_options":sum(int(r['review_option_count']) for r in selected),
            **{metric:int(sum(float(r[metric]) for r in selected)) for metric in METRICS if metric!='review_option_count'}})
    table_report={"table_A_methods":table_methods,
        "table_B_review_option_contrasts":[r for r in contrasts if r['metric']=='review_option_count'],
        "table_C_binary_guardrail_contrasts":[r for r in contrasts if r['metric']!='review_option_count'],
        "notes":["All method counts use all 37 original requests, one request per distinct source image.",
            "Table B: HPID minus comparator, 10000 paired case bootstrap resamples, seed 20261009; 98.75% individual intervals account for four option comparisons by Bonferroni adjustment.",
            "Table C: pointwise descriptive 95% bootstrap intervals and exact two-sided McNemar tests are unadjusted; only option-count contrasts define the primary comparison family.",
            "Exact McNemar p values were added as a guardrail check after the fixed audit; binary improvements are described as observed counts, not confirmatory significance claims.",
            "Sparse paired binary outcomes can yield bootstrap intervals excluding zero while exact McNemar p values exceed .05; interpret these endpoints descriptively.",
            "Every exposed set follows the unchanged rule: normalized semantic matches if any, otherwise all exported instances.",
            "Counts measure interface options, not human time. Equal totals do not demonstrate equivalence. Reused cases do not constitute independent confirmation."]}
    write_json(HERE/'manuscript_tables.json',json.loads(json.dumps(table_report,default=lambda x:x.item())))
    report={"analysis":"Explicit post-hoc full-cohort audit; no algorithm or target changes",
            "unit":"37 requests from 37 distinct source image IDs and source image SHA256s, confirmed by replay",
            "bootstrap":{"iterations":N_BOOT,"seed":SEED,"ordering":"case_id ascending",
                "pairing":"one common resample index matrix for each full-cohort endpoint/contrast",
                "pointwise_coverage":.95,"option_comparison_individual_coverage":.9875,
                "option_comparison_family_size":4,"quantile_method":"NumPy linear"},
            "binary_guardrail_test":{"name":"Exact two-sided McNemar","implementation":"twice smaller Binomial(n_discordant, .5) tail, capped at 1",
                "multiplicity":"unadjusted descriptive guardrails","timing":"added after fixed audit as sparse-outcome check; no change to option-count comparison family"},
            "option_contrasts":[r for r in contrasts if r["metric"]=="review_option_count"],
            "metric_definitions":{
                "review_option_count":"Semantic matches when any exist; otherwise all exported instances.",
                "semantic_correct_any_match":"At least one normalized semantic match reaches target IoU threshold.",
                "geometry_correct_any_exposed":"At least one option actually exposed by the rule reaches target IoU threshold, including fallback options with mismatched labels.",
                "geometry_correct_any_exported":"At least one of all exported instances reaches target IoU threshold regardless of query exposure or label.",
                "unique_and_correct":"Exactly one semantic match that reaches target IoU threshold.",
                "wrong_unique":"Exactly one semantic match below target IoU threshold."},
            "limitations":["No new independent cases; results were known before this post-hoc uncertainty audit.",
                "Option counts are an objective interface proxy, not measured human time or workload.",
                "Equal observed correctness totals do not establish equivalence or noninferiority.",
                "Auxiliary subsets are GT-conditioned post-hoc mechanism diagnostics, not full-cohort or causal evidence.",
                "Only option count contrasts have four-comparison Bonferroni intervals; all guardrail intervals are descriptive pointwise 95%."]}
    # Convert numpy counts generated by comparisons to native JSON values.
    write_json(HERE/"review_burden_report.json",json.loads(json.dumps(report,default=lambda x:x.item())))
    print(json.dumps(report["option_contrasts"],indent=2,default=lambda x:x.item()),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--recompute-only",action="store_true")
    a=p.parse_args()
    if not a.recompute_only:
        replay()
    summarize()
