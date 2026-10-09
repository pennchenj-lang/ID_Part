from __future__ import annotations
from portable_paths import BUNDLE_ROOT, DATA_ROOT, SOURCE_ROOT, PROJECT_ROOT, OUTPUT_DIR, resolve_data
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import importlib.util
import itertools
import json
import os
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

from analyze_revision import (
    ROOT, OUT, BENCH, HOLDOUT, SNAPSHOT, SEED, read_json, write_json,
    write_csv, bootstrap, _load_candidates, _evaluate_result,
)
import numpy as np
from PIL import Image
from hpid_split.export import load_previous_package
import hpid_split.physical_groups as pg
from analyze_paper_results import _match_metrics

ORIGINAL_GATE=pg._candidate_three_stage_verification
CURRENT_GROUP_MODULE=pg
HISTORICAL_SOURCE=Path(os.environ.get('HPID_HISTORICAL_GROUP_SOURCE',str(ROOT/'evidence/gate_ablation/physical_groups_da7d236.py')))
_historical_spec=importlib.util.spec_from_file_location('hpid_split.physical_groups_aug20',HISTORICAL_SOURCE)
HISTORICAL_GROUP_MODULE=importlib.util.module_from_spec(_historical_spec)
sys.modules[_historical_spec.name]=HISTORICAL_GROUP_MODULE
_historical_spec.loader.exec_module(HISTORICAL_GROUP_MODULE)
SOURCE_COMMIT={'test226':'da7d236b7873200692c262d41b5f69196d9dd5fb','holdout42':'be543003632fa739d0b63e9acbffb8d91d99b84a'}
STAGES=['stage_1_semantic','stage_2_structure','stage_3_appearance']
VARIANTS={'STA':(1,1,1),'none':(0,0,0),'S':(1,0,0),'ST':(1,1,0),
          'T':(0,1,0),'A':(0,0,1),'SA':(1,0,1),'TA':(0,1,1)}
GROUP_METRICS=['part_f1_at_025','part_f1_at_050','semantic_f1_at_025',
               'predicted_part_count','mean_matched_boundary_f1_at_025']

def input_fingerprint(candidates,instance_map,records):
    digest=hashlib.sha256(instance_map.tobytes())
    digest.update(json.dumps([r.to_dict() for r in records],sort_keys=True).encode())
    for c in candidates:
        digest.update(c.mask.tobytes())
        digest.update(json.dumps({'name':c.semantic_name,'parent':c.semantic_parent,
                                  'score':c.score,'reliability':c.source_reliability,
                                  'source':c.source,'prompt':c.prompt,'metadata':c.metadata},
                                 sort_keys=True,ensure_ascii=False).encode())
    return digest.hexdigest()

def partition_signature(result):
    return sorted((g.semantic_name,hashlib.sha256((result.group_map==g.group_index).tobytes()).hexdigest()) for g in result.groups)

def switches(original,enabled):
    row=deepcopy(original)
    raw=[bool(row[STAGES[0]]['verified']),
         bool(row[STAGES[1]]['raw_verified']),
         bool(row[STAGES[2]]['raw_verified'])]
    preceding=True
    for k,active,predicate in zip(STAGES,enabled,raw):
        row[k]['release_reason']=row[k]['reason']
        row[k]['evaluated']=preceding
        row[k]['verified']=bool(preceding and (predicate or not active))
        row[k]['bypassed_for_diagnostic']=not bool(active)
        row[k]['reason']=('not_evaluated_previous_stage_failed' if not preceding else
                          'bypassed_for_diagnostic' if not active else
                          'frozen_raw_predicate_pass' if predicate else 'frozen_raw_predicate_fail')
        preceding=row[k]['verified']
    row['accepted']=preceding
    return row

def worker(item):
    global pg,ORIGINAL_GATE
    pg=HISTORICAL_GROUP_MODULE if item['split']=='test226' else CURRENT_GROUP_MODULE
    ORIGINAL_GATE=pg._candidate_three_stage_verification
    package=Path(item['package']);casepath=Path(item['case_path'])
    inst,records=load_previous_package(package)
    cand=tuple(_load_candidates(package))
    image=Image.open(package/'source.png').convert('RGB')
    root=inst>0
    profile=pg._selected_profile(cand)
    fixed_input_digest=input_fingerprint(cand,inst,records)
    calls=[]
    def frozen_gate(candidate,root=None):
        return ORIGINAL_GATE(candidate) if item['split']=='test226' else ORIGINAL_GATE(candidate,root)
    def logged_gate(candidate,root=None,enabled=(1,1,1),variant='STA'):
        raw=frozen_gate(candidate,root)
        modified=switches(raw,enabled)
        calls.append({'split':item['split'],'case_id':item['case_id'],'variant':variant,
                      'candidate_key':candidate.metadata.get('candidate_key'),
                      'semantic_name':candidate.semantic_name,
                      'root_context':'fine_foreground' if root is not None else 'metadata_only',
                      'semantic_raw_pass':raw[STAGES[0]]['verified'],
                      'structure_raw_pass':raw[STAGES[1]]['raw_verified'],
                      'appearance_raw_pass':raw[STAGES[2]]['raw_verified'],
                      'release_accepted':raw['accepted'],'intervention_accepted':modified['accepted']})
        return raw if variant=='STA' else modified
    pg._candidate_three_stage_verification=logged_gate
    try:original=pg.build_physical_groups(inst,records,candidates=cand,image=image)
    finally:pg._candidate_three_stage_verification=ORIGINAL_GATE
    assert np.array_equal(original.group_map,np.asarray(Image.open(package/'group_id_map.tiff'))),item['case_id']
    archived_groups=read_json(package/'groups.json')
    assert [(g.group_index,g.group_id,g.semantic_name) for g in original.groups]==[(g['group_index'],g['group_id'],g['semantic_name']) for g in archived_groups],item['case_id']
    _,_,pool=(pg._profile_candidate_verification(cand,profile) if item['split']=='test226' else pg._profile_candidate_verification(cand,profile,root))
    bykey={c.metadata.get('candidate_key'):c for c in cand}
    assert len(bykey)==len(cand) and None not in bykey,(item['case_id'],'nonunique candidate keys')
    assert all(r['candidate_key'] in bykey for r in pool)
    case=read_json(casepath)
    truth=[np.asarray(Image.open(casepath.parent/r['mask_crop']).convert('L'))>0 for r in case['parts']]
    audits=[]
    complete=[]
    nominated={r['candidate_key'] for r in pool}
    for c in cand:
        complete.append({'split':item['split'],'case_id':item['case_id'],
                         'candidate_key':c.metadata['candidate_key'],'semantic_name':c.semantic_name,
                         'semantic_parent':c.semantic_parent,'source':c.source,'score':c.score,
                         'source_reliability':c.source_reliability,
                         'mask_sha256':hashlib.sha256(c.mask.tobytes()).hexdigest(),
                         'metadata_sha256':hashlib.sha256(json.dumps(c.metadata,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
                         'profile_gate_nomination':c.metadata['candidate_key'] in nominated})
    for row in pool:
        c=bykey[row['candidate_key']]
        replay=frozen_gate(c,root)
        assert replay=={k:v for k,v in row.items() if k not in ['candidate_key','semantic_name','source']}
        audits.append({'split':item['split'],'case_id':item['case_id'],'candidate_key':row['candidate_key'],
                       'semantic_name':c.semantic_name,'mask_sha256':hashlib.sha256(c.mask.tobytes()).hexdigest(),
                       'semantic_pass':row[STAGES[0]]['verified'],
                       'structure_raw_pass':row[STAGES[1]]['raw_verified'],
                       'appearance_raw_pass':row[STAGES[2]]['raw_verified'],
                       'accepted':row['accepted'],
                       'semantic_reason':row[STAGES[0]]['reason'],
                       'structure_reason':row[STAGES[1]]['reason'],
                       'appearance_reason':row[STAGES[2]]['reason']})
    resultrows=[]
    for variant,enabled in VARIANTS.items():
        def modified_gate(candidate,root=None):
            return logged_gate(candidate,root,enabled,variant)
        pg._candidate_three_stage_verification=modified_gate
        try:
            result=original if variant=='STA' else pg.build_physical_groups(inst,records,candidates=cand,image=image)
        finally:pg._candidate_three_stage_verification=ORIGINAL_GATE
        assert input_fingerprint(cand,inst,records)==fixed_input_digest,(item['case_id'],variant,'input mutation')
        # The adapter exposes group masks to the unchanged evaluation protocol.
        adapted=SimpleNamespace(instance_map=result.group_map,instances=[
            SimpleNamespace(instance_index=g.group_index,to_dict=g.to_dict) for g in result.groups])
        metric=_evaluate_result(result=adapted,case=case,case_dir=casepath.parent,expected_domain=item['domain'])
        retained=[bykey[r['candidate_key']].mask for r in pool if switches(r,enabled)['accepted']]
        candidate_metric=_match_metrics(truth,retained,threshold=.25,boundary_tolerance=3)
        resultrows.append({'split':item['split'],'case_id':item['case_id'],'domain':item['domain'],
                           'variant':variant,'pre_group_gate_candidates':len(pool),
                           'complete_exported_candidates':len(cand),'selected_profile':profile,
                           'category':case['object_category'],
                           'fixed_input_sha256':fixed_input_digest,'fixed_inputs_unchanged':True,
                           'retained_candidates':len(retained),
                           'group_map_changed':not np.array_equal(result.group_map,original.group_map),
                           'semantic_partition_changed':partition_signature(result)!=partition_signature(original),
                           'group_map_sha256':hashlib.sha256(result.group_map.tobytes()).hexdigest(),
                           'release_group_map_reproduced':True,
                           **{'group_'+k:metric[k] for k in GROUP_METRICS},
                           **{'candidate_'+k:v for k,v in candidate_metric.items()}})
    provenance={'split':item['split'],'case_id':item['case_id'],'category':case['object_category'],
                'profile':profile,'package':str(package),'case_path':str(casepath),
                'complete_exported_candidates':len(cand),'profile_nominations':len(pool),
                'exact_group_map':True,'exact_group_id_semantic_sequence':True,
                'group_source_commit':SOURCE_COMMIT[item['split']]}
    provenance['fixed_input_sha256']=fixed_input_digest
    for name in ['candidates.json','part_id_map.tiff','parts.json','group_id_map.tiff','groups.json','source.png']:
        if (package/name).exists():provenance[name+'_sha256']=hashlib.sha256((package/name).read_bytes()).hexdigest()
    provenance['case_json_sha256']=hashlib.sha256(casepath.read_bytes()).hexdigest()
    return resultrows,audits,complete,calls,provenance

def inputs():
    bench=read_json(BENCH/'benchmark_summary.json')
    manifest=read_json(bench['source_manifest'])
    cases={r['case_id']:r for r in manifest['cases']}
    rows=[{'split':'test226','case_id':r['case_id'],'domain':r['expected_domain'],
           'package':str(BENCH/r['case_id']),'case_path':cases[r['case_id']]['case_path']}
          for r in bench['cases'] if r['return_code']==0]
    refs={r['case_id']:r for r in read_json(HOLDOUT/'01_sealed_reference_manifest.json')['cases']}
    rows += [{'split':'holdout42','case_id':r['case_id'],'domain':r['expected_domain'],
              'package':str(HOLDOUT/'02_blind_inference'/r['case_id']),
              'case_path':refs[r['case_id']]['case_path']}
             for r in read_json(HOLDOUT/'01_blind_input_manifest.json')['cases']]
    return rows

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=2);parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--output-dir',type=Path,default=OUT/'gate_ablation')
    args=parser.parse_args();items=inputs()
    # Fix worker Python ordering. Targeted repeats still show one-pixel
    # counterfactual variation of unassigned cause; see the saved repeat audit.
    os.environ['PYTHONHASHSEED']=str(SEED)
    if args.limit:
        # Keep trial output separate and cover both archived splits.
        items=[x for split in ('test226','holdout42') for x in [y for y in items if y['split']==split][:args.limit]]
        if args.output_dir==OUT/'gate_ablation':args.output_dir=OUT/'gate_ablation_trial'
    out=args.output_dir
    write_json(out/'group_gate_protocol.json',{
        'scope':'Post-review group-stage gate intervention on the complete archived pre-group-gate pool. The earlier fine-ID fusion and proposal generation are held fixed.',
        'not_scope':'Not an upstream proposal-generation ablation, not an A0-A3 fine-ID ablation, and not a new independent test set.',
        'variants':VARIANTS,'n_cases':len(items),'source_sha256':hashlib.sha256((SNAPSHOT/'src/hpid_split/physical_groups.py').read_bytes()).hexdigest(),
        'historical_test226_source_sha256':hashlib.sha256(HISTORICAL_SOURCE.read_bytes()).hexdigest(),
        'source_commits':SOURCE_COMMIT,
        'historical_source_url':'https://github.com/pennchenj-lang/ID_Part/blob/da7d236b7873200692c262d41b5f69196d9dd5fb/src/hpid_split/physical_groups.py',
        'historical_github_blob_sha':'3de7aa6d829c5380e612d35d2671724f5862ed06',
        'baseline_group_equality':'Every baseline must match the archived pixel map and ordered (Group index, ID, semantic name) records exactly.',
        'input_integrity':'SHA256 over fine map, full Part records, ordered masks, scores, source reliability and full metadata is asserted unchanged after each subset.',
        'version_audit':'Using the later holdout Group implementation on the earlier test226 archive reproduces only178/226 maps. The recovered Aug20 Group implementation reproduces226/226; the Aug23 implementation reproduces42/42. Each split therefore uses its own archived source consistently across all eight gate subsets; splits are never pooled.',
        'seed':SEED,'bootstrap_draws':10000,'reference_used_in_inference':False,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'python_version':sys.version,'platform':platform.platform(),'numpy_version':np.__version__,
        'python_hash_seed':SEED,
        'group_change_definition':'group_map_changed compares numeric arrays; semantic_partition_changed compares sorted (semantic name, binary mask digest) and is invariant to Group index permutations.',
        'paired_unit':'case within split; all eight interventions use each same case; no new independent test cases',
        'ci':'Pointwise percentile paired-case bootstrap; exploratory, not simultaneous; no multiplicity-adjusted superiority claim',
        'empty_nomination_policy':'All cases retained in means and denominators. Zero profile nominations yield zero retained-pool precision/recall/F1.',
        'intervention_boundary':'Only _candidate_three_stage_verification is switched wherever called during Group construction. Profile-specific structural paths, nomination rules, containment/area constraints and final Group cleanup remain unchanged.',
        'candidate_pool_control':'All exported candidates entering build_physical_groups are retained before each switch. Profile, root, fine IDs, masks, scores and metadata remain identical.',
        'switch':'Only the three named predicates in _candidate_three_stage_verification are enabled/bypassed. Raw downstream predicates are re-evaluated even when an earlier predicate is bypassed.'})
    rows=[];audits=[];complete=[];calls=[];provenance=[]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures={executor.submit(worker,x):x for x in items}
        for n,f in enumerate(as_completed(futures),1):
            a,b,c,d,e=f.result();rows+=a;audits+=b;complete+=c;calls+=d;provenance.append(e)
            print(f"gate {n}/{len(items)} {futures[f]['split']} {futures[f]['case_id']}",flush=True)
    rows.sort(key=lambda r:(r['split'],r['case_id'],r['variant']))
    audits.sort(key=lambda r:(r['split'],r['case_id'],r['candidate_key']))
    write_csv(out/'group_gate_cases.csv',rows);write_csv(out/'group_gate_pool.csv',audits)
    write_csv(out/'group_gate_complete_exported_pool.csv',sorted(complete,key=lambda r:(r['split'],r['case_id'],r['candidate_key'])))
    write_csv(out/'group_gate_actual_calls.csv',sorted(calls,key=lambda r:(r['split'],r['case_id'],r['variant'],r['candidate_key'],r['root_context'])))
    write_json(out/'group_gate_input_provenance.json',sorted(provenance,key=lambda r:(r['split'],r['case_id'])))
    summary=[]
    for split in sorted({r['split'] for r in rows}):
        base={r['case_id']:r for r in rows if r['split']==split and r['variant']=='STA'}
        for variant in VARIANTS:
            selected=[r for r in rows if r['split']==split and r['variant']==variant]
            summaryrow={'split':split,'variant':variant,'n':len(selected),
                        'pre_gate_pool_total':sum(r['pre_group_gate_candidates'] for r in selected),
                        'complete_exported_pool_total':sum(r['complete_exported_candidates'] for r in selected),
                        'cases_with_profile_nominations':sum(r['pre_group_gate_candidates']>0 for r in selected),
                        'cases_without_profile_nominations':sum(r['pre_group_gate_candidates']==0 for r in selected),
                        'retained_total':sum(r['retained_candidates'] for r in selected),
                        'changed_group_maps':sum(r['group_map_changed'] for r in selected)}
            summaryrow['changed_semantic_partitions']=sum(r['semantic_partition_changed'] for r in selected)
            metrics=['group_'+k for k in GROUP_METRICS]+['candidate_precision','candidate_recall','candidate_f1']
            for metric in metrics:
                for key,value in bootstrap([r[metric] for r in selected]).items():summaryrow[metric+'_'+key]=value
                for key,value in bootstrap([r[metric]-base[r['case_id']][metric] for r in selected]).items():summaryrow['delta_'+metric+'_'+key]=value
            summary.append(summaryrow)
    write_csv(out/'group_gate_summary.csv',summary)
    print('Complete: '+str(Counter(r['split'] for r in rows)),flush=True)

if __name__=='__main__':main()
