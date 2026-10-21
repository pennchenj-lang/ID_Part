import csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent
with (P/'target_retention.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
with (P/'case_metrics.csv').open(encoding='utf-8-sig',newline='') as f:metrics=list(csv.DictReader(f))
METHODS=['hpid_split_a3','dbscan_fusion','greedy_nms']
SEED=20261009
def mean(x):return float(np.mean(x)) if len(x) else None
def ci(a,q=(.025,.975)):return [float(x) for x in np.quantile(a,q)]
def b(x):return x=='True'
def selected(condition,scope):
    return condition!='automatic_root' if scope=='overall' else condition.startswith('score_') if scope=='score_jitter' else condition.startswith('candidate_') if scope=='candidate_dropout' else condition in ['eroded_root','dilated_root','translated_root']
out={'source':'Prespecified descriptive native-ID sensitivity; HPID actual hierarchical part_id versus baseline evaluation-adapter semantic/side/rank IDs. Different ID rules prevent attribution of any difference solely to fusion quality.','thresholds':{}}
for threshold in ['0.25','0.5']:
    rr=[r for r in rows if r['threshold']==threshold]
    categories={r['case_id']:r['object_category'] for r in rr}
    orig=[r for r in rr if r['condition']=='automatic_root' and r['method']=='hpid_split_a3']
    common=defaultdict(list)
    for r in orig:
        if b(r['three_method_common_initial']):common[r['case_id']].append(r['gt_index'])
    caseids=sorted(common)
    casesall=sorted({r['case_id'] for r in orig})
    tt={'eligible_cases':len(caseids),'common_gt_targets':sum(map(len,common.values())),'excluded_cases':[c for c in casesall if c not in common],'excluded_case_gt_count':sum(1 for r in orig if r['case_id'] not in common),'baseline_initial_correct':{m:sum(b(r['initial_correct']) for r in rr if r['condition']=='automatic_root' and r['method']==m) for m in METHODS},'scope_results':[],'guardrails':[]}
    for scope in ['overall','root','score_jitter','candidate_dropout']:
        accum=defaultdict(list)
        for r in rr:
            if b(r['three_method_common_initial']) and selected(r['condition'],scope):accum[(r['case_id'],r['method'])].append(int(b(r['native_joint_correct_retained'])))
        a=np.array([[mean(accum[(c,m)]) for m in METHODS] for c in caseids])
        rng=np.random.default_rng(SEED);ix=rng.integers(0,len(caseids),(10000,len(caseids)))
        boot=a[ix].mean(axis=1)
        corrected=(.0125,.9875) if scope=='overall' else (.05/12,1-.05/12)
        contrasts=[]
        for j in [1,2]:
            delta=boot[:,0]-boot[:,j]
            contrasts.append({'comparison':'hpid_minus_'+METHODS[j],'difference':float((a[:,0]-a[:,j]).mean()),'ci95':ci(delta),'ci_corrected':ci(delta,corrected)})
        tt['scope_results'].append({'scope':scope,'method_case_macro_rates':dict(zip(METHODS,a.mean(axis=0).tolist())),'method_pooled_rates':{m:mean([int(b(r['native_joint_correct_retained'])) for r in rr if r['method']==m and b(r['three_method_common_initial']) and selected(r['condition'],scope)]) for m in METHODS},'contrasts':contrasts})
        for method in METHODS:
            bycase=defaultdict(list)
            for r in rr:
                if r['method']==method and selected(r['condition'],scope):bycase[r['case_id']].append(int(b(r['native_joint_correct_retained'])))
            value=mean([mean(v) for v in bycase.values()])
            source=[r for r in metrics if r['method']==method and selected(r['condition'],scope)]
            tt['guardrails'].append({'scope':scope,'method':method,'all_gt_jointcorrect_case_macro':value,'part_f1_at_025':mean([float(r['part_f1_at_025']) for r in source]),'part_f1_at_050':mean([float(r['part_f1_at_050']) for r in source])})
    out['thresholds'][threshold]=tt
out['baseline_f1']={m:mean([float(r['part_f1_at_025']) for r in metrics if r['method']==m and r['condition']=='automatic_root']) for m in METHODS}
(P/'native_summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps({k:{'eligible_cases':v['eligible_cases'],'common_gt_targets':v['common_gt_targets'],'scope_results':v['scope_results'],'guardrails':v['guardrails'][:3]} for k,v in out['thresholds'].items()},indent=2))
