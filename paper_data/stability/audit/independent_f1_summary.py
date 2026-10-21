from pathlib import Path
import csv,json,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'audit/statistics'
METHODS=('hpid_split_a3','dbscan_fusion','greedy_nms')
METRICS=('part_f1_at_025','part_f1_at_050','part_f1_at_075','part_f1_mean_025_075','semantic_f1_at_025','object_iou','root_foreground_iou')
with (ROOT/'case_metrics.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
ix={(r['case_id'],r['method'],r['condition']):r for r in rows}
assert len(ix)==6780
cases=sorted({r['case_id'] for r in rows})
assert len(cases)==226
conditions=sorted({r['condition'] for r in rows})
family={r['condition']:r['family'] for r in rows}
scopes={c:[c] for c in conditions}
scopes.update({f:[c for c in conditions if family[c]==f] for f in sorted(set(family.values())) if f!='original'})
scopes['overall_perturbed']=[c for c in conditions if c!='automatic_root']
rng=np.random.default_rng(20261009)
sample=rng.integers(0,226,size=(10000,226))
summaries=[];contrasts=[]
for scope,conds in scopes.items():
    for metric in METRICS:
        vectors={m:np.asarray([np.mean([float(ix[(c,m,k)][metric]) for k in conds]) for c in cases]) for m in METHODS}
        for m,a in vectors.items():
            lo,hi=np.quantile(a[sample].mean(1),(.025,.975))
            summaries.append(dict(scope=scope,metric=metric,method=m,n_cases=226,mean=float(a.mean()),ci95_low=float(lo),ci95_high=float(hi)))
        for baseline in METHODS[1:]:
            d=vectors[METHODS[0]]-vectors[baseline]
            lo,hi=np.quantile(d[sample].mean(1),(.025,.975))
            contrasts.append(dict(scope=scope,metric=metric,baseline=baseline,n_cases=226,mean_paired_difference=float(d.mean()),ci95_low=float(lo),ci95_high=float(hi),interpretation='Descriptive secondary paired 95% interval; no multiplicity-controlled superiority claim'))
def write(name,data):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
write('independent_segmentation_summary.csv',summaries)
write('independent_segmentation_contrasts.csv',contrasts)
print(json.dumps([r for r in summaries if r['metric']=='part_f1_at_025' and r['scope'] in ('automatic_root','root','score_jitter','candidate_dropout','overall_perturbed')],indent=2))
