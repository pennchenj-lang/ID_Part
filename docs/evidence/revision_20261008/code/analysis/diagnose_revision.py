import csv
import hashlib
import json
import subprocess

import numpy as np
from portable_paths import BUNDLE_ROOT, DATA_ROOT, OUTPUT_DIR

ROOT = BUNDLE_ROOT
E = OUTPUT_DIR
R = DATA_ROOT
FRONT = R / 'experiments/paper_v031_identity_frontend_20260828'

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

rows = read(FRONT / '01_same_candidate/same_candidate_postprocessing_cases.csv')
methods = {m: {r['case_id']: r for r in rows if r['method'] == m} for m in {r['method'] for r in rows}}
h = methods['hpid_split_a3']
d = methods['dbscan_fusion']
diagnosis = {'n': len(h), 'methods': {}, 'case_comparison': {}}
for m, data in methods.items():
    diagnosis['methods'][m] = {k: float(np.mean([float(r[k]) for r in data.values()])) for k in ['part_f1_at_025','part_precision_at_025','part_recall_at_025','predicted_part_count','mean_matched_iou_at_025','object_iou','unassigned_root_fraction']}
    diagnosis['methods'][m]['total_matches_at_025'] = sum(round(float(r['part_recall_at_025'])*float(r['truth_part_count'])) for r in data.values())
delta = np.array([float(h[c]['part_f1_at_025'])-float(d[c]['part_f1_at_025']) for c in h])
diagnosis['case_comparison'] = {'hpid_higher': int((delta>1e-10).sum()), 'dbscan_higher': int((delta < -1e-10).sum()), 'equal': int((abs(delta)<=1e-10).sum())}

# Bootstrap objects within categories as a second, taxonomy-clustered diagnostic.
categories = sorted({r['object_category'] for r in h.values()})
per_category = [np.array([delta[i] for i,c in enumerate(h) if h[c]['object_category']==cat]) for cat in categories]
rng = np.random.default_rng(20261008)
means = []
for _ in range(10000):
    groups = rng.integers(len(categories),size=len(categories))
    means.append(np.concatenate([per_category[g] for g in groups]).mean())
diagnosis['category_cluster_bootstrap'] = {'clusters':len(categories),'draws':10000,'seed':20261008,'delta':float(delta.mean()),'low':float(np.quantile(means,.025)),'high':float(np.quantile(means,.975)),'unit':'sample categories with replacement; retain all cases in each selected category'}
(E/'dbscan_diagnosis.json').write_text(json.dumps(diagnosis,indent=2),encoding='utf-8')

probes=[]
for name,relative in [('PartCATSeg','part-catseg'),('HOPS','HOPS')]:
    repo=R/'external_part_baselines_20260828'/relative
    command=[str(R/'.venv/Scripts/python.exe'),'-B','-X','utf8',str(repo/'train_net.py'),'--help']
    run=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',timeout=120,check=False)
    probes.append({'method':name,'command':command,'return_code':run.returncode,'stdout':run.stdout,'stderr':run.stderr,'scope':'local entrypoint environment probe, not a trained inference run','entrypoint_sha256':hashlib.sha256((repo/'train_net.py').read_bytes()).hexdigest()})
(E/'external_entrypoint_probes.json').write_text(json.dumps(probes,indent=2),encoding='utf-8')
print(json.dumps(diagnosis,indent=2))
print([(p['method'],p['return_code']) for p in probes])
