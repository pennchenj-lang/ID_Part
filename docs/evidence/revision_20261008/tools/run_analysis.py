"""Run path-adapted copies without altering the archived numerical evidence."""
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys
B=Path(__file__).resolve().parents[1]
FILES={'taxonomy':'taxonomy_behavior.py','score':'score_provenance.py','routing':'analyze_revision.py','sensitivity':'analyze_revision.py','holdout':'analyze_revision.py','gate':'gate_ablation.py','stage':'stage_diagnosis.py','request':'request_provenance.py'}
def main():
    p=argparse.ArgumentParser();p.add_argument('analysis',choices=FILES);p.add_argument('--data-root',type=Path,default=B/'inputs/runtime');p.add_argument('--output',type=Path,default=B/'replay_results');p.add_argument('--limit',type=int,default=0);p.add_argument('--workers',type=int,default=2);p.add_argument('--check-inputs',action='store_true');a=p.parse_args()
    root=a.data_root.resolve();out=a.output.resolve();required=json.loads((B/'inputs/omitted_input_requirements.json').read_text())
    roles={'gate':{'source_rgb','reference_object','reference_part'},'sensitivity':{'reference_object','reference_part'},'holdout':{'reference_object','reference_part'},'stage':{'reference_object','reference_part'},'request':{'source_rgb','reference_part','request_truth'}}.get(a.analysis,set())
    split='test226' if a.analysis in ['sensitivity','stage'] else 'holdout42' if a.analysis in ['holdout','request'] else None
    missing=[r['path'] for r in required if r['role'] in roles and (split is None or r['split']==split) and not (root/r['path']).is_file()]
    if missing:print(json.dumps({'analysis':a.analysis,'external_inputs_missing':len(missing),'examples':missing[:5],'instruction':'Supply locally obtained inputs identified by inputs/omitted_input_requirements.json; run tools/prepare_inputs.py --check.'},indent=2));raise SystemExit(2)
    if a.check_inputs:print(json.dumps({'analysis':a.analysis,'external_inputs_missing':0}));return
    out.mkdir(parents=True,exist_ok=True);env=dict(os.environ,HPID_DATA_ROOT=str(root),HPID_RESULTS_DIR=str(out),PYTHONHASHSEED='20261008',PYTHONDONTWRITEBYTECODE='1')
    if a.analysis=='score' and not (out/'source_reliability_candidates.csv').exists():shutil.copyfile(B/'evidence/source_reliability_candidates.csv',out/'source_reliability_candidates.csv')
    cmd=[sys.executable,'-B','-X','utf8',str(B/'code/analysis'/FILES[a.analysis])]
    if a.analysis in ['routing','sensitivity','holdout']:cmd.append(a.analysis)
    if a.analysis in ['gate','stage','sensitivity']:cmd+=['--workers',str(a.workers)]
    if a.limit and a.analysis in ['gate','stage','sensitivity','request']:cmd+=['--limit',str(a.limit)]
    if a.analysis=='gate':cmd+=['--output-dir',str(out/'gate_ablation')]
    if a.analysis=='request':cmd+=['--output',str(out)]
    raise SystemExit(subprocess.call(cmd,env=env,cwd=B))
if __name__=='__main__':main()
