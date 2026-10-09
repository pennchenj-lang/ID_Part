from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, os, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
RUNTIME=Path(os.environ.get('HPID_RUNTIME_ROOT','runtime'))
FROZEN=RUNTIME/'code_snapshots/hpid_split_be54300_holdout/src'
PROJECT=Path(os.environ.get('HPID_PROJECT_ROOT','project'))
BENCH=RUNTIME/'experiments/paper_v031/test226_hpid'
MANIFEST=RUNTIME/'datasets/paco/broad_objects_test_holdout_v3_4percat/manifest.json'
PUBLISHED=RUNTIME/'experiments/paper_v031_identity_frontend_20260828/01_same_candidate/same_candidate_postprocessing_cases.csv'
sys.path.insert(0,str(FROZEN))
sys.path.insert(0,str(PROJECT/'scripts'))
import hpid_split
hpid_split.__path__.append(str(PROJECT/'src/hpid_split'))
import cv2, numpy as np
from PIL import Image
from hpid_split.fusion import FusionConfig, fuse_candidates
from hpid_split.postprocess_baselines import BaselineInstance,BaselinePrediction,dbscan_proposal_fusion,greedy_nms_ownership
from hpid_split.paper_eval import _semantic_hungarian
from hpid_split.paco_eval import _normalize
from hpid_split.metrics import binary_iou
from evaluate_same_candidate_postprocessing import _load_candidates,_evaluate
from analyze_automatic_root_robustness import _perturbations,_replace_root,_root_candidate
cv2.setNumThreads(1)
METHODS=('hpid_split_a3','dbscan_fusion','greedy_nms')
CONDITIONS=('automatic_root','eroded_root','dilated_root','translated_root',
    'score_jitter_20261009','score_jitter_20261010','score_jitter_20261011',
    'candidate_dropout_20261009','candidate_dropout_20261010','candidate_dropout_20261011')
THRESHOLDS=(.25,.50)

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def csvout(path,rows):
    if not rows:return
    with Path(path).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)
def mask(path):return np.asarray(Image.open(path).convert('L'),dtype=np.uint8)>=128
def md(mask):return hashlib.sha256(np.packbits(np.asarray(mask,dtype=np.uint8),axis=None).tobytes()).hexdigest()
def family(condition):
    return 'original' if condition=='automatic_root' else 'score_jitter' if condition.startswith('score_') else 'candidate_dropout' if condition.startswith('candidate_') else 'root'
def adapter(prediction):
    groups={}
    for inst in prediction.instances:
        binary=np.asarray(inst.mask,dtype=bool)
        yy,xx=np.nonzero(binary)
        if len(xx): groups.setdefault(inst.semantic_name,[]).append((float(yy.mean()),float(xx.mean()),-len(xx),md(binary),inst))
    records=[]
    for semantic in sorted(groups):
        for rank,row in enumerate(sorted(groups[semantic],key=lambda x:x[:4]),1):
            inst=row[4]
            records.append({'key':f'{semantic}/{rank:03d}','native_id':inst.identity,'semantic_name':inst.semantic_name,'semantic_parent':inst.semantic_parent,'confidence':inst.confidence,'mask':np.asarray(inst.mask,dtype=bool)})
    return records
def stable_cand(c):
    return (c.semantic_name,c.semantic_parent,c.source,c.prompt,float(c.score),float(c.source_reliability),md(c.mask),json.dumps(c.metadata,sort_keys=True,ensure_ascii=False))
def variants(candidates,case_id):
    root=_root_candidate(candidates)
    out={k:(_replace_root(candidates,root,m),m,{'root_hash':md(m),'removed_count':0}) for k,m in _perturbations(root.mask,case_id).items()}
    nonroot=sorted([c for c in candidates if c is not root],key=stable_cand)
    for seed in (20261009,20261010,20261011):
        for fam in ('score_jitter','candidate_dropout'):
            digest=hashlib.sha256(f'{fam}:{seed}:{case_id}'.encode()).digest()
            rng=np.random.default_rng(int.from_bytes(digest[:8],'little'))
            if fam=='score_jitter':
                factors=rng.uniform(.95,1.05,len(nonroot))
                modified=[root]+[replace(c,score=float(np.clip(c.score*f,0,1))) for c,f in zip(nonroot,factors)]
                meta={'factors':factors.tolist(),'removed_count':0,'original_nonroot_count':len(nonroot)}
            else:
                number=round(.10*len(nonroot))
                removed=set(rng.choice(len(nonroot),number,replace=False).tolist())
                modified=[root]+[c for i,c in enumerate(nonroot) if i not in removed]
                meta={'removed_count':number,'original_nonroot_count':len(nonroot),'removed_indexes':sorted(removed),'removed_mask_hashes':[md(nonroot[i].mask) for i in sorted(removed)]}
            out[f'{fam}_{seed}']=(modified,root.mask,meta)
    return out
def cached_prediction(case_id,method,condition,candidates,root,signature):
    folder=HERE/'cache'/case_id
    stem=folder/f'{method}__{condition}'
    meta_path=stem.with_suffix('.json');npz_path=stem.with_suffix('.npz')
    if meta_path.exists() and npz_path.exists():
        meta=read(meta_path)
        if meta['signature']!=signature:raise RuntimeError('cache provenance mismatch '+str(meta_path))
        z=np.load(npz_path);shape=tuple(meta['shape']);size=shape[0]*shape[1]
        records=[{**r,'mask':np.unpackbits(z['masks'][i])[:size].reshape(shape).astype(bool)} for i,r in enumerate(meta['records'])]
        pred=BaselinePrediction(method,tuple(BaselineInstance(r['native_id'],r['semantic_name'],r['semantic_parent'],r['mask'],r['confidence']) for r in records),root,{})
        return pred,records,meta
    started=time.perf_counter();error=None;exact=None
    try:
        if method=='hpid_split_a3':
            result=fuse_candidates(candidates,image_shape=root.shape,config=FusionConfig())
            elapsed=time.perf_counter()-started
            pred=BaselinePrediction(method,tuple(BaselineInstance(r.part_id,r.semantic_name,r.semantic_parent,result.instance_map==r.instance_index,1.0) for r in result.instances),root,{})
            if condition=='automatic_root':exact=bool(np.array_equal(result.instance_map,np.asarray(Image.open(BENCH/case_id/'part_id_map.tiff'))))
        else:
            pred=(dbscan_proposal_fusion if method=='dbscan_fusion' else greedy_nms_ownership)(candidates)
            elapsed=time.perf_counter()-started
    except Exception:
        elapsed=time.perf_counter()-started;error=traceback.format_exc()
        pred=BaselinePrediction(method,tuple(),root,{'error':error})
    records=adapter(pred)
    folder.mkdir(parents=True,exist_ok=True)
    packed=np.stack([np.packbits(r['mask'].reshape(-1)) for r in records]) if records else np.empty((0,(root.size+7)//8),dtype=np.uint8)
    np.savez_compressed(npz_path,masks=packed)
    meta={'signature':signature,'case_id':case_id,'method':method,'condition':condition,'shape':list(root.shape),'records':[{k:v for k,v in r.items() if k!='mask'} for r in records],'runtime_seconds':elapsed,'error':error,'original_hpid_map_exact':exact,'npz_sha256':sha(npz_path)}
    dump(meta_path,meta)
    return pred,records,meta
def scoring_records(records,domain,category,truthnames):
    output=[]
    for r in records:
        isroot=r['semantic_name']==domain
        if isroot and 'body' not in truthnames:continue
        semantic='body' if isroot else _normalize(r['semantic_name'].removeprefix(domain+'_'),domain,object_category=category)
        output.append({**r,'normalized_semantic':semantic})
    return sorted(output,key=lambda r:r['key'])
def worker(job):
    raw,manifestcase,signature=job
    case_id=raw['case_id'];casepath=Path(manifestcase['case_path'])
    target=HERE/'cases'/f'{case_id}.json'
    if target.exists():
        previous=read(target)
        if previous['signature']!=signature:raise RuntimeError('result provenance mismatch')
        return previous
    candidates=_load_candidates(BENCH/case_id)
    predictions={};variantmeta={}
    for condition,(changed,root,details) in variants(candidates,case_id).items():
        variantmeta[condition]=details
        for method in METHODS:
            predictions[(method,condition)]=cached_prediction(case_id,method,condition,changed,root,signature)
    # Ground truth first enters here, after every output and identity has been cached.
    case=read(casepath);domain=raw['expected_domain'];category=case['object_category']
    truths=[mask(casepath.parent/r['mask_crop']) for r in case['parts']]
    semantics=[_normalize(r['part_name'],domain,object_category=category) for r in case['parts']]
    scored={};metrics=[];matches={}
    for (method,condition),(pred,recs,meta) in predictions.items():
        sr=scoring_records(recs,domain,category,set(semantics));scored[(method,condition)]=sr
        sm=_semantic_hungarian(truths,semantics,[r['mask'] for r in sr],[r['normalized_semantic'] for r in sr])
        matches[(method,condition)]={g:(sr[p],float(iou)) for g,p,iou in sm}
        metrics.append({'case_id':case_id,'object_category':category,'method':method,'condition':condition,'family':family(condition),'candidate_count':len(candidates),'runtime_seconds':meta['runtime_seconds'],'error':meta['error'],'original_hpid_map_exact':meta['original_hpid_map_exact'],**_evaluate(pred,case=case,case_dir=casepath.parent,expected_domain=domain)})
    long=[]
    for threshold in THRESHOLDS:
        initial={m:matches[(m,'automatic_root')] for m in METHODS}
        def initial_ok(method,g):return g in initial[method] and initial[method][g][1]>=threshold
        for g,(gt,semantic) in enumerate(zip(truths,semantics)):
            all_common=all(initial_ok(m,g) for m in METHODS)
            pairs={b:initial_ok('hpid_split_a3',g) and initial_ok(b,g) for b in METHODS[1:]}
            for method in METHODS:
                initial_pair=initial[method].get(g)
                initial_key=initial_pair[0]['key'] if initial_pair else None
                initial_iou=initial_pair[1] if initial_pair else 0.
                for condition in CONDITIONS:
                    pair=matches[(method,condition)].get(g)
                    current_key=pair[0]['key'] if pair else None
                    current_iou=pair[1] if pair else 0.
                    correct=bool(pair and current_iou>=threshold)
                    retained=bool(initial_ok(method,g) and correct and current_key==initial_key)
                    samekey=next((r for r in scored[(method,condition)] if r['key']==initial_key),None)
                    long.append({'case_id':case_id,'object_category':category,'gt_index':g,'gt_semantic':semantic,'gt_mask_sha256':md(gt),'method':method,'condition':condition,'family':family(condition),'threshold':threshold,'initial_correct':initial_ok(method,g),'initial_key':initial_key,'initial_native_id':initial_pair[0]['native_id'] if initial_pair else None,'initial_iou':initial_iou,'current_correct':correct,'current_key':current_key,'current_native_id':pair[0]['native_id'] if pair else None,'current_iou':current_iou,'same_key_present':bool(samekey),'same_key_gt_iou':binary_iou(samekey['mask'],gt) if samekey else 0.,'joint_correct_retained':retained,'native_joint_correct_retained':bool(initial_ok(method,g) and correct and pair[0]['native_id']==initial_pair[0]['native_id']),'three_method_common_initial':all_common,'common_initial_hpid_dbscan':pairs['dbscan_fusion'],'common_initial_hpid_nms':pairs['greedy_nms']})
    result={'signature':signature,'case_id':case_id,'object_category':category,'gt_count':len(truths),'candidate_count':len(candidates),'source_manifest_record':manifestcase,'variants':variantmeta,'metrics':metrics,'targets':long}
    dump(target,result)
    return result
def provenance():
    paths=[HERE/'protocol.json',HERE/'run_stability.py',FROZEN/'hpid_split/fusion.py',FROZEN/'hpid_split/instances.py',FROZEN/'hpid_split/metrics.py',FROZEN/'hpid_split/paper_eval.py',FROZEN/'hpid_split/paco_eval.py',PROJECT/'src/hpid_split/postprocess_baselines.py',PROJECT/'scripts/evaluate_same_candidate_postprocessing.py',PROJECT/'scripts/analyze_automatic_root_robustness.py',MANIFEST,BENCH/'benchmark_summary.json',PUBLISHED]
    hashes={str(p):sha(p) for p in paths}
    signature=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    return {'signature':signature,'files':hashes,'python':sys.version,'numpy':np.__version__,'cv2':cv2.__version__,'methods':METHODS,'conditions':CONDITIONS}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    pr=provenance();dump(HERE/'provenance.json',pr)
    manifest={r['case_id']:r for r in read(MANIFEST)['cases'] if r.get('case_path')}
    raw=sorted(read(BENCH/'benchmark_summary.json')['cases'],key=lambda r:r['case_id'])
    assert len(raw)==226 and set(manifest)=={r['case_id'] for r in raw}
    if args.limit:raw=raw[:args.limit]
    results=[]
    with ProcessPoolExecutor(max_workers=min(args.workers,4)) as pool:
        futures={pool.submit(worker,(r,manifest[r['case_id']],pr['signature'])):r['case_id'] for r in raw}
        for f in as_completed(futures):
            result=f.result();results.append(result)
            print(json.dumps({'completed':len(results),'total':len(raw),'case_id':result['case_id'],'errors':sum(bool(r['error']) for r in result['metrics'])}),flush=True)
    results.sort(key=lambda r:r['case_id'])
    allmetrics=[r for x in results for r in x['metrics']];alltargets=[r for x in results for r in x['targets']]
    suffix='_engineering' if args.limit else ''
    csvout(HERE/f'case_metrics{suffix}.csv',allmetrics);csvout(HERE/f'target_retention{suffix}.csv',alltargets)
    with PUBLISHED.open(encoding='utf-8-sig',newline='') as f:published={(r['case_id'],r['method']):r for r in csv.DictReader(f)}
    checks=[]
    for r in allmetrics:
        if r['condition']!='automatic_root':continue
        old=published[(r['case_id'],r['method'])]
        diffs={k:abs(float(r[k])-float(old[k])) for k in old if k in r and k not in ('runtime_seconds','candidate_count') and isinstance(r[k],(float,int)) and not isinstance(r[k],bool)}
        checks.append({'case_id':r['case_id'],'method':r['method'],'max_abs_metric_difference':max(diffs.values(),default=0),'differing_metrics':{k:v for k,v in diffs.items() if v>1e-12},'original_hpid_map_exact':r['original_hpid_map_exact'],'error':r['error'],'published_runtime_seconds':old['runtime_seconds']})
    dump(HERE/f'reproduction_audit{suffix}.json',{'case_count':len(results),'all_metrics_match':all(not r['differing_metrics'] for r in checks),'all_hpid_maps_exact':all(r['original_hpid_map_exact'] for r in checks if r['method']=='hpid_split_a3'),'failure_count':sum(bool(r['error']) for r in allmetrics),'checks':checks})
    dump(HERE/f'completion{suffix}.json',{'cases':len(results),'candidate_count':sum(x['candidate_count'] for x in results),'gt_count':sum(x['gt_count'] for x in results),'prediction_runs':len(allmetrics),'target_rows':len(alltargets),'signature':pr['signature']})
    print('Completed; reproduction audit saved. No inferential summary computed.',flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
