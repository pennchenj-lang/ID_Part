"""Frozen same-candidate routing and controlled request-mask handoff replay.

No foundation model, inpainting model, parameter search or Word generation.
Predictions are constructed before reading each evaluation target mask.
"""
from __future__ import annotations
import argparse, csv, datetime, hashlib, json, os, sys, traceback
from pathlib import Path

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
sys.dont_write_bytecode=True
R=Path(__file__).resolve().parent
SNAPSHOT=Path('__RUNTIME_ROOT__/code_snapshots/hpid_split_be54300_holdout')
PROJECT=Path('__USER_HOME__/Documents/Codex/2026-06-28/new-chat/hpid_split')
sys.path.insert(0,str(SNAPSHOT/'src'))
sys.path.insert(0,str(PROJECT))
import hpid_split
hpid_split.__path__.append(str(PROJECT/'src/hpid_split'))
from scripts.run_semantic_completion_frontend_benchmark import (
    _predictions,_normalize,_normalized_name,_mask_iou)
from hpid_split.completion_frontend import make_controlled_defect
from hpid_split.postprocess_baselines import BaselineInstance,BaselinePrediction
import numpy as np
from PIL import Image
import cv2
cv2.setNumThreads(1)

METHODS=('raw_proposals','greedy_nms','dbscan_fusion','local_pairwise_crf','hpid_split_group_ids')

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def csvread(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvwrite(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
def array_sha(mask):return hashlib.sha256(np.ascontiguousarray(mask,dtype=np.uint8).tobytes()).hexdigest()

def implementation_hashes():
    paths={Path(__file__).resolve()}
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path:
            p=Path(path).resolve()
            if p.suffix=='.py' and (p.is_relative_to(SNAPSHOT) or p.is_relative_to(PROJECT)):
                paths.add(p)
    return {str(p):sha(p) for p in sorted(paths)}

def collect_inputs(requests):
    paths={R.parent/'review_burden/review_cases.csv'}
    for row in requests:
        package=Path(row['package_path'])
        for name in ('source.png','candidates.json','groups.json','group_id_map.tiff','package_manifest.json'):
            paths.add(package/name)
        paths.update(package/c['mask_path'] for c in read(package/'candidates.json'))
        paths.update((Path(row['case_path']),Path(row['target_mask_path'])))
        for key in ('hpid_group_map_path','hpid_groups_path'):
            if row.get(key): paths.add(Path(row[key]))
    return {str(p.resolve()):sha(p) for p in sorted(paths)}

def load_predictions(row):
    predictions=_predictions(Path(row['package_path']))
    if row.get('hpid_group_map_path'):
        labels=np.asarray(Image.open(row['hpid_group_map_path']))
        groups=read(row['hpid_groups_path'])
        predictions['hpid_split_group_ids']=BaselinePrediction('hpid_split_group_ids',tuple(
            BaselineInstance(str(g['group_id']),str(g['semantic_name']),str(g['semantic_name']),
                             labels==int(g['group_index']),1.0) for g in groups),
            predictions['raw_proposals'].root_mask,{'source':'Frozen common Group reconstruction'})
    return predictions

def replay_request(target,out):
    predictions=load_predictions(target)
    # Ground truth is read only after all methods have produced their instances.
    assert sha(target['target_mask_path'])==target['target_mask_sha256']
    truth=np.asarray(Image.open(target['target_mask_path']).convert('L'))>=128
    defect=make_controlled_defect(truth,seed=int(target['defect_seed']),
          fraction=float(target['defect_fraction']),mode=target['defect_mode'])
    normal_target=_normalize(target['target_part_name'],target['expected_domain'],
                             object_category=target['object_category'])
    request_dir=out/'request_masks'/target['request_id'];request_dir.mkdir(parents=True,exist_ok=True)
    Image.fromarray(defect.mask.astype('uint8')*255).save(request_dir/'controlled_defect.png')
    rows=[];instances=[];traces={}
    for method in METHODS:
        pred=predictions[method]
        scored=[(i,_normalized_name(i,expected_domain=target['expected_domain'],
                    object_category=target['object_category']),_mask_iou(i.mask,truth)) for i in pred.instances]
        matched=[v for v in scored if v[1]==normal_target]
        exposed=matched if matched else scored
        selected=max(matched,key=lambda v:(float(v[0].confidence),int(np.count_nonzero(v[0].mask)),v[0].identity)) if matched else None
        best_match=max((s for _,_,s in matched),default=0.0)
        unique=len(matched)==1
        selected_iou=selected[2] if selected else 0.0
        selected_mask=np.asarray(selected[0].mask,dtype=bool) if selected else np.zeros_like(truth)
        request=defect.mask & selected_mask
        request_path=request_dir/(method+'.png')
        Image.fromarray(request.astype('uint8')*255).save(request_path)
        row={k:target[k] for k in ('request_id','case_id','cohort','image_id','source_image_sha256','object_category','domain')}
        row.update(method=method,execution_status='complete',target=target['target_part_name'],normalized_target=normal_target,
            query_match_count=len(matched),all_exported_count=len(scored),review_option_count=len(exposed),
            query_unique=int(unique),query_ambiguous=int(len(matched)>1),query_unresolved=int(not matched),
            query_state='unique' if unique else 'ambiguous' if matched else 'unresolved',
            selected_identity=selected[0].identity if selected else '',selected_part_iou=selected_iou,
            best_review_match_iou=best_match,best_exposed_iou=max((s for _,_,s in exposed),default=0.0),
            best_exported_iou=max((s for _,_,s in scored),default=0.0),
            controlled_defect_pixels=int(defect.mask.sum()),request_pixels=int(request.sum()),
            request_recall=float(request.sum()/max(1,defect.mask.sum())),
            request_mask_path=str(request_path.resolve()),request_mask_sha256=sha(request_path),
            defect_mode=defect.mode,defect_fraction_actual=float(defect.actual_fraction))
        for suffix,threshold in (('025',.25),('050',.50)):
            row.update({f'unique_and_correct_{suffix}':int(unique and selected_iou>=threshold),
                f'wrong_unique_{suffix}':int(unique and selected_iou<threshold),
                f'semantic_correct_any_match_{suffix}':int(best_match>=threshold),
                f'geometry_correct_any_exposed_{suffix}':int(row['best_exposed_iou']>=threshold),
                f'geometry_correct_any_exported_{suffix}':int(row['best_exported_iou']>=threshold),
                f'correct_in_review_set_{suffix}':int(len(matched)>1 and best_match>=threshold)})
        rows.append(row)
        for ins,normal,score in scored:
            instances.append(dict(request_id=target['request_id'],case_id=target['case_id'],method=method,
                identity=ins.identity,semantic=ins.semantic_name,normalized_semantic=normal,
                matches_target=int(normal==normal_target),actual_exposed=int(not matched or normal==normal_target),
                confidence=float(ins.confidence),area_px=int(np.count_nonzero(ins.mask)),target_iou=score,mask_sha256=array_sha(ins.mask)))
        traces[method]=dict(selected_identity=row['selected_identity'],match_count=len(matched),
              instance_count=len(scored),request_mask_sha256=row['request_mask_sha256'])
    # Pixel destinations for every raw target-matching proposal; attribution is
    # descriptive. Gate-level causation requires the separate replay diagnostics.
    dest=predictions['hpid_split_group_ids'].instances
    pixel_flow=[]
    for raw in predictions['raw_proposals'].instances:
        if _normalized_name(raw,expected_domain=target['expected_domain'],object_category=target['object_category'])!=normal_target:continue
        flow=[];covered=np.zeros_like(truth)
        for group in dest:
            overlap=raw.mask & group.mask
            if overlap.any():
                covered |= overlap
                flow.append(dict(identity=group.identity,semantic=group.semantic_name,
                  normalized_semantic=_normalized_name(group,expected_domain=target['expected_domain'],object_category=target['object_category']),
                  candidate_overlap_pixels=int(overlap.sum()),truth_overlap_pixels=int((overlap & truth).sum()),
                  final_target_iou=_mask_iou(group.mask,truth)))
        pixel_flow.append(dict(raw_identity=raw.identity,raw_target_iou=_mask_iou(raw.mask,truth),
              raw_area=int(raw.mask.sum()),raw_truth_overlap=int((raw.mask & truth).sum()),
              not_owned_by_exported_group=int((raw.mask & ~covered).sum()),destinations=flow))
    save(out/'case_traces'/(target['request_id']+'.json'),dict(request_id=target['request_id'],
        methods=traces,raw_target_candidate_pixel_destinations=pixel_flow,
        scope='Observed pixel ownership destinations; not an isolated causal ablation'))
    return rows,instances

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,required=True);ap.add_argument('--protocol-sha256',required=True)
    ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    assert sha(args.protocol)==args.protocol_sha256
    p=read(args.protocol);assert p['status']=='FROZEN_BEFORE_EXPANDED_ROUTING'
    assert sha(args.manifest)==p['manifest_sha256']
    assert implementation_hashes()==p['implementation_hashes']
    for path,digest in p['input_hashes'].items():assert sha(path)==digest,path
    targets=read(args.manifest)['requests']
    assert len(targets)==253 and len({t['image_id'] for t in targets})==253
    assert len({t['request_id'] for t in targets})==253
    assert len({t['source_image_sha256'] for t in targets})==253
    assert sum(t['cohort']=='old37' for t in targets)==37
    assert sum(t['cohort']=='new216' for t in targets)==216
    assert not (args.output/'receipt.json').exists(),'Do not overwrite a sealed run'
    args.output.mkdir(parents=True,exist_ok=True)
    old={(r['case_id'],r['method']):r for r in csvread(R.parent/'review_burden/review_cases.csv')}
    rows=[];instances=[];checks=[];failures=[]
    for index,target in enumerate(targets):
        try:
            new,ins=replay_request(target,args.output)
            if target['cohort']=='old37':
                for row in new:
                    reference=old[row['case_id'],row['method']]
                    for k,v in reference.items():
                        assert k in row,('Missing legacy field',k)
                        if k in ('case_id','method','target','normalized_target','query_state','selected_identity'):
                            assert str(row[k])==v,(row['case_id'],row['method'],k)
                        else:assert abs(float(row[k])-float(v))<1e-12,(row['case_id'],row['method'],k)
                    checks.append(dict(request_id=target['request_id'],method=row['method'],status='PASS'))
            rows.extend(new);instances.extend(ins)
        except Exception as error:
            failures.append(dict(request_id=target['request_id'],error=str(error),traceback=traceback.format_exc()))
            save(args.output/'failures.json',failures)
            # Fail closed on technical or legacy reproduction errors. No request
            # may disappear or masquerade as a zero-option improvement.
            raise
        save(args.output/'progress.json',dict(completed=index+1,planned=len(targets),failures=len(failures)))
        print(json.dumps(dict(completed=index+1,planned=len(targets),request_id=target['request_id'])),flush=True)
    csvwrite(args.output/'routing_cases.csv',rows);csvwrite(args.output/'routing_instances.csv',instances)
    save(args.output/'old37_reproduction.json',dict(status='PASS',checks=checks))
    assert len(rows)==253*5 and len(checks)==37*5
    assert implementation_hashes()==p['implementation_hashes']
    artifacts={str(path.relative_to(args.output)):sha(path)
               for path in sorted(args.output.rglob('*')) if path.is_file()}
    save(args.output/'receipt.json',dict(status='COMPLETE',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        requests=len(targets),case_methods=len(rows),failures=failures,protocol_sha256=sha(args.protocol),
        manifest_sha256=sha(args.manifest),routing_csv_sha256=sha(args.output/'routing_cases.csv'),
        instances_csv_sha256=sha(args.output/'routing_instances.csv'),old37_reproduction='PASS',
        artifact_sha256=artifacts,
        scope='Routing and controlled request-mask handoff only; no new neural restoration or occluded identity inference'))

if __name__=='__main__':main()
