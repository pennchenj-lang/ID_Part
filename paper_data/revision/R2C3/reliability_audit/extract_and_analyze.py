"""Frozen accepted-candidate empirical reliability audit, CPU and read-only inputs.

The existing score/weight is never fitted, modified or interpreted as calibrated.
Subcommands are intentionally sequential: freeze -> extract -> analyze.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import csv
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name] = '1'
sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
WORKSPACE = OUT.parents[2]
EVIDENCE = WORKSPACE/'hpid-publication-20261009/docs/evidence/revision_20261008/evidence'
RUNTIME = Path('__RUNTIME_ROOT__')
BENCH = RUNTIME/'experiments/paper_v031/test226_hpid'
SOURCE = RUNTIME/'code_snapshots/hpid_split_be54300_holdout/src'
sys.path.insert(0,str(SOURCE))
import numpy as np
from PIL import Image
from hpid_split.fusion import FusionConfig, MaskCandidate, _source_family, _source_agreement_factors, _is_broad_scene_layer
from hpid_split.paco_eval import _normalize


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def write_csv(path,value):
    fields=list(dict.fromkeys(k for r in value for k in r))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(value)


def binary(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('L'))>=128


def normalize_prediction(name,domain,category):
    return 'body' if name == domain else _normalize(name.removeprefix(domain+'_'),domain,object_category=category)


def bindings():
    benchmark=read(BENCH/'benchmark_summary.json')
    files=[Path(__file__),BENCH/'benchmark_summary.json',Path(benchmark['source_manifest']),
           EVIDENCE/'score_candidate_provenance.csv',EVIDENCE/'score_protocol.json',
           EVIDENCE/'stage_shared_input_audit.csv',EVIDENCE/'stage_reference_tracks.csv',
           EVIDENCE/'stage_case_metrics.csv',SOURCE/'hpid_split/fusion.py',
           SOURCE/'hpid_split/paco_eval.py',SOURCE/'hpid_split/paco_semantics.py',
           EVIDENCE.parent/'code/analysis/score_provenance.py',
           EVIDENCE.parent/'code/analysis/stage_diagnosis.py']
    return {str(p.resolve()):sha(p) for p in files}


def validate_protocol():
    path=OUT/'protocol_frozen.json'
    assert sha(path)==(OUT/'protocol.sha256').read_text().strip()
    p=read(path)
    assert p['status']=='FROZEN_BEFORE_EMPIRICAL_EXTRACTION_AND_STATISTICS'
    assert p['input_sha256']==bindings(), 'Frozen source/input binding changed'
    return p


def freeze():
    assert not (OUT/'protocol_frozen.json').exists(), 'Refuse to overwrite frozen protocol'
    assert not (OUT/'candidates_all_1951.csv').exists(), 'Results already exist'
    p={'status':'FROZEN_BEFORE_EMPIRICAL_EXTRACTION_AND_STATISTICS',
       'created_utc':datetime.now(timezone.utc).isoformat(),
       'authorized_scope':'Post-review empirical diagnostic reuse of 226 frozen cases and 1951 already accepted candidates; no independent test set.',
       'expected_counts':{'cases':226,'categories':60,'accepted_candidates':1951,'reference_parts':990,
                          'original_evaluation_included_candidates':1830,'root_body_excluded_candidates':121},
       'candidate_definition':'One archived candidates.json record; order/key and all candidates are preserved, without additional area/confidence filtering.',
       's':'Final stored proposal score; heterogeneous detector/model-quality/heuristic/compound provenance, not uniformly a raw detector probability.',
       'r':'Stored source_reliability; allowed above1; no clipping of r outside the frozen weight formula.',
       'a':'Frozen cross-family agreement factor; verify against provenance; defaults imply a=1 on these candidates.',
       'w_formula':'w0=clip((0.45+0.55*clip(s,0,1))*r*a,0.01,0.995); broad scene-layer w=clip(w0*0.72,0.01,0.995), otherwise w=w0.',
       'w_interpretation':'Heuristic fusion weight, not a calibrated correctness probability. No fitting, recalibration, threshold optimization or model rerun.',
       'normalization':'Exact frozen evaluator expected-domain prefix removal and category-dependent canonical alias mapping. No GT-driven relabeling or alternate bridge.',
       'evaluation_included':'Exclude candidate iff semantic_name==expected_domain and normalized GT truth_names contains no body. This is the archived root-as-body evaluator convention, not an inference operation.',
       'iou_definition':'Direct binary intersection/union; candidate masks >=128; best geometric IoU=max over all annotated reference parts in case; best semantic IoU=max over same-normalized-semantic annotated reference parts.',
       'missing_semantic_gt':'If same_semantic_gt_count==0, best_semantic_iou and semantic_hit_025/050 are NULL. No same-class annotation does not establish a false positive. Unknown/unannotated physical parts remain unassessed.',
       'primary_population':'evaluation_included AND same_semantic_gt_count>0',
       'primary_targets':['best_semantic_iou>=0.25','best_semantic_iou>=0.50'],
       'secondary_population':'All1830 evaluation_included candidates',
       'secondary_targets':['best_geometric_iou>=0.25','best_geometric_iou>=0.50'],
       'full_population_retention':'All1951 records retained and flagged; no GT-absence rows silently converted into semantic negatives. Full geometry and same-GT applicability counts retained as descriptive extraction fields.',
       'statistics':{'groups':'All families pooled plus each recorded frozen family, including zero-eligible family rows.',
          'fields':['candidate_n','case_n','category_n','mean_w','matched_fraction','Brier','ECE'],
          'unit_weighting':'Candidate-weighted ratios sum(numerator)/sum(candidate denominator). Candidate observations are clustered, not independent replicates.',
          'bins':{'edges':[i/10 for i in range(11)],'membership':'[left,right), except last [0.9,1.0]; empty bins retained',
                  'fields':['candidate_n','case_n','category_n','mean_w','matched_fraction']},
          'Brier':'mean((w-binary_hit)^2), diagnostic only under the hypothetical interpretation of w as a correctness probability.',
          'ECE':'sum_b(n_b/n)*abs(mean_w_b-matched_fraction_b), fixed10 equal-width bins, diagnostic only under hypothetical probability interpretation.',
          'scope_limit':'Conditional annotation-supported and accepted-pool population. Brier/ECE do not establish calibration success; no tuned bins, corrected scores or deployment decisions.'},
       'bootstrap':{'draws':10000,'seed':20261010,'generator':'numpy.default_rng/PCG64',
          'cluster':'Original 60 object categories, sorted lexicographically; sample60 categories with replacement each draw and retain all eligible candidates of every sampled category, with multiplicity.',
          'shared_draws':'Identical category index draws reused for every family, bin, population, and threshold.',
          'interval':'Pointwise percentile95%, numpy.quantile(method=linear); descriptive exploratory, no multiplicity adjustment or superiority test.',
          'ratio':'Recompute candidate-weighted numerator/denominator inside each cluster draw.',
          'zero_denominator':'Discard only undefined empty denominator draws; report valid_draws and zero_denominator_draws; interval is conditional on nonempty draws. Original empty groups/bins have NULL estimates and CIs.',
          'sparse_groups':'Always retain count/case/category metadata and sparse flags; no candidate-iid intervals.'},
       'input_sha256':bindings()}
    write(OUT/'protocol_frozen.json',p)
    (OUT/'protocol.sha256').write_text(sha(OUT/'protocol_frozen.json')+'\n',encoding='ascii')
    print('FROZEN protocol_sha256',sha(OUT/'protocol_frozen.json'),flush=True)


def extract():
    p=validate_protocol()
    assert not (OUT/'extraction_receipt.json').exists(), 'Preserve completed extraction'
    benchmark=read(BENCH/'benchmark_summary.json')
    assert sha(benchmark['source_manifest'])==benchmark['source_manifest_sha256']
    manifest={r['case_id']:r for r in read(benchmark['source_manifest'])['cases']}
    provenance={(r['case_id'],int(r['candidate_index'])):r for r in rows(EVIDENCE/'score_candidate_provenance.csv')}
    shared={r['case_id']:r for r in rows(EVIDENCE/'stage_shared_input_audit.csv')}
    input_tracks={(r['case_id'],int(r['reference_index'])):r for r in rows(EVIDENCE/'stage_reference_tracks.csv') if r['stage']=='input_pool'}
    config=FusionConfig()
    assert asdict(config)==read(EVIDENCE/'score_protocol.json')['fusion_config']
    all_rows, reference_rows, input_receipts=[],[],[]
    cases=[r for r in benchmark['cases'] if r['return_code']==0]
    for case_number,raw in enumerate(sorted(cases,key=lambda r:r['case_id']),1):
        cid=raw['case_id']; package=BENCH/cid
        assert sha(package/'candidates.json')==shared[cid]['candidates_json_sha256']
        casepath=Path(manifest[cid]['case_path']); case=read(casepath)
        domain=raw['expected_domain']; category=case['object_category']
        gt=case['parts']; truths=[binary(casepath.parent/g['mask_crop']) for g in gt]
        truth_names=[_normalize(g['part_name'],domain,object_category=category) for g in gt]
        assert len(truths)==int(shared[cid]['reference_count'])
        raw_candidates=read(package/'candidates.json')
        candidates=[]; mask_files=[]
        for c in raw_candidates:
            cp=package/c['mask_path']; cm=binary(cp)
            assert cm.shape==truths[0].shape
            assert int(cm.sum())==int(c['area_px'])
            mask_files.append({'path':str(cp),'sha256':sha(cp)})
            candidates.append(MaskCandidate(semantic_name=c['semantic_name'],semantic_parent=c['semantic_parent'],
                mask=cm,score=float(c['score']),source=c['source'],prompt=c.get('prompt',''),
                source_reliability=float(c.get('source_reliability',1)),metadata=c.get('metadata') or {}))
        assert len(candidates)==int(shared[cid]['candidate_count'])
        agreements,_=_source_agreement_factors(candidates,config)
        matrix=np.zeros((len(candidates),len(truths)),dtype=np.float64)
        normalized=[]; included=[]
        for i,(c,a) in enumerate(zip(candidates,agreements)):
            assert int(raw_candidates[i]['candidate_index'])==i+1
            for j,t in enumerate(truths):
                matrix[i,j]=np.count_nonzero(c.mask&t)/max(1,np.count_nonzero(c.mask|t))
            name=normalize_prediction(c.semantic_name,domain,category)
            eligible=c.semantic_name!=domain or 'body' in truth_names
            normalized.append(name);included.append(eligible)
            same=[j for j,n in enumerate(truth_names) if n==name]
            best=int(np.argmax(matrix[i])); sj=max(same,key=lambda j:matrix[i,j]) if same else None
            s=float(c.score);r=float(c.source_reliability);a=float(a)
            clipped=float(np.clip(s,0,1));w0=float(np.clip((.45+.55*clipped)*r*a,.01,.995))
            broad=_is_broad_scene_layer(c);w=float(np.clip(w0*config.scene_layer_fallback_weight,.01,.995)) if broad else w0
            family=_source_family(c); original=provenance[cid,i+1]
            assert family==original['family'] and c.semantic_name==original['semantic_name'] and c.source==original['source']
            for key,value in [('score',s),('source_reliability',r),('agreement',a),('clipped_score',clipped),('weight_before_scene_factor',w0),('effective_weight',w)]:
                assert abs(value-float(original[key]))<1e-12,(cid,i,key)
            assert broad==(original['broad_scene_layer']=='True')
            assert a==1.0
            sem_iou=float(matrix[i,sj]) if sj is not None else None
            all_rows.append({'case_id':cid,'candidate_index':i+1,'candidate_key':c.metadata.get('candidate_key'),
                'object_category':category,'expected_domain':domain,'image_id':case['image_id'],
                'object_annotation_id':case['object_annotation_id'],'semantic_name':c.semantic_name,
                'semantic_parent':c.semantic_parent,'normalized_semantic':name,'source':c.source,'family':family,
                'metadata_source_family':c.metadata.get('source_family'),'s':s,'r':r,'a':a,'clipped_s':clipped,
                'w_before_scene_factor':w0,'broad_scene_layer':int(broad),'w':w,
                'candidate_area_px':int(c.mask.sum()),'image_area_px':int(c.mask.size),
                'candidate_image_fraction':float(c.mask.mean()),'mask_path':str(package/raw_candidates[i]['mask_path']),
                'mask_file_sha256':mask_files[-len(candidates)+i]['sha256'],
                'mask_binary_sha256':hashlib.sha256(c.mask.astype(np.uint8).tobytes()).hexdigest(),
                'is_root_by_parent_equality':int(c.semantic_name==c.semantic_parent),
                'is_evaluation_domain_root':int(c.semantic_name==domain),
                'case_has_normalized_body_gt':int('body' in truth_names),'evaluation_included':int(eligible),
                'evaluation_exclusion_reason':'' if eligible else 'domain_root_excluded_because_no_annotated_body',
                'gt_part_count':len(gt),'same_semantic_gt_count':len(same),
                'semantic_annotation_status':'same_semantic_annotation_present' if same else 'no_same_semantic_annotation_unknown',
                'best_geometric_iou':float(matrix[i,best]),'best_geometric_gt_index':best,
                'best_geometric_gt_annotation_id':gt[best]['annotation_id'],
                'best_geometric_gt_semantic':truth_names[best],
                'best_semantic_iou':sem_iou,'best_semantic_gt_index':sj,
                'best_semantic_gt_annotation_id':gt[sj]['annotation_id'] if sj is not None else None,
                'geometric_hit_025':int(matrix[i,best]>=.25),'geometric_hit_050':int(matrix[i,best]>=.5),
                'semantic_hit_025':int(sem_iou>=.25) if sem_iou is not None else None,
                'semantic_hit_050':int(sem_iou>=.5) if sem_iou is not None else None,
                'primary_semantic_analysis_included':int(eligible and bool(same)),
                'w_is_calibrated_probability':0})
        # Independent extraction check against the already sealed reference-facing pool trace.
        for j,g in enumerate(gt):
            em=[i for i,yes in enumerate(included) if yes]
            sm=[i for i in em if normalized[i]==truth_names[j]]
            observed=input_tracks[cid,j]
            best_any=max((matrix[i,j] for i in em),default=0.0)
            best_sem=max((matrix[i,j] for i in sm),default=0.0)
            assert abs(best_any-float(observed['best_iou']))<1e-12
            assert abs(best_sem-float(observed['best_semantic_iou']))<1e-12
            reference_rows.append({'case_id':cid,'reference_index':j,'annotation_id':g['annotation_id'],
                'raw_part_name':g['part_name'],'normalized_semantic':truth_names[j],
                'mask_path':str(casepath.parent/g['mask_crop']),'mask_sha256':sha(casepath.parent/g['mask_crop']),
                'area_px':int(truths[j].sum()),'stage_input_best_iou_reproduced':float(best_any),
                'stage_input_best_semantic_iou_reproduced':float(best_sem)})
        input_receipts.append({'case_id':cid,'case_json_path':str(casepath),'case_json_sha256':sha(casepath),
            'candidates_json_path':str(package/'candidates.json'),'candidates_json_sha256':sha(package/'candidates.json'),
            'candidate_mask_files':mask_files,'GT_mask_files':[{'path':str(casepath.parent/g['mask_crop']),
                'sha256':sha(casepath.parent/g['mask_crop'])} for g in gt]})
        if case_number%50==0: print('Extracted',case_number,'of226',flush=True)
    assert len(all_rows)==1951 and len(reference_rows)==990
    assert sum(r['evaluation_included'] for r in all_rows)==1830
    assert len(provenance)==len(all_rows)
    write_csv(OUT/'candidates_all_1951.csv',all_rows)
    write_csv(OUT/'candidates_evaluation_1830.csv',[r for r in all_rows if r['evaluation_included']])
    write_csv(OUT/'references_990.csv',reference_rows)
    write(OUT/'case_input_bindings.json',input_receipts)
    receipt={'status':'PASS','protocol_sha256':sha(OUT/'protocol_frozen.json'),
        'script_sha256':sha(__file__),'cases':len(cases),'candidates':len(all_rows),'references':len(reference_rows),
        'evaluation_included':sum(r['evaluation_included'] for r in all_rows),
        'evaluation_excluded':sum(not r['evaluation_included'] for r in all_rows),
        'primary_semantic_eligible':sum(r['primary_semantic_analysis_included'] for r in all_rows),
        'all_candidates_without_same_semantic_gt':sum(r['same_semantic_gt_count']==0 for r in all_rows),
        'evaluation_included_without_same_semantic_gt':sum(r['evaluation_included'] and not r['same_semantic_gt_count'] for r in all_rows),
        'all_weights_equal_frozen_provenance':True,'all_990_reverse_best_iou_tracks_equal_stage_input_pool':True,
        'source_files_modified':False,'GPU_used':False,'model_execution':False,
        'artifacts_sha256':{f:sha(OUT/f) for f in ['candidates_all_1951.csv','candidates_evaluation_1830.csv','references_990.csv','case_input_bindings.json']}}
    validate_protocol()
    write(OUT/'extraction_receipt.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='artifacts_sha256'},indent=2),flush=True)


def interval(values,valid):
    values=np.asarray(values)[valid]
    if not len(values):return None,None,0
    lo,hi=np.quantile(values,[.025,.975],method='linear')
    return float(lo),float(hi),len(values)


def analyze():
    protocol=validate_protocol()
    extraction=read(OUT/'extraction_receipt.json')
    assert extraction['status']=='PASS' and extraction['protocol_sha256']==sha(OUT/'protocol_frozen.json')
    for name,digest in extraction['artifacts_sha256'].items():assert sha(OUT/name)==digest
    assert not (OUT/'analysis_receipt.json').exists(),'Refuse to replace completed statistics'
    rr=rows(OUT/'candidates_all_1951.csv')
    categories=sorted({r['object_category'] for r in rr}); assert len(categories)==60
    families=sorted({r['family'] for r in rr}); cat_index={c:i for i,c in enumerate(categories)}
    rng=np.random.default_rng(20261010)
    draws=rng.integers(len(categories),size=(10000,len(categories)))
    weights=np.zeros((10000,len(categories)),dtype=np.int64)
    for i,d in enumerate(draws):weights[i]=np.bincount(d,minlength=len(categories))
    np.save(OUT/'bootstrap_category_draw_indices.npy',draws,allow_pickle=False)
    np.save(OUT/'bootstrap_category_multiplicities.npy',weights,allow_pickle=False)
    populations={'primary_conditional_semantic':[r for r in rr if r['primary_semantic_analysis_included']=='1'],
                 'secondary_evaluation_geometry':[r for r in rr if r['evaluation_included']=='1']}
    summary_rows,bin_rows=[],[]
    for pop,pr in populations.items():
        prefix='semantic' if pop.startswith('primary') else 'geometric'
        for family in ['ALL']+families:
            selected=pr if family=='ALL' else [r for r in pr if r['family']==family]
            n=len(selected); cids={r['case_id'] for r in selected}; cats={r['object_category'] for r in selected}
            group_base={'population':pop,'family':family,'candidate_n':n,'case_n':len(cids),'category_n':len(cats),
                        'sparse_fewer_than_10_categories':int(len(cats)<10), 'sparse_fewer_than_30_candidates':int(n<30)}
            ids=np.array([cat_index[r['object_category']] for r in selected],dtype=int)
            ww=np.array([float(r['w']) for r in selected],dtype=float)
            bb=np.minimum(9,np.floor(ww*10).astype(int))
            assert np.all((bb>=0)&(bb<=9))
            count=np.bincount(ids,minlength=60).astype(float)
            sumw=np.bincount(ids,weights=ww,minlength=60)
            dn=weights@count; valid=dn>0
            meanw=np.divide(weights@sumw,dn,out=np.zeros(10000),where=valid)
            for suffix in ['025','050']:
                yy=np.array([int(r[prefix+'_hit_'+suffix]) for r in selected],dtype=float)
                hits=np.bincount(ids,weights=yy,minlength=60)
                briers=np.bincount(ids,weights=(ww-yy)**2,minlength=60)
                matched=np.divide(weights@hits,dn,out=np.zeros(10000),where=valid)
                brier=np.divide(weights@briers,dn,out=np.zeros(10000),where=valid)
                point_ece_numerator=0.0; boot_ece_numerator=np.zeros(10000)
                for bn in range(10):
                    ix=bb==bn; bn_n=int(ix.sum())
                    bn_rows=[r for r,b in zip(selected,bb) if b==bn]
                    bcount=np.bincount(ids[ix],minlength=60).astype(float)
                    bsumw=np.bincount(ids[ix],weights=ww[ix],minlength=60)
                    bhits=np.bincount(ids[ix],weights=yy[ix],minlength=60)
                    bdn=weights@bcount; bv=bdn>0
                    bw=np.divide(weights@bsumw,bdn,out=np.zeros(10000),where=bv)
                    by=np.divide(weights@bhits,bdn,out=np.zeros(10000),where=bv)
                    wlo,whi,bvalid=interval(bw,bv); ylo,yhi,_=interval(by,bv)
                    boot_ece_numerator+=np.abs(weights@bhits-weights@bsumw)
                    point_ece_numerator+=abs(float(bhits.sum()-bsumw.sum()))
                    bin_rows.append({'population':pop,'family':family,'iou_threshold':int(suffix)/100,
                        'bin_index':bn,'bin_lower':bn/10,'bin_upper':(bn+1)/10,'upper_inclusive':int(bn==9),
                        'candidate_n':bn_n,'case_n':len({r['case_id'] for r in bn_rows}),
                        'category_n':len({r['object_category'] for r in bn_rows}),
                        'sum_w':float(bsumw.sum()),'matched_n':int(bhits.sum()),
                        'mean_w':float(bsumw.sum()/bn_n) if bn_n else None,
                        'mean_w_ci95_low':wlo,'mean_w_ci95_high':whi,
                        'matched_fraction':float(bhits.sum()/bn_n) if bn_n else None,
                        'matched_fraction_ci95_low':ylo,'matched_fraction_ci95_high':yhi,
                        'valid_bootstrap_draws':bvalid,'zero_denominator_draws':10000-bvalid,
                        'empty_original_bin':int(not bn_n)})
                ece=np.divide(boot_ece_numerator,dn,out=np.zeros(10000),where=valid)
                sr={**group_base,'iou_threshold':int(suffix)/100,'sum_w':float(sumw.sum()),'matched_n':int(hits.sum()),
                    'Brier_sum_squared_error':float(briers.sum()),'ECE_weighted_absolute_gap_numerator':point_ece_numerator,
                    'valid_bootstrap_draws':int(valid.sum()),'zero_denominator_draws':int((~valid).sum())}
                for name,value,samples in [('mean_w',float(sumw.sum()/n) if n else None,meanw),
                    ('matched_fraction',float(hits.sum()/n) if n else None,matched),
                    ('Brier_if_weight_interpreted_as_probability',float(briers.sum()/n) if n else None,brier),
                    ('ECE10_if_weight_interpreted_as_probability',point_ece_numerator/n if n else None,ece)]:
                    lo,hi,_=interval(samples,valid)
                    sr[name]=value;sr[name+'_ci95_low']=lo;sr[name+'_ci95_high']=hi
                summary_rows.append(sr)
    write_csv(OUT/'family_empirical_reliability.csv',summary_rows)
    write_csv(OUT/'fixed10bin_empirical_reliability.csv',bin_rows)
    assert all(sum(r['candidate_n'] for r in bin_rows if r['population']==s['population'] and r['family']==s['family'] and r['iou_threshold']==s['iou_threshold'])==s['candidate_n'] for s in summary_rows)
    assert len(bin_rows)==len(summary_rows)*10
    data={'status':'PASS','protocol_sha256':sha(OUT/'protocol_frozen.json'),
        'extraction_receipt_sha256':sha(OUT/'extraction_receipt.json'),'script_sha256':sha(__file__),
        'all_candidate_count':len(rr),'evaluation_candidate_count':len(populations['secondary_evaluation_geometry']),
        'conditional_semantic_candidate_count':len(populations['primary_conditional_semantic']),
        'families':families,'family_count':len(families),'summary_rows':len(summary_rows),'bin_rows':len(bin_rows),
        'all_empty_bins_retained':True,'bootstrap':{'category_order':categories,'seed':20261010,'draws':10000,
            'draw_index_dtype':str(draws.dtype),'draw_indices_file_sha256':sha(OUT/'bootstrap_category_draw_indices.npy'),
            'draw_multiplicities_file_sha256':sha(OUT/'bootstrap_category_multiplicities.npy'),
            'draw_indices_array_sha256':hashlib.sha256(draws.tobytes()).hexdigest(),
            'candidate_weighted_ratio':True,'all_categories_each_draw_with_replacement':60},
        'pooled_results':[r for r in summary_rows if r['family']=='ALL'],
        'interpretation_limits':[
            'Empirical diagnostic reuse of accepted proposals from an existing evaluated cohort; not independent validation.',
            'Weights are heuristics, not probability predictions. Brier/ECE quantify mismatch only under that hypothetical interpretation; they do not establish calibrated reliability.',
            'Primary semantic results condition on an annotated same-semantic GT part and the original evaluation inclusion rule. Missing annotations are not negative labels.',
            'Geometry success means overlap with any annotated GT part, not correct semantic naming or one-to-one instance matching.',
            'Candidate-weighted means and hit fractions preserve repeated/overlapping proposals. Bootstrap clusters object category and retains all candidates; no iid-candidate inference.',
            'Family names are recorded grouping keys, not guarantees of independent source models. Shared models and derived scores create dependence.',
            'Small family/category support and empty-draw exclusions are reported; intervals are pointwise exploratory and conditional on nonempty denominator.',
            'No models rerun, no confidence threshold selected, no weight fitted or recalibrated, no frozen input edited.']}
    validate_protocol()
    write(OUT/'analysis_summary.json',data)
    write(OUT/'analysis_receipt.json',{'status':'COMPLETE','created_utc':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':sha(OUT/'protocol_frozen.json'),'input_sha256':protocol['input_sha256'],
        'artifacts_sha256':{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='analysis_receipt.json'}})
    print(json.dumps({'status':'PASS','all':len(rr),'primary':len(populations['primary_conditional_semantic']),
        'secondary':len(populations['secondary_evaluation_geometry']),'pooled':data['pooled_results'],
        'receipt_sha256':sha(OUT/'analysis_receipt.json')},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','extract','analyze'])
    {'freeze':freeze,'extract':extract,'analyze':analyze}[parser.parse_args().command]()
