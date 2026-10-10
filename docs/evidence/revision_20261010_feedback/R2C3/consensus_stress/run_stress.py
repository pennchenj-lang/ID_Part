"""Frozen Eq.2 boundary stress test; never changes production sources or Word."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BUNDLE = ROOT / 'hpid-publication-20261009/docs/evidence/revision_20261008'
RUNTIME = Path('__RUNTIME_ROOT__')
os.environ['HPID_DATA_ROOT'] = str(RUNTIME)
os.environ['HPID_RESULTS_DIR'] = str(HERE / 'unused_import_output')
sys.dont_write_bytecode = True
sys.path.insert(0, str(BUNDLE / 'code/analysis'))
import stage_diagnosis as sd
import numpy as np
from PIL import Image
fusion = sd.fusion
MODES = ['family_max_noisy_or', 'candidate_noisy_or', 'all_max', 'family_max_uniform_r']
LEVELS = [.5, .75, 1.0]
SCENARIOS = ['single', 'same_family_5', 'cross_family_5']
METRICS = ['wrong_gt_pairwise_dominance', 'wrong_top_nonroot', 'correct_top_nonroot', 'wrong_evidence', 'correct_evidence']
SEED = 20261010

def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fields); writer.writeheader(); writer.writerows(rows)

def weight(c, uniform=False):
    r = 1.0 if uniform else c.source_reliability
    w = float(np.clip((.45+.55*float(np.clip(c.score,0,1)))*r, .01,.995))
    if fusion._is_broad_scene_layer(c):
        w = float(np.clip(w*fusion.FusionConfig().scene_layer_fallback_weight,.01,.995))
    return w

class Captured(Exception):
    pass

def capture(candidates):
    data = {}
    previous = fusion._parent_support
    def intercept(evidence, taxonomy, accepted, name_to_id):
        data.update(evidence=evidence.copy(), taxonomy=taxonomy, accepted=list(accepted), name_to_id=dict(name_to_id))
        raise Captured()
    fusion._parent_support = intercept
    try:
        fusion.fuse_candidates(candidates, config=fusion.FusionConfig())
    except Captured:
        pass
    finally:
        fusion._parent_support = previous
    assert data
    return data

def aggregate(entries, nclasses, npoints, mode):
    out = np.zeros((nclasses,npoints), np.float32)
    if mode == 'all_max':
        for j,f,q,u in entries: out[j] = np.maximum(out[j],q)
    elif mode == 'candidate_noisy_or':
        for j,f,q,u in entries: out[j] = fusion._combine_noisy_or(out[j],q)
    else:
        groups = {}
        for j,f,q,u in entries:
            p = u if mode == 'family_max_uniform_r' else q
            groups[j,f] = np.maximum(groups[j,f],p) if (j,f) in groups else p
        for (j,f),p in groups.items(): out[j] = fusion._combine_noisy_or(out[j],p)
    return out

def worker(raw, manifest, families):
    cid = raw['case_id']; package = sd.BENCH / cid
    candidates = sd._load_candidates(package)
    before = [sd.mask_sha(c.mask) for c in candidates]
    data = capture(candidates); accepted = data['accepted']; tax = data['taxonomy']; names = data['name_to_id']
    assert len(accepted)==len(candidates), (cid,'unexpected refilter')
    casepath = Path(manifest['case_path']); case = sd.read(casepath)
    domain = raw['expected_domain']; category = case['object_category']
    truths = [np.asarray(Image.open(casepath.parent/r['mask_crop']).convert('L'))>=128 for r in case['parts']]
    truth_names = [sd._normalize(r['part_name'],domain,object_category=category) for r in case['parts']]
    norm = {j:sd.normalized(n,domain,category) for j,n in enumerate(tax.fine_names) if j>0}
    child_ids = [j for j,n in enumerate(tax.fine_names) if j>0 and n!=tax.parent_names[tax.fine_to_parent[j]]]
    soft = [fusion._soft_membership(c.mask) for c in accepted]
    full = [(names[c.semantic_name],fusion._source_family(c),(m*weight(c)).ravel(),(m*weight(c,True)).ravel()) for c,m in zip(accepted,soft)]
    replay = aggregate(full,len(tax.fine_names),data['evidence'].shape[1]*data['evidence'].shape[2],MODES[0]).reshape(data['evidence'].shape)
    error = float(np.max(np.abs(replay-data['evidence'])))
    assert error==0.0, (cid,error)
    agreements,_=fusion._source_agreement_factors(accepted,fusion.FusionConfig())
    assert all(a==1.0 for a in agreements)
    audit = {'case_id':cid,'category':category,'candidate_count':len(candidates),'consensus_max_abs_difference':error,
             'candidates_json_sha256':sha(package/'candidates.json'),'case_json_sha256':sha(casepath),
             'candidate_masks_sha256':hashlib.sha256(''.join(before).encode()).hexdigest(),
             'reference_masks_sha256':hashlib.sha256(''.join(sd.mask_sha(m) for m in truths).encode()).hexdigest()}
    target=None
    # Select the first annotated reference with an unambiguous >=20 pixel ROI
    # and two differently named existing nonroot taxonomy classes. No output
    # score, prediction, correctness, or dominance is used in selection.
    for ri,(truth,tn) in enumerate(zip(truths,truth_names)):
        true_ids=[j for j in child_ids if norm[j]==tn]
        wrong_ids=[j for j in child_ids if norm[j]!=tn and norm[j]!='body']
        if not true_ids or not wrong_ids: continue
        roi=truth.copy()
        for other,on in zip(truths,truth_names):
            if on!=tn and on!='body': roi &= ~other
        if int(roi.sum())<20: continue
        wrong_id=min(wrong_ids,key=lambda j:tax.fine_names[j])
        # Explicitly assert zero annotation overlap for the injected label.
        for other,on in zip(truths,truth_names):
            if on==norm[wrong_id]: assert not np.any(other & roi)
        target=(ri,tn,true_ids,wrong_id,roi); break
    if target is None:
        audit.update(eligible=False,exclusion='No annotated reference with >=20 unambiguous pixels and two distinct normalized nonroot classes.')
        return [],audit
    ri,tn,true_ids,wrong_id,roi=target
    wrong_ids=[j for j in child_ids if norm[j]==norm[wrong_id]]
    wrong_name=tax.fine_names[wrong_id]
    base_families=sorted({fusion._source_family(c) for c in accepted if c.semantic_name==wrong_name})
    assert base_families
    base_family=base_families[0]
    cross_families=[base_family]+[f for f in families if f!=base_family][:4]
    assert len(set(cross_families))==5
    entries=[(j,f,q.reshape(roi.shape)[roi],u.reshape(roi.shape)[roi]) for j,f,q,u in full]
    membership=fusion._soft_membership(roi)[roi]
    audit.update(eligible=True,reference_index=ri,reference_name=tn,wrong_name=wrong_name,wrong_normalized=norm[wrong_id],
                 target_pixels=int(roi.sum()),target_mask_sha256=sd.mask_sha(roi),base_family=base_family,cross_families=json.dumps(cross_families),
                 same_family_max_difference=0.0)
    def measure(e):
        wrong=e[wrong_ids].max(axis=0); correct=e[true_ids].max(axis=0)
        winners=np.array(child_ids)[e[child_ids].argmax(axis=0)]
        return {'wrong_gt_pairwise_dominance':float(np.mean(wrong>correct)),
                'wrong_top_nonroot':float(np.mean(np.isin(winners,wrong_ids))),
                'correct_top_nonroot':float(np.mean(np.isin(winners,true_ids))),
                'wrong_evidence':float(wrong.mean()),'correct_evidence':float(correct.mean())}
    results=[]; common={'case_id':cid,'category':category,'reference_index':ri,'reference_name':tn,'wrong_name':wrong_name,'target_pixels':int(roi.sum())}
    for mode in MODES:
        clean=measure(aggregate(entries,len(tax.fine_names),int(roi.sum()),mode))
        results.append({**common,'mode':mode,'r_level':'clean','scenario':'clean',**clean,**{'delta_clean_'+k:0.0 for k in METRICS}})
        for r in LEVELS:
            q=membership*float(np.clip(r,.01,.995)); u=membership*.995
            single=None; repeated=None
            for scenario in SCENARIOS:
                fs=[base_family] if scenario=='single' else [base_family]*5 if scenario=='same_family_5' else cross_families
                injected=[(wrong_id,f,q,u) for f in fs]
                e=aggregate(entries+injected,len(tax.fine_names),int(roi.sum()),mode)
                if scenario=='single': single=e.copy()
                if scenario=='same_family_5': repeated=e.copy()
                if scenario=='cross_family_5' and mode=='candidate_noisy_or':
                    assert np.array_equal(e,repeated),(cid,'ungrouped family-label dependence')
                if scenario=='same_family_5' and mode!= 'candidate_noisy_or':
                    diff=float(np.max(np.abs(e-single))); assert diff==0,(cid,mode,diff)
                values=measure(e)
                results.append({**common,'mode':mode,'r_level':r,'scenario':scenario,**values,
                                **{'delta_clean_'+k:values[k]-clean[k] for k in METRICS}})
    assert before==[sd.mask_sha(c.mask) for c in candidates]
    return results,audit

def summarize(rows,audits):
    eligible=[a for a in audits if a['eligible']]
    ids=sorted(a['case_id'] for a in eligible); cats={a['case_id']:a['category'] for a in eligible}
    allcats=sorted({a['category'] for a in audits})
    cluster=np.array([allcats.index(cats[c]) for c in ids])
    rng=np.random.default_rng(SEED)
    draws=rng.integers(len(allcats),size=(10000,len(allcats)))
    counts=np.stack([np.bincount(d,minlength=len(allcats)) for d in draws])
    W=counts[:,cluster]; denom=W.sum(axis=1); valid=denom>0
    def stats(v):
        v=np.array(v,float); means=(W[valid]@v)/denom[valid]
        return {'mean':float(v.mean()),'ci95_low':float(np.quantile(means,.025)),'ci95_high':float(np.quantile(means,.975))}
    index={(r['case_id'],r['mode'],str(r['r_level']),r['scenario']):r for r in rows}
    summary=[]; comparisons=[]
    for mode in MODES:
        for r,scenario in [('clean','clean')]+[(str(r),s) for r in LEVELS for s in SCENARIOS]:
            for metric in METRICS:
                for kind in [metric,'delta_clean_'+metric]:
                    summary.append({'mode':mode,'r_level':r,'scenario':scenario,'metric':kind,'n':len(ids),**stats([index[c,mode,r,scenario][kind] for c in ids])})
        for r in LEVELS:
            for scenario in SCENARIOS[1:]:
                for metric in METRICS:
                    comparisons.append({'contrast':scenario+' minus single','mode':mode,'r_level':r,'metric':metric,'n':len(ids),
                                        **stats([index[c,mode,str(r),scenario][metric]-index[c,mode,str(r),'single'][metric] for c in ids])})
    for r in LEVELS:
        for scenario in SCENARIOS:
            for metric in METRICS:
                comparisons.append({'contrast':'candidate_noisy_or minus family_max_noisy_or','mode':'paired_methods','r_level':r,'scenario':scenario,'metric':metric,'n':len(ids),
                    **stats([index[c,'candidate_noisy_or',str(r),scenario][metric]-index[c,'family_max_noisy_or',str(r),scenario][metric] for c in ids])})
    write_csv(HERE/'summary.csv',summary); write_csv(HERE/'paired_comparisons.csv',comparisons)
    return {'case_count':len(audits),'eligible_case_count':len(ids),'eligible_category_count':len(set(cats.values())),
            'total_category_count':len(allcats),'total_candidates':sum(a['candidate_count'] for a in audits),'rows':len(rows),
            'all_226_consensus_arrays_bit_exact':all(a['consensus_max_abs_difference']==0 for a in audits),
            'same_family_repetition_invariant':True,'protocol_sha256':sha(HERE/'protocol.json')}

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=2);args=p.parse_args()
    bench=sd.read(sd.BENCH/'benchmark_summary.json');manifest={r['case_id']:r for r in sd.read(bench['source_manifest'])['cases']}
    cases=[r for r in bench['cases'] if r['return_code']==0];assert len(cases)==226
    families=sorted(r['family'] for r in sd.csv_read(BUNDLE/'evidence/score_family_ranges.csv'));assert len(families)==9
    protocol={'version':1,'frozen_utc':datetime.now(timezone.utc).isoformat(),'cohort':'All 226 existing cases; 1951 archived accepted candidates; diagnostic reuse, not untouched validation.',
      'intervention_boundary':'After candidate eligibility filtering and soft membership, at Eq.2 consensus before parent support. No downstream inference or final masks are claimed.',
      'capture':'Temporarily intercept frozen _parent_support, copy evidence and accepted inputs, abort before downstream operations; no production source edits. Reconstruct entire released consensus array and assert bit equality in every case.',
      'selection':'One target per eligible case, first reference in stored order with >=20 pixels outside all differently named non-body annotations and represented by a nonroot normalized class; injected class lexicographically first existing nonroot class with different normalized name excluding body. Root/background excluded. No prediction used in selection.',
      'injection':'Synthetic semantic conflict with perfect support on selected annotated ROI; s=1, a=1, r in [0.5,0.75,1] (w capped .995). The same soft membership and taxonomy are fixed for all conditions. These are controlled stress strengths, not fitted family estimates.',
      'families':'Base family is lexicographically first recorded family already producing injected class. Single is one copy; same_family_5 is identical single plus four exact copies; cross_family_5 is identical single plus four copies under first four other recorded global family keys. Group labels do not imply statistically independent producers.',
      'modes':MODES,'conditions':['clean']+SCENARIOS,'r_levels':LEVELS,'family_keys':families,
      'uniform_r':'All natural and injected candidates have r=1 in this mode, other score transformation fixed. Consequently its injected weight is .995 at all stress levels.',
      'endpoints':'For each ROI compute wrong-vs-correct evidence dominance (> tie loses); wrong and correct top among existing nonroot classes (argmax lowest taxonomy class id tie); mean wrong/correct evidence. Correct/wrong normalized aliases pooled by max. These are Eq.2 readouts, not final semantic predictions or real false-positive prevalence.',
      'statistics':'Case-macro absolute and paired changes vs clean, 10000 category-cluster percentile bootstrap draws sampling all 60 original categories with replacement, seed 20261010; only eligible cases contribute. Repeated scenarios/pixels are not independent samples. Pointwise exploratory 95% CIs, no multiplicity-adjusted superiority hypothesis tests.',
      'exclusions':'Ineligible cases retained with explicit reason. No outcome-driven exclusions.',
      'input_manifest_sha256':sha(bench['source_manifest']),'fusion_sha256':sha(sd.FUSION),'helper_sha256':sha(Path(sd.__file__)),'script_sha256':sha(__file__)}
    if (HERE/'protocol.json').exists(): raise RuntimeError('Existing frozen protocol: use a new version directory, do not overwrite')
    dump(HERE/'protocol.json',protocol);dump(HERE/'freeze_receipt.json',{'protocol_sha256':sha(HERE/'protocol.json'),'status':'FROZEN_BEFORE_EXECUTION'})
    rows=[];audits=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(worker,r,manifest[r['case_id']],families):r['case_id'] for r in cases}
        for n,job in enumerate(as_completed(jobs),1):
            result,audit=job.result();rows.extend(result);audits.append(audit)
            print(f'consensus {n}/{len(cases)} {jobs[job]} eligible={audit["eligible"]}',flush=True)
    rows.sort(key=lambda r:(r['case_id'],r['mode'],str(r['r_level']),r['scenario']));audits.sort(key=lambda r:r['case_id'])
    write_csv(HERE/'case_results.csv',rows);write_csv(HERE/'case_audit.csv',audits)
    report=summarize(rows,audits);report['finished_utc']=datetime.now(timezone.utc).isoformat()
    dump(HERE/'report.json',report);dump(HERE/'completion_receipt.json',{'status':'COMPLETE','files':{p.name:sha(p) for p in HERE.iterdir() if p.is_file() and p.name!='completion_receipt.json'}})
    print(json.dumps(report),flush=True)

if __name__=='__main__': main()
