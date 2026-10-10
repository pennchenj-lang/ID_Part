"""Frozen R6 source-image paired analysis; never reads masks or changes routes.

Execute only after protocol/code/manifest freeze. One target per source image.
The CLI requires --input-csv --manifest --output and retains all methods/cases.
"""
from __future__ import annotations
import argparse
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys
import numpy as np

METHODS = ('raw_proposals', 'greedy_nms', 'dbscan_fusion', 'local_pairwise_crf', 'hpid_split_group_ids')
HPID, RAW = METHODS[-1], METHODS[0]
STATES = ('correct_unique', 'wrong_unique', 'ambiguous', 'unresolved')
THRESHOLDS = ('025', '050')
SEED, DRAWS = 20261009, 10000
BINARY = ('unique_and_correct', 'wrong_unique', 'semantic_correct_any_match',
          'geometry_correct_any_exposed', 'geometry_correct_any_exported')
PAIR_METRICS = BINARY + ('query_ambiguous', 'query_unresolved')
SUCCESS = {'', '0', 'false', 'none', 'null', 'ok', 'pass', 'passed', 'success',
           'successful', 'complete', 'completed', 'valid'}

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def number(value):
    if str(value).strip().lower() in {'true', 'false'}:
        return float(str(value).strip().lower() == 'true')
    answer = float(value)
    if not math.isfinite(answer):
        raise ValueError(f'Nonfinite numeric input: {value}')
    return answer

def cohort_name(value):
    key = str(value).lower().replace('_', '').replace('-', '')
    if key in {'old', 'old37', 'legacy', 'legacy37', 'original', 'original37'}:
        return 'old37'
    if key in {'new', 'new216', 'added', 'additional', 'additional216'}:
        return 'new216'
    raise ValueError(f'Unknown cohort: {value}')

def source_id(row):
    for key in ('source_image_id', 'image_id', 'source_full_sha256', 'source_image_sha256'):
        if row.get(key) is not None and str(row[key]) != '':
            return str(row[key])
    raise ValueError('Manifest requires source_image_id or image_id')

def wilson(k, n, level=0.95):
    if n == 0:
        return {'n': 0, 'k': int(k), 'proportion': None, 'ci95_low': None, 'ci95_high': None}
    if not 0 <= k <= n:
        raise ValueError((k, n))
    z = NormalDist().inv_cdf((1 + level) / 2)
    p, denom = k / n, 1 + z*z/n
    center = (p + z*z/(2*n))/denom
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denom
    return {'n': int(n), 'k': int(k), 'proportion': p,
            'ci95_low': max(0., center-half), 'ci95_high': min(1., center+half)}

def percentile(values, adjusted=False):
    q = [0.00625, 0.99375] if adjusted else [0.025, 0.975]
    out = np.quantile(values, q, method='linear')
    return [float(out[0]), float(out[1])]

def mcnemar_exact(b, c):
    n = int(b+c)
    return 1. if n == 0 else min(1., 2*sum(math.comb(n, j) for j in range(min(b,c)+1))/(2**n))

def metric_value(row, metric, threshold):
    return row[metric] if metric.startswith('query_') else row[f'{metric}_{threshold}']

def make_draws(ids, units):
    """Preserve the historical candidate-pool strata at their fixed sizes."""
    rng = np.random.default_rng(SEED)
    parts=[]
    for cohort in ('old37','new216'):
        positions=np.array([i for i,rid in enumerate(ids) if units[rid]['cohort']==cohort],dtype=np.int64)
        if len(positions):
            parts.append(positions[rng.integers(0,len(positions),size=(DRAWS,len(positions)))])
    return np.concatenate(parts,axis=1)

def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def state(row, threshold):
    if row['execution_failure']:
        return 'unresolved'
    if row['query_ambiguous']:
        return 'ambiguous'
    if row['query_unresolved']:
        return 'unresolved'
    return 'correct_unique' if row[f'unique_and_correct_{threshold}'] else 'wrong_unique'

def load_data(input_csv, manifest_path):
    manifest = read_json(manifest_path)
    items = manifest if isinstance(manifest, list) else manifest.get('requests', manifest.get('cases'))
    if not isinstance(items, list):
        raise ValueError('Manifest must contain requests or cases list')
    units = {}
    for item in items:
        rid = str(item['request_id'])
        if rid in units:
            raise ValueError(f'Duplicate manifest request_id {rid}')
        units[rid] = {**item, 'request_id': rid, 'cohort': cohort_name(item['cohort']),
                      'source_image_id': source_id(item)}
    if len(units) != 253 or len({u['source_image_id'] for u in units.values()}) != 253:
        raise ValueError('Primary manifest must contain 253 unique requests/source images')
    if sum(u['cohort']=='old37' for u in units.values()) != 37 or sum(u['cohort']=='new216' for u in units.values()) != 216:
        raise ValueError('Expected exact old37 plus new216 cohorts')
    for key in ('source_full_sha256', 'source_image_sha256'):
        hashes = [u[key] for u in units.values() if u.get(key)]
        if hashes and len(hashes) != len(set(hashes)):
            raise ValueError(f'Duplicate canonical full image hash in {key}')
    with Path(input_csv).open(encoding='utf-8-sig', newline='') as f:
        csv_rows = list(csv.DictReader(f))
    numeric = ['query_match_count','all_exported_count','review_option_count','query_unique','query_ambiguous','query_unresolved',
               'selected_part_iou','best_review_match_iou','best_exposed_iou','best_exported_iou']
    numeric += [f'{metric}_{t}' for t in THRESHOLDS for metric in BINARY + ('correct_in_review_set',)]
    data, failures = {}, []
    for original in csv_rows:
        rid, method = original['request_id'], original['method']
        if rid not in units or method not in METHODS or (rid, method) in data:
            raise ValueError(f'Unknown/duplicate request-method: {rid} {method}')
        unit = units[rid]
        if cohort_name(original['cohort']) != unit['cohort']:
            raise ValueError(f'Cohort mismatch {rid}')
        row = dict(original)
        for key in ('domain', 'object_category'):
            a, b = str(original.get(key, '')).strip(), str(unit.get(key, '')).strip()
            if a and b and a != b:
                raise ValueError(f'Metadata mismatch: {rid} {key}')
            row[key] = a or b or 'unknown'
            if not unit.get(key):
                unit[key] = row[key]
        if original.get('image_id') and str(original['image_id']) != str(unit.get('image_id', original['image_id'])):
            raise ValueError(f'image_id mismatch {rid}')
        flags = {key:str(original.get(key, '')).strip().lower() for key in ('failstatus','execution_status','status') if key in original}
        failed = any(value not in SUCCESS for value in flags.values())
        row.update(request_id=rid, cohort=unit['cohort'], source_image_id=unit['source_image_id'],
                   execution_failure=int(failed), status_fields=json.dumps(flags, sort_keys=True))
        for key in numeric:
            if key not in row or row[key] == '':
                if failed:
                    row[key] = 0.
                else:
                    raise ValueError(f'Missing {key}: {rid} {method}')
            else:
                row[key] = number(row[key])
        if failed:
            failures.append({'request_id':rid,'cohort':unit['cohort'],'method':method,'status_fields':row['status_fields'],
                             'reported_review_option_count':row['review_option_count'],'error':original.get('error','')})
            for key in numeric:
                row[key] = 0.
            row['query_unresolved'] = 1.
        if any(row[key] < 0 or row[key] != int(row[key]) for key in ['query_match_count','all_exported_count','review_option_count']):
            raise ValueError(f'Invalid count {rid} {method}')
        if not failed:
            for key in ('selected_part_iou','best_review_match_iou','best_exposed_iou','best_exported_iou'):
                if not 0 <= row[key] <= 1:
                    raise ValueError(f'Invalid IoU {rid} {method} {key}')
            nm = row['query_match_count']
            if [row['query_unique'], row['query_ambiguous'], row['query_unresolved']] != [int(nm==1),int(nm>1),int(nm==0)]:
                raise ValueError(f'Invalid routing partition {rid} {method}')
            exposed = nm if nm else row['all_exported_count']
            if row['review_option_count'] != exposed:
                raise ValueError(f'Exposed option rule differs {rid} {method}')
            for threshold, value in [('025',.25),('050',.50)]:
                expected = {'unique_and_correct':int(nm==1 and row['selected_part_iou']>=value),
                            'wrong_unique':int(nm==1 and row['selected_part_iou']<value),
                            'semantic_correct_any_match':int(nm>0 and row['best_review_match_iou']>=value),
                            'geometry_correct_any_exposed':int(exposed>0 and row['best_exposed_iou']>=value),
                            'geometry_correct_any_exported':int(row['all_exported_count']>0 and row['best_exported_iou']>=value),
                            'correct_in_review_set':int(nm>1 and row['best_review_match_iou']>=value)}
                for metric, target in expected.items():
                    if row[f'{metric}_{threshold}'] != target:
                        raise ValueError(f'Metric/IoU disagreement: {rid} {method} {metric}_{threshold}')
        data[rid,method] = row
    if len(data) != 253*5 or any((rid,m) not in data for rid in units for m in METHODS):
        raise ValueError('Every one of 253 requests requires all five method records, including explicit failures')
    return units, data, failures

def analyze(input_csv, manifest_path, output):
    output = Path(output)
    if (output/'analysis_summary.json').exists():
        raise FileExistsError('Use a fresh analysis output directory to preserve previous evidence')
    units, data, failures = load_data(input_csv, manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    tables = {name:[] for name in ['routing_states_with_wilson','binary_guardrails_with_wilson','conditional_review_rates','method_option_distribution',
                                   'paired_option_contrasts_all_four','paired_correctness_guardrails','route_transition_4x4',
                                   'raw_ambiguous_fixed_subset_flow','correct_option_retention_and_loss','domain_category_failure_summary',
                                   'wrong_unique_diagnostics','execution_failures']}
    tables['execution_failures'] = failures
    draw_receipts, reproduction = [], []
    order = sorted(units, key=lambda rid:(units[rid]['source_image_id'],rid))
    for cohort in ['old37','new216','combined']:
        ids = [rid for rid in order if cohort=='combined' or units[rid]['cohort']==cohort]
        n = len(ids)
        draws = make_draws(ids,units)
        draw_receipts.append({'cohort':cohort,'n':n,'ordered_request_ids':ids,'draw_index_dtype':str(draws.dtype),
                              'fixed_stratum_sizes':{s:sum(units[rid]['cohort']==s for rid in ids) for s in ('old37','new216')},
                              'draw_index_sha256':hashlib.sha256(draws.tobytes()).hexdigest()})
        arrays = {m:{metric:np.array([data[rid,m][metric] for rid in ids],dtype=float)
                     for metric in ['review_option_count','execution_failure']}
                  for m in METHODS}
        for method in METHODS:
            arrays[method].update({f'{metric}_{t}':np.array([metric_value(data[rid,method],metric,t) for rid in ids],dtype=float)
                                   for t in THRESHOLDS for metric in PAIR_METRICS})
        states = {(m,t):[state(data[rid,m],t) for rid in ids] for m in METHODS for t in THRESHOLDS}
        for method in METHODS:
            rows = [data[rid,method] for rid in ids]
            v = arrays[method]['review_option_count']
            tables['method_option_distribution'].append({'cohort':cohort,'method':method,'n':n,'mean':float(v.mean()),
                'median':float(np.median(v)),'q25':float(np.quantile(v,.25)),'q75':float(np.quantile(v,.75)),
                'minimum':float(v.min()),'maximum':float(v.max()),'empty_exposed_n':int((v==0).sum()),
                'execution_failure_n':int(arrays[method]['execution_failure'].sum()),
                'mean_all_exported_count':float(np.mean([r['all_exported_count'] for r in rows]))})
            for threshold in THRESHOLDS:
                for metric in BINARY:
                    tables['binary_guardrails_with_wilson'].append({'cohort':cohort,'method':method,'threshold':threshold,
                        'metric':metric,**wilson(int(arrays[method][f'{metric}_{threshold}'].sum()),n)})
                for s in STATES:
                    tables['routing_states_with_wilson'].append({'cohort':cohort,'method':method,'threshold':threshold,'state':s,
                        **wilson(states[method,threshold].count(s),n)})
                amb = sum(r['query_ambiguous'] for r in rows)
                correct = sum(r[f'correct_in_review_set_{threshold}'] for r in rows)
                unique = sum(r['query_unique'] for r in rows)
                for name,k,denom in [('correct_in_ambiguous_review',correct,amb),
                                     ('correct_among_unique_routes',sum(r[f'unique_and_correct_{threshold}'] for r in rows),unique),
                                     ('wrong_among_unique_routes',sum(r[f'wrong_unique_{threshold}'] for r in rows),unique)]:
                    tables['conditional_review_rates'].append({'cohort':cohort,'method':method,'threshold':threshold,
                                                               'metric':name,**wilson(int(k),int(denom))})
                if cohort=='old37' and threshold=='025' and method in (RAW,HPID):
                    observed=[states[method,threshold].count(s) for s in STATES]
                    expected=[9,5,13,10] if method==RAW else [14,6,6,11]
                    expected_conditional=[9,13] if method==RAW else [1,6]
                    reproduction.append({'method':method,'observed':observed,'expected':expected,
                                         'correct_in_review':[int(correct),int(amb)],'expected_correct_in_review':expected_conditional,
                                         'pass':observed==expected and [int(correct),int(amb)]==expected_conditional})
        for comparator in METHODS[:-1]:
            h=arrays[HPID]['review_option_count']; b=arrays[comparator]['review_option_count']
            d=h-b; boot=d[draws].mean(axis=1)
            lo,hi=percentile(boot); alo,ahi=percentile(boot,True)
            valid=(arrays[HPID]['execution_failure']==0)&(arrays[comparator]['execution_failure']==0)
            clean_n=int(valid.sum())
            option_row={'cohort':cohort,'comparator':comparator,'n':n,'hpid_mean':float(h.mean()),'comparator_mean':float(b.mean()),
                'mean_difference_hpid_minus_comparator':float(d.mean()),'relative_reduction_percent':float(100*(b.mean()-h.mean())/b.mean()) if b.mean() else None,
                'ci95_low':lo,'ci95_high':hi,'ci9875_low':alo,'ci9875_high':ahi,
                'hpid_execution_failure_n':int(arrays[HPID]['execution_failure'].sum()),'comparator_execution_failure_n':int(arrays[comparator]['execution_failure'].sum()),
                'all_case_compactness_superiority_claim_eligible':bool(valid.all()),
                'direction_supported_by_adjusted_interval':bool(ahi<0 and valid.all()),'paired_complete_n':clean_n,
                'paired_complete_mean_difference':float(d[valid].mean()) if clean_n else None,
                'paired_complete_ci95_low':None,'paired_complete_ci95_high':None,
                'family':'combined_four_comparisons_primary' if cohort=='combined' else 'separate_cohort_descriptive_four_comparisons'}
            if clean_n and not valid.all():
                cd=make_draws([rid for rid,ok in zip(ids,valid) if ok],units)
                clo,chi=percentile(d[valid][cd].mean(axis=1))
                option_row.update(paired_complete_ci95_low=clo,paired_complete_ci95_high=chi)
            tables['paired_option_contrasts_all_four'].append(option_row)
            for threshold in THRESHOLDS:
                for metric in PAIR_METRICS:
                    x=arrays[HPID][f'{metric}_{threshold}']; y=arrays[comparator][f'{metric}_{threshold}']
                    diff=x-y; bs=diff[draws].mean(axis=1); lo,hi=percentile(bs)
                    both=int(((x==1)&(y==1)).sum()); gain=int(((x==1)&(y==0)).sum())
                    loss=int(((x==0)&(y==1)).sum()); neither=n-both-gain-loss
                    tables['paired_correctness_guardrails'].append({'cohort':cohort,'comparator':comparator,'threshold':threshold,'metric':metric,'n':n,
                        'hpid_k':int(x.sum()),'comparator_k':int(y.sum()),'both_positive':both,'hpid_only_positive':gain,
                        'comparator_only_positive':loss,'both_negative':neither,'difference_percentage_points':float(100*diff.mean()),
                        'ci95_low_percentage_points':100*lo,'ci95_high_percentage_points':100*hi,
                        'bootstrap_degenerate':bool(np.ptp(bs)==0),'discordant_n':gain+loss,'mcnemar_exact_two_sided_p':mcnemar_exact(gain,loss),
                        'inference_scope':'descriptive_pointwise_no_equivalence_claim'})
                    if metric.startswith('geometry_correct') or metric=='semantic_correct_any_match':
                        for event,k,denom in [('retained',both,n),('lost',loss,n),('gained',gain,n),('neither',neither,n),
                                               ('loss_given_comparator_available',loss,both+loss),('gain_given_comparator_unavailable',gain,gain+neither)]:
                            tables['correct_option_retention_and_loss'].append({'cohort':cohort,'comparator':comparator,'threshold':threshold,
                                                                              'metric':metric,'event':event,**wilson(k,denom)})
                for from_state in STATES:
                    denom=states[comparator,threshold].count(from_state)
                    for to_state in STATES:
                        k=sum(a==from_state and b==to_state for a,b in zip(states[comparator,threshold],states[HPID,threshold]))
                        tables['route_transition_4x4'].append({'cohort':cohort,'comparator':comparator,'threshold':threshold,
                            'from_state':from_state,'to_hpid_state':to_state,'all_cohort_n':n,'all_cohort_proportion':k/n,**wilson(k,denom)})
        for threshold in THRESHOLDS:
            for subset in ['raw_ambiguous','raw_ambiguous_with_correct_option']:
                subset_ids=[rid for rid in ids if data[rid,RAW]['query_ambiguous'] and
                            (subset=='raw_ambiguous' or data[rid,RAW][f'correct_in_review_set_{threshold}'])]
                outcomes=['correct_unique','wrong_unique','ambiguous_with_correct','ambiguous_without_correct',
                          'unresolved_with_correct_fallback','unresolved_without_correct_fallback']
                counts={key:0 for key in outcomes}
                for rid in subset_ids:
                    r=data[rid,HPID]; s=state(r,threshold)
                    if s=='ambiguous': s='ambiguous_with_correct' if r[f'correct_in_review_set_{threshold}'] else 'ambiguous_without_correct'
                    elif s=='unresolved': s='unresolved_with_correct_fallback' if r[f'geometry_correct_any_exposed_{threshold}'] else 'unresolved_without_correct_fallback'
                    counts[s]+=1
                for outcome,k in counts.items():
                    tables['raw_ambiguous_fixed_subset_flow'].append({'cohort':cohort,'threshold':threshold,'subset':subset,
                                                                    'to_hpid_state':outcome,**wilson(k,len(subset_ids))})
            newly_ambiguous=sum(not data[rid,RAW]['query_ambiguous'] and data[rid,HPID]['query_ambiguous'] for rid in ids)
            tables['raw_ambiguous_fixed_subset_flow'].append({'cohort':cohort,'threshold':threshold,'subset':'all_requests',
                                                            'to_hpid_state':'new_hpid_ambiguous_not_raw_ambiguous',**wilson(newly_ambiguous,n)})
        for group_by in ('domain','object_category'):
            categories=sorted({str(units[rid].get(group_by) or 'unknown') for rid in ids})
            for group in categories:
                members=[rid for rid in ids if str(units[rid].get(group_by) or 'unknown')==group]
                gn=len(members)
                for method in METHODS:
                    fail_n=sum(data[rid,method]['execution_failure'] for rid in members)
                    tables['domain_category_failure_summary'].append({'cohort':cohort,'group_by':group_by,'group':group,
                        'source_images_n':gn,'method':method,'threshold':'all','metric':'execution_failure',
                        'sparse_n_lt_10':gn<10,**wilson(fail_n,gn)})
                    for threshold in THRESHOLDS:
                        for s in STATES:
                            k=sum(state(data[rid,method],threshold)==s for rid in members)
                            tables['domain_category_failure_summary'].append({'cohort':cohort,'group_by':group_by,'group':group,
                                'source_images_n':gn,'method':method,'threshold':threshold,'metric':s,'sparse_n_lt_10':gn<10,**wilson(k,gn)})
                        for metric in BINARY[2:]+('correct_in_review_set',):
                            k=sum(data[rid,method][f'{metric}_{threshold}'] for rid in members)
                            denom=sum(data[rid,method]['query_ambiguous'] for rid in members) if metric=='correct_in_review_set' else gn
                            tables['domain_category_failure_summary'].append({'cohort':cohort,'group_by':group_by,'group':group,
                                'source_images_n':gn,'method':method,'threshold':threshold,'metric':metric,'sparse_n_lt_10':gn<10,**wilson(int(k),int(denom))})
                        k=sum(data[rid,method]['query_ambiguous'] and not data[rid,method][f'correct_in_review_set_{threshold}'] for rid in members)
                        tables['domain_category_failure_summary'].append({'cohort':cohort,'group_by':group_by,'group':group,
                            'source_images_n':gn,'method':method,'threshold':threshold,'metric':'ambiguous_without_correct_per_all','sparse_n_lt_10':gn<10,**wilson(k,gn)})
        for rid in ids:
            for method in METHODS:
                r=data[rid,method]
                for threshold in THRESHOLDS:
                    if r[f'wrong_unique_{threshold}']:
                        tables['wrong_unique_diagnostics'].append({'cohort':cohort,'request_id':rid,'case_id':r.get('case_id',''),
                            'source_image_id':units[rid]['source_image_id'],'method':method,'threshold':threshold,'domain':r['domain'],
                            'object_category':r['object_category'],'target':r.get('target',''),'normalized_target':r.get('normalized_target',''),
                            'selected_identity':r.get('selected_identity',''),'selected_part_iou':r['selected_part_iou'],
                            'best_exposed_iou':r['best_exposed_iou'],'best_exported_iou':r['best_exported_iou'],
                            'correct_anywhere_exported':r[f'geometry_correct_any_exported_{threshold}'],
                            'raw_state':state(data[rid,RAW],threshold),'hpid_state':state(data[rid,HPID],threshold),
                            'attribution':'Not inferable from routing CSV alone; requires candidate lineage/stage trace'})
    for name, rows in tables.items():
        write_csv(output/(name+'.csv'),rows)
    summary={'status':'PASS' if all(r['pass'] for r in reproduction) else 'LEGACY_REPRODUCTION_MISMATCH',
             'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'counts':{'old37':37,'new216':216,'combined':253},
             'source_image_independent_n':253,'input_rows':len(data),'methods':list(METHODS),'execution_failure_rows':len(failures),
             'bindings':{'input_csv':str(Path(input_csv).resolve()),'input_csv_sha256':sha(input_csv),
                         'manifest':str(Path(manifest_path).resolve()),'manifest_sha256':sha(manifest_path),'analysis_script_sha256':sha(__file__)},
             'bootstrap':{'draws':DRAWS,'seed':SEED,'generator':'numpy.default_rng/PCG64','quantile_method':'linear',
                          'paired_unit':'source image; one request per image',
                          'stratification':'Fixed historical candidate pools old37/new216. Combined draws retain exactly 37 and 216 images with replacement within each stratum.',
                          'receipts':draw_receipts},
             'versions':{'python':sys.version,'numpy':np.__version__},'legacy37_validation':reproduction,
             'interpretation':['Wilson 95% is marginal, not simultaneous multinomial coverage.',
                'Four Bonferroni 98.75% count intervals define one family on the combined cohort; old/new tables are separate descriptive families.',
                'Other effects and McNemar p values are descriptive; matching counts or zero-discordance bootstrap is not equivalence.',
                'Failed outputs remain unresolved/zero availability and are listed; empty/failure counts must not be promoted as compactness gains.',
                'Correct-in-review conditions on a different ambiguous subset for each method; use paired availability and fixed raw-ambiguous flow for losses.',
                'wrong_among_unique_routes uses wrong_unique/(wrong_unique+correct_unique), alongside wrong_unique/all requests; neither denominator replaces the other.',
                'Domain/category sparse groups are retained with intervals; no subgroup superiority or causal pruning attribution from aggregates.'],
             'scope':'Routing and controlled request-mask handoff on two archived upstream proposal pools with common frozen be54300 postprocessing/grouping. The original 37 LaMA generation demonstrations remain a separate experiment; no new generation-success claim.',
             'outputs':{name:{'rows':len(rows),'sha256':sha(output/(name+'.csv'))} for name,rows in tables.items()}}
    (output/'analysis_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf8')
    print(json.dumps({'status':summary['status'],'cohort_n':253,'failure_rows':len(failures),'output':str(output.resolve())}))
    return 0 if summary['status']=='PASS' else 2

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv',required=True,type=Path)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    raise SystemExit(analyze(args.input_csv,args.manifest,args.output))
