"""Independent saved-result checks; no inference, image modification or model fitting."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,csv,hashlib,json,math

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))

def verify(folder):
    cases=read(folder/'group_gate_cases.csv')
    summaries=read(folder/'group_gate_summary.csv')
    pool=read(folder/'group_gate_pool.csv')
    complete=read(folder/'group_gate_complete_exported_pool.csv')
    calls=read(folder/'group_gate_actual_calls.csv')
    protocol=json.loads((folder/'group_gate_protocol.json').read_text())
    provenance=json.loads((folder/'group_gate_input_provenance.json').read_text())
    bits={k:tuple(v) for k,v in protocol['variants'].items()}
    assert len(cases)==2144 and len(summaries)==16 and len(provenance)==268
    assert len({(r['split'],r['case_id'],r['variant']) for r in cases})==len(cases)
    assert Counter(r['split'] for r in cases)=={'test226':1808,'holdout42':336}
    assert Counter(r['split'] for r in complete)=={'test226':1951,'holdout42':449}
    assert Counter(r['split'] for r in pool)=={'test226':1022,'holdout42':258}
    assert all(r['exact_group_map'] and r['exact_group_id_semantic_sequence'] for r in provenance)
    assert all(r['release_group_map_reproduced']=='True' and r['fixed_inputs_unchanged']=='True' for r in cases)
    exports={(r['split'],r['case_id'],r['candidate_key']):r for r in complete}
    assert len(exports)==len(complete)
    nominated=defaultdict(list)
    for r in pool:
        key=(r['split'],r['case_id'],r['candidate_key'])
        assert key in exports and exports[key]['profile_gate_nomination']=='True'
        assert r['mask_sha256']==exports[key]['mask_sha256']
        nominated[key[:2]].append(r)
    pairs=defaultdict(list)
    for r in cases:pairs[(r['split'],r['case_id'])].append(r)
    for pair,rows in pairs.items():
        assert {r['variant'] for r in rows}==set(bits)
        for field in ['fixed_input_sha256','pre_group_gate_candidates','complete_exported_candidates','selected_profile','category']:
            assert len({r[field] for r in rows})==1,(pair,field)
        for r in rows:
            raw=nominated[pair]
            assert int(r['pre_group_gate_candidates'])==len(raw)
            enabled=bits[r['variant']]
            retained=sum(all((v=='True') or not b for v,b in zip([x['semantic_pass'],x['structure_raw_pass'],x['appearance_raw_pass']],enabled)) for x in raw)
            assert int(r['retained_candidates'])==retained
            if not raw:assert float(r['candidate_precision'])==float(r['candidate_recall'])==float(r['candidate_f1'])==0
    lookup={(r['split'],r['case_id']):r for r in cases if r['variant']=='STA'}
    for s in summaries:
        selected=[r for r in cases if (r['split'],r['variant'])==(s['split'],s['variant'])]
        assert len(selected)==int(s['n'])
        for source,target in [('pre_group_gate_candidates','pre_gate_pool_total'),('retained_candidates','retained_total'),('complete_exported_candidates','complete_exported_pool_total')]:
            assert sum(int(r[source]) for r in selected)==int(s[target])
        assert sum(r['group_map_changed']=='True' for r in selected)==int(s['changed_group_maps'])
        assert sum(int(r['pre_group_gate_candidates'])==0 for r in selected)==int(s['cases_without_profile_nominations'])
        for key,value in s.items():
            if not key.endswith('_mean'):continue
            metric=key[:-5]
            if metric.startswith('delta_'):
                metric=metric[6:]
                vals=[float(r[metric])-float(lookup[(r['split'],r['case_id'])][metric]) for r in selected]
            else:vals=[float(r[metric]) for r in selected]
            assert math.isclose(sum(vals)/len(vals),float(value),rel_tol=1e-12,abs_tol=1e-12),key
            assert float(s[key[:-5]+'_low'])<=float(value)+1e-12<=float(s[key[:-5]+'_high'])+1e-12
    predicate_cache={}
    for r in calls:
        key=(r['split'],r['case_id'],r['candidate_key'],r['root_context'])
        raw=tuple(r[k]=='True' for k in ['semantic_raw_pass','structure_raw_pass','appearance_raw_pass'])
        assert key[:3] in exports
        assert key not in predicate_cache or predicate_cache[key]==raw
        predicate_cache[key]=raw
        expected=all(v or not b for v,b in zip(raw,bits[r['variant']]))
        assert (r['intervention_accepted']=='True')==expected
        assert (r['release_accepted']=='True')==all(raw)
    first=folder/'backups/full_first_pass/group_gate_summary.csv'
    repeat_metric_equal=None
    map_count_discrepancies=[]
    if first.exists():
        old={(s['split'],s['variant']):s for s in read(first)}
        for s in summaries:
            prior=old[(s['split'],s['variant'])]
            for key,value in prior.items():
                if key=='changed_group_maps':
                    if value!=s[key]:map_count_discrepancies.append({'split':s['split'],'variant':s['variant'],'first_pass_count':int(value),'current_count':int(s[key])})
                else:assert value==s[key],(s['split'],s['variant'],key,'repeated metric mismatch')
        repeat_metric_equal=True
    checks={'status':'PASS_WITH_DOCUMENTED_ONE_PIXEL_REPEAT_LIMITATION','case_variant_rows':len(cases),'baseline_exact_maps':len(provenance),
            'baseline_exact_id_semantic_sequences':len(provenance),'unchanged_input_checks':len(cases),
            'complete_exported_candidates':len(complete),'profile_nominations':len(pool),
            'actual_gate_invocations':len(calls),'unique_candidate_contexts':len(predicate_cache),
            'summary_means_recomputed':True,'all_nomination_decisions_recomputed':True,
            'raw_predicates_invariant_across_subsets':True,'zero_nomination_cases_retained':True,
            'independent_confidence_interval_reestimation':False,
            'metric_summaries_identical_to_first_full_pass':repeat_metric_equal,
            'counterfactual_maps_bitwise_repeatable':False,
            'counterfactual_repeat_note':'Targeted fixed-seed repeats show one-pixel differences for calculator__01 and scarf__03. All saved evaluation metrics and confidence intervals are identical across full runs. Cause is not assigned; changing-case counts are descriptive run-specific values.',
            'changed_map_count_discrepancies':map_count_discrepancies,
            'input_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob('group_gate_*')) if p.is_file()}}
    (folder/'gate_validation_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    patterns=Counter((r['split'],''.join('1' if r[k]=='True' else '0' for k in ['semantic_pass','structure_raw_pass','appearance_raw_pass'])) for r in pool)
    with (folder/'gate_raw_predicate_patterns.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['split','raw_S_T_A','n']);w.writerows((s,p,n) for (s,p),n in sorted(patterns.items()))
    print(json.dumps(checks,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dir',type=Path,default=Path(__file__).resolve().parent)
    verify(parser.parse_args().dir)
