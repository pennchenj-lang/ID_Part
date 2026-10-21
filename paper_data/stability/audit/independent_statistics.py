"""Independent, read-only aggregation of frozen per-GT stability evidence.

No inference code or producer aggregation helper is imported. Evidence schema is
adapted only at read_csv_rows(), while endpoint definitions remain fixed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METHODS = ('hpid_split_a3', 'dbscan_fusion', 'greedy_nms')
BASE = 'automatic_root'
SEED = 20261009
REPS = 10000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def truth(v):
    return str(v).lower() in ('true', '1', '1.0')


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_csv_rows(path):
    """Minimal input adaptation; no producer-derived aggregate is trusted."""
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        raw = list(csv.DictReader(f))
    manifest_path = Path(__file__).resolve().parents[1] / 'case_metadata.json'
    manifest = {r['case_id']: r for r in json.loads(manifest_path.read_text(encoding='utf-8-sig'))['cases']}
    assert len(manifest) == 226
    assert len({r['image_id'] for r in manifest.values()}) == 226
    assert len({r['source_image_sha256'] for r in manifest.values()}) == 226
    out = []
    for r in raw:
        out.append({
            'case': r['case_id'],
            'category': r['object_category'],
            'domain': manifest[r['case_id']]['expected_domain'],
            'method': r['method'],
            'condition': r['condition'],
            'family': r['family'],
            'threshold': float(r['threshold']),
            'gt': str(r.get('gt_index', r.get('annotation_id'))),
            'identity': r['current_key'],
            'correct': truth(r['current_correct']),
            'initial_identity': r['initial_key'],
            'initial_correct': truth(r['initial_correct']),
            'producer_retained': truth(r['joint_correct_retained']),
            'producer_common_three': truth(r['three_method_common_initial']),
            'producer_common_pair_dbscan_fusion': truth(r['common_initial_hpid_dbscan']),
            'producer_common_pair_greedy_nms': truth(r['common_initial_hpid_nms']),
        })
    return out


def summarize_interval(samples, point, family_size=1):
    tail = 0.025 / family_size
    low, high = np.quantile(samples, (0.025, 0.975))
    adjlow, adjhigh = np.quantile(samples, (tail, 1-tail))
    return dict(estimate=float(point), ci95_low=float(low), ci95_high=float(high),
                family_size=family_size,
                adjusted_confidence=1-0.05/family_size,
                adjusted_ci_low=float(adjlow), adjusted_ci_high=float(adjhigh))


def paired_bootstrap(values, categories, cluster_universe, key, family_size):
    arr = np.asarray(values, float)
    if not len(arr):
        return {}
    seed = SEED + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(arr), size=(REPS, len(arr)))
    draws = arr[indexes].mean(1)
    result = {f'case_{k}': v for k, v in summarize_interval(draws, arr.mean(), family_size).items()}
    categories = np.asarray(categories)
    universe = list(cluster_universe)
    nums = np.asarray([arr[categories == c].sum() for c in universe])
    dens = np.asarray([(categories == c).sum() for c in universe])
    ix = rng.integers(0, len(universe), size=(REPS, len(universe)))
    totals = dens[ix].sum(1)
    valid = totals > 0
    clustered = nums[ix].sum(1)[valid] / totals[valid]
    result.update({f'category_{k}': v for k, v in summarize_interval(clustered, arr.mean(), family_size).items()})
    result['category_bootstrap_valid_draws'] = int(valid.sum())
    return result


def derive(rows):
    records = {}
    cats, domains, gtsets = {}, {}, defaultdict(set)
    family_by_cond = {}
    for r in rows:
        key = (r['case'], r['threshold'], r['method'], r['condition'], r['gt'])
        assert key not in records, f'Duplicate target record {key}'
        records[key] = r
        cats[r['case']] = r['category']
        domains[r['case']] = r['domain']
        gtsets[r['case']].add(r['gt'])
        if r['condition'] != BASE:
            if r['condition'] in family_by_cond:
                assert family_by_cond[r['condition']] == r['family']
            family_by_cond[r['condition']] = r['family']
    cases = sorted(cats)
    thresholds = sorted({r['threshold'] for r in rows})
    conditions = sorted(family_by_cond)
    families = sorted(set(family_by_cond.values()))
    assert len(cases) == 226, f'Expected all 226 cases; found {len(cases)}'
    assert len(conditions) == 9 and len(families) == 3
    assert all(sum(family_by_cond[c] == fam for c in conditions) == 3 for fam in families)
    assert thresholds == [0.25, 0.5]
    expected = sum(len(gtsets[c]) for c in cases) * len(thresholds) * len(METHODS) * (len(conditions)+1)
    assert len(records) == expected, (len(records), expected)
    percase, cohort = [], []
    for th in thresholds:
        for case in cases:
            gts = sorted(gtsets[case])
            baseline = {m: {g: records[(case, th, m, BASE, g)] for g in gts} for m in METHODS}
            correct = {m: {g for g in gts if baseline[m][g]['correct']} for m in METHODS}
            shared = set.intersection(*(correct[m] for m in METHODS))
            pools = {'common_three': shared, 'all_gt': set(gts)}
            for b in METHODS[1:]:
                pools[f'common_pair_{b}'] = correct[METHODS[0]] & correct[b]
            cohort.append(dict(threshold=th, case_id=case, object_category=cats[case],
                               expected_domain=domains[case], gt_count=len(gts),
                               common_three_count=len(shared),
                               **{f'initial_correct_{m}':len(correct[m]) for m in METHODS},
                               **{f'{k}_count':len(v) for k,v in pools.items() if k.startswith('common_pair_')}))
            for method in METHODS:
                for cond in conditions:
                    retained = set()
                    identity_same = set()
                    for g in gts:
                        original = baseline[method][g]
                        changed = records[(case, th, method, cond, g)]
                        assert changed['initial_identity'] == original['identity']
                        assert changed['initial_correct'] == original['correct']
                        assert changed['producer_common_three'] == (g in shared)
                        for b in METHODS[1:]:
                            assert changed[f'producer_common_pair_{b}'] == (g in pools[f'common_pair_{b}'])
                        same = bool(original['identity']) and original['identity'] == changed['identity']
                        if same:
                            identity_same.add(g)
                        if original['correct'] and changed['correct'] and same:
                            retained.add(g)
                        assert changed['producer_retained'] == (g in retained)
                    methodpools = {**pools, 'method_initial_correct':correct[method]}
                    for poolname, pool in methodpools.items():
                        if poolname.startswith('common_pair_') and method not in (METHODS[0], poolname.removeprefix('common_pair_')):
                            continue
                        numerator, denominator = len(pool & retained), len(pool)
                        percase.append(dict(threshold=th, case_id=case, object_category=cats[case],
                                            expected_domain=domains[case], method=method, condition=cond,
                                            family=family_by_cond[cond], cohort=poolname,
                                            numerator=numerator, denominator=denominator,
                                            rate=numerator/denominator if denominator else None))
    return percase, cohort, cats, domains, families, conditions, family_by_cond


def aggregate(percase, cohorts, cats, domains, families, conditions, family_by_cond):
    grouped = defaultdict(list)
    for r in percase:
        grouped[(r['threshold'],r['cohort'],r['method'],r['condition'])].append(r)
    aggregate_rows = []
    cells = {}
    for (th,pool,m,cond), items in grouped.items():
        for r in items:
            cells[(th,pool,m,cond,r['case_id'])] = r
        valid = [r for r in items if r['denominator']]
        aggregate_rows.append(dict(threshold=th, cohort=pool, method=m, scope=cond,
                                   scope_type='condition', eligible_cases=len(valid),
                                   excluded_cases=len(items)-len(valid),
                                   targets=sum(r['denominator'] for r in items),
                                   retained=sum(r['numerator'] for r in items),
                                   macro=np.mean([r['rate'] for r in valid]) if valid else None,
                                   pooled=sum(r['numerator'] for r in items)/sum(r['denominator'] for r in items) if valid else None))
    scopes = [('overall',conditions,2,'primary_aggregate')] + [(f,[c for c in conditions if family_by_cond[c]==f],6,'family') for f in families] + [(c,[c],18,'condition_exploratory') for c in conditions]
    contrasts = []
    values_by_scope = {}
    for th in (0.25,0.5):
        for pool in ('common_three','all_gt','common_pair_dbscan_fusion','common_pair_greedy_nms'):
            methods = METHODS if pool in ('common_three','all_gt') else (METHODS[0],pool.removeprefix('common_pair_'))
            for scope,conds,family_size,scope_type in scopes:
                for method in methods:
                    values = {}
                    for case in sorted(cats):
                        rows = [cells[(th,pool,method,c,case)] for c in conds]
                        if rows[0]['denominator']:
                            assert len({r['denominator'] for r in rows}) == 1
                            values[case] = np.mean([r['rate'] for r in rows])
                    values_by_scope[(th,pool,method,scope)] = values
                    if len(conds)>1:
                        selected = [cells[(th,pool,method,c,k)] for k in cats for c in conds]
                        den=sum(r['denominator'] for r in selected)
                        aggregate_rows.append(dict(threshold=th,cohort=pool,method=method,scope=scope,scope_type=scope_type,
                                                   eligible_cases=len(values),excluded_cases=len(cats)-len(values),
                                                   targets=den,retained=sum(r['numerator'] for r in selected),
                                                   macro=float(np.mean(list(values.values()))) if values else None,
                                                   pooled=sum(r['numerator'] for r in selected)/den if den else None))
                for baseline in methods[1:]:
                    a=values_by_scope[(th,pool,METHODS[0],scope)]
                    b=values_by_scope[(th,pool,baseline,scope)]
                    assert set(a)==set(b)
                    caseids=sorted(a)
                    differences=[a[k]-b[k] for k in caseids]
                    entry=dict(threshold=th,cohort=pool,scope=scope,scope_type=scope_type,
                               baseline=baseline,eligible_cases=len(caseids),excluded_cases=len(cats)-len(caseids),
                               primary=th==.25 and pool=='common_three' and scope=='overall',
                               hpid_macro=float(np.mean(list(a.values()))) if a else None,
                               baseline_macro=float(np.mean(list(b.values()))) if b else None)
                    key=f'{th}|{pool}|{scope}|{baseline}'
                    entry.update(paired_bootstrap(differences,[cats[k] for k in caseids],sorted(set(cats.values())),key,family_size))
                    contrasts.append(entry)
    strata=[]
    for th in (.25,.5):
        for pool in ('common_three','all_gt'):
            for method in METHODS:
                values=values_by_scope[(th,pool,method,'overall')]
                for stratifier,labelmap in [('domain',domains),('category',cats)]:
                    for label in sorted(set(labelmap.values())):
                        keys=[k for k in values if labelmap[k]==label]
                        strata.append(dict(threshold=th,cohort=pool,method=method,stratifier=stratifier,
                                           label=label,eligible_cases=len(keys),
                                           macro=np.mean([values[k] for k in keys]) if keys else None))
    return aggregate_rows, contrasts, strata


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--targets',type=Path,required=True)
    ap.add_argument('--protocol',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    rows=read_csv_rows(args.targets)
    percase,cohorts,cats,domains,families,conditions,family_by_cond=derive(rows)
    aggregates,contrasts,strata=aggregate(percase,cohorts,cats,domains,families,conditions,family_by_cond)
    write_csv(args.out/'independent_case_rates.csv',percase)
    write_csv(args.out/'independent_cohorts.csv',cohorts)
    write_csv(args.out/'independent_estimates.csv',aggregates)
    write_csv(args.out/'independent_contrasts.csv',contrasts)
    write_csv(args.out/'independent_strata.csv',strata)
    report=dict(status='complete',input_sha256=sha(args.targets),protocol_sha256=sha(args.protocol),
                script_sha256=sha(__file__),target_rows=len(rows),cases=len(cats),categories=len(set(cats.values())),
                domains=len(set(domains.values())),bootstrap_replicates=REPS,bootstrap_seed=SEED,
                conditions=conditions,families=families,
                interval_method='paired case percentile bootstrap; category-cluster sensitivity; Bonferroni individual intervals for simultaneous coverage',
                primary=[r for r in contrasts if r['primary']],
                limits='Conditional primary cohort excludes cases with no common initial correct GT; all-GT evidence must accompany primary. Zero crossing is not proof of equivalence.')
    (args.out/'independent_statistics_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
