"""Independent R6 aggregation and sealed-output audit. Never edits sealed files.

Does not import run_routing or analyze_routing. Raw and HPID instances are read
directly from sealed masks; the three named CPU comparators are reconstructed
with their sealed published implementation to validate the instance ledger.
Routing decisions and statistics below are implemented independently.
"""
from __future__ import annotations
import collections
import csv
import datetime
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import cv2
from PIL import Image
from scipy.stats import norm
cv2.setNumThreads(1)

HERE = Path(__file__).resolve().parent
RUN = HERE / 'routing_run'
STATS = HERE / 'statistics'
SNAP = Path('__RUNTIME_ROOT__/code_snapshots/hpid_split_be54300_holdout/src')
PROJECT = Path('__USER_HOME__/Documents/Codex/2026-06-28/new-chat/hpid_split')
sys.path.insert(0, str(SNAP))
import hpid_split
hpid_split.__path__.append(str(PROJECT / 'src/hpid_split'))
from hpid_split.paco_semantics import canonical_part_token, normalize_paco_name
from hpid_split.postprocess_baselines import greedy_nms_ownership, dbscan_proposal_fusion, local_pairwise_crf

METHODS = ('raw_proposals', 'greedy_nms', 'dbscan_fusion', 'local_pairwise_crf', 'hpid_split_group_ids')
HPID, RAW = METHODS[-1], METHODS[0]
STATES = ('correct_unique', 'wrong_unique', 'ambiguous', 'unresolved')
TOL = 2e-12
COUNTS = collections.Counter()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def csvread(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_sha(path, expected, role):
    actual = sha(path)
    assert actual == expected, (role, str(path), expected, actual)
    COUNTS[role] += 1


def eq(actual, expected, context):
    if expected is None:
        assert actual in (None, ''), (context, actual, expected)
    elif isinstance(expected, (str, bool)):
        if isinstance(expected, bool):
            assert str(actual).lower() == str(expected).lower(), (context, actual, expected)
        else:
            assert str(actual) == expected, (context, actual, expected)
    else:
        assert math.isfinite(float(actual)) and abs(float(actual) - expected) <= TOL, (context, actual, expected)
    COUNTS['value_comparisons'] += 1


def check_fields(actual, expected, context):
    for key, value in expected.items():
        assert key in actual, (context, key, 'missing')
        eq(actual[key], value, (context, key))


def binary(path):
    return np.asarray(Image.open(path).convert('L')) >= 128


def array_sha(mask):
    return hashlib.sha256(np.ascontiguousarray(mask, dtype=np.uint8).tobytes()).hexdigest()


def iou(a, b):
    denominator = int(np.logical_or(a, b).sum())
    return int(np.logical_and(a, b).sum()) / denominator if denominator else 0.


def token(value, unit, predicted=False):
    if predicted and value == unit['expected_domain']:
        return 'body'
    if predicted:
        value = value.removeprefix(unit['expected_domain'] + '_')
    return canonical_part_token(normalize_paco_name(value), unit['expected_domain'],
                                object_category=unit['object_category'])


def wilson(k, n):
    if not n:
        return dict(k=int(k), n=0, proportion=None, ci95_low=None, ci95_high=None)
    z = float(norm.ppf(.975))
    a, b = k + z*z/2, z * math.sqrt(k*(n-k)/n + z*z/4)
    return dict(k=int(k), n=int(n), proportion=k/n,
                ci95_low=max(0., (a-b)/(n+z*z)), ci95_high=min(1., (a+b)/(n+z*z)))


def route_state(row, threshold):
    if row['query_match_count'] == 0:
        return 'unresolved'
    if row['query_match_count'] > 1:
        return 'ambiguous'
    return 'correct_unique' if row[f'unique_and_correct_{threshold}'] else 'wrong_unique'


def main():
    protocol = read(HERE / 'protocol_frozen.json')
    freeze = read(HERE / 'protocol_freeze_receipt.json')
    receipt = read(RUN / 'receipt.json')
    manifest = read(HERE / 'request_manifest_frozen.json')
    analysis = read(STATS / 'analysis_summary.json')
    assert receipt['status'] == 'COMPLETE' and receipt['failures'] == []
    assert analysis['status'] == 'PASS' and analysis['execution_failure_rows'] == 0
    assert protocol['status'] == 'FROZEN_BEFORE_EXPANDED_ROUTING'
    assert datetime.datetime.fromisoformat(protocol['frozen_utc']) < datetime.datetime.fromisoformat(receipt['completed_utc'])
    verify_sha(HERE / 'protocol_frozen.json', freeze['protocol_sha256'], 'freeze_binding')
    verify_sha(HERE / 'protocol_frozen.json', receipt['protocol_sha256'], 'freeze_binding')
    verify_sha(HERE / 'request_manifest_frozen.json', protocol['manifest_sha256'], 'manifest_binding')
    verify_sha(HERE / 'request_manifest_frozen.json', receipt['manifest_sha256'], 'manifest_binding')
    verify_sha(HERE / 'statistical_plan_frozen.json', protocol['statistical_plan_sha256'], 'plan_binding')
    verify_sha(HERE / 'analyze_routing.py', protocol['analysis_script_sha256'], 'analysis_code_binding')
    for path, digest in protocol['implementation_hashes'].items():
        verify_sha(path, digest, 'implementation_file')
    for path, digest in protocol['input_hashes'].items():
        verify_sha(path, digest, 'input_file')
    for relative, digest in protocol['bindings'].items():
        verify_sha(HERE / relative, digest, 'protocol_binding')
    artifacts = {str(p.relative_to(RUN)) for p in RUN.rglob('*') if p.is_file() and p.name != 'receipt.json'}
    assert artifacts == set(receipt['artifact_sha256']), ('artifact_set', artifacts ^ set(receipt['artifact_sha256']))
    for relative, digest in receipt['artifact_sha256'].items():
        verify_sha(RUN / relative, digest, 'routing_artifact')
    for field, filename in [('routing_csv_sha256', 'routing_cases.csv'), ('instances_csv_sha256', 'routing_instances.csv')]:
        verify_sha(RUN / filename, receipt[field], 'routing_csv_binding')
    for name, spec in analysis['outputs'].items():
        verify_sha(STATS / (name + '.csv'), spec['sha256'], 'statistics_artifact')
        eq(len(csvread(STATS / (name + '.csv'))), spec['rows'], (name, 'row_count'))
    for key, path in [('input_csv_sha256', RUN / 'routing_cases.csv'),
                      ('manifest_sha256', HERE / 'request_manifest_frozen.json'),
                      ('analysis_script_sha256', HERE / 'analyze_routing.py')]:
        verify_sha(path, analysis['bindings'][key], 'statistics_binding')
    group_receipt = read(HERE / 'common_groups/preparation_receipt.json')
    verify_sha(HERE / 'common_groups/preparation_receipt.json', manifest['group_preparation_receipt_sha256'], 'group_binding')
    for name, digest in group_receipt['source_sha256'].items():
        verify_sha(Path(group_receipt['source_snapshot']) / 'hpid_split' / name, digest, 'group_code_binding')
    verify_sha(HERE / 'prepare_common_groups.py', group_receipt['script_sha256'], 'group_code_binding')
    print('Sealed input/code/artifact hashes verified', flush=True)

    units = {r['request_id']: r for r in manifest['requests']}
    assert len(units) == 253
    assert len({r['image_id'] for r in units.values()}) == 253
    assert len({r['source_image_sha256'] for r in units.values()}) == 253
    assert collections.Counter(r['cohort'] for r in units.values()) == dict(old37=37, new216=216)
    recorded = {(r['request_id'], r['method']): r for r in csvread(RUN / 'routing_cases.csv')}
    assert len(recorded) == 1265
    ledger = collections.defaultdict(list)
    for row in csvread(RUN / 'routing_instances.csv'):
        ledger[row['request_id'], row['method']].append(row)
    independent = {}
    selected_mask_checks = 0
    for case_index, (rid, unit) in enumerate(units.items(), 1):
        package = Path(unit['package_path'])
        truth = binary(unit['target_mask_path'])
        candidates = []
        own_raw = []
        for index, c in enumerate(read(package / 'candidates.json'), 1):
            mask = binary(package / c['mask_path'])
            candidate = SimpleNamespace(**{k:c.get(k) for k in ('semantic_name','semantic_parent','score','source','prompt')},
                                        source_reliability=c.get('source_reliability',1.), metadata=c.get('metadata',{}), mask=mask)
            candidates.append(candidate)
            confidence = max(0., min(1., .72*float(c['score']) + .28*float(c.get('source_reliability',1.))))
            if mask.sum() >= 6:
                own_raw.append(SimpleNamespace(identity=f'proposal/{index:04d}', semantic_name=c['semantic_name'],
                                               mask=mask, confidence=confidence))
        labels = np.asarray(Image.open(unit['hpid_group_map_path']))
        own_hpid = [SimpleNamespace(identity=g['group_id'], semantic_name=g['semantic_name'],
                                   mask=labels == int(g['group_index']), confidence=1.) for g in read(unit['hpid_groups_path'])]
        source = np.asarray(Image.open(package / 'source.png').convert('RGB'))
        masks_by_method = {
            RAW: own_raw, HPID: own_hpid,
            'greedy_nms': greedy_nms_ownership(candidates).instances,
            'dbscan_fusion': dbscan_proposal_fusion(candidates).instances,
            'local_pairwise_crf': local_pairwise_crf(candidates, source).instances,
        }
        target_token = token(unit['target_part_name'], unit)
        defect = binary(RUN / 'request_masks' / rid / 'controlled_defect.png')
        assert defect.any() and not (defect & ~truth).any()
        for method in METHODS:
            entry = recorded[rid, method]
            assert entry['execution_status'] == 'complete'
            rows = ledger[rid, method]
            actual_masks = {p.identity: p for p in masks_by_method[method]}
            assert len(actual_masks) == len(rows) == len(masks_by_method[method])
            assert len({r['identity'] for r in rows}) == len(rows)
            for instance in rows:
                pred = actual_masks[instance['identity']]
                expected = dict(semantic=pred.semantic_name, confidence=float(pred.confidence),
                                area_px=int(pred.mask.sum()), mask_sha256=array_sha(pred.mask),
                                normalized_semantic=token(pred.semantic_name, unit, True), target_iou=iou(pred.mask, truth))
                check_fields(instance, expected, (rid, method, instance['identity']))
                COUNTS['instance_records_checked'] += 1
            # Core aggregation uses ledger rows, not the routing CSV or runner functions.
            matched = [r for r in rows if r['normalized_semantic'] == target_token]
            exposed = matched or rows
            ordered = sorted(matched, key=lambda r:(float(r['confidence']),int(r['area_px']),r['identity']), reverse=True)
            choice = ordered[0] if ordered else None
            score = float(choice['target_iou']) if choice else 0.
            best_match = max((float(r['target_iou']) for r in matched), default=0.)
            best_exposed = max((float(r['target_iou']) for r in exposed), default=0.)
            best_exported = max((float(r['target_iou']) for r in rows), default=0.)
            expected = dict(query_match_count=len(matched), all_exported_count=len(rows), review_option_count=len(exposed),
                            query_unique=int(len(matched)==1), query_ambiguous=int(len(matched)>1), query_unresolved=int(not matched),
                            query_state='unresolved' if not matched else 'unique' if len(matched)==1 else 'ambiguous',
                            selected_identity=choice['identity'] if choice else '', selected_part_iou=score,
                            best_review_match_iou=best_match, best_exposed_iou=best_exposed, best_exported_iou=best_exported,
                            normalized_target=target_token)
            for t, threshold in [('025',.25), ('050',.5)]:
                expected.update({f'unique_and_correct_{t}':int(len(matched)==1 and score>=threshold),
                                 f'wrong_unique_{t}':int(len(matched)==1 and score<threshold),
                                 f'semantic_correct_any_match_{t}':int(best_match>=threshold),
                                 f'geometry_correct_any_exposed_{t}':int(best_exposed>=threshold),
                                 f'geometry_correct_any_exported_{t}':int(best_exported>=threshold),
                                 f'correct_in_review_set_{t}':int(len(matched)>1 and best_match>=threshold)})
            for ins in rows:
                check_fields(ins, dict(matches_target=int(ins in matched), actual_exposed=int(ins in exposed)), (rid,method,'ledger_flags'))
            check_fields(entry, expected, (rid, method, 'routing'))
            check_fields(entry, {k:unit[k] for k in ('request_id','case_id','cohort','image_id','source_image_sha256','object_category','domain')}, (rid,method,'metadata'))
            selected = actual_masks[choice['identity']].mask if choice else np.zeros(truth.shape, bool)
            request_mask = binary(entry['request_mask_path'])
            assert np.array_equal(request_mask, selected & defect)
            check_fields(entry, dict(request_pixels=int(request_mask.sum()), controlled_defect_pixels=int(defect.sum()),
                                    request_recall=float(request_mask.sum()/defect.sum()), defect_fraction_actual=float(defect.sum()/truth.sum())), (rid,method,'request_mask'))
            selected_mask_checks += 1
            independent[rid,method] = expected
        if case_index % 50 == 0:
            print(f'Independently checked {case_index}/253 cases', flush=True)

    old = {(r['case_id'],r['method']):r for r in csvread(HERE.parent / 'review_burden/review_cases.csv')}
    for rid, unit in units.items():
        if unit['cohort'] == 'old37':
            for method in METHODS:
                expected = independent[rid,method] | dict(case_id=unit['case_id'], method=method, image_id=unit['image_id'], target=unit['target_part_name'])
                check_fields(old[unit['case_id'],method], expected, (rid,method,'legacy'))
                COUNTS['old37_case_methods_exact'] += 1

    def ids_for(cohort):
        return [rid for rid, u in units.items() if cohort == 'combined' or u['cohort'] == cohort]

    def group_rows(cohort, method):
        return [independent[rid,method] for rid in ids_for(cohort)]

    def stat_check(name, predicate):
        table = csvread(STATS / (name + '.csv'))
        for row in table:
            check_fields(row, predicate(row), (name, tuple(row.items())[:5]))
        COUNTS['statistics_rows_checked'] += len(table)

    stat_check('routing_states_with_wilson', lambda r: wilson(sum(route_state(x,r['threshold'])==r['state'] for x in group_rows(r['cohort'],r['method'])),len(ids_for(r['cohort']))))
    stat_check('binary_guardrails_with_wilson', lambda r: wilson(sum(x[r['metric']+'_'+r['threshold']] for x in group_rows(r['cohort'],r['method'])),len(ids_for(r['cohort']))))

    def conditional(r):
        xs=group_rows(r['cohort'],r['method']); t=r['threshold']
        if r['metric']=='correct_in_ambiguous_review':
            return wilson(sum(x[f'correct_in_review_set_{t}'] for x in xs),sum(x['query_ambiguous'] for x in xs))
        metric='unique_and_correct' if r['metric']=='correct_among_unique_routes' else 'wrong_unique'
        return wilson(sum(x[f'{metric}_{t}'] for x in xs),sum(x['query_unique'] for x in xs))
    stat_check('conditional_review_rates', conditional)

    def option_dist(r):
        xs=group_rows(r['cohort'],r['method']); v=np.array([x['review_option_count'] for x in xs])
        return dict(n=len(xs), mean=float(v.mean()),median=float(np.median(v)),q25=float(np.quantile(v,.25)),q75=float(np.quantile(v,.75)),
                    minimum=int(v.min()),maximum=int(v.max()),empty_exposed_n=int((v==0).sum()),execution_failure_n=0,
                    mean_all_exported_count=float(np.mean([x['all_exported_count'] for x in xs])))
    stat_check('method_option_distribution', option_dist)

    def transition(r):
        ids=ids_for(r['cohort']);t=r['threshold']
        eligible=[rid for rid in ids if route_state(independent[rid,r['comparator']],t)==r['from_state']]
        k=sum(route_state(independent[rid,HPID],t)==r['to_hpid_state'] for rid in eligible)
        return wilson(k,len(eligible)) | dict(all_cohort_n=len(ids),all_cohort_proportion=k/len(ids))
    stat_check('route_transition_4x4', transition)

    def paired_counts(r):
        ids=ids_for(r['cohort']); metric=r['metric'] if r['metric'].startswith('query_') else r['metric']+'_'+r['threshold']
        pairs=collections.Counter((independent[rid,HPID][metric],independent[rid,r['comparator']][metric]) for rid in ids)
        both,gain,loss,neither=(pairs[k] for k in ((1,1),(1,0),(0,1),(0,0)))
        return len(ids),both,gain,loss,neither

    def retention(r):
        n,b,g,l,z=paired_counts(r)
        k,denom={'retained':(b,n),'lost':(l,n),'gained':(g,n),'neither':(z,n),
                 'loss_given_comparator_available':(l,b+l),'gain_given_comparator_unavailable':(g,g+z)}[r['event']]
        return wilson(k,denom)
    stat_check('correct_option_retention_and_loss', retention)

    def guardrail(r):
        n,b,g,l,z=paired_counts(r); discordant=g+l
        p=min(1.,2*sum(math.comb(discordant,k) for k in range(min(g,l)+1))/2**discordant) if discordant else 1.
        return dict(n=n,hpid_k=b+g,comparator_k=b+l,both_positive=b,hpid_only_positive=g,comparator_only_positive=l,
                    both_negative=z,difference_percentage_points=100*(g-l)/n,discordant_n=discordant,mcnemar_exact_two_sided_p=p)
    stat_check('paired_correctness_guardrails', guardrail)

    def raw_flow(r):
        ids=ids_for(r['cohort']);t=r['threshold']
        if r['subset']=='all_requests':
            k=sum(not independent[rid,RAW]['query_ambiguous'] and independent[rid,HPID]['query_ambiguous'] for rid in ids)
            return wilson(k,len(ids))
        selected=[rid for rid in ids if independent[rid,RAW]['query_ambiguous'] and
                  (r['subset']=='raw_ambiguous' or independent[rid,RAW][f'correct_in_review_set_{t}'])]
        counts=collections.Counter()
        for rid in selected:
            x=independent[rid,HPID]; s=route_state(x,t)
            if s=='ambiguous': s='ambiguous_with_correct' if x[f'correct_in_review_set_{t}'] else 'ambiguous_without_correct'
            if s=='unresolved': s='unresolved_with_correct_fallback' if x[f'geometry_correct_any_exposed_{t}'] else 'unresolved_without_correct_fallback'
            counts[s]+=1
        return wilson(counts[r['to_hpid_state']],len(selected))
    stat_check('raw_ambiguous_fixed_subset_flow', raw_flow)

    def domain_counts(r):
        ids=[rid for rid in ids_for(r['cohort']) if units[rid][r['group_by']]==r['group']]
        xs=[independent[rid,r['method']] for rid in ids];m=r['metric'];t=r['threshold'];n=len(xs)
        if m=='execution_failure': k=0
        elif m in STATES:k=sum(route_state(x,t)==m for x in xs)
        elif m=='ambiguous_without_correct_per_all':k=sum(x['query_ambiguous'] and not x[f'correct_in_review_set_{t}'] for x in xs)
        else:k=sum(x[f'{m}_{t}'] for x in xs)
        den=sum(x['query_ambiguous'] for x in xs) if m=='correct_in_review_set' else n
        return wilson(k,den) | dict(source_images_n=n,sparse_n_lt_10=n<10)
    stat_check('domain_category_failure_summary', domain_counts)

    bootstrap_checks=[]
    comparison_rows={(r['cohort'],r['comparator']):r for r in csvread(STATS/'paired_option_contrasts_all_four.csv')}
    for cohort in ('old37','new216','combined'):
        ids=sorted(ids_for(cohort),key=lambda rid:(str(units[rid]['image_id']),rid))
        rng=np.random.Generator(np.random.PCG64(20261009))
        blocks=[]
        for stratum in ('old37','new216'):
            index=np.array([i for i,rid in enumerate(ids) if units[rid]['cohort']==stratum],dtype=np.int64)
            if index.size:blocks.append(index[rng.integers(index.size,size=(10000,index.size))])
        draws=np.concatenate(blocks,axis=1)
        draw_receipt=next(x for x in analysis['bootstrap']['receipts'] if x['cohort']==cohort)
        assert ids==draw_receipt['ordered_request_ids']
        assert hashlib.sha256(draws.tobytes()).hexdigest()==draw_receipt['draw_index_sha256']
        for comparator in METHODS[:-1]:
            h=np.array([independent[rid,HPID]['review_option_count'] for rid in ids],dtype=float)
            b=np.array([independent[rid,comparator]['review_option_count'] for rid in ids],dtype=float)
            diff=h-b;sample=diff[draws].mean(axis=1)
            q=np.quantile(sample,[.025,.975,.00625,.99375],method='linear')
            values=dict(n=len(ids),hpid_mean=float(h.mean()),comparator_mean=float(b.mean()),
                        mean_difference_hpid_minus_comparator=float(diff.mean()),relative_reduction_percent=float(100*(b.mean()-h.mean())/b.mean()),
                        ci95_low=float(q[0]),ci95_high=float(q[1]),ci9875_low=float(q[2]),ci9875_high=float(q[3]),
                        direction_supported_by_adjusted_interval=bool(q[3]<0),all_case_compactness_superiority_claim_eligible=True,
                        paired_complete_n=len(ids),paired_complete_mean_difference=float(diff.mean()))
            check_fields(comparison_rows[cohort,comparator],values,(cohort,comparator,'bootstrap'))
            bootstrap_checks.append(dict(cohort=cohort,comparator=comparator,**values))
    COUNTS['paired_option_bootstrap_contrasts_exact']=len(bootstrap_checks)

    summary=[]
    for cohort in ('old37','new216','combined'):
        for method in METHODS:
            xs=group_rows(cohort,method)
            summary.append(dict(cohort=cohort,method=method,n=len(xs),mean_review_options=sum(x['review_option_count'] for x in xs)/len(xs),
                                thresholds={t:{**dict(collections.Counter(route_state(x,t) for x in xs)),
                                               'semantic_correct_available':sum(x[f'semantic_correct_any_match_{t}'] for x in xs),
                                               'correct_in_review':sum(x[f'correct_in_review_set_{t}'] for x in xs)} for t in ('025','050')}))
    compact_loss=[]
    for cohort in ('old37','new216','combined'):
        for comparator in METHODS[:-1]:
            n,b,g,l,z=paired_counts(dict(cohort=cohort,comparator=comparator,metric='semantic_correct_any_match',threshold='025'))
            compact_loss.append(dict(cohort=cohort,comparator=comparator,n=n,retained=b,gained=g,lost=l,neither=z))
    output=dict(status='PASS',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                scope='Read-only verification of frozen artifacts; independently implemented route decisions/statistics. No new model inference, tuning, or changes to frozen files.',
                counts=dict(COUNTS),request_count=253,case_method_count=1265,unique_source_images=253,
                execution_failures=0,empty_exposed_case_methods=0,request_mask_exact_checks=selected_mask_checks,
                cohorts={'old37':37,'new216':216,'combined':253},method_counts=summary,
                paired_semantic_correct_availability_025=compact_loss,paired_option_bootstrap=bootstrap_checks,
                validation_boundary={'normalization':'Uses the frozen PACO canonical-part mapping as the shared evaluation contract.',
                                    'raw_hpid_masks_confidence':'Independently read candidate and common Group masks, independently recomputed confidences/IoU/areas/hashes.',
                                    'other_baselines':'Sealed published CPU comparator functions reconstruct masks/confidence; independent ledger aggregation follows.',
                                    'statistics':'No import of run_routing.py or analyze_routing.py. Wilson, transitions, conditional denominators, paired loss, exact McNemar and option bootstrap implemented in this audit.',
                                    'bootstrap':'Reproduced all12 option contrasts with original seed, PCG64 and stratum order; binary bootstrap intervals not redundantly rerun.'},
                bindings={str(p):sha(p) for p in [Path(__file__),HERE/'protocol_frozen.json',HERE/'protocol_freeze_receipt.json',HERE/'request_manifest_frozen.json',
                                                 RUN/'receipt.json',RUN/'routing_cases.csv',RUN/'routing_instances.csv',STATS/'analysis_summary.json']})
    (HERE/'independent_routing_audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status='PASS',counts=dict(COUNTS),method_counts=summary,paired_semantic_correct_availability_025=compact_loss),ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
