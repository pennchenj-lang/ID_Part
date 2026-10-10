"""Read existing CSV/JSON only. No matching, masks, model execution or scoring changes."""
from __future__ import annotations
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
WORKSPACE = OUT.parents[2]
EVIDENCE = WORKSPACE/'hpid-publication-20261009/docs/evidence/revision_20261008/evidence'
CODE = EVIDENCE.parent/'code/analysis/stage_diagnosis.py'
T = .25
EPS = 1e-10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(name):
    return json.loads((EVIDENCE/name).read_text(encoding='utf-8-sig'))


def rows(name):
    with (EVIDENCE/name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(name, values):
    fields = list(dict.fromkeys(k for r in values for k in r))
    with (OUT/name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(values)


def write_json(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def main():
    inputs = ['dbscan_diagnosis.json', 'stage_report.json', 'stage_summary.csv',
              'stage_transitions.csv', 'stage_reference_tracks.csv', 'stage_shared_input_audit.csv',
              'stage_validation.json', 'stage_audit_summary.json', 'stage_protocol.json', 'stage_case_metrics.csv']
    bindings = {name: sha(EVIDENCE/name) for name in inputs}
    tracks = rows('stage_reference_tracks.csv')
    cases = rows('stage_case_metrics.csv')
    shared = rows('stage_shared_input_audit.csv')
    sums = rows('stage_summary.csv')
    transitions = rows('stage_transitions.csv')
    protocol = read('stage_protocol.json')
    diagnosis = read('dbscan_diagnosis.json')
    previous_audit = read('stage_audit_summary.json')
    ordered = protocol['stage_order']
    actual = [s for s in ordered if not s.startswith('diagnostic_') and s != 'final_dbscan']
    ti = {(r['case_id'], int(r['reference_index']), r['stage']): r for r in tracks}
    assert len(ti) == len(tracks) == 990 * 17
    ci = {(r['case_id'], r['stage']): r for r in cases}
    assert len(ci) == len(cases) == 226 * 17
    case_ids = sorted({r['case_id'] for r in cases})
    ref_ids = sorted({(r['case_id'], int(r['reference_index'])) for r in tracks})
    assert len(case_ids) == len(shared) == 226 and len(ref_ids) == 990
    assert len({r['category'] for r in shared if 'category' in r} or {r['category'] for r in cases}) == 60
    assert sum(int(r['candidate_count']) for r in shared) == 1951
    assert sum(int(r['reference_count']) for r in shared) == 990
    assert sum(int(r['accepted_count_after_refilter']) for r in shared) == 1951
    assert all(r['input_masks_unchanged'] == 'True' and r['exact_hpid_map_reproduction'] == 'True' for r in shared)
    assert max(float(r['archived_numeric_metric_max_abs_difference']) for r in shared) == 0
    assert sum(int(r['remainder_components_attached']) for r in shared) == 0
    assert sum(int(r['identity_visibility_slivers_dropped']) for r in shared) == 14
    verification = []
    numeric_metrics = ['diagnostic_f1_at_025', 'diagnostic_semantic_f1_at_025', 'best_iou',
                       'best_semantic_iou', 'semantic_union_recall', 'any_union_recall']
    per_case_refs = defaultdict(list)
    for r in tracks:
        per_case_refs[(r['case_id'], r['stage'])].append(r)
    for (case_id, stage), r in ci.items():
        rr = per_case_refs[case_id, stage]
        n = int(r['reference_count'])
        p = int(r['prediction_count'])
        k = sum(float(x['hungarian_iou']) >= T for x in rr)
        sk = sum(float(x['semantic_hungarian_iou']) >= T for x in rr)
        assert n == len(rr) and k == int(r['matches_at_025']) and sk == int(r['semantic_matches_at_025'])
        assert abs(2*k/max(1,p+n) - float(r['diagnostic_f1_at_025'])) < EPS
        assert abs(2*sk/max(1,p+n) - float(r['diagnostic_semantic_f1_at_025'])) < EPS
        for key in numeric_metrics[2:]:
            assert abs(sum(float(x[key]) for x in rr)/n - float(r[key])) < EPS
    for r in sums:
        cc = [ci[cid, r['stage']] for cid in case_ids]
        assert sum(int(x['prediction_count']) for x in cc) == int(r['total_prediction_count'])
        assert sum(int(x['matches_at_025']) for x in cc) == int(r['total_matches_at_025'])
        for key in numeric_metrics:
            assert abs(sum(float(x[key]) for x in cc)/226 - float(r[key+'_mean'])) < EPS
    for r in transitions:
        before, after = r['before'], r['after']
        for key in numeric_metrics:
            value = sum(float(ci[cid,after][key]) - float(ci[cid,before][key]) for cid in case_ids)/226
            assert abs(value - float(r['delta_'+key+'_mean'])) < EPS
        for key in numeric_metrics[2:]:
            ds = [float(ti[cid,ref,after][key])-float(ti[cid,ref,before][key]) for cid,ref in ref_ids]
            assert sum(v < -EPS for v in ds) == int(r[key+'_references_decreased'])
            assert sum(v > EPS for v in ds) == int(r[key+'_references_increased'])

    paired, no_support, detailed_changes, gate_negative = [], [], [], []
    for cid, ref in ref_ids:
        h = ti[cid, ref, 'final_hpid']; b = ti[cid, ref, 'final_dbscan']
        hm = float(h['hungarian_iou']) >= T; bm = float(b['hungarian_iou']) >= T
        state = 'both_matched' if hm and bm else 'dbscan_only' if bm else 'hpid_only' if hm else 'neither_matched'
        hg = float(h['best_iou']) >= T
        subtype = ('no_hpid_geometry_at_threshold' if not hg else 'hpid_geometry_available_but_not_credited_by_saved_assignment') if state == 'dbscan_only' else 'not_applicable'
        row = {'case_id': cid, 'reference_index': ref, 'domain': h['domain'], 'category': h['category'],
               'reference_semantic': h['reference_semantic'], 'reference_pixels': int(h['reference_pixels']),
               'threshold': T, 'paired_assignment_state': state, 'dbscan_only_subtype': subtype,
               'hpid_hungarian_iou': float(h['hungarian_iou']), 'dbscan_hungarian_iou': float(b['hungarian_iou']),
               'hpid_best_iou': float(h['best_iou']), 'dbscan_best_iou': float(b['best_iou']),
               'hpid_best_identity': h['best_identity'], 'dbscan_best_identity': b['best_identity'],
               'hpid_best_semantic': h['best_semantic'], 'dbscan_best_semantic': b['best_semantic'],
               'hpid_best_semantic_iou': float(h['best_semantic_iou']), 'dbscan_best_semantic_iou': float(b['best_semantic_iou']),
               'hpid_semantic_hungarian_iou': float(h['semantic_hungarian_iou']),
               'dbscan_semantic_hungarian_iou': float(b['semantic_hungarian_iou']),
               'hpid_semantic_union_recall': float(h['semantic_union_recall']),
               'dbscan_semantic_union_recall': float(b['semantic_union_recall'])}
        ga = ti[cid,ref,'actual_semantic_argmax']; gg = ti[cid,ref,'after_direct_detail_gate']
        gd = float(gg['best_iou']) - float(ga['best_iou'])
        row['detail_gate_best_iou_delta'] = gd
        row['detail_gate_best_iou_decreased'] = int(gd < -EPS)
        paired.append(row)
        if gd < -EPS:
            gate_negative.append({**row, 'before_gate_best_iou': float(ga['best_iou']), 'after_gate_best_iou': float(gg['best_iou']),
                                  'gate_semantic_union_recall_delta': float(gg['semantic_union_recall'])-float(ga['semantic_union_recall'])})
        if state != 'dbscan_only' or hg:
            continue
        values = [float(ti[cid,ref,s]['best_iou']) for s in actual]
        support = [v >= T for v in values]
        losses = [(actual[i-1],actual[i]) for i in range(1,len(actual)) if support[i-1] and not support[i]]
        gains = [(actual[i-1],actual[i]) for i in range(1,len(actual)) if not support[i-1] and support[i]]
        ever = any(support)
        first_loss = ' -> '.join(losses[0]) if losses else 'no_observed_threshold_loss_never_supported'
        last_loss = ' -> '.join(losses[-1]) if losses else 'no_observed_threshold_loss_never_supported'
        base = {**row, 'input_pool_supported': int(support[0]), 'any_actual_stage_supported': int(ever),
                'first_observed_actual_threshold_loss': first_loss,
                'last_observed_actual_threshold_loss': last_loss,
                'threshold_loss_event_count': len(losses), 'threshold_recovery_event_count': len(gains),
                'recovered_after_first_loss': int(bool(losses) and any(actual.index(a) > actual.index(losses[0][0]) for a,b in gains)),
                'interpretation': 'Observed representation transitions; first loss may recover; not causal or additive.'}
        for stage, value in zip(actual, values):
            base[stage+'_best_iou'] = value
        no_support.append(base)
        for i in range(1, len(actual)):
            before, after = actual[i-1], actual[i]
            detailed_changes.append({'case_id': cid, 'reference_index': ref, 'domain': h['domain'], 'category': h['category'],
                 'reference_semantic': h['reference_semantic'], 'before': before, 'after': after,
                 'before_best_iou': values[i-1], 'after_best_iou': values[i], 'delta_best_iou': values[i]-values[i-1],
                 'before_supported': int(support[i-1]), 'after_supported': int(support[i]),
                 'threshold_transition': 'lost' if support[i-1] and not support[i] else 'gained_or_recovered' if support[i] and not support[i-1] else 'unchanged_support_status',
                 'skips_provisional_argmax_readouts': int(before == 'after_hierarchy_duplicate_filter' and after == 'actual_semantic_argmax')})
    counts = Counter(r['paired_assignment_state'] for r in paired)
    assert counts == {'both_matched': 325, 'dbscan_only': 72, 'hpid_only': 29, 'neither_matched': 564}
    assert len(no_support) == 61 and len(gate_negative) == 195
    assert counts['both_matched']+counts['dbscan_only'] == 397
    assert counts['both_matched']+counts['hpid_only'] == 354
    write_csv('all_990_reference_pairing.csv', paired)
    write_csv('dbscan_only_72_references.csv', [r for r in paired if r['paired_assignment_state'] == 'dbscan_only'])
    write_csv('dbscan_only_hpid_no_geometry_61_actual_stage_tracks.csv', no_support)
    write_csv('dbscan_only_hpid_no_geometry_61_actual_stage_transitions.csv', detailed_changes)
    write_csv('detail_gate_195_negative_references.csv', gate_negative)
    stratified = []
    for dimension in ('domain','category'):
        for value in sorted({r[dimension] for r in paired}):
            subset = [r for r in paired if r[dimension] == value]
            cs = Counter(r['paired_assignment_state'] for r in subset)
            stratified.append({'group_by': dimension, 'group': value, 'n_references': len(subset),
                'n_cases': len({r['case_id'] for r in subset}), **{k:cs[k] for k in counts},
                'dbscan_only_hpid_no_geometry': sum(r['dbscan_only_subtype'] == 'no_hpid_geometry_at_threshold' for r in subset),
                'dbscan_only_hpid_geometry_not_matched': sum(r['dbscan_only_subtype'] == 'hpid_geometry_available_but_not_credited_by_saved_assignment' for r in subset)})
    write_csv('paired_subset_counts_by_domain_category.csv', stratified)
    transition_summary = []
    for before,after in zip(actual,actual[1:]):
        subset = [r for r in detailed_changes if r['before']==before and r['after']==after]
        transition_summary.append({'before': before, 'after': after, 'n_references': len(subset),
           'best_iou_decreased': sum(r['delta_best_iou'] < -EPS for r in subset),
           'best_iou_increased': sum(r['delta_best_iou'] > EPS for r in subset),
           'threshold_support_lost': sum(r['threshold_transition']=='lost' for r in subset),
           'threshold_support_gained_or_recovered': sum(r['threshold_transition']=='gained_or_recovered' for r in subset),
           'reference_mean_best_iou_delta_descriptive_only': sum(r['delta_best_iou'] for r in subset)/len(subset)})
    write_csv('no_geometry_subgroup_actual_transition_summary.csv', transition_summary)
    sums_by_stage = {r['stage']:r for r in sums}
    endpoints = {}
    for stage, method in [('final_hpid','hpid_split_a3'),('final_dbscan','dbscan_fusion')]:
        r = sums_by_stage[stage]
        assert abs(float(r['diagnostic_f1_at_025_mean']) - diagnosis['methods'][method]['part_f1_at_025']) < EPS
        endpoints[stage] = {k:r[k] for k in ['diagnostic_f1_at_025_mean','total_matches_at_025','total_prediction_count',
                     'diagnostic_semantic_f1_at_025_mean','best_iou_mean','best_semantic_iou_mean','semantic_union_recall_mean']}
    summary = {'status': 'PASS', 'created_utc': datetime.now(timezone.utc).isoformat(),
       'mode': 'Read-only arithmetic audit of existing CSV/JSON; no model, masks, rematching, scoring changes or new CI.',
       'counts': {'cases':226,'categories':60,'reference_parts':990,'loaded_accepted_proposals':1951,
                  'evaluated_input_pool_masks':1830,'root_rows_excluded_when_reference_has_no_body':121,
                  'shared_candidates_removed_on_refilter':0,'remainder_components_attached':0,
                  'visibility_slivers_dropped':14,'evaluated_masks_removed_at_final_filter':23},
       'pairing_at_iou_025': dict(counts),
       'dbscan_only_72_split': {'hpid_no_mask_meets_iou_025':61,'hpid_mask_available_but_not_credited_by_saved_hungarian_assignment':11},
       'no_geometry_61_dynamics': {'input_pool_supported':sum(r['input_pool_supported'] for r in no_support),
           'never_supported_in_any_observed_actual_stage':sum(not r['any_actual_stage_supported'] for r in no_support),
           'first_observed_loss_counts':dict(Counter(r['first_observed_actual_threshold_loss'] for r in no_support)),
           'last_observed_loss_counts':dict(Counter(r['last_observed_actual_threshold_loss'] for r in no_support)),
           'recovered_after_first_loss_count':sum(r['recovered_after_first_loss'] for r in no_support)},
       'detail_gate_negative_195_endpoint_partition':dict(Counter(r['paired_assignment_state'] for r in gate_negative)),
       'actual_stage_order_for_threshold_loss':actual,
       'excluded_provisional_readouts':[s for s in ordered if s.startswith('diagnostic_')],
       'endpoint_metrics':endpoints,
       'existing_category_cluster_bootstrap_hpid_minus_dbscan': diagnosis['category_cluster_bootstrap'],
       'bootstrap_note':'Existing reported interval retained; no CI recomputation requested or performed.',
       'arithmetic_checks': {'all_16830_reference_rows_complete':'PASS','all_3842_case_stage_metrics_from_reference_rows':'PASS',
           'all_stage_summary_means_totals':'PASS','all_stage_transition_means_and_reference_direction_counts':'PASS',
           'shared_input_counts_invariants':'PASS','endpoint_diagnosis_consistency':'PASS'},
       'semantic_geometric_distinctions':[
           'Main F1/TP are class-agnostic one-to-one Hungarian matching, with IoU>=.25 applied after saved assignment; not semantic-correct matching.',
           'best_iou is oracle maximum over any predicted mask and permits mask reuse across references. It is not Hungarian IoU or F1.',
           'best_semantic_iou and semantic_hungarian_iou additionally require evaluator-normalized semantic agreement.',
           'semantic_union_recall measures reference-pixel coverage by all same-semantic masks. It is not instance recall or one-to-one match count.',
           'The 195 gate-negative references refer to best_iou decreases, not 195 lost TPs or 195 decreases of semantic union recall (the latter is 63).',
           'Main summary means average references within each case, then cases. Subgroup dynamics are descriptive reference counts/means; references are not independent sampling units.'],
       'limits':[
           'Same input means the 1951 archived accepted fine-fusion candidates, not all original unfiltered proposals; original rejected proposal universe is unavailable.',
           'Actual semantic maps, per-class supports and exclusive identity masks are different representations; changes are sequential observations, not an additive causal decomposition of the DBSCAN gap.',
           'Four diagnostic_* argmax readouts use copied pre-argmax evidence plus a background convention. They are not emitted predictions and are excluded from first-loss classification.',
           'A pool-to-first-actual-map loss combines scoring/evidence changes and exclusive semantic ownership; existing actual snapshots cannot isolate its internal cause.',
           'Fine-fusion direct-detail gate is distinct from later three-stage semantic/structure/appearance verification in Groups; these traces do not analyze Groups.',
           'Best-mask identity may switch at each stage. A support loss may later recover. First and last loss are both recorded.',
           'The paired GT subset is directly identifiable by case_id and zero-based reference_index. Hungarian assigned prediction identity is not saved in these tracks (best_identity is a different field), so individual matching-conflict attribution needs additional pair records.',
           'No archived DBSCAN map identity check is available; historical audit establishes numeric evaluator reproduction, whereas all226 HPID maps were compared exactly.',
           'Historical stage_validation source-hash assertions describe execution-time sources. Packaged scripts have different paths/portability edits; this audit binds current packaged files without asserting original script-hash identity.'],
       'inputs_sha256': bindings, 'code_read_sha256': {str(CODE):sha(CODE)},
       'script_sha256':sha(__file__)}
    write_json('summary.json', summary)
    write_json('receipt.json', {'status':'COMPLETE','summary_sha256':sha(OUT/'summary.json'),
       'input_sha256':bindings,
       'artifacts_sha256':{p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='receipt.json'}})
    assert bindings == {name:sha(EVIDENCE/name) for name in inputs}
    print(json.dumps({k:summary[k] for k in ['status','pairing_at_iou_025','dbscan_only_72_split','no_geometry_61_dynamics','detail_gate_negative_195_endpoint_partition']},indent=2))
    print('receipt_sha256',sha(OUT/'receipt.json'))


if __name__ == '__main__':
    main()
