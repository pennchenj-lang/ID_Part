"""Post-hoc, CPU-only audit of sealed routing outcomes; no model or policy changes.

GT is read only here, after prediction sealing, to measure observed stage masks.
Availability is existential: matching semantic AND IoU >= the frozen threshold.
Stage transitions localize loss but do not establish an isolated causal ablation.
"""
from __future__ import annotations
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

sys.dont_write_bytecode = True
SOURCE = Path('__RUNTIME_ROOT__/code_snapshots/hpid_split_be54300_holdout/src')
sys.path.insert(0, str(SOURCE))
from hpid_split.paco_eval import _normalize

OUT = Path(__file__).resolve().parent
ROOT = OUT.parent
THRESHOLDS = {'025': .25, '050': .50}
METHODS = ['raw_proposals', 'greedy_nms', 'dbscan_fusion', 'local_pairwise_crf', 'hpid_split_group_ids']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def csvwrite(path, data):
    fields = list(dict.fromkeys(k for row in data for k in row))
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def mask(path, binary=False):
    with Image.open(path) as im:
        return (np.asarray(im.convert('L')) >= 128) if binary else np.asarray(im).copy()


def binary_sha(a, packed=False):
    a = np.ascontiguousarray(a, dtype=np.uint8)
    return hashlib.sha256(np.packbits(a.ravel()).tobytes() if packed else a.tobytes()).hexdigest()


def iou(a, b):
    intersection = int(np.count_nonzero(a & b))
    return intersection / max(1, int(np.count_nonzero(a | b)))


def norm(name, request):
    domain = request['expected_domain']
    if name == domain:
        return 'body'
    return _normalize(name.removeprefix(domain + '_'), domain, object_category=request['object_category'])


def fingerprint(c, m):
    payload = {k: c[k] for k in ('semantic_name', 'semantic_parent', 'source')}
    payload.update(prompt=c.get('prompt', ''), score=float(c['score']),
                   source_reliability=float(c.get('source_reliability', 1)),
                   mask_sha256=binary_sha(m, True), metadata=c.get('metadata') or {})
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def verification_rows(obj, key, path=''):
    found = []
    if isinstance(obj, dict):
        if obj.get('candidate_key') == key:
            found.append({'json_path': path, 'record': obj})
        else:
            for k, value in obj.items():
                found.extend(verification_rows(value, key, path + '/' + k))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            found.extend(verification_rows(value, key, path + '/' + str(index)))
    return found


def destinations(candidate_mask, records, labels, truth, identity_key, index_key, request):
    result = []
    for record in records:
        current = labels == int(record[index_key])
        overlap = int(np.count_nonzero(candidate_mask & current))
        if overlap:
            result.append({'identity': record[identity_key], 'semantic': record['semantic_name'],
                           'normalized_semantic': norm(record['semantic_name'], request),
                           'candidate_overlap_pixels': overlap,
                           'truth_overlap_pixels': int(np.count_nonzero(truth & candidate_mask & current)),
                           'final_target_iou': iou(current, truth)})
    result.sort(key=lambda row: -row['candidate_overlap_pixels'])
    return result


def stage_loss_label(raw_available, accepted_available, part_available, group_available):
    if not raw_available or group_available:
        return 'not_a_raw_to_group_loss'
    if not accepted_available:
        return 'candidate_list_filter_subtype_unresolved'
    if not part_available:
        return 'accepted_to_native_part_ownership_gate_cleanup_unresolved'
    return 'native_part_to_group'


def main():
    manifest_path = ROOT/'request_manifest_frozen.json'
    case_path = ROOT/'routing_run/routing_cases.csv'
    instance_path = ROOT/'routing_run/routing_instances.csv'
    receipt_path = ROOT/'routing_run/receipt.json'
    receipt = read(receipt_path)
    assert receipt['status'] == 'COMPLETE' and not receipt['failures']
    assert sha(manifest_path) == receipt['manifest_sha256']
    assert sha(case_path) == receipt['routing_csv_sha256']
    assert sha(instance_path) == receipt['instances_csv_sha256']
    requests = read(manifest_path)['requests']
    routing = rows(case_path)
    instances = rows(instance_path)
    by_case = defaultdict(dict)
    by_instances = defaultdict(list)
    for row in routing:
        by_case[row['request_id']][row['method']] = row
    for row in instances:
        by_instances[(row['request_id'], row['method'])].append(row)
    assert len(requests) == len(by_case) == 253
    assert all(set(value) == set(METHODS) for value in by_case.values())
    results, losses, candidates_output, case_bindings = [], [], [], []
    for index, request in enumerate(requests):
        rid = request['request_id']
        common = ROOT/'common_groups'/rid
        package = Path(request['package_path'])
        replay = read(common/'replay_receipt.json')
        for name, details in replay['outputs'].items():
            assert sha(common/name) == details['sha256'], (rid, name)
        for name in ('candidates.json', 'parts.json', 'part_id_map.tiff'):
            assert sha(package/name) == replay['input_sha256'][name], (rid, name)
        assert sha(request['target_mask_path']) == request['target_mask_sha256']
        tracepath = ROOT/'routing_run/case_traces'/f'{rid}.json'
        assert sha(tracepath) == receipt['artifact_sha256'][str(Path('case_traces')/f'{rid}.json')]
        trace = read(tracepath)
        truth = mask(request['target_mask_path'], True)
        assert int(truth.sum()) == request['target_area_px']
        part_map = mask(package/'part_id_map.tiff')
        group_map = mask(common/'group_id_map.tiff')
        parts = read(common/'parts_with_groups.json')
        archive_parts = {r['part_id']: r for r in read(package/'parts.json')}
        for p in parts:
            original = archive_parts[p['part_id']]
            assert all(p[k] == original[k] for k in ('semantic_name', 'instance_index', 'area_px'))
        assert hashlib.sha256(np.ascontiguousarray(part_map).tobytes()).hexdigest() == replay['archived_part_map_pixels_sha256']
        groups = read(common/'groups.json')
        accepted = read(common/'accepted_candidates_metadata.json')
        accepted_by_key = {r['candidate_key']: r for r in accepted}
        fd = read(common/'fused_diagnostics.json')
        gd = read(common/'grouping_diagnostics.json')
        raw = by_case[rid]['raw_proposals']
        hpid = by_case[rid]['hpid_split_group_ids']
        target = raw['normalized_target']
        assert target == _normalize(request['target_part_name'], request['expected_domain'], object_category=request['object_category'])
        stage_parts = []
        for p in parts:
            current = part_map == int(p['instance_index'])
            stage_parts.append({'part_id': p['part_id'], 'group_id': p['group_id'],
                                'semantic': p['semantic_name'], 'normalized_semantic': norm(p['semantic_name'], request),
                                'target_iou': iou(current, truth), 'area_px': int(current.sum())})
        stage_groups = []
        actual_group_instances = {r['identity']: r for r in by_instances[(rid, 'hpid_split_group_ids')]}
        for g in groups:
            current = group_map == int(g['group_index'])
            val = iou(current, truth)
            observed = actual_group_instances[g['group_id']]
            assert abs(val - float(observed['target_iou'])) < 1e-12
            assert binary_sha(current) == observed['mask_sha256']
            assert norm(g['semantic_name'], request) == observed['normalized_semantic']
            stage_groups.append({'group_id': g['group_id'], 'semantic': g['semantic_name'],
                                 'normalized_semantic': norm(g['semantic_name'], request),
                                 'member_part_ids': g['member_part_ids'], 'target_iou': val,
                                 'area_px': int(current.sum()), 'evidence': g.get('evidence')})
        raw_records = read(package/'candidates.json')
        raw_matches = [r for r in by_instances[(rid, 'raw_proposals')] if r['matches_target'] == '1']
        all_match_rows = []
        loss_any = any(int(raw['semantic_correct_any_match_'+t]) and not int(hpid['semantic_correct_any_match_'+t]) for t in THRESHOLDS)
        detail_candidates = []
        for r in raw_matches:
            ci = int(r['identity'].split('/')[-1]) - 1
            c = raw_records[ci]
            cp = package/c['mask_path']
            expected_mask_sha = next(x['sha256'] for x in replay['candidate_mask_files'] if x['path'] == c['mask_path'])
            assert sha(cp) == expected_mask_sha
            cm = mask(cp, True)
            assert binary_sha(cm) == r['mask_sha256']
            assert abs(iou(cm, truth) - float(r['target_iou'])) < 1e-12
            assert norm(c['semantic_name'], request) == target
            key = fingerprint(c, cm)
            am = accepted_by_key.get(key)
            cr = {'request_id': rid, 'cohort': request['cohort'], 'raw_identity': r['identity'],
                  'semantic': r['semantic'], 'target_iou': float(r['target_iou']),
                  'mask_area': int(cm.sum()), 'raw_candidate_index_zero_based': ci,
                  'upstream_candidate_key': (c.get('metadata') or {}).get('candidate_key'),
                  'fingerprint': key, 'accepted_by_fuser': bool(am),
                  'accepted_index': am['index'] if am else None}
            all_match_rows.append(cr)
            if loss_any and float(r['target_iou']) >= .25:
                pd = destinations(cm, parts, part_map, truth, 'part_id', 'instance_index', request)
                gg = destinations(cm, groups, group_map, truth, 'group_id', 'group_index', request)
                known_trace = next(z for z in trace['raw_target_candidate_pixel_destinations'] if z['raw_identity'] == r['identity'])
                assert {z['identity']: z for z in gg} == {z['identity']: z for z in known_trace['destinations']}
                checks = verification_rows(gd, cr['upstream_candidate_key']) if cr['upstream_candidate_key'] else []
                detail_candidates.append({**cr, 'native_part_pixel_destinations': pd,
                                          'final_group_pixel_destinations': gg,
                                          'unowned_part_pixels': int(np.count_nonzero(cm & (part_map == 0))),
                                          'unowned_group_pixels': int(np.count_nonzero(cm & (group_map == 0))),
                                          'grouping_verification_records': checks})
            candidates_output.append(cr)
        row = {'request_id': rid, 'cohort': request['cohort'], 'domain': request['domain'],
               'object_category': request['object_category'], 'target': request['target_part_name'],
               'normalized_target': target, 'raw_query_state': raw['query_state'],
               'hpid_query_state': hpid['query_state'], 'raw_match_count': int(raw['query_match_count']),
               'hpid_match_count': int(hpid['query_match_count']),
               'raw_best_matching_iou': float(raw['best_review_match_iou']),
               'accepted_best_matching_iou': max((r['target_iou'] for r in all_match_rows if r['accepted_by_fuser']), default=0),
               'native_part_best_matching_iou': max((p['target_iou'] for p in stage_parts if p['normalized_semantic'] == target), default=0),
               'group_best_matching_iou': float(hpid['best_review_match_iou']),
               'native_part_best_any_semantic_iou': max((p['target_iou'] for p in stage_parts), default=0),
               'group_best_any_semantic_iou': float(hpid['best_exported_iou'])}
        for t, threshold in THRESHOLDS.items():
            raw_available = int(raw['semantic_correct_any_match_'+t])
            accepted_available = int(row['accepted_best_matching_iou'] >= threshold)
            part_available = int(row['native_part_best_matching_iou'] >= threshold)
            group_available = int(hpid['semantic_correct_any_match_'+t])
            assert raw_available == int(row['raw_best_matching_iou'] >= threshold)
            first = stage_loss_label(raw_available, accepted_available, part_available, group_available)
            subtype = 'not_applicable'
            correct_parts = [p for p in stage_parts if p['normalized_semantic'] == target and p['target_iou'] >= threshold]
            observed_group_fates = []
            if first == 'native_part_to_group':
                for p in correct_parts:
                    original = next(x for x in parts if x['part_id'] == p['part_id'])
                    pd = destinations(part_map == int(original['instance_index']), groups, group_map, truth, 'group_id', 'group_index', request)
                    dominant = pd[0] if pd else None
                    kind = 'group_geometry_or_cleanup_unresolved'
                    if dominant and dominant['normalized_semantic'] != target:
                        kind = 'dominant_pixels_assigned_to_different_semantic_group'
                    elif dominant:
                        g = next(g for g in groups if g['group_id'] == dominant['identity'])
                        if len(g['member_part_ids']) > 1:
                            kind = 'same_semantic_group_merge_or_geometry_change'
                    observed_group_fates.append({'part_id': p['part_id'], 'part_target_iou': p['target_iou'],
                                                'observed_subtype': kind, 'destinations': pd})
                subtype = '|'.join(sorted({x['observed_subtype'] for x in observed_group_fates}))
            elif first.startswith('accepted_to_native'):
                subtype = ('correct_geometry_exists_under_other_native_part_semantic'
                           if row['native_part_best_any_semantic_iou'] >= threshold else 'no_individual_native_part_meets_geometry_threshold')
            row.update({f'raw_available_{t}': raw_available, f'accepted_available_{t}': accepted_available,
                        f'native_part_available_{t}': part_available, f'group_available_{t}': group_available,
                        f'raw_to_group_lost_{t}': int(raw_available and not group_available),
                        f'first_observed_loss_stage_{t}': first, f'observed_subtype_{t}': subtype,
                        f'hpid_wrong_unique_{t}': int(hpid['wrong_unique_'+t]),
                        f'hpid_any_geometry_available_{t}': int(hpid['geometry_correct_any_exported_'+t])})
            if raw_available and not group_available:
                losses.append({'request_id': rid, 'cohort': request['cohort'], 'domain': request['domain'],
                               'object_category': request['object_category'], 'target': target, 'threshold': t,
                               'raw_query_state': raw['query_state'], 'hpid_query_state': hpid['query_state'],
                               'first_observed_loss_stage': first, 'observed_subtype': subtype,
                               'raw_best_iou': row['raw_best_matching_iou'],
                               'accepted_best_iou': row['accepted_best_matching_iou'],
                               'part_best_iou': row['native_part_best_matching_iou'],
                               'group_best_iou': row['group_best_matching_iou'],
                               'group_any_semantic_best_iou': row['group_best_any_semantic_iou'],
                               'part_to_group_fates': observed_group_fates})
        if loss_any:
            write(OUT/'cases'/f'{rid}.json', {'request': request, 'summary': row,
                  'correct_raw_candidates_at_either_threshold': detail_candidates,
                  'native_parts': stage_parts, 'final_groups': stage_groups,
                  'fused_diagnostics': fd, 'grouping_diagnostics_file': str(common/'grouping_diagnostics.json'),
                  'group_stage_losses': [x for x in losses if x['request_id'] == rid],
                  'interpretation_limit': 'Observed stage masks and logged decisions. No per-pixel pre/post gate or cleanup maps; no isolated causal ablation.'})
        results.append(row)
        case_bindings.append({'request_id': rid, 'replay_receipt_sha256': sha(common/'replay_receipt.json'),
                              'case_trace_sha256': sha(tracepath), 'target_mask_sha256': sha(request['target_mask_path'])})
        if (index + 1) % 50 == 0:
            print('Audited', index + 1, 'of', len(requests), flush=True)

    # Fixed-subset transition decomposition avoids confusing changing conditional denominators.
    transitions, conditional, retention, stage_counts, risks = [], [], [], [], []
    for cohort in ['old37', 'new216', 'combined']:
        rs = [r for r in results if cohort == 'combined' or r['cohort'] == cohort]
        for t, threshold in THRESHOLDS.items():
            for method in ['raw_proposals', 'hpid_split_group_ids']:
                subset = [by_case[r['request_id']][method] for r in rs]
                n = sum(int(x['query_ambiguous']) for x in subset)
                k = sum(int(x['correct_in_review_set_'+t]) for x in subset)
                conditional.append({'cohort': cohort, 'threshold': t, 'method': method,
                                    'ambiguous_n': n, 'correct_in_ambiguous_k': k, 'rate': k/n if n else None})
            retention.append({'cohort': cohort, 'threshold': t, 'n': len(rs),
                              'raw_available': sum(r['raw_available_'+t] for r in rs),
                              'group_available': sum(r['group_available_'+t] for r in rs),
                              'retained': sum(r['raw_available_'+t] and r['group_available_'+t] for r in rs),
                              'lost': sum(r['raw_to_group_lost_'+t] for r in rs),
                              'gained': sum(not r['raw_available_'+t] and r['group_available_'+t] for r in rs)})
            counted = Counter(r['first_observed_loss_stage_'+t] for r in rs if r['raw_to_group_lost_'+t])
            for stage, count in sorted(counted.items()):
                stage_counts.append({'cohort': cohort, 'threshold': t, 'stage': stage, 'count': count})
            for subset_name in ['raw_ambiguous', 'raw_ambiguous_with_correct_option']:
                subset = [r for r in rs if r['raw_query_state'] == 'ambiguous'
                          and (subset_name == 'raw_ambiguous' or r['raw_available_'+t])]
                counts = Counter()
                for r in subset:
                    h = by_case[r['request_id']]['hpid_split_group_ids']
                    if h['query_state'] == 'unique':
                        state = 'correct_unique' if int(h['unique_and_correct_'+t]) else 'wrong_unique'
                    elif h['query_state'] == 'ambiguous':
                        state = 'ambiguous_with_correct' if r['group_available_'+t] else 'ambiguous_without_correct'
                    else:
                        state = 'unresolved_with_correct_fallback' if int(h['geometry_correct_any_exposed_'+t]) else 'unresolved_without_correct_fallback'
                    counts[state] += 1
                for state in ['correct_unique', 'wrong_unique', 'ambiguous_with_correct', 'ambiguous_without_correct', 'unresolved_with_correct_fallback', 'unresolved_without_correct_fallback']:
                    transitions.append({'cohort': cohort, 'threshold': t, 'subset': subset_name,
                                        'to_hpid_state': state, 'n': len(subset), 'k': counts[state]})
        for dimension in ['domain', 'object_category']:
            for value in sorted({r[dimension] for r in rs}):
                subset = [r for r in rs if r[dimension] == value]
                for method in METHODS:
                    mr = [by_case[r['request_id']][method] for r in subset]
                    for t in THRESHOLDS:
                        unique = sum(int(x['query_unique']) for x in mr)
                        wrong = sum(int(x['wrong_unique_'+t]) for x in mr)
                        risks.append({'cohort': cohort, 'group_by': dimension, 'group': value,
                                      'method': method, 'threshold': t, 'n_all': len(mr),
                                      'n_unique': unique, 'wrong_unique': wrong,
                                      'wrong_unique_over_all': wrong/len(mr),
                                      'wrong_unique_given_unique': wrong/unique if unique else None,
                                      'sparse_n_lt_10': len(mr) < 10})
    existing = rows(ROOT/'statistics/raw_ambiguous_fixed_subset_flow.csv')
    lookup = {(x['cohort'], x['threshold'], x['subset'], x['to_hpid_state']): x for x in existing}
    for row in transitions:
        old = lookup[tuple(row[k] for k in ('cohort', 'threshold', 'subset', 'to_hpid_state'))]
        assert (row['n'], row['k']) == (int(old['n']), int(old['k']))
    # Keep complete sealed method outcomes, including all unfavorable and no-raw-option cases.
    csvwrite(OUT/'all_case_stage_availability.csv', results)
    csvwrite(OUT/'all_method_outcomes.csv', routing)
    csvwrite(OUT/'all_target_matching_raw_candidates.csv', candidates_output)
    csvwrite(OUT/'lost_cases.csv', [{k:v for k,v in r.items() if k != 'part_to_group_fates'} for r in losses])
    csvwrite(OUT/'wrong_unique_by_domain_category.csv', risks)
    csvwrite(OUT/'fixed_raw_ambiguous_transitions.csv', transitions)
    csvwrite(OUT/'conditional_review_rates.csv', conditional)
    write(OUT/'input_case_bindings.json', case_bindings)
    stage_patterns = []
    for t, threshold in THRESHOLDS.items():
        selected_losses = [r for r in losses if r['threshold'] == t]
        stage_patterns.append({'threshold': t, 'lost_n': len(selected_losses),
            'final_geometry_correct_under_different_semantic': sum(r['group_any_semantic_best_iou'] >= threshold for r in selected_losses),
            'subtype_counts': dict(Counter(r['observed_subtype'] for r in selected_losses)),
            'domain_counts': dict(Counter(r['domain'] for r in selected_losses))})
    source_references = {
        'raw_export': 'postprocess_baselines.py raw_proposals: original candidate index+1, area>=6',
        'candidate_list_filter': 'fusion.py:1485-1492 _deduplicate -> _filter_candidates_by_instance_caps -> _suppress_hierarchical_duplicates',
        'pixel_gate': 'fusion.py:1769-1796 direct evidence gate changes pixel owner labels; does not remove entries from accepted_candidates',
        'cleanup': 'fusion.py:1814 _clean_labels plus root conservation and later identity extraction',
        'group_builder': 'physical_groups.py:4727 build_physical_groups; observed native Part and final Group masks compared directly',
        'semantic_normalizer': 'paco_eval.py:17 _normalize plus benchmark _normalized_name: expected-domain prefix removal and object-category aliases'}
    report = '''# R6 mechanism and stratified failure audit

All 253 requests and all five methods are retained in the audit tables. Predictions, thresholds, frozen manifests and routing outputs are unchanged. This is a post-inference descriptive audit; ground truth is read only to measure already sealed masks. Primary IoU is 0.25; IoU 0.50 is secondary.

## Where correct options become unavailable

| Threshold | Raw available | HPID Group available | Retained | Lost | Gained | First loss: accepted candidate to Part | First loss: Part to Group |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 71 | 48 | 46 | 25 | 2 | 24 | 1 |
| 0.50 | 41 | 23 | 21 | 20 | 2 | 20 | 0 |

Every lost request still has at least one geometrically correct, semantically matching raw candidate in the fuser's accepted list. Thus none is localized first to candidate-list filtering. At 0.25, 17 of the 24 accepted-to-Part losses retain a correct individual Part geometry under a different semantic name; seven have no individual Part reaching the threshold. At final Group export, 15/25 losses retain correct geometry under another name. At 0.50 this final count is 7/20. These are losses of a matching option, not necessarily removal of its pixels.

Accepted-to-Part is deliberately a combined unresolved stage: the logs do not contain the intermediate pixel-label maps needed to isolate direct evidence gates, ownership competition, identity extraction and cleanup. A global suppression counter or a later verification rejection is not evidence that it alone caused a particular loss.

The single primary-threshold Part-to-Group loss is `new216__laptop_computer__01`: base-panel raw IoU 0.7184, native Part IoU 0.2998, then no matching Group. All 6,799 pixels of that Part are assigned to a bezel Group, which also lists the base-panel Part among its four members. This supports a Group merge/semantic reassignment description. At 0.50 the same case already fails at the native Part stage.

## Why correct-in-review changes

| Cohort | Raw correct / ambiguous | HPID correct / ambiguous | Fate of the fixed raw-correct ambiguous subset at 0.25 |
|---|---:|---:|---|
| old37 | 9/13 | 1/6 | 5 correct unique; 1 correct ambiguous; 2 wrong unique; 1 ambiguous without a correct option |
| new216 | 24/44 | 14/24 | 6 correct unique; 14 correct ambiguous; 1 ambiguous without a correct option; 3 unresolved (one retains correct geometry in the fallback set) |
| combined | 33/57 | 15/30 | 11 correct unique; 15 correct ambiguous; 2 wrong unique; 2 ambiguous without a correct option; 3 unresolved |

These conditional denominators are different subsets. The old37 change 9/13 to 1/6 does not mean eight options were destroyed: five requests resolve correctly, one remains correct in review, and three lose a matching correct option. A fourth old37 loss starts from a raw unique request. Secondary-threshold conditional results and fixed-subset flows are preserved in the CSV/JSON outputs.

All four old37 primary-threshold losses are documented, without selection:

| Request | Original state | Raw best matching IoU | Part best matching IoU | Group best matching IoU | Observed pixel fate |
|---|---|---:|---:|---:|---|
| box__u01 / side | ambiguous | 0.5415 | 0.0070 | 0.0069 | 9,010 of the correct raw proposal's 10,073 pixels belong to a bottom Part; bottom Group still has IoU 0.5315 |
| car_automobile__u02 / windshield | ambiguous | 0.7048 | 0.2203 | 0.2171 | 348 candidate pixels belong to window, 206 to windshield at Part export; window IoU is 0.3722 |
| scissors__u01 / handle | ambiguous | 0.2546 | 0.1419 | 0.1419 | Candidate pixels split among handle (4,632), body (3,523) and other Parts; body Group IoU is 0.3553 |
| shoe__u01 / upper | unique | 0.4350 | 0.0934 | 0.2069 | Candidate pixels spread across body and visual/detail Parts; Group merging recovers some upper area but remains below 0.25 |

The three highest raw-IoU primary-threshold losses, an outcome-independent-of-narrative ranking within all losses, are jar__01 (lid, 0.9510), jar__04 (lid, 0.8891) and clock__03 (base, 0.8335). Each correct candidate is accepted, but no matching native Part survives. The jars retain final geometry under another semantic (best-any IoU 0.9485 and 0.3555); clock does not (0.1082). Full candidate-to-Part/Group destinations and original logged verifications are in their case JSON files.

## Wrong unique results by domain

The denominator is shown both for all requests and for unique results. These descriptive counts are not significance tests and do not imply domain-specific generalization.

| Domain | Wrong unique at 0.25 | All requests | Unique results | Wrong among unique |
|---|---:|---:|---:|---:|
| container | 12 | 80 | 22 | 54.5% |
| daily_object | 7 | 37 | 7 | 100% |
| device | 6 | 66 | 18 | 33.3% |
| tool_prop | 6 | 38 | 12 | 50.0% |
| furniture | 3 | 20 | 4 | 75.0% |
| vehicle | 2 | 12 | 6 | 33.3% |

Combined HPID has 36 wrong unique outcomes among 69 unique outputs (36/253 among all requests), compared with raw 39/77 (39/253 overall). Counts for greedy NMS, DBSCAN fusion and local pairwise CRF are respectively 38/72, 45/84 and 36/63. Reduced ambiguity therefore does not by itself establish safer unique routing. At category level bucket contributes 3 wrong unique of 4 requests; box, car, hat, knife, scissors, table, telephone and towel each contribute 2. These category sample sizes are only 3–7, so the full category table is exploratory rather than a ranked performance claim. Both thresholds, all methods and both cohorts are retained in `wrong_unique_by_domain_category.csv`.

## Evidence and reproducibility

`all_case_stage_availability.csv` contains all 253 requests; `all_method_outcomes.csv` retains all 1,265 sealed method rows; `all_target_matching_raw_candidates.csv` retains every matching raw candidate. `lost_cases.csv` has 45 case-threshold rows, corresponding to 30 unique requests. Each of the 30 case JSON files contains native Part/Group IoUs, exact accepted-candidate fingerprints, observed candidate pixel destinations and relevant original verification records. Input and output SHA-256 bindings are in `summary.json` and `receipt.json`.

The audit checks every common-Group artifact against the preparation receipt, the archived native Part map hash against the already exact replay, all Group mask hashes/IoUs against sealed routing instances, all target-matching raw masks/IoUs against sealed routing instances, loss candidate destinations against sealed traces, and all fixed-subset flows against the independent statistics. No model execution, new counterfactual experiment or parameter adjustment was performed.
'''
    (OUT/'findings.md').write_text(report, encoding='utf-8')
    summary = {'status': 'PASS', 'created_utc': datetime.now(timezone.utc).isoformat(),
               'scope': 'Post-hoc descriptive mechanism and risk audit; primary IoU=.25, secondary IoU=.50.',
               'cases': len(results), 'method_rows': len(routing),
               'lost_case_union_count': len({r['request_id'] for r in losses}),
               'first_observed_loss_stage_counts': stage_counts, 'retention': retention,
               'observed_loss_patterns': stage_patterns, 'source_references': source_references,
               'conditional_correct_in_review': conditional, 'fixed_subset_transitions': transitions,
               'bindings': {str(p): sha(p) for p in [Path(__file__), manifest_path, case_path, instance_path, receipt_path,
                   ROOT/'common_groups/preparation_receipt.json', ROOT/'statistics/analysis_summary.json', SOURCE/'hpid_split/paco_eval.py', SOURCE/'hpid_split/paco_semantics.py']},
               'verification': {'sealed_routing_hashes': 'PASS', 'all_253_common_group_files': 'PASS',
                                'native_part_maps_previously_exact_replay_and_archive_hashes': 'PASS',
                                'all_group_mask_hashes_and_ious_equal_sealed_routing_instances': 'PASS',
                                'all_raw_matching_mask_hashes_and_ious_equal_sealed_routing_instances': 'PASS',
                                'lost_candidate_pixel_destinations_equal_sealed_traces': 'PASS',
                                'fixed_raw_ambiguous_flows_equal_independent_statistics': 'PASS'},
               'limitations': [
                   'Candidate-list acceptance is matched by complete candidate fingerprint including metadata and mask hash.',
                   'First observed loss identifies an interface boundary, not an isolated causal contribution; later stages may recover options.',
                   'Accepted-to-Part loss cannot be separated into direct evidence gate, pixel ownership, identity extraction or cleanup without intermediate maps/ablation. It is explicitly unresolved.',
                   'Group verification rejection is a logged decision, not proof it alone caused an earlier fusion loss. Global counters are not candidate-specific causes.',
                   'Semantic nonmatch may retain correct geometry under another label and is not equivalent to deleting pixels.',
                   'Correct-in-review denominators are method-specific ambiguous subsets; fixed-raw-subset transitions explain their changes.',
                   'Domain/category risk aggregates are descriptive with sparse categories; no significance claims or threshold tuning.',
                   'GT used only for this sealed post-inference audit; no prediction, strategy, threshold, model, Word, frozen input or sealed output modified.'],
               'GPU_used': False, 'model_execution': False,
               'output_sha256': {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ('summary.json', 'receipt.json')}}
    write(OUT/'summary.json', summary)
    write(OUT/'receipt.json', {'status': 'COMPLETE', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'cases': len(results), 'lost_case_union': len({r['request_id'] for r in losses}),
        'case_threshold_loss_rows': len(losses), 'audit_status': 'PASS',
        'input_hashes': summary['bindings'],
        'artifact_sha256': {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'receipt.json'}})
    print(json.dumps({'status': 'PASS', 'retention': retention, 'stage_counts': stage_counts,
                      'summary_sha256': sha(OUT/'summary.json')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
