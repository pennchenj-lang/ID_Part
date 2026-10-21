"""Independent audit; no stress production helper is imported or modified."""
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
RELIABILITY = HERE.parent / 'reliability_audit'
RUNTIME = Path('__RUNTIME_ROOT__')
SNAPSHOT = RUNTIME / 'code_snapshots/hpid_split_be54300_holdout'
BENCH = RUNTIME / 'experiments/paper_v031/test226_hpid'
sys.dont_write_bytecode = True
sys.path.insert(0, str(SNAPSHOT / 'src'))
from hpid_split import fusion
from hpid_split.paco_eval import _normalize
import cv2
cv2.setNumThreads(1)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def mask_sha(mask):
    return hashlib.sha256(np.asarray(mask, dtype=np.uint8).tobytes()).hexdigest()

def binary(path):
    return np.asarray(Image.open(path).convert('L')) >= 128

def load_candidates(package):
    return [fusion.MaskCandidate(semantic_name=c['semantic_name'], semantic_parent=c['semantic_parent'],
                                 mask=binary(package / c['mask_path']), score=float(c['score']), source=c['source'],
                                 prompt=c.get('prompt', ''), source_reliability=float(c.get('source_reliability', 1)),
                                 metadata=dict(c.get('metadata') or {})) for c in read(package / 'candidates.json')]

protocol = read(HERE / 'protocol.json')
post = read(HERE / 'post_run_receipt.json')
completion = read(HERE / 'completion_receipt.json')
clarification = read(HERE / 'protocol_clarification.json')
for name, digest in post['files'].items():
    assert sha(HERE / name) == digest, name
prior_mismatches = [name for name, digest in completion['files'].items() if sha(HERE / name) != digest]
assert prior_mismatches == ['run.log'] == post['in_process_receipt_mismatches']
assert sha(HERE / 'protocol.json') == read(HERE / 'freeze_receipt.json')['protocol_sha256']
assert sha(HERE / 'run_stress.py') == protocol['script_sha256']
assert sha(Path(fusion.__file__)) == protocol['fusion_sha256']
assert clarification['analysis_changed'] is False and clarification['original_protocol_preserved'] is True
audits = rows(HERE / 'case_audit.csv')
raw = rows(HERE / 'case_results.csv')
summary = rows(HERE / 'summary.csv')
pairs = rows(HERE / 'paired_comparisons.csv')
assert len(audits) == len({r['case_id'] for r in audits}) == 226
assert sum(int(r['candidate_count']) for r in audits) == 1951
assert all(float(r['consensus_max_abs_difference']) == 0 for r in audits)
eligible = sorted(r['case_id'] for r in audits if r['eligible'] == 'True')
assert len(eligible) == 150 and len(raw) == 6000
assert Counter(r['case_id'] for r in raw) == Counter({cid: 40 for cid in eligible})
index = {(r['case_id'], r['mode'], r['r_level'], r['scenario']): r for r in raw}
assert len(index) == len(raw)
MODES = protocol['modes']
LEVELS = [str(r) for r in protocol['r_levels']]
METRICS = ['wrong_gt_pairwise_dominance', 'wrong_top_nonroot', 'correct_top_nonroot', 'wrong_evidence', 'correct_evidence']
expected = {(cid, mode, 'clean', 'clean') for cid in eligible for mode in MODES}
expected.update((cid, mode, level, scenario) for cid in eligible for mode in MODES for level in LEVELS for scenario in protocol['conditions'][1:])
assert set(index) == expected
for cid in eligible:
    for mode in MODES:
        clean = index[cid, mode, 'clean', 'clean']
        for level in LEVELS:
            single = index[cid, mode, level, 'single']
            same = index[cid, mode, level, 'same_family_5']
            cross = index[cid, mode, level, 'cross_family_5']
            for metric in METRICS:
                if mode != 'candidate_noisy_or':
                    assert float(single[metric]) == float(same[metric])
                else:
                    assert float(same[metric]) == float(cross[metric])
                for row in [single, same, cross]:
                    assert abs(float(row['delta_clean_' + metric]) - (float(row[metric]) - float(clean[metric]))) < 1e-14
                if mode == 'family_max_uniform_r':
                    for scenario in protocol['conditions'][1:]:
                        assert float(index[cid, mode, level, scenario][metric]) == float(index[cid, mode, LEVELS[0], scenario][metric])

# Independent cluster resampling via category-sum indexing, not root's case weights.
categories = sorted({r['category'] for r in audits})
assert len(categories) == 60
by_case = {r['case_id']: r for r in audits}
groups = [np.asarray([i for i, cid in enumerate(eligible) if by_case[cid]['category'] == cat], dtype=int) for cat in categories]
draws = np.random.default_rng(20261010).integers(60, size=(10000, 60))
denoms = np.asarray([len(g) for g in groups])[draws].sum(axis=1)
assert np.all(denoms > 0)
max_stat_error = 0.0
stat_checks = 0

def check_stats(row, values):
    global max_stat_error, stat_checks
    values = np.asarray(values, float)
    cat_sums = np.asarray([values[g].sum() for g in groups])
    boot = cat_sums[draws].sum(axis=1) / denoms
    computed = (values.mean(), np.quantile(boot, .025), np.quantile(boot, .975))
    for key, value in zip(['mean', 'ci95_low', 'ci95_high'], computed):
        error = abs(float(row[key]) - value)
        max_stat_error = max(max_stat_error, error)
        assert error < 1e-12, (row, key, error)
        stat_checks += 1

for row in summary:
    check_stats(row, [float(index[cid, row['mode'], row['r_level'], row['scenario']][row['metric']]) for cid in eligible])
for row in pairs:
    if row['contrast'].endswith(' minus single'):
        scenario = row['contrast'].split(' minus ')[0]
        values = [float(index[cid, row['mode'], row['r_level'], scenario][row['metric']]) - float(index[cid, row['mode'], row['r_level'], 'single'][row['metric']]) for cid in eligible]
    else:
        assert row['contrast'] == 'candidate_noisy_or minus family_max_noisy_or'
        values = [float(index[cid, 'candidate_noisy_or', row['r_level'], row['scenario']][row['metric']]) - float(index[cid, 'family_max_noisy_or', row['r_level'], row['scenario']][row['metric']]) for cid in eligible]
    check_stats(row, values)

bench = read(BENCH / 'benchmark_summary.json')
manifest = {r['case_id']: r for r in read(bench['source_manifest'])['cases']}
assert sha(bench['source_manifest']) == protocol['input_manifest_sha256']
details = {r['case_id']: r for r in bench['cases'] if r['return_code'] == 0}
fixed_samples = eligible[:3]
pixel_audits = []
selection_verified = 0

def weight(c, uniform=False):
    value = float(np.clip((.45 + .55 * float(np.clip(c.score, 0, 1))) * (1. if uniform else c.source_reliability), .01, .995))
    if fusion._is_broad_scene_layer(c):
        value = float(np.clip(value * .72, .01, .995))
    return value

def independent_aggregate(entries, nclasses, shape, mode):
    result = np.zeros((nclasses, *shape), np.float32)
    if mode == 'all_max':
        for j, f, q, u in entries:
            result[j] = np.maximum(result[j], q)
    elif mode == 'candidate_noisy_or':
        for j, f, q, u in entries:
            result[j] = 1. - (1. - result[j]) * (1. - q)
    else:
        cells = {}
        for j, f, q, u in entries:
            support = u if mode == 'family_max_uniform_r' else q
            cells[j, f] = np.maximum(cells[j, f], support) if (j, f) in cells else support.copy()
        for (j, f), support in cells.items():
            result[j] = 1. - (1. - result[j]) * (1. - support)
    return result

for cid in sorted(by_case):
    stored = by_case[cid]
    package = BENCH / cid
    casepath = Path(manifest[cid]['case_path'])
    assert sha(package / 'candidates.json') == stored['candidates_json_sha256']
    assert sha(casepath) == stored['case_json_sha256']
    candidates = load_candidates(package)
    assert len(candidates) == int(stored['candidate_count'])
    candidate_hashes = [mask_sha(c.mask) for c in candidates]
    assert hashlib.sha256(''.join(candidate_hashes).encode()).hexdigest() == stored['candidate_masks_sha256']
    case = read(casepath)
    truths = [binary(casepath.parent / part['mask_crop']) for part in case['parts']]
    assert hashlib.sha256(''.join(mask_sha(t) for t in truths).encode()).hexdigest() == stored['reference_masks_sha256']
    domain, category = details[cid]['expected_domain'], case['object_category']
    truth_names = [_normalize(p['part_name'], domain, object_category=category) for p in case['parts']]
    tax = fusion.taxonomy_from_candidates(candidates)
    normalized = {j: ('body' if name == domain else _normalize(name.removeprefix(domain + '_'), domain, object_category=category)) for j, name in enumerate(tax.fine_names) if j > 0}
    child = [j for j in normalized if tax.fine_names[j] != tax.parent_names[tax.fine_to_parent[j]]]
    selected = None
    for ri, (truth, name) in enumerate(zip(truths, truth_names)):
        true_ids = [j for j in child if normalized[j] == name]
        wrong_ids = [j for j in child if normalized[j] not in (name, 'body')]
        if not true_ids or not wrong_ids:
            continue
        roi = truth.copy()
        for other, other_name in zip(truths, truth_names):
            if other_name != name and other_name != 'body':
                roi &= ~other
        if roi.sum() < 20:
            continue
        wrong_id = min(wrong_ids, key=lambda j: tax.fine_names[j])
        selected = (ri, name, true_ids, wrong_id, roi)
        break
    assert (selected is not None) == (stored['eligible'] == 'True'), cid
    selection_verified += 1
    if selected is None:
        assert stored['exclusion']
        continue
    ri, name, true_ids, wrong_id, roi = selected
    assert ri == int(stored['reference_index']) and name == stored['reference_name']
    assert tax.fine_names[wrong_id] == stored['wrong_name']
    assert int(roi.sum()) == int(stored['target_pixels']) and mask_sha(roi) == stored['target_mask_sha256']
    for truth, gt_name in zip(truths, truth_names):
        if gt_name == normalized[wrong_id]:
            assert not np.any(truth & roi)
    family = sorted({fusion._source_family(c) for c in candidates if c.semantic_name == tax.fine_names[wrong_id]})[0]
    cross = [family] + [f for f in protocol['family_keys'] if f != family][:4]
    assert family == stored['base_family'] and cross == json.loads(stored['cross_families'])
    if cid not in fixed_samples:
        continue
    captured = {}
    class CaptureFinished(Exception):
        pass
    def intercept(evidence, taxonomy, accepted, name_to_id):
        captured.update(evidence=evidence.copy(), taxonomy=taxonomy, accepted=list(accepted), lookup=name_to_id.copy())
        raise CaptureFinished()
    original_parent = fusion._parent_support
    fusion._parent_support = intercept
    try:
        fusion.fuse_candidates(candidates, config=fusion.FusionConfig())
    except CaptureFinished:
        pass
    finally:
        fusion._parent_support = original_parent
    assert captured and captured['taxonomy'].fine_names == tax.fine_names
    entries = []
    for c in captured['accepted']:
        m = fusion._soft_membership(c.mask)
        entries.append((captured['lookup'][c.semantic_name], fusion._source_family(c), m * weight(c), m * weight(c, True)))
    replay = independent_aggregate(entries, len(tax.fine_names), roi.shape, 'family_max_noisy_or')
    assert np.array_equal(replay, captured['evidence']), cid
    roi_entries = [(j, f, q[roi], u[roi]) for j, f, q, u in entries]
    membership = fusion._soft_membership(roi)[roi]
    wrong_ids = [j for j in child if normalized[j] == normalized[wrong_id]]
    max_pixel_summary_error = 0.
    comparisons = 0
    for mode in MODES:
        for level, scenario in [('clean', 'clean')] + [(level, scenario) for level in LEVELS for scenario in protocol['conditions'][1:]]:
            new_entries = list(roi_entries)
            if scenario != 'clean':
                effective = float(np.clip(float(level), .01, .995))
                added_families = [family] if scenario == 'single' else [family] * 5 if scenario == 'same_family_5' else cross
                new_entries += [(wrong_id, f, membership * effective, membership * .995) for f in added_families]
            evidence = independent_aggregate(new_entries, len(tax.fine_names), (int(roi.sum()),), mode)
            wrong = evidence[wrong_ids].max(axis=0)
            correct = evidence[true_ids].max(axis=0)
            winner = np.asarray(child)[evidence[child].argmax(axis=0)]
            calculated = [np.mean(wrong > correct), np.mean(np.isin(winner, wrong_ids)), np.mean(np.isin(winner, true_ids)), wrong.mean(), correct.mean()]
            row = index[cid, mode, level, scenario]
            for metric, value in zip(METRICS, calculated):
                error = abs(float(row[metric]) - float(value))
                max_pixel_summary_error = max(max_pixel_summary_error, error)
                assert error == 0., (cid, mode, level, scenario, metric, error)
                comparisons += 1
    assert candidate_hashes == [mask_sha(c.mask) for c in candidates]
    pixel_audits.append({'case_id': cid, 'reference_index': ri, 'roi_pixels': int(roi.sum()), 'roi_sha256': mask_sha(roi),
                         'consensus_array_bit_exact': True, 'independent_stress_metric_comparisons': comparisons,
                         'maximum_stress_metric_error': max_pixel_summary_error})

# Independent reliability Brier/ECE recomputation; no missing annotation is a zero target.
rel_receipt = read(RELIABILITY / 'analysis_receipt.json')
rel_protocol = read(RELIABILITY / 'protocol_frozen.json')
assert sha(RELIABILITY / 'protocol_frozen.json') == rel_receipt['protocol_sha256']
for path, digest in rel_receipt['input_sha256'].items():
    assert sha(path) == digest, path
for name, digest in rel_receipt['artifacts_sha256'].items():
    assert sha(RELIABILITY / name) == digest, name
all_candidates = rows(RELIABILITY / 'candidates_all_1951.csv')
assert len(all_candidates) == 1951
missing = [r for r in all_candidates if int(r['same_semantic_gt_count']) == 0]
assert all(r['best_semantic_iou'] == '' and r['semantic_hit_025'] == '' and r['semantic_hit_050'] == '' and r['primary_semantic_analysis_included'] == '0' for r in missing)
primary = [r for r in all_candidates if r['evaluation_included'] == '1' and int(r['same_semantic_gt_count']) > 0]
assert len(primary) == 892
assert all((r in primary) == (r['primary_semantic_analysis_included'] == '1') for r in all_candidates)
rel_summary = read(RELIABILITY / 'analysis_summary.json')
pooled = {str(r['iou_threshold']): r for r in rel_summary['pooled_results'] if r['population'] == 'primary_conditional_semantic'}
rel_categories = sorted({r['object_category'] for r in all_candidates})
assert rel_categories == categories
assert np.array_equal(draws, np.load(RELIABILITY / 'bootstrap_category_draw_indices.npy', allow_pickle=False))
rel_groups = [np.asarray([i for i, r in enumerate(primary) if r['object_category'] == cat], dtype=int) for cat in categories]
rel_n = np.asarray([len(g) for g in rel_groups])
rel_denom = rel_n[draws].sum(axis=1)
weights = np.asarray([float(r['w']) for r in primary])
bins = np.minimum(9, np.floor(weights * 10).astype(int))
rel_out = []
for threshold, suffix in [(.25, '025'), (.5, '050')]:
    y = np.asarray([int(r['semantic_hit_' + suffix]) for r in primary])
    assert np.array_equal(y, np.asarray([float(r['best_semantic_iou']) >= threshold for r in primary], dtype=int))
    squared = (weights - y) ** 2
    brier = float(squared.mean())
    residual = weights - y
    ece = sum(abs(float(residual[bins == b].sum())) for b in range(10)) / len(primary)
    sq_by_cat = np.asarray([squared[g].sum() for g in rel_groups])
    boot_brier = sq_by_cat[draws].sum(axis=1) / rel_denom
    boot_ece_numerator = np.zeros(10000)
    for b in range(10):
        sums = np.asarray([residual[g][bins[g] == b].sum() for g in rel_groups])
        boot_ece_numerator += np.abs(sums[draws].sum(axis=1))
    boot_ece = boot_ece_numerator / rel_denom
    stored = pooled[str(threshold)]
    calculated = {'Brier_if_weight_interpreted_as_probability': brier,
                  'ECE10_if_weight_interpreted_as_probability': ece}
    for key, values in [('Brier_if_weight_interpreted_as_probability', boot_brier), ('ECE10_if_weight_interpreted_as_probability', boot_ece)]:
        calculated[key + '_ci95_low'] = float(np.quantile(values, .025))
        calculated[key + '_ci95_high'] = float(np.quantile(values, .975))
    errors = [abs(calculated[k] - float(stored[k])) for k in calculated]
    assert max(errors) < 1e-12
    rel_out.append({'iou_threshold': threshold, 'candidate_n': len(primary), 'matched_n': int(y.sum()),
                    'Brier': brier, 'ECE10': ece, 'maximum_point_and_interval_error': max(errors)})

report = {'status': 'PASS', 'created_utc': datetime.now(timezone.utc).isoformat(),
          'auditor_scope': 'Independent code/design review, source and output binding, raw-table arithmetic, all-case eligibility reconstruction, fixed-first-three raw-pixel Eq2 stress replay; not a new model experiment.',
          'audit_script_sha256': sha(__file__), 'protocol_sha256': sha(HERE / 'protocol.json'),
          'post_run_receipt_sha256': sha(HERE / 'post_run_receipt.json'),
          'bound_files': post['files'], 'n_cases': 226, 'n_eligible': 150, 'n_ineligible': 76, 'n_rows': 6000,
          'n_categories_original': 60, 'n_categories_eligible': len({by_case[cid]['category'] for cid in eligible}),
          'all_conditions_and_three_r_levels_complete': True, 'all_same_family_repetition_metrics_zero_difference': True,
          'candidate_noisy_or_same_cross_family_5_identical': True, 'uniform_r_levels_identical_as_defined': True,
          'all_226_original_consensus_reconstruction_audits_zero_error': True,
          'selection_reconstructed_from_inputs_all_cases': selection_verified, 'selection_uses_no_stress_outcomes': True,
          'raw_summary_and_pair_stats_checks': stat_checks, 'maximum_statistics_error': max_stat_error,
          'fixed_sample_rule': 'First three eligible case IDs in lexicographic order, fixed before raw replay; no result-based sample choice',
          'pixel_replay_samples': pixel_audits,
          'log_seal_history': {'original_receipt_only_mismatch': prior_mismatches, 'post_run_seal_verified': True, 'scientific_artifacts_unchanged': True},
          'protocol_clarification': {'post_execution_start': True, 'original_retained': True, 'no_analysis_or_selection_change': True,
                                     'correct_boundary': 'Selection is conditioned on archived candidate-derived taxonomy and family keys; no candidate score, predicted overlap, correctness or stress outcome is used.'},
          'reliability_audit': {'status': 'PASS', 'protocol_sha256': sha(RELIABILITY / 'protocol_frozen.json'),
                                 'all_candidates': 1951, 'primary_candidates': 892, 'same_semantic_gt_missing_all_candidates': len(missing),
                                 'same_semantic_gt_missing_evaluation_candidates': sum(r['evaluation_included'] == '1' for r in missing),
                                 'missing_annotations_preserved_null_not_negative': True, 'recomputed': rel_out},
          'publication_limits': ['Controlled conflict inside annotation-supported ROI; no claim of real FP prevalence or protection from high-confidence errors.',
                                 'Eq2 evidence readouts stop before hierarchical ownership and final segmentation; no final-mask robustness claim.',
                                 'Same-family exact-duplicate invariance is a mathematical property under correct family assignment, not proof that all correlated producer errors are suppressed.',
                                 'Family keys do not establish statistical independence; use every stress level and all eligibility exclusions.',
                                 'Brier/ECE are conditional diagnostics if heuristic w is interpreted as probability, not a fitted or validated calibration model.',
                                 'All intervals are pointwise exploratory category-cluster intervals on a reused benchmark.']}
(HERE / 'independent_audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({k: report[k] for k in ['status', 'n_cases', 'n_eligible', 'n_rows', 'raw_summary_and_pair_stats_checks', 'maximum_statistics_error', 'pixel_replay_samples', 'reliability_audit']}))
