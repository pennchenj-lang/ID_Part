"""Independent arithmetic audit of existing R2C4 sensitivity data; no model run."""
import ast
import csv
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
BUNDLE = ROOT / 'hpid-publication-20261009/docs/evidence/revision_20261008'
EVIDENCE = BUNDLE / 'evidence'
FUSION = Path('__RUNTIME_ROOT__/code_snapshots/hpid_split_be54300_holdout/src/hpid_split/fusion.py')
PUBLISHED_FUSION = BUNDLE / 'sources/frozen_be54300/src/hpid_split/fusion.py'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

sources = [EVIDENCE / name for name in ['sensitivity_cases.csv', 'sensitivity_summary.csv', 'sensitivity_protocol.json']]
sources += [FUSION, PUBLISHED_FUSION, BUNDLE / 'code/analysis/analyze_revision.py']
before = {str(path): sha(path) for path in sources}
protocol = read(EVIDENCE / 'sensitivity_protocol.json')
rows = csv_rows(EVIDENCE / 'sensitivity_cases.csv')
previous = {r['variant']: r for r in csv_rows(EVIDENCE / 'sensitivity_summary.csv')}
variants = list(protocol['variants'])
assert len(variants) == len(previous) == 50
ids = sorted({r['case_id'] for r in rows})
assert len(ids) == protocol['cases'] == 226
assert len(rows) == 11300
lookup = {(r['case_id'], r['variant']): r for r in rows}
assert len(lookup) == len(rows)
assert set(lookup) == set(itertools.product(ids, variants))
assert all((r['exact_release_map'] == 'True') == (float(r['pixel_change_fraction']) == 0.0) for r in rows)
assert all(r['exact_release_map'] in ('True', 'False') for r in rows)
assert all(r['exact_release_map'] == 'True' for r in rows if r['variant'] == 'release')
metrics = ['part_f1_at_025', 'part_recall_at_025', 'part_f1_at_050', 'part_f1_at_075',
           'semantic_f1_at_025', 'root_foreground_iou', 'mean_matched_boundary_f1_at_025', 'predicted_part_count']
columns = [(variant, metric) for variant in variants for metric in metrics]
values = np.asarray([[float(lookup[cid, variant][metric]) for variant, metric in columns] for cid in ids])
base = {metric: np.asarray([float(lookup[cid, 'release'][metric]) for cid in ids]) for metric in metrics}
differences = values - np.stack([base[metric] for variant, metric in columns], axis=1)

# The original generator draws cases, not object-category clusters. Reconstruct
# exactly those draws, then use case multiplicities for an independent mean calculation.
assert protocol['seed'] == 20261008
draws = np.random.default_rng(20261008).integers(0, 226, (10000, 226))
counts = np.zeros((10000, 226), dtype=np.int64)
np.add.at(counts, (np.arange(10000)[:, None], draws), 1)
boot = counts @ values / 226
boot_delta = counts @ differences / 226
points = values.mean(axis=0)
delta_points = differences.mean(axis=0)
bounds = np.quantile(boot, [.025, .975], axis=0)
delta_bounds = np.quantile(boot_delta, [.025, .975], axis=0)
maximum_error = 0.
comparisons = 0
recomputed = {variant: {'n': 226, 'changed_maps': sum(lookup[cid, variant]['exact_release_map'] == 'False' for cid in ids),
                        'mean_pixel_change_fraction': float(np.mean([float(lookup[cid, variant]['pixel_change_fraction']) for cid in ids])),
                        'metrics': {}} for variant in variants}
for col, (variant, metric) in enumerate(columns):
    expected = previous[variant]
    assert recomputed[variant]['changed_maps'] == int(expected['changed_cases'])
    numbers = {'mean': points[col], 'low': bounds[0, col], 'high': bounds[1, col],
               'delta_mean': delta_points[col], 'delta_low': delta_bounds[0, col], 'delta_high': delta_bounds[1, col]}
    for key, number in numbers.items():
        original_key = ('delta_' + metric + '_' + key[6:]) if key.startswith('delta_') else metric + '_' + key
        error = abs(float(expected[original_key]) - number)
        maximum_error = max(maximum_error, error)
        assert error < 1e-12, (variant, original_key, error)
        comparisons += 1
    recomputed[variant]['metrics'][metric] = {key: float(value) for key, value in numbers.items()}

local = {'detail_bonus': ['detail_bonus=0.064', 'detail_bonus=0.096'],
         'orphan_support': ['orphan_support=0.096', 'orphan_support=0.144'],
         'hierarchy_strength': ['hierarchy_strength=0.52', 'hierarchy_strength=0.78'],
         'conflict_penalty': ['conflict_penalty=0.576', 'conflict_penalty=0.864']}
for parameter, names in local.items():
    original = {'detail_bonus': .08, 'orphan_support': .12, 'hierarchy_strength': .65, 'conflict_penalty': .72}[parameter]
    for name, multiplier in zip(names, [.8, 1.2]):
        spec = protocol['variants'][name]
        target = 'specificity_host_suppression' if parameter == 'conflict_penalty' else parameter
        expected = 1 - original * multiplier if parameter == 'conflict_penalty' else original * multiplier
        assert len(spec) == 1 and abs(spec[target] - expected) < 1e-14
joint = [name for name in variants if name.startswith('joint_')]
assert len(joint) == 16
for setting in itertools.product([.8, 1.2], repeat=4):
    name = 'joint_' + '_'.join(map(str, setting))
    assert name in joint
    for key, expected in zip(['detail_bonus', 'orphan_support', 'hierarchy_strength', 'specificity_host_suppression'],
                             [.08 * setting[0], .12 * setting[1], .65 * setting[2], 1 - .72 * setting[3]]):
        assert abs(protocol['variants'][name][key] - expected) < 1e-14

assert before[str(FUSION)] == before[str(PUBLISHED_FUSION)] == protocol['code_sha256'] == 'd9bb3f738b262109d88587dae5b71796d312a461790d66fe7dfb689f1c372bb5'
source = FUSION.read_text(encoding='utf-8')
config_class = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'FusionConfig')
defaults = {n.target.id: ast.literal_eval(n.value) for n in config_class.body if isinstance(n, ast.AnnAssign)}
assert defaults['detail_bonus'] == .08
assert defaults['specificity_host_suppression'] == .28
assert defaults['scene_layer_fallback_weight'] == .72
assert defaults['orphan_support'] == .12 and defaults['hierarchy_strength'] == .65
for expression in ['outside_penalty = config.orphan_support**config.hierarchy_strength',
                   'evidence[class_id] *= np.where(allowed, 1.0, outside_penalty)',
                   '& (support >= config.orphan_support)',
                   'config.specificity_child_evidence_floor_ratio',
                   'evidence[list(taxonomy.detail_ids)] +=']:
    assert expression in source, expression
assert protocol['variants']['conflict_penalty=0'] == {'specificity_host_suppression': 1}
assert protocol['variants']['disable_use_specificity_ownership'] == {'use_specificity_ownership': False}
assert all(sha(path) == digest for path, digest in before.items())

ablation_names = ['detail_bonus=0', 'conflict_penalty=0', 'orphan_support=0', 'hierarchy_strength=0',
                  'disable_use_parent_support', 'disable_use_specificity_ownership']
joint_f1 = [recomputed[name]['metrics']['part_f1_at_025']['mean'] for name in joint]
joint_semantic = [recomputed[name]['metrics']['semantic_f1_at_025']['mean'] for name in joint]
result = {'status': 'PASS', 'analysis_type': 'Independent arithmetic and static source audit of existing sensitivity outputs; no model inference or parameter changes',
          'created_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': before,
          'counts': {'variants_including_release': 50, 'cases_per_variant': 226, 'rows': 11300, 'joint_settings': 16, 'local_one_at_a_time_settings': 8},
          'statistics': {'unit': 'case/image, not object-category cluster', 'draws': 10000, 'seed': 20261008,
                         'estimand': 'case-macro mean and paired variant-minus-release difference',
                         'interval': 'pointwise descriptive percentile 95%; no multiplicity correction',
                         'case_order': 'lexicographic case_id, matching original analysis CSV sort',
                         'all_original_statistics_recomputed': comparisons, 'maximum_absolute_error': maximum_error},
          'release': recomputed['release'],
          'local_20percent': {parameter: {name: recomputed[name] for name in names} for parameter, names in local.items()},
          'joint_20percent': {'part_f1_at_025_min': min(joint_f1), 'part_f1_at_025_max': max(joint_f1),
                              'semantic_f1_at_025_min': min(joint_semantic), 'semantic_f1_at_025_max': max(joint_semantic),
                              'changed_maps_min': min(recomputed[name]['changed_maps'] for name in joint),
                              'changed_maps_max': max(recomputed[name]['changed_maps'] for name in joint),
                              'settings': {name: recomputed[name] for name in joint}},
          'ablations': {name: recomputed[name] for name in ablation_names},
          'nonoptimality_example': {'variant': 'detail_bonus=0.12', **recomputed['detail_bonus=0.12']},
          'all_50_settings_recomputed': recomputed,
          'parameter_semantics': {
              'detail_bonus': 'Add 0.08 only for detail classes where original direct spatial support >0.38, after parent/residual/child-floor and eligible-host suppression. This does not disable the direct-detail gate when set to zero.',
              'conflict_penalty': 'Penalty 0.72 means retained host-evidence multiplier 0.28 on eligible overlap union. It is distinct from scene_layer_fallback_weight=0.72 in candidate weighting.',
              'conflict_zero': 'conflict_penalty=0 only sets host suppression multiplier to1; child-floor ratio0.90 and its eligibility logic remain active. Disabling specificity ownership removes both child-floor update and host suppression.',
              'parent_factor': 'For nonroot classes with existing hard parent support, multiply evidence by1 inside dilated support and orphan_support**hierarchy_strength outside; default0.12**0.65=' + str(.12**.65) + '. Parent attenuation precedes residual and child-floor updates.',
              'orphan_dual_use': 'orphan_support is also the minimum dilated parent support in the direct-detail gate. Its intervention is not an isolated exponent-factor ablation.',
              'equation3': 'S_k(x)=[1-0.72 o_k(x)] E_k(x)+0.08 d_k(x), where E already includes parent attenuation, parent-residual and child-floor updates; bonus is not multiplied by those earlier factors.'},
          'source_anchors': {'fusion.py': {'defaults': [37,89], 'scene_weight_fallback': [1532,1539], 'parent_attenuation': [1585,1605],
                                          'parent_residual': [1610,1632], 'specificity_floor_and_host': [1635,1753], 'detail_bonus': [1755,1758], 'direct_gate_orphan': [1785,1788]},
                             'analyze_revision.py': 'variants(); case_worker(); run_sensitivity(); bootstrap(); published portable copy retained in source hashes'},
          'claim_limits': ['Post-review diagnostic reuse of226 cases and archived accepted candidates, not independent validation of development overfitting.',
                           'Local stable aggregate F1 does not prove optimal constants, statistical equivalence, or stable pixel/identity maps.',
                           'Accepted-pool replay does not rerun upstream proposal generation or the original serial Group gates and does not establish full-pipeline robustness.',
                           'detail_bonus=.12 gives higher observed geometric F1 than release; retain this evidence and do not call the release optimum.',
                           'Original intervals are case-bootstrap intervals, not the60-category bootstrap used by other diagnostics.']}
(OUT / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
receipt = {'status': 'PASS', 'created_utc': datetime.now(timezone.utc).isoformat(), 'audit_script_sha256': sha(__file__),
           'summary_sha256': sha(OUT / 'summary.json'), 'source_files_unchanged': True, 'source_sha256': before,
           'existing_output_rows': 11300, 'recomputed_statistics': comparisons, 'maximum_absolute_error': maximum_error,
           'new_model_experiments': 0, 'Word_files_modified': 0}
(OUT / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': 'PASS', 'rows': len(rows), 'variants': len(variants), 'recomputed_statistics': comparisons,
                  'maximum_error': maximum_error, 'joint_f1_range': [min(joint_f1), max(joint_f1)],
                  'joint_changed_maps_range': [result['joint_20percent']['changed_maps_min'], result['joint_20percent']['changed_maps_max']],
                  'summary_sha256': receipt['summary_sha256']}))
