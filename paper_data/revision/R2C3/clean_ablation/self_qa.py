"""Author implementation self-QA, not an independent audit or new experiment."""
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent

def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def csv_read(name):
    with (ROOT / name).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

protocol = read('protocol_frozen.json')
receipt = read('run_receipt.json')
files = read('file_manifest.json')
jobs = read('jobs_frozen.json')
rows = csv_read('cases.csv')
summary = csv_read('summary.csv')
deltas = csv_read('paired_deltas.csv')
assert sha(ROOT / 'protocol_frozen.json') == (ROOT / 'protocol_frozen.sha256').read_text().strip() == receipt['protocol_sha256']
assert sha(ROOT / 'run_clean_ablation.py') == protocol['script_sha256'] == receipt['script_sha256']
assert sha(ROOT / 'file_manifest.json') == protocol['file_manifest_sha256']
assert sha(ROOT / 'jobs_frozen.json') == protocol['jobs_sha256']
for item in files['source_files'] + files['input_files']:
    assert sha(item['path']) == item['sha256'], item['path']
fusion_files = [item for item in files['source_files'] if item['path'].endswith('src\\hpid_split\\fusion.py') or item['path'].endswith('src/hpid_split/fusion.py')]
assert len(fusion_files) == 1
assert fusion_files[0]['sha256'] == 'd9bb3f738b262109d88587dae5b71796d312a461790d66fe7dfb689f1c372bb5'
assert len(jobs) == len({j['case_id'] for j in jobs}) == len({j['image_id'] for j in jobs}) == 226
assert sum(j['candidate_count'] for j in jobs) == 1951
assert len(rows) == len({(r['case_id'], r['variant']) for r in rows}) == 678
assert len(list((ROOT / 'receipts').glob('*.json'))) == 226
for job in jobs:
    record = read('receipts/' + job['case_id'] + '.json')
    assert record['status'] == 'COMPLETE'
    assert datetime.fromisoformat(protocol['frozen_utc']) < datetime.fromisoformat(record['started_utc'])
    assert record['original_candidates_unchanged']
    assert record['released_endpoint_max_difference'] == 0
    assert record['variants']['released']['bitexact_archived_map']
    assert record['candidate_json_sha256'] == sha(Path(job['package']) / 'candidates.json')
    assert all(record['variants'][name]['candidate_input_unchanged'] for name in protocol['variant_order'])

# Recompute all bootstrap summaries with multiplicity weights per case,
# rather than the execution script's sums indexed by sampled categories.
categories = sorted({j['category'] for j in jobs})
cat_idx = np.asarray([categories.index(j['category']) for j in jobs])
draws = np.random.default_rng(protocol['statistics']['seed']).integers(len(categories), size=(10000, len(categories)))
multiplicity = np.zeros((10000, len(categories)), dtype=np.int64)
np.add.at(multiplicity, (np.arange(10000)[:, None], draws), 1)
weights = multiplicity[:, cat_idx]
denom = weights.sum(axis=1)
lookup = {(r['case_id'], r['variant']): r for r in rows}
stats = {(r['variant'], r['metric']): r for r in summary}
contrasts = {(r['contrast'], r['metric']): r for r in deltas}
max_error = 0.0
checks = 0
for metric in protocol['metrics']:
    original = np.asarray([float(lookup[j['case_id'], 'released'][metric]) for j in jobs])
    for variant in protocol['variant_order']:
        values = np.asarray([float(lookup[j['case_id'], variant][metric]) for j in jobs])
        boot = weights @ values / denom
        calculated = {'mean': values.mean(), 'ci95_low': np.quantile(boot, .025), 'ci95_high': np.quantile(boot, .975)}
        for key, value in calculated.items():
            error = abs(float(stats[variant, metric][key]) - value)
            max_error = max(max_error, error)
            assert error < 1e-12, (variant, metric, key, error)
            checks += 1
        if variant != 'released':
            diff = values - original
            boot = weights @ diff / denom
            calculated = {'mean_difference': diff.mean(), 'ci95_low': np.quantile(boot, .025), 'ci95_high': np.quantile(boot, .975),
                          'cases_increased': int((diff > 1e-12).sum()), 'cases_decreased': int((diff < -1e-12).sum()),
                          'cases_equal': int((abs(diff) <= 1e-12).sum())}
            for key, value in calculated.items():
                error = abs(float(contrasts[variant + '_minus_released', metric][key]) - value)
                max_error = max(max_error, error)
                assert error < 1e-12, (variant, metric, key, error)
                checks += 1

report = {'status': 'PASS', 'audit_type': 'implementation author self-QA; not an independent audit',
          'protocol_sha256': sha(ROOT / 'protocol_frozen.json'), 'run_receipt_sha256': sha(ROOT / 'run_receipt.json'),
          'self_qa_script_sha256': sha(__file__), 'n_cases': 226, 'n_rows': 678,
          'n_source_files_verified': len(files['source_files']), 'n_input_files_verified': len(files['input_files']),
          'all_case_receipts_complete': True, 'all_frozen_files_unchanged': True,
          'protocol_frozen_before_every_inference': True, 'original_candidates_unchanged': True,
          'released_bitexact_maps': 226, 'released_endpoint_comparisons': receipt['released_endpoint_comparisons'],
          'released_endpoint_maximum_error': 0.0, 'arithmetic_checks': checks, 'maximum_arithmetic_error': max_error,
          'scored_part_totals': {variant: sum(int(float(r['predicted_part_count'])) for r in rows if r['variant'] == variant) for variant in protocol['variant_order']}}
(ROOT / 'self_qa.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report))
