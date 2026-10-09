"""Audit the two recorded bootstrap intervals without changing either analysis."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

_parser=argparse.ArgumentParser();_parser.add_argument("--data-root",type=Path,required=True);_parser.add_argument("--output",type=Path,required=True);_args=_parser.parse_args()
_args.output.parent.mkdir(parents=True,exist_ok=True)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = _args.data_root/'experiments/paper_v031_identity_frontend_20260828/03_completion_group_frontend'

def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def interval(values, seed):
    values = np.asarray(values, dtype=np.float64)
    indexes = np.random.default_rng(seed).integers(0, len(values), size=(10000, len(values)))
    means = values[indexes].mean(axis=1)
    return {'mean': float(values.mean()), 'low': float(np.quantile(means, .025)),
            'high': float(np.quantile(means, .975))}

source = rows(SOURCE/'completion_frontend_cases.csv')
original = rows(SOURCE/'completion_frontend_paired.csv')
current = rows(ROOT/'evidence/synthesis_diagnostics.csv')
audit = json.loads((ROOT/'evidence/routing_audit.json').read_text())
indexed = {(r['case_id'], r['method']): r for r in source}
case_ids = sorted({r['case_id'] for r in source})
checks = []
for metric, column, key, metric_index in [
    ('hidden_region_psnr', 'delta_psnr', 'mean_psnr_delta', 11),
    ('completion_request_recall', 'delta_request_recall', 'mean_request_recall_delta', 6),
]:
    differences = {cid: float(indexed[cid, 'hpid_split_group_ids'][metric]) - float(indexed[cid, 'raw_proposals'][metric]) for cid in case_ids}
    assert len(differences) == len(current) == 37
    assert all(differences[r['case_id']] == float(r[column]) for r in current)
    orig_row = next(r for r in original if r['comparison'] == 'hpid_split_group_ids_minus_raw_proposals' and r['metric'] == metric)
    orig_seed = 20260828 + 1000 + metric_index
    orig_replay = interval([differences[cid] for cid in case_ids], orig_seed)
    post_replay = interval([float(r[column]) for r in current], 20261008)
    orig_expected = {'mean': float(orig_row['mean_paired_difference']), 'low': float(orig_row['ci95_low']), 'high': float(orig_row['ci95_high'])}
    assert orig_replay == orig_expected
    assert post_replay == audit[key]
    checks.append({'metric': metric, 'case_values_equal': True, 'n': 37,
        'original': {'seed': orig_seed, 'draws': 10000, 'case_order': 'sorted case identifiers', 'recorded_equals_recomputed': True, **orig_replay},
        'post_review': {'seed': 20261008, 'draws': 10000, 'case_order': 'stored completion export order', 'recorded_equals_recomputed': True, **post_replay}})

report = {'status': 'PASS', 'checks': checks,
    'interpretation': 'Both intervals use precisely the same37 paired values and the same case-paired percentile estimator. Different seeds and case order generate different finite Monte Carlo draws; this is not a data or estimand change. Keep the original frozen interval in the manuscript and label the later interval as a post-review recomputation.',
    'recommended_supplement_note': 'Table S27 retains the original 10,000-draw case-paired percentile intervals. Its HPID-minus-raw PSNR contrast used sorted case IDs and seed 20261839 (base20260828 plus1000 plus metric index11), giving -0.7431dB [-1.6322,0.0153]. The post-review source-data audit uses the same37 paired values in export order with seed20261008, giving [-1.6787,0.0206]. The difference reflects finite Monte Carlo resampling, not changed outcomes; both intervals include zero.',
    'source_locations': ['hpid_split/scripts/run_semantic_completion_frontend_benchmark.py:32,41-58,192-202,223-240', 'revision_20261008/analyze_revision.py:24,66-71,214-226'],
    'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE/'completion_frontend_cases.csv', SOURCE/'completion_frontend_paired.csv', ROOT/'evidence/synthesis_diagnostics.csv', ROOT/'evidence/routing_audit.json']}}
_args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({'status': report['status'], 'checks': checks}, indent=2))
