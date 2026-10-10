"""Frozen-input, CPU-only full-fusion ablation; GT is used only by evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent
RUNTIME = Path('__RUNTIME_ROOT__')
SNAPSHOT = RUNTIME / 'code_snapshots/hpid_split_be54300_holdout'
PROJECT = Path('__USER_HOME__/Documents/Codex/2026-06-28/new-chat/hpid_split')
BENCH = RUNTIME / 'experiments/paper_v031/test226_hpid'
ARCHIVED = RUNTIME / 'experiments/paper_v031_identity_frontend_20260828/01_same_candidate/same_candidate_postprocessing_cases.csv'
BASELINE = PROJECT / 'src/hpid_split/postprocess_baselines.py'
EVALUATOR = PROJECT / 'scripts/evaluate_same_candidate_postprocessing.py'
sys.dont_write_bytecode = True
sys.path.insert(0, str(SNAPSHOT / 'src'))
sys.path.insert(0, str(PROJECT / 'scripts'))

import cv2
import numpy as np
import scipy
from PIL import Image
from hpid_split import fusion

spec = importlib.util.spec_from_file_location('hpid_split.postprocess_baselines', BASELINE)
baseline = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = baseline
spec.loader.exec_module(baseline)
from evaluate_same_candidate_postprocessing import _evaluate, _load_candidates

cv2.setNumThreads(1)
VARIANTS = ('released', 'uniform_r', 'no_consensus')
METRICS = ('part_f1_at_025', 'part_f1_at_050', 'semantic_f1_at_025', 'semantic_f1_at_050', 'predicted_part_count')
SEED = 20261010
B = 10000


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def csv_write(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def array_sha(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def candidate_signature(candidates):
    rows = []
    for c in candidates:
        rows.append({'semantic_name': c.semantic_name, 'semantic_parent': c.semantic_parent,
                     'score': c.score, 'source_reliability': c.source_reliability,
                     'source': c.source, 'prompt': c.prompt, 'metadata': c.metadata,
                     'mask_sha256': array_sha(c.mask)})
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def freeze():
    if (OUT / 'protocol_frozen.json').exists():
        raise RuntimeError('Refusing to overwrite an existing frozen protocol')
    bench = read(BENCH / 'benchmark_summary.json')
    manifest_path = Path(bench['source_manifest'])
    manifest = read(manifest_path)
    lookup = {r['case_id']: r for r in manifest['cases']}
    rows = sorted([r for r in bench['cases'] if r['return_code'] == 0], key=lambda r: r['case_id'])
    assert len(rows) == 226
    assert len({lookup[r['case_id']]['image_id'] for r in rows}) == 226
    with ARCHIVED.open(encoding='utf-8-sig', newline='') as stream:
        previous = {r['case_id']: r for r in csv.DictReader(stream) if r['method'] == 'hpid_split_a3'}
    inputs = {BENCH / 'benchmark_summary.json', manifest_path, ARCHIVED}
    jobs = []
    n_candidates = 0
    for row in rows:
        cid = row['case_id']
        package = BENCH / cid
        case_path = Path(lookup[cid]['case_path'])
        case = read(case_path)
        candidates = read(package / 'candidates.json')
        n_candidates += len(candidates)
        inputs.update([package / 'candidates.json', package / 'part_id_map.tiff', package / 'parts.json', case_path, case_path.parent / 'object_mask_crop.png'])
        inputs.update(package / c['mask_path'] for c in candidates)
        inputs.update(case_path.parent / p['mask_crop'] for p in case['parts'])
        jobs.append({'case_id': cid, 'domain': row['expected_domain'], 'category': case['object_category'],
                     'case_path': str(case_path), 'package': str(package), 'archived': previous[cid],
                     'image_id': lookup[cid]['image_id'], 'candidate_count': len(candidates)})
    assert n_candidates == 1951
    sources = sorted((SNAPSHOT / 'src/hpid_split').rglob('*.py')) + [BASELINE, EVALUATOR, Path(__file__).resolve()]
    file_manifest = {'input_files': [{'path': str(p), 'sha256': sha(p)} for p in sorted(inputs)],
                     'source_files': [{'path': str(p), 'sha256': sha(p)} for p in sources]}
    dump(OUT / 'file_manifest.json', file_manifest)
    dump(OUT / 'jobs_frozen.json', jobs)
    protocol = {
        'frozen_utc': utc(), 'study': 'R2.3 clean full-fusion fixed-candidate ablation',
        'scope': '226 archived test images; 1951 accepted candidates; no upstream model rerun or parameter tuning',
        'variants': {'released': 'FusionConfig() and original candidates',
                     'uniform_r': 'dataclasses.replace(candidate, source_reliability=1.0); all other values and FusionConfig() unchanged',
                     'no_consensus': 'original candidates; dataclasses.replace(FusionConfig(), use_consensus=False); all other settings unchanged'},
        'variant_order': list(VARIANTS), 'n_cases': 226, 'n_candidates': 1951,
        'n_source_images': 226, 'n_categories': len({r['category'] for r in jobs}),
        'metrics': list(METRICS),
        'evaluation': 'Original evaluate_same_candidate_postprocessing._evaluate, including original root/body handling, semantic normalization and Hungarian matching. GT read only after all three fusion calls in each case.',
        'input_mutation_policy': 'dataclasses.replace changes only source_reliability; mask and metadata are shared unchanged; hash original and each variant before and after inference',
        'reproduction': 'released instance_map bit-exact against archived TIFF; all available numeric evaluator endpoints compared at absolute tolerance 1e-10',
        'statistics': {'estimand': 'case-macro mean and paired variant-minus-released difference',
                       'bootstrap': 'sample sorted object-category clusters with replacement, keeping all cases in each selected cluster; preserve pairing and reweight by sampled case count',
                       'iterations': B, 'seed': SEED, 'interval': '2.5 and 97.5 percentiles; descriptive pointwise 95%, no superiority or multiplicity-adjusted inference',
                       'all_contrasts': 'uniform_r minus released; no_consensus minus released, for all five metrics; retain all results'},
        'failure_policy': 'Any case failure or reproduction mismatch makes study status FAIL; do not substitute zeros or silently drop failures',
        'interpretation': 'uniform_r is a full-pipeline reliability substitution, affecting every downstream use of r and potentially candidate filtering/ownership; not an Eq.2-only causal isolation. no_consensus replaces cross-family noisy-OR by maximum while preserving downstream rules. Existing benchmark reuse and fixed accepted-pool scope are explicit.',
        'base_config': asdict(fusion.FusionConfig()),
        'script_sha256': sha(__file__), 'file_manifest_sha256': sha(OUT / 'file_manifest.json'),
        'jobs_sha256': sha(OUT / 'jobs_frozen.json'),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'opencv': cv2.__version__},
    }
    dump(OUT / 'protocol_frozen.json', protocol)
    (OUT / 'protocol_frozen.sha256').write_text(sha(OUT / 'protocol_frozen.json') + '\n', encoding='ascii')
    print(json.dumps({'status': 'FROZEN', 'sha256': sha(OUT / 'protocol_frozen.json'), 'cases': len(jobs), 'categories': protocol['n_categories'], 'candidates': n_candidates}), flush=True)


def verify_files():
    protocol = read(OUT / 'protocol_frozen.json')
    assert sha(OUT / 'protocol_frozen.json') == (OUT / 'protocol_frozen.sha256').read_text().strip()
    assert sha(__file__) == protocol['script_sha256']
    assert sha(OUT / 'file_manifest.json') == protocol['file_manifest_sha256']
    assert sha(OUT / 'jobs_frozen.json') == protocol['jobs_sha256']
    files = read(OUT / 'file_manifest.json')
    for row in files['input_files'] + files['source_files']:
        if sha(row['path']) != row['sha256']:
            raise RuntimeError('Frozen file changed: ' + row['path'])
    return protocol


def worker(job):
    started = utc()
    package = Path(job['package'])
    candidates = _load_candidates(package)
    original_signature = candidate_signature(candidates)
    root_mask = np.asarray(baseline._root_candidate(candidates).mask, dtype=bool)
    results = {}
    checks = {}
    for variant in VARIANTS:
        selected = [replace(c, source_reliability=1.0) for c in candidates] if variant == 'uniform_r' else candidates
        config = replace(fusion.FusionConfig(), use_consensus=False) if variant == 'no_consensus' else fusion.FusionConfig()
        if variant == 'uniform_r':
            assert all(a.mask is b.mask and a.metadata is b.metadata for a, b in zip(selected, candidates))
        signature_before = candidate_signature(selected)
        t0 = time.perf_counter()
        result = fusion.fuse_candidates(selected, config=config)
        elapsed = time.perf_counter() - t0
        assert signature_before == candidate_signature(selected), (job['case_id'], variant, 'variant input mutated')
        assert original_signature == candidate_signature(candidates), (job['case_id'], variant, 'archived input mutated')
        instances = tuple(baseline.BaselineInstance(identity=r.part_id, semantic_name=r.semantic_name,
                          semantic_parent=r.semantic_parent, mask=result.instance_map == r.instance_index,
                          confidence=1.0) for r in result.instances)
        results[variant] = (baseline.BaselinePrediction(method=variant, instances=instances, root_mask=root_mask, diagnostics={}), elapsed)
        checks[variant] = {'fusion_seconds': elapsed, 'instance_map_sha256': array_sha(result.instance_map),
                           'candidate_input_unchanged': True, 'input_signature': signature_before,
                           'accepted_candidate_count': result.diagnostics['accepted_candidate_count'],
                           'output_part_count': len(result.instances)}
        if variant == 'released':
            archived_map = np.asarray(Image.open(package / 'part_id_map.tiff'))
            assert np.array_equal(result.instance_map, archived_map), (job['case_id'], 'released map mismatch')
            checks[variant]['bitexact_archived_map'] = True
    # No reference masks or labels enter any fusion call above.
    case_path = Path(job['case_path'])
    case = read(case_path)
    rows = []
    max_diff = 0.0
    matched_endpoints = 0
    for variant in VARIANTS:
        prediction, elapsed = results[variant]
        metrics = _evaluate(prediction, case=case, case_dir=case_path.parent, expected_domain=job['domain'])
        if variant == 'released':
            for key, value in metrics.items():
                if key in job['archived']:
                    diff = abs(value - float(job['archived'][key]))
                    max_diff = max(max_diff, diff)
                    matched_endpoints += 1
                    assert diff <= 1e-10, (job['case_id'], key, value, job['archived'][key])
        rows.append({'case_id': job['case_id'], 'domain': job['domain'], 'category': job['category'], 'image_id': job['image_id'],
                     'variant': variant, 'candidate_count': len(candidates), 'fusion_seconds': elapsed,
                     **{key: metrics[key] for key in METRICS}})
    receipt = {'case_id': job['case_id'], 'status': 'COMPLETE', 'started_utc': started, 'completed_utc': utc(),
               'candidate_json_sha256': sha(package / 'candidates.json'), 'original_candidates_signature': original_signature,
               'original_candidates_unchanged': original_signature == candidate_signature(candidates),
               'released_endpoint_max_difference': max_diff, 'released_endpoint_count': matched_endpoints, 'variants': checks}
    dump(OUT / 'receipts' / (job['case_id'] + '.json'), receipt)
    return rows, receipt


def summarize(rows, jobs):
    lookup = {(r['case_id'], r['variant']): r for r in rows}
    ids = [j['case_id'] for j in jobs]
    categories = sorted({j['category'] for j in jobs})
    groups = [np.asarray([i for i, j in enumerate(jobs) if j['category'] == category]) for category in categories]
    rng = np.random.default_rng(SEED)
    draws = rng.integers(len(groups), size=(B, len(groups)))
    sizes = np.asarray([len(g) for g in groups], dtype=float)
    denoms = sizes[draws].sum(axis=1)
    summary, deltas = [], []
    for metric in METRICS:
        original = np.asarray([lookup[cid, 'released'][metric] for cid in ids])
        for variant in VARIANTS:
            values = np.asarray([lookup[cid, variant][metric] for cid in ids])
            sums = np.asarray([values[g].sum() for g in groups])
            boot = sums[draws].sum(axis=1) / denoms
            summary.append({'variant': variant, 'metric': metric, 'n_cases': len(ids), 'n_categories': len(groups),
                            'mean': float(values.mean()), 'ci95_low': float(np.quantile(boot, .025)), 'ci95_high': float(np.quantile(boot, .975))})
            if variant != 'released':
                differences = values - original
                sums = np.asarray([differences[g].sum() for g in groups])
                boot = sums[draws].sum(axis=1) / denoms
                deltas.append({'contrast': variant + '_minus_released', 'metric': metric, 'n_cases': len(ids), 'n_categories': len(groups),
                               'mean_difference': float(differences.mean()), 'ci95_low': float(np.quantile(boot, .025)), 'ci95_high': float(np.quantile(boot, .975)),
                               'cases_increased': int((differences > 1e-12).sum()), 'cases_decreased': int((differences < -1e-12).sum()),
                               'cases_equal': int((np.abs(differences) <= 1e-12).sum())})
    csv_write(OUT / 'summary.csv', summary)
    csv_write(OUT / 'paired_deltas.csv', deltas)
    return summary, deltas


def run(workers):
    protocol = verify_files()
    if (OUT / 'run_receipt.json').exists():
        raise RuntimeError('Refusing to overwrite a prior run receipt')
    jobs = read(OUT / 'jobs_frozen.json')
    (OUT / 'receipts').mkdir(exist_ok=True)
    started = utc()
    t0 = time.perf_counter()
    rows, receipts, failures = [], [], []
    print(json.dumps({'status': 'STARTED', 'started_utc': started, 'protocol_sha256': sha(OUT / 'protocol_frozen.json'), 'workers': workers}), flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(worker, job): job['case_id'] for job in jobs}
        for index, future in enumerate(as_completed(pending), 1):
            cid = pending[future]
            try:
                case_rows, receipt = future.result()
                rows.extend(case_rows)
                receipts.append(receipt)
            except Exception as exc:
                failures.append({'case_id': cid, 'error': repr(exc)})
                dump(OUT / 'receipts' / (cid + '.json'), {'case_id': cid, 'status': 'FAIL', 'error': repr(exc), 'completed_utc': utc()})
            if index % 20 == 0 or index == len(jobs):
                print(json.dumps({'finished': index, 'total': len(jobs), 'failures': len(failures), 'elapsed_seconds': time.perf_counter() - t0}), flush=True)
    verify_files()
    rows.sort(key=lambda row: (row['case_id'], VARIANTS.index(row['variant'])))
    if rows:
        csv_write(OUT / 'cases.csv', rows)
    report = {'status': 'FAIL' if failures else 'COMPLETE', 'started_utc': started, 'completed_utc': utc(),
              'elapsed_seconds': time.perf_counter() - t0, 'protocol_sha256': sha(OUT / 'protocol_frozen.json'),
              'script_sha256': sha(__file__), 'expected_cases': len(jobs), 'completed_cases': len(receipts), 'result_rows': len(rows),
              'failures': failures, 'frozen_files_unchanged_before_after': True,
              'released_bitexact_maps': sum(r['variants']['released']['bitexact_archived_map'] for r in receipts),
              'released_endpoint_comparisons': sum(r['released_endpoint_count'] for r in receipts),
              'released_maximum_endpoint_difference': max((r['released_endpoint_max_difference'] for r in receipts), default=None),
              'candidate_inputs_unchanged_all_cases': all(r['original_candidates_unchanged'] for r in receipts),
              'frozen_utc_precedes_first_case': all(protocol['frozen_utc'] < r['started_utc'] for r in receipts),
              'interpretation': protocol['interpretation']}
    if not failures:
        assert len(receipts) == 226 and len(rows) == 678
        summary, deltas = summarize(rows, jobs)
        report['summary'] = summary
        report['paired_deltas'] = deltas
    dump(OUT / 'run_receipt.json', report)
    print(json.dumps({key: report[key] for key in ['status', 'completed_cases', 'result_rows', 'elapsed_seconds', 'released_bitexact_maps', 'released_maximum_endpoint_difference']}), flush=True)
    if failures:
        raise RuntimeError('Failed cases retained; see run_receipt.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    if args.freeze == args.run:
        raise SystemExit('Specify exactly one of --freeze or --run')
    if args.freeze:
        freeze()
    else:
        run(args.workers)
