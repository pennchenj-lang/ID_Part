"""Recompute all fixed review-option statistics using Python + NumPy only.

Usage: python recompute.py --output-dir /your/local/output
The data and expected tables are read relative to this file. No image data,
network, upstream model, or computer-specific path is needed.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
METHODS = ['raw_proposals', 'greedy_nms', 'dbscan_fusion',
           'local_pairwise_crf', 'hpid_split_group_ids']
HPID = METHODS[-1]
METRICS = ['review_option_count', 'query_unique', 'query_ambiguous',
           'query_unresolved'] + [
    f'{metric}_{threshold}' for threshold in ('025', '050') for metric in
    ('unique_and_correct', 'wrong_unique', 'semantic_correct_any_match',
     'geometry_correct_any_exposed', 'geometry_correct_any_exported')]


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def interval(values, draws):
    bs = values[draws].mean(axis=1)
    return {'n': len(values), 'mean': float(values.mean()),
            'ci95_low': float(np.quantile(bs, .025)),
            'ci95_high': float(np.quantile(bs, .975))}, bs


def exact_mcnemar(gains, losses):
    n = gains + losses
    return min(1., 2 * sum(math.comb(n, k)
                          for k in range(min(gains, losses) + 1)) / 2 ** n)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'recomputed')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    records = read_csv(ROOT / 'data/review_cases.csv')
    cases = sorted({r['case_id'] for r in records})
    indexed = {(r['case_id'], r['method']): r for r in records}
    assert len(cases) == 37 and len(records) == len(indexed) == 185
    assert set(indexed) == {(c, m) for c in cases for m in METHODS}
    units = read_csv(ROOT / 'data/source_units.csv')
    assert len(units) == 37
    assert len({r['image_id'] for r in units}) == 37
    assert len({r['source_image_sha256'] for r in units}) == 37
    assert all(int(r['review_option_count']) >= 1 for r in records)
    draws = np.random.default_rng(20261009).integers(0, 37, size=(10000, 37))
    summaries, contrasts, subsets, exact = [], [], [], []
    for method in METHODS:
        for metric in METRICS:
            values = np.array([float(indexed[c, method][metric]) for c in cases])
            ci, _ = interval(values, draws)
            summaries.append({'method': method, 'metric': metric,
                              'sum': float(values.sum()), **ci})
    for method in METHODS[:-1]:
        for metric in METRICS:
            h = np.array([float(indexed[c, HPID][metric]) for c in cases])
            b = np.array([float(indexed[c, method][metric]) for c in cases])
            difference = h - b
            ci, bs = interval(difference, draws)
            gains = int((difference > 0).sum())
            losses = int((difference < 0).sum())
            row = {'comparator': method, 'metric': metric,
                   'hpid_sum': float(h.sum()), 'comparator_sum': float(b.sum()),
                   'hpid_mean': float(h.mean()), 'comparator_mean': float(b.mean()),
                   'hpid_only_or_higher': gains, 'comparator_only_or_higher': losses,
                   'equal': int((difference == 0).sum()), **ci}
            if metric == 'review_option_count':
                row.update(relative_reduction=float(1 - h.mean()/b.mean()),
                           ci98_75_low=float(np.quantile(bs, .00625)),
                           ci98_75_high=float(np.quantile(bs, .99375)))
            else:
                row.update(mcnemar_discordant_n=gains+losses,
                           mcnemar_exact_two_sided_p=exact_mcnemar(gains, losses))
                exact.append({'comparator': method, 'metric': metric,
                              'hpid_only_or_higher': gains,
                              'comparator_only_or_higher': losses,
                              'discordant_n': gains+losses,
                              'unadjusted_exact_two_sided_p': exact_mcnemar(gains, losses)})
            contrasts.append(row)
        definitions = [('same_query_state', None)] + [
            (f'both_{metric}_{threshold}', f'{metric}_{threshold}')
            for threshold in ('025', '050') for metric in
            ('semantic_correct_any_match', 'geometry_correct_any_exposed')]
        for subset, metric in definitions:
            if metric is None:
                keep = [c for c in cases if indexed[c, HPID]['query_state'] == indexed[c, method]['query_state']]
            else:
                keep = [c for c in cases if int(indexed[c, HPID][metric]) and int(indexed[c, method][metric])]
            if not keep:
                raise AssertionError('Unexpected empty fixed subset')
            h = np.array([float(indexed[c, HPID]['review_option_count']) for c in keep])
            b = np.array([float(indexed[c, method]['review_option_count']) for c in keep])
            subdraws = np.random.default_rng(20261009).integers(0, len(keep), size=(10000, len(keep)))
            ci, _ = interval(h-b, subdraws)
            subsets.append({'comparator': method, 'subset': subset,
                            'case_ids': ';'.join(keep), 'hpid_mean': float(h.mean()),
                            'comparator_mean': float(b.mean()), **ci})
    products = [('method_summary.csv', summaries, ('method', 'metric')),
                ('paired_contrasts.csv', contrasts, ('comparator', 'metric')),
                ('auxiliary_subsets.csv', subsets, ('comparator', 'subset'))]
    numeric_checks = 0
    for filename, rows, keys in products:
        expected = {tuple(r[k] for k in keys): r for r in read_csv(ROOT / 'results' / filename)}
        actual = {tuple(r[k] for k in keys): r for r in rows}
        assert actual.keys() == expected.keys(), filename
        for key, row in actual.items():
            for field, value in expected[key].items():
                if field in keys or value == '':
                    continue
                if field == 'case_ids':
                    assert row[field] == value
                    continue
                assert abs(float(row[field]) - float(value)) <= 1e-12, (filename, key, field)
                numeric_checks += 1
        write_csv(out / filename, rows)
    write_csv(out / 'exact_mcnemar_guardrails.csv', exact)
    verification = {'status': 'PASS', 'requests': 37, 'method_request_records': 185,
                    'distinct_source_images': 37, 'numeric_reference_checks': numeric_checks,
                    'tolerance': 1e-12, 'iterations': 10000, 'seed': 20261009,
                    'numpy': np.__version__,
                    'exact_test_timing': 'Sparse binary guardrail check added after the fixed audit; unadjusted and descriptive.'}
    (out / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
