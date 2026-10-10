"""Recount the immutable historical routing CSV; no new inference/scoring."""
import collections
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = Path('__RUNTIME_ROOT__/experiments/paper_v031_identity_frontend_20260828/03_completion_group_frontend/routing_correctness_cases.csv')
EVIDENCE = HERE.parent.parent / 'hpid-publication-20261009/docs/evidence/revision_20261008/evidence'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flag(row, field):
    return int(float(row[field]))


def state(row):
    if flag(row, 'query_ambiguous'):
        return 'ambiguous'
    if flag(row, 'query_unresolved'):
        return 'unresolved'
    return 'unique_correct' if flag(row, 'unique_and_correct_025') else 'unique_wrong'


def main():
    with SOURCE.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 185
    methods = sorted({r['method'] for r in rows})
    counts = {}
    for method in methods:
        subset = [r for r in rows if r['method'] == method]
        assert len(subset) == 37
        counts[method] = dict(n=37)
        for threshold in ['025', '050']:
            ambiguous_n = sum(flag(r, 'query_ambiguous') for r in subset)
            counts[method][threshold] = {
                'unique_correct': sum(flag(r, f'unique_and_correct_{threshold}') for r in subset),
                'wrong_unique': sum(flag(r, f'wrong_unique_{threshold}') for r in subset),
                'ambiguous': ambiguous_n,
                'unresolved': sum(flag(r, 'query_unresolved') for r in subset),
                'correct_in_review_numerator': sum(flag(r, f'correct_in_review_set_{threshold}') for r in subset),
                'correct_in_review_denominator': ambiguous_n,
            }
    raw = {r['case_id']: r for r in rows if r['method'] == 'raw_proposals'}
    hpid = {r['case_id']: r for r in rows if r['method'] == 'hpid_split_group_ids'}
    transitions = collections.Counter((state(raw[c]), state(hpid[c])) for c in raw)
    paired, nine = [], []
    for case_id in raw:
        first, second = raw[case_id], hpid[case_id]
        available_raw = bool(flag(first, 'unique_and_correct_025') or flag(first, 'correct_in_review_set_025'))
        available_hpid = bool(flag(second, 'unique_and_correct_025') or flag(second, 'correct_in_review_set_025'))
        row = dict(case_id=case_id, raw_state=state(first), hpid_state=state(second),
                   raw_semantic_correct_available=available_raw, hpid_semantic_correct_available=available_hpid,
                   raw_best_semantic_iou=float(first['best_review_match_iou']),
                   hpid_best_semantic_iou=float(second['best_review_match_iou']))
        paired.append(row)
        if flag(first, 'correct_in_review_set_025'):
            nine.append(row)
    availability = collections.Counter((r['raw_semantic_correct_available'], r['hpid_semantic_correct_available']) for r in paired)
    report_path = EVIDENCE / 'request_report.json'
    report = json.loads(report_path.read_text(encoding='utf-8-sig'))
    payload = dict(status='HISTORICAL_READ_ONLY_RECOUNT', source=str(SOURCE), source_sha256=sha(SOURCE),
                   counts=counts, raw_to_hpid_state_transitions=[dict(raw=a, hpid=b, n=n) for (a,b),n in transitions.items()],
                   paired_semantic_correct_availability={f'raw_{a}_hpid_{b}': n for (a,b),n in availability.items()},
                   original_nine_correct_ambiguous_requests=nine, all_paired_requests=paired,
                   interpretation='Raw9/13 and HPID1/6 are conditional on different ambiguous-query sets. Of the same original9 with a correct ambiguous option,5 become uniquely correct,1 remains ambiguous with correct support,2 become wrongly unique and1 stays ambiguous without correct support. Six of those original9 retain a correct semantic option.',
                   substantive_issue='37 limits precision, but observed wrong-unique outputs and lost correct support are substantive failures. More cases improve precision and test reproducibility; they do not by themselves repair routing.',
                   candidate_pruning_evidence=report['verification'],
                   mechanism_claim_boundaries=report['claim_boundaries'],
                   mechanism_diagnostics=report['precise_failures'],
                   evidence_files={str(p): sha(p) for p in sorted(EVIDENCE.glob('request_*')) if p.suffix in {'.csv','.json'}},
                   script_sha256=sha(__file__))
    out = HERE / 'original37_audit.json'
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(path=str(out), sha256=sha(out), counts=counts, paired_availability=payload['paired_semantic_correct_availability'])))


if __name__ == '__main__':
    main()
