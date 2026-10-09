from pathlib import Path
import csv,json
import independent_statistics as audit

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'audit/statistics'
rows=audit.read_csv_rows(ROOT/'target_retention.csv')
with (ROOT/'target_retention.csv').open(encoding='utf-8-sig',newline='') as f:raw=list(csv.DictReader(f))
assert len(raw)==len(rows)
for r,s in zip(rows,raw):
    r['identity']=s['current_native_id']
    r['initial_identity']=s['initial_native_id']
    r['producer_retained']=audit.truth(s['native_joint_correct_retained'])
case,cohorts,cats,domains,families,conditions,familymap=audit.derive(rows)
means,contrasts,strata=audit.aggregate(case,cohorts,cats,domains,families,conditions,familymap)
audit.write_csv(OUT/'independent_native_estimates.csv',means)
audit.write_csv(OUT/'independent_native_contrasts.csv',contrasts)

checks=[]
for source,independent in [('native_summary.json',means),('summary_crosscheck.json',list(csv.DictReader((OUT/'independent_estimates.csv').open(encoding='utf-8-sig',newline=''))))]:
    produced=json.loads((ROOT/source).read_text(encoding='utf-8'))
    for threshold,section in produced['thresholds'].items():
        th=float(threshold)
        cohortrows=[r for r in cohorts if r['threshold']==th]
        assert section['eligible_cases']==sum(r['common_three_count']>0 for r in cohortrows)
        assert section['common_gt_targets']==sum(r['common_three_count'] for r in cohortrows)
        for result in section['scope_results']:
            for method,rate in result['method_case_macro_rates'].items():
                own=next(r for r in independent if float(r['threshold'])==th and r['cohort']=='common_three' and r['scope']==result['scope'] and r['method']==method)
                difference=abs(float(own['macro'])-rate)
                assert difference<1e-12
                checks.append(dict(source=source,threshold=th,scope=result['scope'],method=method,difference=difference))

report=dict(status='pass',point_estimates_checked=len(checks),max_difference=max(r['difference'] for r in checks),checks=checks,
            native_primary=[r for r in contrasts if r['primary']],
            interpretation='Native-ID sensitivity compares complete method+ID-rule packages. It cannot isolate the fusion contribution. Neither native nor common-wrapper primary estimates support HPID superiority.')
(OUT/'native_and_producer_crosscheck.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='checks'},indent=2))
