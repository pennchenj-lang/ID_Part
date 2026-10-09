from pathlib import Path
import numpy as np,json
p=Path(__file__).parent
report=[]
for cid in ['calculator__01','scarf__03']:
    a=np.load(p/f'gate_hashseed_fixed_a_{cid}.npz');b=np.load(p/f'gate_hashseed_fixed_b_{cid}.npz')
    for variant in a.files:
        report.append({'case_id':cid,'variant':variant,'pixels_differ_across_fixed_seed_runs':int(np.count_nonzero(a[variant]!=b[variant])),
                       'pixels_differ_from_baseline_a':int(np.count_nonzero(a[variant]!=a['STA'])),
                       'pixels_differ_from_baseline_b':int(np.count_nonzero(b[variant]!=b['STA']))})
(p/'gate_repeat_pixel_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
