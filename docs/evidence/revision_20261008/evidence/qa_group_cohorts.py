from pathlib import Path
import json,hashlib
from PIL import Image
import numpy as np
R=Path(r"${REVISION_ROOT}")
A=Path(r"${RUNTIME_ROOT}/experiments/paper_v031/test226_hpid")
B=Path(r"${RUNTIME_ROOT}/paper_backups/20260824_before_serial_fig1_release/desktop_package_before_serial_fig1/03_实验数据与统计/HPID-Split_对象条件评估_v0.3.1/HPID-Split_v0.3.1_226例完整物理组回归_D盘链接")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
summary=json.loads((B/'benchmark_summary.json').read_text(encoding='utf-8-sig'))
rows=[]
for c in summary['cases']:
 case=c['case_id'];a=A/case;b=B/case
 row={'case_id':case,'b_exists':b.exists()}
 for n in ['source.png','part_id_map.tiff','group_id_map.tiff']:
  p,q=a/n,b/n
  row[n+'_exist']=p.exists() and q.exists()
  if p.exists() and q.exists():
   x,y=np.array(Image.open(p)),np.array(Image.open(q))
   row[n+'_array_equal']=x.shape==y.shape and np.array_equal(x,y)
 for root,label in [(a,'a'),(b,'b')]:
  p=root/'candidates.json'
  row[label+'_candidate_file_exists']=p.exists()
  if p.exists():
   d=json.loads(p.read_text(encoding='utf-8-sig'));row[label+'_candidate_n']=len(d if isinstance(d,list) else d['candidates'])
 rows.append(row)
out={'earlier_pool':str(A),'later_release_regression':str(B),'summary_sha256':sha(B/'benchmark_summary.json'),'source_benchmark':summary['source_benchmark'],'regroup_flags':{k:summary[k] for k in ['candidate_generation_rerun','fine_part_map_changed']},'note':'Flags compare later regroup to its own source benchmark, not to earlier fine-ID pool.','release_group_f1':summary['aggregate']['editable_group_macro_metrics']['part_discovery_f1_at_025'],'cases':rows}
out['aggregate']={k:sum(r.get(k,0) for r in rows) for k in ['source.png_array_equal','part_id_map.tiff_array_equal','group_id_map.tiff_array_equal','a_candidate_n','b_candidate_n']}
(R/'evidence/qa_group_cohorts.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print({k:v for k,v in out.items() if k!='cases'})

