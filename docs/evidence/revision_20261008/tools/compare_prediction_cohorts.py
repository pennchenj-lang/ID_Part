from pathlib import Path
import json,hashlib
from PIL import Image
import numpy as np
import argparse
_parser=argparse.ArgumentParser();_parser.add_argument('--earlier',type=Path,required=True);_parser.add_argument('--later',type=Path,required=True);_parser.add_argument('--output',type=Path,required=True);_args=_parser.parse_args()
_args.output.parent.mkdir(parents=True,exist_ok=True)
A=_args.earlier
B=_args.later
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
_args.output.write_text(json.dumps(out,indent=2),encoding='utf-8')
print({k:v for k,v in out.items() if k!='cases'})

