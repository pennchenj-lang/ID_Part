"""Portable CPU recomputation from included overlap evidence (NumPy/SciPy)."""
from pathlib import Path
import argparse,csv,json
import numpy as np
from scipy.optimize import linear_sum_assignment

def read(p):return json.loads(p.read_text(encoding='utf8'))
def box_iou(a,b):
 if a is None or b is None:return 0.
 inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
 union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
 return inter/union if union else 0.
def recompute(row,condition):
 p=row[condition];out={}
 for layer in ['groups','parts']:
  x=p[layer];matrix=np.asarray(x['iou_float32'],dtype=np.float32)
  if matrix.size:
   a,b=linear_sum_assignment(1-matrix);v=matrix[a,b]
  else:v=np.array([])
  for suffix,t in [('025',.25),('050',.5),('075',.75)]:
   tp=int((v>=t).sum());den=x['truth_count']+x['prediction_count']
   out[f'{layer}_f1_{suffix}']=2*tp/den if den else 0.
 for src,dst in [('root_union','root_iou'),('union','foreground_iou')]:
  x=p[src];out[dst]=x['intersection_pixels']/x['union_pixels'] if x['union_pixels'] else 0.
 out['root_bbox_iou']=box_iou(p['root_union']['bbox'],row['truth_object_bbox'])
 out['domain_correct']=int(p['selected_domain']==row['expected_domain'])
 out['profile_correct']=int(row['expected_profile'] in p['selected_profiles'])
 out['valid_package']=p['valid_package']
 out['root_success_025']=int(out['root_iou']>=.25);out['root_success_050']=int(out['root_iou']>=.5)
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--evidence',type=Path,default=Path(__file__).resolve().parent);ap.add_argument('--output',type=Path,default=Path('overlap_recomputation.json'));args=ap.parse_args();r=args.evidence
 cases=read(r/'overlap_evidence.json')['cases'];assert len(cases)==42 and len({x['image_id'] for x in cases})==42
 cases=sorted(cases,key=lambda x:x['case_id']);saved={}
 for mode,file in [('crop','baseline_projected_cases.csv'),('full','full_image_cases.csv')]:
  with (r/file).open(encoding='utf-8-sig',newline='') as f:saved[mode]={x['case_id']:x for x in csv.DictReader(f)}
 values={};checks=0;maxerr=0.
 for mode in ['crop','full']:
  values[mode]=[recompute(row,mode) for row in cases]
  for row,out in zip(cases,values[mode]):
   for key,val in out.items():
    err=abs(val-float(saved[mode][row['case_id']][key]));assert err<=1e-12,(row['case_id'],mode,key,err);checks+=1;maxerr=max(maxerr,err)
 summary=read(r/'full_image_summary.json')['metrics'];ix=np.random.default_rng(20261009).integers(0,42,(10000,42));stats={}
 for key in values['crop'][0]:
  x=np.array([v[key] for v in values['crop']]);y=np.array([v[key] for v in values['full']]);d=y-x
  stats[key]={'crop_mean':float(x.mean()),'full_mean':float(y.mean()),'difference':float(d.mean()),'paired_ci95':np.quantile(d[ix].mean(axis=1),[.025,.975]).tolist()}
  for field,value in stats[key].items():
   assert np.allclose(value,summary[key][field],atol=1e-12,rtol=0),(key,field,value,summary[key][field]);checks+=np.size(value)
 result={'status':'PASS','cases':42,'per_case_and_statistic_checks':checks,'maximum_case_numeric_error':maxerr,'reproduced_metrics':stats,'scope':'Recomputes scored outcomes from saved pairwise overlaps and pixel counts; rerunning neural inference requires the externally obtained original images and frozen assets.'}
 args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='reproduced_metrics'}))
if __name__=='__main__':main()
