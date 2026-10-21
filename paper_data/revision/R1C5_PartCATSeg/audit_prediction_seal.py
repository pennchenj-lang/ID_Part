"""Audit terminal prediction files only; never open references or scoring outputs."""
from pathlib import Path
import datetime, hashlib, json
import numpy as np
from PIL import Image

R=Path(__file__).resolve().parent
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

audit=read(R/'experiment_execution_audit.json')
checks=[]
def check(name, passed, evidence): checks.append(dict(name=name,status='PASS' if passed else 'FAIL',evidence=evidence))
P=R/'cohort_predictions'
manifest, receipt, progress, run=[read(P/p) for p in ['predictions_manifest.json','receipt.json','progress.json','run.json']]
protocol=read(R/'protocol_frozen.json')
inputs=read(R/'inference_manifest.json')
expected_ids=[x['anonymous_id'] for x in inputs['cases']]
check('all_42_terminal_complete',manifest['status']=='SEALED' and receipt['status']=='completed'
      and run['status']=='COHORT_INFERENCE_COMPLETE' and progress['status']=='COMPLETED'
      and manifest['complete_cases']==receipt['complete_cases']==42 and manifest['failed_cases']==receipt['failed_cases']==0
      and all([x['anonymous_id'] for x in doc['cases']]==expected_ids for doc in [manifest,receipt,progress])
      and all(x['status']=='COMPLETE' for doc in [manifest,receipt,progress] for x in doc['cases']),
      dict(count=42,complete_cases=manifest['complete_cases'],failed_cases=manifest['failed_cases'],exact_input_order=True))
check('manifest_receipt_and_run_agree',receipt['predictions_manifest_sha256']==sha(P/'predictions_manifest.json')
      and run['cohort_result']==receipt and progress['cases']==manifest['cases']
      and all(r=={k:v for k,v in m.items() if k in r} for r,m in zip(receipt['cases'],manifest['cases'])),
      dict(manifest_sha256=sha(P/'predictions_manifest.json'),receipt_sha256=sha(P/'receipt.json'),run_sha256=sha(P/'run.json')))
final_hashes={}
for oldcheck in audit['checks']:
    if oldcheck['name']=='frozen_asset_bindings':
        for key,old in oldcheck['evidence'].items():
            current=sha(R/old['path'])
            final_hashes[key]=dict(path=old['path'],expected=old['expected'],actual=current)
check('frozen_inputs_runner_assets_unchanged',sha(R/'protocol_frozen.json')==audit['protocol_sha256']
      and all(x['expected']==x['actual'] for x in final_hashes.values()),final_hashes)
binding_map={'protocol_sha256':audit['protocol_sha256'],
             'input_manifest_sha256':sha(R/'inference_manifest.json'),
             'label_registry_sha256':sha(R/'label_registry.json'),
             'adapter_sha256':sha(R/'smoke_inference.py')}
check('terminal_manifests_use_frozen_bindings',all(manifest[k]==v for k,v in binding_map.items())
      and receipt['inference_manifest_sha256']==binding_map['input_manifest_sha256']
      and all(receipt[k]==binding_map[k] for k in ['protocol_sha256','label_registry_sha256','adapter_sha256']),binding_map)
case_audits=[]
for original, record in zip(inputs['cases'],manifest['cases']):
    path,png=Path(record['label_path']),Path(record['png_path'])
    a=np.load(path,allow_pickle=False)
    with Image.open(png) as im: b=np.asarray(im)
    expected_shape=original['size_wh'][::-1]
    good=(record['input_sha256']==original['input_sha256']==sha(original['input_path'])
          and sha(path)==record['label_sha256'] and sha(png)==record['png_sha256']
          and a.dtype==np.dtype('uint16') and a.ndim==2 and list(a.shape)==expected_shape==record['shape']
          and np.array_equal(a,b) and a.size>0 and int(a.min())>=0 and int(a.max())<=456
          and path.parent==P/'labels' and png.parent==P/'labels')
    case_audits.append(dict(anonymous_id=record['anonymous_id'],status='PASS' if good else 'FAIL',
                           dtype=str(a.dtype),shape=list(a.shape),minimum_label=int(a.min()),maximum_label=int(a.max()),
                           label_sha256=sha(path),png_sha256=sha(png),npy_png_pixel_identical=bool(np.array_equal(a,b))))
check('all_label_files_hash_dtype_size_range',len(case_audits)==42 and all(x['status']=='PASS' for x in case_audits),case_audits)
actual_files={p.name for p in (P/'labels').iterdir() if p.is_file()}
expected_files={c['anonymous_id']+suffix for c in inputs['cases'] for suffix in ['.npy','.png']}
check('exact_42_npy_42_png_outputs',actual_files==expected_files,
      dict(file_count=len(actual_files),unexpected=sorted(actual_files-expected_files),missing=sorted(expected_files-actual_files)))
source_manifest=read(R/'source_manifest.json')
source=Path(source_manifest['copied_source'])
bad=[rel for rel,h in source_manifest['files'].items() if sha(source/rel)!=h]
check('source_unchanged_after_inference',not bad,dict(file_count=len(source_manifest['files']),mismatches=bad))
raw=run['checkpoint_raw_audit']
check('final_runtime_and_checkpoint_match_controls',raw['status']=='PASS' and raw['checkpoint_key_count']==raw['model_key_count']==804
      and not raw['missing_keys'] and not raw['unexpected_keys'] and not raw['incorrect_shapes']
      and run['runtime']['numerical_settings']==protocol['executed_numerical_controls']['runtime']['numerical_settings']
      and run['input_provenance']['annotations_read'] is False and manifest['annotations_read'] is False,
      dict(checkpoint_raw_audit=raw,numerical_settings=run['runtime']['numerical_settings'],input_provenance=run['input_provenance']))
seal=dict(status='PASS' if all(x['status']=='PASS' for x in checks) else 'FAIL',
          audited_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,
          audit_script_sha256=sha(__file__),scope='No references, GT masks, accuracy or scoring outputs opened. No prediction or frozen file modified.')
audit['final_seal']=seal
audit['pending']=[] if seal['status']=='PASS' else ['Resolve final seal audit failures']
audit['status']='PASS' if audit['status']=='PASS' and seal['status']=='PASS' else 'FAIL'
audit['cohort_observed_status']=run['status']
(R/'experiment_execution_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
(R/'prediction_seal_audit.json').write_text(json.dumps(seal,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(status=seal['status'],checks=len(checks),case_count=len(case_audits),failed_checks=[x['name'] for x in checks if x['status']!='PASS']),indent=2))
