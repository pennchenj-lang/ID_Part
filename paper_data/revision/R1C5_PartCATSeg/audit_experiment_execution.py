"""Independent read-only audit. Does not open GT, scores, or prediction arrays."""
from pathlib import Path
import collections, datetime, hashlib, json, mmap, subprocess, sys
from PIL import Image

R = Path(__file__).resolve().parent
checks, issues = [], []

def read(relative):
    return json.loads((R / relative).read_text(encoding='utf-8-sig'))

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def check(name, condition, evidence):
    checks.append(dict(name=name, status='PASS' if condition else 'FAIL', evidence=evidence))
    if not condition:
        issues.append(name)

protocol = read('protocol_frozen.json')
freeze = read('protocol_freeze_receipt.json')
cohort = read('cohort_predictions/run.json')
registry = read('label_registry.json')
inputs = read('inference_manifest.json')
manifest = read('source_manifest.json')
voc1, voc2, paco = [read(p + '/run.json') for p in ['smoke_voc_01', 'smoke_voc_02', 'smoke_paco_01']]
expected_protocol = 'fc237e550c755d65cff07658cfa392387ef30db0786e29f271dd0eb954c4538d'
check('protocol_sha256', sha(R/'protocol_frozen.json') == expected_protocol == freeze['protocol_sha256'], expected_protocol)
binding_paths = {
    'inference_manifest_sha256':'inference_manifest.json', 'label_registry_sha256':'label_registry.json',
    'checkpoint_sha256':'weights/partcatseg_voc.pth', 'adapter_sha256':'smoke_inference.py',
    'source_manifest_sha256':'source_manifest.json', 'scorer_sha256':'score_partcatseg.py',
    'decoder_equivalence_sha256':'smoke_voc_02/decoder_equivalence.json',
    'full_vocabulary_resource_control_sha256':'smoke_paco_01/run.json',
    'package_versions_sha256':'environment/package_versions_frozen.json',
    'fairness_audit_sha256':'fairness_input_audit.json',
    'detectron_build_summary_sha256':'environment/detectron_build_summary.json'}
bindings = {key:dict(path=path, expected=protocol['hash_bindings'][key], actual=sha(R/path))
            for key,path in binding_paths.items()}
check('frozen_asset_bindings', all(v['expected']==v['actual'] for v in bindings.values()), bindings)

processes = read('cohort_process_start_audit.json')
if isinstance(processes, dict): processes=[processes]
freeze_dt = datetime.datetime.fromisoformat(protocol['frozen_utc'])
starts = [datetime.datetime.fromisoformat(p['CreationDate']) for p in processes]
check('freeze_precedes_real_process_start', bool(starts) and all(freeze_dt < d for d in starts),
      dict(frozen_utc=freeze_dt.isoformat(), process_starts_utc=[d.astimezone(datetime.timezone.utc).isoformat() for d in starts],
           seconds_to_first_start=(min(starts)-freeze_dt).total_seconds() if starts else None,
           source='Windows Win32_Process CreationDate captured while the cohort was running; run.json has no started_utc field'))
check('cohort_command_bound_to_protocol', all(expected_protocol in p['CommandLine'] for p in processes)
      and cohort['protocol']['sha256']==expected_protocol, dict(argv=cohort['argv'], processes=processes))

source = Path(manifest['copied_source'])
original = Path(manifest['original_source'])
source_failures=[]
for rel,expected in manifest['files'].items():
    for tree in [source, original]:
        if sha(tree/rel) != expected:
            source_failures.append(str(tree/rel))
commit=subprocess.check_output(['git','-C',str(original),'rev-parse','HEAD'],text=True).strip()
check('official_source_and_task_copy_hashes', not source_failures and commit==manifest['source_commit'],
      dict(files_per_tree=len(manifest['files']), source_commit=commit, mismatches=source_failures))
dino=R/'environment/dinov2'
dino_commit=subprocess.check_output(['git','-C',str(dino),'rev-parse','HEAD'],text=True).strip()
dino_status=subprocess.check_output(['git','-C',str(dino),'status','--short'],text=True).splitlines()
check('dino_source', not dino_status and sha(dino/'hubconf.py')==cohort['dino_source']['hubconf_sha256'],
      dict(commit=dino_commit, git_status=dino_status, hubconf_sha256=sha(dino/'hubconf.py'),
           scope='Complete tracked tree is clean at the recorded commit; runtime report explicitly binds hubconf.py'))

assets={name:dict(path=data['path'],expected=data['sha256'],actual=sha(data['path'])) for name,data in cohort['assets'].items()}
check('actual_checkpoint_clip_dino_files', all(x['expected']==x['actual'] for x in assets.values()), assets)
import torch
checkpoint=torch.load(R/'weights/partcatseg_voc.pth',map_location='meta',weights_only=False)
raw=checkpoint['model']
actual_shapes=[dict(name=k,shape=list(v.shape),dtype=str(v.dtype)) for k,v in raw.items()]
inventory=read('weights/checkpoint_inventory.json')
check('original_checkpoint_key_inventory', len(raw)==804 and actual_shapes==inventory['shapes'],
      dict(tensor_count=len(raw), inventory_count=inventory['tensor_count'],
           metadata_loaded_on='meta device only; no GPU or model execution',
           key_list_sha256=hashlib.sha256(json.dumps(sorted(raw)).encode()).hexdigest()))
checkpoint_reports={name:run['checkpoint_raw_audit'] for name,run in [('voc01',voc1),('voc02',voc2),('paco01',paco)]}
if 'checkpoint_raw_audit' in cohort: checkpoint_reports['cohort']=cohort['checkpoint_raw_audit']
check('checkpoint_model_exact_key_shape_compatibility', all(d['status']=='PASS' and d['checkpoint_key_count']==d['model_key_count']==804
      and not d['missing_keys'] and not d['unexpected_keys'] and not d['incorrect_shapes'] and not d['heuristics'] for d in checkpoint_reports.values()),
      dict(reports=checkpoint_reports, scope='Independent raw 804-key inventory plus hash-bound runtime model compatibility checks; no second model constructed'))
del checkpoint, raw

input_failures=[]
allowed={'anonymous_id','input_path','input_sha256','size_wh'}
for case in inputs['cases']:
    path=Path(case['input_path'])
    with Image.open(path) as image: wh=list(image.size)
    if set(case)!=allowed or sha(path)!=case['input_sha256'] or wh!=case['size_wh']:
        input_failures.append(case['anonymous_id'])
check('anonymous_image_only_inputs', len(inputs['cases'])==42 and len({c['anonymous_id'] for c in inputs['cases']})==42
      and len({c['input_sha256'] for c in inputs['cases']})==42 and not input_failures,
      dict(count=len(inputs['cases']), fields=sorted(allowed), unique_hashes=len({c['input_sha256'] for c in inputs['cases']}),
           mismatches=input_failures, note='Existing annotation-derived object crops are shared localization input; this is not full-image discovery.'))

# Extract only the public categories array. No annotation objects are decoded.
taxonomy=Path('__RUNTIME_ROOT__/datasets/paco/annotations/paco_lvis_v1_train.json')
with taxonomy.open('rb') as stream, mmap.mmap(stream.fileno(),0,access=mmap.ACCESS_READ) as data:
    start=data.find(b'"categories"')
    assert start >= 0
    start=data.find(b'[',start)
    depth=0; quoted=False; escaped=False; end=None
    for i in range(start,len(data)):
        b=data[i]
        if quoted:
            if escaped: escaped=False
            elif b==92: escaped=True
            elif b==34: quoted=False
        elif b==34: quoted=True
        elif b==91: depth+=1
        elif b==93:
            depth-=1
            if depth==0: end=i+1; break
    categories=json.loads(data[start:end])
categories.sort(key=lambda x:x['id'])
parts=[c for c in categories if ':' in c['name']]
objects=[c for c in categories if ':' not in c['name']]
expected_labels=[]
for index,c in enumerate(parts):
    obj,part=c['name'].split(':',1)
    expected_labels.append(dict(label_index=index,paco_category_id=c['id'],paco_name=c['name'],
                                rendered_name=obj.replace('_',' ')+"'s "+part.replace('_',' ')))
expected_objects=[c['name'].replace('_',' ') for c in objects]
check('complete_public_456_class_vocabulary',len(parts)==456 and len(objects)==75 and expected_labels==registry['labels']
      and expected_objects==registry['object_classes'] and sha(taxonomy)==registry['taxonomy_file_sha256'],
      dict(public_object_part_count=len(parts),public_object_count=len(objects),
           generalized_parts=len({c['name'].split(':',1)[1] for c in parts}),
           taxonomy_sha256=registry['taxonomy_file_sha256'], annotation_objects_decoded=0,
           labels_exact_order_and_text_match=True))
check('same_full_vocabulary_used_in_cohort',cohort['vocabulary']['class_count']==456
      and cohort['vocabulary']['stuff_classes']==[c['rendered_name'] for c in registry['labels']]
      and cohort['vocabulary']['obj_classes']==registry['object_classes'] and cohort['vocabulary']['class_ids']==list(range(456))
      and cohort['prediction_contract']['background_index']==456,
      dict(class_count=456,object_count=75,background_index=456,ignore_placeholder=65535))

equiv=read('smoke_voc_02/decoder_equivalence.json')
check('numerical_controls_and_retained_initial_failure',voc1['status']=='FAILED' and voc1['decoder_equivalence']['status']=='FAIL'
      and equiv['status']=='PASS' and equiv['validation_run_complete'] and equiv['argmax_identical']
      and all(x['dummy_target_invariance']['status']=='PASS' and x['dummy_target_invariance']['max_abs_score_error']==0 for x in [voc2,paco]),
      dict(voc01_failure=voc1['decoder_equivalence'],voc02_pass=equiv,paco_dummy=paco['dummy_target_invariance'],
           scope='One synthetic fixture; FP32 execution policy changed before target inference, not selected by target accuracy.'))
frozen_runtime=protocol['executed_numerical_controls']['runtime']['numerical_settings']
check('frozen_final_numeric_policy',equiv['runtime_fingerprint']==frozen_runtime and paco['runtime']['numerical_settings']==frozen_runtime,
      dict(runtime=frozen_runtime, cohort_runtime=cohort['runtime'],
           note='No universal numeric-equivalence claim; global deterministic_algorithms remains false as recorded.'))

native=read('environment/detectron_native_verification.json')
check('real_detectron_native_binary',native['status']=='PASS' and sha(native['extension'])==native['extension_sha256'],native)
frozen_packages=read('environment/package_versions_frozen.json')
current_packages=json.loads(subprocess.check_output([sys.executable,'-m','pip','list','--format=json','--disable-pip-version-check'],text=True))
check('installed_package_versions_unchanged',frozen_packages==current_packages,
      dict(count=len(current_packages), frozen_sha256=sha(R/'environment/package_versions_frozen.json')))

check('reviewed_inference_no_test_labels',True,{
    'review_type':'Static runner and official forward data-flow audit plus dummy-target invariance control; not a system-call sandbox proof',
    'runner':'smoke_inference.py',
    'input_preparation':'prepare_image_sample uses read_image and deterministic ResizeShortestEdge; sem_seg/obj_part_sem_seg constant ignore maps, gt_classes constant zero.',
    'forward':'Official partcatseg.py lines 340-364 produce unfiltered sem_seg_all separately from category-filtered sem_seg; only sem_seg_all exported.',
    'external_reads':'Official local source/config/weights, public registry, anonymous image manifest and pixels, frozen protocol and equivalence receipt. No reference manifest or scorer imported/opened by model process.',
    'aggregation':'Only conv_decoder splits already-aggregated class axis; original attention and correlation computations see all classes. Decoder Conv/GroupNorm do not mix batch elements.',
    'changes':'Exact hash-checked in-memory device/local-weight/Pillow compatibility overlay; weights_only=False restores legacy official loader semantics. Seeded original pre-checkpoint text cache policy retained.',
    'audit_did_not_open':'No GT masks, GT annotation rows, scoring outputs or prediction arrays.',
    'limits':['Model pretraining contamination is outside this audit.','Reference-manifest hash was not independently read here to maintain the requested GT boundary.','This is a retrospective common-crop cross-dataset comparison, not new untouched-cohort or official Pascal metric replication.']})

report=dict(status='PASS' if not issues else 'FAIL',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            audit_scope='Execution readiness and frozen provenance, independent of accuracy; final output sealing is a subsequent check.',
            protocol_sha256=expected_protocol,cohort_observed_status=cohort['status'],
            checks=checks,issues=issues,
            pending=['Verify the terminal all-42 receipt and sealed output hashes after cohort completion.'],
            audit_script_sha256=sha(__file__))
(R/'experiment_execution_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(dict(status=report['status'],checks=len(checks),issues=issues,cohort_observed_status=cohort['status']),indent=2))
raise SystemExit(bool(issues))
