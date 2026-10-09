"""Stage a bounded public evidence bundle; never uploads or edits source experiments."""
from __future__ import annotations
import argparse,csv,hashlib,io,json,re,shutil,zipfile
from pathlib import Path

ANALYSIS=['analyze_revision.py','gate_ablation.py','stage_diagnosis.py','request_provenance.py','score_provenance.py','taxonomy_behavior.py','diagnose_revision.py']
HELPERS=['analyze_cross_domain_fusion_ablation.py','analyze_paper_results.py','evaluate_same_candidate_postprocessing.py','audit_completion_routing.py']
PROMPT_CONFIGS={'general_asset_prompts.json','paco_modern_extensions.json','game_asset_extensions.json'}

def digest(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def portability_validation(rev,out):
    replay=rev/'public_replay_check_LOCAL_ONLY';checks=[]
    def rows(p):
        with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    for file,folder in [('holdout_root_cases.csv','results_holdout'),('routing_intervals.csv','results'),('routing_transitions.csv','results'),('score_candidate_provenance.csv','results')]:
        p=replay/folder/file
        if p.exists():
            same=p.read_bytes()==(rev/'evidence'/file).read_bytes();assert same,file
            checks.append({'analysis_file':file,'replay_rows':len(rows(p)),'comparison':'byte-exact CSV','passed':same})
    selected=[('sensitivity_cases.csv','results_sensitivity/sensitivity_cases.csv',['case_id','variant'],set()),('stage_case_metrics.csv','results_stage/stage_trial_case_metrics.csv',['case_id','stage'],set()),('request_endpoints.csv','results_request/request_endpoints.csv',['case_id','method'],set())]
    gp=list((replay/'results_gate').rglob('group_gate_cases.csv')) if (replay/'results_gate').exists() else []
    if gp:selected.append(('gate_ablation/group_gate_cases.csv',str(gp[0].relative_to(replay)),['split','case_id','variant'],{'fixed_input_sha256','group_map_sha256','group_map_changed','semantic_partition_changed'}))
    for original,relative,key,ignored in selected:
        p=replay/relative
        if not p.exists():continue
        old={tuple(r[k] for k in key):r for r in rows(rev/'evidence'/original)};new=rows(p)
        for r in new:
            base=old[tuple(r[k] for k in key)]
            for k,v in r.items():
                if k not in ignored:assert v==base[k],(original,key,k,v,base[k])
        checks.append({'analysis_file':original,'replay_rows':len(new),'comparison':'all fields exact except documented input-path hashes and one-pixel map-change fields' if ignored else 'all subset fields exact','passed':True})
    taxonomy=replay/'results/taxonomy_behavior.json'
    if taxonomy.exists():
        result=read(taxonomy);original=read(rev/'evidence/taxonomy_behavior.json')
        assert result['all_assertions_passed'] and result['fixture_count']==43
        assert result['fixtures']==original['fixtures'] and result['frozen_sources']==original['frozen_sources']
        checks.append({'analysis_file':'taxonomy_behavior.json','replay_rows':43,'comparison':'all fixtures and frozen source hashes exact','passed':True})
    data={'status':'PASS' if len(checks)==9 else 'INCOMPLETE','checks':checks,'execution_scope':{'taxonomy':'43 fixtures passed','score':'all226 cases/1951 candidates','routing':'all37 requests','holdout':'all42 cases/84 paired rows','sensitivity':'3 cases x50 settings','gate':'3 cases per split x8 subsets','stage':'3 cases x17 stage endpoints','request':'3 requests x3 stored routing methods'},'input_preparation':'4404 generated files extracted and hash-verified;1770 separately supplied local files hash-verified. Missing-input gate check failed clearly before supply. Source experiments were read-only.','limitations':'Subset smoke checks establish path portability, not new full independent experiments. Existing full evidence remains the scientific result. Gate one-pixel counterfactual repeat limitation remains.','private_test_directory_included':False}
    save(out/'PORTABILITY_VALIDATION.json',data)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runtime',type=Path,required=True);ap.add_argument('--project',type=Path,required=True);ap.add_argument('--output',type=Path)
    args=ap.parse_args();rev=Path(__file__).resolve().parent;runtime=args.runtime.resolve();project=args.project.resolve();out=(args.output or rev/'public_evidence').resolve()
    out.mkdir(parents=True,exist_ok=True)
    audit=[];members=[];omitted=[];case_manifest=[]
    roots=[(str(rev),'${REVISION_ROOT}'),(str(project),'${PROJECT_ROOT}'),(str(runtime),'${RUNTIME_ROOT}'),(str(project.parent),'${WORKSPACE_ROOT}'),('${USER_HOME}','${USER_HOME}')]
    def clean_text(s):
        for root,token in roots:
            for version in [root.replace('\\','/'),root.replace('/','\\').replace('\\','\\\\'),root.replace('/','\\')]:s=s.replace(version,token)
        return s
    def clean_obj(o):
        if isinstance(o,str):return clean_text(o)
        if isinstance(o,list):return [clean_obj(x) for x in o]
        if isinstance(o,dict):return {clean_text(str(k)):clean_obj(v) for k,v in o.items() if not str(k).startswith('suggested_')}
        return o
    def cleaned(p):
        b=p.read_bytes()
        if p.suffix.lower()=='.json':return (json.dumps(clean_obj(json.loads(b.decode('utf-8-sig'))),ensure_ascii=False,indent=2)+'\n').encode()
        if p.suffix.lower() in {'.py','.csv','.md','.txt','.log','.toml'}:return clean_text(b.decode('utf-8-sig')).encode('utf-8')
        return b
    def put(src,rel,sanitize=True):
        data=cleaned(src) if sanitize else src.read_bytes();dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        audit.append({'source':clean_text(str(src)),'public_path':str(Path(rel)).replace('\\','/'),'original_sha256':digest(src.read_bytes()),'public_sha256':digest(data),'bytes':len(data),'path_redacted':data!=src.read_bytes()})
    # Only experimental outputs. Manuscript extracts and local-only visual audits are excluded.
    for p in sorted((rev/'evidence').iterdir()):
        if p.is_file() and p.suffix in {'.csv','.json','.log','.py'} and not any(t in p.name for t in ['trial','LOCAL_ONLY','qa_content','qa_equation','qa_scientific','gate_probe']):put(p,Path('evidence')/p.name)
    # Drop this stale manuscript-layout QA copy from earlier staging, preserving its local original.
    stale=out/'evidence/qa_equation_native_check.json'
    if stale.exists():
        assert stale.resolve().is_relative_to(out)
        stale.unlink()
    gate_allow=['group_gate_protocol.json','group_gate_cases.csv','group_gate_summary.csv','group_gate_complete_exported_pool.csv','group_gate_pool.csv','group_gate_actual_calls.csv','group_gate_input_provenance.json','group_gate_full_run.log','gate_validation_checks.json','gate_repeat_pixel_audit.json','gate_raw_predicate_patterns.csv','gate_baseline_audit.json','gate_baseline_audit.csv','gate_baseline_audit_historical.json','gate_baseline_audit_historical.csv','gate_report.txt','gate_runner_provenance_note.txt','gate_verify_results.py']
    for name in gate_allow:
        p=rev/'evidence/gate_ablation'/name
        if p.exists():put(p,Path('evidence/gate_ablation')/name)
    for pattern in ['gate_baseline_audit.py','gate_baseline_audit*.log','gate_verify_results.log','gate_compare_repeats.py','gate_hashseed*']:
        for p in sorted((rev/'evidence/gate_ablation').glob(pattern)):
            if p.is_file():put(p,Path('evidence/gate_ablation')/p.name)
    for folder in ['full_first_pass','full_integrity_pass','final_executed']:
        for p in sorted((rev/'evidence/gate_ablation/backups'/folder).iterdir()):
            if p.is_file() and p.suffix in {'.csv','.json','.log','.py'}:put(p,Path('evidence/gate_ablation/backups')/folder/p.name)
    # Algorithm source and configuration snapshots; no model checkpoints or data photographs.
    snap=runtime/'code_snapshots/hpid_split_be54300_holdout'
    for dirname,pattern in [('src','*.py'),('configs','*.json')]:
        for p in sorted((snap/dirname).rglob(pattern)):put(p,Path('sources/frozen_be54300')/p.relative_to(snap),False)
    for name in ['LICENSE','THIRD_PARTY.md','pyproject.toml','FROZEN_RELEASE.md']:
        if (snap/name).exists():put(snap/name,Path('sources/frozen_be54300')/name)
    put(rev/'evidence/gate_ablation/physical_groups_da7d236.py','sources/historical_da7d236/physical_groups.py',False)
    for name in HELPERS:put(project/'scripts'/name,Path('sources/project/hpid_split/scripts')/name)
    put(project/'src/hpid_split/postprocess_baselines.py','sources/project/hpid_split/src/hpid_split/postprocess_baselines.py',False)
    for p in sorted((project/'configs').rglob('*.json')):
        rel=p.relative_to(project/'configs')
        put(p,Path('sources/project/hpid_split/configs')/rel,p.name not in PROMPT_CONFIGS)
        # Earlier staging retained this source alias; sanitize it consistently if present.
        alias=Path('sources/project/configs')/rel
        if (out/alias).exists():put(p,alias,p.name not in PROMPT_CONFIGS)
    for name in ANALYSIS:
        p=rev/name;put(p,Path('code/original')/name)
        text=p.read_text(encoding='utf-8-sig')
        # Portable copies differ only in I/O paths, recursive manifest path resolution and output destination.
        text=re.sub(r'^ROOT = .+$','ROOT = BUNDLE_ROOT',text,flags=re.M)
        text=re.sub(r'^PROJECT = .+$','PROJECT = PROJECT_ROOT',text,flags=re.M)
        text=re.sub(r'^RUNTIME = .+$','RUNTIME = DATA_ROOT',text,flags=re.M)
        text=re.sub(r'^R = .+$','R = DATA_ROOT',text,flags=re.M)
        text=re.sub(r'^SNAPSHOT = .+$','SNAPSHOT = SOURCE_ROOT',text,flags=re.M)
        text=re.sub(r'^OUT = .+$','OUT = OUTPUT_DIR',text,flags=re.M)
        text=re.sub(r'^E = .+$','E = OUTPUT_DIR',text,flags=re.M)
        text=text.replace('ROOT / "evidence"','OUTPUT_DIR').replace("ROOT / 'evidence'",'OUTPUT_DIR')
        text=text.replace('if args.limit and args.output == OUTPUT_DIR:', 'if args.limit and args.output == BUNDLE_ROOT / "evidence":')
        text=re.sub(r'return json.loads\(([^\n]+)\)',r'return resolve_data(json.loads(\1))',text)
        prelude='from portable_paths import BUNDLE_ROOT, DATA_ROOT, SOURCE_ROOT, PROJECT_ROOT, OUTPUT_DIR, resolve_data\n'
        if 'from __future__ import annotations\n' in text:text=text.replace('from __future__ import annotations\n','from __future__ import annotations\n'+prelude,1)
        else:text=prelude+text
        target=out/'code/analysis'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(text,encoding='utf-8')
    (out/'code/analysis/portable_paths.py').write_text(PORTABLE_PATHS,encoding='utf-8')
    # Preserve cross-package comparison calculations, replacing only input/output paths.
    qa=(rev/'evidence/qa_group_cohorts.py').read_text(encoding='utf-8-sig')
    qa=re.sub(r'^R=Path\(.+\)$',"import argparse\n_parser=argparse.ArgumentParser();_parser.add_argument('--earlier',type=Path,required=True);_parser.add_argument('--later',type=Path,required=True);_parser.add_argument('--output',type=Path,required=True);_args=_parser.parse_args()\n_args.output.parent.mkdir(parents=True,exist_ok=True)",qa,flags=re.M)
    qa=re.sub(r'^A=Path\(.+\)$','A=_args.earlier',qa,flags=re.M)
    qa=re.sub(r'^B=Path\(.+\)$','B=_args.later',qa,flags=re.M)
    qa=qa.replace("(R/'evidence/qa_group_cohorts.json')",'_args.output')
    (out/'tools/compare_prediction_cohorts.py').parent.mkdir(parents=True,exist_ok=True)
    (out/'tools/compare_prediction_cohorts.py').write_text(qa,encoding='utf-8')
    psnr=rev/'evidence/qa_psnr_intervals.py'
    if psnr.exists():
        code=psnr.read_text(encoding='utf-8-sig')
        code=code.replace('import numpy as np','import numpy as np\nimport argparse\n_parser=argparse.ArgumentParser();_parser.add_argument("--data-root",type=Path,required=True);_parser.add_argument("--output",type=Path,required=True);_args=_parser.parse_args()\n_args.output.parent.mkdir(parents=True,exist_ok=True)')
        code=re.sub(r'^SOURCE = .+$',"SOURCE = _args.data_root/'experiments/paper_v031_identity_frontend_20260828/03_completion_group_frontend'",code,flags=re.M)
        code=code.replace("(ROOT/'evidence/qa_psnr_intervals.json').write_text",'_args.output.write_text')
        (out/'tools/verify_psnr_intervals.py').write_text(code,encoding='utf-8')
    # Explicit numeric metadata and generated prediction masks only.
    bench=runtime/'experiments/paper_v031/test226_hpid';hold=runtime/'experiments/paper_v031_untouched_group_holdout_42_20260825_r5';front=runtime/'experiments/paper_v031_identity_frontend_20260828'
    bench_data=read(bench/'benchmark_summary.json');manifest=Path(bench_data['source_manifest']);lookup={r['case_id']:r for r in read(manifest)['cases']}
    refs={r['case_id']:r for r in read(hold/'01_sealed_reference_manifest.json')['cases']}
    items=[('test226',r['case_id'],bench/r['case_id'],Path(lookup[r['case_id']]['case_path'])) for r in bench_data['cases'] if r['return_code']==0]
    items += [('holdout42',r['case_id'],hold/'02_blind_inference'/r['case_id'],Path(refs[r['case_id']]['case_path'])) for r in read(hold/'01_blind_input_manifest.json')['cases']]
    zip_path=out/'inputs/prediction_packages.zip';zip_path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        seen=set()
        def add(p):
            rel=p.relative_to(runtime).as_posix()
            if rel in seen:return
            seen.add(rel);raw=p.read_bytes();data=cleaned(p)
            info=zipfile.ZipInfo(rel,(2026,10,8,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,data)
            members.append({'path':rel,'bytes':len(data),'sha256':digest(data),'original_sha256':digest(raw),'path_redacted':raw!=data})
        def omit(p,role,split=None,cid=None):
            if not p.exists():raise FileNotFoundError(p)
            rel=p.relative_to(runtime).as_posix()
            if not any(r['path']==rel for r in omitted):omitted.append({'path':rel,'role':role,'split':split,'case_id':cid,'bytes':p.stat().st_size,'sha256':digest(p.read_bytes())})
        add(bench/'benchmark_summary.json');add(manifest)
        for n in ['01_blind_input_manifest.json','01_sealed_reference_manifest.json','02_blind_inference_manifest.json']:add(hold/n)
        for split,cid,pkg,casepath in items:
            case=read(casepath);add(casepath)
            for name in ['candidates.json','part_id_map.tiff','parts.json','group_id_map.tiff','groups.json','taxonomy.json']:add(pkg/name)
            for r in read(pkg/'candidates.json'):add(pkg/r['mask_path'])
            omit(pkg/'source.png','source_rgb',split,cid)
            omit(casepath.parent/'object_mask_crop.png','reference_object',split,cid)
            for r in case['parts']:omit(casepath.parent/r['mask_crop'],'reference_part',split,cid)
            case_manifest.append({'split':split,'case_id':cid,'package':pkg.relative_to(runtime).as_posix(),'reference_case':casepath.relative_to(runtime).as_posix(),**{k:case[k] for k in ['dataset','dataset_repository','image_id','object_annotation_id','object_category','crop_box_xyxy','source_annotation_sha256','source_image_sha256','source_image_url','source_image_urls_from_annotation'] if k in case},'parts':[{'annotation_id':p['annotation_id'],'part_name':p['part_name'],'mask_crop':p['mask_crop']} for p in case['parts']]})
        for folder in ['01_same_candidate','03_completion_group_frontend']:
            for p in sorted((front/folder).glob('*.csv')):add(p)
        target_manifest=front/'02_completion_frontend/completion_target_manifest.json';add(target_manifest)
        for t in read(target_manifest)['selected_cases']:
            cid=t['case_id'];base=front/'03_completion_group_frontend/cases'/cid
            omit(base/'truth_part.png','request_truth','holdout42',cid)
            for method in ['raw_proposals','dbscan_fusion','hpid_split_group_ids']:add(base/method/'selected_part.png')
        for rel in ['.venv/Lib/site-packages/transformers/models/grounding_dino/processing_grounding_dino.py','.venv/Lib/site-packages/transformers/pipelines/mask_generation.py']:add(runtime/rel)
    save(out/'inputs/archive_members.json',members);save(out/'inputs/omitted_input_requirements.json',omitted);save(out/'inputs/case_reconstruction_manifest.json',case_manifest)
    from PIL import Image
    pixel_counts={'binary_candidate_masks':0,'binary_selected_masks':0,'integer_id_maps':0};mode_counts={}
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if Path(name).suffix.lower() not in {'.png','.tif','.tiff'}:continue
            with Image.open(io.BytesIO(z.read(name))) as im:
                mode_counts[im.mode]=mode_counts.get(im.mode,0)+1
                if Path(name).suffix.lower()=='.png':
                    assert im.mode in {'1','L'},(name,im.mode)
                    assert set(im.get_flattened_data()) <= {0,1,255},name
                    kind='binary_candidate_masks' if '/candidate_masks/' in name else 'binary_selected_masks'
                    pixel_counts[kind]+=1
                else:
                    assert Path(name).name in {'part_id_map.tiff','group_id_map.tiff'},name
                    assert im.mode in {'I','I;16','I;16L','I;16B'},(name,im.mode)
                    assert im.getextrema()[0]>=0,name
                    pixel_counts['integer_id_maps']+=1
    assert pixel_counts=={'binary_candidate_masks':2400,'binary_selected_masks':111,'integer_id_maps':536},pixel_counts
    import numpy as np
    npz_audit=[]
    for p in sorted((out/'evidence/gate_ablation').glob('*.npz')):
        with np.load(p,allow_pickle=False) as arrays:
            for key in arrays.files:
                arr=arrays[key];assert arr.ndim==2 and arr.dtype==np.uint16,(p.name,key)
                npz_audit.append({'file':p.relative_to(out).as_posix(),'variant':key,'shape':list(arr.shape),'dtype':str(arr.dtype)})
    save(out/'PREDICTION_PIXEL_AUDIT.json',{'status':'PASS','image_members':sum(pixel_counts.values()),'counts':pixel_counts,'modes':mode_counts,'counterfactual_npz_arrays':npz_audit,'method':'Every PNG is single-channel binary; every TIFF is a nonnegative integer fine/Group ID map. Four supplementary repeat-audit NPZ files contain only32 two-dimensional uint16 ID maps. File-member whitelist also excludes source/reference masks. No RGB image members are included.'})
    # Standard-library preparation and inspection utilities.
    for name,body in [('prepare_inputs.py',PREPARE_INPUTS),('run_analysis.py',RUN_ANALYSIS),('verify_bundle.py',VERIFY_BUNDLE)]:
        p=out/'tools'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(body,encoding='utf-8')
    if (rev/'verify_public_metrics.py').exists():put(rev/'verify_public_metrics.py','tools/verify_public_metrics.py')
    final_numeric=rev/'qa/public_bundle_numeric_privacy_audit.json'
    if final_numeric.exists() and read(final_numeric)['status']=='PASS':put(final_numeric,'verification/public_metrics_and_privacy.json')
    put(Path(__file__),'code/packaging/prepare_public_evidence.py')
    for p in (runtime/'.venv/Lib/site-packages/transformers-5.12.1.dist-info').rglob('LICENSE*'):
        if p.is_file():put(p,Path('licenses/transformers')/p.name,False)
    save(out/'LICENSE_AUDIT.json',{'checked_date':'2026-10-08','sources':[{'url':'https://github.com/facebookresearch/paco','finding':'README limits its license statement to source code; lists separate annotation/image downloads.'},{'url':'https://github.com/facebookresearch/paco/blob/main/LICENSE','finding':'MIT license covers software and associated documentation; this check does not establish redistribution permission for separate dataset images/reference mask files.'},{'url':'https://github.com/facebookresearch/paco/blob/main/docs/PACO_DATASET.md','finding':'PACO-LVIS derives from LVIS images and provides image/annotation IDs; the image records carry their own license identifiers.'}],'decision':'Source photos and reference-mask pixels excluded. Numeric IDs, crop coordinates, source URLs and hashes permit reconstruction from separately obtained copies. Included binary prediction masks and fine/Group maps are generated HPID outputs. No inference of a common image license is made.','repo_license':'HPID source snapshots retain their original LICENSE and THIRD_PARTY notices; no new licensing declaration is fabricated.'})
    (out/'README.md').write_text(README,encoding='utf-8')
    (out/'requirements-replay.txt').write_text('numpy==2.5.1\nscipy==1.18.0\nopencv-python==5.0.0\nPillow==12.3.0\n',encoding='utf-8')
    save(out/'SOURCE_COPY_AUDIT.json',audit)
    save(out/'INPUT_AVAILABILITY.json',{'cases':len(items),'test_cases':226,'holdout_cases':42,'generated_prediction_candidates':2400,'archive_members':len(members),'archive_bytes':zip_path.stat().st_size,'omitted_files':len(omitted),'omitted_bytes':sum(r['bytes'] for r in omitted),'numerical_verification_requires_images':False,'image_level_replay_requires_omitted_inputs':True,'source_images_in_bundle':False,'reference_mask_pixels_in_bundle':False,'input_paths_resolved_by':'code/analysis/portable_paths.py resolves ${RUNTIME_ROOT}, ${PROJECT_ROOT} and related markers recursively when loading manifests. No original drive letters are required.'})
    portability_validation(rev,out)
    manifest_out=[]
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name!='FILE_MANIFEST.json':manifest_out.append({'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p.read_bytes())})
    save(out/'FILE_MANIFEST.json',manifest_out)
    print(json.dumps({'output':str(out),'files':len(manifest_out)+1,'bytes':sum(r['bytes'] for r in manifest_out),'archive_members':len(members),'omitted_inputs':len(omitted)},indent=2))

PORTABLE_PATHS=r'''from pathlib import Path
import os
BUNDLE_ROOT=Path(__file__).resolve().parents[2]
DATA_ROOT=Path(os.environ.get('HPID_DATA_ROOT',BUNDLE_ROOT/'inputs/runtime')).resolve()
SOURCE_ROOT=BUNDLE_ROOT/'sources/frozen_be54300'
PROJECT_ROOT=BUNDLE_ROOT/'sources/project'
OUTPUT_DIR=Path(os.environ.get('HPID_RESULTS_DIR',BUNDLE_ROOT/'replay_results')).resolve()
OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('HPID_HISTORICAL_GROUP_SOURCE',str(BUNDLE_ROOT/'sources/historical_da7d236/physical_groups.py'))
def resolve_data(value):
    if isinstance(value,dict):return {k:resolve_data(v) for k,v in value.items()}
    if isinstance(value,list):return [resolve_data(v) for v in value]
    if not isinstance(value,str):return value
    replacements={'${RUNTIME_ROOT}':DATA_ROOT,'${PROJECT_ROOT}':PROJECT_ROOT/'hpid_split','${REVISION_ROOT}':BUNDLE_ROOT,'${WORKSPACE_ROOT}':PROJECT_ROOT,'${BUNDLE_ROOT}':BUNDLE_ROOT}
    for marker,path in replacements.items():
        if marker in value:value=value.replace(marker,str(path)).replace('\\','/')
    return value
'''

PREPARE_INPUTS=r'''"""Extract generated outputs and optionally add separately supplied local inputs."""
from pathlib import Path
import argparse,hashlib,json,shutil,zipfile
B=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,default=B/'inputs/runtime');p.add_argument('--local-runtime',type=Path);p.add_argument('--extract',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args();root=a.data_root.resolve()
    if a.extract:
        with zipfile.ZipFile(B/'inputs/prediction_packages.zip') as z:
            for member in z.infolist():
                target=(root/member.filename).resolve()
                if not target.is_relative_to(root):raise ValueError('Unsafe archive member')
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(member))
    required=json.loads((B/'inputs/omitted_input_requirements.json').read_text())
    if a.local_runtime:
        if root==a.local_runtime.resolve():raise ValueError('Replay root must differ from archival runtime')
        for r in required:
            src=a.local_runtime/r['path'];dst=root/r['path']
            if not src.is_file() or sha(src)!=r['sha256']:raise ValueError('Missing or hash-mismatched locally supplied input: '+r['path'])
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    missing=[r['path'] for r in required if not (root/r['path']).is_file()]
    bad=[r['path'] for r in required if (root/r['path']).is_file() and sha(root/r['path'])!=r['sha256']]
    generated=json.loads((B/'inputs/archive_members.json').read_text());genmissing=[r['path'] for r in generated if not (root/r['path']).is_file()];genbad=[r['path'] for r in generated if (root/r['path']).is_file() and sha(root/r['path'])!=r['sha256']]
    report={'generated_files':len(generated),'generated_missing':len(genmissing),'generated_hash_mismatches':genbad,'external_files':len(required),'external_missing':len(missing),'external_hash_mismatches':bad,'complete_image_replay_inputs':not(missing or bad or genmissing or genbad),'first_missing_external':missing[:5]}
    print(json.dumps(report,indent=2))
    if bad or genbad:raise SystemExit(2)
if __name__=='__main__':main()
'''

RUN_ANALYSIS=r'''"""Run path-adapted copies without altering the archived numerical evidence."""
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys
B=Path(__file__).resolve().parents[1]
FILES={'taxonomy':'taxonomy_behavior.py','score':'score_provenance.py','routing':'analyze_revision.py','sensitivity':'analyze_revision.py','holdout':'analyze_revision.py','gate':'gate_ablation.py','stage':'stage_diagnosis.py','request':'request_provenance.py'}
def main():
    p=argparse.ArgumentParser();p.add_argument('analysis',choices=FILES);p.add_argument('--data-root',type=Path,default=B/'inputs/runtime');p.add_argument('--output',type=Path,default=B/'replay_results');p.add_argument('--limit',type=int,default=0);p.add_argument('--workers',type=int,default=2);p.add_argument('--check-inputs',action='store_true');a=p.parse_args()
    root=a.data_root.resolve();out=a.output.resolve();required=json.loads((B/'inputs/omitted_input_requirements.json').read_text())
    roles={'gate':{'source_rgb','reference_object','reference_part'},'sensitivity':{'reference_object','reference_part'},'holdout':{'reference_object','reference_part'},'stage':{'reference_object','reference_part'},'request':{'source_rgb','reference_part','request_truth'}}.get(a.analysis,set())
    split='test226' if a.analysis in ['sensitivity','stage'] else 'holdout42' if a.analysis in ['holdout','request'] else None
    missing=[r['path'] for r in required if r['role'] in roles and (split is None or r['split']==split) and not (root/r['path']).is_file()]
    if missing:print(json.dumps({'analysis':a.analysis,'external_inputs_missing':len(missing),'examples':missing[:5],'instruction':'Supply locally obtained inputs identified by inputs/omitted_input_requirements.json; run tools/prepare_inputs.py --check.'},indent=2));raise SystemExit(2)
    if a.check_inputs:print(json.dumps({'analysis':a.analysis,'external_inputs_missing':0}));return
    out.mkdir(parents=True,exist_ok=True);env=dict(os.environ,HPID_DATA_ROOT=str(root),HPID_RESULTS_DIR=str(out),PYTHONHASHSEED='20261008',PYTHONDONTWRITEBYTECODE='1')
    if a.analysis=='score' and not (out/'source_reliability_candidates.csv').exists():shutil.copyfile(B/'evidence/source_reliability_candidates.csv',out/'source_reliability_candidates.csv')
    cmd=[sys.executable,'-B','-X','utf8',str(B/'code/analysis'/FILES[a.analysis])]
    if a.analysis in ['routing','sensitivity','holdout']:cmd.append(a.analysis)
    if a.analysis in ['gate','stage','sensitivity']:cmd+=['--workers',str(a.workers)]
    if a.limit and a.analysis in ['gate','stage','sensitivity','request']:cmd+=['--limit',str(a.limit)]
    if a.analysis=='gate':cmd+=['--output-dir',str(out/'gate_ablation')]
    if a.analysis=='request':cmd+=['--output',str(out)]
    raise SystemExit(subprocess.call(cmd,env=env,cwd=B))
if __name__=='__main__':main()
'''

VERIFY_BUNDLE=r'''from pathlib import Path
import hashlib,json,zipfile
B=Path(__file__).resolve().parents[1]
files=json.loads((B/'FILE_MANIFEST.json').read_text())
actual={p.relative_to(B).as_posix() for p in B.rglob('*') if p.is_file() and '.git' not in p.parts}
declared={r['path'] for r in files}|{'FILE_MANIFEST.json'}
assert actual==declared,{'unlisted_files':sorted(actual-declared),'missing_files':sorted(declared-actual)}
for r in files:
    p=B/r['path'];assert p.is_file(),r['path'];assert len(p.read_bytes())==r['bytes'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],r['path']
with zipfile.ZipFile(B/'inputs/prediction_packages.zip') as z:
    members=json.loads((B/'inputs/archive_members.json').read_text());assert len(members)==len(z.namelist())
    for r in members:assert hashlib.sha256(z.read(r['path'])).hexdigest()==r['sha256'],r['path']
    forbidden=[n for n in z.namelist() if Path(n).name in {'source.png','source_crop.png','truth_part.png','object_mask_crop.png'} or '/parts_crop/' in n or 'LOCAL_ONLY' in n]
    assert not forbidden,forbidden
assert not list(B.rglob('*LOCAL_ONLY*'))
print(json.dumps({'status':'PASS','files_checked':len(files),'archive_members_checked':len(members),'source_photos_and_reference_mask_pixels_excluded':True},indent=2))
'''

README='''# HPID-Split post-review experimental evidence — 2026-10-08

This additive evidence bundle accompanies the constrained IEEE Access revision. It preserves the measured results, including adverse findings. It does not replace frozen releases/tags, create independent new test cases, claim upstream model retraining, or establish a learned-baseline accuracy comparison.

## Contents and scope

- `evidence/`: full numerical CSV/JSON results, protocols and logs for the 50-setting fine-ID sensitivity, 42-case paired root diagnostic, shared-input/stage audit, 37-request provenance, six unique failures, taxonomy fixtures, score provenance and eight-subset Group gate experiment.
- `evidence/gate_ablation/`: 2,144 condition rows. Earlier226 Group results use recovered `da7d236`; holdout42 uses `be54300`. All268 baseline maps and ID/semantic sequences reproduce exactly. Candidate/fine-map inputs are fixed. Two counterfactual cases have one-pixel repeat differences of unassigned cause despite a fixed hash seed; reported metrics and intervals agree across full repeated runs. Read the report before interpreting changed-map counts.
- `evidence/qa_group_cohorts.json` and its audit script distinguish the earlier 1,951-candidate test226 package used by these fixed-pool analyses (Group F1@0.25=0.3661978203) from the later 2,188-candidate v0.3.1 Group regression package (0.4042812032). The 226 RGB inputs match, but only 19 fine maps and 20 Group maps are identical. These are separate prediction packages on the same cases; the results are not pooled. The additional later historical package needed to rerun this cross-package audit is not included.
- `sources/`: frozen algorithm/configuration sources, historical Group source, and unchanged comparison/evaluation helpers. `code/original/` retains analysis code with machine paths redacted; `code/analysis/` contains path-adapted runnable copies. The adapter changes I/O paths and recursively resolves manifest tokens; it does not retune algorithms.
- `inputs/prediction_packages.zip`: 268 sets of generated candidate masks/scores/metadata, fine/Group maps and records, numerical frontend tables and manifest metadata. No source photographs or reference-mask pixels are included.
- `inputs/case_reconstruction_manifest.json`: original PACO image/object/part annotation IDs, crop boxes, source URLs and hashes. `omitted_input_requirements.json` lists every externally supplied file and expected SHA256. This is an explicit availability boundary, not a self-contained image-level replay archive.
- `FILE_MANIFEST.json`, `SOURCE_COPY_AUDIT.json`: public checksums/sizes and original-to-public path-redaction provenance. Numerical values are not beautified. Local paths become `${RUNTIME_ROOT}`, `${PROJECT_ROOT}` and related markers.
- `PREDICTION_PIXEL_AUDIT.json` confirms that all 3,047 image members are binary generated masks or integer ID maps, with no RGB members. `sources/project/hpid_split/` is the canonical helper location used by the runner; retained source aliases contain the same path-sanitized material.
- Gate `backups/` retain complete first-pass, integrity-pass and final-run CSV/JSON/log records. Four supplementary NPZ files contain only 32 uint16 counterfactual ID maps documenting the repeat limitation; they contain no RGB or reference pixels. `code/packaging/prepare_public_evidence.py` records the staging procedure and requires the original local source archives.

## Verify the public numerical package without images

File integrity uses only the Python standard library. The independent numerical checker additionally requires NumPy; it imports no algorithm modules, images or models:

```sh
python tools/verify_bundle.py
python tools/verify_public_metrics.py . --report ../metric_verification.json
```

The optional `--privacy` flag also requires Pillow and decodes every published prediction image to check its single-channel binary-mask or integer-ID-map format. The saved `verification/public_metrics_and_privacy.json` records 275,151 numeric consistency checks, 1,283 independently recomputed intervals and the filename/text/pixel audit. Write fresh reports outside this bundle if retaining exact manifest integrity.

The metric checker uses saved CSV/JSON values. It does not infer new masks or independently validate image annotations. Its report distinguishes what was recomputed from what is an archived execution assertion.

The optional cross-package audit can be repeated when the additional later historical package is separately available:

```sh
python tools/compare_prediction_cohorts.py --earlier /path/to/earlier226 --later /path/to/later226 --output /path/to/results/qa_group_cohorts.json
```

The saved export-contract QA records checks against original package manifests, visible-part masks and linked diagnostics beyond the minimal replay inputs. Its original audit source is included; these extra archived package files must be supplied to repeat that file-link audit.

`evidence/qa_psnr_intervals.json` verifies that the original and post-review PSNR intervals use the same 37 paired values. They differ because of bootstrap seed and case order, not changed outcomes. After extracting the generated archive, `python tools/verify_psnr_intervals.py --data-root /path/to/replay_data --output /path/to/results/psnr_interval_audit.json` repeats both intervals using only the included numerical tables. No external image files are required for this check.

## Prepare image-level replay

Recorded environment: Python3.12.2, NumPy2.5.1, SciPy1.18.0, OpenCV5.0.0, Pillow12.3.0; exact recorded versions are in `evidence/stage_run_environment.json`. The replay requirement file records these versions. GPU checkpoints and detector inference are unnecessary for fixed-pool analyses. The complete old model-generation workflow is outside this bundle.

```sh
python tools/prepare_inputs.py --data-root /path/to/replay_data --extract --check
```

This first command must report missing external files. On Windows, use a short absolute data-root path (for example `D:/hpid_replay`) because archived case filenames are long. Obtain the source images and PACO-LVIS annotations under their applicable terms and reproduce the crops/reference masks specified by the reconstruction manifest. Place them at the listed relative paths and verify their hashes. Users with the original local archive may copy only the omitted files, without altering that archive:

```sh
python tools/prepare_inputs.py --data-root /path/to/replay_data --local-runtime /path/to/original_runtime --check
```

PACO's official [README](https://github.com/facebookresearch/paco) provides annotation/image acquisition instructions and describes the repository license as applying to source code. The [MIT license](https://github.com/facebookresearch/paco/blob/main/LICENSE) and [dataset description](https://github.com/facebookresearch/paco/blob/main/docs/PACO_DATASET.md) did not establish a blanket permission to redistribute the separate images/reference-mask files in this audit; these files are therefore omitted. Included prediction masks/maps are generated HPID outputs. No image license is fabricated. Existing HPID source-license notices remain unchanged.

## Run analyses into a new output directory

```sh
python tools/run_analysis.py taxonomy --output /path/to/results
python tools/run_analysis.py score --data-root /path/to/replay_data --output /path/to/results
python tools/run_analysis.py routing --data-root /path/to/replay_data --output /path/to/results
python tools/run_analysis.py sensitivity --data-root /path/to/replay_data --output /path/to/results --workers 2
python tools/run_analysis.py holdout --data-root /path/to/replay_data --output /path/to/results
python tools/run_analysis.py gate --data-root /path/to/replay_data --output /path/to/results --workers 2
python tools/run_analysis.py stage --data-root /path/to/replay_data --output /path/to/results --workers 2
python tools/run_analysis.py request --data-root /path/to/replay_data --output /path/to/results
```

Use `--limit 3` for gate/stage/request/sensitivity smoke checks; trial output must be separate from complete runs. `--check-inputs` reports missing external files before image-level analysis. The runner writes outside `evidence/` by default and does not modify original experiments. External PartCATSeg/HOPS entrypoint failure logs describe the recorded local environment; they are not trained accuracy runs and are not promised to recur on a different installation.

Root42 and parameter sweeps reuse the same archived cases. Gate nominees are not all exported candidates. Stage diagnostics include unions/intermediate argmax proxies and must not be mistaken for interventions or added samples. Routing uniqueness is not calibrated confidence; retain the six failures and adverse PSNR outcomes. See each protocol for matching rules, denominators, paired units, confidence intervals and limitations.
'''

if __name__=='__main__':main()
