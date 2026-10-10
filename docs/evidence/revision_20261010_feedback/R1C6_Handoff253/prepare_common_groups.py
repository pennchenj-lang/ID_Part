"""Rebuild one common be54300 Groups interface from frozen proposal evidence.

Only four manifest fields are consumed: request_id, case_id, cohort, package_path.
No target masks, reference annotations, requests or routing outcomes are read.
All archive production FusionConfig fields are restored for both cohorts.
Original37 exact reproduction and all253 exact Part maps are mandatory gates.
"""
from __future__ import annotations
from dataclasses import asdict, fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import traceback

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SOURCE = Path('__RUNTIME_ROOT__/code_snapshots/hpid_split_be54300_holdout/src')
sys.path.insert(0, str(SOURCE))
import cv2
import numpy as np
from PIL import Image
from hpid_split.fusion import FusionConfig, MaskCandidate, fuse_candidates
from hpid_split.physical_groups import build_physical_groups
cv2.setNumThreads(1)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def serialize_numpy(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False, default=serialize_numpy), encoding='utf-8')


def canonical(value):
    return json.loads(json.dumps(value, default=serialize_numpy))


def array_hash(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def binary_hash(value):
    return hashlib.sha256(np.packbits(np.asarray(value, dtype=np.uint8).ravel()).tobytes()).hexdigest()


def production_config(raw):
    names = {field.name for field in fields(FusionConfig)}
    kwargs = {}
    for key, value in raw.items():
        actual = key if key in names else 'use_' + key
        if actual not in names:
            raise ValueError('Unknown archived FusionConfig field: ' + key)
        if actual in kwargs:
            raise ValueError('Duplicate config alias: ' + actual)
        kwargs[actual] = value
    return FusionConfig(**kwargs)


def group_mask_ids(labels, groups):
    return {row['group_id']: binary_hash(labels == int(row['group_index'])) for row in groups}


def partition_equal(old, new):
    if old.shape != new.shape or not np.array_equal(old == 0, new == 0):
        return False
    pairs = np.unique(np.stack((old.ravel(), new.ravel()), axis=1), axis=0)
    return len(pairs) == len(np.unique(old)) == len(np.unique(new))


def main():
    manifest_path = HERE / 'request_manifest_draft.json'
    output = HERE / 'common_groups'
    overrides_path = HERE / 'group_overrides.json'
    if output.exists() or overrides_path.exists():
        raise FileExistsError('Preserve prior evidence: common_groups or group_overrides already exists')
    manifest_hash = sha(manifest_path)
    draft = read(manifest_path)
    requests = [{key: row[key] for key in ('request_id', 'case_id', 'cohort', 'package_path')}
                for row in draft['requests']]
    del draft
    if len(requests) != 253 or len({row['request_id'] for row in requests}) != 253:
        raise ValueError('Expected 253 unique requests')
    if sum(row['cohort']=='old37' for row in requests) != 37 or sum(row['cohort']=='new216' for row in requests) != 216:
        raise ValueError('Expected old37/new216 cohorts')
    for row in requests:
        if not re.fullmatch('[A-Za-z0-9_-]{1,150}', row['request_id']):
            raise ValueError('Unsafe request_id')
    # Original37 is a hard gate before any expanded-cohort Group generation.
    requests.sort(key=lambda row: (row['cohort'] != 'old37', row['request_id']))
    source_hashes = {name: sha(SOURCE/'hpid_split'/name) for name in
                     ('fusion.py','physical_groups.py','instances.py','taxonomy.py','export.py','cli.py')}
    config_rows = {}
    raw_variants = set()
    full_variants = set()
    for row in requests:
        path = Path(row['package_path'])/'package_manifest.json'
        raw = read(path)['algorithm']['fusion_config']
        config = production_config(raw)
        raw_variants.add(json.dumps(raw, sort_keys=True))
        full_variants.add(json.dumps(asdict(config), sort_keys=True))
        config_rows[row['request_id']] = {'archive_manifest_sha256':sha(path),'raw':raw,'config':config}
    if len(raw_variants) != 1 or len(full_variants) != 1:
        raise ValueError('Archive production configurations differ; stop before choosing a common config')
    common_config = config_rows[requests[0]['request_id']]['config']
    complete_config = asdict(common_config)
    output.mkdir(parents=True)
    progress_path = output/'preparation_receipt.json'
    receipt = {'status':'RUNNING','started_utc':datetime.now(timezone.utc).isoformat(),
      'source_snapshot':str(SOURCE),'source_sha256':source_hashes,'script_sha256':sha(__file__),
      'request_manifest_path':str(manifest_path),'request_manifest_sha256':manifest_hash,
      'consumed_manifest_fields':['request_id','case_id','cohort','package_path'],
      'ground_truth_read':False,'routing_or_score_results_read':False,'GPU_used':False,
      'archive_files_modified':False,'manifest_modified':False,'grouping_provisional_scene_labels':False,
      'production_config_archive':json.loads(next(iter(raw_variants))),
      'production_config_complete':complete_config,
      'config_differences_from_unmodified_FusionConfig_defaults':{
          key:value for key,value in complete_config.items() if value != asdict(FusionConfig())[key]},
      'planned_cases':253,'original37_required_exact':True,'cases':[]}
    write(progress_path,receipt)
    overrides=[]
    start=time.perf_counter()
    try:
        for request in requests:
            request_id=request['request_id']
            package=Path(request['package_path'])
            case_dir=output/request_id
            case_dir.mkdir()
            files=['package_manifest.json','source.png','candidates.json','parts.json','part_id_map.tiff','groups.json','group_id_map.tiff']
            input_hashes={filename:sha(package/filename) for filename in files}
            if input_hashes['package_manifest.json'] != config_rows[request_id]['archive_manifest_sha256']:
                raise RuntimeError('Archive config changed during preparation: '+request_id)
            archived_candidates=read(package/'candidates.json')
            candidates=[]
            candidate_mask_files=[]
            for candidate in archived_candidates:
                path=package/candidate['mask_path']
                candidate_mask_files.append({'path':candidate['mask_path'],'sha256':sha(path)})
                with Image.open(path) as im:
                    mask=np.asarray(im.convert('L'))>=128
                candidates.append(MaskCandidate(
                    semantic_name=str(candidate['semantic_name']),semantic_parent=str(candidate['semantic_parent']),
                    mask=mask,score=float(candidate['score']),source=str(candidate['source']),
                    prompt=str(candidate.get('prompt','')),source_reliability=float(candidate.get('source_reliability',1)),
                    metadata=dict(candidate.get('metadata') or {})))
            with Image.open(package/'source.png') as im:
                image=im.convert('RGB')
            with Image.open(package/'part_id_map.tiff') as im:
                archived_part_map=np.asarray(im).copy()
            with Image.open(package/'group_id_map.tiff') as im:
                archived_group_map=np.asarray(im).copy()
            archived_groups=read(package/'groups.json')
            archived_parts=read(package/'parts.json')
            before=time.perf_counter()
            fused=fuse_candidates(candidates,image_shape=archived_part_map.shape,config=common_config)
            if not np.array_equal(fused.instance_map,archived_part_map):
                raise RuntimeError('Part map changed; halt entire preparation: '+request_id)
            grouping=build_physical_groups(fused.instance_map,fused.instances,
                                           candidates=fused.accepted_candidates,image=image,
                                           provisional_scene_labels=False)
            groups=canonical([group.to_dict() for group in grouping.groups])
            part_records=canonical([record.to_dict() for record in grouping.records])
            map_exact=bool(np.array_equal(grouping.group_map,archived_group_map))
            ids_exact=[g['group_id'] for g in groups]==[g['group_id'] for g in archived_groups]
            json_exact=groups==archived_groups
            fields_to_compare=list(part_records[0]) if part_records else []
            archived_part_records=[{key:part.get(key) for key in fields_to_compare} for part in archived_parts]
            parts_exact=part_records==archived_part_records
            comparison={'part_map_exact':True,'group_map_exact':map_exact,'group_ids_exact':ids_exact,
                        'groups_json_exact':json_exact,'part_records_with_group_memberships_exact':parts_exact,
                        'group_partition_equal_up_to_numeric_relabeling':partition_equal(archived_group_map,grouping.group_map),
                        'group_id_associated_masks_exact':group_mask_ids(archived_group_map,archived_groups)==group_mask_ids(grouping.group_map,groups),
                        'different_group_pixels':int(np.count_nonzero(grouping.group_map!=archived_group_map)),
                        'archived_group_count':len(archived_groups),'common_group_count':len(groups)}
            if request['cohort']=='old37' and not (map_exact and ids_exact and json_exact and parts_exact):
                write(case_dir/'comparison.json',comparison)
                raise RuntimeError('Original37 Group reproduction gate failed: '+request_id)
            map_path=case_dir/'group_id_map.tiff'
            groups_path=case_dir/'groups.json'
            Image.fromarray(grouping.group_map.astype(np.uint16)).save(map_path)
            write(groups_path,groups)
            write(case_dir/'parts_with_groups.json',part_records)
            write(case_dir/'fused_diagnostics.json',fused.diagnostics)
            write(case_dir/'grouping_diagnostics.json',grouping.diagnostics)
            light_candidates=[]
            for index,candidate in enumerate(fused.accepted_candidates):
                packed_hash=binary_hash(candidate.mask)
                key_payload={'semantic_name':candidate.semantic_name,'semantic_parent':candidate.semantic_parent,
                             'source':candidate.source,'prompt':candidate.prompt,'score':float(candidate.score),
                             'source_reliability':float(candidate.source_reliability),'mask_sha256':packed_hash,
                             'metadata':candidate.metadata}
                stable_key=hashlib.sha256(json.dumps(key_payload,sort_keys=True,default=serialize_numpy).encode()).hexdigest()
                light_candidates.append({'index':index,'candidate_key':stable_key,**key_payload,
                                         'mask_shape':list(candidate.mask.shape),'mask_area':int(np.count_nonzero(candidate.mask))})
            write(case_dir/'accepted_candidates_metadata.json',light_candidates)
            with Image.open(map_path) as im:
                if not np.array_equal(np.asarray(im),grouping.group_map):
                    raise RuntimeError('Saved Group map differs from reconstruction')
            if read(groups_path)!=groups:
                raise RuntimeError('Saved Groups JSON differs from reconstruction')
            if not all(sha(package/name)==digest for name,digest in input_hashes.items()):
                raise RuntimeError('Archived principal input changed during processing')
            if not all(sha(package/item['path'])==item['sha256'] for item in candidate_mask_files):
                raise RuntimeError('Archived candidate mask changed during processing')
            detail={**request,'status':'PASS','comparison_to_archive':comparison,
                    'input_sha256':input_hashes,'candidate_mask_files':candidate_mask_files,
                    'source_RGB_pixels_sha256':array_hash(np.asarray(image)),
                    'archived_part_map_pixels_sha256':array_hash(archived_part_map),
                    'refused_part_map_pixels_sha256':array_hash(fused.instance_map),
                    'production_config_complete':complete_config,'elapsed_seconds':time.perf_counter()-before,
                    'input_candidate_count':len(candidates),'accepted_candidate_count':len(fused.accepted_candidates),
                    'outputs':{p.name:{'path':str(p.resolve()),'sha256':sha(p)} for p in [
                        map_path,groups_path,case_dir/'parts_with_groups.json',case_dir/'fused_diagnostics.json',
                        case_dir/'grouping_diagnostics.json',case_dir/'accepted_candidates_metadata.json']}}
            write(case_dir/'replay_receipt.json',detail)
            overrides.append({'request_id':request_id,'hpid_group_map_path':str(map_path.resolve()),
                              'hpid_groups_path':str(groups_path.resolve()),'hpid_group_map_sha256':sha(map_path),
                              'hpid_groups_sha256':sha(groups_path)})
            receipt['cases'].append({'request_id':request_id,'cohort':request['cohort'],'case_id':request['case_id'],
                                     'comparison_to_archive':comparison,'replay_receipt_path':str((case_dir/'replay_receipt.json').resolve()),
                                     'replay_receipt_sha256':sha(case_dir/'replay_receipt.json')})
            if len(receipt['cases'])%20==0 or len(receipt['cases'])==37:
                write(progress_path,receipt)
                print('Prepared',len(receipt['cases']),'of253; original37 gate', 'PASS' if len(receipt['cases'])>=37 else 'PENDING',flush=True)
        if sha(manifest_path)!=manifest_hash:
            raise RuntimeError('Request manifest changed during Group preparation')
        if not all(sha(SOURCE/'hpid_split'/name)==digest for name,digest in source_hashes.items()):
            raise RuntimeError('Frozen grouping source changed during preparation')
        # Overrides are published only after every mandatory gate passes.
        order={row['request_id']:i for i,row in enumerate(read(manifest_path)['requests'])}
        overrides.sort(key=lambda row:order[row['request_id']])
        write(overrides_path,overrides)
        receipt['status']='COMPLETE'
        receipt['completed_utc']=datetime.now(timezone.utc).isoformat()
        receipt['elapsed_seconds']=time.perf_counter()-start
        receipt['group_overrides']={'path':str(overrides_path.resolve()),'sha256':sha(overrides_path),'count':len(overrides)}
        receipt['summary']={cohort:{'count':sum(row['cohort']==cohort for row in receipt['cases']),
            'part_map_exact_count':sum(row['cohort']==cohort and row['comparison_to_archive']['part_map_exact'] for row in receipt['cases']),
            'group_map_exact_count':sum(row['cohort']==cohort and row['comparison_to_archive']['group_map_exact'] for row in receipt['cases']),
            'full_groups_JSON_exact_count':sum(row['cohort']==cohort and row['comparison_to_archive']['groups_json_exact'] for row in receipt['cases']),
            'group_ID_exact_count':sum(row['cohort']==cohort and row['comparison_to_archive']['group_ids_exact'] for row in receipt['cases']),
            'partition_changed_count':sum(row['cohort']==cohort and not row['comparison_to_archive']['group_partition_equal_up_to_numeric_relabeling'] for row in receipt['cases'])}
            for cohort in ('old37','new216')}
        receipt['interpretation']='Common frozen be54300 grouping version applied to both cohorts. Differences from historical test226 Groups reflect interface/version alignment, not selection by GT or routing performance.'
        write(progress_path,receipt)
        print(json.dumps({'status':receipt['status'],'summary':receipt['summary'],'group_overrides_sha256':sha(overrides_path),
                          'preparation_receipt_sha256':sha(progress_path),'elapsed_seconds':receipt['elapsed_seconds']}),flush=True)
    except Exception as error:
        receipt['status']='FAILED_STOPPED'
        receipt['error']={'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()}
        write(progress_path,receipt)
        raise


if __name__=='__main__':
    main()
