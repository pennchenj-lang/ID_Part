"""Independent loss coverage, mask lineage and stage-boundary audit."""
import collections
import datetime
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
import independent_routing_audit as common
import numpy as np
from PIL import Image
from hpid_split import fusion
from hpid_split.physical_groups import build_physical_groups

HERE = Path(__file__).resolve().parent
MECH = HERE / 'mechanism_analysis'


def fingerprint(candidate, mask_hash):
    value = {key:candidate.get(key, default) for key, default in
             [('semantic_name',''),('semantic_parent',''),('source',''),('prompt',''),('score',0.),
              ('source_reliability',1.),('metadata',{})]}
    value['score'] = float(value['score'])
    value['source_reliability'] = float(value['source_reliability'])
    value['mask_sha256'] = mask_hash
    return json.dumps(value, sort_keys=True)


def packed_sha(mask):
    return hashlib.sha256(np.packbits(np.asarray(mask,dtype=np.uint8).ravel()).tobytes()).hexdigest()


def native_rows(records, labels, unit, group=False):
    return [dict(identity=r['group_id'] if group else r['part_id'],semantic=r['semantic_name'],
                 normalized_semantic=common.token(r['semantic_name'],unit,True),
                 mask=labels==int(r['group_index'] if group else r['instance_index'])) for r in records]


def best(rows, truth, target=None):
    return max((common.iou(r['mask'],truth) for r in rows if target is None or r['normalized_semantic']==target),default=0.)


def flows(mask, truth, rows):
    output=[];owned=np.zeros(mask.shape,bool)
    for row in rows:
        overlap=mask & row['mask']
        if overlap.any():
            assert not (owned & overlap).any(), 'Overlapping owner masks violate partition'
            owned |= overlap
            output.append(dict(identity=row['identity'],semantic=row['semantic'],normalized_semantic=row['normalized_semantic'],
                               candidate_overlap_pixels=int(overlap.sum()),truth_overlap_pixels=int((overlap & truth).sum()),
                               final_target_iou=common.iou(row['mask'],truth)))
    missing=int((mask & ~owned).sum())
    assert sum(r['candidate_overlap_pixels'] for r in output)+missing==int(mask.sum())
    assert sum(r['truth_overlap_pixels'] for r in output)+int((mask & ~owned & truth).sum())==int((mask & truth).sum())
    return output,missing


def pointer(value,path):
    for item in path.strip('/').split('/'):
        value=value[int(item)] if isinstance(value,list) else value[item.replace('~1','/').replace('~0','~')]
    return value


def json_native(value):
    return json.loads(json.dumps(value,default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist()))


def main():
    final_receipt=common.read(MECH/'receipt.json')
    assert final_receipt['status']=='COMPLETE' and final_receipt['audit_status']=='PASS'
    common.verify_sha(MECH/'receipt.json','80a695971cb0a72dc3155138e6e0c64a114dda5286027cc47a45c38108a8fd12','final_mechanism_receipt')
    common.verify_sha(MECH/'summary.json','6b0eb80b3f9a722b8bf50daa5282e5b3759accaf9813b86e9514ed6b4bc85543','final_mechanism_summary')
    actual_files={str(p.relative_to(MECH)) for p in MECH.rglob('*') if p.is_file() and p.name!='receipt.json'}
    assert actual_files==set(final_receipt['artifact_sha256'])
    for path,digest in final_receipt['input_hashes'].items():common.verify_sha(path,digest,'mechanism_input')
    for name,digest in final_receipt['artifact_sha256'].items():common.verify_sha(MECH/name,digest,'mechanism_artifact')
    manifest=common.read(HERE/'request_manifest_frozen.json')
    units={r['request_id']:r for r in manifest['requests']}
    routes={(r['request_id'],r['method']):r for r in common.csvread(HERE/'routing_run/routing_cases.csv')}
    reported_loss=common.csvread(MECH/'lost_cases.csv')
    reported_stages={r['request_id']:r for r in common.csvread(MECH/'all_case_stage_availability.csv')}
    expected_loss={(rid,t) for rid in units for t in ('025','050')
                   if int(routes[rid,'raw_proposals'][f'semantic_correct_any_match_{t}'])==1
                   and int(routes[rid,'hpid_split_group_ids'][f'semantic_correct_any_match_{t}'])==0}
    observed_loss={(r['request_id'],r['threshold']) for r in reported_loss}
    assert len(observed_loss)==len(reported_loss) and observed_loss==expected_loss
    loss_ids=sorted({rid for rid,t in expected_loss})
    assert len(loss_ids)==30
    assert {p.stem for p in (MECH/'cases').glob('*.json')}==set(loss_ids)
    prep=common.read(HERE/'common_groups/preparation_receipt.json')
    prep_index={r['request_id']:r for r in prep['cases']}
    # All old37 losses plus deterministic one new example for each reported stage.
    replay_ids={rid for rid,t in expected_loss if units[rid]['cohort']=='old37'}
    for stage in sorted({r['first_observed_loss_stage'] for r in reported_loss if r['cohort']=='new216'}):
        replay_ids.add(min(r['request_id'] for r in reported_loss if r['cohort']=='new216' and r['first_observed_loss_stage']==stage))
    checks=[];stage_counts=collections.Counter();bindings={};replays=[];gate_count=0;flow_count=0
    for rid in loss_ids:
        unit=units[rid];package=Path(unit['package_path']);folder=HERE/'common_groups'/rid
        cpath=MECH/'cases'/(rid+'.json'); detail=common.read(cpath)
        receipt=common.read(folder/'replay_receipt.json')
        common.verify_sha(folder/'replay_receipt.json',prep_index[rid]['replay_receipt_sha256'],'case_receipt')
        for name,digest in receipt['input_sha256'].items():common.verify_sha(package/name,digest,'case_input')
        for output in receipt['outputs'].values():common.verify_sha(output['path'],output['sha256'],'common_group_artifact')
        bindings[str(cpath)]=common.sha(cpath)
        truth=common.binary(unit['target_mask_path']);target=common.token(unit['target_part_name'],unit)
        raw=common.read(package/'candidates.json'); accepted=common.read(folder/'accepted_candidates_metadata.json')
        part_map=np.asarray(Image.open(package/'part_id_map.tiff'))
        group_map=np.asarray(Image.open(unit['hpid_group_map_path']))
        parts=native_rows(common.read(package/'parts.json'),part_map,unit)
        groups=native_rows(common.read(unit['hpid_groups_path']),group_map,unit,True)
        diagnostics=common.read(folder/'grouping_diagnostics.json')
        fused_diag=common.read(folder/'fused_diagnostics.json')
        accepted_index=collections.defaultdict(list)
        for a in accepted:accepted_index[fingerprint(a,a['mask_sha256'])].append(a['index'])
        candidates=[];raw_target=[];accepted_target=[];candidate_by_identity={}
        for index,c in enumerate(raw,1):
            mask=common.binary(package/c['mask_path'])
            candidates.append(fusion.MaskCandidate(c['semantic_name'],c['semantic_parent'],mask,float(c['score']),c['source'],
                                                   c.get('prompt',''),float(c.get('source_reliability',1.)),dict(c.get('metadata') or {})))
            sig=fingerprint(c,packed_sha(mask));matches=accepted_index.get(sig,[])
            candidate_by_identity[f'proposal/{index:04d}']=(c,mask,matches)
            if mask.sum()>=6 and common.token(c['semantic_name'],unit,True)==target:
                raw_target.append(common.iou(mask,truth))
                if matches:accepted_target.append(common.iou(mask,truth))
        values=dict(raw_best_matching_iou=max(raw_target,default=0.),accepted_best_matching_iou=max(accepted_target,default=0.),
                    native_part_best_matching_iou=best(parts,truth,target),group_best_matching_iou=best(groups,truth,target),
                    native_part_best_any_semantic_iou=best(parts,truth),group_best_any_semantic_iou=best(groups,truth))
        common.check_fields(reported_stages[rid],values,(rid,'stage_masks'))
        common.check_fields(detail['summary'],values,(rid,'case_stage_masks'))
        stage_labels={}
        for t,threshold in [('025',.25),('050',.5)]:
            flags=[values[k]>=threshold for k in ('raw_best_matching_iou','accepted_best_matching_iou','native_part_best_matching_iou','group_best_matching_iou')]
            actual_loss=flags[0] and not flags[3]
            stage='not_a_raw_to_group_loss'
            if actual_loss:
                if not flags[1]:stage='raw_to_accepted_candidate_filtering'
                elif not flags[2]:stage='accepted_to_native_part_ownership_gate_cleanup_unresolved'
                else:stage='native_part_to_group'
                stage_counts[unit['cohort'],t,stage]+=1
            common.check_fields(reported_stages[rid],{f'raw_available_{t}':int(flags[0]),f'accepted_available_{t}':int(flags[1]),
                f'native_part_available_{t}':int(flags[2]),f'group_available_{t}':int(flags[3]),f'raw_to_group_lost_{t}':int(actual_loss),
                f'first_observed_loss_stage_{t}':stage},(rid,t))
            stage_labels[t]=stage
        expected_correct={identity for identity,(c,m,a) in candidate_by_identity.items()
                          if m.sum()>=6 and common.token(c['semantic_name'],unit,True)==target and common.iou(m,truth)>=.25}
        correct_rows=detail['correct_raw_candidates_at_either_threshold']
        assert {r['raw_identity'] for r in correct_rows}==expected_correct
        for cr in correct_rows:
            c,mask,accepted_matches=candidate_by_identity[cr['raw_identity']]
            common.check_fields(cr,dict(target_iou=common.iou(mask,truth),mask_area=int(mask.sum()),
                                       accepted_by_fuser=bool(accepted_matches)),(rid,cr['raw_identity']))
            if accepted_matches:assert cr['accepted_index'] in accepted_matches
            for field,owners,unowned_field in [('native_part_pixel_destinations',parts,'unowned_part_pixels'),
                                               ('final_group_pixel_destinations',groups,'unowned_group_pixels')]:
                computed,missing=flows(mask,truth,owners)
                observed={r['identity']:r for r in cr[field]}
                assert set(observed)=={r['identity'] for r in computed}
                for row in computed:common.check_fields(observed[row['identity']],row,(rid,cr['raw_identity'],field))
                eq_value=cr[unowned_field]
                # Diagnostic represents only pixels left outside the labeled partition.
                common.eq(eq_value,missing,(rid,cr['raw_identity'],unowned_field))
                flow_count+=1
            for record in cr['grouping_verification_records']:
                assert pointer(diagnostics,record['json_path'])==record['record']
                assert record['record'].get('candidate_key')==c.get('metadata',{}).get('candidate_key')
                gate_count+=1
        for loss in detail['group_stage_losses']:
            for fate in loss['part_to_group_fates']:
                part=next(p for p in parts if p['identity']==fate['part_id'])
                common.eq(fate['part_target_iou'],common.iou(part['mask'],truth),(rid,'part_to_group_iou'))
                computed,missing=flows(part['mask'],truth,groups)
                observed={r['identity']:r for r in fate['destinations']}
                assert set(observed)=={r['identity'] for r in computed}
                for row in computed:common.check_fields(observed[row['identity']],row,(rid,'part_to_group_flow'))
                if rid=='new216__laptop_computer__01' and loss['threshold']=='025':
                    assert len(computed)==1 and computed[0]['candidate_overlap_pixels']==6799
                    assert computed[0]['normalized_semantic']=='bezel' and missing==0
                    group_record=next(g for g in common.read(unit['hpid_groups_path']) if g['group_id']==computed[0]['identity'])
                    assert fate['part_id'] in group_record['member_part_ids'] and len(group_record['member_part_ids'])==4
        if rid in replay_ids:
            # Read-only observers record semantic-map support without changing math.
            capture=[]
            original=fusion._clean_labels
            def observed_cleanup(labels,taxonomy,*args,**kwargs):
                def observe(stage,value):
                    candidates_for_stage=[dict(normalized_semantic=common.token(name,unit,True),mask=value==i)
                                          for i,name in enumerate(taxonomy.fine_names) if i]
                    capture.append(dict(stage=stage,best_target_semantic_union_iou=best(candidates_for_stage,truth,target)))
                observe('before_semantic_cleanup',labels)
                answer=original(labels,taxonomy,*args,**kwargs)
                observe('after_semantic_cleanup',answer)
                return answer
            fusion._clean_labels=observed_cleanup
            try:
                fused=fusion.fuse_candidates(candidates,image_shape=part_map.shape,config=fusion.FusionConfig(**receipt['production_config_complete']))
            finally:fusion._clean_labels=original
            assert np.array_equal(fused.instance_map,part_map)
            assert json_native(fused.diagnostics)==fused_diag
            with Image.open(package/'source.png') as im:
                result=build_physical_groups(fused.instance_map,fused.instances,candidates=fused.accepted_candidates,
                                             image=im.convert('RGB'),provisional_scene_labels=False)
            assert np.array_equal(result.group_map,group_map)
            assert json_native([g.to_dict() for g in result.groups])==common.read(unit['hpid_groups_path'])
            assert json_native(result.diagnostics)==diagnostics
            replay=dict(request_id=rid,part_map_exact=True,group_map_exact=True,fused_and_group_diagnostics_exact=True,
                        semantic_union_observations=capture,instance_stage_values=values)
            replays.append(replay)
        checks.append(dict(request_id=rid,cohort=unit['cohort'],candidate_correct_count=len(correct_rows),
                           stage_values=values,stage_labels=stage_labels,gate_records_verified=sum(len(r['grouping_verification_records']) for r in correct_rows)))
    summary=common.read(MECH/'summary.json')
    for row in summary['first_observed_loss_stage_counts']:
        if row['cohort']=='combined':n=sum(v for (c,t,s),v in stage_counts.items() if t==row['threshold'] and s==row['stage'])
        else:n=stage_counts[row['cohort'],row['threshold'],row['stage']]
        common.eq(row['count'],n,('stage_summary',row))
    primary=[r for r in checks if (r['request_id'],'025') in expected_loss]
    secondary=[r for r in checks if (r['request_id'],'050') in expected_loss]
    accepted_to_part=[r for r in primary if r['stage_labels']['025']=='accepted_to_native_part_ownership_gate_cleanup_unresolved']
    narrative_counts=dict(primary_accepted_to_part=len(accepted_to_part),
                          primary_accepted_to_part_with_other_part_geometry=sum(r['stage_values']['native_part_best_any_semantic_iou']>=.25 for r in accepted_to_part),
                          primary_final_group_with_other_geometry=sum(r['stage_values']['group_best_any_semantic_iou']>=.25 for r in primary),
                          secondary_final_group_with_other_geometry=sum(r['stage_values']['group_best_any_semantic_iou']>=.50 for r in secondary))
    assert narrative_counts==dict(primary_accepted_to_part=24,primary_accepted_to_part_with_other_part_geometry=17,
                                   primary_final_group_with_other_geometry=15,secondary_final_group_with_other_geometry=7)
    output=dict(status='PASS',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                loss_cases_at_025=sum(t=='025' for rid,t in expected_loss),loss_cases_at_050=sum(t=='050' for rid,t in expected_loss),
                unique_loss_cases_checked=len(loss_ids),complete_loss_coverage=True,old37_all4_checked=True,
                candidate_pixel_flow_partitions_checked=flow_count,verification_records_compared_to_original=gate_count,
                mechanism_artifacts_verified=len(final_receipt['artifact_sha256']),reader_facing_mechanism_counts_verified=narrative_counts,
                single_primary_group_loss_membership_verified=True,
                all_loss_cases=checks,read_only_exact_replays=replays,
                first_observed_loss_stage_counts=summary['first_observed_loss_stage_counts'],
                interpretation=['Every reported loss is recovered directly from sealed routing and every loss-case stage boundary is confirmed from candidate, accepted metadata, native Part and final Group masks.',
                                'At IoU .25,24/25 losses first appear between accepted masks and native Part instances;1/25 appears between native Part and Group. None is first observed as candidate-list rejection.',
                                'This locates an interface boundary, not a causal contribution of a specific gate, ownership rule or cleanup step. No isolated ablation was run.',
                                'The4 old losses already fall below .25 in native Part outputs; scissors later Group rejection is not evidence it caused the earlier loss.',
                                'Verification records are read from the original grouped diagnostics at their JSON paths; pixel flows are independently recomputed, not accepted from report labels.'],
                bindings={**bindings,**{str(p):common.sha(p) for p in [Path(__file__),MECH/'receipt.json',MECH/'summary.json',MECH/'findings.md',MECH/'lost_cases.csv',MECH/'all_case_stage_availability.csv',
                                                                            HERE/'request_manifest_frozen.json',HERE/'routing_run/receipt.json',HERE/'common_groups/preparation_receipt.json']}})
    (HERE/'mechanism_independent_audit.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(dict(status='PASS',loss025=output['loss_cases_at_025'],loss050=output['loss_cases_at_050'],
                          loss_cases=len(loss_ids),read_only_replay_ids=sorted(replay_ids),flows=flow_count,gate_records=gate_count)),flush=True)


if __name__=='__main__':main()
