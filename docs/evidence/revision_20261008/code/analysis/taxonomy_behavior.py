"""Executable boundary fixtures, not accuracy samples or an OOD benchmark."""
from __future__ import annotations
from portable_paths import BUNDLE_ROOT, DATA_ROOT, SOURCE_ROOT, PROJECT_ROOT, OUTPUT_DIR, resolve_data
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import sys
from analyze_revision import SNAPSHOT, OUT, write_json
import numpy as np
from hpid_split.fusion import MaskCandidate, FusionConfig, fuse_candidates, taxonomy_from_candidates
from hpid_split.taxonomy import Taxonomy
from hpid_split.prompt_bank import PromptBank, DomainPrompt, PartPrompt
from hpid_split.physical_groups import _candidate_three_stage_verification

rows=[]
def check(name, operation, predicate, purpose):
    try:
        observed=operation()
        outcome={'returned':observed}
    except Exception as exc:
        outcome={'exception':type(exc).__name__, 'message':str(exc)}
    passed=bool(predicate(outcome))
    rows.append({'fixture':name,'purpose':purpose,'assertion_passed':passed,**outcome})
    if not passed:
        raise AssertionError(rows[-1])

def raised(out): return out.get('exception')=='ValueError'
def select(domain, label, **kw):
    parts, profile, info=domain.select_parts(label, **kw)
    return {'selected_profile':profile,'parts':[p.semantic_name for p in parts], 'reason':info['selection_reason']}

bank=PromptBank.from_json(SNAPSHOT/'configs/general_asset_prompts.json')
for domain in bank.domains:
    if domain.part_profiles:
        check('unknown_category_'+domain.name,lambda d=domain:select(d,'zz_unlisted_organism_314159'),
              lambda o:o.get('returned',{}).get('selected_profile') is None,
              'Unknown label uses configured fallback inventory; no novel hierarchy inference.')
        check('invalid_profile_hint_'+domain.name,lambda d=domain:select(d,'zz_unlisted_organism_314159',profile_hint='zz_missing_profile'),raised,
              'An explicitly nonexistent profile hint raises instead of silently selecting a profile.')
    check('unlisted_part_label_'+domain.name,lambda d=domain:d.match_part('zz_unknown_part_314159'),
          lambda o:'returned' in o and o['returned'] is None,
          'String matching does not invent a semantic name for a completely unmatched token.')

base_part=PartPrompt(semantic_name='fixture_part',prompts=('fixture part',))
def fixture_bank(part):
    PromptBank((DomainPrompt(name='fixture_root',root_prompts=('fixture root',),parts=(part,)),))
    return 'constructed'
for field in ['semantic_parent','query_parent','fallback_query_parent','assembly_parent']:
    check('unknown_'+field,lambda f=field:fixture_bank(replace(base_part,**{f:'missing_parent'})),raised,
          'Prompt-bank construction validates referenced parent membership.')
check('semantic_parent_cycle',lambda:fixture_bank(replace(base_part,semantic_parent='fixture_part')),raised,
      'Prompt-bank construction rejects semantic-parent cycles.')
check('invalid_taxonomy_parent_index',lambda:Taxonomy(('background','part'),('background','root'),(0,2),('part',)).to_dict(),raised,
      'Explicit taxonomy rejects an out-of-range numerical parent index.')

mask=np.zeros((32,32),bool);mask[8:24,8:24]=True
candidate=MaskCandidate('zz_unlisted_part','zz_unlisted_parent',mask,0.9,'grounded-sam-fixture')
def fused_summary(candidates,taxonomy=None):
    result=fuse_candidates(candidates, taxonomy=taxonomy, config=FusionConfig(minimum_area_px=1))
    return {'fine_names':list(result.taxonomy.fine_names),'parents':list(result.taxonomy.parent_names),
            'output_names':[x.semantic_name for x in result.instances],
            'output_pixels':int(np.count_nonzero(result.instance_map))}
check('dynamic_taxonomy_arbitrary_nonempty_labels',lambda:fused_summary([candidate]),
      lambda o:'zz_unlisted_part' in o.get('returned',{}).get('fine_names',[]) and o['returned']['output_pixels']>0,
      'Direct low-level fusion constructs a taxonomy from supplied labels; it does not validate an external inventory.')
check('dynamic_taxonomy_conflicting_parents',lambda:taxonomy_from_candidates([candidate,replace(candidate,semantic_parent='other_parent')]).to_dict(),raised,
      'The same semantic name cannot be assigned two different parents.')
explicit=Taxonomy(('background','fixture_part','fixture_root'),('background','fixture_root'),(0,1,1),('fixture_part',))
check('explicit_taxonomy_unknown_class',lambda:fused_summary([candidate],explicit),raised,
      'Explicit taxonomy excludes an unknown candidate class.')
check('empty_semantic_name',lambda:replace(candidate,semantic_name=''),raised,
      'Candidate constructor rejects an empty semantic name.')
check('gate_direct_source_unlisted_name',lambda:_candidate_three_stage_verification(candidate),
      lambda o:o.get('returned',{}).get('accepted') is True,
      'The named Group gate can trust direct-source provenance without vocabulary membership validation; this is not OOD detection.')
check('gate_nonempty_unverified_profile_string',lambda:_candidate_three_stage_verification(replace(candidate,source='fixture',metadata={'selected_part_profile':'zz_missing_profile','geometric_support':0.8})),
      lambda o:o.get('returned',{}).get('accepted') is True,
      'This low-level Group gate tests a nonempty profile provenance field, not profile membership in PromptBank.')
check('gate_no_semantic_provenance',lambda:_candidate_three_stage_verification(replace(candidate,source='fixture')),
      lambda o:o.get('returned',{}).get('accepted') is False,
      'Without direct-source or inventory provenance, this candidate fails the semantic stage.')

files=['fusion.py','taxonomy.py','prompt_bank.py','physical_groups.py']
write_json(OUT/'taxonomy_behavior.json',{'protocol':'Deterministic synthetic API boundary fixtures; these are not PACO cases or empirical OOD accuracy samples.',
    'fixture_count':len(rows),'all_assertions_passed':all(x['assertion_passed'] for x in rows),
    'frozen_sources':{f:hashlib.sha256((SNAPSHOT/'src/hpid_split'/f).read_bytes()).hexdigest() for f in files},
    'config_sha256':hashlib.sha256((SNAPSHOT/'configs/general_asset_prompts.json').read_bytes()).hexdigest(),'fixtures':rows})
print(json.dumps({'fixture_count':len(rows),'all_assertions_passed':True},indent=2))
