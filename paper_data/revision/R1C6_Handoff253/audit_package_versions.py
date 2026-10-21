"""Read archived provenance; no predictions or routing outcomes computed."""
import collections
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def distribution(values):
    counts = collections.Counter(canonical(v) for v in values)
    return [dict(value=json.loads(v), count=n) for v, n in counts.most_common()]


def main():
    inventory = read(HERE / 'data_inventory.json')
    old42 = read(inventory['sources']['original42_reference_manifest'])['cases']
    source_rows = [('original42', x['case_id'], Path(x['historical_package'])) for x in old42]
    source_rows += [('test226', x['case_id'], Path(x['package_path'])) for x in inventory['all_cases']]
    rows = []
    for cohort, case_id, path in source_rows:
        manifest_path = path / 'package_manifest.json'
        diag_path = path / 'inference_diagnostics.json'
        m, d = read(manifest_path), read(diag_path)
        algorithm = m['algorithm']
        gen = d['candidate_generation']
        models = gen.get('models', {})
        # Remove image-derived outputs only; preserve all scalar/static model settings.
        static_models = {k: v for k, v in models.items() if k not in {'automatic_asset_queries'}}
        candidate_rows = read(path / 'candidates.json')
        missing = [c['mask_path'] for c in candidate_rows if not (path / c['mask_path']).is_file()]
        rows.append(dict(cohort=cohort, case_id=case_id, package_path=str(path),
                         manifest_sha256=sha(manifest_path), diagnostics_sha256=sha(diag_path),
                         format_version=m.get('format_version'), algorithm_version=algorithm.get('version'),
                         prompt_bank_sha256=algorithm.get('prompt_bank_sha256'),
                         diagnostic_prompt_bank_sha256=d.get('prompt_bank', {}).get('sha256'),
                         fusion_config=algorithm.get('fusion_config'),
                         candidate_models=algorithm.get('candidate_models'),
                         candidate_model_configuration=static_models,
                         root_mode=d.get('root_routing', {}).get('mode'),
                         router_index_sha256=d.get('global_asset_proposal', {}).get('index_manifest_sha256'),
                         explicit_asset_prompt=models.get('asset_prompt'),
                         explicit_asset_prompt_domain=models.get('asset_prompt_domain'),
                         explicit_asset_prompt_profile=models.get('asset_prompt_profile'),
                         manifest_gt_used=m.get('inference_uses_ground_truth'),
                         algorithm_gt_used=algorithm.get('ground_truth_used'),
                         candidate_gt_used=gen.get('ground_truth_used'),
                         router_gt_used=d.get('global_asset_proposal', {}).get('ground_truth_used'),
                         router_target_mask_used=d.get('global_asset_proposal', {}).get('target_mask_used'),
                         adaptive_route=gen.get('proposal_first_execution_route'),
                         candidate_count=len(candidate_rows), missing_candidate_masks=missing))
    fields = ['format_version', 'algorithm_version', 'prompt_bank_sha256', 'fusion_config', 'candidate_models',
              'root_mode', 'router_index_sha256', 'explicit_asset_prompt', 'explicit_asset_prompt_domain',
              'explicit_asset_prompt_profile', 'manifest_gt_used', 'algorithm_gt_used', 'candidate_gt_used',
              'router_gt_used', 'router_target_mask_used']
    summary = {}
    for cohort in ['original42', 'test226']:
        subset = [r for r in rows if r['cohort'] == cohort]
        summary[cohort] = {f: distribution(r[f] for r in subset) for f in fields}
        summary[cohort]['case_count'] = len(subset)
        summary[cohort]['packages_with_missing_candidate_masks'] = sum(bool(r['missing_candidate_masks']) for r in subset)
        summary[cohort]['adaptive_route_present_count'] = sum(r['adaptive_route'] is not None for r in subset)
        all_keys = sorted(set().union(*(r['candidate_model_configuration'] for r in subset)))
        summary[cohort]['candidate_model_field_distributions'] = {
            k: distribution(r['candidate_model_configuration'].get(k, '__ABSENT__') for r in subset) for k in all_keys}
    all_model_keys = sorted(set(summary['original42']['candidate_model_field_distributions']) |
                            set(summary['test226']['candidate_model_field_distributions']))
    config_differences = {}
    for k in all_model_keys:
        values = {cohort: {canonical(r['candidate_model_configuration'].get(k, '__ABSENT__')) for r in rows
                          if r['cohort'] == cohort} for cohort in summary}
        if values['original42'] != values['test226']:
            config_differences[k] = {cohort: [json.loads(v) for v in sorted(vals)] for cohort, vals in values.items()}
    reproduction_path = HERE.parent / 'reproduction_audit.json'
    payload = dict(status='READ_ONLY_ARCHIVED_PROVENANCE_AUDIT', summary=summary,
                   candidate_configuration_fields_with_different_observed_values=config_differences,
                   inference_condition='All archived packages record no GT, blank asset prompt, automatic primary root routing; original42 and test226 have different historical prompt banks and candidate-generation configuration metadata.',
                   pairing='Five methods within a case must use the exact same cached accepted candidates. No upstream candidate regeneration is required to make that within-case comparison.',
                   pooling='Keep old37/new216 separate as primary strata. Any pooled result is a labeled archival mixture, not evidence from one identical upstream pipeline.',
                   known_part_reproduction={'path': str(reproduction_path), 'sha256': sha(reproduction_path),
                                            'case_count': read(reproduction_path)['case_count'],
                                            'all_hpid_maps_exact': read(reproduction_path)['all_hpid_maps_exact'],
                                            'limit': 'Given cached candidate masks; does not establish identical candidate generation.'},
                   group_reproduction='Assigned to independent full_image_doc_audit agent; not run by this script.',
                   rows=rows, script_sha256=sha(__file__))
    out = HERE / 'package_version_audit.json'
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(path=str(out), sha256=sha(out), differing_config_fields=list(config_differences),
                          summary={cohort: {k: v for k, v in s.items() if k != 'candidate_model_field_distributions' and k != 'fusion_config'}
                                   for cohort, s in summary.items()}), ensure_ascii=False))


if __name__ == '__main__':
    main()
