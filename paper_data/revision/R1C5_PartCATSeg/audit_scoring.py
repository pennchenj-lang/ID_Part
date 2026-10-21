"""Independent CPU mask/metric audit; never imports the scorer or model.

OpenCV CCL replaces scipy.ndimage; explicit bounded-mask intersections replace
bincount IoUs; a standalone rectangular Hungarian implementation replaces SciPy.
Frozen semantic alias tables are read as AST literals, not imported as code.
Only this audit's JSON report is written. No thresholds, cohorts or outputs change.
"""
from __future__ import annotations
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import re
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
SNAPSHOT = Path('__RUNTIME_ROOT__/code_snapshots/hpid_split_be54300_holdout')
SEMANTICS = SNAPSHOT / 'src/hpid_split/paco_semantics.py'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def mask(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('L')) >= 128


def assignment(weight):
    """Minimum-cost augmenting paths; rows <= columns; float64 arithmetic."""
    if not weight.size:
        return []
    flipped = weight.shape[0] > weight.shape[1]
    cost = 1.0 - (weight.T if flipped else weight)
    n, m = cost.shape
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p, way = np.zeros(m + 1, dtype=int), np.zeros(m + 1, dtype=int)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        best = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = np.inf, 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                current = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if current < best[j]:
                    best[j], way[j] = current, j0
                if best[j] < delta:
                    delta, j1 = best[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    best[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    pairs = [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j]]
    return [(b, a) for a, b in pairs] if flipped else pairs


def geometry(matrix):
    pairs = assignment(matrix)
    t, p = matrix.shape
    out = {'truth_part_count': t, 'predicted_part_count': p,
           'assignment_iou_sum': float(sum(matrix[a, b] for a, b in pairs))}
    for threshold, suffix in [(0.25, '025'), (0.5, '050'), (0.75, '075')]:
        hits = int(sum(matrix[a, b] >= threshold for a, b in pairs))
        out['matched_count_at_' + suffix] = int(hits)
        out['part_f1_at_' + suffix] = 2 * hits / (t + p) if t + p else 0.0
        out['part_precision_at_' + suffix] = hits / p if p else 0.0
        out['part_recall_at_' + suffix] = hits / t if t else 0.0
    return out


def iou(a, b):
    intersection = int(np.count_nonzero(a & b))
    union = int(np.count_nonzero(a)) + int(np.count_nonzero(b)) - intersection
    return intersection / union if union else 0.0


def alias_tables():
    values = {}
    for node in ast.parse(SEMANTICS.read_text()).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in ('_DOMAIN_ALIASES', '_CATEGORY_ALIASES'):
                values[node.target.id] = ast.literal_eval(node.value)
    return values['_DOMAIN_ALIASES'], values['_CATEGORY_ALIASES']


DOMAIN_ALIAS, CATEGORY_ALIAS = alias_tables()


def norm(name):
    return re.sub('[^a-z0-9]+', '_', name.lower()).strip('_')


def canonical(name, domain, category=''):
    token = norm(name)
    return CATEGORY_ALIAS.get(norm(category), {}).get(token, DOMAIN_ALIAS.get(domain, {}).get(token, token))


def predicted_token(full_name, expected_domain):
    if full_name == expected_domain:
        return 'body'
    prefix = expected_domain + '_'
    return canonical(full_name[len(prefix):] if full_name.startswith(prefix) else full_name, expected_domain)


def union_by_token(masks, tokens):
    out = {}
    for m, t in zip(masks, tokens):
        out[t] = m.copy() if t not in out else (out[t] | m)
    return out


def semantic(truth, prediction):
    vals = [iou(m, prediction[t]) if t in prediction else 0.0 for t, m in truth.items()]
    return {'mean_semantic_union_iou': sum(vals) / len(vals),
            'semantic_union_recall_at_025': sum(v >= 0.25 for v in vals) / len(vals),
            'semantic_label_precision': len(set(truth) & set(prediction)) / max(1, len(prediction))}


def component_masks(labels):
    comps = []
    for label in np.unique(labels):
        if label == 456:
            continue
        count, components, stats, _ = cv2.connectedComponentsWithStats(
            np.ascontiguousarray(labels == label, dtype=np.uint8), connectivity=8, ltype=cv2.CV_32S)
        for index in range(1, count):
            x, y, w, h, area = (int(v) for v in stats[index])
            cropped = components[y:y+h, x:x+w] == index
            assert int(cropped.sum()) == area
            comps.append((int(label), x, y, cropped, area))
    return comps


def descriptive(values):
    a = np.asarray(values)
    return {'min': float(a.min()), 'q25': float(np.quantile(a, .25)),
            'median': float(np.median(a)), 'mean': float(a.mean()),
            'q75': float(np.quantile(a, .75)), 'max': float(a.max())}


def bootstrap(rows, metrics):
    # Reproduce the protocol seed, but apply draws by multiplicity weights instead
    # of the scorer's indexed tensor means; percentile interpolation explicit.
    rng = np.random.default_rng(20261009)
    samples = rng.integers(0, len(rows), size=(10000, len(rows)))
    weights = np.array([np.bincount(draw, minlength=len(rows)) for draw in samples])
    out = {}
    for key in metrics:
        a = np.array([r['hpid'][key] for r in rows])
        b = np.array([r['partcatseg'][key] for r in rows])
        differences = a - b
        distribution = np.sort(weights @ differences / len(rows))
        bounds = []
        for q in (.025, .975):
            at = (len(distribution) - 1) * q
            lower = int(np.floor(at))
            bounds.append(float(distribution[lower] + (distribution[int(np.ceil(at))] - distribution[lower]) * (at - lower)))
        out[key] = {'hpid_mean': float(a.mean()), 'partcatseg_mean': float(b.mean()),
                    'paired_hpid_minus_partcatseg': float(differences.mean()),
                    'paired_percentile_95ci': bounds, 'n': len(rows)}
    return out


def main():
    sanity_rng = np.random.default_rng(150)
    for n,m in [(2,3),(3,2),(3,4),(4,3),(3,3)]:
        for _ in range(5):
            matrix = sanity_rng.random((n,m))
            observed = sum(matrix[a,b] for a,b in assignment(matrix))
            weights = matrix if n<=m else matrix.T
            expected = max(sum(weights[i,j] for i,j in enumerate(order))
                           for order in itertools.permutations(range(weights.shape[1]),weights.shape[0]))
            assert abs(observed-expected)<1e-12
    refs_path = ROOT.parent / 'full_image_review4/sealed_reference_manifest.json'
    summary_path = ROOT / 'scoring/summary.json'
    recorded = {r['anonymous_id']: r for r in read(ROOT / 'scoring/cases.json')}
    prediction_path = ROOT / 'cohort_predictions/predictions_manifest.json'
    pred_manifest = read(prediction_path)
    preds = {r['anonymous_id']: r for r in pred_manifest['cases']}
    refs = read(refs_path)['cases']
    registry = read(ROOT / 'label_registry.json')['labels']
    domains = {r['category']: r['expected_domain'] for r in read(SNAPSHOT / 'configs/paco_broad_categories.json')['categories']}
    mappings, predicted_categories, predicted_domains = [], [], []
    for row in registry:
        cat, part = row['paco_name'].split(':', 1)
        domain = domains.get(cat)
        mappings.append(domain + '_' + canonical(part, domain, cat) if domain else 'unmapped_paco_' + str(row['paco_category_id']))
        predicted_categories.append(cat)
        predicted_domains.append(domain)
    assert len(refs) == len(recorded) == len(preds) == 42
    assert len({r['image_id'] for r in refs}) == 42
    assert set(recorded) == set(preds) == {r['anonymous_id'] for r in refs}
    protocol = read(ROOT / 'protocol_frozen.json')
    source_summary = read(summary_path)
    hashes = {str(path): sha(path) for path in [refs_path, summary_path, prediction_path,
              ROOT/'score_partcatseg.py', ROOT/'smoke_inference.py', ROOT/'label_registry.json',
              ROOT/'protocol_frozen.json', SEMANTICS]}
    assert sha(ROOT/'score_partcatseg.py') == protocol['hash_bindings']['scorer_sha256']
    assert sha(ROOT/'smoke_inference.py') == protocol['hash_bindings']['adapter_sha256']
    assert sha(ROOT/'label_registry.json') == protocol['hash_bindings']['label_registry_sha256']
    assert sha(prediction_path) == read(ROOT/'cohort_predictions/receipt.json')['predictions_manifest_sha256']
    records, discrepancies, all_areas, background_fractions, raw_counts = [], [], [], [], []
    class_pixels, component_counts = Counter(), Counter()
    excluded_roots = []
    geometry_keys = ['part_f1_at_025','part_f1_at_050','part_f1_at_075','part_precision_at_025','part_recall_at_025',
                     'predicted_part_count','truth_part_count','matched_count_at_025','matched_count_at_050','matched_count_at_075']
    semantic_keys = ['mean_semantic_union_iou','semantic_union_recall_at_025','semantic_label_precision']
    for r in refs:
        case_id = r['anonymous_id']
        case_path = Path(r['case_path'])
        assert sha(case_path) == r['case_sha256']
        gt = read(case_path)
        truth_masks = [mask(case_path.parent/p['mask_crop']) for p in gt['parts']]
        truth_tokens = [canonical(p['part_name'],r['expected_domain'],gt['object_category']) for p in gt['parts']]
        truth_union = union_by_token(truth_masks,truth_tokens)
        true_areas = [int(m.sum()) for m in truth_masks]
        hpid_root = Path(r['historical_package'])
        all_hpid = read(hpid_root/'parts.json')
        keep = [p for p in all_hpid if p['semantic_name'] != r['expected_domain'] or 'body' in truth_tokens]
        removed = [p for p in all_hpid if p not in keep]
        hp_masks = [mask(hpid_root/p['mask_visible_path']) for p in keep]
        h_matrix = np.array([[iou(t,p) for p in hp_masks] for t in truth_masks], dtype=np.float64)
        if not hp_masks:
            h_matrix = np.zeros((len(truth_masks),0))
        hp_tokens = [predicted_token(p['semantic_name'],r['expected_domain']) for p in keep]
        hpid = {**geometry(h_matrix), **semantic(truth_union,union_by_token(hp_masks,hp_tokens))}
        prediction = preds[case_id]
        assert prediction['status'] == 'COMPLETE'
        path = Path(prediction['label_path'])
        assert sha(path) == prediction['label_sha256']
        labels = np.load(path,allow_pickle=False)
        assert labels.shape == truth_masks[0].shape and labels.dtype == np.uint16 and labels.min() >= 0 and labels.max() <= 456
        comps = component_masks(labels)
        p_matrix = np.zeros((len(truth_masks),len(comps)),dtype=np.float64)
        for pi,(_,x,y,pred,area) in enumerate(comps):
            h,w = pred.shape
            for ti,true in enumerate(truth_masks):
                inter = int(np.count_nonzero(true[y:y+h,x:x+w] & pred))
                p_matrix[ti,pi] = inter/(true_areas[ti]+area-inter)
        present, counts = np.unique(labels,return_counts=True)
        p_masks = [labels == val for val in present if val != 456]
        p_tokens = [predicted_token(mappings[int(val)],r['expected_domain']) for val in present if val != 456]
        pc = {**geometry(p_matrix), **semantic(truth_union,union_by_token(p_masks,p_tokens))}
        for method, result in [('hpid',hpid),('partcatseg',pc)]:
            for key in geometry_keys+semantic_keys:
                delta = abs(float(result[key])-float(recorded[case_id][method][key]))
                if delta > 1e-12:
                    discrepancies.append({'case':case_id,'method':method,'metric':key,'absolute_error':delta})
        areas = [c[-1] for c in comps]
        all_areas.extend(areas)
        background = int(np.count_nonzero(labels==456))
        background_fractions.append(background/labels.size)
        raw_counts.append(int(len(present)-(456 in present)))
        mapped_registry_tokens = {predicted_token(name,r['expected_domain']) for name in mappings}
        missing_ref_tokens = sorted(set(truth_union)-mapped_registry_tokens)
        annotation_union = np.logical_or.reduce(truth_masks)
        annotated_foreground = int(np.count_nonzero(annotation_union & (labels!=456)))
        annotated_background = int(np.count_nonzero(annotation_union & (labels==456)))
        wrong_cat = wrong_domain = unmapped = same_domain_wrong_cat = 0
        wrong_cat_annotated = wrong_domain_annotated = 0
        for index,pixels in zip(present,counts):
            index,pixels = int(index),int(pixels)
            if index == 456:
                continue
            class_pixels[index] += pixels
            if predicted_domains[index] is None:
                unmapped += pixels
            if predicted_categories[index] != gt['object_category']:
                wrong_cat += pixels
                wrong_cat_annotated += int(np.count_nonzero(annotation_union & (labels==index)))
                if predicted_domains[index] == r['expected_domain']:
                    same_domain_wrong_cat += pixels
            if predicted_domains[index] != r['expected_domain']:
                wrong_domain += pixels
                wrong_domain_annotated += int(np.count_nonzero(annotation_union & (labels==index)))
        for c in comps:
            component_counts[c[0]] += 1
        if removed:
            excluded_roots.append({'anonymous_id':case_id,'count':len(removed),
                                   'pixels':sum(int(mask(hpid_root/p['mask_visible_path']).sum()) for p in removed)})
        record = {'anonymous_id':case_id,'image_id':r['image_id'],'object_category':gt['object_category'],
                  'hpid':hpid,'partcatseg':pc,'diagnostics':{
                  'pixels':labels.size,'background_pixels':background,'background_fraction':background/labels.size,
                  'raw_classes':raw_counts[-1],'components':len(comps),'component_area_min':min(areas) if areas else None,
                  'component_area_median':float(np.median(areas)) if areas else None,
                  'components_le4':sum(a<=4 for a in areas),'components_le16':sum(a<=16 for a in areas),
                  'components_le64':sum(a<=64 for a in areas),'components_le256':sum(a<=256 for a in areas),
                  'wrong_object_pixels':wrong_cat,'wrong_domain_pixels':wrong_domain,'unmapped_pixels':unmapped,
                  'annotated_foreground_pixels':annotated_foreground,'annotated_background_pixels':annotated_background,
                  'wrong_object_annotated_foreground_pixels':wrong_cat_annotated,
                  'wrong_domain_annotated_foreground_pixels':wrong_domain_annotated,
                  'same_domain_wrong_object_pixels':same_domain_wrong_cat,
                  'uncovered_reference_semantic_tokens':missing_ref_tokens,
                  'reference_semantic_classes':len(truth_union),'hpid_root_parts_excluded':len(removed)}}
        records.append(record)
        print(case_id,'HPID',round(hpid['part_f1_at_025'],6),'PC',round(pc['part_f1_at_025'],6),'components',len(comps),flush=True)
    metrics = [key for key in source_summary['metrics']]
    calculated = bootstrap(records,metrics)
    summary_errors = {}
    for key in metrics:
        errors = {field:float(np.max(np.abs(np.asarray(value)-np.asarray(source_summary['metrics'][key][field]))))
                  for field,value in calculated[key].items()}
        summary_errors[key] = errors
    total_pixels = sum(r['diagnostics']['pixels'] for r in records)
    total_bg = sum(r['diagnostics']['background_pixels'] for r in records)
    foreground = total_pixels-total_bg
    area_array = np.array(all_areas)
    assert int(area_array.sum()) == foreground
    annotated_foreground = sum(r['diagnostics']['annotated_foreground_pixels'] for r in records)
    annotated_background = sum(r['diagnostics']['annotated_background_pixels'] for r in records)
    diagnostics = {'total_pixels':total_pixels,'foreground_pixels':foreground,'background_pixels':total_bg,
      'pooled_background_fraction':total_bg/total_pixels,'per_image_background_fraction':descriptive(background_fractions),
      'components_total':len(all_areas),'per_image_components':descriptive([r['partcatseg']['predicted_part_count'] for r in records]),
      'hpid_per_image_parts':descriptive([r['hpid']['predicted_part_count'] for r in records]),
      'per_image_truth_parts':descriptive([r['hpid']['truth_part_count'] for r in records]),
      'per_image_raw_semantic_classes':descriptive(raw_counts),'observed_registry_class_count':len(class_pixels),
      'observed_predicted_object_categories':len({predicted_categories[k] for k in class_pixels}),
      'component_area_pixels':descriptive(all_areas),
      'small_component_counts':{str(t):int(np.sum(area_array<=t)) for t in [1,4,16,64,256]},
      'small_component_pixel_share':{str(t):float(area_array[area_array<=t].sum()/foreground) for t in [1,4,16,64,256]},
      'wrong_object_foreground_fraction':sum(r['diagnostics']['wrong_object_pixels'] for r in records)/foreground,
      'wrong_domain_foreground_fraction':sum(r['diagnostics']['wrong_domain_pixels'] for r in records)/foreground,
      'same_domain_wrong_object_foreground_fraction':sum(r['diagnostics']['same_domain_wrong_object_pixels'] for r in records)/foreground,
      'unmapped_prediction_pixels':sum(r['diagnostics']['unmapped_pixels'] for r in records),
      'registry_unmapped_class_count':sum(domain is None for domain in predicted_domains),
      'unmapped_present_class_count':sum(predicted_domains[k] is None for k in class_pixels),
      'unmapped_predicted_components':sum(component_counts[k] for k in component_counts if predicted_domains[k] is None),
      'annotated_region_diagnostics':{'foreground_pixels':annotated_foreground,'background_pixels':annotated_background,
        'background_fraction':annotated_background/(annotated_background+annotated_foreground),
        'wrong_object_foreground_fraction':sum(r['diagnostics']['wrong_object_annotated_foreground_pixels'] for r in records)/annotated_foreground,
        'wrong_domain_foreground_fraction':sum(r['diagnostics']['wrong_domain_annotated_foreground_pixels'] for r in records)/annotated_foreground,
        'note':'Post-inference descriptive diagnostics only; these masks never clip or repair any prediction or score.'},
      'uncovered_reference_semantic_tokens':[{'anonymous_id':r['anonymous_id'],'tokens':r['diagnostics']['uncovered_reference_semantic_tokens']}
                                           for r in records if r['diagnostics']['uncovered_reference_semantic_tokens']],
      'top20_prediction_classes_by_pixels':[{'label_index':k,'paco_name':registry[k]['paco_name'],'pixels':v,
                                            'components':component_counts[k]} for k,v in class_pixels.most_common(20)],
      'hpid_root_exclusion':{'cases':len(excluded_roots),'excluded_parts':sum(r['count'] for r in excluded_roots),'details':excluded_roots},
      'primary_zero_cases':{method:sum(r[method]['part_f1_at_025']==0 for r in records) for method in ['hpid','partcatseg']},
      'primary_pairwise':{'hpid_higher':sum(r['hpid']['part_f1_at_025']>r['partcatseg']['part_f1_at_025'] for r in records),
                          'equal':sum(r['hpid']['part_f1_at_025']==r['partcatseg']['part_f1_at_025'] for r in records),
                          'partcatseg_higher':sum(r['hpid']['part_f1_at_025']<r['partcatseg']['part_f1_at_025'] for r in records)}}
    assert all(sha(path)==value for path,value in hashes.items()),'Frozen inputs changed during read-only audit'
    result = {'status':'PASS' if not discrepancies and max(max(v.values()) for v in summary_errors.values())<1e-12 else 'FAIL',
      'created_utc':datetime.now(timezone.utc).isoformat(),'independent_cases':42,'metrics':calculated,
      'case_discrepancies':discrepancies,'summary_absolute_errors':summary_errors,'diagnostics':diagnostics,
      'independence':{'component_engine':'OpenCV connectedComponentsWithStats, 8-connectivity',
        'IoU':'Explicit cropped binary component intersections and complete GT/component areas, float64',
        'matching':'Independent augmenting-path Hungarian implementation; no scipy import',
        'bootstrap':'Same frozen RNG draws converted to multiplicity weights; explicit sorted linear percentile interpolation',
        'semantic':'Alias dictionaries parsed as AST literals; canonicalization, unions and scoring reimplemented',
        'scorer_imported':False,'model_imported':False,'GPU_used':False,
        'Hungarian_sanity':'25 random small rectangular matrices checked against exhaustive permutation optimum'},
      'input_sha256':hashes,'audit_script_sha256':sha(__file__),'cases':records,
      'scope':'Retrospective VOC-checkpoint to full PACO456 transfer on 42 shared localized crops with an all-component CCL adapter. Not an official PartCATSeg benchmark or universal algorithm ranking.',
      'interpretation_limits':[
        'The all-component instance adapter converts semantic fragments into separate predictions without area filtering. Low geometric precision can reflect fragmentation as well as mask quality.',
        'HPID inherited root-body inclusion is annotation-conditioned; PartCATSeg retains every non-background semantic component. This native-contract asymmetry must remain explicit.',
        'The semantic bridge is a coarse domain/part ontology: wrong objects within the same domain can share tokens; cross-domain predictions retain prefixes and are not repaired using target GT.',
        'Semantic union means are conditional on annotated reference tokens. Unannotated predicted classes do not enter that IoU average, but separate label-presence precision and all geometric components retain them.',
        'Descriptive fragmentation cutoffs in this audit do not alter any prediction, threshold, matching, cohort or reported endpoint.',
        'The primary paired bootstrap uses 42 unique source images and 10000 fixed draws. Secondary intervals are descriptive pointwise, not multiplicity-adjusted claims.'
      ]}
    target = ROOT/'scoring_independent_audit.json'
    target.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'primary':calculated['part_f1_at_025'],
                      'semantic':calculated['mean_semantic_union_iou'],'diagnostics':diagnostics},ensure_ascii=False))


if __name__ == '__main__':
    main()
