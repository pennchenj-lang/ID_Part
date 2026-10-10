"""Independent CPU-only scorer for sealed PartCATSeg predictions.

No model is imported or run. The primary score preserves the released HPID
PACO geometric matching rule. Semantic-union IoU is an annotation-conditioned
comparison under the frozen canonical ontology, not official benchmark mIoU.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import mmap
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


HERE = Path(__file__).resolve().parent
SNAPSHOT = Path("__RUNTIME_ROOT__/code_snapshots/hpid_split_be54300_holdout")
REFERENCES = HERE.parent / "full_image_review4/sealed_reference_manifest.json"
PUBLIC_TAXONOMY = Path("__RUNTIME_ROOT__/datasets/paco/annotations/paco_lvis_v1_train.json")
sys.path.insert(0, str(SNAPSHOT / "src"))
from hpid_split.paco_semantics import canonical_part_token, normalize_paco_name


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def binary_mask(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8) >= 128


def iou(a, b):
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else 0.0


def gt_token(name, domain, category):
    return canonical_part_token(name, domain, object_category=category)


def predicted_token(semantic, expected_domain):
    # Mirrors the already-frozen evaluator. The expected domain does not repair
    # a prediction in another domain: its prefix remains part of its token.
    if semantic == expected_domain:
        return "body"
    return canonical_part_token(semantic.removeprefix(expected_domain + "_"), expected_domain)


def load_reference(row):
    case_path = Path(row["case_path"])
    case = read_json(case_path)
    if row.get("case_sha256") and sha256(case_path) != row["case_sha256"]:
        raise ValueError("Reference case checksum mismatch: " + row["anonymous_id"])
    truth_rows = case["parts"]
    masks = [binary_mask(case_path.parent / part["mask_crop"]) for part in truth_rows]
    if not masks or len({mask.shape for mask in masks}) != 1:
        raise ValueError("Reference masks missing or inconsistent")
    with Image.open(case_path.parent / "source_crop.png") as image:
        if (image.height, image.width) != masks[0].shape:
            raise ValueError("Reference mask/source crop size mismatch")
    domain = row["expected_domain"]
    category = case["object_category"]
    tokens = [gt_token(part["part_name"], domain, category) for part in truth_rows]
    unions = {}
    for token, mask in zip(tokens, masks, strict=True):
        unions[token] = unions.get(token, np.zeros(mask.shape, dtype=bool)) | mask
    overlap_count = np.zeros(masks[0].shape, dtype=np.uint16)
    for mask in masks:
        overlap_count += mask
    return {
        "case": case, "rows": truth_rows, "masks": masks, "tokens": tokens,
        "unions": unions, "domain": domain, "shape": masks[0].shape,
        "annotation_pixels": int(np.count_nonzero(overlap_count)),
        "overlapping_annotation_pixels": int(np.count_nonzero(overlap_count > 1)),
        "include_hpid_root_body": "body" in tokens,
    }


def load_hpid(row, ref):
    package = Path(row["historical_package"])
    rows = read_json(package / "parts.json")
    rows = [r for r in rows if r.get("semantic_name") != ref["domain"] or ref["include_hpid_root_body"]]
    masks = [binary_mask(package / r["mask_visible_path"]) for r in rows]
    if any(mask.shape != ref["shape"] for mask in masks):
        raise ValueError("Historical HPID/reference dimension mismatch")
    unions = {}
    for r, mask in zip(rows, masks, strict=True):
        token = predicted_token(str(r["semantic_name"]), ref["domain"])
        unions[token] = unions.get(token, np.zeros(ref["shape"], dtype=bool)) | mask
    matrix = np.zeros((len(ref["masks"]), len(masks)), dtype=np.float32)
    for ti, truth in enumerate(ref["masks"]):
        for pi, prediction in enumerate(masks):
            matrix[ti, pi] = iou(truth, prediction)
    return matrix, unions, {"predicted_part_count": len(masks), "root_body_included": ref["include_hpid_root_body"]}


def geometric_scores(matrix):
    truth_count, prediction_count = matrix.shape
    assigned = []
    if matrix.size:
        ti, pi = linear_sum_assignment(1.0 - matrix)
        assigned = [(int(t), int(p), float(matrix[t, p])) for t, p in zip(ti, pi, strict=True)]
    result = {"truth_part_count": truth_count, "predicted_part_count": prediction_count, "assignments": assigned}
    for threshold, key in [(0.25, "025"), (0.50, "050"), (0.75, "075")]:
        matched = sum(score >= threshold for _, _, score in assigned)
        precision = matched / prediction_count if prediction_count else 0.0
        recall = matched / truth_count if truth_count else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        result.update({f"part_f1_at_{key}": f1, f"part_precision_at_{key}": precision,
                       f"part_recall_at_{key}": recall, f"matched_count_at_{key}": matched})
    return result


def semantic_scores(truth_unions, predicted_unions):
    # Full crop union masks, with no clipping to the annotated object/part region.
    # Only actually annotated reference semantic classes enter this mean. Report
    # the separate predicted-class precision and counts so hallucinated classes
    # cannot be hidden behind the reference-conditioned mean.
    rows = []
    for token, truth in sorted(truth_unions.items()):
        prediction = predicted_unions.get(token)
        rows.append({"semantic": token, "union_iou": iou(truth, prediction) if prediction is not None else 0.0,
                     "truth_pixels": int(truth.sum()), "predicted_pixels": int(prediction.sum()) if prediction is not None else 0})
    overlaps = [row["union_iou"] for row in rows]
    tp_tokens = set(truth_unions) & set(predicted_unions)
    return {
        "mean_semantic_union_iou": float(np.mean(overlaps)) if overlaps else 0.0,
        "semantic_union_recall_at_025": sum(x >= 0.25 for x in overlaps) / max(1, len(overlaps)),
        "semantic_label_precision": len(tp_tokens) / max(1, len(predicted_unions)),
        "annotated_semantic_class_count": len(truth_unions),
        "predicted_semantic_class_count": len(predicted_unions),
        "unmatched_predicted_semantic_classes": sorted(set(predicted_unions) - set(truth_unions)),
        "semantic_union_rows": rows,
    }


def read_registry(path):
    data = read_json(path)
    labels = data.get("labels", data.get("classes"))
    if not isinstance(labels, list) or len(labels) != 456:
        raise ValueError("Expected complete public PACO 456-label registry")
    canonical = []
    for position, item in enumerate(labels):
        index = int(item.get("label_index", item.get("index", position)))
        raw_name = item.get("paco_name", item.get("category_name", item.get("name")))
        category_id = item.get("paco_category_id", item.get("category_id"))
        if index != position or not raw_name or ":" not in raw_name or category_id is None:
            raise ValueError("Registry must have contiguous ordered indices and public PACO pair names/ids")
        canonical.append({"label_index": index, "paco_category_id": int(category_id), "paco_name": raw_name})
    if len({r["paco_category_id"] for r in canonical}) != 456 or len({r["paco_name"] for r in canonical}) != 456:
        raise ValueError("Duplicate PACO registry classes")
    # Read the public training taxonomy only, without loading annotations or
    # deriving a vocabulary from present classes in the evaluation cohort.
    with PUBLIC_TAXONOMY.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as contents:
        start = contents.find(b'"categories":')
        if start < 0:
            raise ValueError("Public PACO categories field missing")
        categories, _ = json.JSONDecoder().raw_decode(contents[start + 13:start + 1_000_013].decode("utf-8").lstrip())
    pairs = sorted([item for item in categories if ":" in item["name"]], key=lambda item: item["id"])
    expected = [{"label_index": i, "paco_category_id": item["id"], "paco_name": item["name"]} for i, item in enumerate(pairs)]
    if canonical != expected:
        raise ValueError("Registry differs from complete public PACO categories in numeric id order")
    background = int(data.get("background_index", 456))
    if background != 456:
        raise ValueError("Expected frozen background index456")
    return canonical, background


def category_domains():
    catalog = read_json(SNAPSHOT / "configs/paco_broad_categories.json")
    return {item["category"]: item["expected_domain"] for item in catalog["categories"]}


def registry_semantics(registry, domains):
    semantic_names = []
    for item in registry:
        category, part = item["paco_name"].split(":", 1)
        domain = domains.get(category)
        if domain is None:
            # Never borrow the GT category/domain to interpret this prediction.
            semantic = "unmapped_paco_" + str(item["paco_category_id"])
        else:
            semantic = domain + "_" + canonical_part_token(part, domain, object_category=category)
        semantic_names.append(semantic)
    return semantic_names


def component_map(labels, background):
    components = np.zeros(labels.shape, dtype=np.int32)
    # Position k-1 gives the source semantic label of component id k.
    source_labels = []
    offset = 0
    for label in sorted(int(v) for v in np.unique(labels) if int(v) != background):
        connected, number = ndimage.label(labels == label, structure=np.ones((3, 3), dtype=np.uint8))
        foreground = connected > 0
        components[foreground] = connected[foreground] + offset
        source_labels.extend([label] * int(number))
        offset += int(number)
    return components, source_labels


def component_iou_matrix(truth_masks, components, count):
    areas = np.bincount(components.ravel(), minlength=count + 1)[1:].astype(np.int64)
    matrix = np.zeros((len(truth_masks), count), dtype=np.float32)
    for index, truth in enumerate(truth_masks):
        intersections = np.bincount(components[truth], minlength=count + 1)[1:]
        unions = int(truth.sum()) + areas - intersections
        matrix[index] = np.divide(intersections, unions, out=np.zeros(count, dtype=float), where=unions > 0)
    return matrix, areas


def load_labels(path, shape, background):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        labels = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            labels = np.asarray(image)
    if labels.ndim != 2 or labels.shape != shape or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Prediction must be an integer label image at original crop dimensions")
    if labels.min() < 0 or labels.max() > background:
        raise ValueError("Label values outside frozen0..456; ignore sentinel cannot be a prediction")
    return labels


def evaluate_labels(labels, ref, semantic_names, background):
    components, source_labels = component_map(labels, background)
    matrix, areas = component_iou_matrix(ref["masks"], components, len(source_labels))
    unions = {}
    unmapped_pixels = 0
    for index in np.unique(labels):
        index = int(index)
        if index == background:
            continue
        token = predicted_token(semantic_names[index], ref["domain"])
        mask = labels == index
        unions[token] = unions.get(token, np.zeros(ref["shape"], dtype=bool)) | mask
        if semantic_names[index].startswith("unmapped_paco_"):
            unmapped_pixels += int(mask.sum())
    return {
        **geometric_scores(matrix), **semantic_scores(ref["unions"], unions),
        "predicted_raw_semantic_class_count": int(len(np.unique(labels[labels != background]))),
        "foreground_pixels": int(np.count_nonzero(labels != background)),
        "unmapped_prediction_pixels": unmapped_pixels,
        "unmapped_predicted_components": sum(semantic_names[label].startswith("unmapped_paco_") for label in source_labels),
        "component_area_min": int(areas.min()) if len(areas) else 0,
        "component_area_median": float(np.median(areas)) if len(areas) else 0.0,
        "component_area_max": int(areas.max()) if len(areas) else 0,
        "component_source_labels": source_labels,
    }


def zero_result(ref):
    return {**geometric_scores(np.zeros((len(ref["masks"]), 0), dtype=np.float32)),
            **semantic_scores(ref["unions"], {}), "foreground_pixels": 0,
            "unmapped_prediction_pixels": 0, "unmapped_predicted_components": 0}


def prediction_rows(path):
    manifest = read_json(path)
    cases = manifest["cases"]
    if not isinstance(cases, list):
        raise ValueError("predictions manifest cases must be a list")
    indexed = {row["anonymous_id"]: row for row in cases}
    if len(indexed) != len(cases):
        raise ValueError("Duplicate anonymous prediction id")
    return indexed


def validate_seal(receipt_path, manifest_path, registry_path, rows, reference_ids):
    receipt = read_json(receipt_path)
    if receipt.get("status") not in ("completed", "complete", "sealed"):
        raise ValueError("Inference receipt is not explicitly terminal/sealed")
    for field, path in [("predictions_manifest_sha256", manifest_path), ("label_registry_sha256", registry_path)]:
        if receipt.get(field) != sha256(path):
            raise ValueError("Inference receipt checksum mismatch: " + field)
    if set(rows) != set(reference_ids):
        raise ValueError("Manifest must explicitly retain all42 known cases, including failures")
    records = {r["anonymous_id"]: r for r in receipt.get("cases", [])}
    if set(records) != set(reference_ids):
        raise ValueError("Receipt must bind all42 terminal case records")
    for anonymous_id, row in rows.items():
        status = str(row.get("status", "")).lower()
        if status not in ("complete", "completed", "failed"):
            raise ValueError("Prediction row not in supported terminal state: " + anonymous_id)
        # Normalize only the parsed in-memory row. The original manifest and
        # receipt bytes remain unchanged and retain their sealed checksums.
        row["status"] = status
        record = records[anonymous_id]
        record_status = str(record.get("status", "")).lower()
        if record_status != status:
            raise ValueError("Receipt/manifest case state mismatch: " + anonymous_id)
        if status == "failed":
            continue
        path = Path(row["label_path"])
        if not path.is_absolute():
            path = Path(manifest_path).resolve().parent / path
        if not path.is_file() or record.get("label_sha256") != sha256(path):
            raise ValueError("Sealed prediction file missing or modified: " + anonymous_id)
        row["resolved_label_path"] = str(path)
    return receipt


def paired_summary(cases, keys, seed=20261009, draws=10000):
    # Existing cohort has one target per source image. Reject accidental repeats
    # instead of silently treating image-correlated targets as independent.
    if len({case["image_id"] for case in cases}) != len(cases):
        raise ValueError("Repeated source images require cluster bootstrap")
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(cases), size=(draws, len(cases)))
    result = {}
    for key in keys:
        hpid = np.asarray([case["hpid"][key] for case in cases], dtype=float)
        learned = np.asarray([case["partcatseg"][key] for case in cases], dtype=float)
        differences = hpid - learned
        ci = np.percentile(differences[sample_indices].mean(axis=1), [2.5, 97.5])
        result[key] = {"hpid_mean": float(hpid.mean()), "partcatseg_mean": float(learned.mean()),
                       "paired_hpid_minus_partcatseg": float(differences.mean()),
                       "paired_percentile_95ci": ci.tolist(), "n": len(cases)}
    return result


def check_baseline(references, output_path):
    # Only archived HPID outputs/reference data are read in this mode. This is
    # safe to run while PartCATSeg inference is still pending.
    checks = []
    means = []
    for row in references:
        ref = load_reference(row)
        matrix, unions, _ = load_hpid(row, ref)
        scores = {**geometric_scores(matrix), **semantic_scores(ref["unions"], unions)}
        historical_path = Path(row["historical_package"]).parents[1] / "03_scored_holdout/cases" / (row["case_id"] + ".json")
        historical = read_json(historical_path)
        pairs = [("part_f1_at_025", "part_discovery_f1_at_025"),
                 ("part_precision_at_025", "part_discovery_precision_at_025"),
                 ("part_recall_at_025", "part_discovery_recall_at_025"),
                 ("mean_semantic_union_iou", "mean_semantic_union_iou"),
                 ("semantic_label_precision", "semantic_union_precision"),
                 ("semantic_union_recall_at_025", "semantic_union_recall")]
        differences = {ours: abs(scores[ours] - float(historical[old])) for ours, old in pairs}
        passed = max(differences.values()) <= 1e-7 and scores["predicted_part_count"] == historical["predicted_part_count"]
        checks.append({"anonymous_id": row["anonymous_id"], "case_id": row["case_id"], "status": "PASS" if passed else "FAIL",
                       "absolute_differences": differences, "historical_score_sha256": sha256(historical_path)})
        means.append(scores["part_f1_at_025"])
    result = {"status": "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL",
              "scope": "Historical HPID reproducibility only; no PartCATSeg outputs read", "case_count": len(checks),
              "hpid_mean_part_f1_at_025": float(np.mean(means)), "checks": checks,
              "scorer_sha256": sha256(__file__), "created_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output_path, result)
    return result


def self_check():
    # Diagonal pixels join under 8-connectivity; separated islands remain two.
    labels = np.full((5, 5), 456, dtype=np.uint16)
    labels[0, 0] = labels[1, 1] = labels[4, 4] = 0
    labels[0, 4] = 1
    components, source = component_map(labels, 456)
    assert source == [0, 0, 1] and components[0, 0] == components[1, 1]
    truth = [labels == 0, labels == 1]
    fast, _ = component_iou_matrix(truth, components, len(source))
    direct = np.asarray([[iou(t, components == n + 1) for n in range(len(source))] for t in truth], dtype=np.float32)
    assert np.array_equal(fast, direct)
    assert geometric_scores(np.ones((1, 1), dtype=np.float32))["part_f1_at_025"] == 1.0
    assert geometric_scores(np.empty((1, 0), dtype=np.float32))["part_f1_at_025"] == 0.0
    assert geometric_scores(np.array([[0.25]], dtype=np.float32))["part_f1_at_025"] == 1.0
    assert geometric_scores(np.array([[0.249]], dtype=np.float32))["part_f1_at_025"] == 0.0
    # Predictions outside the reference region reduce semantic IoU; no GT clip.
    assert semantic_scores({"x": labels == 1}, {"x": labels != 456})["mean_semantic_union_iou"] == 0.25
    assert predicted_token("furniture_leg", "device") != "leg"
    return {"status": "PASS", "checks": 8}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=REFERENCES)
    parser.add_argument("--predictions-manifest", type=Path)
    parser.add_argument("--label-registry", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--output", type=Path, default=HERE / "scoring")
    parser.add_argument("--baseline-check-only", action="store_true")
    parser.add_argument("--self-check-only", action="store_true")
    args = parser.parse_args()
    sanity = self_check()
    if args.self_check_only:
        print(json.dumps(sanity))
        return
    references = read_json(args.references)["cases"]
    if len(references) != 42 or len({r["anonymous_id"] for r in references}) != 42:
        raise ValueError("Expected original all42 cohort")
    if args.baseline_check_only:
        result = check_baseline(references, args.output / "baseline_reproduction.json")
        print(json.dumps({k: v for k, v in result.items() if k != "checks"}))
        if result["status"] != "PASS":
            raise SystemExit(1)
        return
    if not all([args.predictions_manifest, args.label_registry, args.receipt]):
        parser.error("Scoring requires --predictions-manifest, --label-registry and --receipt")
    registry, background = read_registry(args.label_registry)
    pred_rows = prediction_rows(args.predictions_manifest)
    receipt = validate_seal(args.receipt, args.predictions_manifest, args.label_registry, pred_rows,
                            [r["anonymous_id"] for r in references])
    semantics = registry_semantics(registry, category_domains())
    baseline_check = check_baseline(references, args.output / "baseline_reproduction.json")
    if baseline_check["status"] != "PASS":
        raise ValueError("Historical baseline reproducibility failed")
    cases = []
    for row in references:
        ref = load_reference(row)
        matrix, unions, hpid_meta = load_hpid(row, ref)
        historical_scores = {**geometric_scores(matrix), **semantic_scores(ref["unions"], unions), **hpid_meta}
        pred = pred_rows[row["anonymous_id"]]
        status = pred["status"]
        error = pred.get("error")
        if status == "failed":
            learned_scores = zero_result(ref)
        else:
            try:
                labels = load_labels(pred["resolved_label_path"], ref["shape"], background)
                learned_scores = evaluate_labels(labels, ref, semantics, background)
                status = "complete"
            except (ValueError, OSError) as exc:
                # Output format failure stays in the all-case denominator. An
                # altered/missing sealed file is a fatal seal error above.
                status, error = "invalid_output", str(exc)
                learned_scores = zero_result(ref)
        case_result = {
            "anonymous_id": row["anonymous_id"], "case_id": row["case_id"], "image_id": row["image_id"],
            "object_category": ref["case"]["object_category"], "expected_domain": ref["domain"],
            "status": status, "error": error, "hpid": historical_scores, "partcatseg": learned_scores,
            "shape_hw": list(ref["shape"]), "annotated_pixels": ref["annotation_pixels"],
            "overlapping_annotation_pixels": ref["overlapping_annotation_pixels"],
        }
        cases.append(case_result)
        write_json(args.output / "cases" / (row["anonymous_id"] + ".json"), case_result)
    metrics = ["part_f1_at_025", "part_f1_at_050", "part_f1_at_075", "part_precision_at_025",
               "part_recall_at_025", "mean_semantic_union_iou", "semantic_union_recall_at_025", "semantic_label_precision"]
    summary = {
        "status": "COMPLETE", "created_utc": datetime.now(timezone.utc).isoformat(),
        "case_count": len(cases), "valid_partcatseg_outputs": sum(case["status"] == "complete" for case in cases),
        "primary_endpoint": "part_f1_at_025", "comparison_direction": "HPID minus PartCATSeg",
        "statistics": {"seed": 20261009, "bootstrap_draws": 10000, "unit": "unique source image", "CI": "paired percentile95%"},
        "metrics": paired_summary(cases, metrics),
        "semantic_scope": "Annotation-conditioned class-union IoU in frozen canonical ontology; not official dense benchmark mIoU. Full crop unions, no GT clipping; mean over annotated reference semantics. Semantic label precision is label-presence precision, not IoU-threshold precision.",
        "adapter": "Each non-background raw semantic label split by8-connectivity; every component retained, no area filter. HPID native Parts/root-body rule unchanged.",
        "limitations": "Semantic model plus deterministic component adapter; wrong target classes and unmapped classes remain primary geometric predictions. This is cross-dataset object-crop transfer with a shared annotation-derived crop, not full-image discovery.",
        "inputs": {"reference_manifest_sha256": sha256(args.references), "predictions_manifest_sha256": sha256(args.predictions_manifest),
                   "label_registry_sha256": sha256(args.label_registry), "receipt_sha256": sha256(args.receipt),
                   "scorer_sha256": sha256(__file__), "paco_semantics_sha256": sha256(SNAPSHOT / "src/hpid_split/paco_semantics.py")},
        "self_check": sanity, "baseline_reproduction": baseline_check["status"],
    }
    write_json(args.output / "cases.json", cases)
    write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
