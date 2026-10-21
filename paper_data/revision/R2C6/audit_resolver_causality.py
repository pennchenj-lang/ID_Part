"""Deterministic unit interventions on the actual archived benchmark resolver.

No model inference, performance improvement experiment, or source-package edits.
The archived audit main() is executed with fixed prediction objects and captured
in-memory I/O; the current historical helper is also called directly.
"""
from __future__ import annotations

import ast
import copy
import csv
import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
WORK = HERE.parent.parent
R6 = WORK / "stability_work/handoff_review6"
ARCHIVE = WORK / "hpid-publication-20261009/docs/evidence/revision_20261008"
OLD = Path("__RUNTIME_ROOT__/experiments/paper_v031_identity_frontend_20260828")
CASES = OLD / "03_completion_group_frontend"
SIX = ("telephone__u01", "bottle__u01", "box__u01", "car_automobile__u02", "shoe__u01", "screwdriver__u01")
METHOD = "hpid_split_group_ids"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def csvread(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def mask(path):
    return np.asarray(Image.open(path).convert("L")) >= 128


def fingerprint(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.uint8).tobytes()).hexdigest()


def source_record(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def function_record(module, name):
    path = Path(module.__file__)
    source = path.read_text(encoding="utf-8-sig")
    node = next(n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.FunctionDef) and n.name == name)
    return {**source_record(path), "function": name, "lines": [node.lineno, node.end_lineno],
            "function_source_sha256": hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()}


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    replay = load_module("r6_resolver_audit_environment", R6 / "run_routing.py")
    import scripts.run_semantic_completion_frontend_benchmark as benchmark
    import hpid_split.paco_eval as paco_eval
    import hpid_split.paco_semantics as paco_semantics
    import hpid_split.postprocess_baselines as baselines

    archived_path = ARCHIVE / "sources/project/scripts/audit_completion_routing.py"
    archived = load_module("archived_routing_audit_for_c6", archived_path)
    original_manifest_path = OLD / "02_completion_frontend/completion_target_manifest.json"
    original_manifest = read(original_manifest_path)
    original_targets = {r["case_id"]: r for r in original_manifest["selected_cases"]}
    manifest = read(R6 / "request_manifest_frozen.json")
    targets = {r["case_id"]: r for r in manifest["requests"] if r["cohort"] == "old37"}
    original_routes = {(r["case_id"], r["method"]): r for r in csvread(CASES / "routing_correctness_cases.csv")}
    original_front = {(r["case_id"], r["method"]): r for r in csvread(CASES / "completion_frontend_cases.csv")}
    assert set(SIX) == {c for (c,m),r in original_routes.items() if m == METHOD and float(r["wrong_unique_025"]) == 1.0}

    # All 37 archived packages are loaded through the actual historical adapter.
    predictions, selected, truths, package_rows = {}, {}, {}, {}
    input_paths = {original_manifest_path, R6 / "request_manifest_frozen.json", CASES / "routing_correctness_cases.csv",
                   CASES / "completion_frontend_cases.csv", archived_path, Path(benchmark.__file__),
                   Path(paco_eval.__file__), Path(paco_semantics.__file__), Path(baselines.__file__), Path(__file__)}
    for case, t in targets.items():
        package = Path(t["package_path"])
        labels = np.asarray(Image.open(package / "group_id_map.tiff"))
        pred = benchmark._hpid_prediction(package, labels > 0)
        predictions[case] = pred
        q = original_targets[case]
        item, matches = benchmark._resolve_target(pred, target_name=q["target_part_name"],
                expected_domain=q["expected_domain"], object_category=q["object_category"])
        selected[case] = item
        truths[case] = mask(t["target_mask_path"])
        package_rows[case] = read(package / "groups.json")
        assert q["target_mask_sha256"] == sha(t["target_mask_path"])
        assert len(matches) == int(float(original_routes[case, METHOD]["query_match_count"]))
        if item is not None:
            assert item.identity == original_front[case, METHOD]["selected_identity"]
            assert np.array_equal(item.mask, mask(CASES / "cases" / case / METHOD / "selected_part.png"))
        input_paths.update([package / "groups.json", package / "group_id_map.tiff", Path(t["target_mask_path"])])
    before = {str(p.resolve()): sha(p) for p in sorted(input_paths)}

    original_read_text, original_write_text = Path.read_text, Path.write_text
    sentinel = HERE / "__in_memory_manifest__.json"

    def execute_archived(preds, case_ids):
        """Call the archived main(), preserving the original decision statements."""
        request_manifest = {"case_count": len(case_ids), "selected_cases": []}
        for case in case_ids:
            t = copy.deepcopy(original_targets[case])
            t["target_mask_relative_path"] = str(Path(targets[case]["target_mask_path"]).resolve())
            request_manifest["selected_cases"].append(t)
        captured = {}

        def read_override(path, *args, **kwargs):
            if path.resolve() == sentinel.resolve():
                return json.dumps(request_manifest)
            return original_read_text(path, *args, **kwargs)

        def write_override(path, text, *args, **kwargs):
            assert path.resolve() == (HERE / "routing_correctness_report.json").resolve()
            captured["report"] = json.loads(text)
            return len(text)

        argv = [str(archived_path), "--target-manifest", str(sentinel), "--inference-root", str(HERE),
                "--reference-root", str(HERE), "--output", str(HERE), "--bootstrap-iterations", "1"]
        with patch.object(sys, "argv", argv), patch.object(archived, "METHODS", (METHOD,)), \
             patch.object(archived, "_predictions", side_effect=lambda p: {METHOD: preds[p.name]}), \
             patch.object(archived, "_write_csv", side_effect=lambda p, r: captured.update({p.name: r})), \
             patch.object(Path, "read_text", read_override), patch.object(Path, "write_text", write_override):
            assert archived.main() == 0
        # Internal one-resample summaries are deliberately discarded. They are
        # incidental to executing main() and are NOT inferential results.
        return {r["case_id"]: r for r in captured["routing_correctness_cases.csv"]}

    archived_baseline = execute_archived(predictions, list(targets))
    for case, observed in archived_baseline.items():
        for key, value in observed.items():
            expected = original_routes[case, METHOD][key]
            assert value == expected if isinstance(value, str) else abs(float(value) - float(expected)) < 1e-12, (case,key,value,expected)

    def replace_selected(pred, old, new):
        return replace(pred, instances=tuple(new if i.identity == old.identity else i for i in pred.instances))

    def adapter_with_review_flag(case, flag):
        package = Path(targets[case]["package_path"])
        group_path = package / "groups.json"
        altered = copy.deepcopy(package_rows[case])
        for group in altered:
            if group["group_id"] == selected[case].identity:
                group["review_required"] = flag

        def read_override(path, *args, **kwargs):
            if path.resolve() == group_path.resolve():
                return json.dumps(altered)
            return original_read_text(path, *args, **kwargs)
        with patch.object(Path, "read_text", read_override):
            return benchmark._hpid_prediction(package, predictions[case].root_mask)

    scenario_predictions = {name: {} for name in (
        "unchanged", "selected_mask_empty", "selected_mask_full", "selected_mask_equals_reference",
        "review_required_false", "review_required_true", "selected_confidence_zero", "selected_confidence_one",
        "add_second_identity_same_semantic", "remove_sole_matching_identity")}
    case_records = []
    for case in SIX:
        p, s = predictions[case], selected[case]
        scenario_predictions["unchanged"][case] = p
        for name, change in (("selected_mask_empty", np.zeros_like(s.mask)),
                             ("selected_mask_full", np.ones_like(s.mask)),
                             ("selected_mask_equals_reference", truths[case])):
            scenario_predictions[name][case] = replace_selected(p, s, replace(s, mask=change.copy()))
        for value in (False, True):
            revised = adapter_with_review_flag(case, value)
            assert all((a.identity,a.semantic_name,a.semantic_parent,a.confidence) == (b.identity,b.semantic_name,b.semantic_parent,b.confidence)
                       and np.array_equal(a.mask,b.mask) for a,b in zip(p.instances,revised.instances,strict=True))
            scenario_predictions["review_required_" + str(value).lower()][case] = revised
        for label,value in (("zero",0.0),("one",1.0)):
            scenario_predictions["selected_confidence_"+label][case] = replace_selected(p,s,replace(s,confidence=value))
        scenario_predictions["add_second_identity_same_semantic"][case] = replace(p,instances=p.instances+(replace(s,identity=s.identity+"/__unit_second_identity__"),))
        scenario_predictions["remove_sole_matching_identity"][case] = replace(p,instances=tuple(i for i in p.instances if i.identity != s.identity))
        group = next(g for g in package_rows[case] if g["group_id"] == s.identity)
        case_records.append({"case_id":case,"original_query":original_targets[case],"package_path":targets[case]["package_path"],
                             "selected_group_original_record":group,"adapter_instance_fields":[f.name for f in fields(s)],
                             "adapter_selected_confidence":s.confidence,"original_selected_iou":archived_baseline[case]["selected_part_iou"],
                             "original_quality_related_fields":{k:v for k,v in group.items() if any(t in k.lower() for t in ("quality","review","confiden","evidence","verif"))}})

    observations = []
    for scenario, preds in scenario_predictions.items():
        actual = execute_archived(preds, list(SIX))
        for case in SIX:
            r = actual[case]
            q = original_targets[case]
            item, matches = benchmark._resolve_target(preds[case],target_name=q["target_part_name"],
                    expected_domain=q["expected_domain"],object_category=q["object_category"])
            assert len(matches) == r["query_match_count"]
            expected_state = "ambiguous" if scenario == "add_second_identity_same_semantic" else "unresolved" if scenario == "remove_sole_matching_identity" else "unique"
            assert r["query_"+expected_state] == 1.0
            if expected_state == "unique":
                assert item.identity == selected[case].identity
                assert (item.semantic_name,item.semantic_parent) == (selected[case].semantic_name,selected[case].semantic_parent)
            if scenario == "selected_mask_empty": assert r["selected_part_iou"] == 0
            if scenario == "selected_mask_equals_reference":
                assert r["selected_part_iou"] == 1
                assert r["wrong_unique_025"] == 0 and r["unique_and_correct_025"] == 1
            if scenario.startswith("review_required_"):
                assert r == archived_baseline[case]
            observations.append({"case_id":case,"scenario":scenario,"state":expected_state,
                                 "selected_identity":item.identity if item else None,
                                 "selected_mask_sha256":fingerprint(item.mask) if item else None,
                                 "actual_archived_audit_row":r})

    after = {str(p.resolve()): sha(p) for p in sorted(input_paths)}
    assert before == after
    mechanism = read(R6 / "mechanism_independent_audit.json")
    assert mechanism["status"] == "PASS" and mechanism["old37_all4_checked"]
    frozen_source = ARCHIVE / "sources/frozen_be54300/src/hpid_split"
    frozen_checks = {name: {"published_archive":source_record(frozen_source/name), "runtime_import":source_record(Path(module.__file__)),
                           "bytes_identical":sha(frozen_source/name)==sha(module.__file__)}
                     for name,module in (("paco_eval.py",paco_eval),("paco_semantics.py",paco_semantics))}
    assert all(v["bytes_identical"] for v in frozen_checks.values())
    report = {
        "status":"PASS_WITH_HISTORICAL_HELPER_PROVENANCE_LIMITATION", "created_utc":datetime.now(timezone.utc).isoformat(),
        "scope":"Resolver-unit causal interventions on the controlled benchmark interface only. No model inference, no upstream fusion counterfactual, no new population performance claims, no Word edits.",
        "execution":{"archived_main_called":True,"current_resolve_target_called_directly":True,
                     "archived_main_baseline_requests_exact":37,"original_wrong_unique_cases_exact":6,
                     "unit_case_scenarios_including_unchanged_control":len(observations),
                     "non_control_case_scenarios":len(observations)-len(SIX),"inputs_unchanged":True,
                     "in_memory_IO":"Only fixed predictions and manifest/file I/O are patched. The archived main decision statements and current helper bodies are executed unchanged. No mock decision function is used.",
                     "unit_design":"Mask interventions deliberately modify resolver input objects, retaining names/IDs/semantic-parent fields. They are not valid alternative fuser exports or accuracy-improvement experiments. GT masks are used only for the reference-mask unit intervention and scoring; no model receives GT.",
                     "discarded_internal_summaries":"Archived main() internally runs its summary bootstrap with one resample to finish execution; these unused summaries are discarded and are not reported as statistics."},
        "source_bindings":{"archived_audit":source_record(archived_path),"original_target_manifest":source_record(original_manifest_path),
                           "adapter":function_record(benchmark,"_hpid_prediction"),"resolver":function_record(benchmark,"_resolve_target"),
                           "name_adapter":function_record(benchmark,"_normalized_name"),"semantic_normalizer":function_record(paco_eval,"_normalize"),
                           "category_alias_normalizer":function_record(paco_semantics,"canonical_part_token"),
                           "baseline_types":source_record(baselines.__file__),"frozen_normalizers":frozen_checks,
                           "all_input_sha256":before},
        "provenance_limit":"The published archived routing audit and be54300 normalizers are executed directly. The historical generation helper/Group adapter has no copy in the published frozen source tree; this audit binds its current local source hash and establishes exact agreement with all 37 archived Group routes and saved selected masks. It does not prove that the entire current helper file is byte-identical to its 2026-08-28 version.",
        "original_six_cases":case_records,"unit_observations":observations,
        "established_execution_causal_claims":[
            "At this benchmark resolver interface, the number of normalized semantic-name matches determines unique/ambiguous/unresolved. Exactly one match causes the unique branch regardless of its geometry, review_required flag, or confidence magnitude.",
            "The Group adapter drops the actual exported review_required/evidence fields and assigns each Group a constant confidence of 1.0. This is a property of this benchmark adapter, not a conclusion about every production package consumer.",
            "For all six archived wrong-unique cases, empty/full/reference mask substitutions leave the unique branch and selected identity unchanged although evaluated IoU changes; reference substitution reaches IoU=1 and changes the external correctness label from wrong-unique to correct-unique without changing the resolver's unique branch.",
            "For all six, changing only exported review_required to false or true produces identical adapter instances and identical route/evaluation rows.",
            "For all six, adding a second same-semantic identity changes the state to ambiguous; removing the sole matching identity changes it to unresolved. These are deterministic unit-level causal interventions.",
            "The six actual original selected Groups all have review_required=false. The interventions with review_required=true show that even this warning would not change the benchmark branch; they do not show that such a warning existed in those six original outputs.",
            "Unique is a match-count state, not a calibrated quality/confidence assertion. Setting the sole match's confidence to 0 or 1 leaves its ID and unique branch unchanged. The adapter's constant 1.0 is not established as the cause of the bad original mask or sole-match error."
        ],
        "upstream_stage_evidence":{"source":source_record(R6 / "mechanism_independent_audit.json"),
            "facts":"All old37 losses were independently traced. Box side proposal pixels mostly occupy bottom, car windshield pixels divide among window/windshield, and shoe upper pixels divide among body/details; their matching Group IoUs remain below .25.",
            "causal_boundary":"These observed pixel destinations and first-loss stages are execution facts. Existing endpoint/log comparisons do not isolate a particular ownership coefficient, eligibility gate, instance extraction, or cleanup operation as the sole cause. The resolver intervention does not establish the upstream cause of the bad masks."},
        "prohibited_inferences":["All production resolvers ignore review flags.","The six original outputs were high-confidence model errors.",
            "The original aliases were proved wrong or caused the geometry error.","A specific ownership constant caused all six masks to fail.",
            "These unit interventions improve or estimate segmentation/routing performance.","A GT mask would be available at deployment."]}
    path = HERE / "resolver_causal_audit.json"
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":report["status"],"baseline_exact":37,"unit_observations":len(observations),
                      "original_review_flags":{r["case_id"]:r["original_quality_related_fields"] for r in case_records},
                      "report":str(path),"report_sha256":sha(path),"script_sha256":sha(__file__)},ensure_ascii=False))


if __name__ == "__main__":
    main()
