"""Validate completed stage traces against archived endpoints and invariants."""
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARCHIVE = Path(r"${RUNTIME_ROOT}\experiments\paper_v031_identity_frontend_20260828\01_same_candidate\same_candidate_postprocessing_cases.csv")


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    cases = read(HERE / "stage_case_metrics.csv")
    archived = {(r["case_id"], r["method"]): r for r in read(ARCHIVE)}
    discrepancy = 0.0
    for row in cases:
        if row["stage"] not in {"final_hpid", "final_dbscan"}:
            continue
        method = "hpid_split_a3" if row["stage"] == "final_hpid" else "dbscan_fusion"
        original = archived[row["case_id"], method]
        for actual, expected in [("diagnostic_f1_at_025", "part_f1_at_025"),
                                 ("diagnostic_semantic_f1_at_025", "semantic_f1_at_025"),
                                 ("prediction_count", "predicted_part_count")]:
            difference = abs(float(row[actual]) - float(original[expected]))
            discrepancy = max(discrepancy, difference)
            assert difference < 1e-10, (row["case_id"], actual, difference)
    tracks = read(HERE / "stage_reference_tracks.csv")
    index = {(r["case_id"], r["reference_index"], r["stage"]): r for r in tracks}
    references = 0
    for row in tracks:
        if row["stage"] == "identity_plus_eligible_remainder":
            after = index[row["case_id"], row["reference_index"], "after_remainder_attachment_before_filter"]
            for metric in ["semantic_union_recall", "any_union_recall"]:
                assert abs(float(row[metric]) - float(after[metric])) < 1e-12
            references += 1
        elif row["stage"] == "after_remainder_attachment_before_filter":
            final = index[row["case_id"], row["reference_index"], "final_hpid"]
            for metric in ["semantic_union_recall", "any_union_recall"]:
                assert float(final[metric]) <= float(row[metric]) + 1e-12
    protocol = json.loads((HERE / "stage_protocol.json").read_text(encoding="utf-8"))
    for filename, digest_key in [(protocol["fusion_source"], "fusion_sha256"),
                                 (protocol["baseline_source"], "baseline_sha256"),
                                 (str(HERE.parent / "stage_diagnosis.py"), "script_sha256")]:
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == protocol[digest_key]
    result = {
        "case_count": len({r["case_id"] for r in cases}),
        "reference_count": references,
        "final_metrics_vs_archived_maximum_difference": discrepancy,
        "remainder_union_preserved_all_references": True,
        "final_filter_support_is_subset_all_references": True,
        "recorded_source_hashes_match": True,
        "independent_validation": "Separate table/invariant check; not an independent dataset or replication.",
    }
    (HERE / "stage_validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
