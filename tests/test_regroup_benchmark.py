from __future__ import annotations

import hashlib
import json

from scripts.regroup_benchmark import _register_benchmark_evaluation


def test_benchmark_evaluation_is_registered_in_package_manifest(tmp_path) -> None:
    payload = tmp_path / "payload.txt"
    payload.write_text("payload", encoding="utf-8")
    evaluation = tmp_path / "paco_evaluation.json"
    evaluation.write_text('{"uses_ground_truth": true}', encoding="utf-8")
    manifest_path = tmp_path / "package_manifest.json"
    manifest_path.write_text('{"files": []}', encoding="utf-8")

    _register_benchmark_evaluation(tmp_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = {row["path"]: row for row in manifest["files"]}
    assert set(rows) == {"paco_evaluation.json", "payload.txt"}
    assert rows["paco_evaluation.json"]["sha256"] == hashlib.sha256(
        evaluation.read_bytes()
    ).hexdigest()
    assert manifest["benchmark_evaluation_uses_ground_truth"] is True
