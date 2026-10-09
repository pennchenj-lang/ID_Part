from pathlib import Path
import json, hashlib
from collections import Counter
R=Path(r"${REVISION_ROOT}")
roots=[Path(r"${RUNTIME_ROOT}/experiments/paper_v031/test226_hpid"),Path(r"${RUNTIME_ROOT}/experiments/paper_v031_untouched_group_holdout_42_20260825_r5/02_blind_inference")]
part_fields=["part_id","group_id","semantic_name","semantic_parent","assembly_parent_id","mask_visible_path","bbox_visible"]
manifest_fields=["candidate_audit_path","diagnostics_path","quality_status","quality_report_path"]
counts=Counter();failures=[]
for root in roots:
 for path in sorted(root.glob("*/parts.json")):
  p=path.parent; parts=json.loads(path.read_text(encoding="utf-8-sig"));m=json.loads((p/"package_manifest.json").read_text(encoding="utf-8-sig"))
  counts["packages"]+=1;counts["parts"]+=len(parts);counts["status_"+m["quality_status"]]+=1
  for f in manifest_fields:
   if f not in m:failures.append([str(p),f,"missing"])
  for f in ["candidate_audit_path","diagnostics_path","quality_report_path"]:
   if m.get(f) is not None and not (p/m[f]).exists():failures.append([str(p),f,"missing_target"])
  for row in parts:
   for f in part_fields:
    if f not in row:failures.append([str(p),row["part_id"],f,"missing"])
   if not (p/row["mask_visible_path"]).exists():failures.append([str(p),row["part_id"],"maskfile_missing"])
   if row["group_id"] is None:counts["null_group_ids"]+=1
out={"status":"PASS" if not failures else "FAIL","counts":dict(counts),"failures":failures,
 "verified_part_fields":part_fields,"verified_manifest_fields":manifest_fields,
 "source":"Frozen export.py:342-371,390-410; quality.py:512-524; instances.py:11-26",
 "note":"Actual268 archived packages checked. A field's presence and valid file link do not validate the semantic correctness of its label."}
(R/"evidence/qa_contract_schema.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
print(out)

