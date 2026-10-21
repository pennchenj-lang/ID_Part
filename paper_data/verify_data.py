"""Verify deposited evidence files against MANIFEST.json (standard library only)."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
errors = []
for record in manifest["files"]:
    path = root / record["path"]
    if not path.is_file():
        errors.append(record["path"] + ": missing")
        continue
    content = path.read_bytes()
    if len(content) != record["bytes"] or hashlib.sha256(content).hexdigest() != record["sha256"]:
        errors.append(record["path"] + ": digest or size mismatch")
if errors:
    raise SystemExit("\n".join(errors))
print(f"Verified {len(manifest['files'])} evidence files.")
