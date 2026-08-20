from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

FORMAT = "HPID local regression suite"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit and deduplicate an HPID local regression manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT:
        raise ValueError(f"manifest format must be {FORMAT!r}")
    seen_case_ids: set[str] = set()
    seen_hashes: dict[str, str] = {}
    rows: list[dict[str, object]] = []
    for raw in manifest.get("cases", []):
        case_id = str(raw["case_id"])
        if case_id in seen_case_ids:
            raise ValueError(f"duplicate case_id: {case_id}")
        seen_case_ids.add(case_id)
        image_path = Path(raw["image"]).expanduser()
        if not image_path.is_absolute():
            image_path = (args.manifest.parent / image_path).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        digest = _sha256(image_path)
        if digest in seen_hashes:
            raise ValueError(
                f"duplicate image content: {case_id} and {seen_hashes[digest]}"
            )
        seen_hashes[digest] = case_id
        with Image.open(image_path) as image:
            width, height = image.size
        rows.append(
            {
                **raw,
                "image": str(image_path),
                "image_sha256": digest,
                "width": width,
                "height": height,
            }
        )
    audit = {
        "format": "HPID local regression suite audit",
        "source_manifest": str(args.manifest.resolve()),
        "case_count": len(rows),
        "cases": rows,
    }
    payload = json.dumps(audit, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
