from __future__ import annotations

import argparse
import json
from pathlib import Path

from hpid_split.asset_routing import (
    Siglip2AssetEncoder,
    build_asset_routing_index,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a train-only SigLIP 2 router for modern asset categories."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    encoder = Siglip2AssetEncoder(
        args.model,
        device=args.device,
        local_files_only=args.local_files_only,
        batch_size=args.batch_size,
    )
    manifest = build_asset_routing_index(args.manifest, args.output, encoder)
    print(
        json.dumps(
            {
                "asset_count": manifest["asset_count"],
                "label_count": manifest["label_count"],
                "arrays_sha256": manifest["arrays_sha256"],
                "output": str(args.output.resolve()),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
