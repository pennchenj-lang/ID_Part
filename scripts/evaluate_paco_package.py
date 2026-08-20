from __future__ import annotations

import argparse
import json
from pathlib import Path

from hpid_split.paco_eval import evaluate_paco_package


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate one HPID package against a materialized PACO case."
    )
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--expected-domain", required=True)
    parser.add_argument("--expected-profile")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = evaluate_paco_package(
        args.package,
        args.case,
        expected_domain=args.expected_domain,
        expected_profile=args.expected_profile,
    )
    output = args.output or args.package / "paco_evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"domain_correct={result['domain_correct']} "
        f"object_iou={result['object_iou']:.4f} "
        f"part_f1={result['part_discovery_f1_at_025']:.4f} "
        f"semantic_recall={result['semantic_part_recall']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
