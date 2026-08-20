from __future__ import annotations

from dataclasses import dataclass

from .fusion import MaskCandidate
from .root_routing import candidate_root_key


@dataclass(frozen=True)
class EnsembleEvidenceGateConfig:
    """Keep ensemble output semantic when a closed profile is well supported."""

    minimum_named_semantics: int = 3


@dataclass(frozen=True)
class EnsembleEvidenceGateResult:
    candidates: tuple[MaskCandidate, ...]
    diagnostics: dict[str, object]


def filter_unresolved_ensemble_regions(
    candidates: list[MaskCandidate] | tuple[MaskCandidate, ...],
    roots: list[MaskCandidate] | tuple[MaskCandidate, ...],
    config: EnsembleEvidenceGateConfig | None = None,
) -> EnsembleEvidenceGateResult:
    """Drop unnamed visual tiles once an ensemble has enough named part evidence.

    The root candidate remains in the candidate set, so fusion assigns every
    removed tile back to the canonical object instead of leaving a hole. Open-set
    objects and weakly resolved profiles keep their generic regions.
    """

    config = config or EnsembleEvidenceGateConfig()
    roots_by_key = {
        key: root
        for root in roots
        if (key := candidate_root_key(root)) is not None
    }
    named_by_root: dict[str, set[str]] = {key: set() for key in roots_by_key}
    for candidate in candidates:
        root_key = candidate_root_key(candidate)
        root = roots_by_key.get(root_key)
        if root is None or candidate is root:
            continue
        if bool(candidate.metadata.get("generic_visual_region")):
            continue
        if candidate.semantic_name == root.semantic_name:
            continue
        named_by_root[root_key].add(candidate.semantic_name)

    guarded_roots = {
        root_key
        for root_key, root in roots_by_key.items()
        if root.metadata.get("selected_part_profile")
        and len(named_by_root[root_key]) >= config.minimum_named_semantics
    }
    output: list[MaskCandidate] = []
    removed_rows: list[dict[str, object]] = []
    for candidate in candidates:
        root_key = candidate_root_key(candidate)
        generic_visual = bool(candidate.metadata.get("generic_visual_region")) and bool(
            candidate.metadata.get("visual_region")
        )
        if generic_visual and root_key in guarded_roots:
            physical_evidence = candidate.metadata.get("physical_region_gate")
            vlm_confirmed = bool(
                isinstance(physical_evidence, dict)
                and physical_evidence.get("vlm_physical_supported")
            )
            if not vlm_confirmed:
                removed_rows.append(
                    {
                        "candidate_key": candidate.metadata.get("candidate_key"),
                        "root_key": root_key,
                        "semantic_name": candidate.semantic_name,
                        "reason": "unresolved_visual_region_after_named_consensus",
                    }
                )
                continue
        output.append(candidate)

    return EnsembleEvidenceGateResult(
        tuple(output),
        {
            "algorithm": "hpid-conservative-ensemble-evidence-gate-v1",
            "status": "completed",
            "input_candidate_count": len(candidates),
            "output_candidate_count": len(output),
            "guarded_root_count": len(guarded_roots),
            "removed_generic_region_count": len(removed_rows),
            "named_semantics_by_root": {
                key: sorted(values) for key, values in sorted(named_by_root.items())
            },
            "removed_candidates": removed_rows,
            "ground_truth_used": False,
        },
    )
