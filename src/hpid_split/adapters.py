from __future__ import annotations

import numpy as np

from .fusion import MaskCandidate
from .inference import SplitPrediction
from .taxonomy import Taxonomy


def semantic_prediction_candidates(
    prediction: SplitPrediction,
    taxonomy: Taxonomy,
    *,
    semantic_parent: str,
    name_mapping: dict[str, str] | None = None,
    source: str = "hpid-split-net",
    source_reliability: float = 0.88,
    minimum_score: float = 0.20,
) -> list[MaskCandidate]:
    """Adapt a learned semantic prediction to the model-independent fusion API."""
    mapping = name_mapping or {}
    fine_lookup = {name: index for index, name in enumerate(taxonomy.fine_names)}
    candidates: list[MaskCandidate] = []
    for record in prediction.instances:
        semantic_name = mapping.get(record.semantic_name, record.semantic_name)
        mask = prediction.instance_map == record.instance_index
        class_id = fine_lookup[record.semantic_name]
        if not mask.any():
            continue
        score = float(np.mean(prediction.fine_probabilities[class_id][mask]))
        if score < minimum_score:
            continue
        candidates.append(
            MaskCandidate(
                semantic_name=semantic_name,
                semantic_parent=semantic_parent,
                mask=mask,
                score=score,
                source=source,
                source_reliability=source_reliability,
                metadata={
                    "model_part_id": record.part_id,
                    "model_semantic_name": record.semantic_name,
                },
            )
        )
    return candidates
