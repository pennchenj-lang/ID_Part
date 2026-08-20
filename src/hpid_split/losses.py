from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .taxonomy import Taxonomy


@dataclass(frozen=True)
class LossWeights:
    fine_ce: float = 1.0
    parent_ce: float = 0.45
    detail_tversky: float = 0.55
    boundary_bce: float = 0.25
    hierarchy: float = 0.20


def label_boundary(labels: torch.Tensor) -> torch.Tensor:
    boundary = torch.zeros_like(labels, dtype=torch.bool)
    boundary[:, 1:, :] |= labels[:, 1:, :] != labels[:, :-1, :]
    boundary[:, :-1, :] |= labels[:, 1:, :] != labels[:, :-1, :]
    boundary[:, :, 1:] |= labels[:, :, 1:] != labels[:, :, :-1]
    boundary[:, :, :-1] |= labels[:, :, 1:] != labels[:, :, :-1]
    return boundary & (labels != 0)


def detail_tversky_loss(
    fine_logits: torch.Tensor,
    targets: torch.Tensor,
    detail_ids: tuple[int, ...],
    alpha: float = 0.35,
    beta: float = 0.65,
) -> torch.Tensor:
    probabilities = fine_logits.softmax(dim=1)
    losses: list[torch.Tensor] = []
    for class_id in detail_ids:
        truth = (targets == class_id).float()
        if not torch.any(truth):
            continue
        prediction = probabilities[:, class_id]
        true_positive = (prediction * truth).sum(dim=(1, 2))
        false_positive = (prediction * (1.0 - truth)).sum(dim=(1, 2))
        false_negative = ((1.0 - prediction) * truth).sum(dim=(1, 2))
        score = (true_positive + 1.0) / (
            true_positive + alpha * false_positive + beta * false_negative + 1.0
        )
        losses.append(1.0 - score.mean())
    if not losses:
        return fine_logits.sum() * 0.0
    return torch.stack(losses).mean()


def aggregate_fine_probabilities(
    fine_probabilities: torch.Tensor, taxonomy: Taxonomy
) -> torch.Tensor:
    output = fine_probabilities.new_zeros(
        fine_probabilities.shape[0],
        taxonomy.num_parent_classes,
        fine_probabilities.shape[2],
        fine_probabilities.shape[3],
    )
    for fine_id, parent_id in enumerate(taxonomy.fine_to_parent):
        output[:, parent_id] += fine_probabilities[:, fine_id]
    return output.clamp_min(1e-7)


def hpid_split_loss(
    outputs: dict[str, torch.Tensor],
    fine_target: torch.Tensor,
    taxonomy: Taxonomy,
    fine_class_weights: torch.Tensor | None = None,
    parent_class_weights: torch.Tensor | None = None,
    weights: LossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = weights or LossWeights()
    mapping = torch.as_tensor(taxonomy.fine_to_parent, device=fine_target.device)
    parent_target = mapping[fine_target]
    fine_ce = F.cross_entropy(outputs["fine"], fine_target, weight=fine_class_weights)
    parent_ce = F.cross_entropy(
        outputs["parent"], parent_target, weight=parent_class_weights
    )
    detail = detail_tversky_loss(outputs["fine"], fine_target, taxonomy.detail_ids)

    boundary_target = label_boundary(fine_target).float()
    positive = boundary_target.sum()
    negative = boundary_target.numel() - positive
    positive_weight = (negative / positive.clamp_min(1.0)).clamp(1.0, 25.0)
    boundary = F.binary_cross_entropy_with_logits(
        outputs["boundary"][:, 0],
        boundary_target,
        pos_weight=positive_weight,
    )

    fine_parent = aggregate_fine_probabilities(outputs["fine"].softmax(dim=1), taxonomy)
    parent_probabilities = outputs["parent"].softmax(dim=1).clamp_min(1e-7)
    hierarchy = (
        0.5
        * (
            F.kl_div(fine_parent.log(), parent_probabilities, reduction="batchmean")
            + F.kl_div(parent_probabilities.log(), fine_parent, reduction="batchmean")
        )
        / (fine_target.shape[-1] * fine_target.shape[-2])
    )

    total = (
        weights.fine_ce * fine_ce
        + weights.parent_ce * parent_ce
        + weights.detail_tversky * detail
        + weights.boundary_bce * boundary
        + weights.hierarchy * hierarchy
    )
    components = {
        "loss": float(total.detach()),
        "fine_ce": float(fine_ce.detach()),
        "parent_ce": float(parent_ce.detach()),
        "detail_tversky": float(detail.detach()),
        "boundary_bce": float(boundary.detach()),
        "hierarchy": float(hierarchy.detach()),
    }
    return total, components
