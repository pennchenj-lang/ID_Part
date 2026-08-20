from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.segmentation import (
    LRASPP_MobileNet_V3_Large_Weights,
    lraspp_mobilenet_v3_large,
)

from .taxonomy import Taxonomy


class SharedLRASPPHeads(nn.Module):
    """Shared LR-ASPP projection with fine, parent, and boundary heads."""

    def __init__(
        self, low_channels: int, high_channels: int, taxonomy: Taxonomy
    ) -> None:
        super().__init__()
        hidden = 128
        self.high_projection = nn.Sequential(
            nn.Conv2d(high_channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.high_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, hidden, 1, bias=False),
            nn.Sigmoid(),
        )
        self.fine_low = nn.Conv2d(low_channels, taxonomy.num_fine_classes, 1)
        self.fine_high = nn.Conv2d(hidden, taxonomy.num_fine_classes, 1)
        self.parent_low = nn.Conv2d(low_channels, taxonomy.num_parent_classes, 1)
        self.parent_high = nn.Conv2d(hidden, taxonomy.num_parent_classes, 1)
        self.boundary_low = nn.Sequential(
            nn.Conv2d(low_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
        )
        self.boundary_high = nn.Conv2d(hidden, 1, 1)

    def forward(self, features: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        low = features["low"]
        high_raw = features["high"]
        high = self.high_projection(high_raw) * self.high_gate(high_raw)
        high = F.interpolate(
            high, size=low.shape[-2:], mode="bilinear", align_corners=False
        )
        return {
            "fine": self.fine_low(low) + self.fine_high(high),
            "parent": self.parent_low(low) + self.parent_high(high),
            "boundary": self.boundary_low(low) + self.boundary_high(high),
        }


class HPIDSplitNet(nn.Module):
    """Hierarchy-constrained network for fine-part decomposition.

    The model never accepts annotations in ``forward``. Ground truth is consumed
    only by the external loss function during training.
    """

    def __init__(self, taxonomy: Taxonomy, pretrained_backbone: bool = True) -> None:
        super().__init__()
        weights = (
            LRASPP_MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
        )
        base = lraspp_mobilenet_v3_large(weights=weights)
        self.backbone = base.backbone
        self.heads = SharedLRASPPHeads(40, 960, taxonomy)
        self.taxonomy = taxonomy

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        spatial_size = images.shape[-2:]
        outputs = self.heads(self.backbone(images))
        return {
            name: F.interpolate(
                value, size=spatial_size, mode="bilinear", align_corners=False
            )
            for name, value in outputs.items()
        }
