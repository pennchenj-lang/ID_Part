"""HPID-Split public API."""

from .fusion import FusionConfig, FusionResult, MaskCandidate, fuse_candidates
from .instances import PartInstance, semantic_to_part_ids
from .model import HPIDSplitNet
from .relational import (
    RelationalAppearanceConfig,
    RelationalCandidateGeneration,
    propose_relational_candidates,
)
from .taxonomy import Taxonomy, default_character_taxonomy

__all__ = [
    "FusionConfig",
    "FusionResult",
    "HPIDSplitNet",
    "MaskCandidate",
    "PartInstance",
    "RelationalAppearanceConfig",
    "RelationalCandidateGeneration",
    "Taxonomy",
    "default_character_taxonomy",
    "fuse_candidates",
    "propose_relational_candidates",
    "semantic_to_part_ids",
]

__version__ = "0.3.8"
