"""Configuration options and utilities for models."""
from enum import Enum


class NMFModel(Enum):
    """Enum outlining compatible NMF model variants."""

    STANDARD = "standard"
    HYPERSURFACE = "hypersurface"
    L2_1 = "l2_1"
    L1_ROBUST = "l1_robust"
