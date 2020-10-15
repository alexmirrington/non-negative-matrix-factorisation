"""Configuration options and utilities for models."""
from enum import Enum


class Model(Enum):
    """Enum outlining compatible NMF model variants."""

    STANDARD = "standard"
    HYPERSURFACE = "hypersurface"
    L21 = "l21"
    L1_ROBUST = "l1_robust"
