"""Configuration options and utilities for noise functions."""
from enum import Enum


class Noise(Enum):
    """Enum outlining compatible noise function variants."""

    UNIFORM = "uniform"
    GAUSSIAN = "gaussian"
    SALT_AND_PEPPER = "salt_and_pepper"
    MISSING_BLOCK = "missing_block"
