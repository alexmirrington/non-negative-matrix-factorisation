"""Configuration options and utilities for datasets."""
from enum import Enum


class Dataset(Enum):
    """Enum outlining compatible dataset names."""

    ORL = "orl"
    YALEB = "yaleb"
