"""Utilities for logging metrics to wandb."""
from typing import Any, Dict

import wandb

from .base import Logger


class WandbLogger(Logger):
    """Class that logs metrics to wandb."""

    def __init__(self, commit: bool = True):
        """Initialise a `WandbLogger` instance."""
        self.commit = commit

    def __call__(self, metrics: Dict[str, Any]) -> None:
        """Log a set of key-value pairs."""
        wandb.log(metrics, commit=self.commit)
