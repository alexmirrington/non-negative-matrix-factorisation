"""Utilities for logging metrics to a file in JSON lines format."""
import json
from pathlib import Path
from typing import Any, Dict

from .base import Logger


class JSONLLogger(Logger):
    """Class that logs metrics to a file in JSON lines format."""

    def __init__(self, path: Path) -> None:
        """Initialise a `JSONLLogger` instance."""
        self._path = path

    @property
    def path(self) -> Path:
        """Get the path of the file that is being written to."""
        return self._path

    def __call__(self, metrics: Dict[str, Any]) -> None:
        """Log a set of key-value pairs."""
        with open(self._path, "a") as f:
            f.write(json.dumps(metrics) + "\n")
