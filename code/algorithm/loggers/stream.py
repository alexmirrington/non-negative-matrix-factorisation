"""Utilities for logging metrics to stdout."""
import io
import sys
from typing import Any, Dict

from termcolor import colored

from .base import Logger


class StreamLogger(Logger):
    """Class that logs metrics to a file in JSON lines format."""

    def __init__(
        self, stream: io.IOBase = sys.stdout, coloured: bool = True, newline: bool = True
    ) -> None:
        """Initialise a `StreamLogger` instance."""
        self._stream = stream
        self.coloured = coloured
        self.newline = newline

    def __call__(self, metrics: Dict[str, Any]) -> None:
        """Log a set of key-value pairs."""
        output = ""

        for idx, (key, value) in enumerate(metrics.items()):
            color = None
            if self.coloured:
                color = "cyan"
                if "iteration" in key:
                    color = "magenta"
            if isinstance(value, float):
                value = f"{value:.4f}"
            valstr = colored(value, color=color) if color else value
            output += f"{key}: {valstr} "
        output = output.rstrip()
        if not self.newline:
            output = f"{output}\r"
        print(output, end="\n" if self.newline else "")
