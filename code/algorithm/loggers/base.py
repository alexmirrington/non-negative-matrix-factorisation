"""Abstract base classes for metric logging."""
from abc import ABC, abstractmethod
from typing import Any, Dict


class Logger(ABC):
    """Abstract base class for all metric loggers."""

    @abstractmethod
    def __call__(self, metrics: Dict[str, Any]) -> None:
        """Log a set of key-value pairs."""
