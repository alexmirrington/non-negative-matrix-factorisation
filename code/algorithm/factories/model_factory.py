"""Classes to aid model creation given config parameters."""
import argparse

import numpy as np
from config import Model
from models import L21NMF, HypersurfaceNMF, L1RobustNMF, StandardNMF
from models.base import NMFAlgorithm


class ModelFactory:
    """Factory class to aid model creation given config parameters."""

    def __init__(self):
        """Initialise a `ModelFactory` instance."""
        self._factory_methods = {
            Model.STANDARD: ModelFactory._create_standard_nmf,
            Model.HYPERSURFACE: ModelFactory._create_hypersurface_nmf,
            Model.L21: ModelFactory._create_l21_nmf,
            Model.L1_ROBUST: ModelFactory._create_l1_robust_nmf,
        }

    def create(self, data: np.ndarray, config: argparse.Namespace) -> NMFAlgorithm:
        """Create a model from a dataset and config."""
        method = self._factory_methods.get(config.model)
        if method is None:
            raise NotImplementedError()
        return method(data, config)

    @staticmethod
    def _create_standard_nmf(data: np.ndarray, config: argparse.Namespace) -> NMFAlgorithm:
        return StandardNMF(data, n_components=config.components)

    @staticmethod
    def _create_hypersurface_nmf(data: np.ndarray, config: argparse.Namespace) -> NMFAlgorithm:
        return HypersurfaceNMF(data, n_components=config.components)

    @staticmethod
    def _create_l21_nmf(data: np.ndarray, config: argparse.Namespace) -> NMFAlgorithm:
        return L21NMF(data, n_components=config.components)

    @staticmethod
    def _create_l1_robust_nmf(data: np.ndarray, config: argparse.Namespace) -> NMFAlgorithm:
        # lam=0.3 is recommended in the paper.
        return L1RobustNMF(data, n_components=config.components, lam=0.3)
