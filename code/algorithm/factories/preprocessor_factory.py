"""Classes to aid model creation given config parameters."""
import argparse
from typing import Callable

import numpy as np
from preprocessing import gaussian, missing_block, salt_and_pepper, uniform

from config import Noise


class PreprocessorFactory:
    """Factory class to aid preprocessor creation given config parameters."""

    def __init__(self):
        """Initialise a `PreprocessorFactory` instance."""
        self._factory_methods = {
            Noise.SALT_AND_PEPPER: PreprocessorFactory._create_salt_and_pepper,
            Noise.MISSING_BLOCK: PreprocessorFactory._create_missing_block,
            Noise.UNIFORM: PreprocessorFactory._create_uniform,
            Noise.GAUSSIAN: PreprocessorFactory._create_gaussian,
        }

    def create(self, config: argparse.Namespace) -> Callable[[np.ndarray], np.ndarray]:
        """Create a model from a dataset and config."""
        method = self._factory_methods.get(config.noise)
        if method is None:
            return None
        return method(config)

    @staticmethod
    def _create_salt_and_pepper(
        config: argparse.Namespace,
    ) -> Callable[[np.ndarray], np.ndarray]:
        return lambda data: salt_and_pepper(data, config.scale, config.noise_p, config.noise_r)

    @staticmethod
    def _create_missing_block(
        config: argparse.Namespace,
    ) -> Callable[[np.ndarray], np.ndarray]:
        return lambda data: missing_block(data, config.noise_blocksize, num_blocks=config.noise_blocks, fill=config.scale)

    @staticmethod
    def _create_uniform(
        config: argparse.Namespace,
    ) -> Callable[[np.ndarray], np.ndarray]:
        return lambda data: uniform(data, mean=config.noise_mean, std=config.noise_std, min=0, max=config.scale)

    @staticmethod
    def _create_gaussian(config: argparse.Namespace) -> Callable[[np.ndarray], np.ndarray]:
        return lambda data: gaussian(data, mean=config.noise_mean, std=config.noise_std, min=0, max=config.scale)
