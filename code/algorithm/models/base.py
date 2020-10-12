"""This model contains a base class with shared methods amongst the different NMF methods."""
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

class NMFAlgorithm(ABC):
    """Base class for the different NMF methods."""

    def _init_matrix(self, shape: Tuple):
        """Initalise a matrix of the given size.


        TODO:
        Currently using random initialisation similar to sklearn random initialisation. It is
        worthwhile reading about some other reasonable techniques and implementing, since there
        seems to be some papers on the topic.

        Args
        ---
        shape: Shape of the output matrix.
        """
        scale = np.sqrt(self.X.mean() / self.k)
        return scale * np.random.rand(*shape)

    @abstractmethod
    def reconstruction_error(self):
        """Return the reconstruction error between the original data and reconstructed data."""
        return

    @abstractmethod
    def fit(self, max_iter: int, tol: float):
        """Update the dictionary and other matrices until convergence.

        The optimisation will stop after the maximum number of iterations, or when

        Args
        ---
        max_iter: Maximum number of iterations to run the update steps for
        tol: Tolerance to consider convergence has stopped
        """
        return
