"""This module implements the Multiplicative Update Rule NMF algorithm.

The model is based on that introduced in the COMP5328 Advanced Machine Learning lectures.
"""
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np

from .base import NMFAlgorithm


class StandardNMF(NMFAlgorithm):
    """NMF model with optimisation using Multiplicative Update algorithm.

    Objective is min ||X-WH||_F^2 where W is an elementwise positive d x k matrix, and H is an
    elementwise positive k x n matrix.
    """

    def __init__(self, input_data: np.ndarray, n_components: int):
        """Initialise the NMF model.

        Args
        ---
        input_data: The original data to be decomposed. Shape (n_features, n_samples).
        n_components: Number of components in the decomposed data.
        """
        super().__init__()
        self.X = input_data
        self.d = input_data.shape[0]
        self.n = input_data.shape[1]
        self.k = n_components
        self.W, self.H = self._init_matrices([(self.d, self.k), (self.k, self.n)])

    def reconstructed_data(self) -> np.ndarray:
        """Return the reconstruction of the input data."""
        return self.W @ self.H

    def fit(
        self,
        max_iter: int = 10000,
        tol: float = 1e-4,
        callback_freq: int = 16,
        callbacks: Optional[Iterable[Callable[[Dict[str, Any]], Any]]] = None,
        clean_data: Optional[np.ndarray] = None,
        true_labels: Optional[np.ndarray] = None,
    ):
        """Update matrices W and H using the multiplicative update algorithm.

        The optimisation will stop after the maximum number of iterations or when the
        relative difference between the previous error and current error is less than threshold
        tolerance. That is, when (previous_err - current_err)/previous_err < tol.

        This stopping criterion is commonly applied to NMF algorithms, and the following paper
        and tutorial mention it as a simple convergence criterion or baseline.


        F. G. Germain and G. J. Mysore, "Stopping Criteria for Non-Negative Matrix Factorization
        Based Supervised and Semi-Supervised Source Separation," in IEEE Signal Processing Letters,
        vol. 21, no. 10, pp. 1284-1288, Oct. 2014, doi: 10.1109/LSP.2014.2331981.

        S. Essid and A. Ozerov, "A Tutorial on Nonnegative Matrix Factorisation with
        Applications to Audiovisual Content Analysis," Telecom ParisTech / Technicolor,
        July 2014, https://www.cs.rochester.edu/u/jliu/CSC-576/NMF-tutorial.pdf


        Args
        ---
        max_iter: Maximum number of iterations to run.
        tol: Tolerance for minimum relative change in error before stopping convergence.
        callback_freq: How many iterations between logging metrics. If -1, do not log.
        callbacks: A sequence of functions, each of which takes a dictionary of
            metrics as input, used for logging.
        clean_data: The clean data to use for calculating relative reconstruction error,
            not used for training. Shape: (n_features, n_samples)
        true_labels: The true data labels to evaluate clustering results,
            not used for training. Shape: (n_samples,)
        """
        prev_error = self.abs_reconstruction_error(self.X)
        for iter in range(max_iter):
            # Update W
            self._update_W()
            # Update R
            self._update_H()

            error = self.abs_reconstruction_error(self.X)

            if iter % callback_freq == callback_freq - 1 and callback_freq > 0:
                self._log(iter, error, clean_data, true_labels, callbacks)

            if (prev_error - error) / prev_error < tol:
                self._log(iter, error, clean_data, true_labels, callbacks)
                print(f"Converged after {iter + 1} iterations.")
                return
            prev_error = error
        self._log(iter, error, clean_data, true_labels, callbacks)
        print(f"Converged after reaching the maximum number of iterations ({max_iter}).")

    def _update_H(self) -> np.ndarray:
        """Update H with respect to the objective.

        H_(i,j)^(k+1)=H_(i,j)^(k)((W^(k^(TT))X)_(i,j))/((W^(k^(TT))W^(k)H^(k))_(i,j))
        """
        numerator = self.W.T @ self.X
        denominator = self.W.T @ self.W @ self.H
        self.H = self.H * (numerator / denominator)

    def _update_W(self) -> np.ndarray:
        """Update W with respect to the objective.

        W_(i,j)^(k+1)=W_(i,j)^(k)((XH^(k+1^(TT)))_(i,j))/((W^(k)H^(k+1)H^(k+1^(TT)))_(i,j))
        """
        numerator = self.X @ self.H.T
        denominator = self.W @ self.H @ self.H.T
        self.W = self.W * (numerator / denominator)
