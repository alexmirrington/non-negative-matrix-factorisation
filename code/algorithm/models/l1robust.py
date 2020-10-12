"""This module implements NMF with a regularised, robust L1-norm objective.

The model was introduced in the following paper.

Zhang, L., Chen, Z., Zheng, M. et al. Robust non-negative matrix factorization.
Front. Electr. Electron. Eng. China 6, 192–200 (2011).
https://doi.org/10.1007/s11460-011-0128-0
"""
import numpy as np
from .base import NMFAlgorithm

class L1RobustNMF(NMFAlgorithm):
    """NMF algorithm with L1 norm based objective with regularisation.

    The original data is modelled as WH + S, where S models error as a sparse matrix.
    The objective is min_(W,H,S)|X-WH-S|_(F)^(2)+lambda|S|_(1), where W is an elementwise
    positive d x k matrix and H is an elementwise positive k x n matrix.
    """

    def __init__(self, input_data: np.ndarray, n_components: int, lam: float):
        """Initialise the NMF model.

        Args
        ---
        input_data: The original data to be decomposed. Shape (n_features, n_samples).
        n_components: Number of components in the decomposed data.
        lambda: The parameter that controls regularisation.
        """
        super().__init__()
        self.X = input_data
        self.d = input_data.shape[0]
        self.n = input_data.shape[1]
        self.k = n_components
        self.W = self._init_matrix((self.d, self.k))
        self.H = self._init_matrix((self.k, self.n))
        self.lam = lam


    def fit(self, max_iter=100, tol=1e-4):
        """Update matrices D and R using the multiplicative update algorithm.

        The optimisation will stop after the maximum number of iterations.
        TODO: Figure out what a good way of thresholding is to stop before max_iters.
        """

        for iter in range(max_iter+1):
            # Update S
            self.S = self._update_S()
            # Update W
            self.W = self._update_W()
            # Update H
            self.H = self._update_H()

            # Normalise W and H
            self.W = self._normalise_W()
            self.H = self._normalise_H()

            if iter % 10 == 0:
                reconstruction_error = np.linalg.norm(self.X - (self.W @ self.H + self.S))
                print("Reconstruction error: ", reconstruction_error)

    def reconstruction_error(self):
        """Return the reconstruction error between the original data and reconstructed data."""
        return np.linalg.norm(self.X - (self.W @ self.H + self.S))


    def _update_W(self):
        """Update W with respect to the objective."""

        numerator = np.abs((self.S - self.X) @ self.H.T) - ((self.S - self.X) @ self.H.T)
        denominator = 2 * self.W @ self.H @ self.H.T
        return self.W * numerator / denominator


    def _update_H(self):
        """Update H with respect to the objective."""
        numerator = np.abs(self.W.T @ (self.S - self.X)) - (self.W.T @ (self.S - self.X))
        denominator = 2 * self.W.T @ self.W @ self.H
        return self.H * numerator / denominator

    def _update_S(self):
        """Update S with respect to the objective."""
        new_S = self.X - self.W @ self.H

        #print((new_S <= self.lam/2) & (new_S >= -self.lam/2))
        new_S = np.where((new_S <= self.lam/2) & (new_S >= -self.lam/2), 0, new_S)
        new_S = np.where(new_S > self.lam/2, new_S - self.lam/2, new_S)
        new_S = np.where(new_S < -self.lam/2, new_S + self.lam/2, new_S)
        return new_S

    def _normalise_W(self):
        """Normalise W as W_{ij} = W_{ij}/sqrt(sum_k(W_{kj}^2))."""
        # Denominator is matrix same size as W but with the column norm value
        # repeated at every element in same column.
        divisor = np.sqrt(np.sum(self.W**2, axis=0)) * np.ones(self.W.shape)
        return self.W / divisor

    def _normalise_H(self):
        """Normalise H as H_{ij} = H_{ij} * sqrt(sum_k(W_{ki}^2)).

        All elements in ith row of H get multiplied by the ith column norm of W.
        """
        multiplier = np.sqrt(np.sum(self.W**2, axis=0))[:, np.newaxis] * np.ones(self.H.shape)
        return self.H * multiplier
