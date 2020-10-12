"""This module implements NMF with an L2,1 objective function."""
import numpy as np
from .base import NMFAlgorithm

class L21NMF(NMFAlgorithm):
    """NMF model with objective function to minimise the L2,1 norm of the reconstruction.

    Objective is min ||X - WH||_{2,1} s.t. W is an elementwise positive d x k matrix and
    H is an elementwise k x n matrix.
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
        self.W = self._init_matrix((self.d, self.k))
        self.H = self._init_matrix((self.k, self.n))

    def fit(self, max_iter=100, tol=1e-4):
        """Update matrices W and H using the multiplicative update algorithm.

        The optimisation will stop after the maximum number of iterations.
        TODO: Figure out what a good way of thresholding is to stop before max_iters.
        """

        for iter in range(max_iter+1):
            # Update W
            self.W = self._update_W()
            # Update H
            self.H = self._update_H()

            if iter % 10 == 0:
                self.reconstruction_error = np.linalg.norm(self.X - (self.W @ self.H))
                print("Reconstruction error: ", self.reconstruction_error)

    def reconstruction_error(self):
        """Return the reconstruction error between the original data and reconstructed data."""
        return np.linalg.norm(self.X - (self.W @ self.H))

    def _calculate_diag(self):
        """Calculates the diagonal matrix used for the updates.

        This is given by:
        W_(ii)=1//sqrt(sum_(j=1)^(p)(X-WH)_(ji)^(2))=1//|x_(i)-Wh_(i)|.
        """
        diags = np.zeros(self.n)
        for i in range(self.n):
            diags[i] = 1 / np.linalg.norm(self.X[:, i] - self.W @ self.H[:, i])
        return np.diag(diags)

    def _update_W(self):
        """Update W with respect to the objective."""
        diag = self._calculate_diag()
        numerator = self.X @ diag @ self.H.T
        denominator = self.W @ self.H @ diag @ self.H.T
        return self.W * numerator / denominator

    def _update_H(self):
        """Update H with respect to the objective."""
        diag = self._calculate_diag()
        numerator = self.W.T @ self.X @ diag
        denominator = self.W.T @ self.W @ self.H @ diag
        return self.H * numerator / denominator
