"""This module implements the Multiplicative Update Rule NMF algorithm."""
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
        self.W, self.H = self._init_matrix([(self.d, self.k), (self.k, self.n)])


    def reconstructed_data(self):
        """Return the reconstruction of the input data."""
        return self.W @ self.H

    def fit(self, max_iter=100, tol=1e-4):
        """Update matrices W and H using the multiplicative update algorithm.

        The optimisation will stop after the maximum number of iterations.
        TODO: Figure out what a good way of thresholding is to stop before max_iters.
        """

        for iter in range(max_iter+1):
            # Update W
            self.W = self._update_W()
            # Update R
            self.H = self._update_H()

            if iter % 10 == 0:
                # Current workaround
                print("Reconstruction error: ", self.abs_reconstruction_error(target=self.X))

    def _update_H(self):
        """Update H with respect to the objective.

        H_(i,j)^(k+1)=H_(i,j)^(k)((W^(k^(TT))X)_(i,j))/((W^(k^(TT))W^(k)H^(k))_(i,j))
        """
        numerator = self.W.T @ self.X
        denominator = self.W.T @ self.W @ self.H
        return self.H * (numerator/denominator)

    def _update_W(self):
        """Update W with respect to the objective.

        W_(i,j)^(k+1)=W_(i,j)^(k)((XH^(k+1^(TT)))_(i,j))/((W^(k)H^(k+1)H^(k+1^(TT)))_(i,j))
        """
        numerator = self.X @ self.H.T
        denominator = self.W @ self.H @ self.H.T
        return self.W * (numerator/denominator)
