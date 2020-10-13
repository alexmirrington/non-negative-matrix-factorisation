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
        self.W, self.H = self._init_matrix([(self.d, self.k), (self.k, self.n)])
        self.lam = lam


    def fit(self, max_iter: int = 10000, tol: float = 1e-4, output_freq: int = 20):
        """Update matrices W and H using the multiplicative update algorithm.

        The optimisation will stop after the maximum number of iterations or when
        ||WH^(k+1) - WH^(k)|| / ||WH^(k)|| < tol.

        This stopping criterion is used since (previous_err - current_err)/previous_err < tol
        is not a meaningful measure of the convergence of WH, given that S is by definition set to
        approximate the difference between X and WH.

        The idea for this stopping criterion comes from the following paper, which applies it to
        a different model.

        A. B. Hamza and D. J. Brady, "Reconstruction of reflectance spectra using robust
        nonnegative matrix factorization," in IEEE Transactions on Signal Processing,
        vol. 54, no. 9, pp. 3637-3642, Sept. 2006, doi: 10.1109/TSP.2006.879282.

        Args
        ---
        max_iter: Maximum number of iterations to run.
        tol: Tolerance for minimum relative change in error before stopping convergence.
        output_freq: How many iterations between printing reconstruction error.
                        If -1, do not print.
        """
        prev_WH = self.W @ self.H
        for iter in range(max_iter+1):
            # Update S
            self._update_S()
            # Update W
            self._update_W()
            # Update H
            self._update_H()

            # Normalise W and H
            self._normalise_W()
            self._normalise_H()

            new_WH = self.W @ self.H
            error = self.abs_reconstruction_error(self.X)

            if iter % output_freq == 0 and output_freq > 0:
                # Current workaround
                print("Reconstruction error (including S approximating noise): ", error)

            if np.linalg.norm(new_WH - prev_WH)/np.linalg.norm(prev_WH) < tol:
                print(f"Finished convergence after {iter} iterations.")
                return
            prev_WH = new_WH
        print(f"Finished convergence after reaching the maximum number of iterations.")


    def reconstructed_data(self) -> np.ndarray:
        """Return the reconstruction of the input data."""
        return self.W @ self.H + self.S


    def _update_W(self) -> np.ndarray:
        """Update W with respect to the objective."""
        numerator = np.abs((self.S - self.X) @ self.H.T) - ((self.S - self.X) @ self.H.T)
        denominator = 2 * self.W @ self.H @ self.H.T
        self.W = self.W * numerator / denominator


    def _update_H(self) -> np.ndarray:
        """Update H with respect to the objective."""
        numerator = np.abs(self.W.T @ (self.S - self.X)) - (self.W.T @ (self.S - self.X))
        denominator = 2 * self.W.T @ self.W @ self.H
        self.H = self.H * numerator / denominator

    def _update_S(self):
        """Update S with respect to the objective."""
        new_S = self.X - self.W @ self.H

        #print((new_S <= self.lam/2) & (new_S >= -self.lam/2))
        new_S = np.where((new_S <= self.lam/2) & (new_S >= -self.lam/2), 0, new_S)
        new_S = np.where(new_S > self.lam/2, new_S - self.lam/2, new_S)
        new_S = np.where(new_S < -self.lam/2, new_S + self.lam/2, new_S)
        self.S = new_S

    def _normalise_W(self):
        """Normalise W as W_{ij} = W_{ij}/sqrt(sum_k(W_{kj}^2))."""
        # Denominator is matrix same size as W but with the column norm value
        # repeated at every element in same column.
        divisor = np.sqrt(np.sum(self.W**2, axis=0)) * np.ones(self.W.shape)
        self.W =  self.W / divisor

    def _normalise_H(self):
        """Normalise H as H_{ij} = H_{ij} * sqrt(sum_k(W_{ki}^2)).

        All elements in ith row of H get multiplied by the ith column norm of W.
        """
        multiplier = np.sqrt(np.sum(self.W**2, axis=0))[:, np.newaxis] * np.ones(self.H.shape)
        self.H = self.H * multiplier
