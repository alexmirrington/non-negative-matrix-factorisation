"""This module implements NMF with an L2,1 objective function.

The model was introduced in the following paper.

Deguang Kong, Chris Ding, and Heng Huang. 2011.
Robust nonnegative matrix factorization using L21-norm.
In Proceedings of the 20th ACM international conference on Information
and knowledge management (CIKM '11). Association for Computing Machinery,
New York, NY, USA, 673–682. DOI:https://doi.org/10.1145/2063576.2063676
"""
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
        self.W, self.H = self._init_matrices([(self.d, self.k), (self.k, self.n)])

    def fit(self, max_iter: int = 10000, tol: float = 1e-4, output_freq: int = 20):
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
        output_freq: How many iterations between printing reconstruction error.
                        If -1, do not print.
        """
        prev_error = self.abs_reconstruction_error(self.X)
        for iter in range(max_iter + 1):
            # Update W
            self._update_W()
            # Update R
            self._update_H()

            error = self.abs_reconstruction_error(self.X)

            if iter % output_freq == 0 and output_freq > 0:
                # Current workaround
                print("Reconstruction error: ", error)

            if (prev_error - error) / prev_error < tol:
                print(f"Finished convergence after {iter} iterations.")
                return
            prev_error = error
        print(f"Finished convergence after reaching the maximum number of iterations.")

    def reconstructed_data(self):
        """Return the reconstruction of the input data."""
        return self.W @ self.H

    def _calculate_diag(self) -> np.ndarray:
        """Calculate the diagonal matrix used for the updates.

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
        self.W = self.W * numerator / denominator

    def _update_H(self):
        """Update H with respect to the objective."""
        diag = self._calculate_diag()
        numerator = self.W.T @ self.X @ diag
        denominator = self.W.T @ self.W @ self.H @ diag
        self.H = self.H * numerator / denominator
