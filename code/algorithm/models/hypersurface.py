"""This module implements NMF with a hypersurface cost objective function.

The model was introduced in the following paper.

A. B. Hamza and D. J. Brady, "Reconstruction of reflectance spectra using robust
nonnegative matrix factorization," in IEEE Transactions on Signal Processing,
vol. 54, no. 9, pp. 3637-3642, Sept. 2006, doi: 10.1109/TSP.2006.879282.
"""
from typing import Any, Callable, Dict, Iterable, Optional

import numpy as np

from .base import NMFAlgorithm


class HypersurfaceNMF(NMFAlgorithm):
    r"""NMF model with iterative optimisation of hypersurface cost function.

    The objective is min :math:`\sqrt(1 + ||X - WH|| ^ 2) - 1` where :math:`W`
    is an elementwise positive :math:`d \times k` matrix and H is an elementwise
    positive :math:`k \times n` matrix.
    """

    def __init__(self, input_data: np.ndarray, n_components: int):
        """Initialise the NMF model.

        Args
        ---
        input_data: The original data to be decomposed. Shape (n_features, n_samples
        n_components: Number of components in the decomposed data
        """
        super().__init__()
        self.X = input_data
        self.d = input_data.shape[0]
        self.n = input_data.shape[1]
        self.k = n_components
        self.W, self.H = self._init_matrices([(self.d, self.k), (self.k, self.n)])

    def reconstructed_data(self):
        """Return the reconstruction of the input data."""
        return self.W @ self.H

    def reconstruction_error(self):
        """Return the reconstruction error between the original data and reconstructed data."""
        return np.linalg.norm(self.X - (self.W @ self.H))

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
        print("Converged after reaching the maximum number of iterations.")

    def _update_W(self):
        """Update W with respect to the objctive.

        The update rule is
        W_(ik)^((t+1)) = W_(ik)^((t)) -
                         alpha_(ik)^((t)) *
                         ((WHH^(T))_(ik)^((t))-(WH^(T))_(ik)^((t))) /
                         (sqrt(1+|X-WH|))

        The step size alpha is found by Armijo search.
        """
        alpha = None
        numerator = (self.W @ self.H @ self.H.T) - (self.X @ self.H.T)
        denominator = np.sqrt(1 + np.linalg.norm(self.X - (self.W @ self.H)))
        deriv = numerator / denominator
        alpha = self._get_armijo_coeff_W(deriv)
        self.W = self.W - alpha * deriv

    def _update_H(self):
        """Update H with respect to the objective.

        The update rule is
        H_(kj)^((t+1))=H_(kj)^((t)) -
                       beta_(kj)^((t)) *
                       ((W^(T)WH)_(kj)^((t))-(W^(T)X)_(kj)^((t))) /
                       (sqrt(1+|X-WH|))

        The step size beta is found by Armijo search.
        """
        numerator = (self.W.T @ self.W @ self.H) - (self.W.T @ self.X)
        denominator = np.sqrt(1 + np.linalg.norm(self.X - (self.W @ self.H)))
        deriv = numerator / denominator
        beta = self._get_armijo_coeff_H(deriv)
        self.H = self.H - beta * deriv

    # TODO: CURRENTLY ARMIJO STEP SIZE IS SAME FOR ALL INDICES OF THE MATRIX!
    # I'm having trouble figuring out how to do the Armijo line search. In the paper there
    # is an alpha and beta value for each (i, j) of the matrix, so we must do it per element if
    # they are not meant to be all the same.
    # However I'm unsure if this means we evaluate the objective function in 1D also or not. If not
    # I can't see how one wouldn't end up with the same learning rate for all anyway.

    # Currently its not converging before underflow error.

    # I have implemented it as best I could extract from the lecture slides, but I couldn't
    # really understand how to apply the Armijo search in the reference the paper makes to
    # 'Iterative Methods for Linear and Nonlinear Equations'.

    # Would love a hand here if any ideas.

    def _get_armijo_coeff_W(
        self,
        deriv: np.ndarray,
        initial_step_size: float = 0.05,
        alpha: float = 0.5,
        beta: float = 0.5,
    ):
        """Find the armijo coefficient for updating W in the current iteration."""
        step_size = initial_step_size  # Initial step size value
        while not self._check_armijo_criterion_W(deriv, step_size, alpha):
            step_size = beta * step_size
        return step_size

    def _check_armijo_criterion_W(self, deriv: np.ndarray, step_size: float, alpha: float):
        """Check whether the armijo criterion is satisfied for the proposed update to W."""
        obj_val = 0.5 * (np.sqrt(1 + np.linalg.norm(self.X - self.W @ self.H)) - 1)
        proposed_W = self.W - step_size * deriv
        proposed_obj_val = 0.5 * (np.sqrt(1 + np.linalg.norm(self.X - proposed_W @ self.H)) - 1)

        if proposed_obj_val - obj_val <= -step_size * alpha * np.linalg.norm(deriv) ** 2:
            return True
        return False

    def _get_armijo_coeff_H(
        self, deriv: np.ndarray, initial_step_size: float = 1, alpha: float = 0.5, beta: float = 0.5
    ):
        """Find the armijo coefficient for updating H in the current iteration."""
        step_size = initial_step_size  # Initial step size value
        while not self._check_armijo_criterion_H(deriv, step_size, alpha):
            step_size = beta * step_size
        return step_size

    def _check_armijo_criterion_H(self, deriv: np.ndarray, step_size: float, alpha: float):
        """Check whether the armijo criterion is satisfied for the proposed update to H."""
        obj_val = 0.5 * (np.sqrt(1 + np.linalg.norm(self.X - self.W @ self.H)) - 1)
        proposed_H = self.H - step_size * deriv
        proposed_obj_val = 0.5 * (np.sqrt(1 + np.linalg.norm(self.X - self.W @ proposed_H)) - 1)

        if proposed_obj_val - obj_val <= -step_size * alpha * np.linalg.norm(deriv) ** 2:
            return True
        return False
