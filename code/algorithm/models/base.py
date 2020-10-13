"""This model contains a base class with shared methods amongst the different NMF methods."""
from abc import ABC, abstractmethod
from typing import Tuple, List
import numpy as np

class NMFAlgorithm(ABC):
    """Base class with shared functionality for the different NMF methods."""

    def _init_matrix(self, shapes: List[Tuple], init_type: str = 'nndsvdar') -> Tuple[np.ndarray]:
        """Initalise a matrix of the given size.

        Acknowledgements
        ---
        The initialisation options are inspired by those available in sklearn, along with
        several papers on the topic.

        -
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html
        -
        was used as inspiration for the 'random' option.

        -
        C. Boutsidis, E. Gallopoulos,
        SVD based initialization: A head start for nonnegative matrix factorization,
        Pattern Recognition,
        Volume 41, Issue 4,
        2008,
        Pages 1350-1362,
        ISSN 0031-3203,
        https://doi.org/10.1016/j.patcog.2007.09.010.
        -
        used for the 'nndsvda' and 'nndsvdar' option,
        particularly Table 1 algorithm and the discussion from Section 2.3 for avoiding zeros.


        Args
        ---
        shape: Shape of the output matrix.
        init_type: Type of initialisation.
                        Options
                        ---
                        'random': Initialise with a random uniform distribution scaled to
                                    sqrt(input_data.mean() / n_components).
                        'nndsvda': Initialise using the SVD-based method from
                                    https://doi.org/10.1016/j.patcog.2007.09.010,
                                    avoiding any zeros by adding average of the input_data
                                    where necessary.
                        'nndsvdar': Initialise using the SVD-based method from
                                    https://doi.org/10.1016/j.patcog.2007.09.010,
                                    avoiding any zeros by adding uniform random number in
                                    range [0, mean(input_data)/100].
        """
        # Sanity checks on calls from model __init__'s
        assert len(shapes) == 2
        assert shapes[0][1] == shapes[1][0]


        if init_type == 'random':
            scale = np.sqrt(self.X.mean() / self.k)
            return scale * np.random.rand(*shapes[0]), scale * np.random.rand(*shapes[1])

        elif init_type == 'nndsvda' or 'nndsvdar':
            if self.k >= self.n and self.k >= self.d:
                raise ValueError("Number of components must be less than min(n_features, n_samples)"
                                 " to use nndsvd. Use init_type='random' instead.")
            U, s, V = np.linalg.svd(self.X, full_matrices=False)
            # Keep only n_components largest singular triplets
            U, s, V = U[:, :self.k], s[:self.k], V[:self.k, :]

            W = np.zeros(shapes[0])
            H = np.zeros(shapes[1])

            def pos(in_arr: np.ndarray):
                """Keep only positive values, setting negative values to 0."""
                return np.where(in_arr >= 0, in_arr, 0)
            def neg(in_arr: np.ndarray):
                """Keep only the magnitude of negative values, setting positive values to 0."""
                return np.where(in_arr < 0, -in_arr, 0)

            # For details about the algorithm, including justification with several important
            # theorems, please see the original paper at
            # https://doi.org/10.1016/j.patcog.2007.09.010
            for j in range(0, self.k):
                x, y = U[:, j], V[j, :]

                xp, xn, yp, yn = pos(x), neg(x), pos(y), neg(y)

                xpnrm, ypnrm = np.linalg.norm(xp), np.linalg.norm(yp)
                mp = xpnrm * ypnrm

                xnnrm, ynnrm = np.linalg.norm(xn), np.linalg.norm(yn)
                mn = xnnrm * ynnrm

                if mp > mn:
                    u, v = xp / xpnrm, yp / ypnrm
                    sigma = mp

                else:
                    u, v = xn / xnnrm, yn / ynnrm
                    sigma = mn

                W[:, j] = np.sqrt(s[j] * sigma) * u
                H[j, :] = np.sqrt(s[j] * sigma) * v

            X_mean = np.abs(self.X.mean())

            if init_type == 'nndsvda':
                # Avoid any zeros in initialisation by adding mean of input data
                W = np.where(W == 0, X_mean, W)
                H = np.where(H == 0, X_mean, H)

            elif init_type == 'nndsvdar':
                # Avoid any zeros by adding a random uniform number in [0, X.mean()/100]
                scale = X_mean / 100
                W = np.where(W == 0, np.random.rand()*scale, W)
                H = np.where(H == 0, np.random.rand()*scale, H)

            return W, H

        else:
            raise ValueError("init_type must be one of 'random', 'nndsvda', nndsvdar.")

    def abs_reconstruction_error(self, target: np.ndarray) -> float:
        """Return the Frobenius norm of the difference between target and reconstructed data."""
        return np.linalg.norm(target - self.reconstructed_data())

    @abstractmethod
    def reconstructed_data(self) -> np.ndarray:
        """Return the reconstruction of the input data."""
        return

    @abstractmethod
    def fit(self, max_iter: int, tol: float) -> None:
        """Update the dictionary and other matrices until convergence.

        The optimisation will stop after the maximum number of iterations, or when

        Args
        ---
        max_iter: Maximum number of iterations to run the update steps for
        tol: Tolerance to consider convergence has stopped
        """
        return
