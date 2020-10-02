"""This module implements the Multiplicative Update Rule NMF algorithm."""
from typing import Tuple
import numpy as np


class MultiplicativeUpdateNMF:
    """NMF model with optimisation using Multiplicative Update algorithm.

    Objective is min ||X-DR||_F^2 where D is an elementwise positive d x k matrix, and R is an
    elementwise k x n matrix.
    """

    def __init__(self, input_data, n_components: int):
        """Initialise the NMF model.

        Args
        ---
        orig_data: The original data to be decomposed. Shape (n_features, n_samples).
        n_components: Number of components in the decomposed data.
        """
        self.X = input_data
        self.d = input_data.shape[0]
        self.n = input_data.shape[1]
        self.k = n_components
        self.D = self._init_matrix((self.d, self.k))
        self.R = self._init_matrix((self.k, self.n))

        self.reconstruction_error = np.linalg.norm(self.X - (self.D @ self.R))
        print("Reconstruction error: ", self.reconstruction_error)


    def fit(self, max_iter=100, tol=1e-4):
        """Update matrices D and R using the multiplicative update algorithm.

        The optimisation will stop after the maximum number of iterations.
        TODO: Figure out what a good way of thresholding is to stop before max_iters.
        """

        for iter in range(max_iter+1):
            # Update D
            self.D = self._new_D()
            # Update R
            self.R = self._new_R()

            if iter % 10 == 0:
                self.reconstruction_error = np.linalg.norm(self.X - (self.D @ self.R))
                print("Reconstruction error: ", self.reconstruction_error)

    def _new_R(self):
        """Update R with respect to the objective.

        R_(i,j)^(k+1)=R_(i,j)^(k)((D^(k^(TT))X)_(i,j))/((D^(k^(TT))D^(k)R^(k))_(i,j))
        """
        numerator = self.D.T @ self.X
        denominator = self.D.T @ self.D @ self.R
        return self.R * (numerator/denominator)

    def _new_D(self):
        """Update D with respect to the objective.

        D_(i,j)^(k+1)=D_(i,j)^(k)((XR^(k+1^(TT)))_(i,j))/((D^(k)R^(k+1)R^(k+1^(TT)))_(i,j))
        """
        numerator = self.X @ self.R.T
        denominator = self.D @ self.R @ self.R.T
        return self.D * (numerator/denominator)

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
