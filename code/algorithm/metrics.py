"""Module containing implementation of metrics to evaluate NMF performance."""
import numpy as np
import sklearn.metrics


def relative_reconstruction_error(clean_data: np.ndarray,
                                  reconstructed: np.ndarray
                                  ) -> float:
    """Calculate relative reconstruction error RRE=||V_hat-WH||_F/||V_hat||_F.

    where V_hat is the uncorrupted data and WH is the reconstructed data from the NMF model.
    Note that for some models the reconstruction may be calculated with a different formula.

    Args
    ---
    clean_data: The clean data. Shape: (n_pixels, n_samples).
    reconstructed: The reconstructed data from NMF. Shape: (n_pixels, n_samples).

    Returns
    ---
    The RRE value as a float.
    """
    return np.linalg.norm(clean_data - reconstructed) / np.linalg.norm(clean_data)


def average_accuracy(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """Evaluate the average accuracy of clustering results using sklearn.

    Args
    ---
    Y_true: The ground truth labels.
    Y_pred: The predicted label from the clustering.

    Returns
    ---
    Average accuracy as a float.
    """
    return sklearn.metrics.accuracy_score(Y_true, Y_pred)


def normalised_mutual_info(Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
    """Evaluate normalised mutual info of clustering results using sklearn.

    Args
    ---
    Y_true: The ground truth labels.
    Y_pred: The predicted label from the clustering.

    Returns
    ---
    NMI as a float.
    """
    return sklearn.metrics.normalized_mutual_info_score(Y_true, Y_pred)
