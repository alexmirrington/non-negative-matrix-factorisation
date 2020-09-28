"""Module containing implementation of metrics to evaluate NMF performance."""
import numpy as np
import sklearn.metrics


def relative_reconstruction_error(V_hat, W, H):
    """Calculate relative reconstruction error RRE=||V_hat-WH||_F/||V_hat||_F.

    Args
    ---
    V_hat: The clean data. Shape: (n_pixels, n_samples).
    W: The 'dictionary'. Shape: (n_pixels, n_components).
    H: The 'components'. Shape: (n_components, n_samples).

    W and H are the factorisation results from NMF, whose product approximates the original data.

    Returns
    ---
    The RRE value as a float.
    """
    return np.linalg.norm(V_hat - W.dot(H)) / np.linalg.norm(V_hat)


def average_accuracy(Y_hat, Y_pred):
    """Evaluate the average accuracy of clustering results using sklearn.

    Args
    ---
    Y_hat: The ground truth labels.
    Y_pred: The predicted label from the clustering.

    Returns
    ---
    Average accuracy as a float.
    """
    return sklearn.metrics.accuracy_score(Y, Y_pred)


def normalised_mutual_info(Y_hat, Y_pred):
    """Evaluate normalised mutual info of clustering results using sklearn.

    Args
    ---
    Y_hat: The ground truth labels.
    Y_pred: The predicted label from the clustering.

    Returns
    ---
    NMI as a float.
    """
    return sklearn.metrics.normalized_mutual_info_score(Y_hat, Y_pred)
