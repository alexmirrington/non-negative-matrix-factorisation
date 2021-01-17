"""Module containing implementation of metrics to evaluate NMF performance."""
from collections import Counter

import numpy as np
import sklearn.cluster
import sklearn.metrics


def relative_reconstruction_error(clean_data: np.ndarray, reconstructed: np.ndarray) -> float:
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


def assign_cluster_labels(features: np.ndarray, Y_true: np.ndarray) -> np.ndarray:
    """Assign cluster labels to input_data using k means algorithm.

    Args
    ---
    features: Array of features to cluster. Shape: (n_samples, n_features).
                For NMF evaluation, this is typically H.T, the transpose of the compressed
                data features that the NMF model builds.
    Y_true: The gold labels. For NMF evaluation this will typically be the dataset labels.

    Returns
    ---
    An array of predicted cluster labels in range [0, n_clusters). Shape: (n_samples,)
    """
    kmeans = sklearn.cluster.KMeans(n_clusters=len(set(Y_true))).fit(features)
    Y_pred = np.zeros(Y_true.shape)
    for i in set(kmeans.labels_):
        ind = kmeans.labels_ == i
        Y_pred[ind] = Counter(Y_true[ind]).most_common(1)[0][0]  # assign label.
    return Y_pred
