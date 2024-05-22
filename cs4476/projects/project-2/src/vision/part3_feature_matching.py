import numpy as np

from typing import Tuple


def compute_feature_distances(
    features1: np.ndarray, features2: np.ndarray
) -> np.ndarray:
    """
    This function computes a list of distances from every feature in one array
    to every feature in another.

    Using Numpy broadcasting is required to keep memory requirements low.

    Note: Using a double for-loop is going to be too slow. One for-loop is the
    maximum possible. Vectorization is needed.
    See numpy broadcasting details here:
        https://cs231n.github.io/python-numpy-tutorial/

    Args:
        features1: A numpy array of shape (n1,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
        features2: A numpy array of shape (n2,feat_dim) representing a second
            set of features (n1 not necessarily equal to n2)

    Returns:
        dists: A numpy array of shape (n1,n2) which holds the distances (in
            feature space) from each feature in features1 to each feature in
            features2
    """

    sum_square1 = np.sum(features1**2, axis=1)
    sum_square2 = np.sum(features2**2, axis=1)

    cross_term = 2 * np.dot(features1, features2.T)

    dists = np.sqrt(
        sum_square1[:, np.newaxis] + sum_square2[np.newaxis, :] - cross_term
    )

    return dists


def match_features_ratio_test(
    features1: np.ndarray, features2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest-neighbor distance ratio feature matching.

    This function does not need to be symmetric (e.g. it can produce different
    numbers of matches depending on the order of the arguments).

    To start with, simply implement the "ratio test", equation 7.18 in section
    7.1.3 of Szeliski. There are a lot of repetitive features in these images,
    and all of their descriptors will look similar. The ratio test helps us
    resolve this issue (also see Figure 11 of David Lowe's IJCV paper).

    You should call `compute_feature_distances()` in this function, and then
    process the output.

    Args:
        features1: A numpy array of shape (n1,feat_dim) representing one set of
            features, where feat_dim denotes the feature dimensionality
        features2: A numpy array of shape (n2,feat_dim) representing a second
            set of features (n1 not necessarily equal to n2)

    Returns:
        matches: A numpy array of shape (k,2), where k is the number of matches.
            The first column is an index in features1, and the second column is
            an index in features2
        confidences: A numpy array of shape (k,) with the real valued confidence
            for every match

    'matches' and 'confidences' can be empty, e.g., (0x2) and (0x1)
    """

    dists = compute_feature_distances(features1, features2)

    idx_sorted = np.argsort(dists, axis=1)

    ratios = dists[np.arange(dists.shape[0]), idx_sorted[:, 0]] / (
        dists[np.arange(dists.shape[0]), idx_sorted[:, 1]] + 1e-10
    )

    ratio_threshold = 0.8
    good_ratios_mask = ratios < ratio_threshold

    matches = np.column_stack(
        (np.nonzero(good_ratios_mask)[0], idx_sorted[good_ratios_mask, 0])
    )
    confidences = 1 - ratios[good_ratios_mask]

    sorted_confidence_idx = np.argsort(-confidences)
    matches = matches[sorted_confidence_idx]
    confidences = confidences[sorted_confidence_idx]

    return matches, confidences
