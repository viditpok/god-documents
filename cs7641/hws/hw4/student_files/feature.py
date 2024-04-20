import numpy as np


def create_nl_feature(X):
    """
    TODO - Create additional features and add it to the dataset

    returns:
        X_new - (N, d + num_new_features) array with
                additional features added to X such that it
                can classify the points in the dataset.
    """

    x1 = X[:, 0]
    x2 = X[:, 1]

    squared_features = np.square(X)
    interaction_feature = (X[:, 0] * X[:, 1]).reshape(-1, 1)
    sin_x1 = np.sin(X[:, 0])
    sin_x2 = np.sin(X[:, 1])
    cos_x1 = np.cos(X[:, 0])
    cos_x2 = np.cos(X[:, 1])

    X_new = np.hstack(
        [
            X,
            squared_features,
            interaction_feature,
            sin_x1.reshape(-1, 1),
            sin_x2.reshape(-1, 1),
            cos_x1.reshape(-1, 1),
            cos_x2.reshape(-1, 1),
        ]
    )

    return X_new
