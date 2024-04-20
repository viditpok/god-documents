import numpy as np
from pca import PCA
from regression import Regression


class Slope(object):

    def __init__(self):
        pass

    @staticmethod
    def pca_slope(X, y):
        """
        Calculates the slope of the first principal component given by PCA

        Args:
            x: N x 1 array of feature x
            y: N x 1 array of feature y
        Return:
            slope: (float) scalar slope of the first principal component
        """
        data = np.hstack((X, y))
        pca = PCA()
        pca.fit(data)
        data_transformed = pca.transform(data)
        slope = np.polyfit(data_transformed[:, 0], y.flatten(), 1)[0]
        return slope

    @staticmethod
    def lr_slope(X, y):
        """
        Calculates the slope of the best fit returned by linear_fit_closed()

        For this function don't use any regularization

        Args:
            X: N x 1 array corresponding to a dataset
            y: N x 1 array of labels y
        Return:
            slope: (float) slope of the best fit
        """
        reg = Regression()
        X_aug = np.hstack((np.ones((X.shape[0], 1)), X))
        theta = reg.linear_fit_closed(X_aug, y)
        slope = theta[1]
        return slope

    @classmethod
    def addNoise(cls, c, x_noise=False, seed=1):
        """
        Creates a dataset with noise and calculates the slope of the dataset
        using the pca_slope and lr_slope functions implemented in this class.

        Args:
            c: (float) scalar, a given noise level to be used on Y and/or X
            x_noise: (Boolean) When set to False, X should not have noise added
                    When set to True, X should have noise.
                    Note that the noise added to X should be different from the
                    noise added to Y. You should NOT use the same noise you add
                    to Y here.
            seed: (int) Random seed
        Return:
            pca_slope_value: (float) slope value of dataset created using pca_slope
            lr_slope_value: (float) slope value of dataset created using lr_slope
        """
        np.random.seed(seed)
        X = np.linspace(0, 10, 100).reshape(-1, 1)
        y = 3 * X.squeeze() + np.random.normal(0, c, X.shape[0])

        if x_noise:
            X = X + np.random.normal(0, c, X.shape)

        pca_slope_value = cls.pca_slope(X, y.reshape(-1, 1))
        lr_slope_value = cls.lr_slope(X, y.reshape(-1, 1))

        return pca_slope_value, lr_slope_value
