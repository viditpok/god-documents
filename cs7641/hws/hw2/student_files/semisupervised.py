"""
File: semisupervised.py
Project: autograder_test_files
File Created: September 2020
Author: Shalini Chaudhuri (you@you.you)
Updated: February 2024, Kanishk
"""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm

SIGMA_CONST = 1e-6
LOG_CONST = 1e-32


def complete_(data):
    """
    Args:
        data: N x (D+1) numpy array where the last column is the labels
    Return:
        labeled_complete: n x (D+1) array (n <= N) where values contain both complete features and labels
    """
    is_label_not_nan = ~np.isnan(data[:, -1])
    is_feature_not_nan = ~np.isnan(data[:, :-1]).any(axis=1)
    labeled_complete = data[np.logical_and(is_label_not_nan, is_feature_not_nan)]
    return labeled_complete


def incomplete_(data):
    """
    Args:
        data: N x (D+1) numpy array where the last column is the labels
    Return:
        labeled_incomplete: n x (D+1) array (n <= N) where values contain incomplete features but complete labels
    """
    is_feature_nan = np.isnan(data[:, :-1]).any(axis=1)
    is_label_not_nan = ~np.isnan(data[:, -1])
    labeled_incomplete = data[np.logical_and(is_feature_nan, is_label_not_nan)]
    return labeled_incomplete


def unlabeled_(data):
    """
    Args:
        data: N x (D+1) numpy array where the last column is the labels
    Return:
        unlabeled_complete: n x (D+1) array (n <= N) where values contain complete features but incomplete labels
    """
    is_label_nan = np.isnan(data[:, -1])
    is_feature_not_nan = ~np.isnan(data[:, :-1]).any(axis=1)
    unlabeled_complete = data[np.logical_and(is_label_nan, is_feature_not_nan)]
    return unlabeled_complete


class CleanData(object):
    def __init__(self):
        pass

    def pairwise_dist(self, x, y):
        """
        Args:
            x: N x D numpy array
            y: M x D numpy array
        Return:
            dist: N x M array, where dist[i, j] is the euclidean distance between
            x[i, :] and y[j, :]
        """
        dist = np.sqrt(((x[:, np.newaxis, :] - y[np.newaxis, :, :]) ** 2).sum(axis=2))
        return dist

    def __call__(self, incomplete_points, complete_points, K, **kwargs):
        """
        Function to clean or "fill in" NaN values in incomplete data points based on
        the average value for that feature for the K-nearest neighbors in the complete data points.

        Args:
            incomplete_points: N_incomplete x (D+1) numpy array, the incomplete labeled observations
            complete_points:   N_complete   x (D+1) numpy array, the complete labeled observations
            K: integer, corresponding to the number of nearest neighbors you want to base your calculation on
            kwargs: any other args you want
        Return:
            clean_points: (N_complete + N_incomplete) x (D+1) numpy array, containing both the complete points and recently filled points

        Notes:
            (1) The first D columns are features, and the last column is the class label
            (2) There may be more than just 2 class labels in the data (e.g. labels could be 0,1,2 or 0,1,2,...,M)
            (3) There will be at most 1 missing feature value in each incomplete data point (e.g. no points will have more than one NaN value)
            (4) You want to find the k-nearest neighbors, from the complete dataset, with the same class labels;
            (5) There may be missing values in any of the features. It might be more convenient to address each feature at a time.
            (6) Do NOT use a for-loop over N_incomplete; you MAY use a for-loop over the M labels and the D features (e.g. omit one feature at a time)
            (7) You do not need to order the rows of the return array clean_points in any specific manner
        """
        labels = np.unique(complete_points[:, -1])
        cleaned_points = []

        for label in labels:
            for feature_index in range(incomplete_points.shape[1] - 1):

                incomplete_label_points = incomplete_points[
                    (incomplete_points[:, -1] == label)
                    & np.isnan(incomplete_points[:, feature_index])
                ]
                complete_label_points = complete_points[complete_points[:, -1] == label]

                if incomplete_label_points.size == 0:
                    continue

                valid_incomplete_points = np.delete(
                    incomplete_label_points, feature_index, axis=1
                )
                valid_complete_points = np.delete(
                    complete_label_points, feature_index, axis=1
                )

                dist = self.pairwise_dist(
                    valid_incomplete_points[:, :-1], valid_complete_points[:, :-1]
                )
                for i, point in enumerate(incomplete_label_points):

                    nearest_indices = np.argsort(dist[i])[:K]

                    mean_value = np.mean(
                        complete_label_points[nearest_indices, feature_index]
                    )
                    incomplete_label_points[i, feature_index] = mean_value

                cleaned_points.append(incomplete_label_points)

        cleaned_points = np.vstack(cleaned_points)
        clean_points = np.vstack((cleaned_points, complete_points))
        return clean_points


def mean_clean_data(data):
    """
    Args:
        data: N x (D+1) numpy array where only last column is guaranteed non-NaN values and is the labels
    Return:
        mean_clean: N x (D+1) numpy array where each NaN value in data has been replaced by the mean feature value
    Notes:
        (1) When taking the mean of any feature, do not count the NaN value
        (2) Return all values to max one decimal point
        (3) The labels column will never have NaN values
    """
    means = np.nanmean(data[:, :-1], axis=0)
    indices = np.where(np.isnan(data[:, :-1]))
    data[indices] = np.take(means, indices[1])
    mean_clean = np.round(data, 1)
    return mean_clean


class SemiSupervised(object):
    def __init__(self):
        pass

    def softmax(self, logit):
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array where softmax has been applied row-wise to input logit
        """
        e_logit = np.exp(logit - np.max(logit, axis=1, keepdims=True))
        return e_logit / np.sum(e_logit, axis=1, keepdims=True)

    def logsumexp(self, logit):
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:])
        """
        max_logit = np.max(logit, axis=1, keepdims=True)
        return max_logit + np.log(
            np.sum(np.exp(logit - max_logit), axis=1, keepdims=True)
        )

    def normalPDF(self, logit, mu_i, sigma_i):
        """
        Args:
            logit: N x D numpy array
            mu_i: 1xD numpy array, the center for the ith gaussian.
            sigma_i: 1xDxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            pdf: 1xN numpy array, the probability distribution of N data for the ith gaussian

        Hint:
            np.diagonal() should be handy.
        """
        D = logit.shape[1]
        denom = np.sqrt((2 * np.pi) ** D * np.linalg.det(sigma_i))
        diff = logit - mu_i
        exp_term = np.exp(
            -0.5 * np.sum(np.dot(diff, np.linalg.inv(sigma_i)) * diff, axis=1)
        )
        return exp_term / denom

    def _init_components(self, points, K, **kwargs):
        """
        Args:
            points: Nx(D+1) numpy array, the observations
            K: number of components
            kwargs: any other args you want
        Return:
            pi: numpy array of length K; contains the prior probabilities of each class k
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
        Hint:
            As explained in the algorithm, you need to calculate the values of mu, sigma and pi based on the labelled dataset
        """
        labels = points[:, -1]
        D = points.shape[1] - 1
        pi = np.zeros(K)
        mu = np.zeros((K, D))
        sigma = np.zeros((K, D, D))

        for k in range(K):

            class_points = points[labels == k, :-1]
            pi[k] = class_points.shape[0] / float(points.shape[0])
            mu[k] = np.mean(class_points, axis=0)

            sigma[k] = np.diag(np.var(class_points, axis=0) + 1e-6)

        return pi, mu, sigma

    def _ll_joint(self, points, pi, mu, sigma, **kwargs):
        """
        Args:
            points: NxD numpy array, the observations
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
        Return:
            ll(log-likelihood): NxK array, where ll(i, j) = log pi(j) + log NormalPDF(points_i | mu[j], sigma[j])
        """
        K = len(pi)
        N = points.shape[0]
        ll = np.zeros((N, K))
        for k in range(K):
            ll[:, k] = np.log(pi[k]) + self.normalPDF(points, mu[k], sigma[k])
        return ll

    def _E_step(self, points, pi, mu, sigma, **kwargs):
        """
        Args:
            points: NxD numpy array, the observations
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
        Return:
            gamma: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.

        Hint: You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
        """
        ll_joint = self._ll_joint(points, pi, mu, sigma)
        log_prob = ll_joint - self.logsumexp(ll_joint)
        gamma = np.exp(log_prob)
        return gamma

    def _M_step(self, points, gamma, **kwargs):
        """
        Args:
            points: NxD numpy array, the observations
            gamma: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.

        Hint:  There are formulas in the slide.
        """
        N, D = points.shape
        K = gamma.shape[1]

        Nk = np.sum(gamma, axis=0)
        pi = Nk / N
        mu = np.dot(gamma.T, points) / Nk[:, np.newaxis]

        sigma = np.zeros((K, D, D))
        for k in range(K):
            diff = points - mu[k]
            sigma[k] = np.dot(gamma[:, k] * diff.T, diff) / Nk[k]
            sigma[k] += np.eye(D) * SIGMA_CONST

        return pi, mu, sigma

    def __call__(
        self, points, K, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, **kwargs
    ):
        """
        Args:
            points: N x (D+1) numpy array, where
                - N is
                - D is the number of features,
                - the last column is the point labels (when available) or NaN for unlabeled points
            K: integer, number of clusters
            max_iters: maximum number of iterations
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want
        Return:
            pi, mu, sigma: (1xK np array, KxD numpy array, KxDxD numpy array)
        """
        labeled_data = points[~np.isnan(points[:, -1])]
        pi, mu, sigma = self._init_components(labeled_data, K)
        prev_ll = None

        for iteration in range(max_iters):
            gamma = self._E_step(points[:, :-1], pi, mu, sigma)
            pi, mu, sigma = self._M_step(points[:, :-1], gamma)

            current_ll = np.sum(self._ll_joint(points[:, :-1], pi, mu, sigma))
            if prev_ll is not None and (abs(current_ll - prev_ll) < abs_tol or abs((current_ll - prev_ll) / prev_ll) < rel_tol):
                break
            prev_ll = current_ll

        return pi, mu, sigma

    def predict(self, points, pi, mu, sigma):
        ll_joint = self._ll_joint(points, pi, mu, sigma)
        log_prob = ll_joint - self.logsumexp(ll_joint)
        gamma = np.exp(log_prob)
        return np.argmax(gamma, axis=1)


class ComparePerformance(object):
    def __init__(self):
        pass

    @staticmethod
    def accuracy_semi_supervised(training_data, validation_data, K: int) -> float:
        """
        Train a classification model using your SemiSupervised object on the training_data.
        Classify the validation_data using the trained model
        Return the accuracy score of the model's predicted classification of the validation_data

        Args:
            training_data: N_t x (D+1) numpy array, where
                - N_t is the number of data points in the training set,
                - D is the number of features, and
                - the last column represents the labels (when available) or a flag that allows you to separate the unlabeled data.
            validation_data: N_v x(D+1) numpy array, where
                - N_v is the number of data points in the validation set,
                - D is the number of features, and
                - the last column are the labels
            K: integer, number of clusters for SemiSupervised object
        Return:
            accuracy: floating number

        Note: (1) validation_data will NOT include any unlabeled points
              (2) you may use sklearn accuracy_score: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
        """

        semi_supervised_model = SemiSupervised()

        pi, mu, sigma = semi_supervised_model(training_data, K)

        predictions = semi_supervised_model.predict(
            validation_data[:, :-1], pi, mu, sigma
        )

        accuracy = accuracy_score(validation_data[:, -1], predictions)

        return accuracy

    @staticmethod
    def accuracy_GNB(training_data, validation_data) -> float:
        """
        Train a Gaussion Naive Bayes classification model (sklearn implementation) on the training_data.
        Classify the validation_data using the trained model
        Return the accuracy score of the model's predicted classification of the validation_data

        Args:
            training_data: N_t x (D+1) numpy array, where
                - N is the number of data points in the training set,
                - D is the number of features, and
                - the last column represents the labels
            validation_data: N_v x (D+1) numpy array, where
                - N_v is the number of data points in the validation set,
                - D is the number of features, and
                - the last column are the labels
        Return:
            accuracy: floating number

        Note: (1) both training_data and validation_data will NOT include any unlabeled points
              (2) use sklearn implementation of Gaussion Naive Bayes: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
        """

        X_train = training_data[:, :-1]
        y_train = training_data[:, -1]

        gnb = GaussianNB()
        gnb.fit(X_train, y_train)

        X_validation = validation_data[:, :-1]
        y_validation = validation_data[:, -1]
        predictions = gnb.predict(X_validation)
        accuracy = accuracy_score(y_validation, predictions)
        return accuracy
