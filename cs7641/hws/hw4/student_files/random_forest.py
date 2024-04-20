import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.tree import ExtraTreeClassifier


class RandomForest(object):
    def __init__(self, n_estimators, max_depth, max_features, random_seed=None):

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.random_seed = random_seed
        self.bootstraps_row_indices = []
        self.feature_indices = []
        self.out_of_bag = []
        self.decision_trees = [
            ExtraTreeClassifier(max_depth=max_depth, criterion="entropy")
            for i in range(n_estimators)
        ]

    def _bootstrapping(self, num_training, num_features, random_seed=None):
        """
        TODO:
        - Randomly select a sample dataset of size num_training with replacement from the original dataset.
        - Randomly select certain number of features (num_features denotes the total number of features in X,
          max_features denotes the percentage of features that are used to fit each decision tree) without replacement from the total number of features.

        Args:
        - num_training: number of data points in the bootstrapped dataset.
        - num_features: number of features in the original dataset.

        Return:
        - row_idx: the row indices corresponding to the row locations of the selected samples in the original dataset.
        - col_idx: the column indices corresponding to the column locations of the selected features in the original feature list.
        Reference: https://en.wikipedia.org/wiki/Bootstrapping_(statistics)
        Hint 1: Please use np.random.choice. First get the row_idx first, and then second get the col_idx.
        Hint 2:  If you are getting a Test Failed: 'bool' object has no attribute 'any' error, please try flooring, or converting to an int, the number of columns needed for col_idx. Using np.ceil() can cause an autograder error.
        """
        np.random.seed(random_seed)

        row_idx = np.random.choice(num_training, size=num_training, replace=True)

        col_idx = np.random.choice(
            num_features, size=int(self.max_features * num_features), replace=False
        )
        return row_idx, col_idx

    def bootstrapping(self, num_training, num_features):

        np.random.seed(self.random_seed)
        for i in range(self.n_estimators):
            total = set(list(range(num_training)))
            row_idx, col_idx = self._bootstrapping(num_training, num_features)
            total = total - set(row_idx)
            self.bootstraps_row_indices.append(row_idx)
            self.feature_indices.append(col_idx)
            self.out_of_bag.append(total)

    def fit(self, X, y):
        """
        TODO:
        Train decision trees using the bootstrapped datasets.
        Note that you need to use the row indices and column indices.
        X: NxD numpy array, where N is number
           of instances and D is the dimensionality of each
           instance
        y: 1D numpy array of size (N,), the predicted labels
        Returns:
            None. Calling this function should train the decision trees held in self.decision_trees
        """
        self.bootstrapping(X.shape[0], X.shape[1])

        for i in range(self.n_estimators):

            X_sample = X[self.bootstraps_row_indices[i]][:, self.feature_indices[i]]
            y_sample = y[self.bootstraps_row_indices[i]]

            self.decision_trees[i].fit(X_sample, y_sample)

    def OOB_score(self, X, y):

        accuracy = []
        for i in range(len(X)):
            predictions = []
            for t in range(self.n_estimators):
                if i in self.out_of_bag[t]:
                    predictions.append(
                        self.decision_trees[t].predict(
                            np.reshape(X[i][self.feature_indices[t]], (1, -1))
                        )[0]
                    )
            if len(predictions) > 0:
                accuracy.append(np.sum(predictions == y[i]) / float(len(predictions)))
        return np.mean(accuracy)

    def predict(self, X):
        N = X.shape[0]
        y = np.zeros((N, 7))
        for t in range(self.n_estimators):
            X_curr = X[:, self.feature_indices[t]]
            y += self.decision_trees[t].predict_proba(X_curr)
        pred = np.argmax(y, axis=1)
        return pred

    def plot_feature_importance(self, data_train):
        """
        TODO:
        -Display a bar plot showing the feature importance of every feature in
        one decision tree of your choice from the tuned random_forest from Q3.2.
        Args:
            data_train: This is the orginal data train Dataframe containg data AND labels.
                Hint: you can access labels with data_train.columns
        Returns:
            None. Calling this function should simply display the aforementioned feature importance bar chart
        """
        tree_index = 0
        tree = self.decision_trees[tree_index]

        feature_importances = tree.feature_importances_

        feature_names = data_train.columns

        non_zero_indices = np.where(feature_importances != 0)[0]
        feature_importances = feature_importances[non_zero_indices]
        feature_names = feature_names[non_zero_indices]

        sorted_indices = np.argsort(feature_importances)[::-1]
        feature_importances = feature_importances[sorted_indices]
        feature_names = feature_names[sorted_indices]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(feature_importances)), feature_importances, align="center")
        plt.yticks(range(len(feature_importances)), feature_names)
        plt.xlabel("Feature Importance")
        plt.ylabel("Feature")
        plt.title("Feature Importance for Decision Tree {}".format(tree_index))
        plt.show()

    def select_hyperparameters(self):
        """
        Hyperparameter tuning Question
        TODO: assign a value to n_estimators, max_depth, max_features
        Args:
            None
        Returns:
            n_estimators: int number (e.g 2)
            max_depth: int number (e.g 4)
            max_features: a float between 0.0-1.0 (e.g 0.1)
        """
        n_estimators = 100
        max_depth = 20
        max_features = 0.5

        return n_estimators, max_depth, max_features