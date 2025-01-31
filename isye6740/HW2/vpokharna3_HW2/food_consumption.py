import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class FoodConsumptionPCA:
    def __init__(self, input_path="data/food-consumption.csv"):
        """
        LOAD the data
        """
        self.data = pd.read_csv(input_path, index_col=0)
        self.food_items = self.data.columns
        self.countries = self.data.index

    def country_pca(self):
        """
        Returns (m, 2) numpy array for the first 2 principal components with food as feature vector.
        """
        data_matrix = self.data.values
        mean_vector = np.mean(data_matrix, axis=0)
        centered_matrix = data_matrix - mean_vector
        covariance_matrix = np.cov(centered_matrix.T)
        _, eigenvectors = np.linalg.eigh(covariance_matrix)
        top_eigenvectors = eigenvectors[:, -2:]
        principal_components = np.dot(centered_matrix, top_eigenvectors)
        return principal_components

    def food_pca(self, num_dim=2):
        """
        Returns (m, 2) numpy array the first 2 principal components with country consumptions as feature vector.
        """
        data_matrix = self.data.T.values
        mean_vector = np.mean(data_matrix, axis=0)
        centered_matrix = data_matrix - mean_vector
        covariance_matrix = np.cov(centered_matrix.T)
        _, eigenvectors = np.linalg.eigh(covariance_matrix)
        top_eigenvectors = eigenvectors[:, -2:]
        principal_components = np.dot(centered_matrix, top_eigenvectors)
        return principal_components

    def plot_pca(self, principal_components, labels, title, x_label, y_label):
        plt.figure(figsize=(10, 8))
        plt.scatter(
            principal_components[:, 0], principal_components[:, 1], c="blue", alpha=0.7
        )
        for i, label in enumerate(labels):
            plt.text(
                principal_components[i, 0],
                principal_components[i, 1],
                label,
                fontsize=9,
            )
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(alpha=0.5)
        plt.show()


pca_analyzer = FoodConsumptionPCA(input_path="data/food-consumption.csv")

# part a
country_pca_result = pca_analyzer.country_pca()
pca_analyzer.plot_pca(
    country_pca_result,
    labels=pca_analyzer.countries,
    title="PCA of Countries by Food Consumption",
    x_label="Principal Component 1",
    y_label="Principal Component 2",
)

# part b
food_pca_result = pca_analyzer.food_pca()
pca_analyzer.plot_pca(
    food_pca_result,
    labels=pca_analyzer.food_items,
    title="PCA of Food Items by Country Consumption",
    x_label="Principal Component 1",
    y_label="Principal Component 2",
)
