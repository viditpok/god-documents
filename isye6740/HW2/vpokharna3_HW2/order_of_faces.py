import numpy as np
import scipy.io as sio
from scipy.sparse.csgraph import shortest_path
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.pyplot as plt
import networkx as nx
import random


class OrderOfFaces:
    def __init__(self, images_path="data/isomap.mat"):
        """
        Load the data from input files provided.
        """
        self.data = sio.loadmat(images_path)["images"]
        self.data = self.data.T
        self.num_images, self.image_dim = self.data.shape
        np.random.seed(6740)

    def get_adjacency_matrix(self, epsilon):
        """
        This method returns the adjacency matrix for given epsilon (kernel width)

        Inputs:
            epsilon (int): kernel width

        Output:
            2d numpy array which can directly be used with plt.imshow(...) .
        """
        adjacency_matrix = np.zeros((self.num_images, self.num_images))
        for i in range(self.num_images):
            for j in range(i + 1, self.num_images):
                dist = np.linalg.norm(self.data[i] - self.data[j])
                if dist < epsilon:
                    adjacency_matrix[i, j] = dist
                    adjacency_matrix[j, i] = dist
        return adjacency_matrix

    def get_best_epsilon(self):
        """
        Returns the best epsilon for ISOMAP.
        This could be a hardcoded value or a strategy implemented by code.
        """

        return 13

    def isomap(self, epsilon):
        """
        Returns the first 2 principal components for the low embedding space.

        Inputs:
            epsilon (int): kernel width

        Output:
            (m, 2) numpy array.
        """
        adjacency_matrix = self.get_adjacency_matrix(epsilon=epsilon)
        shortest_paths = shortest_path(adjacency_matrix, directed=False)
        m = self.num_images
        I = np.identity(m)
        ones = np.ones(m)
        H = I - (1 / m) * np.outer(ones, ones.T)
        C = (-1 / 2) * H @ (shortest_paths**2) @ H
        U_C, Sig_C, _ = np.linalg.svd(C)
        dim_1 = U_C[:, 0] * np.sqrt(Sig_C[0])
        dim_2 = U_C[:, 1] * np.sqrt(Sig_C[1])
        return np.column_stack((dim_1, dim_2))

    def pca(self):
        """
        Returns the first 2 principal components for the low embedding space.

        Output:
            (m, 2) numpy array .
        """
        mean_vector = np.mean(self.data, axis=0)
        centered_data = self.data - mean_vector
        covariance_matrix = np.cov(centered_data, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        indices = np.argsort(-eigenvalues)[:2]
        top_eigenvectors = eigenvectors[:, indices]
        principal_components = np.dot(centered_data, top_eigenvectors)
        return principal_components

    def visualize_graph(self, adjacency_matrix, sample_indices):
        graph = nx.Graph()
        num_nodes = adjacency_matrix.shape[0]
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if adjacency_matrix[i, j] > 0:
                    graph.add_edge(i, j, weight=adjacency_matrix[i, j])
        pos = nx.spring_layout(graph, seed=6740)
        plt.figure(figsize=(12, 8))
        nx.draw_networkx_edges(graph, pos, alpha=0.4)
        nx.draw_networkx_nodes(graph, pos, node_size=20, alpha=0.7, node_color="blue")
        for idx in sample_indices:
            nx.draw_networkx_nodes(
                graph, pos, nodelist=[idx], node_size=100, node_color="red"
            )
            plt.text(pos[idx][0], pos[idx][1], f"Img {idx}", fontsize=8, color="red")

        plt.title("Nearest Neighbor Graph Visualization")
        plt.axis("off")
        plt.show()

    def visualize_embedding_with_images(self, embedding, sample_count=40):
        random.seed(6740)
        sample_indices = random.sample(range(self.num_images), sample_count)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(
            embedding[:, 0], embedding[:, 1], c="blue", alpha=0.6, label="Data Points"
        )

        for idx in sample_indices:
            img = self.data[idx].reshape(64, 64).T
            ab = AnnotationBbox(
                OffsetImage(img, cmap="gray", zoom=0.5),
                (embedding[idx, 0], embedding[idx, 1]),
                pad=0.1,
            )
            ax.add_artist(ab)

        plt.title("ISOMAP Embedding with Selected Images")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.show()


order_of_faces = OrderOfFaces()

# part a
best_epsilon = order_of_faces.get_best_epsilon()
adjacency_matrix = order_of_faces.get_adjacency_matrix(best_epsilon)
plt.imshow(adjacency_matrix, cmap="viridis")
plt.colorbar()
plt.title(f"Adjacency Matrix (ε = {best_epsilon})")
plt.show()

sample_indices = [0, 100, 200, 300, 400]
order_of_faces.visualize_graph(adjacency_matrix, sample_indices)

# part b
isomap_embedding = order_of_faces.isomap(best_epsilon)
order_of_faces.visualize_embedding_with_images(isomap_embedding)

# part c
pca_embedding = order_of_faces.pca()
fig, ax = plt.subplots(figsize=(10, 10))
ax.scatter(pca_embedding[:, 0], pca_embedding[:, 1], color="blue", label="Data Points")
ax.set_title("PCA Embedding with Selected Images")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
sample_indices = np.random.choice(pca_embedding.shape[0], size=40, replace=False)
for i in sample_indices:
    img = order_of_faces.data[i, :].reshape(64, 64).T
    ab = AnnotationBbox(
        OffsetImage(img, cmap="gray", zoom=0.5), (pca_embedding[i, 0], pca_embedding[i, 1]), pad=0.1
    )
    ax.add_artist(ab)
ax.legend(loc="best")
plt.show()
