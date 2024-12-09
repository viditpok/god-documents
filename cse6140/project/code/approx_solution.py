"""
approx_solution.py

Uses a minimum spanning tree (MST) and preorder traversal to approximate a TSP tour.
Randomizes the start node with a specified seed for varied results.
"""

import networkx as nx
import random
from utils import euclidean_distance


def approximate_tsp(points, seed):
    # set the random seed
    if seed is not None:
        random.seed(seed)

    # create a graph and add edges with distances
    G = nx.Graph()
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distance = euclidean_distance(points[i], points[j])
            G.add_edge(
                points[i][0], points[j][0], weight=distance
            )  # use vertex IDs for nodes

    # generate the mst
    mst = nx.minimum_spanning_tree(G)

    # choose a random start node for the preorder traversal
    start_node = random.choice(points)[0]  # select a random node ID from points
    preorder_nodes = list(nx.dfs_preorder_nodes(mst, start_node))

    # calculate the tour distance and complete the cycle
    total_distance = 0
    tour = []
    for i in range(len(preorder_nodes)):
        current_node = preorder_nodes[i]
        next_node = preorder_nodes[(i + 1) % len(preorder_nodes)]
        total_distance += G[current_node][next_node]["weight"]
        tour.append(current_node)

    # return the best distance and path found
    return total_distance, tour
