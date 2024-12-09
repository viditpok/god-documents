"""
exact_solution.py

Finds the shortest path by evaluating all permutations of points with a cutoff time.
Returns the best distance and path found.
"""

from itertools import permutations
import time
from utils import euclidean_distance


def exact_tsp(points, cutoff):
    # edge cases
    if not points:
        return 0, []
    if len(points) == 1:
        return 0, [points[0]]

    # initialize variables
    start_time = time.time()
    best_distance = float("inf")
    best_path = []

    # iterate over all permutations of points
    for perm in permutations(points):
        # check for cutoff time
        if time.time() - start_time > cutoff:
            print("Time cutoff reached, returning best solution found so far")
            break

        # calculate total distance
        current_distance = 0
        for i in range(len(perm)):
            current_distance += euclidean_distance(perm[i], perm[(i + 1) % len(perm)])

        # update best solution
        if current_distance < best_distance:
            best_distance = current_distance
            best_path = perm

    # return the best distance and path found
    return best_distance, best_path
