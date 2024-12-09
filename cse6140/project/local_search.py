"""
local_search.py

Optimizes a TSP tour using simulated annealing with a cooling schedule.
Randomly explores neighboring solutions, accepting improvements or probabilistic worsened moves.
Returns best tour found within the cutoff time.
"""

import random
import math
import time
from heapq import heappush, heappop
from utils import euclidean_distance


def generate_2_approximation(points):
    """
    Generate a 2-approximation for TSP using the MST-based approach.
    Steps:
    1. Build a Minimum Spanning Tree (MST) using Prim's algorithm.
    2. Perform a pre-order traversal of the MST to generate a TSP tour.
    """
    validate_points(points)  # validate input points

    if len(points) < 2:
        return points

    # build the MST using Prim's algorithm
    mst = build_mst(points)

    # perform a pre-order traversal of the MST to generate a TSP tour
    tour = preorder_traversal(mst, points)

    # close the cycle by appending the starting point at the end
    tour.append(tour[0])

    return tour


def build_mst(points):
    """
    Build a Minimum Spanning Tree (MST) using Prim's algorithm.
    Returns the MST as an adjacency list.
    """
    n = len(points)
    mst = {i: [] for i in range(n)}  # adjacency list for the MST
    visited = [False] * n
    min_heap = []

    # Start from vertex 0
    visited[0] = True
    for j in range(1, n):
        dist = euclidean_distance(points[0], points[j])
        heappush(min_heap, (dist, 0, j))  # (distance, from, to)

    while min_heap:
        weight, u, v = heappop(min_heap)
        if not visited[v]:
            visited[v] = True
            mst[u].append(v)
            mst[v].append(u)
            for w in range(n):
                if not visited[w]:
                    dist = euclidean_distance(points[v], points[w])
                    heappush(min_heap, (dist, v, w))

    return mst


def preorder_traversal(mst, points):
    """
    Perform a pre-order traversal of the MST to generate a TSP tour.
    """
    visited = set()
    tour = []

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        tour.append(points[node])
        for neighbor in mst[node]:
            dfs(neighbor)

    dfs(0)
    
    # Add check to ensure all vertices are visited
    if len(visited) != len(points):
        missing = set(range(len(points))) - visited
        for vertex in missing:
            tour.append(points[vertex])
            
    return tour


def simulated_annealing_tsp_with_approximation(tour, initial_temperature, cooling_rate, cutoff, seed=None):
   if seed is not None:
       random.seed(seed)

   # Get required vertices and mapping
   required_vertices = set(p[0] for p in tour) 
   vertex_to_point = {p[0]: p for p in tour}
   
   # Start with all vertices exactly once
   current_solution = [vertex_to_point[v] for v in sorted(required_vertices)]
   
   best_solution = current_solution[:]
   best_distance = calculate_total_distance(best_solution)
   
   current_temperature = initial_temperature
   start_time = time.time()

   while current_temperature > 1 and (time.time() - start_time) < cutoff:
       # generate a neightbour using 2-opt (swap two edges)
       new_solution = two_opt_swap(current_solution[:])
       
       # Get unique vertices while preserving order
       seen = set()
       new_solution = [x for x in new_solution if x[0] not in seen and not seen.add(x[0])]
       
       # Skip if not all vertices included
       if set(p[0] for p in new_solution) != required_vertices:
           continue
           
       current_distance = calculate_total_distance(current_solution)
       new_distance = calculate_total_distance(new_solution)
       
       # decide whether to accept the new solution
       if new_distance < current_distance or random.random() < math.exp((current_distance - new_distance) / current_temperature):
           current_solution = new_solution[:]
           
           # update the best solution found so far
           if new_distance < best_distance:
               best_solution = new_solution[:]
               best_distance = new_distance

       current_temperature *= cooling_rate

   # Final check for all vertices without duplicates
   seen = set()
   best_solution = [x for x in best_solution if x[0] not in seen and not seen.add(x[0])]
   
   missing = required_vertices - set(p[0] for p in best_solution)
   for v in missing:
       best_solution.append(vertex_to_point[v])
           
   return best_distance, [p[0] for p in best_solution]
   # return the best distance and path as vertex IDs

def two_opt_swap(tour):
    """
    Perform a 2-opt swap to generate a neighboring solution.
    Select two indices i and j, reverse the segment between them.
    """
    new_tour = tour[:]
    i, j = sorted(random.sample(range(len(tour)), 2))
    new_tour[i : j + 1] = reversed(
        new_tour[i : j + 1]
    )  # reverse the segment between i and j
    return new_tour


def calculate_total_distance(tour):
    """
    Helper function to calculate the total distance of a tour.
    """
    total_distance = 0
    for i in range(len(tour)):
        total_distance += euclidean_distance(
            tour[i], tour[(i + 1) % len(tour)]  # complete the cycle
        )
    return total_distance


def validate_points(points):
    """
    Validate input points to ensure they are well-formed.
    """
    if len(points) < 2:
        raise ValueError("At least two points are required for TSP.")
    if not all(isinstance(p, tuple) and len(p) == 3 for p in points):
        raise ValueError("Each point must be a tuple (vertex_id, x, y).")


def validate_tour(tour):
    """
    Validate that the input is a valid tour.
    """
    if len(tour) < 2 or not all(isinstance(p, tuple) and len(p) == 3 for p in tour):
        raise ValueError(
            "Input must be a valid tour in the form [(vertex_id, x, y), ...]."
        )
