"""
utils.py

Includes functions to parse TSP data from a file and calculate Euclidean distance between points.
"""

import math


def parse_data(file_path):
    points = []
    with open(file_path, "r") as f:
        section_found = False
        for line in f:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                section_found = True
                continue
            if not section_found:
                continue
            if line == "EOF":
                break
            parts = line.split()
            vertex_id = int(parts[0])
            x, y = float(parts[1]), float(parts[2])
            points.append((vertex_id, x, y))
    return points


def euclidean_distance(point1, point2):
    return round(math.sqrt((point1[1] - point2[1]) ** 2 + (point1[2] - point2[2]) ** 2))


def validate_points(points):
    if len(points) < 2:
        raise ValueError("At least two points are required for TSP.")
    if not all(isinstance(p, tuple) and len(p) == 3 for p in points):
        raise ValueError("Each point must be a tuple (vertex_id, x, y).")
