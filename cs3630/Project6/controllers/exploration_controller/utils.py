import random
import math
import numpy as np
import pickle

# Node object for RRT
class Node(object):
    """Class representing a node in RRT
    """

    def __init__(self, coord, parent=None):
        super(Node, self).__init__()
        self.coord = coord    # 2D coordinate of the node in the map
        self.parent = parent  # parent node in the RRT tree

    @property
    def x(self):
        return self.coord[0]

    @property
    def y(self):
        return self.coord[1]
    
    @property
    def xy(self):
        return (self.coord[0], self.coord[1])

    def __getitem__(self, key):
        assert (key == 0 or key == 1)
        return self.coord[key]


""" Some math utilies, feel free to use any of these!!!
"""

# euclian distance in grid world
def grid_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def grid_node_distance(node_a, node_b):
    return math.sqrt((node_a.x - node_b.x) ** 2 + (node_a.y - node_b.y) ** 2)


# utils for 2d rotation, given frame \theta
def rotate_point(x, y, heading_deg):
    c = math.cos(math.radians(heading_deg))
    s = math.sin(math.radians(heading_deg))
    xr = x * c + y * s
    yr = -x * s + y * c
    return xr, yr

def diff_heading_deg(heading1, heading2):
    """
    Arguments:
        heading1: Angle (degrees)
        heading2: Angle (degrees)

    Returns:
        dh: Difference in heading1 and headin2 in range (-180,180] (degrees)
    """
    dh = heading1 - heading2
    while dh > 180:
        dh -= 360
    while dh <= -180:
        dh += 360
    return dh


def find_line(p1, p2):
    """ Find the line that connects two points p1 and p2 in the form y=mx+c
    """
    m = 0 if p2[0] == p1[0] else (p2[1]-p1[1])/(p2[0]-p1[0])
    c = p2[1] - m*p2[0]
    return m, c


def find_dist(m, c, p):
    return abs(m*p[0]-p[1]+c)/math.sqrt(m**2 + 1)


def find_centroid(points):
    """
    Finds centroid of a set of 2D coords
    """
    sum_x = sum([p[0] for p in points])
    sum_y = sum([p[1] for p in points])
    return sum_x/len(points), sum_y/len(points)


