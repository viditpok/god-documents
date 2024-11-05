import numpy as np


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

    def __getitem__(self, key):
        assert (key == 0 or key == 1)
        return self.coord[key]


def get_dist(p, q):
    dist = ((p.x - q.x) ** 2 + (p.y - q.y) ** 2) ** 0.5
    return dist


################################################################################
######## helper function for line segment intersection check   #################
# credit:http://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect
################################################################################
def is_zero(val):
    return abs(val) < 1e-2


def is_on_segment(p, q, r):
    if (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and
                q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y)):
        return True
    return False


def get_orientation(p, q, r):
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    if is_zero(val):
        # colinear
        return 0
    elif val > 0:
        # clockwise
        return 1
    else:
        # counter-clockwise
        return 2


def is_intersect(p1, q1, p2, q2):
    """
    Check the line from p1 to q1 intersects with the line from p2 to q2.
    """
    o1 = get_orientation(p1, q1, p2)
    o2 = get_orientation(p1, q1, q2)
    o3 = get_orientation(p2, q2, p1)
    o4 = get_orientation(p2, q2, q1)

    if (o1 != o2 and o3 != o4):
        return True
    if (is_zero(o1) and is_on_segment(p1, p2, q1)):
        return True
    if (is_zero(o2) and is_on_segment(p1, q2, q1)):
        return True
    if (is_zero(o3) and is_on_segment(p2, p1, q2)):
        return True
    if (is_zero(o4) and is_on_segment(p2, q1, q2)):
        return True
    return False


if __name__ == '__main__':
    p1, q1 = Node((1, 1)), Node((10, 1))
    p2, q2 = Node((1, 2)), Node((10, 2))
    print(is_intersect(p1, q1, p2, q2))
    p1, q1 = Node((10, 0)), Node((0, 10))
    p2, q2 = Node((0, 0)), Node((10, 10))
    print(is_intersect(p1, q1, p2, q2))
    p1, q1 = Node((-5, -5)), Node((0, 0))
    p2, q2 = Node((1, 1)), Node((10, 10))
    print(is_intersect(p1, q1, p2, q2))
    print(p1[0])
    print(p1[1])
