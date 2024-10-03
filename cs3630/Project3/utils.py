import setting
import random
random.seed(setting.RANDOM_SEED)
import numpy as np
np.random.seed(setting.RANDOM_SEED)
import math
from typing import Tuple

def grid_distance(x1: int, y1: int, x2: int, y2: int) -> float:
    """
    
    Calculate the Euclidean distance between two points in a grid world.

    Arguments:
        x1, y1: int
            Coordinates of the first point.
        x2, y2: int
            Coordinates of the second point.

    Returns:
        float
            Euclidean distance between the two points.
    """
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def rotate_point(x: float, y: float, heading_deg: float) -> Tuple:
    """
    Rotate a point (x, y) around the origin by a given angle in degrees.

    Arguments:
        x, y: float
            Coordinates of the point to be rotated.
        heading_deg: float
            Angle of rotation in degrees.

    Returns:
        Tuple[float, float] (xr, yr)
            Coordinates of the rotated point.
    """
    heading_rad = math.radians(heading_deg)
    xr = x * math.cos(heading_rad) - y * math.sin(heading_rad)
    yr = y * math.cos(heading_rad) + x * math.sin(heading_rad)
    return xr, yr

def add_gaussian_noise(variable: float, sigma: float = 1.0) -> float:
    """
    Add zero-mean Gaussian noise to the input variable.

    Arguments: 
        variable: float
            Input variable to which noise will be added.
        sigma: float
            Standard deviation of the Gaussian noise.

    Returns:
        float
            Variable with added Gaussian noise.
    """
    return random.gauss(0, sigma) + variable

def diff_heading_deg(heading1, heading2):
    """
    Return the difference between two angles, heading1 - heading2.

    Return value always in range (-180, 180] degrees.
    """
    dh = heading1 - heading2
    while dh > 180:
            dh -= 360
    while dh <= -180:
            dh += 360
    return dh


def compute_mean_pose(particles, confident_dist=1):
    """ 
    Compute the mean pose for all particles.

    (This is not part of the particle filter algorithm but rather an
    addition to show the "best belief" for current pose)
    """
    m_x, m_y, m_count = 0, 0, 0
    # for rotation average
    m_hx, m_hy = 0, 0
    for p in particles:
        m_count += 1
        m_x += p.x
        m_y += p.y
        m_hx += math.sin(math.radians(p.h))
        m_hy += math.cos(math.radians(p.h))

    if m_count == 0:
        return -1, -1, 0, False

    m_x /= m_count
    m_y /= m_count

    # average rotation
    m_hx /= m_count
    m_hy /= m_count
    m_h = math.degrees(math.atan2(m_hx, m_hy));

    # Now compute how good that mean is -- check how many particles
    # actually are in the immediate vicinity
    m_count = 0
    for p in particles:
        if grid_distance(p.x, p.y, m_x, m_y) < 1:
            m_count += 1

    return m_x, m_y, m_h, m_count > len(particles) * 0.95


