from grid import *
from particle import Particle
from utils import *
import setting
import numpy as np

np.random.seed(setting.RANDOM_SEED)
from itertools import product
from typing import List, Tuple


def create_random(count: int, grid: CozGrid) -> List[Particle]:
    """
    Returns a list of <count> random Particles in free space.

    Parameters:
        count: int, the number of random particles to create
        grid: a Grid, passed in to motion_update/measurement_update
            see grid.py for definition

    Returns:
        List of Particles with random coordinates in the grid's free space.
    """
    particles = []

    for _ in range(count):
        x, y = grid.random_free_place()
        heading = random.uniform(0, 360)
        particle = Particle(x=x, y=y, heading=heading)
        particles.append(particle)

    return particles


def motion_update(
    old_particles: List[Particle], odometry_measurement: Tuple, grid: CozGrid
) -> List[Particle]:
    """
    Implements the motion update step in a particle filter.
    Refer to setting.py and utils.py for required functions and noise parameters

    NOTE: the GUI will crash if you have not implemented this method yet. To get around this, try setting new_particles = old_particles.

    Arguments:
        old_particles: List
            list of Particles representing the belief before motion update p(x_{t-1} | u_{t-1}) in *global coordinate frame*
        odometry_measurement: Tuple
            noisy estimate of how the robot has moved since last step, (dx, dy, dh) in *local robot coordinate frame*

    Returns:
        a list of NEW particles representing belief after motion update \tilde{p}(x_{t} | u_{t})
    """
    motion_particles = []
    
    for p in old_particles:
        local_x, local_y = odometry_measurement[:2]
        x, y = rotate_point(local_x, local_y, p.h)
        p.x += x
        p.y += y
        p.x = add_gaussian_noise(p.x, setting.ODOM_TRANS_SIGMA)
        p.y = add_gaussian_noise(p.y, setting.ODOM_TRANS_SIGMA)
        p.h += odometry_measurement[2]
        p.h = add_gaussian_noise(p.h, setting.ODOM_HEAD_SIGMA)
        motion_particles.append(p)
        
    return motion_particles


def generate_marker_pairs(
    robot_marker_list: List[Tuple], particle_marker_list: List[Tuple]
) -> List[Tuple]:
    """Pair markers in order of closest distance

    Arguments:
    robot_marker_list -- List of markers observed by the robot: [(x1, y1, h1), (x2, y2, h2), ...]
    particle_marker_list -- List of markers observed by the particle: [(x1, y1, h1), (x2, y2, h2), ...]

    Returns: List[Tuple] of paired robot and particle markers: [((xr1, yr1, hr1), (xp1, yp1, hp1)), ((xr2, yr2, hr2), (xp2, yp2, hp2)), ...]
    """
    marker_pairs = []

    while robot_marker_list and particle_marker_list:

        closest_pair = None
        closest_dist = float("inf")

        for robot_marker, particle_marker in product(
            robot_marker_list, particle_marker_list
        ):
            distance = grid_distance(
                robot_marker[0], robot_marker[1], particle_marker[0], particle_marker[1]
            )
            if distance < closest_dist:
                closest_dist = distance
                closest_pair = (robot_marker, particle_marker)

        marker_pairs.append(closest_pair)
        robot_marker_list.remove(closest_pair[0])
        particle_marker_list.remove(closest_pair[1])

    return marker_pairs


def marker_likelihood(robot_marker: Tuple, particle_marker: Tuple) -> float:
    """Calculate likelihood of reading this marker using Gaussian PDF.
    The standard deviation of the marker translation and heading distributions
    can be found in setting.py

    Some functions in utils.py might be useful in this section

    Arguments:
    robot_marker -- Tuple (x,y,theta) of robot marker pose
    particle_marker -- Tuple (x,y,theta) of particle marker pose

    Returns: float probability
    """
    x_r, y_r, h_r = robot_marker
    x_p, y_p, h_p = particle_marker

    distance = grid_distance(x_r, y_r, x_p, y_p)

    heading_diff = diff_heading_deg(h_r, h_p)

    trans_prob = np.exp(-0.5 * (distance / setting.MARKER_TRANS_SIGMA) ** 2)
    heading_prob = np.exp(-0.5 * (heading_diff / setting.MARKER_HEAD_SIGMA) ** 2)

    return trans_prob * heading_prob


def particle_likelihood(
    robot_marker_list: List[Tuple], particle_marker_list: List[Tuple]
) -> float:
    """Calculate likelihood of the particle pose being the robot's pose

    Arguments:
    robot_marker_list -- List of markers (x,y,theta) observed by the robot
    particle_marker_list -- List of markers (x,y,theta) observed by the particle

    Returns: float probability
    """

    l = 1.0
    marker_pairs = generate_marker_pairs(robot_marker_list, particle_marker_list)
    if not marker_pairs:
        return 0
    for robot_marker, particle_marker in marker_pairs:

        pair_likelihood = marker_likelihood(robot_marker, particle_marker)
        l *= pair_likelihood
    return l


def measurement_update(
    particles: List[Particle], measured_marker_list: List[Tuple], grid: CozGrid
) -> List[Particle]:
    """Particle filter measurement update

    NOTE: the GUI will crash if you have not implemented this method yet. To get around this, try setting measured_particles = particles.

    Arguments:
    particles -- input list of particle represents belief \tilde{p}(x_{t} | u_{t})
            before measurement update (but after motion update)

    measured_marker_list -- robot detected marker list, each marker has format:
            measured_marker_list[i] = (rx, ry, rh)
            rx -- marker's relative X coordinate in robot's frame
            ry -- marker's relative Y coordinate in robot's frame
            rh -- marker's relative heading in robot's frame, in degree

            * Note that the robot can only see markers which is in its camera field of view,
            which is defined by ROBOT_CAMERA_FOV_DEG in setting.py
                            * Note that the robot can see mutliple markers at once, and may not see any one

    grid -- grid world map, which contains the marker information,
            see grid.py and CozGrid for definition
            Can be used to evaluate particles

    Returns: the list of particles represents belief p(x_{t} | u_{t})
            after measurement update
    """
    measuredMarkerListLength, num, weights = len(measured_marker_list), 0, []
    for particle in particles:
        if measuredMarkerListLength:
            robotMarkers = particle.read_markers(grid=grid)
            markersLength = len(robotMarkers)
            if not markersLength:
                weights.append(0)
            elif not grid.is_free(x=particle.x, y=particle.y):
                weights.append(0)
                num += 1
            else:
                lst, confidence = [], 1
                for measured_marker in measured_marker_list:
                    if markersLength:
                        worst = min(robotMarkers, key=lambda m: grid_distance(x1=measured_marker[0], y1=measured_marker[1], x2=m[0], y2=m[1]))
                        robotMarkers.remove(worst)
                        markersLength = len(robotMarkers)
                        lst.append((measured_marker, worst))
                for marker, worst in lst:
                    confidence *= math.exp(-1 * (math.pow(grid_distance(marker[0], marker[1], worst[0], worst[1]), 2) / 
                                  (2 * math.pow(setting.MARKER_TRANS_SIGMA, 2)) + 
                                  math.pow(diff_heading_deg(marker[2], worst[2]), 2) / 
                                  (2 * math.pow(setting.MARKER_HEAD_SIGMA, 2))))
                weights.append(confidence)
    particleLength, weightsSum = len(particles), sum(weights)
    if measuredMarkerListLength < 1 or weightsSum == 0:
        weights = particleLength * [(1 / float(particleLength))]
    else:
        weights = [weight / weightsSum for weight in weights]
    beliefList = [Particle(x=particle.x, y=particle.y, heading=particle.h) 
                  for particle in np.random.choice(particles, size=(particleLength - min(particleLength, 50 + num)), p=weights).tolist()]
    for x, y in [grid.random_free_place() for i in range(min(particleLength, 50 + num))]:
        beliefList.append(Particle(x=x, y=y))
    return beliefList