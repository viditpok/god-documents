# Vidit Pokharna

import numpy as np
from setting import *
np.random.seed(RANDOM_SEED)
from itertools import product
from environment import *
from geometry import SE2
from sensors import MarkerMeasure
from utils import *
import math

# ------------------------------------------------------------------------
def create_random(count:int, env:Environment):
    """
    Create a list of random particles in the environment.
    """
    return [env.random_free_pose() for _ in range(count)]

# ------------------------------------------------------------------------
def motion_update(particles: list[SE2], odometry: SE2) -> list[SE2]:
    """
    Motion update that moves the particles according to the odometry.
    Args:
        * particles (list[SE2]): a list of particles before the motion update.
        * odometry (SE2): relative transform of the robot pose, i.e., T^{k}_{k+1} with k being the time step number.
    Return:
        (list[SE2]): list of particles after the motion update.
    """
    new_particles = []
    for particle in particles:
        noisy_odo = odometry.add_noise(MOTION_TRANS_SIGMA, MOTION_TRANS_SIGMA, MOTION_HEAD_SIGMA)
        new_particle = particle.compose(noisy_odo)
        new_particles.append(new_particle)
    return new_particles

# ------------------------------------------------------------------------
def generate_marker_pairs(robot_marker_measures: list[MarkerMeasure],
                          particle_marker_measures: list[MarkerMeasure]) -> tuple[list, list, list]:
    """ Pair markers in order of closest distance
        Args:
            * robot_marker_measures (list[MarkerMeasure]) -- List of marker measures observed by the robot.
            * particle_marker_measures (list[MarkerMeasure]) -- List of marker measures observed by the particle.
        Return: 
            * (tuple[list[tuple[MarkerMeasure, MarkerMeasure]], list[MarkerMeasure], list[MarkerMeasure]]):
                - the first entry corresponds to a list of matched marker pairs.
                - the second entry is a list of unmatched markers in robot_marker_measures.
                - the third entry is a list of unmatched markers in particle_marker_measures.
    """
    marker_pairs = []
    robot_marker_measures = robot_marker_measures.copy()
    while len(robot_marker_measures) > 0 and len(particle_marker_measures) > 0:
        all_pairs = product(robot_marker_measures, particle_marker_measures, )
        r_m, p_m = min(all_pairs, key=lambda p: abs(p[0].angle - p[1].angle))
        
        marker_pairs.append((r_m, p_m))
        robot_marker_measures.remove(r_m)
        particle_marker_measures.remove(p_m)
        pass
    return marker_pairs, robot_marker_measures, particle_marker_measures

# ------------------------------------------------------------------------
def marker_likelihood(robot_marker: MarkerMeasure, particle_marker: MarkerMeasure) -> float:
    """ Calculate likelihood of reading this marker using Gaussian PDF.
        Args:
            * robot_marker(MarkerMeasure): measurements of the marker by the robot.
            * particle_marker(MarkerMeasure): measurements of the marker by the particle.
        Return:
            (float): likelihood of the marker pair.
    """
    depth_diff = abs(robot_marker.depth - particle_marker.depth)
    angle_dff = abs(robot_marker.angle - particle_marker.angle)
    range_diff = abs(robot_marker.lidar_range - particle_marker.lidar_range)

    # calculate the likelihood of this marker using the gaussian pdf
    exp1 = (depth_diff**2)/(2*CAMERA_DEPTH_SIGMA**2)
    exp2 = (angle_dff**2)/(2*CAMERA_HEADING_SIGMA**2)
    exp3 = (range_diff**2)/(2*LIDAR_RANGE_SIGMA**2)
    l = math.exp(-(exp1+exp2+exp3))
    return l

# ------------------------------------------------------------------------
def particle_likelihood(robot_marker_list: list[MarkerMeasure], particle_marker_list: list[MarkerMeasure]) -> float:
    """ Calculate likelihood of the particle pose being the robot's pose.
        Hint:
            * You can treat the unmatched particle marker measures as detection failures. It indicates that the
              robot fails to detect the marker that it is supposed to observe.
            * You can treat the unmatched robot marker measures as spurious detections. It indicates that the
              robot generates a false detection of a marker that does not exist.
            * We have provided the functions "compute_spurious_detection_rate" and "compute_detection_failure_rate"
              that compute the spurious detection rate and the detection failure rate.
            * We have provided implementations of "generate_marker_pairs", "marker_likelihood", as you have already
              implemented them in Project 3.
        Args:
            * robot_marker_list (list[MarkerMeasure]): List of markers observed by the robot.
            * particle_marker_list (list[MarkerMeasure]): List of markers observed by the particle.
        Returns:
            * (float): likelihood of the paritcle.
    """
    l = 1.0
    marker_pairs, unmatched_robot_markers, unmatched_particle_markers = generate_marker_pairs(robot_marker_list, particle_marker_list)
    
    for robot_marker, particle_marker in marker_pairs:
        l *= marker_likelihood(robot_marker, particle_marker)
        
    for unmatched_robot_marker in unmatched_robot_markers:
        l *= compute_spurious_detection_rate(unmatched_robot_marker)

    for unmatched_particle_marker in unmatched_particle_markers:
        l *= 1 - compute_detection_failure_rate(unmatched_particle_marker)

    return l

# ------------------------------------------------------------------------
def comptue_particle_weights(particles:list[SE2], robot_marker_measures:list[MarkerMeasure], env:Environment) -> list[float]:
    """
    Comptues the importance of the particles given the robot marker measures.
    Args
        * particles (list[SE2]): all particles.
    Returns
        * (list[float]): importance weights corresponding to particles.
    """
    particle_weights = []
    for particle in particles:
        particle_marker_measures = env.read_marker_measures(particle)
        weight = particle_likelihood(robot_marker_measures, particle_marker_measures)
        particle_weights.append(weight)
    return particle_weights

# ------------------------------------------------------------------------
def resample_particles(particles:list[SE2], particle_weights:list[float], env:Environment)->list[SE2]:
    """
    Resample particles using the provided importance weights of particles.
    Args:
        particles(list[SE2]): list of particles to sample from.
        particle_weights(list[float]): importance weights corresponding to particles.
    Return:
        (list[SE2]): resampled particles according to weights.
    """
    # normalize the particle weights
    weight_sum = float(sum(particle_weights))
    if weight_sum < 1e-5:
        return create_random(PARTICLE_COUNT, env)
    norm_weights = [i / weight_sum for i in particle_weights]

    # resample remaining particles using the computed particle weights
    measured_particles = np.random.choice(particles, PARTICLE_COUNT, p=norm_weights).tolist()
    return measured_particles

# ------------------------------------------------------------------------
class ParticleFilter:
    # Constructor
    def __init__(self, env: Environment):
        self.env = env
        self.particles = create_random(PARTICLE_COUNT, env)

    # Update the estimates using motion odometry and sensor measurements
    def update(self, odometry: SE2, marker_measures: list[MarkerMeasure]) -> None:
        """
        Update the particles through motion update and measurement update.
        Hint:
            * You can use function compute_measurements to generate the depth, angle, range measures.
        Args:
            * odometry (SE2): relative transform of the robot pose, i.e., T^{k}_{k+1} with k being the time step number.
            * marker_measures (list[MarkerMeasure]): depth, angle, range measurements of markers observed by the robot.
        Return: None
        """
        motion_particles = motion_update(self.particles, odometry)
        motion_particle_weights = comptue_particle_weights(motion_particles, marker_measures, self.env)
        new_particles = resample_particles(motion_particles, motion_particle_weights, self.env)
        self.particles = new_particles

    # compute the best pose estimate
    def compute_best_estimate(self) -> SE2:
        """
        Compute the best estimate using the particles. Outliers are ignored.
        Return:
            * (SE2): best estimated robot pose.
        """
        # comptue average pose
        mean_pose = SE2.mean(self.particles)
        # filter out outliers
        neighbor_distance = 0.1
        neighbor_poses = []
        while len(neighbor_poses) < PARTICLE_COUNT * 0.05:
            neighbor_distance *= 2
            neighbor_poses = poses_within_dist(mean_pose, self.particles, neighbor_distance)
        best_estimate = SE2.mean(neighbor_poses)
        return best_estimate
