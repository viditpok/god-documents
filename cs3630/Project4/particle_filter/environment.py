# Vidit Pokharna

import numpy as np
import math
from geometry import SE2, Point
from setting import *
from utils import read_marker_positions, read_walls, point_in_rectangle, line_rectangle_intersect
from sensors import MarkerMeasure
import json

# Compute the detection failure rate given the ground-truth measurement.
def compute_detection_failure_rate(marker_measure: MarkerMeasure):
    """
    Generally, it is more likely to have a detection failure when the marker is near the edge of an image,
    or the marker is too close to the camera.
    Args:
        * marker_measure (MarkerMeasure): ground-truth depth, angle, range measurements of a marker.
    Return:
        * (float within range[0, 1]): probability that there is a detection failure of the marker by the robot.
    """
    edge_angle = min(abs(marker_measure.angle + ROBOT_CAMERA_FOV/2), abs(marker_measure.angle-ROBOT_CAMERA_FOV/2))
    c = edge_angle / marker_measure.lidar_range

    threshold = 0.1
    if c < threshold:
        return np.interp(c, [0, 0.1], [EDGE_DETECTION_FAILURE_RATE, NOMINAL_DETECTION_FAILURE_RATE])
    else:
        return NOMINAL_DETECTION_FAILURE_RATE

def compute_spurious_detection_rate(marker_measure: MarkerMeasure):
    """
    Assume constant spurious detection rate.
    Args:
        * marker_measure (MarkerMeasure): measurements of a marker by the robot.
    Return:
        * (float within range[0, 1]): probability that the observation is a spurious detection.
    """
    return NOMINAL_SPURIOUS_DETECTION_RATE

# Environment containing robot parameters and marker info
class Environment:
    # Constructor
    def __init__(self, config_file_path: str):
        """
        The attributes includes:
            * robot_radius (float): radius of the robot
            * wheel_radius (float): radius of the wheels
            * fov (float): field of view of the cameras, expressed in radians
            * baseline (float): distance between two cameras
            * T_r_c (SE2): T^r_c, pose of camera coordinate expressed in the robot coordinate frame.
            * T_r_l (SE2): T^l_r, pose of lidar coordinate expressed in the robot coordinate frame.
            * markers (list[Point]): positions of markers in the world coordinate frame.
            * wall_poses (list[SE2]): poses of wall obstacles in the world coordinate frame.
            * wall_dimensions (list[tuple[float, float]]): length, width of wall obstacles.
            * x_min (float): smallest possible x coordinate of robot pose in the environment.
            * x_max (float): largest possible x coordinate of robot pose in the environment.
            * y_min (float): smallest possible y coordinate of robot pose in the environment.
            * y_max (float): largest possible y coordinate of robot pose in the environment.
        """
        with open(config_file_path, "r") as file:
            configs = json.load(file)
        self.axle_length = configs["axle_length"]
        self.robot_radius = configs["robot_radius"]
        self.wheel_radius = configs["wheel_radius"]
        self.fov = configs["fov"]
        self.baseline = configs["camera_baseline"]
        self.T_r_c = SE2(*configs["camera_pose"])
        self.T_r_l = SE2(*configs["lidar_pose"])
        world_file = os.path.join(WORLD_PATH, configs["world_file"])
        self.markers = read_marker_positions(world_file)
        self.wall_poses, self.wall_dimensions = read_walls(world_file)
        x_limits = configs["x_range"]
        y_limits = configs["y_range"]
        self.x_min = x_limits[0] + self.robot_radius
        self.x_max = x_limits[1] - self.robot_radius
        self.y_min = y_limits[0] + self.robot_radius
        self.y_max = y_limits[1] - self.robot_radius

    # Generate a random pose free of obstacles
    def random_free_pose(self) -> SE2:
        while True:
            x = np.random.uniform(self.x_min, self.x_max)
            y = np.random.uniform(self.y_min, self.y_max)
            h = np.random.uniform(-np.pi, np.pi)
            pose = SE2(x, y, h)
            if (self.is_free(pose)):
                return pose

    # Check if a pose is free from collision with obstacles
    def is_free(self, pose: SE2) -> bool:
        if pose.x < self.x_min or pose.x > self.x_max:
            return False
        if pose.y < self.y_min or pose.y > self.y_max:
            return False
        for wall_pose, wall_dim in zip(self.wall_poses, self.wall_dimensions):
            if point_in_rectangle(pose.position(), wall_pose, wall_dim):
                return False
        return True

    # Check if a point is visible from the pose in the field of view without occlusions.
    def visible(self, pose: SE2, point: Point) -> bool:
        """
        Args:
            pose (SE2): pose of the sensor in the world frame.
            point (Point): target point in the world frame.
        Return:
            (bool): if the point is visible to the sensor at the pose.
        """
        point_sensor = pose.inverse().transform_point(point)
        angle = np.arctan2(point_sensor.y, point_sensor.x)
        if abs(angle) > self.fov/2:
            return False
        for wall_pose, wall_dim in zip(self.wall_poses, self.wall_dimensions):
            if line_rectangle_intersect(pose.position(), point, wall_pose, wall_dim):
                return False
        return True

    # Generate expected ground-truth marker measurements given the pose of the robot
    def read_marker_measures(self, T_w_r: SE2) -> list[MarkerMeasure]:
        """
        With a given robot pose, generate the ground-truth measurements of all visible markers.
        Hints:
            * You can use the visible() function to check if a marker is visible to the robot.
            * You can find the location of all markers in the world frame in self.markers.
            * You may find the following naming conventions helpful:
                - T_a_b to represent an SE(2) T^a_b (refer to Lecture 10).
                - P_a to represent a point expressed in the coordinate frame a.
                - Let w refer to world frame, T_w_a can be used to represented the pose of coordinate frame a.
            * Then, the following rules applies:
                - T_a_c = T_a_b.compose(T_b_c)
                - T_c_a = T_a_c.inverse()
                - P_a = T_a_b.transform_point(P_b)
        Args:
            * T_w_r (SE2): pose of the robot in the world frame.
        Return:
            * (list[MarkerMeasure]): List of measurements of all markers. Ordering does not matter.
        """
        marker_measures = []
        
        # Compute transformations from world frame to camera and lidar frames
        T_w_c = T_w_r.compose(self.T_r_c)
        T_w_l = T_w_r.compose(self.T_r_l)

        for Pm_w in self.markers:
            Pm_c = T_w_c.inverse().transform_point(Pm_w)
            
            depth = Pm_c.x
            angle = math.atan2(Pm_c.y, Pm_c.x)
            
            Pm_l = T_w_l.inverse().transform_point(Pm_w)
            lidar_range = math.sqrt(Pm_l.x**2 + Pm_l.y**2)

            if self.visible(T_w_c, Pm_w):
                marker_measures.append(MarkerMeasure(depth=depth, angle=angle, lidar_range=lidar_range))

        return marker_measures

    # Kinematics of a differential drive robot
    def diff_drive_kinematics(self, omega_l: float, omega_r: float) -> tuple[float, float]:
        """
        For a differntial drive robot, compute the forward and rotation speed given wheel speeds.
        Hint:
            * You can find the axle length and wheel radius in self.axle_length, self.wheel_radius.
        Args:
            * omega_l (float): rotational speed of left wheel (in radian/second).
            * omega_r (float): rotational speed of right wheel (in radian/second).
        Return:
            * (tuple[float, float]):
                - the first entry is the forward speed (in meter/second).
                - the second entry is the counterclockwise rotational speed of the robot (in radian/second).
        """
        v_x = self.wheel_radius * (omega_r + omega_l) / 2
        omega = self.wheel_radius * (omega_r - omega_l) / self.axle_length
        return v_x, omega

    # Compute the odometry of a differential drive robot
    def diff_drive_odometry(self, omega_l: float, omega_r: float, dt: float) -> SE2:
        """
        Compute the odometry the robot travels within a time step.
        Args:
            * omega_l (float): rotational speed of left wheel (in radian/second).
            * omega_r (float): rotational speed of right wheel (in radian/second).
            * dt (float): time step duration (in second).
        Return:
            *(SE2): relative transform of robot pose T^{k}_{k+1}, where k denotes the index of time step.
        """
        v_x, omega = self.diff_drive_kinematics(omega_l, omega_r)
        if math.fabs(omega) < 1e-5:
            return SE2(v_x * dt, 0, omega * dt)
        curve_radius = v_x / omega
        curve_angle = omega * dt
        dx = curve_radius * math.sin(curve_angle)
        dy = curve_radius * (1-math.cos(curve_angle))
        dh = curve_angle
        return SE2(dx, dy, dh)
