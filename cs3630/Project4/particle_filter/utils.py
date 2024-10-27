import os
import csv
from geometry import SE2, Point
import math
import cv2
import numpy as np
from setting import *

# ------------------------------------------------------------------------
def read_poses(file_path: str) -> list[SE2]:
    """
    Read the ground-truth poses of the robot stored through running Webots.
    Args:
        * file_path(str): path to the pose storage file.
    Return:
        * (list[SE2]): ground-truth path of each time step.
    """
    poses = []
    poses.append(None)     # append none for step 0
    with open(file_path) as csvfile:
        pose_reader = csv.reader(csvfile)
        rows = [row for row in pose_reader]
        for row in rows[1:]:
            x = float(row[1])
            y = float(row[2])
            h = float(row[3])
            poses.append(SE2(x, y, h))
    return poses

# ------------------------------------------------------------------------
def read_lidar(file_path: str) -> list[list[float]]:
    """
    Read the stored lidar measurements stored through running Webots.
    Args:
        * file_path(str): path to the lidar measurements storage file.
    Return:
        * (list[list[float]]): the lidar measurements (as a list of distances
          at each angle) of each time step.
    """
    lidar_arrays = []
    lidar_arrays.append(None)    # append none for step 0
    with open(file_path) as csvfile:
        lidar_reader = csv.reader(csvfile)
        rows = [row for row in lidar_reader]
        for row in rows[1:]:
            lidar_array = [float(d) for d in row]
            lidar_arrays.append(lidar_array)
    return lidar_arrays

# ------------------------------------------------------------------------
def read_images(folder_path: str, step: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the camera images stored through running Webots.
    Args:
        * folder_path (str): path to the image storage folder.
        * step (int): time step.
    Return:
        * (tuple[np.ndarray, np.ndarray]): left and right images recorded by
          the cameras at the specified time step.
    """
    img_l = cv2.imread(os.path.join(folder_path, str(step) + '_camera_l.jpg'))
    img_r = cv2.imread(os.path.join(folder_path, str(step) + '_camera_r.jpg'))
    return img_l, img_r

# ------------------------------------------------------------------------
def read_odometry(file_path: str):
    """
    Read the odometry stored through running Webots.
    Args:
        * file_path(str): path to the odometry storage file.
    Return:
        * (list[tuple[float, float, float]]):
            * first entry: left wheel speed in radian/second.
            * second entry: right wheel speed in radian/second.
            * thrid entry: time step duration.
    """
    odometry = []
    odometry.append(None)    # append none for step 0
    with open(file_path) as csvfile:
        odometry_reader = csv.reader(csvfile)
        rows = [row for row in odometry_reader]
        idx = int(0)
        for row in rows[1:]:
            step_data = [float(d) for d in row]
            idx+=1
            odometry.append((step_data[1], step_data[2], step_data[3]))
    return odometry

# ------------------------------------------------------------------------
def integrate_odo(env, start_step, end_step, odometry_steps):
    """
    Integrate odometry computed from differential drive of several time steps.
    """
    transform_odo = SE2(0, 0, 0)
    for k in range(start_step, end_step):
        omega_l, omega_r, dt = odometry_steps[k]
        odo_k = env.diff_drive_odometry(omega_l, omega_r, dt)
        transform_odo = transform_odo.compose(odo_k)
    return transform_odo


# ------------------------------------------------------------------------
# Following functions are helpers for geometrical calcuations. You do not 
# need to use them for implementing the particle filter algorithm.
# ------------------------------------------------------------------------

# ------------------------------------------------------------------------
def axis_angle_to_rotation_matrix(axis, angle):
    """
    Convert an axis-angle representation to a rotation matrix.
    """
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    return np.array([
        [x*x*C + c, x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, y*y*C + c, y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, z*z*C + c]
    ])

# ------------------------------------------------------------------------
def rotation_matrix_to_euler_angles(R):
    """
    Convert a rotation matrix to Euler angles (roll, pitch, yaw).
    """
    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

# ------------------------------------------------------------------------
def axis_angle_to_euler_angles(rotation):
    """
    Convert axis-angle rotation to Euler angles.
    """
    axis = rotation[0:3]
    angle = rotation[3]
    R = axis_angle_to_rotation_matrix(axis, angle)
    return rotation_matrix_to_euler_angles(R)

# ------------------------------------------------------------------------
def read_marker_positions(wbt_file_path):
    """
    Read the positions of all markers from the webot world file.
    Return the positions of all markers in the world frame as a list.
    """
    reading_direction_panel = False
    marker_positions = []
    with open(wbt_file_path, "r") as file:
        for line in file:
            if "DirectionPanel" in line:
                reading_direction_panel = True
                continue
            if reading_direction_panel and "translation" in line:
                x, y = line.strip().split()[1:3]
                marker_positions.append(Point(float(x), float(y)))
                reading_direction_panel = False
                continue
            else:
                reading_direction_panel = False
    return marker_positions

# ------------------------------------------------------------------------
def read_walls(wbt_file_path):
    """
    Read the pose and dimensions of all wall from the webot world file.
    Each wall is represented as a rectangle.
    The wall coordinate frame is positioned at the center of the rectangle.
    The dimension of the wall is in the form of [width, height] of the rectangle.
    """
    reading_wall = False
    wall_poses = []
    wall_dimensions = []
    with open(wbt_file_path, "r") as file:
        for line in file:
            if "Wall" in line:
                reading_wall = True
                continue
            if reading_wall:
                if "translation" in line:
                    x, y = line.strip().split()[1:3]
                    translation = Point(float(x), float(y))
                    continue
                if "rotation" in line:
                    rotation = line.strip().split()[1:5]
                    for i in range(len(rotation)):
                        rotation[i] = float(rotation[i])
                    h = axis_angle_to_euler_angles(rotation)[2]
                    wall_poses.append(SE2(translation.x, translation.y, h))
                    continue
                if "size" in line:
                    dimension = line.strip().split()[1:4]
                    for i in range(len(dimension)):
                        dimension[i] = float(dimension[i])
                    wall_dimensions.append(dimension)
                    reading_wall = False
                    continue

    return wall_poses, wall_dimensions

# ------------------------------------------------------------------------
def line_intersection(p1:Point, p2:Point, p3:Point, p4:Point) -> Point:
    """
    Compute the intersection point of two lines (p1-p2) and (p3-p4).
    """
    # Denominator for ua and ub
    denominator = ((p4.y - p3.y) * (p2.x - p1.x)) - ((p4.x - p3.x) * (p2.y - p1.y))

    # Make sure the denominator is not zero to avoid division by zero
    if denominator == 0:
        return None  # Lines are parallel

    # ua and ub are the fractions of the line where the intersection occurs
    ua = (((p4.x - p3.x) * (p1.y - p3.y)) - ((p4.y - p3.y) * (p1.x - p3.x))) / denominator
    ub = (((p2.x - p1.x) * (p1.y - p3.y)) - ((p2.y - p1.y) * (p1.x - p3.x))) / denominator

    # If the fractions are between 0 and 1, the lines intersect
    if 0 <= ua <= 1 and 0 <= ub <= 1:
        # Calculate the intersection point
        intersection_x = p1.x + ua * (p2.x - p1.x)
        intersection_y = p1.y + ua * (p2.y - p1.y)
        return Point(intersection_x, intersection_y)
    else:
        return None  # Lines do not intersect within the line segments


# ------------------------------------------------------------------------
def point_on_segment(p1, p2, p3):
    """
    Check if point (p3) lies on line segment (p1-p2).
    """
    return min(p1.x, p2.x) <= p3.x <= max(p1.x, p2.x) and min(p1.y, p2.y) <= p3.y <= max(p1.y, p2.y)

# ------------------------------------------------------------------------
def line_segment_intersect(p1, p2, p3, p4):
    p_intersect = line_intersection(p1, p2, p3, p4)
    if p_intersect is None:
        return False
    return point_on_segment(p1, p2, p_intersect) and point_on_segment(p3, p4, p_intersect)

# ------------------------------------------------------------------------
def line_rectangle_intersect(p1:Point, p2:Point, rect_pose:SE2, rect_dim: list[float]):
    """
    Check if line segment (p1-p2) intersects with a rectangle.
    """
    rect_width, rect_height, _ = rect_dim
    a, b = rect_width/2, rect_height/2
    p1_l = Point(-a, -b)
    p2_l = Point(a, -b)
    p3_l = Point(a, b)
    p4_l = Point(-a, b)
    p1_w = rect_pose.transform_point(p1_l)
    p2_w = rect_pose.transform_point(p2_l)
    p3_w = rect_pose.transform_point(p3_l)
    p4_w = rect_pose.transform_point(p4_l)
    if line_segment_intersect(p1, p2, p1_w, p2_w):
        return True
    if line_segment_intersect(p1, p2, p3_w, p2_w):
        return True
    if line_segment_intersect(p1, p2, p1_w, p4_w):
        return True
    if line_segment_intersect(p1, p2, p3_w, p4_w):
        return True
    return False

# ------------------------------------------------------------------------
def point_in_rectangle(point:Point, rect_pose:SE2, rect_dim: list[float]):
    """
    Check if a point is in the rectangle.
    """
    rect_width, rect_height, _ = rect_dim
    a, b = rect_width/2, rect_height/2
    point_local = rect_pose.inverse().transform_point(point)
    if -a < point_local.x < a and -b < point_local.y < b:
        return True
    return False

# ------------------------------------------------------------------------
def calculate_pose(gps_value, compass_value):
    """
    Transform from translation (x, y, z) and north direction into SE2(x, y, h)
    """
    x, y, _ = gps_value
    h = math.atan2(compass_value[0], compass_value[1])
    return SE2(x, y, h)

# ------------------------------------------------------------------------
def rotate_point(x, y, heading):
    """
    Rotate a point (x, y) by a given angle in radians.
    """
    xr, yr = 0, 0
    c = math.cos(heading)
    s = math.sin(heading)
    xr = x * c + y * -s
    yr = x * s + y * c
    return xr, yr

# ------------------------------------------------------------------------
def diff_heading_rad(heading1, heading2):
    """
    heading angle difference = heading1 - heading2
    return value always in range (-180, 180] in deg
    """
    dh = heading1 - heading2
    while dh > math.pi:
        dh -= 2*math.pi
    while dh <= -math.pi:
        dh += 2*math.pi
    return dh

# ------------------------------------------------------------------------
def check_confident(est_pose, robot_pose):
    distance_trans = math.sqrt((est_pose.x - robot_pose.x)**2 + (est_pose.y - robot_pose.y)**2)
    distance_rot = math.fabs(diff_heading_rad(est_pose.h, robot_pose.h))
    return distance_trans < Err_trans and distance_rot < math.radians(Err_rot)

# ------------------------------------------------------------------------
def pose_distance(pose1, pose2):
    """
    Distance between two poses that incorporates both position and orientation.
    """
    diff_x = pose1.x - pose2.x
    diff_y = pose1.y - pose2.y
    diff_h = diff_heading_rad(pose1.h, pose2.h)
    distance = math.sqrt(diff_x**2 + diff_y**2 + diff_h**2)
    return distance

# ------------------------------------------------------------------------
def poses_within_dist(ref_pose, poses, distance):
    """
    Return neighboring poses within specified distance to the reference pose.
    """
    neighbor_poses = []
    for pose in poses:
        if pose_distance(ref_pose, pose) < distance:
            neighbor_poses.append(pose)
    return neighbor_poses
