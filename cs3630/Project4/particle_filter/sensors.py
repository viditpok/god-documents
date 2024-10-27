import math
import unittest
import os
import cv2
from contour import box_measure
from setting import *
from utils import *
import numpy as np
from itertools import product

"""
Measurements of a marker, including
    * depth, angle measurement (in radians) from camera
    * range measurement from lidar
"""
class MarkerMeasure:
    # Constructor
    def __init__(self, depth:float, angle:float, lidar_range:float) -> None:
        """
        depth (float): depth of marker in camera frame.
        angle (float): angle of marker in camera frame (in radians).
        lidar_range (float): range of marker in lidar frame.
        """
        self.depth = depth
        self.angle = angle
        self.lidar_range = lidar_range

    # Function for printing marker measurement
    def __str__(self):
        return f"[depth: {self.depth},\t angle: {self.angle},\trange: {self.lidar_range}]"

    # Function for printing a list of marker measurements
    def __repr__(self):
        return f"[depth: {self.depth},\t angle: {self.angle},\trange: {self.lidar_range}]\n"

"""
Compute the depth and angle given marker centroid in stereo images.
(modified from Project2 solution code)
"""
def compute_depth_angle(centroid1, centroid2, width):        
    # Step 1: Calculate focal length of the camera
    focal_length = width / (2 * math.tan(ROBOT_CAMERA_FOV / 2))
    
    # Step 2: Distance calculation
    disparity = abs(centroid1[0] - centroid2[0])
    if disparity == 0:
        disparity = 1
    depth = ROBOT_CAMERA_BASELINE * focal_length / disparity

    # Step 3: Get ratio between fov and image width
    k = ROBOT_CAMERA_FOV / width

    # Step 4: Calculating the pixel difference between image centre and centroid1 pixel
    diff_pixel = width // 2 - centroid1[0]

    if diff_pixel < 0:
        # If marker makes negative angle with camera left
        # Step 5: Compute the angle using outputs from step 3 and 4
        theta = math.pi /2 + diff_pixel * k 

        # Step 6: Compute the angle of the centroid from the center of the robot 
        right_triangle_base = depth / math.tan(theta)

        dist_diff = right_triangle_base - ROBOT_CAMERA_BASELINE / 2

        angle = math.atan2(depth, dist_diff) - math.pi/2
    
    elif diff_pixel > 0:
        # If marker makes positive angle with camera left
        # Step 5: Compute the angle using outputs from step 3 and 4
        theta = math.pi /2 - diff_pixel * k 

        # Step 6: Compute the angle of the centroid from the center of the robot 
        right_triangle_base = depth / math.tan(theta)

        dist_diff = right_triangle_base + ROBOT_CAMERA_BASELINE / 2

        angle = math.pi/2 - math.atan2(depth, dist_diff)

    else:
        # If marker makes zero angle with camera left
        # Step 5: Compute the angle using 
        angle = math.pi/2 - math.atan2(depth, ROBOT_CAMERA_BASELINE/2)

    return depth, angle

"""
In the case of multiple markers, generate pairs of centroids in left and right images based on pixel distance.
"""
def generate_centroid_pairs(centroids_l, centroids_r):
    centroid_pairs = []
    while len(centroids_l) > 0 and len(centroids_r) > 0:
        # 1. find the (particle marker,robot marker) pair with shortest grid distance
        all_pairs = product(centroids_l, centroids_r, )
        c_l, c_r = min(all_pairs, key=lambda p: (p[0][0] - p[1][0])**2 + (p[0][1] - p[1][1])**2)
        
        # 2. add this pair to centroid_pairs and remove markers from corresponding lists
        if abs(c_l[1] - c_r[1])<20 and c_l[0] > c_r[0]:
            centroid_pairs.append((c_l, c_r))
        centroids_l.remove(c_l)
        centroids_r.remove(c_r)
        pass
    return centroid_pairs

"""
For all visible markers, compute measurements in depth, angle for stereo camera, and range for lidar.
"""
def compute_measurements(img_l: np.ndarray, img_r: np.ndarray, lidar_array: list[float]) -> list[MarkerMeasure]:
    """
    Detect markers in the images, and generate measurements of depth, angle and range for all detected markers.
    The depth, angle are measured in the camera's coordinate frame.
    The range is measureed in lidar's coordinate frame.
    Args:
        * img_l(np.ndarray with shape [height, width, 3]): image recorded from the left camera.
        * img_r(np.ndarray with shape [height, width, 3]): image recorded from the right camera.
        * lidar_array(list[float] of 360 elements): lidar array recorded at each angle (in degree) counter-clockwisely.
    Return:
        * (list[MarkerMeasure]): measurements of all detected markers in the image.
    """
    centroids_l, magnitudes_l = box_measure(img_l)
    centroids_r, magnitudes_r = box_measure(img_r)
    centroid_pairs = generate_centroid_pairs(centroids_l, centroids_r)

    width = img_l.shape[1] # value in pixels
    measurements = []
    for centroid_l, centroid_r in centroid_pairs:
        depth, angle = compute_depth_angle(centroid_l, centroid_r, width)
        angle_deg = angle * 180/math.pi
        # adjust to range of [0, 359]
        lidar_index = round(angle_deg) % 360
        lidar_range = lidar_array[lidar_index]
        if -ROBOT_CAMERA_FOV/ 2 < angle < ROBOT_CAMERA_FOV/2 and depth < 3:
            measurements.append(MarkerMeasure(depth, angle, lidar_range))
    return measurements


class TestSensor(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSensor, self).__init__(*args, **kwargs)

    def test_camera(self):
        scenario_name = "large_circle"
        lidar_arrays = read_lidar(os.path.join(LIDAR_PATH, f"lidar_{scenario_name}.csv"))
        image_folder = os.path.join(IMAGE_PATH, scenario_name)
        step = 785
        img_l, img_r = read_images(image_folder, step)
        measurements = compute_measurements(img_l, img_r, lidar_arrays[step])
        print(measurements)

if __name__ == '__main__':
    unittest.main()
