import math
from contour import box_measure
import numpy as np


def vision_lidar_distance_calculation(image, lidar_range_array, fov):
    """
    Calculates distance and angle using camera and lidar.

    Arguments:
        image: Image from the camera
        lidar_range_array: Array of lidar distances
        fov: Field of view of the camera

    Returns:
        distance: Distance of the object from the robot (meters)
        angle: Heading of the object's centroid with respect to the robot (degrees)
    """

    centroid = box_measure(image)
    if not centroid or len(centroid) == 0:
        print("Warning: Invalid centroid detected.")
        return None, None

    image_width = image.shape[1]

    image_center_x = image_width / 2
    pixel_diff = image_center_x - centroid[0]

    focal_length = image_width / (2 * math.tan(fov / 2))
    alpha_radians = math.atan(pixel_diff / focal_length)
    alpha = math.degrees(alpha_radians)

    lidar_index = int((alpha + 360) % 360)

    distance = lidar_range_array[lidar_index]

    return distance, alpha
