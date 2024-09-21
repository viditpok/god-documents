import math
from contour import box_measure


def vision_only_depth_calculation(image1, image2, fov, camera_translation):
    """
    Arguments:
        image1: image from the first camera
        image2: image from the second camera
        fov: field of view of the cameras, both cameras have the same value in our robot
        camera_translation: horizontal displacement between the two cameras (c)

    Returns:
        depth: perpendicular distance of the object from the two cameras (meters)
        beta: angle (heading) of the marker's centroid with respect to the robot center (degrees)
    """

    centroid1 = box_measure(image1)
    centroid2 = box_measure(image2)

    if (
        centroid1 is None
        or centroid2 is None
        or len(centroid1) == 0
        or len(centroid2) == 0
    ):
        return None, None

    image_width = image1.shape[1]
    focal_length = image_width / (2 * math.tan(fov / 2))

    x_disparity = abs(centroid1[0] - centroid2[0])

    if x_disparity == 0:
        return None, None

    depth = (camera_translation * focal_length) / x_disparity

    image_center = image_width / 2
    pixel_offset1 = centroid1[0] - image_center
    alpha_radians = math.atan(pixel_offset1 / focal_length)
    alpha = math.degrees(alpha_radians)

    third_angle = math.atan(camera_translation / (2 * depth))
    third_angle = math.degrees(third_angle)
    beta = third_angle - alpha

    if centroid1[0] < image_center:
        beta = abs(beta)
    else:
        beta = -abs(beta)

    return depth, beta
