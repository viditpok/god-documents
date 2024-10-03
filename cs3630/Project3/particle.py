import setting
import random
random.seed(setting.RANDOM_SEED)
import math

from utils import *
from grid import *




""" Particle class (base class for robot)
    A class for particle, each particle contains x, y, and heading information
"""
class Particle(object):

    # data members
    x = "X coordinate in world frame"
    y = "Y coordinate in world frame"
    h = "Heading angle in world frame in degree. h = 0 when robot's head (camera) points to positive X"

    # functions members

    def __init__(self, x, y, heading=None):
        if heading is None:
            heading = random.uniform(0, 360)
        self.x = x
        self.y = y
        self.h = heading

    def __repr__(self):
        return "(x = %f, y = %f, heading = %f deg)" % (self.x, self.y, self.h)

    @property
    def xy(self):
        return self.x, self.y

    @property
    def xyh(self):
        return self.x, self.y, self.h

    def read_markers(self, grid):
        """ Helper function to simulate markers measurements by robot's camera
            Only markers in robot's camera view (in FOV) will be in the list

            Arguments:
            grid -- map grid with marker information

            Return: robot detected marker list, each marker has format:
                    measured_marker_list[i] = (rx, ry, rh)
                    rx -- marker's relative X coordinate in robot's frame
                    ry -- marker's relative Y coordinate in robot's frame
                    rh -- marker's relative heading in robot's frame, in degree
        """
        marker_list = []
        for marker in grid.markers:
            m_x, m_y, m_h = parse_marker_info(marker[0], marker[1], marker[2])
            # rotate marker into robot frame
            mr_x, mr_y = rotate_point(m_x - self.x, m_y - self.y, -self.h)
            if math.fabs(math.degrees(math.atan2(mr_y, mr_x))) < setting.ROBOT_CAMERA_FOV_DEG / 2.0:
                mr_h = diff_heading_deg(m_h, self.h)
                marker_list.append((mr_x, mr_y, mr_h))
        return marker_list



""" Robot class
    A class for robot, contains same x, y, and heading information as particles
    but with some more utitilies for robot motion / collision checking
"""
class Robot(Particle):

    def __init__(self, x, y, h):
        super(Robot, self).__init__(x, y, h)

    #def __init__(self, grid):
    #    super(Robot, self).__init__(*grid.random_free_place())

    def __repr__(self):
        return "(x = %f, y = %f, heading = %f deg)" % (self.x, self.y, self.h)

    # return a random robot heading angle
    def chose_random_heading(self):
        return random.uniform(0, 360)

    def read_markers(self, grid):
        """ Helper function to simulate markers measurements by robot's camera
            Only markers in robot's camera view (in FOV) will be in the list

            Arguments:
            grid -- map grid with marker information

            Return: robot detected marker list, each marker has format:
                    measured_marker_list[i] = (rx, ry, rh)
                    rx -- marker's relative X coordinate in robot's frame
                    ry -- marker's relative Y coordinate in robot's frame
                    rh -- marker's relative heading in robot's frame, in degree
        """
        detected_markers = []
        for marker in super(Robot, self).read_markers(grid):
            # Drop detected markers randomly according to the detection failure rate
            if random.random() >= setting.DETECTION_FAILURE_RATE:
                detected_markers.append(marker)

            # Randomly add spurious markers
            if random.random() < setting.SPURIOUS_DETECTION_RATE:
                fake_marker_x = random.random() * grid.width / 2.
                fake_marker_y = math.radians(setting.ROBOT_CAMERA_FOV_DEG / 2.0) * (random.random()*2 - 1.) * fake_marker_x
                fake_marker = fake_marker_x, fake_marker_y, (random.random() - 0.5) * 360
                detected_markers.append(fake_marker)

        return detected_markers

    def move(self, odom):
        """ Move the robot with a steering angle and dist drive forward.
            Note that the robot *drive first, then turn head*.

            Arguments:
            odom -- odometry to move (dx, dy, dh) in *robot local frame*

            No return
        """

        dx, dy = rotate_point(odom[0], odom[1], self.h)
        self.x += dx
        self.y += dy
        self.h = self.h + odom[2]


    def check_collsion(self, odom, grid):
        """ Check whether moving the robot will cause collision.
            Note this function will *not* move the robot

            Arguments:
            odom -- odometry to move (dx, dy, dh) in robot local frame

            Return: True if will cause collision, False if will not be a collision
        """
        dx, dy = rotate_point(odom[0], odom[1], self.h)
        if grid.is_free(self.x+dx, self.y+dy):
            return False
        return True

