import numpy as np
import time
import math
import threading

SPEED_UP_FACTOR = 16 # factor of play-back speed in visualization

class DDRobot():
    """
    Differential drive robot used in simulation
    """
    ROBOT_WIDTH = 25
    ROBOT_LENGTH = 40
    HEAD_WIDTH = 5
    ROBOT_CORNERS = np.array(
        [
            [ROBOT_LENGTH / 2 - HEAD_WIDTH, ROBOT_WIDTH / 2, 1],
            [ROBOT_LENGTH / 2 - HEAD_WIDTH, -ROBOT_WIDTH / 2, 1],
            [-ROBOT_LENGTH / 2, -ROBOT_WIDTH / 2, 1],
            [-ROBOT_LENGTH / 2, ROBOT_WIDTH / 2, 1]
        ]
    ).T
    ROBOT_HEAD_CORNERS = np.array(
        [
            [ROBOT_LENGTH / 2, ROBOT_WIDTH / 2, 1],
            [ROBOT_LENGTH / 2, -ROBOT_WIDTH / 2, 1],
            [ROBOT_LENGTH / 2 - HEAD_WIDTH, -ROBOT_WIDTH / 2, 1],
            [ROBOT_LENGTH / 2 - HEAD_WIDTH, ROBOT_WIDTH / 2, 1]
        ]
    ).T
    VISION_DISTANCE = 30.0

    def __init__(self, x, y, map=None):
        self.__x = x
        self.__y = y
        self.__theta = 0
        self.map = map
        self.trajectory = [[self.__x, self.__y]]
        self.updated = threading.Event()

    @property
    def x(self):
        return self.__x
    
    @property
    def y(self):
        return self.__y
    
    @property
    def theta(self):
        return self.__theta
    
    @staticmethod
    def get_transform_matrix(x, y, theta):
        return np.array(
            [
                [np.cos(theta), -np.sin(theta), x],
                [np.sin(theta), np.cos(theta), y],
                [0, 0, 1]
            ]
        )
    
    def turn_in_place(self, angle_rad):
        """
        Rotate the robot in place counterclockwise for a certain angle

        Args:
            angle_rad (float): angle to rotate in radians
        """
        TIME_TO_ROTATE = 0.5 / SPEED_UP_FACTOR
        ANGLE_PER_FRAME = angle_rad / math.ceil(TIME_TO_ROTATE * 60)
        for i in range(math.ceil(TIME_TO_ROTATE * 60)):
            # 60 fps
            self.__theta += ANGLE_PER_FRAME
            self.updated.set()
            time.sleep(1/60)
    
    def move_forward(self, distance_mm, mm_per_sec=10):
        """
        Command the robot to move forward for a certain distance

        Args:
            distance_mm (float): distance to move forward in mm, 0 <= distance_mm <= 30 otherwise clipped
            mm_per_sec (float, optional): speed of the robot in mm/s. Defaults to 10.
        """
        distance_mm = np.clip(distance_mm, 0, 30)
        TIME_TO_MOVE = distance_mm / mm_per_sec / SPEED_UP_FACTOR
        DISTANCE_PER_FRAME = distance_mm / math.ceil(TIME_TO_MOVE * 60)
        for i in range(math.ceil(TIME_TO_MOVE * 60)):
            self.__x += DISTANCE_PER_FRAME * np.cos(self.__theta)
            self.__y += DISTANCE_PER_FRAME * np.sin(self.__theta)
            self.updated.set()
            self.check_valid()
            time.sleep(1/60)
        self.trajectory.append([self.x, self.y])

    def check_valid(self):
        """
        Check if the current robot pose is in collision free and within map bounds
        """
        if self.map is not None and self.map.is_inside_obstacles(self, True):
            print(f"Robot is in obstacle, invalid position: {self.x}, {self.y}")
            raise Exception("Robot is in obstacle")

        if self.map is not None and not self.map.is_inbound(self):
            print(f"Robot is out of map bounds, invalid position: {self.x}, {self.y}")
            raise Exception("Robot is out of map bounds")

    def __add_turn_noise(self):
        pass

    def __add_move_noise(self):
        pass

    def get_corners(self):
        """
        Get the four corners of the robot body for visualization
        """
        corners = DDRobot.get_transform_matrix(self.__x, self.__y, self.__theta) @ DDRobot.ROBOT_CORNERS
        return [(corners[0, i], corners[1, i]) for i in range(4)]
    
    def get_head_corners(self):
        """
        Get the four corners of the robot head for visualization
        """
        corners = DDRobot.get_transform_matrix(self.__x, self.__y, self.__theta) @ DDRobot.ROBOT_HEAD_CORNERS
        return [(corners[0, i], corners[1, i]) for i in range(4)]




if __name__ == '__main__':
    """
    Simple test drive of the robot.
    """
    r = DDRobot(0, 0)
    r.turn_in_place(np.pi/2)
    r.move_forward(200, 50)
    print(r.x, r.y, r.theta)
