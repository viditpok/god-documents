from controller import Robot
from geometry import SE2, Point
import math
import numpy as np
from rrt import RRT_visualize
from map import Map
from gui import *
import time
import os
import sys

MAP_NAME = "/maps/maze6.json"
TIME_STEP = 64
MAX_SPEED = 6.28

class MoveRobot:

    def __init__(self,robot,path,TIME_STEP,MAX_SPEED):
        self.robot = robot
        self.path = path
        self.TIME_STEP = TIME_STEP
        self.MAX_SPEED = MAX_SPEED
        self.step = 1
        self.last_step_measured = 0

        # get a handler to the motors and set target position to infinity (speed control)
        self.leftMotor = robot.getDevice('left wheel motor')
        self.rightMotor = robot.getDevice('right wheel motor')

        # Set initial wheel positions
        self.leftMotor.setPosition(float('inf'))
        self.rightMotor.setPosition(float('inf'))

        # Get position sensors for wheels
        self.left_ps = self.robot.getDevice('left wheel sensor')
        self.right_ps = self.robot.getDevice('right wheel sensor')
        self.left_ps.enable(self.TIME_STEP)
        self.right_ps.enable(self.TIME_STEP)

        self.wheel_position = [self.left_ps.getValue(), self.right_ps.getValue()]
        print("wheel initial positions", self.wheel_position)

        # Get absolute position of robot
        self.gps = self.robot.getDevice('gps')
        self.gps.enable(self.TIME_STEP)

        # Get coordinate system of robot
        self.compass = self.robot.getDevice('compass')
        self.compass.enable(self.TIME_STEP)

        # Initialize motors
        self.leftMotor.setVelocity(0.0)
        self.rightMotor.setVelocity(0.0)
        self.robot.step(TIME_STEP)

    def get_robot_pose(self):
        """
        Transform from translation (x, y, z) and north direction into SE2(x, y, h)
        """
        x, y, _ = self.gps.getValues()
        compass_value = self.compass.getValues()
        h = math.atan2(compass_value[0], compass_value[1])
        return SE2(x, y, h)
        
    def get_motion_info(self):

        dt = (self.step - self.last_step_measured) * TIME_STEP / 1000 #in second
        new_wheel_position = [self.left_ps.getValue(), self.right_ps.getValue()]
        wheel_dis_traveled = [new_wheel_position[0] - self.wheel_position[0], new_wheel_position[1] - self.wheel_position[1]]
        self.wheel_position = new_wheel_position
        omega_l, omega_r = wheel_dis_traveled[0] / dt, wheel_dis_traveled[1] / dt
        self.last_step_measured = self.step
        return omega_l, omega_r, dt

    def move_forward(self,next_coords,speed = 1,min_distance = 0.1):

        print("move forward...")
        while robot.step(TIME_STEP) != -1:

            # Set wheel velocity
            self.rightMotor.setVelocity(speed*self.MAX_SPEED)
            self.leftMotor.setVelocity(speed*self.MAX_SPEED)

            # Stop moving when the robot is within min_distance of the next_coords
            current_dist = math.sqrt((self.get_robot_pose().x-next_coords[0])**2 + (self.get_robot_pose().y-next_coords[1])**2)
            if current_dist <= min_distance:
                # Stop moving by setting robot wheel velocities to 0
                self.leftMotor.setVelocity(0.0)
                self.rightMotor.setVelocity(0.0)
                break

        time.sleep(0.1)

    def turn_in_place(self,expected_heading,min_angle):
        speed = 0.1

        while self.robot.step(TIME_STEP) != -1:
            # Turn clockwise
            if expected_heading < self.get_robot_pose().h:
                self.rightMotor.setVelocity(speed * MAX_SPEED * -1)
                self.leftMotor.setVelocity(speed * MAX_SPEED * 1)
            # Turn counterclockwise
            else:
                self.rightMotor.setVelocity(speed * MAX_SPEED * 1)
                self.leftMotor.setVelocity(speed * MAX_SPEED * -1)

            # Stop turning if robot is at expected heading
            heading_diff = abs(self.get_robot_pose().h - expected_heading)
            if heading_diff < min_angle:
                # Stop turning by setting robot wheel velocities to 0
                self.leftMotor.setVelocity(0.0)
                self.rightMotor.setVelocity(0.0)
                break
            
        time.sleep(0.5)

    def follow_path(self):
        for node in self.path:
            min_distance = 0.1
            min_angle = 0.02 #radians

            # Turn to angle of next coord
            print(node,' turn in place')
            expected_heading = math.atan2(node[1]-self.get_robot_pose().y,node[0]-self.get_robot_pose().x)
            self.turn_in_place(expected_heading,min_angle=min_angle)
            #  Move towards point
            self.move_forward(next_coords=node,speed=1,min_distance=min_distance)
            
        print('At the goal!')

        return

if __name__ == "__main__":
    
    # Create the Robot instance.
    robot = Robot()

    # Call RRT to get path
    map_name = os.path.dirname(__file__) + MAP_NAME #change MAP_NAME to explore a different map
    print(map_name)
    map = Map(map_name, exploration_mode=False)

    path = RRT_visualize(map)

    print(path,'start')
    
    # Move robot
    MoveRobot(robot,path[1:],TIME_STEP,MAX_SPEED).follow_path()
