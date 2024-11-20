from controller import Robot, Supervisor
from geometry import SE2, Point
import math
import numpy as np
from robot_gui import RobotEnv, RobotEnvThread
from gui import *
import time
import os
import sys

from grid import Grid
from gui import GUIWindow
from robot import Robot_Sim
from exploration import exploration_state_machine
from utils import *
from generate_noise import add_noise, add_offset_noise

TIME_STEP = 64
MAX_SPEED = 6.28

class MoveRobot:

    def __init__(self,robot,path,TIME_STEP,MAX_SPEED):
        self.robot = robot
        self.path = path
        self.TIME_STEP = TIME_STEP
        self.MAX_SPEED = MAX_SPEED
        self.step = 1
        self.step_vis = 1
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

    def move_forward(self,next_coords,speed = 5,min_distance = 0.1):

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

        time.sleep(0.0001)

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
            
        time.sleep(0.00001)

    def move_to_next_coord(self, node):
        
        min_distance = 0.1
        min_angle = 0.02 #radians
        expected_heading = math.atan2(node[1]-self.get_robot_pose().y,node[0]-self.get_robot_pose().x)
        self.turn_in_place(expected_heading,min_angle=min_angle)
        self.move_forward(next_coords=node,speed=1,min_distance=min_distance)
        return
    
    def follow_path(self):
        for node in self.path:
            min_distance = 0.1
            min_angle = 0.02 #radians
            # Turn to angle of next coord
            print(node,' turn in place')
            expected_heading = math.atan2(node[1]-self.get_robot_pose().y,node[0]-self.get_robot_pose().x)
            print("Expected heading webots: ", expected_heading)
            self.turn_in_place(expected_heading,min_angle=min_angle)
            #  Move towards point
            self.move_forward(next_coords=node,speed=1,min_distance=min_distance)
            
        print('At the goal!')

        return

class RobotEnvController:

    def __init__(self, robbie, grid, webot_robot, program_state='exploration', testing=False):
        self.robbie = robbie
        self.grid = grid
        self.webot_robot = webot_robot
        self.program_state = program_state
        self.testing = testing
        self.robbie.markers_found_or_picked = robbie.read_marker_around(grid)
        self.used_rrt=True
        self.prev_cx=100
        self.prev_cy=100
        self.counter = 0


    def update(self):

        if self.program_state == 'exploration':
            self.robbie = exploration_state_machine(robbie=self.robbie, grid=self.grid)
        
        self.update_motion() # update where the robot is on the grid and what it can see
    
    def update_motion(self):

        self.robbie.dt = self.robbie.TIMESTEP
        self.robbie.move_diff_drive(self.grid, self.robbie.vl, self.robbie.vr, self.robbie.dt)
        
        # convert robbie.next_coord to continuous
        c_x, c_y = self.grid.discrete_to_cont(self.robbie.x, self.robbie.y)
        
        self.counter+=1
        if self.counter%10==0:
            webot_robot.move_to_next_coord([c_x, c_y])
        self.prev_cx,self.prev_cy=c_x,c_y

        if self.program_state == 'exploration':
            # read markers around
            marker_list = self.robbie.read_marker_around(self.grid)

            #update markers found        
            if len(self.robbie.markers_found_or_picked) != len(set(self.robbie.markers_found_or_picked).union(set(marker_list))):
                self.robbie.markers_found_or_picked = list(set(self.robbie.markers_found_or_picked).union(set(marker_list)))
                fname = self.grid.fname
                print(f'{fname} found {len(self.robbie.markers_found_or_picked)}/{self.grid.LANDMARKS_TOTAL} markers')
        
        if self.program_state == 'tasks': 
            # read markers around
            self.robbie.pickup_marker(self.grid)                


# thread to run robot environment when GUI is on
class RobotEnvThreadController(threading.Thread):

    def __init__(self, robot_env, gui=None, testing=False, time_limit=None):
        threading.Thread.__init__(self, daemon=True)
        self.robot_env = robot_env
        self.gui = gui
        self.testing = testing
        self.time_limit = time_limit if time_limit else math.inf
        self.robbie = robot_env.robbie
        self.grid = self.robot_env.grid
        self.map_name = self.robot_env.grid.fname

    def run(self):
        print(f'Running on map: {self.map_name}')
        start_time = time.time()
        if self.testing:
            while True:
                try:
                    self.robot_env.update()
                    elapsed_time = time.time() - start_time
                except Exception as e:
                    print(f'Exception in {self.map_name}: {e}! Thread terminated.')
                    break
                if elapsed_time > self.time_limit:
                    print(f'Run time exceeds {self.time_limit:.2f} seconds.')
                    break
                if len(self.robbie.markers_found_or_picked) == self.grid.LANDMARKS_TOTAL:
                    print(f'{self.map_name} done in {elapsed_time:.2f} seconds. Good job!')
                    break
        else:
            while True:
                self.robot_env.update()
                self.gui.show_robot(self.robot_env.robbie)
                self.gui.updated.set()

                time.sleep(robot_pause_time)
                if len(self.robbie.markers_found_or_picked) == self.grid.LANDMARKS_TOTAL:
                    elapsed_time = time.time() - start_time
                    print(f'{self.map_name} done in {elapsed_time:.2f} seconds. Good job!')
                    time.sleep(3)
                    stopevent.set()
                    break
        
        return self.robbie.markers_found_or_picked, self.grid.LANDMARKS_TOTAL

if __name__ == "__main__":
    
    # # Create the Robot instance.
    robot = Robot()

    global stopevent
    
    folder_path = os.path.dirname(os.getcwd())
    
    # Set the map path you want to test
    maze_name = "maze1"
    map_filename = os.path.join(folder_path, "exploration_controller","maps",maze_name+".json")
    
    #  Set state 'exploration'
    state = 'exploration'

    if len(sys.argv) > 1:
        map_filename = sys.argv[1]

    robot_pause_time = 0.0
	
    grid = Grid(map_filename)

    # initial robot transformation (X, Y, yaw in deg)
    robot_init_pose = grid.start
    robot_init_pose.append(0) # needs to be 3d
    
    robbie = Robot_Sim(robot_init_pose[0], robot_init_pose[1], robot_init_pose[2]) # Running from webots
    webot_robot = MoveRobot(robot, None, TIME_STEP,MAX_SPEED)
    
    robot_env = RobotEnvController(robbie, grid, webot_robot)
    robot_env.program_state = state
    
    stopevent = threading.Event()
    gui = GUIWindow(grid,robot_env.program_state, stopevent)
    robot_thread = RobotEnvThreadController(robot_env, gui, testing=False)
    robot_thread.start()
    gui.start()