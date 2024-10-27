import sys
import os
# Add the parent directory to sys.path
project_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
pf_path = os.path.join(project_path, "particle_filter")
sys.path.append(os.path.abspath(pf_path))
import cv2
from controller import Robot
import csv
import math
import time
import threading
import json
from datetime import datetime
from environment import Environment
from particle_filter import ParticleFilter
from setting import *
from utils import calculate_pose, check_confident
from gui import GUIWindow
from geometry import SE2
from sensors import compute_measurements


correct_est_count = 0

robot = Robot()

camera_r = robot.getDevice('camera_1')
camera_r.enable(100)

camera_l = robot.getDevice('camera_2')
camera_l.enable(100)

lidar = robot.getDevice('lidar')
lidar.enable(TIME_STEP)

# get a handler to the motors and set target position to infinity (speed control)
leftMotor = robot.getDevice('left wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor = robot.getDevice('right wheel motor')
rightMotor.setPosition(float('inf'))



left_ps = robot.getDevice('left wheel sensor')
left_ps.enable(TIME_STEP)
right_ps = robot.getDevice('right wheel sensor')
right_ps.enable(TIME_STEP)



gps = robot.getDevice('gps')
gps.enable(TIME_STEP)
compass = robot.getDevice('compass')
compass.enable(TIME_STEP)


world_name = '_'.join(os.path.basename(__file__).split("_")[1:3])
IMAGEFOLDER = os.path.join(IMAGE_PATH, world_name)
LIDARPATH = os.path.join(LIDAR_PATH, f"lidar_{world_name}.csv")
POSEPATH = os.path.join(POSE_PATH, f"pose_{world_name}.csv")
ODOMETRYPATH = os.path.join(ODOMETRY_PATH, f"odometry_{world_name}.csv")
CONFIGPATH = os.path.join(CONFIG_PATH, f"config_{world_name}.json")
env = Environment(CONFIGPATH)

particle_filter = ParticleFilter(env)

gui = GUIWindow(CONFIGPATH)

step = 1
last_step_measured = 0




def get_robot_pose():
    return calculate_pose(gps.getValues(), compass.getValues())


#initialize motors
leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)
robot.step(TIME_STEP)

wheel_position = [left_ps.getValue(), right_ps.getValue()]
print("wheel initial positions", wheel_position)

def get_motion_info():
    global last_step_measured, wheel_position
    dt = (step - last_step_measured) * TIME_STEP / 1000 #in second
    new_wheel_position = [left_ps.getValue(), right_ps.getValue()]
    wheel_dis_traveled = [new_wheel_position[0] - wheel_position[0], new_wheel_position[1] - wheel_position[1]]
    wheel_position = new_wheel_position
    omega_l, omega_r = wheel_dis_traveled[0] / dt, wheel_dis_traveled[1] / dt
    last_step_measured = step
    return omega_l, omega_r, dt



def initialize_output_data():
    if not DATA_CAPTURE_MODE:
        return
    if not os.path.exists(IMAGEFOLDER):
        os.makedirs(IMAGEFOLDER)

    if not os.path.exists(LIDAR_PATH):
        os.makedirs(LIDAR_PATH)

    if not os.path.exists(POSE_PATH):
        os.makedirs(POSE_PATH)

    if not os.path.exists(ODOMETRY_PATH):
        os.makedirs(ODOMETRY_PATH)

    with open(ODOMETRYPATH, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['step', 'omega_l', 'omega_r', 'dt'])

    with open(POSEPATH, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['step', 'x', 'y', 'h'])

    with open(LIDARPATH, 'w', newline='') as file:
            writer = csv.writer(file)
            columns = ['step'] + [degree for degree in range(1, 361)]
            writer.writerow(columns)

def output_data():
    global step
    if not DATA_CAPTURE_MODE:
        return

    
    camera_r.saveImage(os.path.join(IMAGEFOLDER, f"{step}_camera_r.jpg"), IMAGE_QUALITY)
    camera_l.saveImage(os.path.join(IMAGEFOLDER, f"{step}_camera_l.jpg"), IMAGE_QUALITY)


    with open(ODOMETRYPATH, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([step, *get_motion_info()])

    with open(POSEPATH, 'a', newline='') as file:
        writer = csv.writer(file)
        current_pose = get_robot_pose()
        writer.writerow([step, current_pose.x, current_pose.y, current_pose.h])

    with open(LIDARPATH, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(lidar.getRangeImage())

    step += 1

def pf_step():
    global step, correct_est_count
    if DATA_CAPTURE_MODE:
        output_data()
        return
    if step % 5 != 0 or step < 10:
        step += 1
        return
    
    # capture images and lidar array
    print ("============================", step, "============================")
    camera_l.saveImage("img_l.png",IMAGE_QUALITY)
    img_l = cv2.imread("img_l.png")
    camera_r.saveImage("img_r.png",IMAGE_QUALITY)
    img_r = cv2.imread("img_r.png")
    lidar_range_array = lidar.getRangeImage()

    # get robot pose
    robot_pose = get_robot_pose()
    print("robot pose ", robot_pose)
    
    # compute odometry from wheel speeds
    odometry = env.diff_drive_odometry(*get_motion_info())

    # compute marker measurements from sensor data
    marker_measures = compute_measurements(img_l, img_r, lidar_range_array)

    # update pf
    particle_filter.update(odometry, marker_measures)
    est_pose = particle_filter.compute_best_estimate()
    
    confident = check_confident(est_pose, robot_pose)
    if confident:
        correct_est_count += 1

    # visualization
    gui.show_particles(particle_filter.particles)
    gui.show_robot(robot_pose, marker_measures)
    gui.show_mean(est_pose, confident)
    gui.show_lidar_array(robot_pose, lidar_range_array)
    gui.updated.set()
    
    step += 1

def move_forward(dis_to_wall = 0.27, distance = float('inf'), speed = 1):
    global time
    print("move forward...")
    x1, y1 = get_robot_pose().x, get_robot_pose().y
    while robot.step(TIME_STEP) != -1:
        rightMotor.setVelocity(speed*MAX_SPEED)
        leftMotor.setVelocity(speed*MAX_SPEED)
        
        pf_step()
        if lidar.getRangeImage()[0] < dis_to_wall:
            break
        
        x2, y2 = get_robot_pose().x, get_robot_pose().y
        if math.sqrt((x2 - x1)**2 + (y2 - y1)**2) >= distance:
            break

        pass
    time.sleep(0.1)

def turn_90_degrees(clockwise = True):
    speed = 0.1
    heading = get_robot_pose().h
    while robot.step(TIME_STEP) != -1:
        rightMotor.setVelocity(speed * MAX_SPEED * -1 if clockwise else 1)
        leftMotor.setVelocity(speed * MAX_SPEED * 1 if clockwise else -1)
        pf_step()
        heading_diff = abs(get_robot_pose().h - heading)
        heading_diff = min(heading_diff, 2 * math.pi - heading_diff)
        if heading_diff >= 1.5:
            break
        pass
    time.sleep(0.5)

def turn_left():
    print("turn left...")
    turn_90_degrees(False)

def turn_right():
    print("turn right...")
    turn_90_degrees(True)

class MainThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self, daemon=True)

    def run(self):
        import traceback
        try:
            global correct_est_count

            if DATA_CAPTURE_MODE:
                print("running in data capture mode, particle filter will not run")

            start_time = datetime.now()
            initialize_output_data()

            # turn in place 720 degrees
            for _ in range(8):
                turn_right()
            
            if not DATA_CAPTURE_MODE:
                with open(CONFIGPATH, "r") as file:
                    configs = json.load(file)
                    baseline = configs["num_correct_est_baseline"]
                print('number of correct estimats:', correct_est_count)
                final_score = min(100, round(correct_est_count / baseline, 2) * 100)
                print('score: ', final_score)

            print(f"elapsed time in seconds: {(datetime.now() - start_time).total_seconds()}")
            print("trajectory finished")

        except Exception as e:
            traceback.print_exc()

        robot.step(TIME_STEP)


if __name__ == "__main__":
    
    
    main_thread = MainThread()
    main_thread.start()
    gui.start()
    main_thread.join()
    gui.join()
    
    