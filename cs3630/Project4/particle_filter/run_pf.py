
import sys
import os
import threading
import json
from datetime import datetime
from particle_filter import ParticleFilter
from environment import Environment
from gui import GUIWindow
from utils import *
from setting import *
from sensors import compute_measurements

SCENARIO_NAME = "simple_world1"  #simple_world1 or maze_world1

IMAGEFOLDER = os.path.join(IMAGE_PATH, SCENARIO_NAME)
LIDARPATH = os.path.join(LIDAR_PATH, f"lidar_{SCENARIO_NAME}.csv")
POSEPATH = os.path.join(POSE_PATH, f"pose_{SCENARIO_NAME}.csv")
ODOMETRYPATH = os.path.join(ODOMETRY_PATH, f"odometry_{SCENARIO_NAME}.csv")
CONFIGPATH = os.path.join(CONFIG_PATH, f"config_{SCENARIO_NAME}.json")
poses = read_poses(POSEPATH)
lidar_arrays = read_lidar(LIDARPATH)
odometry_steps = read_odometry(ODOMETRYPATH)

gui = GUIWindow(CONFIGPATH)
correct_est_count = 0

def run_scenario():
    global correct_est_count
    env = Environment(CONFIGPATH)
    particle_filter = ParticleFilter(env)
    
    start_step = 10
    end_step = len(poses)
    step_skip = 5
    for i in range(start_step, end_step, step_skip):
        print (i, "/", end_step - end_step % step_skip)
        robot_pose = poses[i]

        # Compute odometry from wheel speeds.
        odometry = integrate_odo(env, i-step_skip, i, odometry_steps)

        # Compute marker measurements from sensor data.
        img_l, img_r = read_images(IMAGEFOLDER, i)
        lidar_range_array = lidar_arrays[i]
        marker_measures = compute_measurements(img_l, img_r, lidar_range_array)

        # Particle filter update.
        particle_filter.update(odometry, marker_measures)
        est_pose = particle_filter.compute_best_estimate()

        # Check if estimate is correct within threshold.
        confident = check_confident(est_pose, robot_pose)
        if confident:
            correct_est_count += 1

        # Visualization.
        gui.show_particles(particle_filter.particles)
        gui.show_robot(robot_pose, marker_measures)
        gui.show_mean(est_pose, confident)
        gui.show_lidar_array(robot_pose, lidar_range_array)
        gui.updated.set()

class MainThread(threading.Thread):

    def __init__(self):
        threading.Thread.__init__(self, daemon=True)

    def run(self):

        start_time = datetime.now()

        run_scenario()

        
        with open(CONFIGPATH, "r") as file:
            configs = json.load(file)
            baseline = configs["num_correct_est_baseline"]
        print('number of correct estimats:', correct_est_count)
        final_score = min(100, round(correct_est_count / baseline, 2) * 100)
        print('score: ', final_score)

        print(f"elapsed time in seconds: {(datetime.now() - start_time).total_seconds()}")
        print("trajectory finished")
        sys.exit()


if __name__ == '__main__':
    main_thread = MainThread()
    main_thread.start()
    gui.start()
    main_thread.join()
    gui.join()
    