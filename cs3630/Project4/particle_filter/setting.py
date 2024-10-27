# Set randome seed to None for random testing, to 0 for deterministic testing
RANDOM_SEED = 2024
# RANDOM_SEED = 0

PARTICLE_COUNT = 4000       # Total number of particles in your filter

# odometry Gaussian noise model
ODOM_TRANS_SIGMA = 0.05     # translational err in inch (grid unit)
ODOM_HEAD_SIGMA = 0.01         # rotational err in rad
MOTION_TRANS_SIGMA = 0.03
MOTION_HEAD_SIGMA = 0.03


# marker measurement Gaussian noise model
CAMERA_DEPTH_SIGMA = 1e5          # set to a verge large number as sometimes the depth can be incorrect 
CAMERA_HEADING_SIGMA = 0.1        # in radians
LIDAR_RANGE_SIGMA = 0.05

PARTICLE_MAX_SHOW = 200     # Max number of particles to be shown in GUI (for speed up)

ROBOT_CAMERA_BASELINE = 0.06
ROBOT_CAMERA_FOV = 0.84   # Robot camera FOV in degree
ROBOT_RADIUS = 0.1

TIME_STEP = 64
MAX_SPEED = 6.28
IMAGE_QUALITY = 100

###################################
## Non ideal robot detection model
###################################
## Feel free to modify the values for debugging
NOMINAL_DETECTION_FAILURE_RATE = 1e-2
EDGE_DETECTION_FAILURE_RATE = 1e-1
NOMINAL_SPURIOUS_DETECTION_RATE = 1e-2

# translational error allow
Err_trans = 0.1
# orientation erro allow in degree
Err_rot = 10

# set as True if in data capture mode
DATA_CAPTURE_MODE = True


import os
PROJECT_PATH = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(PROJECT_PATH, "data")
IMAGE_PATH = os.path.join(DATA_PATH, "images")
LIDAR_PATH = os.path.join(DATA_PATH, "lidar")
POSE_PATH = os.path.join(DATA_PATH, "pose")
ODOMETRY_PATH = os.path.join(DATA_PATH, "odometry")
CONFIG_PATH = os.path.join(PROJECT_PATH, "config")
RESULT_PATH = os.path.join(PROJECT_PATH, "results")
WORLD_PATH = os.path.join(PROJECT_PATH, "worlds")

