"""proj2_robot_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor

import os
import cv2
import copy
import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt
from contour import box_measure
from vision_only_calculation import vision_only_depth_calculation
from controller import Robot, Camera, Lidar
import math

def test_vision_only(robot, camera_1, camera_2):

    iteration = 0
    while robot.step(timestep) != -1 and iteration < 10:

        # Get camera feed
            # https://stackoverflow.com/questions/58286019/webots-displaying-processed-numpy-image-opencv-python
        
        image_l = camera_2.getImage()
        camera_2.saveImage("img_l.png",100)
        img_l = cv2.imread("img_l.png")

        image_r = camera_1.getImage()
        camera_1.saveImage("img_r.png",100)
        img_r = cv2.imread("img_r.png")

        fov = camera_1.getFov()
        camera_translation = 0.06
        
        print('Depth, Angle calculation using vision only :', vision_only_depth_calculation(img_l, img_r, fov, camera_translation))
        
        print('------------------------')       
        #os.remove("img1.png")
        #os.remove("img2.png")

    exit()

    
if __name__ == "__main__":

    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    
    camera_1 = Camera('camera_1')
    camera_1.enable(100)
    
    camera_2 = Camera('camera_2')
    camera_2.enable(100)

    test_vision_only(robot, camera_1, camera_2)
