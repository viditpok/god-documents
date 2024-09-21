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
from vision_lidar_calculation import vision_lidar_distance_calculation
from controller import Robot, Camera, Lidar
import math

def test_vision_lidar(robot, camera_1, lidar):

    iteration = 0

    while robot.step(timestep) != -1 and iteration < 10:

        # Get camera feed
            # https://stackoverflow.com/questions/58286019/webots-displaying-processed-numpy-image-opencv-python

        image = camera_1.getImage()
        camera_1.saveImage("img.png",100)
        img = cv2.imread("img.png")

        fov = camera_1.getFov()
        camera_translation = 0.06
        
        # Get lidar feed
        lidar = robot.getDevice('lidar')
        lidar.enable(timestep)
        lidar_range_array = lidar.getRangeImage()   

        with open('lidar.txt', 'w') as file:
            for value in lidar_range_array:
                file.write(f'{value:.2f}\n')
        
        print('Distance, Angle calculation using vision + lidar :', vision_lidar_distance_calculation(img, lidar_range_array, fov))
           
        print('----------------------')
        #os.remove("img.png")
        iteration += 1

    exit()

if __name__ == "__main__":

    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
   
    camera_1 = Camera('camera_1')
    camera_1.enable(100)
    
    lidar = Lidar('lidar')
    lidar.enablePointCloud()
    
    test_vision_lidar(robot, camera_1, lidar)
