import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import copy
import os

def detect_signs_location(image, resize_shape = (500,500), HSV_lower = (40, 20, 62), HSV_upper = (80, 255, 255), 
                          dilate_iterations = 1, erode_iterations = 1, contour_threshold_area = 200, plot_image = True):    
    '''
    arguments: 
        image: if path to image not given, uses this
        resize_shape: shape to resize input images to. Set to None to not resize
        HSV_lower: lower bound of HSV colour space to be used as a mask
        HSV_upper: upper bound of HSV colour space to be used as a mask
        dilate_iterations: number of iterations to dilate
        erode_iterations: number of iterations to erode
        contour_threshold_area: threshold area used to discard contours (usually some smaller contours are noise)
        plot_image: boolean value to determine whether to plot outputs or not
    
    return values:
        Centroids: list of tuples (x,y)
        Dimensions: list of tuples (w,h)

        to find the corners of the boxes: (x + or - w/2, y + or - h/2)

    This function follows the following steps, plotting after each step
    step 1: read image, resize it, convert to HSV colour space and create a mask
    step 2: erodes and dilates the mask in that order to remove noise in mask 
    step 3: uses the mask to create contours on the image
    step 4: creates boxes around the contours that have an area greater than threshold area
    step 5: finds the centroids and dimensions of the boxes
    '''

    #step1
    if resize_shape != None:
        image = cv2.resize(image, resize_shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    mask = cv2.inRange(image_hsv, HSV_lower, HSV_upper)
    if(HSV_lower[0] > HSV_upper[0]):
        mask = cv2.inRange(image_hsv, HSV_lower, (180,HSV_upper[1],HSV_upper[2])) + cv2.inRange(image_hsv, (0,HSV_lower[1],HSV_lower[2]), HSV_upper)
    #cv2.imshow('Mask 1',cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    #cv2.waitKey(1000)


    #step 2
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=erode_iterations)
    mask = cv2.dilate(mask, kernel, iterations=dilate_iterations)
    #cv2.imshow('Mask 2',cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
    #cv2.waitKey(1000)


    #step 3
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    copy_image = copy.deepcopy(image)
    if resize_shape != None:
        copy_image = cv2.resize(copy_image, resize_shape)
    cv2.drawContours(copy_image, contours, -1, (0, 255, 0), 3)
    #cv2.imshow('Mask 3',cv2.cvtColor(copy_image, cv2.COLOR_BGR2RGB))
    #cv2.waitKey(1000)


    #step 4
    Boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) > contour_threshold_area:
            x, y, w, h = cv2.boundingRect(cnt)
            Boxes.append((x,y,w,h))
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #cv2.imshow('Mask 4',cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #cv2.waitKey(1000)

    #step 5
    Centroids = []
    Dimensions = []

    for bounding_rect in Boxes:
        x, y, w, h = bounding_rect
        Centroids.append((int(x+w/2), int(y + h/2)))
        Dimensions.append(w*h)
        
    return Centroids, Dimensions


def box_measure(image, resize_shape = None, HSV_lower_green = (30, 30, 50),HSV_upper_green = (90, 255, 255), 
                HSV_lower_red = (170, 75, 50),HSV_upper_red = (20, 255, 255),
                HSV_lower_blue= (90, 75, 30),HSV_upper_blue = (130, 255, 255),
                dilate_iterations = 1, erode_iterations = 1, contour_threshold_area = 200, plot_image = True):
    '''
    arguments: 
        image_path : path to image (if None, uses image)
        image: if path to image not given, uses this
        resize_shape: shape to resize input images to. Set to None to not resize
        HSV_lower_green: lower bound of HSV colour space to be used as a mask for green signs
        HSV_upper_green: upper bound of HSV colour space to be used as a mask for green signs
        HSV_lower_red: lower bound of HSV colour space to be used as a mask for red signs
        HSV_upper_red: upper bound of HSV colour space to be used as a mask for red signs
        HSV_lower_blue: lower bound of HSV colour space to be used as a mask for blue signs
        HSV_upper_blue: upper bound of HSV colour space to be used as a mask for blue signs
        dilate_iterations: number of iterations to dilate
        erode_iterations: number of iterations to erode
        contour_threshold_area: threshold area used to discard contours (usually some smaller contours are noise)
        plot_image: boolean value to determine whether to plot outputs or not
    
    return values:
        Centroids: list of tuples (x,y)

    This function uses detect_signs_location() to combine the centroids and dimensions for red, green and blue signs.
    '''

    if resize_shape != None:
        image = cv2.resize(image, resize_shape)

  
    Centroids_red, Dimensions_red = detect_signs_location(image, resize_shape = resize_shape, HSV_lower = HSV_lower_red, HSV_upper = HSV_upper_red,
                                                            dilate_iterations = dilate_iterations, erode_iterations = dilate_iterations,
                                                            contour_threshold_area = contour_threshold_area, plot_image = True)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    if Centroids_red != []:
#         uncomment the lines below to show the centroid and the image
#         image = cv2.circle(image, Centroids_red[np.argmax(Dimensions_red)], radius=10, color=(0, 0, 255), thickness=-1)
#         cv2.imshow('Mask',cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#         cv2.waitKey(1000)
        return Centroids_red[np.argmax(Dimensions_red)]
        
    else: 
        return Centroids_red
