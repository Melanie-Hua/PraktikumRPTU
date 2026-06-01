# ttest SPDX-FileCopyrightText: 2019 Dave Astels for Adafruit Industries
#
# SPDX-License-Identifier: MIT
"""
Consume LIDAR measurement file and create an image for display.
Adafruit invests time and resources providing this open source code.
Please support Adafruit and open source hardware by purchasing
products from Adafruit!
Written by Dave Astels for Adafruit Industries
Copyright (c) 2019 Adafruit Industries
Licensed under the MIT license.
All text above must be included in any redistribution.
"""
import os
from math import cos, sin, pi, floor
from sys import exception
from rplidar import RPLidar
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import cv2 as cv

height = 1000 # for the display window, not the lidar data
width = 1000 
blank_image = np.zeros((height,width,3), np.uint8) # creates a black image with 3 channes RGB with the data type 8bit integers 
# Set up the display
cv.imshow("LIDAR",blank_image)
cv.waitKey(1) #waits 1 millisecond for a key event

# Setup the RPLidar
PORT_NAME = '/dev/ttyUSB0'


lidar = RPLidar(PORT_NAME, baudrate=115200, timeout=3) # creates LIDAR object with No motor PWM control pin, port name and serial timeout as waiting time for data
# used to scale data to fit on the screen
max_distance = 0
#pylint: disable=redefined-outer-name,global-statement
scan_data = [0]*360 #creates a list of 360 zeros to store the distance data for each degree of the scan, initialized to zero
pixels = np.zeros((360,2),dtype=np.int16) #creats a 360x2 table to store the x and y values
tolerance = 50
try:

    for scan in lidar.iter_scans(): # starts continuous scan loop, iter_scans() is a generator that yields scan data as it is received from the LIDAR, each scan is a list of tuples (quality, angle, distance)
        for (_, angle, distance) in scan: # each scan is a touple of (quality, angle, distance). _ because we dont need quality
            index=min([359, floor(angle)]) # rounds every angle downward so theres no angle greater than 359. floor --> an rounds down to the nearest integer, min --> returns the smaller of the two values, so if floor(angle) is greater than 359, it will return 359 instead
            scan_data[index] = distance #in the 360 long touple, store the measured distance as index for the corresponding angle 
            radians = angle*pi/180 # converts angle from degrees to radians, since the math functions in python use radians
            y = -sin(radians)*distance/20+500 # this is because the angles are looking up
            x = cos(radians)*distance/20+500 # given angle and distance from origin point calculate x and y. /20 to scale to fit in the image and +500 to put origin in the center of the image
            x=int(x) #make it a interger
            y=int(y)
            #blank_image[x-2:x+2, y-2:y+2] = (255, 255, 255)
            pixels[index] = [x,y] # stores x and y as touple in the pixels array at the index corresponding to the angle, so we can later use it to plot the points on the image

            #print(distance)
        #print(scan_data)
        #zeros = np.where(np.array(scan_data)<0.001,1,0)
        #num_zeros = np.sum(zeros)
        #if num_zeros<tolerance:
        #    break
        #print(num_zeros) 
except KeyboardInterrupt:
    print('Stopping.')

lidar.stop()
lidar.disconnect()


