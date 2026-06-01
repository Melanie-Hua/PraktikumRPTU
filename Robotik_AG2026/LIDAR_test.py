# SPDX-FileCopyrightText: 2019 Dave Astels for Adafruit Industries
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
from adafruit_rplidar import RPLidar
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import cv2 as cv

height = 1000
width = 1000 
blank_image = np.zeros((height,width,3), np.uint8)
# Set up the display
cv.imshow("LIDAR",blank_image)
cv.waitKey(1)

# Setup the RPLidar
PORT_NAME = '/dev/ttyUSB0'



lidar = RPLidar(None, PORT_NAME, timeout=3)
# used to scale data to fit on the screen
max_distance = 0
#pylint: disable=redefined-outer-name,global-statement
scan_data = [0]*360
pixels = np.zeros((360,2),dtype=np.int16)
tolerance = 50
try:
    print(lidar.info)
    for scan in lidar.iter_scans():
        for (_, angle, distance) in scan:
            index=min([359, floor(angle)])
            scan_data[index] = distance
            radians = angle*pi/180
            y = -sin(radians)*distance/20+500 # this is because the angles are looking up
            x = cos(radians)*distance/20+500
            x=int(x)
            y=int(y)
            #blank_image[x-2:x+2, y-2:y+2] = (255, 255, 255)
            pixels[index] = [x,y]
        blank_image = np.zeros((height,width,3), np.uint8)
        for pixel in pixels:    
            blank_image[pixel[0]-2:pixel[0]+2,pixel[1]-2:pixel[1]+2] = (255,255,255)
        cv.imshow("LIDAR",blank_image)
        cv.waitKey(1)

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

exit()
plt.plot(scan_data)
plt.ylabel('Distance (mm)')
plt.xlabel('Angle (degrees)')
plt.savefig(f"./angle-to-degree-{tolerance}.png")
plt.close()
plt.scatter(pixels[:,0],pixels[:,1])
plt.ylabel('y')
plt.xlabel('x')
plt.savefig(f"picture-{tolerance}.png")
plt.close()

fig, axs = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={'projection': 'polar'},
                        layout='constrained')
ax = axs
ax.plot(np.linspace(0,2*pi,360), scan_data)
ax.set_rlabel_position(-22.5)  # Move radial labels away from plotted line
ax.grid(True)

ax.set_title("A line plot on a polar axis", va='bottom')

plt.savefig(f"./polarplot-{tolerance}.png")