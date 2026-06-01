

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

from math import floor
from rplidar import RPLidar, RPLidarException
import numpy as np
import threading
import time

# Setup the RPLidar
# used to scale data to fit on the screen
max_distance = 0
#pylint: disable=redefined-outer-name,global-statement
scan_data = [0]*360 #in mm
pixels = np.zeros((360,2),dtype=np.int16)

# interrupt check for any obstacles 26.05.2026
roboterwidth = 60
horizontal_buffer = 15
vertical_buffer = 20
measure_samples = 3
measure_interval = 0
following_distance = 200
min_distance = following_distance - vertical_buffer
driving_direction = 0 # get from CV group

class RingBuffer:
    """ Class that implements a not-yet-full buffer. """
    def __init__(self, bufsize):
        self.bufsize = bufsize
        self.data = []

    class __Full:
        """ Class that implements a full buffer. """
        def add(self, x):
            """ Add an element overwriting the oldest one. """
            self.data[self.currpos] = x
            self.currpos = (self.currpos+1) % self.bufsize
        def get(self):
            """ Return list of elements in correct order. """
            return self.data[self.currpos:]+self.data[:self.currpos]

    def add(self,x):
        """ Add an element at the end of the buffer"""
        self.data.append(x)
        if len(self.data) == self.bufsize:
            # Initializing current position attribute
            self.currpos = 0
            # Permanently change self's class from not-yet-full to full
            self.__class__ = self.__Full

    def get(self):
        """ Return a list of elements from the oldest to the newest. """
        return self.data
    
scan_data_buffer = RingBuffer(3)
class LIDAR():
    def __init__(self,dimensions_robot,not_aus_time=10,PORT_NAME = '/dev/ttyUSB0'):
        self.dimensions_robot = dimensions_robot # in mm (width, length)
        self.lidar = None
        self.lidar = RPLidar(port=PORT_NAME, baudrate=115200, timeout=1)
        self.measure = threading.Event()
        self.calculate = threading.Event()
        self.detect_obstacle = threading.Event()
        self.measuring_thread = threading.Thread(target=self.measuring_loop)
        self.logic_thread = threading.Thread(target=self.simple_logic_loop)
        self.speed = 10 #mm/s
        self.direction = 0 #degrees
        self.not_aus_time = not_aus_time
        self.stop_window =[]
        self.scan_error_count = 0
        self.last_scan_error_time = 0.0
        self.target_angle = 0
        

    def start_measuring(self):
        self.measure.set()
        self.lidar.start_motor()
        self.measuring_thread.start()

    def stop_measuring(self):
        self.measure.clear()
        if self.measuring_thread.is_alive():
            self.measuring_thread.join()
        if self.lidar is not None:
            self.lidar.stop()
            self.lidar.stop_motor()

    def start_logic(self):
        self.detect_obstacle.clear()
        self.calculate.set()
        self.logic_thread.start()

    def stop_logic(self):
        self.calculate.clear()
        if self.logic_thread.is_alive():
            self.logic_thread.join()

    def measuring_loop(self):
        global max_distance
        global scan_data
        while self.measure.is_set():
            try:
                for scan in self.lidar.iter_scans(max_buf_meas=5000, min_len=10):
                    if not self.measure.is_set():
                        print("Stopping measuring loop")
                        break
                    for (quality, angle, distance) in scan:
                        #if angle <= 5:
                            #print("Quality {}; Angle: {}; Distance: {}".format(quality, angle, distance))
                        index = min(359, floor(angle))
                        scan_data[index] = distance
                scan_data_buffer.add(scan_data.copy())
                self.scan_error_count = 0
            except RPLidarException as error:
                if not self.measure.is_set():
                    break
                self._recover_scan_stream(error)

    def _recover_scan_stream(self, error):
        now = time.time()
        if now - self.last_scan_error_time > 3:
            self.scan_error_count = 0
        self.last_scan_error_time = now
        self.scan_error_count += 1

        print(f"LIDAR scan warning: {error}. Resync attempt {self.scan_error_count}")
        try:
            self.lidar.stop()
            self.lidar.disconnect()
            time.sleep(0.2)
            self.lidar.connect()
            self.lidar.start_motor()
            time.sleep(0.3)
        except Exception as recovery_error:
            print(f"LIDAR recovery failed: {recovery_error}")
            time.sleep(0.5)  

    def average_scan(self, scan_data_list):  
        # Creating ring buffer for averaging scan distances
        scan_data_zeros = np.where(np.array(scan_data_list) == 0, 0, 1)  # Mask to identify zero values
        divisor = np.sum(scan_data_zeros, axis=0)  # Count non-zero entries for each angle
        scan_data_sum = np.sum(scan_data_list, axis=0)  # Sum of distances for each angle
        with np.errstate(divide='ignore', invalid='ignore'):
            averaged_scan = np.where(divisor > 0, scan_data_sum / divisor, 0)  # Avoid division by zero
        
        # Return averaged distances as array
        return np.array(averaged_scan, dtype=float)
    
    def is_region_clear(self, direction):
        global scan_data_buffer
        # Calculate scan window angle
        measure_interval = int(np.degrees(np.arctan(((roboterwidth + horizontal_buffer)/2)/(following_distance - vertical_buffer))))
      #  print(f"Direction: {self.direction}° | Scan window: ±{measure_interval}°")
        

        # Get averaged distances from ring buffer
        distances = self.average_scan(scan_data_buffer.get())
        direction_window_idx = np.arange((direction-measure_interval), (direction+measure_interval), dtype=int) % 360
        scan_segment = distances[direction_window_idx]  # focus on a sector around the direction of movement
        obstacle_mask = (scan_segment < min_distance) & (scan_segment > 0)
        if np.any(obstacle_mask):
            return False
        return True

    def simple_logic_loop(self):
        global scan_data
        # Calculate scan window angle
        measure_interval = int(np.degrees(np.arctan(((roboterwidth + horizontal_buffer)/2)/(following_distance - vertical_buffer))))
        print("Starting simple logic loop")
        print(f"Direction: {self.direction}° | Scan window: ±{measure_interval}°")
        last_direction = self.direction
        
        while self.calculate.is_set():
            # Get averaged distances from ring buffer
            distances = self.average_scan(scan_data_buffer.get())
            direction_window_idx = np.arange((self.direction-measure_interval), (self.direction+measure_interval), dtype=int) % 360
            scan_segment = distances[direction_window_idx]  # focus on a sector around the direction of movement
            obstacle_mask = (scan_segment < min_distance) & (scan_segment > 0)
            if np.any(obstacle_mask):
                left_obstacle_sum = np.sum(obstacle_mask[:len(obstacle_mask)//2])
                right_obstacle_sum = np.sum(obstacle_mask[len(obstacle_mask)//2:])
                
                if left_obstacle_sum > right_obstacle_sum:
                    self.direction = (min(direction_window_idx[obstacle_mask]-measure_interval)%360)
                else:
                    self.direction = (min(direction_window_idx[obstacle_mask]+measure_interval)%360)
                print(f"New direction: {self.direction}")
                self.detect_obstacle.set()
                # return False # to indicate obstacle detected and direction change needed
                #reset obstacle pixels due to update policy 
                for angle in direction_window_idx[obstacle_mask]:
                    scan_data[angle] = 0.0
                time.sleep(3)
            else:
                self.detect_obstacle.clear()
            
            time.sleep(0.1)

    def logic_loop(self):
        global scan_data
        # compute window half-angle in degrees (use np.degrees for clarity)
        window_half_size = int(np.degrees(np.arctan2(self.dimensions_robot[0] / 2, self.dimensions_robot[1] / 2))) + 1
        angles = np.arange(-window_half_size, window_half_size + 1)
        angles_rad = np.deg2rad(angles)

        # short_sides: use sin(theta) (corrects the original "90 - rad" mistake)
        sin_vals = np.sin(angles_rad)
        sin_safe = np.where(np.abs(sin_vals) < 1e-6, np.inf, sin_vals)
        # not_aus_time is a time; this array is in time-units and will be multiplied by speed later
        short_sides = self.not_aus_time / sin_safe

        while self.calculate.is_set():
            future_edge_angle = int(np.degrees(np.arctan2(self.dimensions_robot[0]/2,
                                                          self.speed*self.not_aus_time + self.dimensions_robot[1]/2))) + 1
            window = short_sides[future_edge_angle:future_edge_angle+1] * self.speed
            self.stop_window = window
            # build scan_segment using modular indices to preserve length
            start_index = (self.direction - future_edge_angle) % 360
            idxs = (start_index + np.arange(0, 2*future_edge_angle + 1)) % 360
            scan_segment = np.array([scan_data[i] for i in idxs], dtype=float)

            # compare scan distances with allowed window distances (window is same length as scan_segment)
            # guarded subtract to avoid in-place alias issues
            diff = scan_segment - window
            if np.any(diff < 0):
                print("Obstacle detected, stopping")
                time.sleep(5)

    def get_driving_direction(self):
        if self.driving_direction == 1: 
            target_angle = 0
            return target_angle
        elif self.driving_direction == -1: 
            target_angle = 180
            return target_angle
        else:
            print("Please enter valid driving_direction")
            raise ValueError("Invalid driving_direction")
        
    #destructor
    def __del__(self):
        if getattr(self, "lidar", None) is not None:
            self.lidar.disconnect()
            print("LIDAR disconnected")

def get_front_sector(data, half_width=10):
    return [data[i % 360] for i in range(-half_width, half_width + 1)]


def is_obstacle_in_path():
    lidar = LIDAR(dimensions_robot=(500,500))
    lidar.start_measuring()
    lidar.start_logic()
    try:
        while True:
            #print(f"\rin front {scan_data}", end="", flush=True)
            print(f"Is region clear: {lidar.is_region_clear(0)}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        print()
        print('Stopping.')
        lidar.stop_measuring()
        lidar.stop_logic()

if __name__ == "__main__":
    if is_obstacle_in_path():
        print("Obstacle detected in path.")
    else:
        print("No obstacle detected in path.")
