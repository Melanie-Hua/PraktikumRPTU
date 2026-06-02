import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import deque
import mediapipe as mp

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode
)

# -----------------------------
# Plot setup
# -----------------------------
plt.ion() # enables intereactive plotting so program doesnt freeze when plotting 

fig, ax = plt.subplots()

stream_width = 640
stream_height = 480
sample_count = 0
average_timestamp = 0
average_closest_distance = 0
timestamp_sum = 0
closest_distance_sum = 0
closest_person_index= -1
x_data = []
y_data = []

line, = ax.plot([], [], 'r-') # r- means red solid line, line is the variable that will store the line object created by plot, which we can update later with new data

ax.set_xlabel("Frame") #label the x axis
ax.set_ylabel("Distance (m)") #label the y axis
ax.set_title("Closest Person Distance")


# -----------------------------
# RealSense setup
# -----------------------------
pipeline1 = rs.pipeline()
configu = rs.config()

configu.enable_stream(rs.stream.color, stream_width, stream_height, rs.format.bgr8, 30)
configu.enable_stream(rs.stream.depth, stream_width, stream_height, rs.format.z16, 30)

pipeline1.start(configu)

# -----------------------------
# MediaPipe setup
# -----------------------------
# Some mediapipe builds (Tasks-only) do not expose `mp.solutions`.
# Try to use it, otherwise fall back to a standard set of pose connections
try:
    POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS # defines which body parts to connect to a skeleton
except Exception:
    # Fallback connections (standard MediaPipe pose connections)
    POSE_CONNECTIONS = [
        (0,1),(0,2),(1,3),(2,4),(3,5),(4,6),(5,7),(6,8),(7,9),(8,10),
        (9,11),(10,12),(11,13),(12,14),(13,15),(14,16),(15,17),(16,18),
        (17,19),(18,20),(19,21),(20,22),(21,23),(22,24),(23,24),(24,26),
        (25,27),(26,28),(27,29),(28,30),(29,31),(30,32),(11,12),(23,24)
    ]

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="pose_landmarker.task"), 
        running_mode=RunningMode.VIDEO, num_poses=5) 
# creates cofiguration setting for pose landmarkers: loads ai model for detection, tells the program that it will process video frames, sets max number of people to detect

timestamp = 0 # timestamp to keep track of current frame number and calculate motion and smooth pose predictions

# -----------------------------
# Create Trackbar
# -----------------------------
cv2.namedWindow("Settings", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Settings", 400, 300)
cv2.createTrackbar("Kp", "Settings", 10, 50, lambda x: None)
cv2.createTrackbar("Target", "Settings", 50, 250, lambda x: None)
Velocity = 0
Target_Distance = 0.5
Kp = 0
previous_position = 200
theoretical_position = 200

# -----------------------------
# Main loop
# -----------------------------
try:

    with PoseLandmarker.create_from_options(options) as landmarker:

        while True:

            frames = pipeline1.wait_for_frames()

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            # Convert to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Flip images
            depth_image = cv2.flip(depth_image, 1)
            color_image = cv2.flip(color_image, 1)

            # Depth colormap for display
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )

            # Convert BGR -> RGB for MediaPipe
            rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=rgb_image
            ) # converts the color image to a format that mediapipe can process

            timestamp += 1 # timestamp counts up by 1 for each frame

            result = landmarker.detect_for_video(
                mp_image, timestamp
            ) # with the poselandmarker as landmarker in the frame mp image, detect the pose landmarks and store them in results

            # --------------------------------
            # Find closest person
            # --------------------------------
            # inital placeholder values for person is none and closest distance to max, so any new distance is closer
            closest_person = None
            closest_distance = 9999

            h, w, _ = color_image.shape # get the height, width and number of channels (channels are redundant here)

            if result.pose_landmarks: #if any humans were deteced 

                closest_person_index = -1 # -1 means no person detected yet, will be updated to the index of the closest person in the results if a valid distance is found
                closest_distance = 9999

                # --------------------------------
                # FIRST PASS:
                # find closest person
                # --------------------------------
                for i, person in enumerate(result.pose_landmarks): # for each person detected in the frame, set an index with enumerate
                    # person contains the pose landmarks 
                    left_shoulder = person[11] #get the coordinates of the left and right shoulder
                    right_shoulder = person[12]

                    # chest center in pixels
                    cx = int(
                        ((left_shoulder.x + right_shoulder.x) / 2) * w
                    )

                    cy = int(
                        ((left_shoulder.y + right_shoulder.y) / 2) * h
                    )

                    if 0 <= cx < w and 0 <= cy < h: # if chest center is withing the frame,

                        distance = depth_frame.get_distance(cx, cy) # get the distance to the chest center from the depth frame

                        if 0 < distance < closest_distance: # if the distance is valid and closer than the closest distance found so far, update closest person and distance
                            closest_distance = distance # update closest distance
                            closest_person_index = i # update closest person index

                # --------------------------------
                # SECOND PASS:
                # draw ALL skeletons
                # --------------------------------
                for i, person in enumerate(result.pose_landmarks): # for each person detected in the frame,

                    points = [] # create empty list to store joint coordinates

                    for lm in person: # for each joint in the detected person,

                        x = int(lm.x * w) # convert normalized coordinates to pixel coordinates
                        y = int(lm.y * h)

                        points.append((x, y)) # append/add the joint coordinates to the points list

                        # draw points on the joints
                        cv2.circle(color_image, (x, y), 4, (0,255,0), -1)

                    # draw skeleton lines with the predefined connections between joints in the POSE_CONNECTIONS list, which is provided by mediapipe and defines which joints to connect to form the skeleton
                    for connection in POSE_CONNECTIONS:
                        a, b = connection # a,b means something like the connection from left sholder to left elbow
                        if a < len(points) and b < len(points): # len asks how many joints were detected. if the index of the joint is smaller than the number of joints detected,
                            cv2.line(color_image, points[a], points[b], 
                                (255,0,0), 2) # draw a line between the two joints defined by the connection with blue color and thickness of 2

                    # --------------------------------
                    # LABEL ONLY CLOSEST PERSON
                    # --------------------------------
                    if i == closest_person_index: # if this person is the closest person, 
                        cv2.putText(
                            color_image, f"{closest_distance:.2f} m",
                            points[0], cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0,255,255), 2) #label them with the distance in meters

                        
            # --------------------------------
            # Plotting
            # --------------------------------
            if closest_person_index != -1:

                timestamp_sum += timestamp
                closest_distance_sum += closest_distance
                sample_count += 1

                # Every 5 VALID detections
                if sample_count == 10:

                    average_timestamp = timestamp_sum / 10
                    average_closest_distance = (closest_distance_sum / 10)%999.9 #999.9 for in case the detected person distance is set at 9999 according to line 131

                    x_data.append(average_timestamp)
                    y_data.append(average_closest_distance)
                    #print(average_closest_distance)

                    # Reset accumulators
                    timestamp_sum = 0
                    closest_distance_sum = 0
                    sample_count = 0


            line.set_xdata(x_data) #moves graphline to new points of x_data
            line.set_ydata(y_data)
            ax.relim() # relim = recalculate limits (for the axes)
            ax.autoscale_view() #autoscales the graph axes
            
            plt.draw()
            plt.pause(0.001) # to give matplotlib time to process so the window doesnt freeze
            # --------------------------------
            # draw lines !!! DOESNT WORK YET!!! need to make a virtual robot thats not influenced by the camera values except for the target distance 
            # --------------------------------    
            Kp = cv2.getTrackbarPos("Kp", "Settings")/10
            Target_Distance = cv2.getTrackbarPos("Target", "Settings")
            if closest_person_index == -1:
                print("SEARCHING: No person detected")
            elif closest_distance <= 5:
                Velocity = Kp * ((closest_distance*100-Target_Distance) - theoretical_position) # simple proportional controller, distance is multiplied by 100 to convert to cm scale to match the target distance scale
                theoretical_position = int(previous_position + Velocity) # convert to pixel scale
                theoretical_position = max(15, min(415, theoretical_position))
                previous_position = theoretical_position
                print("FOLLOWING:", closest_distance, "with", Target_Distance, "cm Target distance")
            else:
                print("LOCKED: Person more than 5m away, not following")
            color_image = cv2.line(color_image, (15, stream_height-15), (415,stream_height-15), (255,255,255), 1) # horizontal center line
            color_image = cv2.line(color_image, (int(closest_distance*100), stream_height-20), (int(closest_distance*100), stream_height-10), (255,0,0), 3)
            color_image = cv2.line(color_image, (int(closest_distance*100-Target_Distance),stream_height-20), (int(closest_distance*100-Target_Distance),stream_height-10), (0,255,0), 3) # left boundary
            color_image = cv2.rectangle(color_image, pt1=(theoretical_position-2, stream_height-17), pt2=(theoretical_position+2, stream_height-13), color=(0, 0, 255), thickness=-1)
            cv2.putText(color_image, f"{theoretical_position/100:.2f} m", (theoretical_position + 15, stream_height-15), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,255,255), 1) # label them with the distance in meters
            # --------------------------------
            # Show windows
            # --------------------------------
            cv2.imshow("Color Frame", color_image)
            cv2.imshow("Depth Frame", depth_colormap)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:

    pipeline1.stop()
    cv2.destroyAllWindows()