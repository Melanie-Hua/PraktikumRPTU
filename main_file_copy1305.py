import pyrealsense2 as rs
import numpy as np
import cv2
import mediapipe as mp

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode 
)

# -----------------------------
# RealSense setup
# -----------------------------
pipeline1 = rs.pipeline()
configu = rs.config()

configu.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
configu.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

pipeline1.start(configu)

# -----------------------------
# MediaPipe setup
# -----------------------------
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS # defines which body parts to connect to a skeleton

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="pose_landmarker.task"), 
        running_mode=RunningMode.VIDEO, num_poses=5
) # creates cofiguration setting for pose landmarkers: loads ai model for detection, tells the program that it will process video frames, sets max number of people to detect

timestamp = 0 # timestamp to keep track of current frame number and calculate motion and smooth pose predictions

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
            # Show windows
            # --------------------------------
            cv2.imshow("Color Frame", color_image)
            cv2.imshow("Depth Frame", depth_colormap)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

finally:

    pipeline1.stop()
    cv2.destroyAllWindows()