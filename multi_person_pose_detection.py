import cv2
import mediapipe as mp
import time

from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

# MediaPipe connection map (for skeleton lines)
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

# -----------------------------
# Setup Pose Landmarker
# -----------------------------
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker.task"),
    running_mode=RunningMode.VIDEO,
    num_poses=5  # detect up to 5 people
)

cap = cv2.VideoCapture(2)  # change to 1 or 2 if needed

timestamp = 0

with PoseLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame
        )

        # VIDEO mode requires timestamps (in ms)
        timestamp += 1

        result = landmarker.detect_for_video(mp_image, timestamp)

        # -----------------------------
        # DRAW POSE LANDMARKS
        # -----------------------------
        if result.pose_landmarks:

            for person in result.pose_landmarks:

                h, w, _ = frame.shape
                points = []

                # Convert landmarks to pixel coordinates
                for lm in person:
                    x = int(lm.x * w)
                    y = int(lm.y * h)
                    points.append((x, y))

                    # draw joints
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                # Draw skeleton connections
                for connection in POSE_CONNECTIONS:
                    a, b = connection

                    if a < len(points) and b < len(points):
                        cv2.line(frame, points[a], points[b], (255, 0, 0), 2)

        cv2.imshow("Multi-Person Pose", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()