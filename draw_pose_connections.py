import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose #selects the pose module inside mediapipe 
pose = mp_pose.Pose() #creates an instance of the pose class to process the video frames
mp_draw = mp.solutions.drawing_utils #gives tools to draw the skeleton automatically

cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #frame = cv2.flip(frame, 1)

    #rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = pose.process(frame) # sends the image to mediapipe so it analyses it and returns results

    if results.pose_landmarks: #if a person was detected:
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        ) # draw the skeleton on frame with the detected joints/landmarks and use POSE_CONNECTIONS that tells the program what to connect with each other

    cv2.imshow("Skeleton", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 