import pyrealsense2 as rs
import numpy as np
import cv2

# Configure the pipeline
pipeline1 = rs.pipeline()
configu = rs.config()

#reference points
x_o = 0
y_o = 240 
x_c = 0.0
y_c = 0.0

# Enable both color and depth streams
# variable where rs.config() is stored.enable_stream(rs.stream.color or depth, width, height, rs.format.whats measured (bgr/z depth)how many bits, fps)
configu.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
configu.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Start streaming
pipeline1.start(configu)

# Optional: colorize depth for display
# raw 16bit integers into colors: value 1200 --> 1,2m and colorizer maps into a color
#colorizer = rs.colorizer()

stop = cv2.imread('stop.png')

cv2.namedWindow("Settings", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Settings", 400, 300)
cv2.createTrackbar("color min", "Settings", 150, 179, lambda x: None)
cv2.createTrackbar("color max", "Settings", 179, 179, lambda x: None)
cv2.createTrackbar("saturation min", "Settings", 100, 255, lambda x: None)
cv2.createTrackbar("saturation max", "Settings", 255, 255, lambda x: None)
cv2.createTrackbar("brightness min", "Settings", 50, 255, lambda x: None)
cv2.createTrackbar("brightness max", "Settings", 255, 255, lambda x: None)

kernel = np.ones((5, 5), np.uint8)
# create a trackbar

try:
    while True:
        frames = pipeline1.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
        
        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        
        # Colorize depth for visualization
        #depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        hmin = cv2.getTrackbarPos("color min", "Settings")
        hmax = cv2.getTrackbarPos("color max", "Settings")
        smin = cv2.getTrackbarPos("saturation min", "Settings")
        smax = cv2.getTrackbarPos("saturation max", "Settings")
        vmin = cv2.getTrackbarPos("brightness min", "Settings")
        vmax = cv2.getTrackbarPos("brightness max", "Settings")

        # convert to HSV
        hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

        filter_upper = np.array([hmax, smax, vmax], np.uint8)
        filter_lower = np.array([hmin, smin, vmin], np.uint8)
        mask = cv2.inRange(hsv_image, filter_lower, filter_upper)
        mask = cv2.dilate(mask, kernel)
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        (contours, hierarchy) = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # tree lets you know what contour is inside another
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cx = x + w // 2
            cy = y + h // 2
            blob_coordinates = (cx, cy)
            color_image = cv2.circle(color_image, blob_coordinates, 5, (0, 255, 0), -1)
            distance = depth_frame.get_distance(cx, cy)
            label = f"distance: {distance:.2f}m"
            if 0 < distance < 0.65:
                cv2.imshow('STOP', stop)
            color_image = cv2.putText(color_image, label, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        
        # Show frames
        cv2.imshow('Color Frame', color_image)
        cv2.imshow('HSV Frame', hsv_image)
        cv2.imshow('Mask', mask)
        cv2.imshow('Depth Frame', depth_colormap)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline1.stop()
    cv2.destroyAllWindows()