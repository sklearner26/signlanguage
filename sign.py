import numpy as np
import cv2
from collections import deque
import streamlit as st
from PIL import Image

# Function to handle the OpenCV logic for capturing and processing frames
def process_sign_gesture():
    # Initialize variables for OpenCV and color detection
    cv2.namedWindow("Color detectors")
    cv2.createTrackbar("Upper Hue", "Color detectors", 153, 180, lambda x: None)
    cv2.createTrackbar("Upper Saturation", "Color detectors", 255, 255, lambda x: None)
    cv2.createTrackbar("Upper Value", "Color detectors", 255, 255, lambda x: None)
    cv2.createTrackbar("Lower Hue", "Color detectors", 64, 180, lambda x: None)
    cv2.createTrackbar("Lower Saturation", "Color detectors", 72, 255, lambda x: None)
    cv2.createTrackbar("Lower Value", "Color detectors", 49, 255, lambda x: None)

    # Initialize points deque for tracking
    bpoints = [deque(maxlen=1024)]
    gpoints = [deque(maxlen=1024)]
    rpoints = [deque(maxlen=1024)]
    ypoints = [deque(maxlen=1024)]

    # Colors for drawing
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    colorIndex = 0

    # The kernel for dilation
    kernel = np.ones((5, 5), np.uint8)

    # Initial setup for painting window (background)
    paintWindow = np.zeros((471, 636, 3)) + 255
    paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
    paintWindow = cv2.rectangle(paintWindow, (160, 1), (255, 65), colors[0], -1)
    paintWindow = cv2.rectangle(paintWindow, (275, 1), (370, 65), colors[1], -1)
    paintWindow = cv2.rectangle(paintWindow, (390, 1), (485, 65), colors[2], -1)
    paintWindow = cv2.rectangle(paintWindow, (505, 1), (600, 65), colors[3], -1)
    cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "BLUE", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "GREEN", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "RED", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2, cv2.LINE_AA)

    # Start capturing video from webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get current trackbar values
        u_hue = cv2.getTrackbarPos("Upper Hue", "Color detectors")
        u_saturation = cv2.getTrackbarPos("Upper Saturation", "Color detectors")
        u_value = cv2.getTrackbarPos("Upper Value", "Color detectors")
        l_hue = cv2.getTrackbarPos("Lower Hue", "Color detectors")
        l_saturation = cv2.getTrackbarPos("Lower Saturation", "Color detectors")
        l_value = cv2.getTrackbarPos("Lower Value", "Color detectors")
        Upper_hsv = np.array([u_hue, u_saturation, u_value])
        Lower_hsv = np.array([l_hue, l_saturation, l_value])

        # Mask for detecting the color
        Mask = cv2.inRange(hsv, Lower_hsv, Upper_hsv)
        Mask = cv2.erode(Mask, kernel, iterations=1)
        Mask = cv2.morphologyEx(Mask, cv2.MORPH_OPEN, kernel)
        Mask = cv2.dilate(Mask, kernel, iterations=1)

        # Find contours
        cnts, _ = cv2.findContours(Mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center = None

        # If contours are found, draw them
        if len(cnts) > 0:
            cnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            M = cv2.moments(cnt)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        # Send the frame to Streamlit as an image
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Display the frame in Streamlit
        st.image(img_pil, caption="Sign Gesture", use_column_width=True)

        # Break loop if user presses 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
