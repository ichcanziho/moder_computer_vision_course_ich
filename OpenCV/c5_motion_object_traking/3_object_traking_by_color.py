import cv2
from matplotlib import pyplot as plt
import os
import pathlib
import numpy as np

"""
In this lesson we'll learn to:

    1: How to use an HSV Color Filter to Create a Mask and then Track our Desired Object

"""

# ---------------------------------------------------------------
#       1: How to use an HSV Color Filter to Create a Mask and then Track our Desired Object
# ---------------------------------------------------------------

cap = cv2.VideoCapture('../../SRC/videos/bmwm4.mp4')
# define range of color in HSV
lower = np.array([20,50,90])
upper = np.array([40,255,255])

# Create empty points array
points = []

ret, frame = cap.read()
Height, Width = frame.shape[:2]
frame_count = 0
radius = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv_img, lower, upper)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create empty centre array to store centroid center of mass
    center = int(Height / 2), int(Width / 2)

    if len(contours) > 0:
        # Get the largest contour and its center
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)

        # Sometimes small contours of a point will cause a divison by zero error
        try:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        except ZeroDivisionError:
            center = int(Height / 2), int(Width / 2)

        # Allow only contours that have a larger than 25 pixel radius
        if radius > 25:
            # Draw circle and leave the last center creating a trail
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 0, 255), 2)
            cv2.circle(frame, center, 5, (0, 255, 0), -1)
        # Log center points
        points.append(center)
    # If radius large enough, we use 25 pixels
    if radius > 25:

        # loop over the set of tracked points
        for i in range(1, len(points)):
            try:
                cv2.line(frame, points[i - 1], points[i], (0, 255, 0), 2)
            except:
                pass

        # Make frame count zero
        frame_count = 0

    cv2.imshow('Color Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
