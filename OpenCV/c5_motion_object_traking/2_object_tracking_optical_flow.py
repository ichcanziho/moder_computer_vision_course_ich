import cv2
from matplotlib import pyplot as plt
import os
import pathlib
import numpy as np

"""
In this lesson we'll learn to:

    1: How to use Optical Flow in OpenCV
    2: Then use Dense Optical Flow

"""

# ---------------------------------------------------------------
#       1: How to use Optical Flow in OpenCV
# ---------------------------------------------------------------

"""
Optical flow is the pattern of apparent motion of image objects between two consecutive frames caused by the movement 
of object or camera. It is 2D vector field where each vector is a displacement vector showing the movement of points 
from first frame to second. Consider the image below.

It shows a ball moving in 5 consecutive frames. The arrow shows its displacement vector. Optical flow has many 
applications in areas like :

    Structure from Motion
    Video Compression
    Video Stabilization

Optical flow works on several assumptions:

    The pixel intensities of an object do not change between consecutive frames.
    Neighbouring pixels have similar motion


"""

cap = cv2.VideoCapture('../../SRC/videos/walking.mp4')
# Set parameters for ShiTomasi corner detection
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Set parameters for lucas kanade optical flow
lucas_kanade_params = dict(winSize=(15, 15),
                           maxLevel=2,
                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
# Used to create our trails for object movement in the image
color = np.random.randint(0, 255, (100, 3))

# Take first frame and find corners in it
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Find initial corner locations
prev_corners = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(prev_frame)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    new_corners, status, errors = cv2.calcOpticalFlowPyrLK(prev_gray,
                                                           frame_gray,
                                                           prev_corners,
                                                           None,
                                                           **lucas_kanade_params)

    # Select and store good points
    good_new = new_corners[status == 1]
    good_old = prev_corners[status == 1]

    # Draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        print(prev_frame.shape, mask.shape)
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

    img = cv2.add(frame, mask)

    # Now update the previous frame and previous points
    prev_gray = frame_gray.copy()
    prev_corners = good_new.reshape(-1, 1, 2)

    cv2.imshow('Optical Flow - Lucas-Kanade', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

# ---------------------------------------------------------------
#       2: Then use Dense Optical Flow
# ---------------------------------------------------------------

"""
Lucas-Kanade method computes optical flow for a sparse feature set (in our example, corners detected using Shi-Tomasi 
algorithm). OpenCV provides another algorithm to find the dense optical flow. It computes the optical flow for all the 
points in the frame. It is based on Gunner Farneback’s algorithm which is explained in “Two-Frame Motion Estimation 
Based on Polynomial Expansion” by Gunner Farneback in 2003.

Below sample shows how to find the dense optical flow using above algorithm. We get a 2-channel array with optical flow
vectors, (u,v). We find their magnitude and direction. We color code the result for better visualization.

    Direction corresponds to Hue value of the image.
    Magnitude corresponds to Value plane. See the code below:

"""

cap = cv2.VideoCapture('../../SRC/videos/walking.mp4')

# Get first frame
ret, first_frame = cap.read()
previous_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(first_frame)
hsv[...,1] = 255

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    next_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Computes the dense optical flow using the Gunnar Farneback’s algorithm
    flow = cv2.calcOpticalFlowFarneback(previous_gray, next_frame,
                                        None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # use flow to calculate the magnitude (speed) and angle of motion
    # use these values to calculate the color to reflect speed and angle
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = angle * (180 / (np.pi / 2))
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    final = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Store current image as previous image
    previous_gray = next_frame

    cv2.imshow('Dense Optical Flow', final)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
