import cv2
from matplotlib import pyplot as plt
import os
import pathlib
import numpy as np

"""
In this lesson we'll learn to:

    1: How to use the Mean Shift Algorithm in OpenCV
    2: Use CAM-SHIFT in OpenCV

"""


# Define our imshow function
def imshow(title="Image", img=None, size=10, save=True):
    # we have added the size parameter to control de fig-size of the image while plotting with matplotlib, for that
    # we obtain the aspect ratio of the image dividing the width by the height.
    h, w = img.shape[0], img.shape[1]
    aspect_ratio = w / h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(title)
    if save:
        folder = str(pathlib.Path().resolve()).split("/")[-1]
        f_name = os.path.basename(__file__).split(".")[0]
        if not os.path.exists(f"../outputs/{folder}"):
            os.makedirs(f"../outputs/{folder}")
        if not os.path.exists(f"../outputs/{folder}/{f_name}"):
            os.makedirs(f"../outputs/{folder}/{f_name}")
        plt.savefig(f"../outputs/{folder}/{f_name}/{title}.png")
    plt.show()


# ---------------------------------------------------------------
#       1: How to use the Mean Shift Algorithm in OpenCV
# ---------------------------------------------------------------
"""


The intuition behind the meanshift is simple. Consider you have a set of points. (It can be a pixel distribution like 
histogram backpropagation). You are given a small window ( may be a circle) and you have to move that window to the 
area of maximum pixel density (or maximum number of points). It is illustrated in the simple image given below: 

Mean shift is a hill climbing algorithm which involves shifting this kernel iteratively to a higher density region until 
convergence. Every shift is defined by a mean shift vector. The mean shift vector always points toward the direction 
of the maximum increase 

"""

cap = cv2.VideoCapture('../../SRC/videos/slow.mp4')

# take first frame of the video
ret, frame = cap.read()

# setup initial location of window
r, h, c, w = 250, 90, 400, 125  # simply hardcoded the values
track_window = (c, r, w, h)

# set up the ROI for tracking
roi = frame[r:r + h, c:c + w]
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # apply mean-shift to get the new location
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)

    # Draw it on image
    x, y, w, h = track_window
    img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    cv2.imshow('Tracking Mean-shift', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

# ---------------------------------------------------------------
#       2: Use CAM-SHIFT in OpenCV
# ---------------------------------------------------------------
"""
It is almost same as meanshift, but it returns a rotated rectangle (that is our result) and box parameters (used to be 
passed as search window in next iteration). 
"""
cap = cv2.VideoCapture('../../SRC/videos/slow.mp4')

# take first frame of the video
ret, frame = cap.read()

# setup initial location of window
r, h, c, w = 250, 90, 400, 125  # simply hardcoded the values
track_window = (c, r, w, h)

# set up the ROI for tracking
roi = frame[r:r + h, c:c + w]
hsv_roi = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by at least 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # apply cam-shift to get the new location
    ret, track_window = cv2.CamShift(dst, track_window, term_crit)

    # Draw it on image
    pts = cv2.boxPoints(ret)
    pts = np.int0(pts)
    img2 = cv2.polylines(frame, [pts], True, 255, 2)

    cv2.imshow('Tracking Cam-shift', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
