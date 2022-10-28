import cv2
from matplotlib import pyplot as plt
import os
import pathlib
import numpy as np

"""
In this lesson we'll learn to:

    1: Background Subtraction with Gaussian Mixture-based Background/Foreground Segmentation Algorithm
    2: Improved adaptive Gaussian mixture model for background subtraction
    3: Foreground Subtraction

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


# ----------------------------------------
#       1: Background Subtraction with Gaussian Mixture-based Background/Foreground Segmentation Algorithm
# ----------------------------------------

"""
Background subtraction (BS) is a common and widely used technique for generating a foreground mask (namely, a binary 
image containing the pixels belonging to moving objects in the scene) by using static cameras.

As the name suggests, BS calculates the foreground mask performing a subtraction between the current frame and a 
background model, containing the static part of the scene or, more in general, everything that can be considered as 
background given the characteristics of the observed scene.

Background modeling consists of two main steps:

    1: Background Initialization;
    2: Background Update.

In the first step, an initial model of the background is computed, while in the second step that model is updated in 
order to adapt to possible changes in the scene.

"""

cap = cv2.VideoCapture('../../SRC/videos/walking.mp4')

w = int(cap.get(3))
h = int(cap.get(4))
# out = cv2.VideoWriter('walking_output_GM.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (w, h))
foreground_background = cv2.createBackgroundSubtractorMOG2()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    foreground_mask = foreground_background.apply(frame)
    # kernel = np.ones((3, 3), np.uint8)
    # foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
    # foreground_mask = cv2.bitwise_and(frame, frame, mask=foreground_mask)

    # cv2.imshow('frame', frame)
    cv2.imshow('foreground frame', foreground_mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

# ----------------------------------------
#       2: Improved adaptive Gaussian mixture model for background subtraction
# ----------------------------------------
cap = cv2.VideoCapture('../../SRC/videos/walking.mp4')

w = int(cap.get(3))
h = int(cap.get(4))
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
foreground_background = cv2.createBackgroundSubtractorKNN()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    foreground_mask = foreground_background.apply(frame)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
    cv2.imshow('foreground frame', foreground_mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

# ----------------------------------------
#       3: Foreground Subtraction
# ----------------------------------------
cap = cv2.VideoCapture('../../SRC/videos/walking.mp4')

w = int(cap.get(3))
h = int(cap.get(4))
ret, frame = cap.read()
average = np.float32(frame)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 0.01 is the weight of image, play around to see how it changes
    cv2.accumulateWeighted(frame, average, 0.01)

    # Scales, calculates absolute values, and converts the result to 8-bit
    background = cv2.convertScaleAbs(average)
    cv2.imshow("Frame", frame)
    cv2.imshow('background frame', background)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
