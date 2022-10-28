import cv2
from matplotlib import pyplot as plt
import os
import pathlib
import numpy as np

"""
In this lesson we'll learn to:

    1: How to use the HSV Color Space to Filter by Color

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
#       1: How to use the HSV Color Space to Filter by Color
# ----------------------------------------

# https://www.researchgate.net/publication/331742081/figure/fig1/AS:736357450256386@1552572705787/
# HSV-Color-Space-4-After-changing-the-color-space-the-values-of-all-the-pixels-on-the.png


image = cv2.imread('../../SRC/images/truck.jpg')

# define range of BLUE color in HSV
lower = np.array([90, 0, 0])
upper = np.array([135, 255, 255])

# Convert image from RBG/BGR to HSV, so we easily filter
hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Use inRange to capture only the values between lower & upper
mask = cv2.inRange(hsv_img, lower, upper)

# Perform Bitwise AND on mask and our original frame
res = cv2.bitwise_and(image, image, mask=mask)

imshow('Original Truck', image)
imshow('Truck mask', mask)
imshow('Filtered Color Only Truck', res)

image = cv2.imread("../../SRC/images/Hillary.jpg")

img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# lower mask (0-10)
lower_red = np.array([0, 0, 0])
upper_red = np.array([10, 255, 255])
mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

# upper mask (170-180)
lower_red = np.array([170, 0, 0])
upper_red = np.array([180, 255, 255])
mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

# join masks
mask = mask0 + mask1

# Perform Bitwise AND on mask and our original frame
res = cv2.bitwise_and(image, image, mask=mask)

imshow('Original Hillary', image)
imshow('Hillary mask', mask)
imshow('Filtered Color Only Hillary', res)
