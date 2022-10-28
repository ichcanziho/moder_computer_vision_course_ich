import cv2
from matplotlib import pyplot as plt
import os
import pathlib
import numpy as np

"""
In this lesson we'll learn to:

    1: Use OpenCV's getPerspectiveTransform
    2: Use findContours to get corners and automate perspective Transform

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
#       1: Use OpenCV's getPerspectiveTransform
# ----------------------------------------

image = cv2.imread('../../SRC/images/scan.jpg')

# Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow('After thresholding', th2)

# Use a copy of your image e.g. edged.copy(), since findContours alters the image
contours, hierarchy = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw all contours, note this overwrites the input image (inplace operation)
# Use '-1' as the 3rd parameter to draw all
cv2.drawContours(image, contours, -1, (0, 255, 0), thickness=2)
imshow('Contours overlaid on original image', image)

print("Number of Contours found = " + str(len(contours)))

# Approximate our contour above to just 4 points using approxPolyDP

# Sort contours large to small by area

sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
approx = None
# loop over the contours
for cnt in sorted_contours:
    # approximate the contour
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.05 * perimeter, True)

    if len(approx) == 4:
        break

# Our x, y coordinates of the four corners
print("Our 4 corner points are:")
print(approx)


# ----------------------------------------
#       2: Use findContours to get corners and automate perspective Transform
# ----------------------------------------

# Note: We manually matched the order of the points

# Order obtained here is top left, bottom left, bottom right, top right
inputPts = np.float32(approx)

outputPts = np.float32([[0, 0],
                        [0, 800],
                        [500, 800],
                        [500, 0]])

# Get our Transform Matrix, M
M = cv2.getPerspectiveTransform(inputPts, outputPts)
# Apply the transform Matrix M using Warp Perspective
dst = cv2.warpPerspective(image, M, (500, 800))

imshow("Perspective", dst)


def sort_corners(corners: np.array):
    corners = corners.reshape((4, 2))
    corners = corners.tolist()
    corners.sort(key=lambda x: x[0])
    left_corners = corners[:2]
    right_corners = corners[2:]
    top_left = left_corners[0] if left_corners[0][0] > left_corners[0][1] else left_corners[1]
    bottom_left = left_corners[0] if left_corners[0][0] < left_corners[0][1] else left_corners[1]
    top_right = right_corners[0] if right_corners[0][0] > right_corners[0][1] else right_corners[1]
    bottom_right = right_corners[0] if right_corners[0][0] < right_corners[0][1] else right_corners[1]
    sorted_corners = [top_left, top_right, bottom_right, bottom_left]
    width = np.linalg.norm(np.array(top_left) - np.array(top_right))
    height = np.linalg.norm(np.array(top_left) - np.array(bottom_left))
    ratio = width/height
    sorted_corners = np.array(sorted_corners)
    sorted_corners = sorted_corners.reshape((4, 1, 2))
    np.float32(sorted_corners)
    return sorted_corners, ratio


def get_new_corners(ratio, height):
    width = int(ratio*height)
    return np.float32([[0, 0], [0, height], [width, height], [width, 0]])


