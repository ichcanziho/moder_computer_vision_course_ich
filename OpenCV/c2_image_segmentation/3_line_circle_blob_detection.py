import cv2
from matplotlib import pyplot as plt
import os
import numpy as np

"""
In this lesson we'll learn to:

    1: Hough lines
    2: Probabilistic Hough lines
    3: Hough Circles
    4: Blob Detection

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
        f = os.path.basename(__file__)
        f = f.split(".")[0]
        plt.savefig(f"../outputs/c2_image_segmentation/{f}/{title}.png")
    plt.show()


# ----------------------------------------
#       1: Hough lines
# ----------------------------------------

"""
The Hough transform takes a binary edge map as input and attempts to locate edges placed as straight lines. The idea of 
the Hough transform is, that every edge point in the edge map is transformed to all possible lines that could pass 
through that point.

cv2.HoughLines(binarized/thresholded image, 𝜌 accuracy, 𝜃 accuracy, threshold)

Threshold here is the minimum vote for it to be considered a line
"""
image = cv2.imread('../../SRC/images/soduku.jpg')
imshow('Original sodoku', image)

# Grayscale and Canny Edges extracted
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 170, apertureSize=3)

# Run HoughLines using a rho accuracy of 1 pixel
# theta accuracy of np.pi / 180 which is 1 degree
# Our line threshold is set to 240 (number of points on-line)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 240)

# We iterate through each line and convert it to the format
# required by cv2.lines (i.e. requiring end points)
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * a)
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * a)
    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

imshow('Hough Lines', image)

# ----------------------------------------
#       2: Probabilistic Hough lines
# ----------------------------------------

"""
A Hough Transform is considered probabilistic if it uses random sampling of the edge points. These algorithms can be 
divided based on how they map image space to parameter space.

cv2.HoughLinesP(binarized image, 𝜌 accuracy, 𝜃 accuracy, threshold, minimum line length, max line gap)
"""

# Grayscale and Canny Edges extracted
image = cv2.imread('../../SRC/images/soduku.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 170, apertureSize=3)

# Again we use the same rho and theta accuracies
# However, we specific a minimum vote (pts along line) of 100
# and Min line length of 3 pixels and max gap between lines of 25 pixels
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, 3, 25)
print(lines.shape)

for x in range(0, len(lines)):
    for x1, y1, x2, y2 in lines[x]:
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

imshow('Probabilistic Hough Lines', image)

# ----------------------------------------
#       3: Hough Circles
# ----------------------------------------

"""
cv2.HoughCircles(image, method, dp, MinDist, param1, param2, minRadius, MaxRadius)

Method - currently only cv2.HOUGH_GRADIENT available
dp - Inverse ratio of accumulator resolution
MinDist - the minimum distance between the center of detected circles
param1 - Gradient value used in the edge detection
param2 - Accumulator threshold for the HOUGH_GRADIENT method (lower allows more circles to be detected 
(false positives))
minRadius - limits the smallest circle to this size (via radius)
MaxRadius - similarly sets the limit for the largest circles

"""

image = cv2.imread('../../SRC/images/Circles_Packed_In_Square_11.jpeg')
imshow('Circles', image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.medianBlur(gray, 5)

circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1.2, 25)

cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)

circles = np.uint16(np.around(circles))

for i in circles[0, :]:
    # draw the outer circle
    cv2.circle(image, (i[0], i[1]), i[2], (0, 0, 255), 5)

    # draw the center of the circle
    cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 8)

imshow('Detected circles', image)

# ----------------------------------------
#       4: Blob Detection
# ----------------------------------------
"""
The function cv2.drawKeypoints takes the following arguments:

cv2.drawKeypoints(input image, keypoints, blank_output_array, color, flags)

flags:

cv2.DRAW_MATCHES_FLAGS_DEFAULT
cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG
cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS

"""
# Read image
image = cv2.imread("../../SRC/images/Sunflowers.jpg")
imshow("Original flowers", image)

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create()

# Detect blobs.
keypoints = detector.detect(image)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of
# the circle corresponds to the size of blob
blank = np.zeros((1, 1))
blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 255, 0),
                          cv2.DRAW_MATCHES_FLAGS_DEFAULT)

# Show keypoints
imshow("Blobs", blobs)
