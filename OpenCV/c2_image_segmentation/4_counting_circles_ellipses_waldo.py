import cv2
from matplotlib import pyplot as plt
import os
import numpy as np

"""
In this lesson we'll learn to:

    1: Mini Project on Counting Circular Blobs
    2: Mini Project on using Template Matching to find Waldo

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
#       1: Mini Project on Counting Circular Blobs
# ----------------------------------------

# Load image
image = cv2.imread("../../SRC/images/blobs.jpg", 0)
imshow('Original Blolb Image', image)

# Initialize the detector using the default parameters
detector = cv2.SimpleBlobDetector_create()

# Detect blobs
keypoints = detector.detect(image)

# Draw blobs on our image as red circles
blank = np.zeros((1, 1))
blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 0, 255),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs = len(keypoints)
text = "Total Number of Blobs: " + str(len(keypoints))
cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 255), 2)

# Display image with blob keypoints
imshow("Blobs using default parameters", blobs)

# Set our filtering parameters
# Initialize parameter setting using cv2.SimpleBlobDetector
params = cv2.SimpleBlobDetector_Params()

# Set Area filtering parameters
params.filterByArea = True
params.minArea = 100

# Set Circularity filtering parameters
params.filterByCircularity = True
params.minCircularity = 0.9

# Set Convexity filtering parameters
params.filterByConvexity = False
params.minConvexity = 0.2

# Set inertia filtering parameters
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(image)

# Draw blobs on our image as red circles
blank = np.zeros((1, 1))
blobs = cv2.drawKeypoints(image, keypoints, blank, (0, 255, 0),
                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

number_of_blobs = len(keypoints)
text = "Number of Circular Blobs: " + str(len(keypoints))
cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

# Show blobs
imshow("Filtering Circular Blobs Only", blobs)

# ----------------------------------------
#       2: Mini Project on using Template Matching to find Waldo
# ----------------------------------------
"""
Notes on Template Matching There are a variety of methods to perform template matching, but in this case we are 
using the correlation coefficient which is specified by the flag cv2.TM_CCOEFF. 

So what exactly is the cv2.matchTemplate function doing? Essentially, this function takes a “sliding window” of our 
waldo query image and slides it across our puzzle image from left to right and top to bottom, one pixel at a time. 
Then, for each of these locations, we compute the correlation coefficient to determine how “good” or “bad” the match 
is. 

Regions with sufficiently high correlation can be considered “matches” for our waldo template. From there, 
all we need is a call to cv2.minMaxLoc on Line 22 to find where our “good” matches are. That’s really all there is to 
template matching! 
"""
template = cv2.imread('../../SRC/images/waldo.jpg')
imshow('Template Waldo', template)
# Load input image and convert to grayscale
image = cv2.imread('../../SRC/images/WaldoBeach.jpg')
imshow('Where is Waldo', image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load Template image
template = cv2.imread('../../SRC/images/waldo.jpg', 0)

result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Create Bounding Box
top_left = max_loc
bottom_right = (top_left[0] + 50, top_left[1] + 50)
cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 5)

imshow('Where is Waldo FOUND', image)
