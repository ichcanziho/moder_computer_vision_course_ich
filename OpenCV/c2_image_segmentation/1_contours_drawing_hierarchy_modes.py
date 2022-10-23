import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage.filters import threshold_local

"""
In this lesson we'll learn to:

    1: Using findContours
    2: Drawing Contours
    3: Hierachy of Contours
    4: Contouring Modes (Simple vs Approx)

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
        plt.savefig(f"../outputs/c2_image_segmentation/1_contours_drawing_hierarchy_modes/{title}.png")
    plt.show()


# Let's load a simple image license plate image
image = cv2.imread('../../SRC/images/LP.jpg')
imshow('Input Image', image)

# ----------------------------------------
#       1: Using findContours
# ----------------------------------------

"""
cv2.findContours(image, Retrieval Mode, Approximation Method)

Retrieval Modes

    - RETR_LIST - Retrieves all the contours, but doesn't create any parent-child relationship. Parents and kids are 
      equal under this rule, and they are just contours. ie they all belongs to same hierarchy level.
    - RETR_EXTERNAL - eturns only extreme outer flags. All child contours are left behind.
    - RETR_CCOMP - This flag retrieves all the contours and arranges them to a 2-level hierarchy. ie external contours 
      of the object (ie its boundary) are placed in hierarchy-1. And the contours of holes inside object (if any) is 
      placed in hierarchy-2. If any object inside it, its contour is placed again in hierarchy-1 only. And its hole in 
      hierarchy-2 and so on.
    - RETR_TREE - It retrieves all the contours and creates a full family hierarchy list.

Approximation Method Options

    - cv2.CHAIN_APPROX_NONE – Stores all the points along the line(inefficient!)
    - cv2.CHAIN_APPROX_SIMPLE – Stores the end points of each line
    
"""

# Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow('After thresholding', th2)

V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]
T = threshold_local(V, 25, offset=15, method="gaussian")
thresh_skimage = (V > T).astype("uint8") * 255
imshow("threshold_local", thresh_skimage)
# Finding Contours
# Use a copy of your image e.g. edged.copy(), since findContours alters the image
contours, hierarchy = cv2.findContours(th2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
contours_sk, hierarchy_sk = cv2.findContours(thresh_skimage, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# NOTE: For findContours to work, the background has to be black and foreground (i.e. the text or objects)
# Otherwise you'll need to invert the image by using cv2..bitwise_not(input_image)

# ----------------------------------------
#       2: Drawing Contours
# ----------------------------------------

# Draw all contours, note this overwrites the input image (inplace operation)
# Use '-1' as the 3rd parameter to draw all

cv2.drawContours(image, contours, -1, (0, 255, 0), thickness=2)
imshow('Contours overlaid on original image', image)

cv2.drawContours(image, contours_sk, -1, (0, 255, 0), thickness=2)
imshow('Contours overlaid on original image skimage', image)

print("Number of Contours found = " + str(len(contours)))

# ----------------------------------------
#       3: Hierarchy of Contours
# ----------------------------------------

"""
Official Doc - https://docs.opencv.org/master/d9/d8b/tutorial_py_contours_hierarchy.html

Hierachry

This array stores 4 values for each contour:

First term is the index of the Next contour
Second term is the index of the Previous contour
Third term is the index of the parent contour
Forth term is the index of the child contour
"""

# 3.1: RETR_LIST
# Retrieves all the contours, but doesn't create any parent-child relationship. Parents and kids are equal under this
# rule, and they are just contours. ie they all belongs to same hierarchy level.
image = cv2.imread('../../SRC/images/LP.jpg')

# Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow('After thresholding', th2)

# Use a copy of your image e.g. edged.copy(), since findContours alters the image
contours, hierarchy = cv2.findContours(th2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

# Draw all contours, note this overwrites the input image (inplace operation)
# Use '-1' as the 3rd parameter to draw all
cv2.drawContours(image, contours, -1, (0, 255, 0), thickness=2)
imshow('Contours overlaid on original image', image)

print("Number of Contours found = " + str(len(contours)))
print(hierarchy)

# 3.2: RETR_EXTERNAL
# Returns only extreme outer flags. All child contours are left behind.

image = cv2.imread('../../SRC/images/LP.jpg')

# Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow('After thresholding', th2)

# Use a copy of your image e.g. edged.copy(), since findContours alters the image
contours, hierarchy = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Draw all contours, note this overwrites the input image (inplace operation)
# Use '-1' as the 3rd parameter to draw all
cv2.drawContours(image, contours, -1, (0, 255, 0), thickness=2)
imshow('Contours overlaid on original image', image, size=10)

print("Number of Contours found = " + str(len(contours)))
print(hierarchy)

# 3.3: RETR_CCOMP

# Retrieves all the contours and arranges them to a 2-level hierarchy. ie external contours of the object (ie its
# boundary) are placed in hierarchy-1. And the contours of holes inside object (if any) is placed in hierarchy-2. If
# any object inside it, its contour is placed again in hierarchy-1 only. And its hole in hierarchy-2 and so on.

image = cv2.imread('../../SRC/images/LP.jpg')

# Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow('After thresholding', th2)

# Use a copy of your image e.g. edged.copy(), since findContours alters the image
contours, hierarchy = cv2.findContours(th2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

# Draw all contours, note this overwrites the input image (inplace operation)
# Use '-1' as the 3rd parameter to draw all
cv2.drawContours(image, contours, -1, (0, 255, 0), thickness=2)
imshow('Contours overlaid on original image', image)

print("Number of Contours found = " + str(len(contours)))
print(hierarchy)

# 3.4: RETR_TREE
# It retrieves all the contours and creates a full family hierarchy list.

image = cv2.imread('../../SRC/images/LP.jpg')

# Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow('After thresholding', th2)

# Use a copy of your image e.g. edged.copy(), since findContours alters the image
contours, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# Draw all contours, note this overwrites the input image (inplace operation)
# Use '-1' as the 3rd parameter to draw all
cv2.drawContours(image, contours, -1, (0,255,0), thickness = 2)
imshow('Contours overlaid on original image', image)

print("Number of Contours found = " + str(len(contours)))
print(hierarchy)

# ----------------------------------------
#       4: Contouring Modes (Simple vs Approx)
# ----------------------------------------

# CHAIN_APPROX_NONE
image = cv2.imread('../../SRC/images/LP.jpg')

# Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow('After thresholding', th2)

# Use a copy of your image e.g. edged.copy(), since findContours alters the image
contours, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

# Draw all contours, note this overwrites the input image (inplace operation)
# Use '-1' as the 3rd parameter to draw all
cv2.drawContours(image, contours, -1, (0, 255, 0), thickness=2)
imshow('Contours overlaid on original image', image)

print("Number of Contours found = " + str(len(contours)))
for c in contours:
    print(len(c))

# CHAIN_APPROX_SIMPLE

image = cv2.imread('../../SRC/images/LP.jpg')

# Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow('After thresholding', th2)

# Use a copy of your image e.g. edged.copy(), since findContours alters the image
contours, hierarchy = cv2.findContours(th2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw all contours, note this overwrites the input image (inplace operation)
# Use '-1' as the 3rd parameter to draw all
cv2.drawContours(image, contours, -1, (0, 255, 0), thickness=2)
imshow('Contours overlaid on original image', image)

print("Number of Contours found = " + str(len(contours)))
for c in contours:
    print(len(c))
