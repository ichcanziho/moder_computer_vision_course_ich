import cv2
from matplotlib import pyplot as plt
import os
import pathlib
import numpy as np

"""
In this lesson we'll learn to:

    1: How to use the Watershed algorithm for marker-based image segmentation

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
#       1: How to use the Watershed algorithm for marker-based image segmentation
# ----------------------------------------

"""Any grayscale image can be viewed as a topographic surface where high intensity denotes peaks and hills while low 
intensity denotes valleys. This algorithm uses that analogy and starts filling those low points (valleys) with a 
different colored label ( aka our water). As the water rises, depending on the peaks (gradients) nearby, water from 
different valleys, obviously with different colors will start to merge. To avoid that, you build barriers in the 
locations where water merges. You continue the work of filling water and building barriers until all the peaks are 
under water. The barriers you created gives you the segmentation result. This is the "philosophy" behind the 
watershed. You can visit the CMM webpage on watershed to understand it with the help of some animations. Their 
approach however, gives you over segmented result due to noise or any other irregularities in the image. 

Thus, OpenCV implemented a marker-based watershed algorithm where you specify which are all valley points are to be 
merged and which are not. It gives different labels for our object we know. Label the region which we are sure of 
being the foreground or object with one color (or intensity), label the region which we are sure of being background 
or non-object with another color and finally the region which we are not sure of anything, label it with 0. That is 
our marker. Then apply watershed algorithm. Then our marker will be updated with the labels we gave, 
and the boundaries of objects will have a value of -1. 
"""

# Load image
img = cv2.imread('../../SRC/images/water_coins.jpg')
imshow("Original image", img)

# Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold using OTSU
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

imshow("Thresholded", thresh)

# Removing the touching masks

# noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# sure background area
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Finding sure foreground area
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

imshow("SureFG", sure_fg)
imshow("SureBG", sure_bg)
imshow("unknown", unknown)

# Marker labelling
# Connected Components determines the connectivity of blob-like regions in a binary image.
ret, markers = cv2.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Now, mark the region of unknown with zero
markers[unknown == 255] = 0
print(markers)
markers = cv2.watershed(img, markers)
img[markers == -1] = [255, 0, 0]

imshow("img", img)
