import cv2
from matplotlib import pyplot as plt
import numpy as np

"""
In this lesson we'll learn to:

    1: Dilation
    2: Erosion
    3: Opening
    4: Closing
    5: Canny Edge Detection
    6: Auto Canny Edge Detection

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
        plt.savefig(f"../outputs/c1_image_operations/10_dilation_erosion_edge_detection/{title}.png")
    plt.show()


"""
Dilation – Adds pixels to the boundaries of objects in an image
Erosion – Removes pixels at the boundaries of objects in an image
Opening - Erosion followed by dilation
Closing - Dilation followed by erosion
"""

image = cv2.imread('../../SRC/images/opencv_inv.png', 0)
imshow('Original', image)

# Let's define our kernel size
kernel = np.ones((5, 5), np.uint8)

# ----------------------------------------
#       1: Dilation
# ----------------------------------------

# Dilate here
dilation = cv2.dilate(image, kernel, iterations=1)
imshow('Dilation', dilation)

# ----------------------------------------
#       2: Erosion
# ----------------------------------------
# Now we erode
erosion = cv2.erode(image, kernel, iterations=1)
imshow('Erosion', erosion)
# ----------------------------------------
#       3: Opening
# ----------------------------------------
# Opening - Good for removing noise
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
imshow('Opening', opening)
# ----------------------------------------
#       4: Closing
# ----------------------------------------
# Closing - Good for removing noise
closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
imshow('Closing', closing)
# ----------------------------------------
#       5: Canny Edge Detection
# ----------------------------------------

"""
The first argument is our input image.
The second and third arguments are our minVal and maxVal respectively.
The forth argument is aperture_size. It is the size of Sobel kernel used for find image gradients. By default it is 3.
Edge detection needs a threshold to tell what difference/change should be counted as edge
"""
image = cv2.imread('../../SRC/images/londonxmas.jpeg', 0)

# Canny Edge Detection uses gradient values as thresholds
# The first threshold gradient
canny = cv2.Canny(image, 50, 120)
imshow('Canny 1', canny)

# Wide edge thresholds expect lots of edges
canny = cv2.Canny(image, 10, 200)
imshow('Canny Wide', canny)

# Narrow threshold, expect less edges
canny = cv2.Canny(image, 200, 240)
imshow('Canny Narrow', canny)

canny = cv2.Canny(image, 60, 110)
imshow('Canny 4', canny)


# Then, we need to provide two values: threshold1 and threshold2. Any gradient value larger than threshold2
# is considered to be an edge. Any value below threshold1 is considered not to be an edge.
# Values in between threshold1 and threshold2 are either classiﬁed as edges or non-edges based on how their
# intensities are “connected”. In this case, any gradient values below 60 are considered non-edges
# whereas any values above 120 are considered edges.

# ----------------------------------------
#       6: Auto Canny Edge Detection
# ----------------------------------------

def auto_canny(image):
    # Finds optimal thresholds based on median image pixel intensity
    blurred_img = cv2.blur(image, ksize=(5, 5))
    med_val = np.median(image)
    lower = int(max(0, 0.66 * med_val))
    upper = int(min(255, 1.33 * med_val))
    edges = cv2.Canny(image=blurred_img, threshold1=lower, threshold2=upper)
    return edges


auto_canny = auto_canny(image)
imshow("auto canny", auto_canny)
