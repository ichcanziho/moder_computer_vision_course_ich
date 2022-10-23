import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
In this lesson we'll learn to:

    1: View the individual channels of an RGB Image
    2: Manipulate a color space
    3: Introduce HSV Color Spaces

"""


# Define our imshow function
def imshow(title="Image", image=None, size=2, save=True):
    # we have added the size parameter to control de fig-size of the image while plotting with matplotlib, for that
    # we obtain the aspect ratio of the image dividing the width by the height.
    h, w = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(title)
    if save:
        plt.savefig(f"../outputs/3_color_spaces/{title}.png")
    plt.show()


# Load our input image
img = cv2.imread("../../SRC/images/castara.jpeg")
print("Dimensions of original input image")
print("Original", img.shape)
# Use cv2.split to get each color space separately
B, G, R = cv2.split(img)
print("Dimensions of the BGR channels")
print("B", B.shape)
print("G", G.shape)
print("R", R.shape)

# Each color space on it's on will look like a grayscale as it lacks the other color channels
# imshow("Blue Channel Only", B)

# Let's create a matrix of zeros
# with dimensions of the image h x w
zeros = np.zeros(img.shape[:2], dtype="uint8")
print("black image", zeros.shape)
imshow("zeros", zeros)
# cv2.merge serves for joint layers by creating a new dimension
imshow("Red", cv2.merge([zeros, zeros, R]))
imshow("Green", cv2.merge([zeros, G, zeros]))
imshow("Blue", cv2.merge([B, zeros, zeros]))

# OpenCV's 'split' function splits the image into each color index
B, G, R = cv2.split(img)

# Let's re-make the original image,
merged = cv2.merge([B, G, R])
imshow("Merged", merged)

# Let's amplify the blue color
merged = cv2.merge([B+100, G, R])
imshow("Blue Boost", merged)

# Convert to HSV
hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
imshow('HSV', hsv_image)

# Our imshow function was not prepare to plot HSV images, so we need to convert from HSV to RGB to use our imshow
# function
imshow("HSV to BGR", cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR))

# Switching back to viewing the RGB representation
imshow("Hue", hsv_image[:, :, 0])
imshow("Saturation", hsv_image[:, :, 1])
imshow("Value", hsv_image[:, :, 2])
