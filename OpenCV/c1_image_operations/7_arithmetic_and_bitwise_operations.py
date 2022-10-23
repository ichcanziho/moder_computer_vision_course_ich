import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
In this lesson we'll learn to:

    1: Arithmetic Operations
    2: Bitwise Operations

"""


# Define our imshow function
def imshow(title="Image", img=None, size=4, save=True):
    # we have added the size parameter to control de fig-size of the image while plotting with matplotlib, for that
    # we obtain the aspect ratio of the image dividing the width by the height.
    h, w = img.shape[0], img.shape[1]
    aspect_ratio = w / h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(title)
    if save:
        plt.savefig(f"../outputs/7_arithmetic_and_bitwise_operations/{title}.png")
    plt.show()


# ----------------------------------------
#       1: Arithmetic Operations
# ----------------------------------------

# Adding comma zero in cv2.imread loads our image in as a grayscale image
image = cv2.imread('../../SRC/images/liberty.jpeg', 0)
imshow("Grayscaled Image", image)
print(image)

# Create a matrix of ones, then multiply it by a scaler of 100
# This gives a matrix with same dimensions of our image with all values being 100
M = np.ones(image.shape, dtype="uint8") * 100
print(M)

# Increasing Brightness

# We use this to add this matrix M, to our image
# Notice the increase in brightness
added = cv2.add(image, M)
imshow("Increasing Brightness", added)

# Now if we just added it, look what happens
added2 = image + M
# if we only add the images using the "+" operator the image will tend to overflow and then the results will be
# different. If our original image has a pixel with value 200, and we want to add 100, then the result is 300, but
# the desire output must be clipped to 255 because this is the maximum value for a pixel, besides this limitation when
# our summation is 300 the final value will be 300 - 255 = 45. So our value will be darker than the original value.
imshow("Simple Numpy Adding Results in Clipping", added2)

# Decreasing Brightness

# Likewise, we can also subtract
# Notice the decrease in brightness
subtracted = cv2.subtract(image, M)
imshow("Subtracted", subtracted)

subtracted = image - M
imshow("Subtracted 2", subtracted)

# ----------------------------------------
#         2: Bitwise Operations
# ----------------------------------------

# To demonstrate these operations let's create some simple images

# If you're wondering why only two dimensions, well this is a grayscale image,

# Making a square
square = np.zeros((300, 300), np.uint8)
cv2.rectangle(square, (50, 50), (250, 250), 255, -2)
imshow("square", square)

# Making a ellipse
ellipse = np.zeros((300, 300), np.uint8)
cv2.ellipse(ellipse, (150, 150), (150, 150), 30, 0, 180, 255, -1)
imshow("ellipse", ellipse)

# Shows only where they intersect
And = cv2.bitwise_and(square, ellipse)
imshow("AND", And)

# Shows where either square or ellipse is
bitwiseOr = cv2.bitwise_or(square, ellipse)
imshow("bitwiseOr", bitwiseOr)

# Shows where either exist by itself
bitwiseXor = cv2.bitwise_xor(square, ellipse)
imshow("bitwiseXor", bitwiseXor)

# Shows everything that isn't part of the square
bitwiseNot_sq = cv2.bitwise_not(square)
imshow("bitwiseNot_sq", bitwiseNot_sq)

# Notice the last operation inverts the image totally
