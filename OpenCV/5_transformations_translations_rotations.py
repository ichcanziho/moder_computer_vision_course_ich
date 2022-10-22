import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
In this lesson we'll learn to:

    1: Perform Image Translations
    2: Rotations with getRotationMatrix2D
    3: Rotations with Transpose
    4: Flipping Images

"""


# Define our imshow function
def imshow(title="Image", img=None, size=2, save=True):
    # we have added the size parameter to control de fig-size of the image while plotting with matplotlib, for that
    # we obtain the aspect ratio of the image dividing the width by the height.
    h, w = img.shape[0], img.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(title)
    if save:
        plt.savefig(f"outputs/{title}.png")
    plt.show()

# ----------------------------------------
#       1: Perform Image Translations
# ----------------------------------------

# This an affine transform that simply shifts the position of an image. (left or right).
# We use cv2.warpAffine to implement these transformations.
# cv2.warpAffine(image, T, (width, height))


image = cv2.imread('../SRC/images/Volleyball.jpeg')
imshow("Original", image)

# Store height and width of the image
height, width = image.shape[:2]

# We shift it by quarter of the height and width
quarter_height, quarter_width = height/4, width/4

# Our Translation
#       | 1 0 Tx |
#  T  = | 0 1 Ty |

# T is our translation matrix
T = np.float32([[1, 0, quarter_width], [0, 1, quarter_height]])

# We use warpAffine to transform the image using the matrix, T
img_translation = cv2.warpAffine(image, T, (width, height))
imshow("Translated", img_translation)

# What does T look like
print(T)

print(height, width)

# ----------------------------------------
# 2: Rotations with getRotationMatrix2D
# ----------------------------------------

# cv2.getRotationMatrix2D(rotation_center_x, rotation_center_y, angle of rotation, scale)
# Load our image
image = cv2.imread('../SRC/images/Volleyball.jpeg')
height, width = image.shape[:2]

# Divide by two to rotate the image around its centre
rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 1)

# Input our image, the rotation matrix and our desired final width and height
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
imshow("Rotated 90 degrees with scale = 1", rotated_image)

# some time it's useful to resize the original image and then rotate, this will allow us to avoid cropping the image
# while rotating. Divide by two to rotate the image around its centre
rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 90, 0.5)
print(rotation_matrix)
# Input our image, the rotation matrix and our desired final width and height
rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
imshow("Rotated 90 degrees with scale = 0.5", rotated_image)

# ----------------------------------------
#       3: Rotations with Transpose
# ----------------------------------------

# Transpose is less flexible, because we can only rotate transpose from X to Y
rotated_image = cv2.transpose(image)
imshow("Original", image)
imshow("Rotated using Transpose", rotated_image)

# If we transpose 2 times the image we come back to the original image
rotated_image = cv2.transpose(image)
rotated_image = cv2.transpose(rotated_image)
imshow("Rotated using Transpose", rotated_image)

# ----------------------------------------
#           4: Flipping Images
# ----------------------------------------

# Let's now to a horizontal flip.
flipped = cv2.flip(image, 1)
imshow("Horizontal Flip", flipped)
