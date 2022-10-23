import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
In this lesson we'll learn to:

    1: Drawing Lines
    2: Drawing Rectangles
    3: Drawing Circles
    4: Drawing Polygons
    5: Writing text

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
        plt.savefig(f"../outputs/4_drawing_on_images/{title}.png")
    plt.show()


# Create a black image using numpy to create and array of black
image = np.zeros((512, 512, 3), np.uint8)

# Can we make this in black and white? grayscale
image_gray = np.zeros((512, 512), np.uint8)

# Black would be the same as a greyscale or color image (same for white)
imshow("Black Canvas - RGB Color", image, save=False)
imshow("Black Canvas - Grayscale", image_gray, save=False)

# ----------------------------------------
# Let's draw a line over our black square
# ----------------------------------------

# Note this is an inplace operation, meaning it changes the input image
# Unlike many other OpenCV functions that return a new image leaving the input unaffected
# Remember our image was the black canvas
# cv2.line(image, starting coordinates, ending coordinates, color, thickness)
cv2.line(image, (0, 0), (511, 511), (255, 127, 0), 5)
imshow("Line Example", image)

# ----------------------------------------
#            Drawing Rectangles
# ----------------------------------------

# Create our black canvas again because now it has a line in it
image = np.zeros((512, 512, 3), np.uint8)
# cv2.rectangle(image, starting vertex, opposite vertex, color, thickness)
# Thickness - if positive. Negative thickness means that it is filled
cv2.rectangle(image, (100, 100), (300, 250), (127, 50, 127), 10)
imshow("Rectangle Example", image)

# ----------------------------------------
#            Drawing Circles
# ----------------------------------------

image = np.zeros((512, 512, 3), np.uint8)
# cv2.circle(image, center, radius, color, fill)
cv2.circle(image, (350, 350), 100, (15, 150, 50), -1)
imshow("Circle Example", image)

# ----------------------------------------
#            Drawing Polygons
# ----------------------------------------

image = np.zeros((512, 512, 3), np.uint8)

# Let's define four points
pts = np.array([[10, 50], [400, 50], [90, 200], [50, 500]], np.int32)

# Let's now reshape our points in form  required by polylines
pts = pts.reshape((-1, 1, 2))
# cv2.polylines(image, points, Closed?, color, thickness)
cv2.polylines(image, [pts], True, (0, 0, 255), 3)
imshow("Polygon Example", image)

pts = np.array([[10, 50], [400, 50], [90, 200], [50, 500]], np.int32)
print("polygon points:", pts)
print("original points shape:", pts.shape)
pts = pts.reshape((-1, 1, 2))
print("points reshape:", pts.shape)

# ----------------------------------------
#            Writing Text
# ----------------------------------------

"""

Available Fonts

    FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN
    FONT_HERSHEY_DUPLEX,FONT_HERSHEY_COMPLEX
    FONT_HERSHEY_TRIPLEX, FONT_HERSHEY_COMPLEX_SMALL
    FONT_HERSHEY_SCRIPT_SIMPLEX
    FONT_HERSHEY_SCRIPT_COMPLEX

"""

image = np.zeros((1000, 1000, 3), np.uint8)
ourString = 'Ichcanziho'
# cv2.putText(image, 'Text to Display', bottom left starting point, Font, Font Size, Color, Thickness)
cv2.putText(image, ourString, (155, 290), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3, (40, 200, 0), 4)
imshow("Text Example", image)
