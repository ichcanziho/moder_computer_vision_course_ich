import cv2
import numpy as np
from matplotlib import pyplot as plt

"""
In this lesson we'll learn to:

    1: Convolution Operations
    2: Blurring
    3: De-noising
    4: Sharpening

"""


# Define our imshow function
def imshow(title="Image", img=None, size=5, save=True):
    # we have added the size parameter to control de fig-size of the image while plotting with matplotlib, for that
    # we obtain the aspect ratio of the image dividing the width by the height.
    h, w = img.shape[0], img.shape[1]
    aspect_ratio = w / h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(title)
    if save:
        plt.savefig(f"../outputs/8_convolutions_blurring_sharpening/{title}.png")
    plt.show()


# ----------------------------------------
#       1: Convolution Operations
# ----------------------------------------

# Blurring using Convolutions

image = cv2.imread('../../SRC/images/flowers.jpeg')
imshow('Original Image', image)

# Creating our 3 x 3 kernel
kernel_3x3 = np.ones((3, 3), np.float32) / 9

# We use the cv2.fitler2D to conovlve the kernal with an image
blurred = cv2.filter2D(image, -1, kernel_3x3)
imshow('3x3 Kernel Blurring', blurred)

# Creating our 7 x 7 kernel
kernel_7x7 = np.ones((7, 7), np.float32) / 49

blurred2 = cv2.filter2D(image, -1, kernel_7x7)
imshow('7x7 Kernel Blurring', blurred2)

# ----------------------------------------
#       2: Blurring
# ----------------------------------------

# Other commonly used blurring methods in OpenCV
# Regular Blurring
# Gaussian Blurring
# Median Blurring

image = cv2.imread('../../SRC/images/flowers.jpeg')

# Averaging done by convolving the image with a normalized box filter.
# This takes the pixels under the box and replaces the central element
# Box size needs to odd and positive
blur = cv2.blur(image, (5, 5))
imshow('Averaging', blur)

# Instead of box filter, gaussian kernel
Gaussian = cv2.GaussianBlur(image, (5, 5), 0)
imshow('Gaussian Blurring', Gaussian)

# Takes median of all the pixels under kernel area and central
# element is replaced with this median value
median = cv2.medianBlur(image, 5)
imshow('Median Blurring', median)

"""

Bilateral Filter
    - dst = cv.bilateralFilter(src, d, sigmaColor, sigmaSpace[, dst[, borderType]])
    
    - src = Source 8-bit or floating-point, 1-channel or 3-channel image.
    
    - dst = Destination image of the same size and type as src .
    
    - d = Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is computed 
    from sigmaSpace. 
    
    - sigmaColor = Filter sigma in the color space. A larger value of the parameter means that farther colors within 
    the pixel neighborhood (see sigmaSpace) will be mixed together, resulting in larger areas of semi-equal color. 
    
    - sigmaSpace = Filter sigma in the coordinate space. A larger value of the parameter means that farther pixels 
    will influence each other as long as their colors are close enough (see sigmaColor ). When d>0, it specifies the 
    neighborhood size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace. 
    
    - borderType = border mode used to extrapolate pixels outside of the image

"""

# Bilateral is very effective in noise removal while keeping edges sharp
bilateral = cv2.bilateralFilter(image, 9, 75, 75)
imshow('Bilateral Blurring', bilateral)

# ----------------------------------------
#       3: De-noising
# ----------------------------------------

"""

There are 4 variations of Non-Local Means Denoising:

    - cv2.fastNlMeansDenoising() - works with a single grayscale images
    - cv2.fastNlMeansDenoisingColored() - works with a color image.
    - cv2.fastNlMeansDenoisingMulti() - works with image sequence captured in short period of time (grayscale images)
    -cv2.fastNlMeansDenoisingColoredMulti() - same as above, but for color images.

fastNlMeansDenoisingColored(InputArray src, OutputArray dst, float h=3, float hColor=3, int templateWindowSize=7, 
int searchWindowSize=21)

Parameters for fastNlMeansDenoisingColored:
    - src – Input 8-bit 3-channel image.
    - dst – Output image with the same size and type as src . templateWindowSize – Size in pixels of the template patch 
    that is used to compute weights. Should be odd. Recommended value 7 pixels
    - searchWindowSize – Size in pixels of the window that is used to compute weighted average for given pixel. Should 
    be odd. Affect performance linearly: greater searchWindowsSize - greater denoising time. Recommended value 21 pixels
    - h – Parameter regulating filter strength for luminance component. Bigger h value perfectly removes noise but also 
    removes image details, smaller h value preserves details but also preserves some noise
    - hColor – The same as h but for color components. For most images value equals 10 will be enought to remove colored
    noise and do not distort colors

"""

image = cv2.imread('../../SRC/images/hilton.jpeg')
imshow('Original', image)

dst = cv2.fastNlMeansDenoisingColored(image, None, 6, 6, 7, 21)
imshow('fastNlMeansDenoisingColored', dst)

# ----------------------------------------
#       4: Sharpening
# ----------------------------------------

# Loading our image
image = cv2.imread('../../SRC/images/hilton.jpeg')
imshow('Original', image)

# Create our shapening kernel, remember it must sum to one
kernel_sharpening = np.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])

# applying the sharpening kernel to the image
sharpened = cv2.filter2D(image, -1, kernel_sharpening)
imshow('Sharpened Image', sharpened)
