import cv2
from matplotlib import pyplot as plt
from skimage.filters import threshold_local

"""
In this lesson we'll learn to:

    1: Binarized Images
    2: Thresholding Methods
    3: Adaptive Thresholding
    4: SkImage's Threshold Local

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
        plt.savefig(f"../outputs/c1_image_operations/9_thresholding_binarization_adaptive_thresholding/{title}.png")
    plt.show()


# ----------------------------------------
#       1: Binarized Images
# ----------------------------------------

# Load our image as greyscale
image = cv2.imread('../../SRC/images/scan_book.jpeg', 0)
imshow("Original", image)

# Values below 127 goes to 0 or black, everything above goes to 255 (white)
ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
imshow('1 Threshold Binary @ 127', thresh1)

# ----------------------------------------
#       2: Thresholding Methods
# ----------------------------------------

# Values below 127 go to 255 and values above 127 go to 0 (reverse of above)
ret, thresh2 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
imshow('2 Threshold Binary Inverse @ 127', thresh2)

# Values above 127 are truncated (held) at 127 (the 255 argument is unused)
ret, thresh3 = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
imshow('3 THRESH TRUNC @ 127', thresh3)

# Values below 127 go to 0, above 127 are unchanged
ret, thresh4 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
imshow('4 THRESH TOZERO @ 127', thresh4)

# Reverse of the above, below 127 is unchanged, above 127 goes to 0
ret, thresh5 = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)
imshow('5 THRESH TOZERO INV @ 127', thresh5)

# ----------------------------------------
#       3: Adaptive Thresholding
# ----------------------------------------

"""

- ADAPTIVE_THRESH_MEAN_C
- THRESH_OTSU

cv2.adaptiveThreshold Parameters
cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst]) → dst

    - src – Source 8-bit single-channel image.
    - dst – Destination image of the same size and the same type as src .
    - maxValue – Non-zero value assigned to the pixels for which the condition is satisfied. See the details below.
    - adaptiveMethod – Adaptive thresholding algorithm to use, ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C . 
        See the details below.
    - thresholdType – Thresholding type that must be either THRESH_BINARY or THRESH_BINARY_INV .
    - blockSize – Size of a pixel neighborhood that is used to calculate a threshold value for the pixel: 3, 5, 7, and 
      so on.
    - C – Constant subtracted from the mean or weighted mean. Normally, it is positive but may be zero or negative as 
      well.

"""

image = cv2.imread('../../SRC/images/scan_book.jpeg', 0)
imshow("Original", image)

# Values below 127 goes to 0 (black, everything above goes to 255 (white)
ret, thresh1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
imshow('Threshold Binary', thresh1)

# It's good practice to blur images as it removes noise
# image = cv2.GaussianBlur(image, (3, 3), 0)

# Using adaptiveThreshold
thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)
imshow("Adaptive Mean Thresholding", thresh)

_, th2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow("Otsu's Thresholding", th2)

# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(image, (5, 5), 0)
_, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
imshow("Guassian Otsu's Thresholding", th3)

# ----------------------------------------
#       4: SkImage's Threshold Local
# ----------------------------------------

"""
threshold_local(image, block_size, offset=10)

The threshold_local function, calculates thresholds in regions with a characteristic size block_size surrounding each 
pixel (i.e. local neighborhoods). Each threshold value is the weighted mean of the local neighborhood minus an offset 
value

https://scikit-image.org/docs/stable/auto_examples/applications/plot_thresholding.html
"""

image = cv2.imread('../../SRC/images/scan_book.jpeg')

# We get the Value component from the HSV color space
# then we apply adaptive thresholding to
V = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))[2]
T = threshold_local(V, 25, offset=15, method="gaussian")

# Apply the threshold operation
thresh = (V > T).astype("uint8") * 255
imshow("threshold_local", thresh)
