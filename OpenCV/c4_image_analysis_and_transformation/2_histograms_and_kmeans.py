import cv2
from matplotlib import pyplot as plt
import os
import pathlib
import numpy as np
from sklearn.cluster import KMeans

"""
In this lesson we'll learn to:

    1: View the RGB Histogram representations of images
    2: Use K-Means Clustering to get the dominant colors and their proportions in images

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
#       1: View the RGB Histogram representations of images
# ----------------------------------------

image = cv2.imread('../../SRC/images/input.jpg')
imshow("Input 1", image)

# histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

# We plot a histogram, ravel() flattens our image array
plt.hist(image.ravel(), 256, [0, 256])
plt.show()

# Viewing Separate Color Channels
color = ('b', 'g', 'r')

# We now separate the colors and plot each in the Histogram
for i, col in enumerate(color):
    histogram2 = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(histogram2, color=col)
    plt.xlim([0, 256])

plt.show()

"""
cv2.calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])

    - images : it is the source image of type uint8 or float32. it should be given in square brackets, ie, "[img]".
    - channels : it is also given in square brackets. It is the index of channel for which we calculate histogram. 
      For example, if input is grayscale image, its value is [0]. For color image, you can pass [0], [1] or [2] to 
      calculate histogram of blue, green or red channel respectively.
    - mask : mask image. To find histogram of full image, it is given as "None". But if you want to find histogram of 
      particular region of image, you have to create a mask image for that and give it as mask. 
      (I will show an example later.)
    - histSize : this represents our BIN count. Need to be given in square brackets. For full scale, we pass [256].
      ranges : this is our RANGE. Normally, it is [0,256].

"""

image = cv2.imread('../../SRC/images/tobago.jpg')
imshow("Input", image)

histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

# We plot a histogram, ravel() flatens our image array
plt.hist(image.ravel(), 256, [0, 256])
plt.show()

# Viewing Separate Color Channels
color = ('b', 'g', 'r')

# We now separate the colors and plot each in the Histogram
for i, col in enumerate(color):
    histogram2 = cv2.calcHist([image], [i], None, [256], [0, 256])
    plt.plot(histogram2, color=col)
    plt.xlim([0, 256])

plt.show()


# ----------------------------------------
#       2: Use K-Means Clustering to get the dominant colors and their proportions in images
# ----------------------------------------

def centroid_histogram(clt):
    # Create a histogram for the clusters based on the pixels in each cluster
    # Get the labels for each cluster
    num_labels = np.arange(0, len(np.unique(clt.labels_)) + 1)

    # Create our histogram
    (hist, _) = np.histogram(clt.labels_, bins=num_labels)

    # normalize the histogram, so that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    return hist


def plot_colors(hist, centroids):
    # Create our blank barchart
    bar = np.zeros((100, 500, 3), dtype="uint8")

    x_start = 0
    # iterate over the percentage and dominant color of each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        end = x_start + (percent * 500)
        cv2.rectangle(bar, (int(x_start), 0), (int(end), 100),
                      color.astype("uint8").tolist(), -1)
        x_start = end
    return bar


image = cv2.imread('../../SRC/images/Volleyball.jpeg')
imshow("Input", image)

# We reshape our image into a list of RGB pixels
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.reshape((image.shape[0] * image.shape[1], 3))

number_of_clusters = 3
clt = KMeans(number_of_clusters)
clt.fit(image)

hist = centroid_histogram(clt)
bar = plot_colors(hist, clt.cluster_centers_)

# show our color bart
plt.figure()
plt.axis("off")
plt.imshow(bar)
plt.show()
