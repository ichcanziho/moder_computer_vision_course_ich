import cv2
from matplotlib import pyplot as plt
import os
import pathlib
import numpy as np
from skimage.metrics import structural_similarity

"""
In this lesson we'll learn to:

    1: Compare Images using Mean Squared Error (MSE)
    2: Compare Images using Structural Similarity

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
#       1: Compare Images using Mean Squared Error (MSE)
#       2: Compare Images using Structural Similarity
# ----------------------------------------

# The MSE between the two images is the sum of the squared difference between the two images. This can easily be
# implemented with numpy. The lower the MSE the more similar the images are.

def mse(image1, image2):
    # Images must be of the same dimension
    error = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    error /= float(image1.shape[0] * image1.shape[1])

    return error


def compare(image1, image2):
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    print('MSE = {:.2f}'.format(mse(image1, image2)))
    print('SS = {:.2f}'.format(structural_similarity(image1, image2)))


fireworks1 = cv2.imread('../../SRC/images/fireworks.jpeg')
fireworks2 = cv2.imread('../../SRC/images/fireworks2.jpeg')

M = np.ones(fireworks1.shape, dtype="uint8") * 100
fireworks1b = cv2.add(fireworks1, M)

imshow("fireworks 1", fireworks1)
imshow("Increasing Brightness", fireworks1b)
imshow("fireworks 2", fireworks2)

print("Fireworks 1 vs Fireworks 1")
compare(fireworks1, fireworks1)
print("Fireworks 1 vs Fireworks 2")
compare(fireworks1, fireworks2)
print("Fireworks 1 vs Fireworks 1 brighter")
compare(fireworks1, fireworks1b)
print("Fireworks 2 vs Fireworks 1 brighter")
compare(fireworks2, fireworks1b)
