import cv2
import dlib
from matplotlib import pyplot as plt
import os
import pathlib
import numpy as np

"""
In this lesson we'll learn to:

    1:  Apply Facial Landmark Detection

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


# ---------------------------------------------------------------
#       1: Apply Facial Landmark Detection
# ---------------------------------------------------------------

class TooManyFaces(Exception):
    pass


class NoFaces(Exception):
    pass


def get_landmarks(im, detector, predictor,):
    rects = detector(im, 1)

    if len(rects) > 1:
        raise TooManyFaces
    if len(rects) == 0:
        raise NoFaces

    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,

                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im


PREDICTOR_PATH = "../../SRC/dlib_predictors/shape_predictor_68_face_landmarks.dat"
pred = dlib.shape_predictor(PREDICTOR_PATH)
detec = dlib.get_frontal_face_detector()


image = cv2.imread('../../SRC/images/Trump.jpg')
imshow('Original Trump', image)
landm = get_landmarks(image, detec, pred)
image_with_landmarks = annotate_landmarks(image, landm)
imshow('Result Trump', image_with_landmarks)

image = cv2.imread('../../SRC/images/Hillary.jpg')
imshow('Original Hillary', image)
landm = get_landmarks(image, detec, pred)
image_with_landmarks = annotate_landmarks(image, landm)
imshow('Result Hillary', image_with_landmarks)
