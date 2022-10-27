import cv2
from matplotlib import pyplot as plt
import os
import numpy as np

"""
In this lesson we'll learn to:

    1: To use a Haar cascade Classifier to detect faces
    2: To use a Haar cascade Classifier to detect eyes
    3: To use a Haar cascade Classifier to detect faces and eyes from your webcam

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
        f = os.path.basename(__file__)
        f = f.split(".")[0]
        plt.savefig(f"../outputs/c3_haar_cascade_classifiers/{f}/{title}.png")
    plt.show()


# Object Detection is the ability to detect and classify individual objects in an image and draw a bounding box over
# the object's area.


"""
Developed by Viola and Jones in 2001.

An object detection method that uses a series of classifiers (cascade) to identify objects in an image. They are 
trained to identify one type of object, however, we can use several of them in parallel e.g. detecting eyes and faces 
together. HAAR Classifiers are trained using lots of positive images (i.e. images with the object present) and 
negative images (i.e. images without the object present). 

"""

# ----------------------------------------
#       1: To use a Haar cascade Classifier to detect faces
# ----------------------------------------

# We point OpenCV's CascadeClassifier function to where our
# classifier (XML file format) is stored
face_classifier = cv2.CascadeClassifier('../../SRC/Haarcascades/haarcascade_frontalface_default.xml')

# Load our image then convert it to grayscale
image = cv2.imread('../../SRC/images/Trump.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Our classifier returns the ROI of the detected face as a tuple
# It stores the top left coordinate and the bottom right coordinates
faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# When no faces detected, face_classifier returns and empty tuple
if faces is None:
    print("No faces found")

# We iterate through our faces array and draw a rectangle
# over each face in faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (127, 0, 255), 2)

imshow('Face Detection', image)

# ----------------------------------------
#       2: To use a Haar cascade Classifier to detect eyes
# ----------------------------------------

face_classifier = cv2.CascadeClassifier('../../SRC/Haarcascades/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('../../SRC/Haarcascades/haarcascade_eye.xml')

img = cv2.imread('../../SRC/images/Trump.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_classifier.detectMultiScale(gray, 1.3, 5)

# When no faces detected, face_classifier returns and empty tuple
if faces is None:
    print("No Face Found")

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (127, 0, 255), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    eyes = eye_classifier.detectMultiScale(roi_gray, 1.2, 3)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 255, 0), 2)

imshow('Eye & Face Detection', img)


face_classifier = cv2.CascadeClassifier('../../SRC/Haarcascades/haarcascade_frontalface_default.xml')
eye_classifier = cv2.CascadeClassifier('../../SRC/Haarcascades/haarcascade_eye.xml')


def face_detector(img, size=0.5):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img
    roi_color = None
    for (x, y, w, h) in faces:
        x = x - 50
        w = w + 50
        y = y - 50
        h = h + 50
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_classifier.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)

    roi_color = cv2.flip(roi_color, 1)
    return roi_color


cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    cv2.imshow('Our Face Extractor', face_detector(frame))
    if cv2.waitKey(1) == ord("q"):  # 13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()
