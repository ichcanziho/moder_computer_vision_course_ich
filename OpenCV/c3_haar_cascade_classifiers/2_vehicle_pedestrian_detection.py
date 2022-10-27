import cv2
from matplotlib import pyplot as plt
import os
import pathlib
import numpy as np

"""
In this lesson we'll learn to:

    1: Use a Haar cascade classifier to detect Pedestrians
    2: Use our Haar cascade classifiers on videos
    3: Use a Haar cascade classier to detect Vehicles or cars

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
#       1: To use a Haar cascade Classifier to detect pedestrians
# ----------------------------------------

# Create our video capturing object
cap = cv2.VideoCapture('../../SRC/videos/walking.mp4')

# Load our body classifier
body_classifier = cv2.CascadeClassifier('../../SRC/Haarcascades/haarcascade_fullbody.xml')

# Read first frame
ret, frame = cap.read()

# Ret is True if successfully read
if ret:

    # Grayscale our image for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pass frame to our body classifier
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)

    # Extract bounding boxes for any bodies identified
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

# Release our video capture
cap.release()
imshow("Pedestrian Detector", frame)

# ----------------------------------------
#       2: Use our Haar cascade classifiers on videos
# ----------------------------------------

# Create our video capturing object
cap = cv2.VideoCapture('../../SRC/videos/walking.mp4')

# Get the height and width of the frame (required to be an interfere)
w = int(cap.get(3))
h = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'walking_output.avi' file.
out = cv2.VideoWriter('walking_output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (w, h))

body_detector = cv2.CascadeClassifier('../../SRC/Haarcascades/haarcascade_fullbody.xml')

# Loop once video is successfully loaded
while True:

    ret, frame = cap.read()
    if ret:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Pass frame to our body classifier
        bodies = body_detector.detectMultiScale(gray, 1.2, 3)

        # Extract bounding boxes for any bodies identified
        for (x, y, w, h) in bodies:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Write the frame into the file 'output.avi'
        out.write(frame)
    else:
        break

cap.release()
out.release()

# ----------------------------------------
#       3: Use a Haar cascade classier to detect Vehicles or cars
# ----------------------------------------

# Create our video capturing object
cap = cv2.VideoCapture('../../SRC/videos/cars.mp4')

# Load our vehicle classifier
vehicle_detector = cv2.CascadeClassifier('../../SRC/Haarcascades/haarcascade_car.xml')

# Read first frame
ret, frame = cap.read()

# Ret is True if successfully read
if ret:

    # Grayscale our image for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Pass frame to our body classifier
    vehicles = vehicle_detector.detectMultiScale(gray, 1.4, 2)

    # Extract bounding boxes for any bodies identified
    for (x, y, w, h) in vehicles:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

# Release our video capture
cap.release()
imshow("Vehicle Detector", frame)

# Create our video capturing object
cap = cv2.VideoCapture('../../SRC/videos/cars.mp4')

# Get the height and width of the frame (required to be an interfer)
w = int(cap.get(3))
h = int(cap.get(4))

# Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
out = cv2.VideoWriter('cars_output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (w, h))

vehicle_detector = cv2.CascadeClassifier('../../SRC/Haarcascades/haarcascade_car.xml')

# Loop once video is successfully loaded
while True:

    ret, frame = cap.read()
    if ret:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Pass frame to our body classifier
        vehicles = vehicle_detector.detectMultiScale(gray, 1.2, 3)

        # Extract bounding boxes for any bodies identified
        for (x, y, w, h) in vehicles:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Write the frame into the file 'output.avi'
        cv2.imshow("video", frame)
        out.write(frame)
    else:
        break

cap.release()
out.release()
