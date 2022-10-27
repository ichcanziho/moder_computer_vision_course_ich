import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage.filters import threshold_local

image = cv2.imread("../../SRC/images/ine_n.jpeg")
# Original Image
cv2.imshow("original", image)
# Convert to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# cv2.imshow('After thresholding', th2)
kernel = np.ones((5, 5), np.uint8)
th2 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel)
# cv2.imshow('After thresholding and closing', th2)

# Find contour and sort by contour area
cnts = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

# Find bounding box and extract ROI
ROI = None
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    ROI = image[y:y + h, x:x + w]
    break

cv2.imshow("roi", ROI)
V = cv2.split(cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV))[2]
T = threshold_local(V, 25, offset=15, method="gaussian")
thresh_skimage = (V > T).astype("uint8") * 255
cv2.imshow("threshold_local", thresh_skimage)
gray = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)
_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# cv2.imshow("threshold_otsu", thresh_skimage)
cv2.waitKey(0)


