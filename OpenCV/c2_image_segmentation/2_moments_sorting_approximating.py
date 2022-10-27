import cv2
from matplotlib import pyplot as plt

"""
In this lesson we'll learn to:

    1: Sort Contours by Area
    2: Sort by Left to Right (Great for OCR)
    3: Approximate Contours
    4: Convex Hull
    5: Matching Contours

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
        plt.savefig(f"../outputs/c2_image_segmentation/2_moments_sorting_approximating/{title}.png")
    plt.show()


# ----------------------------------------
#       1: Sort Contours by Area
# ----------------------------------------

# Load our image
image = cv2.imread('../../SRC/images/bunchofshapes.jpg')
imshow('Original Image', image)

# Grayscale our image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Find Canny edges
edged = cv2.Canny(gray, 50, 200)
imshow('Canny Edges', edged)

# Find contours and print how many were found
contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
print("Number of contours found = ", len(contours))

# Draw all contours over blank image
cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
imshow('All Contours', image)


# Function we'll use to display contour area

def get_contour_areas(contours):
    """returns the areas of all contours as list"""
    all_areas = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas.append(area)
    return all_areas


image = cv2.imread('../../SRC/images/bunchofshapes.jpg')

# Let's print the areas of the contours before sorting
print("Contour Areas before sorting...")
print(get_contour_areas(contours))

# Sort contours large to small by area
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

print("Contour Areas after sorting...")
print(get_contour_areas(sorted_contours))

# Iterate over our contours and draw one at a time
for (i, c) in enumerate(sorted_contours):
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cv2.putText(image, str(i + 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.drawContours(image, [c], -1, (255, 0, 0), 3)

imshow('Contours by area', image)


# ----------------------------------------
#       2: Sort by Left to Right (Great for OCR)
# ----------------------------------------

# Functions we'll use for sorting by position
def x_cord_contour(contours):
    """Returns the X cordinate for the contour centroid"""
    if cv2.contourArea(contours) > 10:
        M = cv2.moments(contours)
        return int(M['m10'] / M['m00'])
    else:
        pass


def label_contour_center(image, c):
    """Places a red circle on the centers of contours"""
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])

    # Draw the contour number on the image
    cv2.circle(image, (cx, cy), 10, (0, 0, 255), -1)
    return image


image = cv2.imread('../../SRC/images/bunchofshapes.jpg')
orginal_image = image.copy()

# Computer Center of Mass or centroids and draw them on our image
for (i, c) in enumerate(contours):
    orig = label_contour_center(image, c)

# Showing the Contour centers
imshow("Sorting Left to Right - center", image)

# Sort by left to right using our x_cord_contour function
contours_left_to_right = sorted(contours, key=x_cord_contour, reverse=False)

# Labeling Contours left to right
for (i, c) in enumerate(contours_left_to_right):
    cv2.drawContours(orginal_image, [c], -1, (0, 0, 255), 3)
    M = cv2.moments(c)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    cv2.putText(orginal_image, str(i + 1), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    (x, y, w, h) = cv2.boundingRect(c)

imshow('Sorting Left to Right', orginal_image)

# ----------------------------------------
#       3: Approximate Contours
# ----------------------------------------

"""
Using ApproxPolyDP to approximate contours as a more defined shape
It approximates a contour shape to another shape with less number of vertices depending upon the precision we specify.

cv2.approxPolyDP(contour, Approximation Accuracy, Closed)

    - contour – is the individual contour we wish to approximate
    - Approximation Accuracy – Important parameter is determining the accuracy of the approximation. 
      Small values give precise- approximations, large values give more generic approximation. A good rule of thumb is 
      less than 5% of the contour perimeter
    - Closed – a Boolean value that states whether the approximate contour should be open or closed
"""

# Load image and keep a copy
image = cv2.imread('../../SRC/images/house.jpg')
orig_image = image.copy()
imshow('Original House Image', orig_image)

# Grayscale and binarize
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

# Find contours
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
copy = image.copy()

# Iterate through each contour
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(orig_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.drawContours(image, [c], 0, (0, 255, 0), 2)

imshow('Drawing of Contours', image)
imshow('Bounding Rectangles', orig_image)
# Iterate through each contour and compute the approx contour
for c in contours:
    # Calculate accuracy as a percent of the contour perimeter
    accuracy = 0.03 * cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, accuracy, True)
    cv2.drawContours(copy, [approx], 0, (0, 255, 0), 2)

imshow('Approx Poly DP', copy)

# ----------------------------------------
#       4: Convex Hull
# ----------------------------------------

"""
Convex Hull will look similar to contour approximation, but it is not (Both may provide the same results in some cases).

The cv2.convexHull() function checks a curve for convexity defects and corrects it. Generally speaking, convex curves 
are the curves which are always bulged out, or at-least flat. And if it is bulged inside, it is called convexity 
defects. For example, check the below image of hand. Red line shows the convex hull of hand. The double-sided arrow 
marks shows the convexity defects, which are the local maximum deviations of hull from contours 
"""

image = cv2.imread('../../SRC/images/hand.jpg')
orginal_image = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

imshow('Original Hand Image', image)

# Threshold the image
ret, thresh = cv2.threshold(gray, 176, 255, 0)

# Find contours
contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(image, contours, 0, (0, 255, 0), 2)
imshow('Contours of Hand', image)

# Sort Contours by area and then remove the largest frame contour
n = len(contours) - 1
contours = sorted(contours, key=cv2.contourArea, reverse=False)[:n]

# Iterate through contours and draw the convex hull
for c in contours:
    hull = cv2.convexHull(c)
    cv2.drawContours(orginal_image, [hull], 0, (0, 255, 0), 2)

imshow('Convex Hull', orginal_image)

# ----------------------------------------
#       5: Matching Contours
# ----------------------------------------

# Load the shape template or reference image
template = cv2.imread('../../SRC/images/4star.jpg', 0)
imshow('Template', template)

# Load the target image with the shapes we're trying to match
target = cv2.imread('../../SRC/images/shapestomatch.jpg')
target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

# Threshold both images first before using cv2.findContours
ret, thresh1 = cv2.threshold(template, 127, 255, 0)
ret, thresh2 = cv2.threshold(target_gray, 127, 255, 0)

# Find contours in template
contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

# We need to sort the contours by area so that we can remove the largest
# contour which is the image outline
sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

# We extract the second-largest contour which will be our template contour
template_contour = contours[1]

# Extract contours from second target image
contours, hierarchy = cv2.findContours(thresh2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

closest_contour = []
for c in contours:
    # Iterate through each contour in the target image and
    # use cv2.matchShapes to compare contour shapes
    match = cv2.matchShapes(template_contour, c, 3, 0.0)
    print(match)
    # If the match value is less than 0.15 we
    if match < 0.15:
        closest_contour = c
    else:
        closest_contour = []

cv2.drawContours(target, [closest_contour], -1, (0, 255, 0), 3)
imshow('Output matching', target)
