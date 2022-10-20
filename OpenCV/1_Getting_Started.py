import cv2
from matplotlib import pyplot as plt

"""
Welcome to your first OpenCV Lesson. Here we'll learn to:

    1: Import the OpenCV Model in Python
    2: Load Images
    3: Display Images
    4: Save Images
    5: Getting the Image Dimensions
    
"""


# 1: Displaying Images
def imshow(title="", image=None):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # By default, the imread function of cv2 loads the image into
    # BGR Format, so we need to convert it into RGB to plot properly using matplotlib
    plt.imshow(image_rgb)
    plt.title(title)
    plt.show()


print(cv2.__version__)

img = cv2.imread("../SRC/images/castara.jpeg")

imshow("Displaying our first image", img)
cv2.imshow("Displaying our first image", img)
cv2.waitKey(0)

# 2: Saving images

cv2.imwrite("outputs/output.jpg", img)
cv2.imwrite("outputs/output.png", img)

# 3: Displaying Image Dimensions

print(img.shape)
print('Height of Image: {} pixels'.format(int(img.shape[0])))
print('Width of Image: {} pixels'.format(int(img.shape[1])))
print('Depth of Image: {} colors components'.format(int(img.shape[2])))
