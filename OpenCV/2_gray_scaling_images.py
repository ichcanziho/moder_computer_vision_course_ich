import cv2
from matplotlib import pyplot as plt


# Define our imshow function
def imshow(title="Image", image=None, size=10):
    # we have added the size parameter to control de fig-size of the image while plotting with matplotlib, for that
    # we obtain the aspect ratio of the image dividing the width by the height.
    h, w = image.shape[0], image.shape[1]
    aspect_ratio = w/h
    plt.figure(figsize=(size * aspect_ratio, size))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()


img = cv2.imread("../SRC/images/castara.jpeg")
imshow("Castara, Tobago", img)

# We use cvtColor, to convert to grayscale
# It takes 2 arguments, the first being the input image
# The second being the color space conversion code
gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

imshow("Converted to Grayscale", gray_image)

print("Original Image shape: ", img.shape)
print("Gray scale Image shape:", gray_image.shape)
