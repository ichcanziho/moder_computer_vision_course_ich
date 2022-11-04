import cv2
from matplotlib import pyplot as plt
import os
import pathlib
import numpy as np

"""
In this lesson we'll learn to:

    1:  use the GrabCut Algorithm for background Removal

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
#       1: use the GrabCut Algorithm for background Removal
# ---------------------------------------------------------------

"""

    - User inputs the rectangle. Everything outside this rectangle will be taken as the background. Everything inside 
    rectangle is unknown. 
    
    - The algorithm labels the foreground and background pixels (or it hard-labels)
     
    - Next a Gaussian Mixture Model(GMM) is used to model the foreground and background. 
    
    - Depending on the data we gave,  GMM learns and create new pixel distribution. That is, the unknown pixels are 
    labelled either probable foreground or probable background depending on its relation with the other hard-labelled 
    pixels in terms of color statistics (It is just like clustering). 
    
    - A graph is built from this pixel distribution. Nodes in the graphs are pixels. Additional two nodes are added, 
    Source node and Sink node. Every foreground pixel is connected to Source node and every background pixel is 
    connected to Sink node. 
    
    - The weights of edges connecting pixels to source node/end node are defined by the probability of a pixel being 
    foreground/background. The weights between the pixels are defined by the edge information or pixel similarity. If 
    there is a large difference in pixel color, the edge between them will get a low weight. = Then a mincut algorithm 
    is used to segment the graph. It cuts the graph into two separating source node and sink node with minimum cost 
    function. The cost function is the sum of all weights of the edges that are cut. After the cut, all the pixels 
    connected to Source node become foreground and those connected to Sink node become background. 
    
    - The process is continued until the classification converges. 

"""

# Load our image
image = cv2.imread('../../SRC/images/woman.jpeg')
copy = image.copy()
# Create a mask (of zeros uint8 datatype) that is the same size (width, height) as our original image
mask = np.zeros(image.shape[:2], np.uint8)

bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# Needs to be set Manually or selected with cv2.selectROI()
x1, y1, x2, y2 = 190, 70, 350, 310
start = (x1, y1)
end = (x2, y2)

# Format is X,Y,W,H
rect = (x1, y1, x2 - x1, y2 - y1)

# Show Rectangle
cv2.rectangle(copy, start, end, (0, 0, 255), 3)
imshow("Input Image red squared", copy)

"""
Grabcut Arguments

    > img - Input image
    > mask - It is a mask image where we specify which areas are background, foreground or probable background/
    foreground etc. It is done by the following flags, cv.GC_BGD, cv.GC_FGD, cv.GC_PR_BGD, cv.GC_PR_FGD, or simply pass 
    0,1,2,3 to image.
    > rect - It is the coordinates of a rectangle which includes the foreground object in the format (x,y,w,h)
    > bdgModel, fgdModel - These are arrays used by the algorithm internally. You just create two np.float64 type zero 
    arrays of size (1,65).
    > iterCount - Number of iterations the algorithm should run.
    > mode - It should be cv.GC_INIT_WITH_RECT or cv.GC_INIT_WITH_MASK or combined which decides whether we are drawing 
    rectangle or final touch up strokes.

"""

# Let the algorithm run for 5 iterations. Mode should be cv.GC_INIT_WITH_RECT since we are using rectangle.
# Grabcut modifies the mask image.
# In the new mask image, pixels will be marked with four flags denoting background/foreground as specified above.
# So we modify the mask such that all 0-pixels and 2-pixels are put to 0 (i.e. background) & all 1-pixels and 3-pixels
# are put to 1(i.e. foreground pixels).
# Now our final mask is ready. Just multiply it with input image to get the segmented image.
cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
image = image * mask2[:, :, np.newaxis]

imshow("Mask", mask * 80)
imshow("Mask2", mask2 * 255)
imshow("Image", image)
