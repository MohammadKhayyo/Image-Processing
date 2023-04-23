import cv2
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy

img = cv2.imread("img.png", 0).astype("uint8")
kernel = np.ones((10, 10), dtype=np.uint8)
img_after_erode = cv2.erode(deepcopy(img), kernel)
edges = img - img_after_erode
ret, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
image_copy = edges.copy()

# Define the seed point and the value to fill the region with
seed_point = (0, 0)
fill_value = (255, 255, 255)

# Perform the region filling
cv2.floodFill(image_copy, None, seed_point, fill_value)
image_Fill = np.invert(image_copy)
# Display the original and the filled image
plt.subplot(121)
plt.title('Original')
plt.imshow(edges, cmap='gray')
plt.subplot(122)
plt.title('Filled')
plt.imshow(image_Fill, cmap='gray')
plt.show()
