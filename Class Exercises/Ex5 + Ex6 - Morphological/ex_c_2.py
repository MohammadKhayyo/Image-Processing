import cv2
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy

img = cv2.imread("img.png", 0).astype("uint8")
kernel = np.ones((10, 10), dtype=np.uint8)
img_after_erode = cv2.erode(deepcopy(img), kernel)
edges = img - img_after_erode
ret, edges = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
plt.subplot(121)
plt.title("edges")
plt.imshow(edges, cmap='gray')
plt.subplot(122)
plt.title("img")
plt.imshow(img, cmap='gray')
plt.show()
