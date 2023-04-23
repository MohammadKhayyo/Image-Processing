import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("img.png", 0).astype("uint8")

kernel = np.ones((7, 7), dtype=np.uint8)
image_after_dilate = cv2.dilate(image, kernel)
kernel = np.ones((3, 3), dtype=np.uint8)
image_after_erode = cv2.erode(image_after_dilate, kernel)
plt.subplot(121)
plt.imshow(image_after_erode, cmap='gray')
plt.subplot(122)
plt.imshow(image, cmap='gray')
plt.show()
