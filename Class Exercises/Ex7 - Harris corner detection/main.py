import cv2
import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy

if __name__ == "__main__":
    img = np.zeros([200, 200], dtype=np.int8)
    img[10:60, 100:150] = np.eye(50)
    img[10:100, 10:100] = np.rot90(np.eye(90))
    img[100:150, 10:60] = np.eye(50)
    img[60:150, 60:150] = np.rot90(np.eye(90))
    kernel = np.ones((7, 7), dtype=np.uint8)
    img = np.float32(img)
    img = cv2.dilate(img, None)
    dst = cv2.cornerHarris(img, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    original_image = deepcopy(img)
    corner = deepcopy(img)
    corner[dst > 0.01 * dst.max()] = 255
    ret, corner = cv2.threshold(corner, 200, 255, cv2.THRESH_BINARY)
    img_without_corner = original_image - corner
    corner = np.dstack(
        (corner, np.zeros([corner.shape[0], corner.shape[1]]), np.zeros([corner.shape[0], corner.shape[1]])))
    original_image = cv2.merge([original_image, original_image, original_image])
    img_without_corner = np.dstack((img_without_corner, img_without_corner, img_without_corner))
    img = img_without_corner + corner
    plt.subplot(121)
    plt.title("original image")
    plt.imshow(original_image, cmap='gray')
    plt.subplot(122)
    plt.title("dst")
    plt.imshow(img, cmap='gray')
    plt.show()
