import numpy as np
import cv2
from matplotlib import pyplot as plt


def img_kernel(image, kernel):
    grad_x = cv2.filter2D(image, -1, kernel)
    grad_y = cv2.filter2D(image, -1, kernel)
    return np.absolute(grad_x) + np.absolute(grad_y)


if __name__ == "__main__":
    img = np.zeros([200, 100])
    img[25:75, 25:75] = np.ones([50, 50])
    img[125:175, 25:75] = np.ones([50, 50])
    plt.subplot(121)
    kernel = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]])
    corner = img_kernel(img, kernel)
    corner = cv2.merge([corner, np.zeros([200, 100]), np.zeros([200, 100])])
    img = cv2.merge([img, img, img])
    corner = cv2.filter2D(corner, -1, np.matrix("5,10,5;10,10,10;5,10,5"))
    img_corner = corner + img
    plt.imshow(img_corner, cmap='gray')
    plt.subplot(122)
    plt.imshow(img, cmap='gray')
    plt.show()
