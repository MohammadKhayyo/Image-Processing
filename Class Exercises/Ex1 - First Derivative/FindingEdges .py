import numpy as np
import cv2
from matplotlib import pyplot as plt


def img_kernel(image, kernel):
    grad_x = cv2.filter2D(image, -1, kernel)
    grad_y = cv2.filter2D(image, -1, kernel.T)
    return np.absolute(grad_x) + np.absolute(grad_y)


if __name__ == "__main__":
    img = np.zeros([200, 100])
    img[25:75, 25:75] = np.ones([50, 50])
    img[125:175, 25:75] = np.ones([50, 50])
    plt.subplot(121)
    plt.imshow(img_kernel(img, np.array([[1, -1], [1, -1]])), cmap='gray')
    plt.subplot(122)
    plt.imshow(img, cmap='gray')
    plt.show()
