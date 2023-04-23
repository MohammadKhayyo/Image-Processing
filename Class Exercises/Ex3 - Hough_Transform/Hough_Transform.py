from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy import ndimage
import cv2


def to_ndarray(img):
    im = misc.imread(img, flatten=True)
    im = im.astype('int32')
    return im


def round_angle(angle):
    angle = np.rad2deg(angle) % 180
    if (0 <= angle < 22.5) or (157.5 <= angle < 180):
        angle = 0
    elif 22.5 <= angle < 67.5:
        angle = 45
    elif 67.5 <= angle < 112.5:
        angle = 90
    elif 112.5 <= angle < 157.5:
        angle = 135
    return angle


def gs_filter(img, sigma):
    if type(img) != np.ndarray:
        raise TypeError('Input image must be of type ndarray.')
    else:
        return gaussian_filter(img, sigma)


def gradient_intensity(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32)

    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    D = np.arctan2(Iy, Ix)
    return (G, D)


def suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    for i in range(M):
        for j in range(N):
            # find neighbour pixels to visit from the gradient directions
            where = round_angle(D[i, j])
            try:
                if where == 0:
                    if (img[i, j] >= img[i, j - 1]) and (img[i, j] >= img[i, j + 1]):
                        Z[i, j] = img[i, j]
                elif where == 90:
                    if (img[i, j] >= img[i - 1, j]) and (img[i, j] >= img[i + 1, j]):
                        Z[i, j] = img[i, j]
                elif where == 135:
                    if (img[i, j] >= img[i - 1, j - 1]) and (img[i, j] >= img[i + 1, j + 1]):
                        Z[i, j] = img[i, j]
                elif where == 45:
                    if (img[i, j] >= img[i - 1, j + 1]) and (img[i, j] >= img[i + 1, j - 1]):
                        Z[i, j] = img[i, j]
            except IndexError as e:
                """ Todo: Deal with pixels at the image boundaries. """
                pass
    return Z


def threshold(img, t, T):
    # define gray value of a WEAK and a STRONG pixel
    cf = {
        'WEAK': np.int32(50),
        'STRONG': np.int32(255),
    }

    # get strong pixel indices
    strong_i, strong_j = np.where(img > T)

    # get weak pixel indices
    weak_i, weak_j = np.where((img >= t) & (img <= T))

    # get pixel indices set to be zero
    zero_i, zero_j = np.where(img < t)

    # set values
    img[strong_i, strong_j] = cf.get('STRONG')
    img[weak_i, weak_j] = cf.get('WEAK')
    img[zero_i, zero_j] = np.int32(0)

    return (img, cf.get('WEAK'))


def tracking(img, weak, strong=255):
    M, N = img.shape
    for i in range(M):
        for j in range(N):
            if img[i, j] == weak:
                # check if one of the neighbours is strong (=255 by default)
                try:
                    if ((img[i + 1, j] == strong) or (img[i - 1, j] == strong)
                            or (img[i, j + 1] == strong) or (img[i, j - 1] == strong)
                            or (img[i + 1, j + 1] == strong) or (img[i - 1, j - 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


def hough_line(img):
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0))
    width, height = img.shape
    diag_len = int(np.ceil(np.sqrt(width * width + height * height)))  # max_dist
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)
    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas))
    y_idxs, x_idxs = np.nonzero(img)  # (row, col) indexes to edges
    # Vote in the hough accumulator
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx]) + diag_len)
            accumulator[rho, t_idx] += 1
    return accumulator, thetas, rhos


def find_all_max(image, num_of_corner):
    accumulator, thetas, rhos = hough_line(image)
    rho_theta_of_all_max = {}
    for _ in range(num_of_corner):
        idx = np.argmax(accumulator)
        distance = int(idx / accumulator.shape[1])
        angle = int(idx % accumulator.shape[1])
        rho = rhos[distance]
        theta = thetas[angle]
        rho_theta_of_all_max[idx] = (rho, theta)
        accumulator[distance, angle] = -1
    return rho_theta_of_all_max


if __name__ == "__main__":
    img = np.zeros([200, 200], dtype=np.int8)
    img[10:60, 100:150] = np.eye(50)
    img[10:100, 10:100] = np.rot90(np.eye(90))
    img[100:150, 10:60] = np.eye(50)
    img[60:150, 60:150] = np.rot90(np.eye(90))
    plt.subplot(122)
    plt.imshow(img, cmap='gray')
    corner_max = find_all_max(img, 4)
    all_y = list()
    for key in list(corner_max.keys()):
        rho, theta = corner_max[key]
        m = -np.cos(theta) / np.sin(theta)
        b = rho / np.sin(theta)
        y = m * np.arange(0, img.shape[0]) + b
        all_y.append(y)
    all_dots = []
    for j, y in enumerate(all_y):
        y = [int(val) for val in y]
        for k, y2 in enumerate(all_y):
            if j == k:
                continue
            y2 = [int(val) for val in y2]
            for i in range(len(y) - 1):
                if y[i] == y2[i + 1] or y[i] == y2[i]:
                    all_dots.append((i, y[i]))
                    break
    corner = np.zeros([200, 200])
    for _x, _y in all_dots:
        corner[_x, _y] = 1
    corner = cv2.merge([corner, np.zeros([200, 200]), np.zeros([200, 200])])
    img = cv2.merge([img, img, img])
    corner = cv2.filter2D(corner, -1, np.matrix("5,10,5;10,10,10;5,10,5"))
    img_corner = corner + img
    plt.subplot(121)
    plt.imshow(img_corner, cmap='gray')
    plt.show()
