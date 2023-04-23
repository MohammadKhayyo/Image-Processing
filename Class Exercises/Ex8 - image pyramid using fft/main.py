import cv2
import numpy as np
from matplotlib import pyplot as plt


def get_x_y(image, n_points=4):
    refPt = list()

    def click_and_crop(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            refPt.append([x, y])

    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)
    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF
        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()
        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break
        if len(refPt) == n_points:
            break
    return refPt


def Laplacian_Pyramid_Blending_with_mask(A, B, m, num_levels=6):
    GA = A.copy()
    GB = B.copy()
    GM = m.copy()
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in range(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float64(GA))
        gpB.append(np.float64(GB))
        gpM.append(np.float64(GM))
    lpA = [gpA[num_levels - 1]]
    lpB = [gpB[num_levels - 1]]
    gpMr = [gpM[num_levels - 1]]
    for i in range(num_levels - 1, 0, -1):
        print(i)
        # Laplacian: subtarctupscaledversion of lower# level from current level
        # to get the high frequencies
        LA = np.subtract(gpA[i - 1], cv2.pyrUp(gpA[i]))
        LB = np.subtract(gpB[i - 1], cv2.pyrUp(gpB[i]))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i - 1])  # also reverse the masks
    LS = []
    for la, lb, gm in zip(lpA, lpB, gpMr):
        ls = la * gm + lb * (1.0 - gm)
        ls.dtype = np.float64
        LS.append(ls)
    # now reconstruct
    ls_ = LS[0]
    # ls_.dtype= np.float64
    for i in range(1, num_levels):
        print("LS" + str(i))
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])
    return ls_


if __name__ == "__main__":
    A = cv2.imread("man.png", 0)
    B = cv2.imread("teen.jpg", 0)
    m = cv2.imread("mask.png", 0)
    m = cv2.normalize(m.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)
    width = 400
    height = 400

    A = cv2.resize(A, (width, height), interpolation=cv2.INTER_AREA)
    B = cv2.resize(B, (width, height), interpolation=cv2.INTER_AREA)
    m = cv2.resize(m, (width, height), interpolation=cv2.INTER_AREA)
    pts_A = get_x_y(A)
    pts_B = get_x_y(B)
    pts1 = np.float32(pts_A)
    pts2 = np.float32(pts_B)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    A = cv2.warpPerspective(A, M, A.shape)

    lpb = Laplacian_Pyramid_Blending_with_mask(A, B, m, 5)
    plt.imshow(lpb, cmap='gray')
    plt.show()
