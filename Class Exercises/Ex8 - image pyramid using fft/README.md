# Laplacian Pyramid Blending with Mask

This repository contains a Python implementation of the Laplacian Pyramid blending algorithm. The algorithm is used to blend two images based on a mask.

## Description

The script performs the following tasks:

1. It reads three input images (two for blending and one for the mask).
2. The mask is normalized and all images are resized to match the dimensions of the input images.
3. The perspective transformation of one image is calculated based on four points selected manually.
4. The Laplacian Pyramid blending algorithm is used to blend the two images based on the mask.
5. The resulting image is displayed.

## Dependencies

To run this script, you need the following Python libraries:
- OpenCV (cv2)
- NumPy
- Matplotlib

You can install these dependencies using pip:
```
pip install opencv-python numpy matplotlib
```

## How to Run

1. Clone this repository to your local machine.
2. Run the script using Python:
```
python <name_of_the_script.py>
```

## Code Explanation

1. The script reads the two input images and the mask. The mask is normalized and all images are resized to the same dimensions.

```python
A = cv2.imread("man.png", 0)
B = cv2.imread("teen.jpg", 0)
m = cv2.imread("mask.png", 0)
m = cv2.normalize(m.astype('float64'), None, 0.0, 1.0, cv2.NORM_MINMAX)
```

2. The perspective transformation of image A is calculated based on four points selected manually. The same points are used for image B.

```python
pts_A = get_x_y(A)
pts_B = get_x_y(B)
pts1 = np.float32(pts_A)
pts2 = np.float32(pts_B)
M = cv2.getPerspectiveTransform(pts1, pts2)
A = cv2.warpPerspective(A, M, A.shape)
```

3. The Laplacian Pyramid blending algorithm is used to blend the two images based on the mask.

```python
lpb = Laplacian_Pyramid_Blending_with_mask(A, B, m, 5)
```

4. The resulting blended image is displayed using matplotlib.

```python
plt.imshow(lpb, cmap='gray')
plt.show()
```

## Contributing

If you want to contribute to this project and make it better, your help is very welcome. You can make constructive issues, feature requests, pull requests or contribute with anything else.

![alt text](https://github.com/MohammadKhayyo/Image-Processing/blob/main/Class%20Exercises/Ex8%20-%20image%20pyramid%20using%20fft/output.png)