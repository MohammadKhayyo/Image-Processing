# Corner Detection using Harris Corner Detector

This repository contains Python code to detect corners in an image using the Harris Corner Detection algorithm. The image used is a grayscale image created from NumPy arrays.

## Description

The script performs the following tasks:

1. It creates an image using NumPy arrays and different geometric shapes such as rectangles and rotated eyes.
2. Applies dilation to enhance the edges of the image.
3. Uses the Harris Corner Detection algorithm to detect the corners in the image.
4. Highlights the detected corners in the original image and displays the results.

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

1. The script begins by creating a blank image of size 200x200 using NumPy zeros. Several geometric shapes, such as rectangles and rotated eyes, are then added to the image.

```python
img = np.zeros([200, 200], dtype=np.int8)
img[10:60, 100:150] = np.eye(50)
img[10:100, 10:100] = np.rot90(np.eye(90))
img[100:150, 10:60] = np.eye(50)
img[60:150, 60:150] = np.rot90(np.eye(90))
```

2. The script applies dilation to the image. This is done to enhance the edges of the geometric shapes in the image.

```python
img = np.float32(img)
img = cv2.dilate(img, None)
```

3. It uses the Harris Corner Detection algorithm to detect the corners in the image.

```python
dst = cv2.cornerHarris(img, 2, 3, 0.04)
dst = cv2.dilate(dst, None)
```

4. The corners are then highlighted by assigning them a value of 255 (white). The corners are also extracted and a separate image is created that only contains the corners.

```python
corner = deepcopy(img)
corner[dst > 0.01 * dst.max()] = 255
ret, corner = cv2.threshold(corner, 200, 255, cv2.THRESH_BINARY)
```

5. It merges the corner image with the original image and displays them side by side using matplotlib.

```python
plt.subplot(121)
plt.title("original image")
plt.imshow(original_image, cmap='gray')
plt.subplot(122)
plt.title("dst")
plt.imshow(img, cmap='gray')
plt.show()
```

![alt text](https://github.com/MohammadKhayyo/Image-Processing/blob/main/Class%20Exercises/Ex7%20-%20Harris%20corner%20detection/output.png)
