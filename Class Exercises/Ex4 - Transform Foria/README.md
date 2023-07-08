# Image Morphology using OpenCV

This repository contains Python code that performs morphological operations on images using OpenCV. Specifically, the script uses Dilation and Erosion to manipulate the image.

## Description

The code reads an image from file, applies dilation, followed by erosion and then visualizes the original and the transformed images side by side.

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
2. Place the image you want to process in the same directory and rename it to "img.png" or modify the code to point to your image file.
3. Run the script using Python:
```
python <name_of_the_script.py>
```

## Code Explanation

The script follows these steps:

1. Imports the required modules: cv2 for image processing, numpy for numerical operations, and matplotlib to display the images.

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt
```

2. Reads an image from the file "img.png" in grayscale mode and converts it to an 8-bit unsigned integer array.

```python
image = cv2.imread("img.png", 0).astype("uint8")
```

3. Defines a 7x7 structuring element (kernel) and applies dilation to the image. Dilation adds pixels to the boundaries of objects in an image, which can help in joining broken parts of an object.

```python
kernel = np.ones((7, 7), dtype=np.uint8)
image_after_dilate = cv2.dilate(image, kernel)
```

4. Defines a 3x3 structuring element and applies erosion to the dilated image. Erosion removes pixels at the boundaries of objects in an image, which can help in removing small noise, disjointed parts of an object, etc.

```python
kernel = np.ones((3, 3), dtype=np.uint8)
image_after_erode = cv2.erode(image_after_dilate, kernel)
```

5. Finally, it uses matplotlib to display the original and the transformed images side by side.

```python
plt.subplot(121)
plt.imshow(image_after_erode, cmap='gray')
plt.subplot(122)
plt.imshow(image, cmap='gray')
plt.show()
```

**Note**: The size of the structuring element (kernel) and the order of applying dilation and erosion can greatly impact the final results. Feel free to modify these as per your requirements.

## Results

![alt text](https://github.com/MohammadKhayyo/Image-Processing/blob/main/Class%20Exercises/Ex4%20-%20Transform%20Foria/output.png)