# Image Processing with OpenCV

This repository contains Python code for two distinct image processing tasks performed using the OpenCV library. The tasks are:

1. Edge Detection - Extracting the edges of objects present in the image.
2. Image Filling - Filling the regions inside the edges of objects detected in the image.

## Description

The first script reads an image, applies erosion to it and then subtracts the eroded image from the original image to extract the edges of objects present in the image. 

The second script follows up on the first one. It uses the edges detected by the first script, performs region filling inside those edges, and then visualizes the filled and the original images side by side.

## Dependencies

The scripts require the following Python libraries:

- OpenCV (cv2)
- NumPy
- Matplotlib

These can be installed using pip:
```
pip install opencv-python numpy matplotlib
```

## How to Run

1. Clone this repository to your local machine.
2. Place the image you want to process in the same directory and rename it to "img.png" or modify the scripts to point to your image file.
3. Run the scripts using Python:
```
python <name_of_the_script.py>
```

## Code Explanation

### Edge Detection

1. The script reads an image from the file "img.png" in grayscale mode and converts it to an 8-bit unsigned integer array.

2. It then creates a 10x10 structuring element and applies erosion on the image. Erosion removes pixels at the boundaries of objects in an image.

3. The edges are extracted by subtracting the eroded image from the original image.

4. The image is then binarized using a threshold of 50. All pixels with values below 50 are set to 0 (black), and those with values above 50 are set to 255 (white).

5. Finally, it displays the extracted edges and the original image side by side using matplotlib.

### Image Filling

1. The script starts by following the same steps as the edge detection script to detect edges in the image.

2. It then performs region filling inside the detected edges. The seed point for the flood fill operation is (0, 0), and the fill value is 255 (white).

3. The filled image is then inverted because the floodFill operation fills the outside of the object.

4. Lastly, it displays the original and the filled images side by side using matplotlib.

## Results

![alt text](https://github.com/MohammadKhayyo/Image-Processing/blob/main/Class%20Exercises/Ex5%20%2B%20Ex6%20-%20Morphological/ex_c_2.png)

![alt text](https://github.com/MohammadKhayyo/Image-Processing/blob/main/Class%20Exercises/Ex5%20%2B%20Ex6%20-%20Morphological/ex_c_3.png)