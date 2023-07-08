# Python Image Processing Script

This Python script contains several functions for image processing, including gradient computation, non-maximum suppression, and Hough transformation.

## Dependencies

To run this script, you need the following Python libraries:

- `scipy`
- `numpy`
- `matplotlib`
- `cv2`

Make sure you have them installed. If not, use pip:

`pip install scipy numpy matplotlib opencv-python`

## Functions

### `to_ndarray(img)`

This function takes an image and converts it to a numpy ndarray. The image is read in grayscale and then converted to a 32-bit integer ndarray.

### `round_angle(angle)`

The `round_angle` function takes an angle in radians and normalizes it to one of 0, 45, 90, or 135 degrees, representing the direction of the gradient.

### `gs_filter(img, sigma)`

This function applies a Gaussian filter to the input image. This is done to reduce noise and detail in the image.

### `gradient_intensity(img)`

This function computes the gradient of the input image using the Sobel operator. It returns the gradient magnitude and direction at each pixel.

### `suppression(img, D)`

This function performs non-maximum suppression on the input image, thinning the edges.

### `threshold(img, t, T)`

This function thresholds the input image. Pixels with intensity above the high threshold (T) are marked as strong. Pixels with intensity between the low (t) and high (T) thresholds are marked as weak. All other pixels are set to zero.

### `tracking(img, weak, strong=255)`

This function tracks and extends weak edges in the input image. Weak edges are only preserved if they are connected to strong edges.

### `hough_line(img)`

This function performs the Hough transform to detect lines in the image. It returns the Hough accumulator, the range of rho values, and the range of theta values.

### `find_all_max(image, num_of_corner)`

This function uses the Hough transform to find the corners in the image. It returns the rho and theta of the corners.

## Usage

The script is used by running the code in the main section, which creates an image with shapes, performs Hough transform to find the corners, and then displays the original and processed image.

## Results

![alt text](https://github.com/MohammadKhayyo/Image-Processing/blob/main/Class%20Exercises/Ex3%20-%20Hough_Transform/output.png)
