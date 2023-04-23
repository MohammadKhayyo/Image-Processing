import cv2 as cv
import os
import numpy as np
from skimage.morphology import skeletonize as skelt
from scipy import ndimage


def normalize_image(image_data, target_mean=100.0, target_variance=100.0):
    """
    Normalize an input image to a specified mean and variance.
    :param image_data (numpy.ndarray): A numpy array representing the input image.
    :param target_mean (float): A float representing the desired mean of the normalized image. Default is 100.0.
    :param target_variance (float): A float representing the desired variance of the normalized image. Default is 100.0.
    :return:numpy.ndarray: A numpy array representing the normalized image with the specified mean and variance.
    :source: https://pdfs.semanticscholar.org/6e86/1d0b58bdf7e2e2bb0ecbf274cee6974fe13f.pdf equation 21
    """
    # Compute the mean of the input image
    image_mean = np.mean(image_data)

    # Compute the variance of the input image
    image_variance = np.var(image_data)

    # Normalize the input image by subtracting its mean, multiplying it by the square root of the target variance
    # divided by the image variance, and then adding the target mean
    normalized_image = (image_data - image_mean) * np.sqrt(target_variance / image_variance) + target_mean

    # Convert the normalized image to an unsigned integer array
    return normalized_image.astype(np.uint8)


def segment_image(img_norm, window_size=16, threshold=0.3):
    """
    This function segments an input grayscale image by applying a binary mask based on the variance of local image blocks.
    It then performs morphological opening and closing to remove noise and produce a final segmented image.

    :param img_norm: A normalized grayscale image as a NumPy array.
    :param window_size: The size of local image blocks as an integer (default=16).
    :param threshold: The threshold value used to segment the fingerprint image.
            Default value is 0.3.
    :return: A tuple containing the segmented image, the normalized image, and the binary mask.
    :source: https://pdfs.semanticscholar.org/6e86/1d0b58bdf7e2e2bb0ecbf274cee6974fe13f.pdf
    """
    # Get number of rows and columns in the image
    rows, cols = img_norm.shape
    threshold = np.std(img_norm) * threshold
    # Initialize an array to store the local variance of each block
    image_variance = np.zeros(img_norm.shape)

    # Initialize a binary mask with the same shape as the image
    mask = np.ones_like(img_norm)

    # Loop over each block of the image
    for cols_block in range(cols // window_size):
        for rows_block in range(rows // window_size):
            # Determine the indices of the current block
            row_start = rows_block * window_size
            col_start = cols_block * window_size
            row_end = row_start + window_size
            col_end = col_start + window_size

            # Calculate the standard deviation of the current block
            blk_std_dev = np.std(img_norm[row_start:row_end, col_start:col_end])

            # Store the standard deviation of the current block in the corresponding location of the image_variance array
            image_variance[row_start:row_end, col_start:col_end] = blk_std_dev

    # Set all pixel values in the mask to 0 where the corresponding value in image_variance is below the threshold
    mask[image_variance < threshold] = 0

    # Create a structuring element for morphological operations
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (window_size * 2, window_size * 2))

    # Perform morphological opening on the mask
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    # Perform morphological closing on the mask
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # Apply the mask to the normalized image to obtain the segmented image
    seg_img = img_norm * mask

    # Normalize the image
    img_norm = (img_norm - np.mean(img_norm)) / np.std(img_norm)

    # Normalize the image again, this time only using pixel values where mask == 0
    img_norm = (img_norm - np.mean(img_norm[mask == 0])) / np.std(img_norm[mask == 0])

    # Return the segmented image, the normalized image, and the mask
    return seg_img, img_norm, mask


def calculate_angles(image, window_size=16, smooth=True):
    """
    Calculate the angles of gradients for each block in an image.

    :param image: A 2D numpy array representing the image.
    :param window_size: An integer representing the size of each block.
    :param smooth: A boolean representing whether or not to apply Gaussian smoothing to the image before processing.
    :return: A 2D numpy array representing the angles of gradients for each block in the image.
    """
    # Get the rows and columns of the image
    rows, cols = image.shape

    # Define a Sobel operator for edge detection in x and y directions
    sobel_operator = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

    # Define a vertical Sobel operator by transposing the horizontal one
    vertical_sobel_operator = np.array(sobel_operator).astype(np.int_)
    horizontal_sobel_operator = np.transpose(vertical_sobel_operator).astype(np.int_)

    # Create a list of angles for each block
    angles = [[] for _ in range(1, rows, window_size)]

    # Apply Gaussian smoothing to the image if smooth is True
    if smooth:
        image = ndimage.gaussian_filter(image, sigma=1.5)

    # Apply vertical and horizontal Sobel operators to the image to get gradient values
    G_x = ndimage.convolve(image / 125, vertical_sobel_operator, mode='nearest') * 125
    G_y = ndimage.convolve(image / 125, horizontal_sobel_operator, mode='nearest') * 125

    # Iterate over each block of the image and calculate the angle of the gradient
    for row in range(1, rows, window_size):
        for col in range(1, cols, window_size):
            # Initialize the numerator and denominator for the angle calculation
            nominator = 0
            denominator = 0

            # Iterate over each pixel in the block and update the numerator and denominator
            for r in range(row, min(row + window_size, rows - 1)):
                for c in range(col, min(col + window_size, cols - 1)):
                    # Round the gradient values to integers
                    G_x_r_c = round(G_x[r, c])
                    G_y_r_c = round(G_y[r, c])

                    # Update the numerator and denominator
                    nominator += 2 * G_x_r_c * G_y_r_c
                    denominator += G_x_r_c ** 2 - G_y_r_c ** 2

            # If the nominator or denominator is not 0, calculate the angle and append it to the list
            if nominator or denominator:
                angle = (np.pi + np.arctan2(nominator, denominator)) / 2
                angles[int((row - 1) // window_size)].append(angle)
            else:
                angles[int((row - 1) // window_size)].append(0)

    # Convert the list of angles to a numpy array and return it
    return np.array(angles)


def calculate_frequency(im, orientation, kernel_size, min_wavelength, max_wavelength):
    """
    Calculates the frequency block of an image.

    Parameters:
        im (numpy.ndarray): The input image.
        orientation (float): The orientation angle in radians.
        kernel_size (int): The size of the square structuring element for dilation.
        min_wavelength (float): The minimum wavelength for the frequency block.
        max_wavelength (float): The maximum wavelength for the frequency block.

    Returns:
        numpy.ndarray: The frequency block of the image.
    Source:
        https://pdfs.semanticscholar.org/ca0d/a7c552877e30e1c5d87dfcfb8b5972b0acd9.pdf pg.14
    """
    # Calculate cosine and sine of twice the orientation angle.
    cos_orientation = np.cos(2 * orientation)
    sin_orientation = np.sin(2 * orientation)

    # Calculate the block orientation by taking the arctangent of sine and cosine.
    block_orientation = np.arctan2(sin_orientation, cos_orientation) / 2

    # Rotate the image by the calculated block orientation angle.
    rotated_image = ndimage.rotate(im, block_orientation / np.pi * 180 + 90, reshape=False, order=3, mode='nearest')

    # Crop the rotated image to a square shape.
    crop_size = int(np.fix(im.shape[0] / np.sqrt(2)))
    offset = int(np.fix((im.shape[0] - crop_size) / 2))
    rotated_image = rotated_image[offset:offset + crop_size, offset:offset + crop_size]

    # Calculate the sum of pixel values along each column of the rotated image.
    ridge_sum = np.sum(rotated_image, axis=0)

    # Dilate the ridge sum using a square structuring element of given kernel size.
    dilation = ndimage.grey_dilation(ridge_sum, size=kernel_size, structure=np.ones(kernel_size))

    # Calculate ridge noise by subtracting the ridge sum from the dilated ridge sum.
    ridge_noise = np.abs(dilation - ridge_sum)

    # Find the maximum points where the ridge noise is below a given threshold and ridge sum is above the mean ridge sum.
    peak_threshold = 2
    max_points = (ridge_noise < peak_threshold) & (ridge_sum > np.mean(ridge_sum))
    max_indices = np.where(max_points)[0]

    # Calculate the frequency block by determining the wavelength of the maximum points and comparing it with the given wavelength bounds.
    if len(max_indices) < 2:
        freq_block = np.zeros(im.shape)
    else:
        wavelength = (max_indices[-1] - max_indices[0]) / (len(max_indices) - 1)
        if min_wavelength <= wavelength <= max_wavelength:
            freq_block = 1 / wavelength * np.ones(im.shape)
        else:
            freq_block = np.zeros(im.shape)

    # Return the frequency block.
    return freq_block


def estimate_ridge_frequency(image, mask, orientation, window_size=16, kernel_size=5, min_wavelength=5,
                             max_wavelength=15):
    """
    Estimates the median frequency of ridges in the input image.

    Parameters:
        image (numpy.ndarray): The input image.
        mask (numpy.ndarray): The mask for the input image.
        orientation (numpy.ndarray): The orientation angles of the input image.
        window_size (int): The size of the blocks to use for frequency estimation.
        kernel_size (int): The size of the square structuring element for dilation.
        min_wavelength (float): The minimum wavelength for the frequency block.
        max_wavelength (float): The maximum wavelength for the frequency block.

    Returns:
        numpy.ndarray: The estimated median frequency.
    """
    # Create a numpy array with zeros that has the same shape as the input image
    frequency = np.zeros_like(image)
    # Get the number of rows and columns of the input image
    rows, cols = image.shape
    # Loop through each block of the image, with the size of "window_size"
    for row in range(0, rows - window_size, window_size):
        for col in range(0, cols - window_size, window_size):
            # Get the orientation angle for this block
            angle_block = orientation[row // window_size][col // window_size]
            # If there is an angle, calculate the frequency using the calculate_frequency function
            if angle_block:
                image_block = image[row:row + window_size, col:col + window_size]
                frequency[row:row + window_size, col:col + window_size] = calculate_frequency(image_block, angle_block,
                                                                                              kernel_size,
                                                                                              min_wavelength,
                                                                                              max_wavelength)
    # Apply the mask to the frequency array
    frequency *= mask
    # Get all non-zero elements in the frequency array
    non_zero_elems_in_freq = frequency[frequency > 0]
    # Calculate the median of non-zero elements and apply the mask to it
    median_freq = np.median(non_zero_elems_in_freq) * mask
    # Return the median frequency
    return median_freq


def gabor_ridge_filter(image, orientation, frequency, kx=0.65, ky=0.65, angle_increment=3):
    """
    Applies a Gabor filter to an image to extract ridges at specific orientations and frequencies.

    Parameters:
        image (numpy array): Input image.
        orientation (numpy array): Orientation of the ridges in radians.
        frequency (numpy array): Frequency of the ridges.
        kx (float): Scaling factor for sigma_x.
        ky (float): Scaling factor for sigma_y.
        angle_inc (int): Angle increment for the Gabor filter.

    Returns:
        numpy array: The filtered image.
    Source:
        https://airccj.org/CSCP/vol7/csit76809.pdf pg.91
    """

    image = np.double(image)  # convert image to double precision
    num_rows, num_cols = image.shape  # get number of rows and columns in the image
    result = np.zeros((num_rows, num_cols))  # initialize an array of zeros with the same shape as the image

    flattened_frequency = np.round(
        frequency.flatten() * 100) / 100  # flatten the frequency array, round it to 2 decimal places, and store it in a new array
    unique_frequencies = np.unique(
        flattened_frequency[flattened_frequency > 0])  # get unique frequencies that are greater than 0
    sigma_x = 1 / unique_frequencies * kx  # calculate the sigma value for the x-axis using unique frequencies and kx
    sigma_y = 1 / unique_frequencies * ky  # calculate the sigma value for the y-axis using unique frequencies and ky
    window_size = int(np.round(
        3 * np.max([sigma_x, sigma_y])))  # determine the block size using the maximum value between sigma_x and sigma_y

    x, y = np.meshgrid(np.linspace(-window_size, window_size, 2 * window_size + 1),
                       # create a meshgrid with x and y coordinates
                       np.linspace(-window_size, window_size, 2 * window_size + 1))
    reference_filter = np.exp(-(((x ** 2) / (sigma_x ** 2)) + ((y ** 2) / (sigma_y ** 2)))) * np.cos(
        2 * np.pi * unique_frequencies[
            0] * x)  # create a reference filter using the x and y coordinates, sigmas, and the unique frequency value at index 0
    gabor_filter = np.array(
        [ndimage.rotate(reference_filter, -(degree * angle_increment + 90), reshape=False) for degree in
         range(0,
               180 // angle_increment)])  # create a gabor filter by rotating the reference filter at different angles and store it in an array

    max_orientation_index = np.round(
        180 / angle_increment)  # determine the maximum orientation index based on the angle increment
    orientation_index = np.round(
        orientation / np.pi * 180 / angle_increment)  # calculate the orientation index by converting radians to degrees and dividing by the angle increment
    orientation_index[
        orientation_index < 1] += max_orientation_index  # adjust the orientation index by adding the max orientation index for values less than 1
    orientation_index[
        orientation_index > max_orientation_index] -= max_orientation_index  # adjust the orientation index by
    # subtracting the max orientation index for values greater than the max orientation index

    valid_rows, valid_cols = np.where(frequency > 0)  # find the rows and columns with frequency values greater than 0
    final_index = np.where(
        (valid_rows > window_size) & (valid_rows < num_rows - window_size) & (valid_cols > window_size) & (
                valid_cols < num_cols - window_size))  # find the rows and columns that are within the valid range

    # apply the Gabor filter to the image blocks at the specified orientations and frequencies
    for k in range(np.shape(final_index)[1]):
        row = valid_rows[final_index[0][k]]
        col = valid_cols[final_index[0][k]]
        img_block = image[row - window_size:row + window_size + 1, col - window_size:col + window_size + 1]
        result[row][col] = np.sum(img_block * gabor_filter[int(orientation_index[row // 16][col // 16]) - 1])

    gabor_image = 255 - np.array((result < 0) * 255).astype(np.uint8)
    return gabor_image


def skeletonize(image):
    """
    Applies skeletonization to the input image, returning the skeletonized image.

    Args:
        image (ndarray): Input image to be skeletonized. Must be a grayscale image.

    Returns:
        ndarray: Skeletonized image.

    Raises:
        TypeError: If the input image is not a grayscale ndarray.

    """
    if len(image.shape) != 2:
        raise TypeError("Input image must be a grayscale ndarray.")

    # Convert black pixels to 1.0 and white pixels to 0.0
    binary_image = np.zeros_like(image)
    binary_image[image == 0] = 1.0

    # Skeletonize binary image
    skeleton = skelt(binary_image)

    # Convert skeleton back to original pixel values
    output = np.zeros_like(image)
    output[skeleton] = 255

    # Invert the image
    output = cv.bitwise_not(output)

    return output


def minutiae_at(pixels, row, col):
    """
    Determines the type of minutiae at the given pixel location in a binary image.

    Args:
        pixels (ndarray): Binary image represented as a 2D ndarray of integers (0 or 1).
        row (int): Row index of the pixel to be analyzed.
        col (int): Column index of the pixel to be analyzed.

    Returns:
        str: The type of minutiae at the given pixel location. Possible values are "ending", "bifurcation", and "none".

    """
    # check if the pixel value is 1
    if pixels[row][col] == 1:
        # define a list of 8-neighbor coordinates for the given pixel
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        # get the values of 8-neighbor pixels for the given pixel
        neighbor_values = [pixels[row + x][col + y] for x, y in neighbors]
        # calculate the number of times the values of 8-neighbor pixels change from 1 to 0 or vice versa
        num_crossings = sum(
            abs(neighbor_values[k] - neighbor_values[k + 1]) for k in range(len(neighbor_values) - 1)) // 2
        # if there is only one change in values among the 8-neighbor pixels, return "ending"
        if num_crossings == 1:
            return "ending"
        # if there are three changes in values among the 8-neighbor pixels, return "bifurcation"
        if num_crossings == 3:
            return "bifurcation"
    # if the pixel value is not 1, return "none"
    return "none"


def calculate_minutes(im, kernel_size=3):
    """
     Calculates the number of minutes in the fingerprint image.

     Parameters:
     im: ndarray
         The grayscale image of the fingerprint.
     kernel_size: int, optional (default=3)
         The size of the kernel used for smoothing the image.

     Returns:
     ndarray
         The image showing the number of minutes at each pixel.
     """
    # Convert the image into a binary image with 0's and 1's using a threshold of 10
    binary_image = (im < 10).astype(np.int8)

    # Create a new image with the same size as the original image, filled with zeros and float data type
    minutes_problems_image = np.zeros_like(im, dtype=np.float_)

    # Loop over every pixel in the image except for the edges of the kernel
    for col in range(1, im.shape[1] - kernel_size // 2):
        for row in range(1, im.shape[0] - kernel_size // 2):
            # Determine whether the current pixel is a bifurcation or an ending using the 'minutiae_at' function
            minutiae = minutiae_at(binary_image, row, col)

            # Add 1.0 to the 'minutes_problems_image' at the current pixel if the minutiae is a bifurcation
            if minutiae == "bifurcation":
                minutes_problems_image[row, col] += 1.0

            # Add 2.5 to the 'minutes_problems_image' at the current pixel if the minutiae is an ending
            if minutiae == "ending":
                minutes_problems_image[row, col] += 2.5

    # Return the image with the minutiae values
    return minutes_problems_image


def count_lines(image):
    """
    Counts the number of lines in an image along the main and secondary diagonals.

    Parameters:
    image (numpy.ndarray): Input image.

    Returns:
    tuple: A tuple containing the number of lines and the name of the diagonal with more lines.
    """
    # Creates a 3x3 kernel filled with ones.
    kernel = np.ones((3, 3), dtype=np.uint8)
    # Erodes the input image using the kernel created above.
    eroded_image = cv.erode(image, kernel)
    # Thresholds the eroded image to create a binary image.
    thresholded_image = eroded_image.copy()
    thresholded_image[eroded_image < 127] = 0
    thresholded_image[eroded_image >= 127] = 255
    # Gets the number of rows and columns in the image.
    rows, cols = image.shape
    # Initializes variables to count lines along the main diagonal.
    is_white = True
    main_diagonal = 0
    # Loops over the rows of the image.
    for i in range(image.shape[0]):
        # If a black pixel is encountered along the main diagonal, the line count is incremented.
        if thresholded_image[i][i] == 0 and is_white:
            main_diagonal += 1
            is_white = False
        # If a white pixel is encountered along the main diagonal, the next line count will be for a different line.
        if thresholded_image[i][i] == 255:
            is_white = True

    # Initializes variables to count lines along the secondary diagonal.
    is_white = True
    secondary_diagonal = 0
    # Loops over the rows of the image again.
    for i in range(image.shape[0]):
        # If a black pixel is encountered along the secondary diagonal, the line count is incremented.
        if thresholded_image[rows - i - 1][i] == 0 and is_white:
            secondary_diagonal += 1
            is_white = False
        # If a white pixel is encountered along the secondary diagonal, the next line count will be for a different line.
        if thresholded_image[rows - i - 1][i] == 255:
            is_white = True

    # Compares the line counts along the main diagonal and secondary diagonal and returns the larger one along with its name.
    if main_diagonal <= secondary_diagonal:
        return secondary_diagonal, "secondary_diagonal"
    else:
        return main_diagonal, "main_diagonal"


def draw_diagonal(image, start_row, start_col, end_row, end_col, diagonal_number, window_size, y, x, text_color):
    """
    Draws a line on the image between the starting and ending coordinates, along with the diagonal number and the text "ridges"
    above and below the line.

    Args:
        image: A NumPy array representing the input image.
        start_row: An integer representing the starting row coordinate of the line.
        start_col: An integer representing the starting column coordinate of the line.
        end_row: An integer representing the ending row coordinate of the line.
        end_col: An integer representing the ending column coordinate of the line.
        diagonal_number: An integer representing the diagonal number associated with the line.
        window_size: An integer representing the size of a block in pixels.
        y: An integer representing the x-coordinate of the text.
        x: An integer representing the y-coordinate of the text.
        text_color: A tuple representing the color of the text.

    Returns:
        None
    """
    # Draw a line on the image using the starting and ending coordinates and a red color.
    cv.line(image, (start_col, start_row), (end_col, end_row), (0, 0, 255), 1)
    # Add the diagonal number as text to the image above the line using the provided position, font, color and scale.
    cv.putText(image, f' {diagonal_number}', (y, x - window_size), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    # Add the word "ridges" as text to the image below the line using the provided position, font, color and scale.
    cv.putText(image, "ridges", (y, x), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)


def calculate_singularities(thin_image, minutia_problems_image, window_size, mask, image):
    """
    Finds the block with the fewest number of minutia problems that is fully covered by the mask, calculates the
    diagonal number of the block, and draws it on a copy of the input image.

    Parameters:
        thin_image (numpy.ndarray): binary image containing the thinned fingerprint image
        minutia_problems_image (numpy.ndarray): image containing the number of minutia problems for each pixel in the thin_image
        window_size (int): size of the blocks to search in the thin_image
        mask (numpy.ndarray): binary image indicating the region of interest in the thin_image
        image (numpy.ndarray): grayscale image containing the original fingerprint

    Returns:
        tuple: resulting image with diagonal and block outlined (numpy.ndarray), diagonal number (int), number of minutia
               problems in the best block (int)
    """
    # Create a copy of the input image as an RGB image
    result_on_image = cv.cvtColor(image.copy(), cv.COLOR_GRAY2RGB)

    # Initialize variables to store information about the best block found
    Less_block_with_minutia_problems = None
    min_minutia_problems = (window_size * 8) ** 2
    diagonal_number = 0

    # Get the dimensions of the thin_image
    rows, cols = thin_image.shape

    # Iterate over all blocks of size 120x120 in the thin_image
    for i in range(1, cols - 9):
        for j in range(1, rows - 9):
            # Check if the current block is fully covered by the mask
            mask_flag = np.sum(mask[i * window_size:(i + 8) * window_size, j * window_size:(j + 8) * window_size])
            if mask_flag == (window_size * 8) ** 2:
                # Calculate the number of minutia problems in the current block
                minutia_problems = np.sum(
                    minutia_problems_image[i * window_size:(i + 8) * window_size,
                    j * window_size:(j + 8) * window_size])
                # Check if the current block has fewer minutia problems than the previous best block
                if minutia_problems <= min_minutia_problems:
                    min_minutia_problems = minutia_problems
                    Less_block_with_minutia_problems = [i * window_size, (i + 8) * window_size, j * window_size,
                                                        (j + 8) * window_size]

    # If a Less_block_with_minutia_problems was found, calculate the diagonal number of the block and draw it on the result_on_image
    if Less_block_with_minutia_problems:
        # Create a copy of the best block from the thin_image
        copy_im = (thin_image[Less_block_with_minutia_problems[0]: Less_block_with_minutia_problems[1],
                   Less_block_with_minutia_problems[2]: Less_block_with_minutia_problems[3]]).copy()
        # Calculate the diagonal number of the best block
        diagonal_number, type_diag = count_lines(copy_im)
        # Calculate the center coordinates of the best block
        y = (Less_block_with_minutia_problems[3] - Less_block_with_minutia_problems[2]) // 2 + \
            Less_block_with_minutia_problems[2]
        # Draw the diagonal on the result_on_image based on the type of diagonal found
        if type_diag == "main_diagonal":
            x = (Less_block_with_minutia_problems[1] - Less_block_with_minutia_problems[0]) // 3 + \
                Less_block_with_minutia_problems[0]
            draw_diagonal(result_on_image, Less_block_with_minutia_problems[0], Less_block_with_minutia_problems[2],
                          Less_block_with_minutia_problems[1], Less_block_with_minutia_problems[3], diagonal_number,
                          window_size, y,
                          x, (0, 255, 255))
        elif type_diag == "secondary_diagonal":
            x = 2 * (Less_block_with_minutia_problems[1] - Less_block_with_minutia_problems[0]) // 3 + \
                Less_block_with_minutia_problems[0]
            draw_diagonal(result_on_image, Less_block_with_minutia_problems[0], Less_block_with_minutia_problems[3],
                          Less_block_with_minutia_problems[1], Less_block_with_minutia_problems[2], diagonal_number,
                          window_size, y,
                          x, (0, 255, 255))
        # Draw a rectangle around the best block on the result_on_image
        cv.rectangle(result_on_image, (Less_block_with_minutia_problems[2], Less_block_with_minutia_problems[0]),
                     (Less_block_with_minutia_problems[3], Less_block_with_minutia_problems[1]), (0, 0, 255), 1)
    # Return the result_on_image along with the diagonal number and the number of minutia problems for the best block
    return result_on_image, diagonal_number, min_minutia_problems


def gender_fingerprint_pipeline(image, threshold_level=0.3, window_size=16):
    """
    This function processes an input fingerprint image to extract its gender-specific features.

    Args:
        image: A 2D grayscale image of a fingerprint.
        threshold_level (float): The threshold value used to segment the fingerprint image.
            Default value is 0.3.
        window_size :

    Returns:
        singularities_img: A 2D image indicating the location of singularities (bifurcations and endings) in the
            processed fingerprint image.
        diagonal_number (int): The number of diagonals in the processed fingerprint image.
        min_minutia_problems (int): The minimum number of minutia problems for any block in the processed
            fingerprint image.
    """
    # Crop the bottom 32 rows of the input image
    img_cut = image[:-32, :]
    # Set the block size to 16 pixels

    # Normalize the input image to have zero mean and unit variance
    normalized_img = normalize_image(img_cut)
    # cv.imshow("img_cut", img_cut)

    # Segment the normalized image into blocks and return the segmented image,
    # the normalized image, and the mask indicating which blocks are valid
    segmented_img, norm_image, mask = segment_image(normalized_img, threshold=threshold_level)

    # Calculate the ridge angles of the segmented image using a Gaussian filter
    angles = calculate_angles(segmented_img)
    try:
        # Estimate the ridge frequency of the normalized image using the mask,
        # the calculated angles, and a Gabor filter
        median_freq = estimate_ridge_frequency(norm_image, mask, angles)
        # Apply a Gabor filter to the normalized image using the calculated angles
        # and the estimated ridge frequency
        gabor_img = gabor_ridge_filter(norm_image, angles, median_freq)
    except Exception:
        # If there is an error in the ridge frequency estimation or filtering,
        # print the error and return the original cropped image, a diagonal number of 0,
        # and a minutia problem count of (window_size * 8) ** 2
        return img_cut, 0, (window_size * 8) ** 2
    # Thin the filtered image using a skeletonization algorithm
    thinned_image = skeletonize(gabor_img)
    # Calculate the number of minutia problems for each block in the thinned image
    minutia_problems_image = calculate_minutes(thinned_image)
    # Set the block size to 15 pixels
    window_size = 15
    # Calculate the singularities (bifurcations and endings) in the thinned image
    # and return the resulting image, the diagonal number, and the minimum number of
    # minutia problems for any block
    singularities_img, diagonal_number, min_minutia_problems = calculate_singularities(thinned_image,
                                                                                       minutia_problems_image,
                                                                                       window_size, mask,
                                                                                       image)
    return singularities_img, diagonal_number, min_minutia_problems


if __name__ == "__main__":
    # Define input and output directories
    input_dir = './input_images'
    output_dir = './output_images/'
    # Create the output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    threshold_max_level = 0.45
    threshold_min_level = 0.2
    threshold_levels = [0.45, 0.35, 0.3, 0.25, 0.2]
    step = -0.05
    window_size = 16
    # Iterate over each image in the input directory
    for imgName in os.listdir(input_dir):
        print("Process on image" + imgName)
        # Load the image in grayscale
        img_dir = os.path.join(input_dir, imgName)
        img = cv.imread(img_dir)
        img_gry = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        # Find the best segmentation threshold using a loop that gradually reduces the threshold
        the_best_result = None
        min_minutes_problems = (window_size * 8) ** 2
        for th_level in threshold_levels:
            results, diagonal_number, number_minutes_problems = gender_fingerprint_pipeline(img_gry,
                                                                                            threshold_level=th_level,
                                                                                            window_size=window_size)
            if diagonal_number != 0 and number_minutes_problems < min_minutes_problems:
                the_best_result = results, diagonal_number, number_minutes_problems, th_level
                min_minutes_problems = number_minutes_problems
        # If a valid segmentation threshold was found, save the segmented fingerprint to the output directory
        if the_best_result:
            results, diagonal_number, number_minutes_problems, th_level = the_best_result
            print(f'the image with name: {imgName} has count diag = {diagonal_number}')
            cv.imwrite(output_dir + imgName, results)
            print("The process was successful with threshold level = " + str(th_level))
        # If a valid segmentation threshold was not found, save the original grayscale image to the output directory
        else:
            cv.imwrite(output_dir + imgName, img)
            print("Process failure")
