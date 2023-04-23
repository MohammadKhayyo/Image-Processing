Name: Mohammad Khayyo
============================================================================================================
General Hough Transform
This Python script uses the General Hough Transform (GHT) to detect shapes in an image and plot the locations of the shapes. The GHT is a technique for detecting the presence of a predetermined shape in an image, regardless of its orientation or scale.

How it works
The GHT works by constructing a reference table (R-table) for the shape to be detected. The R-table maps the gradient orientations of the edge points in the shape to the corresponding positions of the points relative to a reference point in the shape.

To detect the shape in an image, the GHT first applies the Canny edge detection algorithm to the image to identify the edge points. It then calculates the gradient orientations of the edge points and accumulates the corresponding positions in the R-table for each gradient orientation. The resulting accumulator image shows the locations of the shape in the image, with higher values indicating a higher likelihood of the shape being present at that location.
The Program choose image and finds letters M and K.
I use General Hough Transform to find the lines and then edges.
============================================================================================================
Functions:
1) gradient_orientation(image) : """Calculate the gradient orientation for edge point in the image"""
2) build_r_table(image, origin): """Build the R-table from the given shape image and a reference point"""
3) accumulate_gradients(r_table, grayImage): """Perform a General Hough Transform with the given image and R-table"""
4)general_hough(query_image): """ Uses a accumulate_gradients to detect shapes in an image and create nice output """
