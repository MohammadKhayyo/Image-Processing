# General Hough Transform for Shape Detection

This repository contains a Python implementation of the General Hough Transform for shape detection. This algorithm is primarily used for object detection and recognition in an image.

## Description

The script performs the following tasks:

1. It reads an input image and applies the General Hough Transform to detect the presence and location of specific shapes, namely the letters 'M' and 'K'.
2. The algorithm calculates the gradient orientation of the edges of the image.
3. It then builds an R-table, a shape descriptor used for the General Hough Transform, for each shape in the image.
4. Accumulator matrix is calculated using the R-tables which gives the possible locations of the shape in the image.
5. Finally, it visualizes the detected shapes by highlighting the areas in the input image where the specific shapes (i.e., 'M' and 'K') have been detected.

## Dependencies

To run this script, you need the following Python libraries:

- NumPy
- Collections
- SciKit-Image (skimage)
- SciPy
- Tkinter
- Matplotlib

You can install these dependencies using pip:
```
pip install numpy collections scikit-image scipy matplotlib
```

## How to Run

1. Clone this repository to your local machine.
2. Run the script using Python:
```
python <name_of_the_script.py>
```
The script will open a file dialog for you to select the image you want to detect shapes on.

## Code Explanation

1. The script reads an image from the path selected through the file dialog.

```python
file_path = filedialog.askopenfilename(filetypes=(("image type", "*.jpg"), ("image type", "*.PNG")))
image = plt.imread(file_path)
image = image[:, :, 0]
```

2. The General Hough Transform is applied to the image to detect shapes.
```python
general_hough(image)
```

3. The 'general_hough' function applies the General Hough Transform to the image, attempting to detect the presence and location of the shapes 'M' and 'K'. The areas in the image where the shapes are detected are highlighted.

