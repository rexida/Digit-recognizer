# Handwritten Digit Recognizer

A simple handwritten digit recognition project built with Python, Tkinter, NumPy, Pillow, Matplotlib, and scikit-learn.

# How it works

The program uses the `load_digits()` dataset from scikit-learn to train a Support Vector Classifier (SVM) to recognize handwritten digits from 0 to 9.

The dataset contains 1,797 examples of handwritten digits. Each digit is represented as an 8×8 grayscale image, giving the model 64 pixel features per image.

The dataset is split into training and testing data. The SVM learns patterns from the training examples and is then evaluated on examples it has not seen before. The model achieves about 98.6% accuracy on the test split used in this project.

# Recognizing a user drawing

When a digit is drawn on the Tkinter canvas, the program:

1. Stores the drawing as a grayscale PIL image.
2. Finds the bounding box containing the drawing.
3. Crops away the unused black space.
4. Makes the cropped image square so the digit is not stretched.
5. Resizes it to 8×8 pixels to match the training data.
6. Converts the pixel values from the 0–255 range to the 0–16 range used by the dataset.
7. Flattens the 8×8 image into 64 features.
8. Sends those 64 features to the trained SVM.
9. Prints the predicted digit.

# Technologies

* Python
* Tkinter
* NumPy
* Pillow
* Matplotlib
* scikit-learn

# Installation

Install the required libraries with:

py -m pip install -r requirements.txt

# Run

py main.py


Draw a digit on the canvas and press **Predict**.

# Notes

The model's test accuracy is measured on the scikit-learn dataset, while handwritten digits drawn by a user can look different from the training examples. Because of this, real-world predictions may be less accurate than the reported test accuracy.
