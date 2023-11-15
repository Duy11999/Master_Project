# python3 ShearPocket_Measurement.py
# Shear Pocket dimension detection
import cv2
import numpy as np
from object_detector import *

def euclidean_distance(point1, point2):
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

# Load Aruco detector
parameters = cv2.aruco.DetectorParameters()
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

# Load Object Detector
detector = HomogeneousBgDetector()

# Load the image
img1 = cv2.imread("../Images/Pocket2-Trans.jpg")

# Get Aruco marker
corners, _, _ = cv2.aruco.detectMarkers(img1, aruco_dict, parameters=parameters)

# Aruco Perimeter
aruco_perimeter = cv2.arcLength(corners[0], True)

# Pixel to mm ratio
pixel_cm_ratio = aruco_perimeter / 600
print(pixel_cm_ratio)

# Load the image
img = cv2.imread('../Images/P2.jpg')


# Convert the image to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the HSV range and create a mask
l_b = np.array([0, 0, 138])
u_b = np.array([255, 255, 255])
mask = cv2.inRange(hsv, l_b, u_b)

# Find contours in the mask
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort the contours by area in descending order
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]  # Get the two largest contours

# Iterate through the two largest contours found
for contour in contours:
    # Get the bounding rectangle for the current contour
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)  # Get four vertices of the rotated rectangle
    box_int = np.intp(box)  # Convert vertices to integers
    cv2.drawContours(img, [box_int], 0, (0, 255, 0), 2)

    # Extract the top-left, top-right, bottom-right, bottom-left corners
    top_left, top_right, bottom_right, bottom_left = box

    # Calculate the width and height in cm based on your pixel-to-cm ratio
    width = euclidean_distance(top_left, top_right) / pixel_cm_ratio
    height = euclidean_distance(top_left, bottom_left) / pixel_cm_ratio

    # Printing the calculated width and height along with the corners
    print("Width:", width)
    print("Height:", height)
    print("Top-Left:", top_left)
    print("Top-Right:", top_right)
    print("Bottom-Left:", bottom_left)
    print("Bottom-Right:", bottom_right)
    print("-" * 30)

# Display the result
cv2.namedWindow("OriginalImg", cv2.WINDOW_NORMAL)
cv2.imshow("OriginalImg", img1)
cv2.namedWindow("result", cv2.WINDOW_NORMAL)
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
