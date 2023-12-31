# python3 Calculate_dimension.py
import cv2
from object_detector import *
import numpy as np
import imutils


#Load Object Detector
detector = HomogeneousBgDetector()

def detect_aruco_corners(img, aruco_type):
    # Define names of each possible ArUco tag OpenCV supports
    ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
    }

    # Load the input image from disk and resize it
    print("[INFO] Loading image...")
    image = cv2.imread(img)
    image = imutils.resize(image, width=600)

    # Verify that the supplied ArUCo tag exists and is supported by OpenCV
    if ARUCO_DICT.get(aruco_type, None) is None:
        print("[INFO] ArUCo tag of '{}' is not supported".format(aruco_type))
        return None

    # Load the ArUCo dictionary, grab the ArUCo parameters, and detect the markers
    print("[INFO] Detecting '{}' tags...".format(aruco_type))
    arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
    arucoParams = cv2.aruco.DetectorParameters()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

    # Verify *at least* one ArUco marker was detected
    if len(corners) > 0:
        # Flatten the ArUco IDs list
        ids = ids.flatten()

        # Display all detected marker IDs
        print("Detected marker IDs:")
        for markerID in ids:
            print(markerID)

        # Choose the marker to use based on its ID
        chosen_id = int(input("Enter the ID of the marker to use: "))

        # Find the chosen marker index
        chosen_marker_index = -1
        for i, markerID in enumerate(ids):
            if markerID == chosen_id:
                chosen_marker_index = i
                break

        if chosen_marker_index != -1:
            # Retrieve the chosen marker's corners
            chosen_marker_corners = corners[chosen_marker_index][0]

            # Convert the corner points to the expected data type
            chosen_marker_corners = chosen_marker_corners.astype(int)

            # Reshape the corners to match the expected format
            chosen_marker_corners = chosen_marker_corners.reshape((-1, 1, 2))

            # Return the corners of the chosen marker
            return np.array(chosen_marker_corners)

    # If no corners are found or the chosen marker is not detected, return None
    return None

# Example usage
img = cv2.imread('../Images/a(1).jpg')
aruco_type = "DICT_5X5_50"

chosen_marker_corners = detect_aruco_corners(img, aruco_type)
if chosen_marker_corners is not None:
    # Convert chosen_marker_corners to a numpy array
    chosen_marker_corners = np.array(chosen_marker_corners)

    # Calculate the perimeter of the chosen marker
    aruco_perimeter = cv2.arcLength(chosen_marker_corners, True)

    # Calculate the pixel-to-millimeter ratio
    pixel_cm_ratio = aruco_perimeter / 400

    print(pixel_cm_ratio)
else:
    print("No marker detected or invalid marker ID chosen.")

img = cv2.imread('../Images/a(1).jpg')
#Draw Polygon around marker
int_conners = np.intp(chosen_marker_corners)
cv2.polylines(img,int_conners,True, (200,255,0),3)

contours = detector.detect_objects(img)
cv2.namedWindow('image1', cv2.WINDOW_NORMAL)
cv2.imshow("image1",img)


#Draw objects boundaries:
for cnt in contours:
    # Get rect
    rect = cv2.minAreaRect(cnt)
    (x,y), (w,h), angle = rect # width and height in pixel of object on image

    #Get Width and Height of Objects by applying Ration pixel to cm
    object_width = w/pixel_cm_ratio
    object_height = h/pixel_cm_ratio

    #Display rectangle
    box = cv2.boxPoints(rect)
    box = np.intp(box) # array contains 4 coordinates of 4 corners
    # Draw polygons
    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -2)
    cv2.polylines(img, [box], True, (255, 126, 0), 3)
    cv2.namedWindow('img2', cv2.WINDOW_NORMAL)
    cv2.imshow("img2", img)
    print(object_height)
    print(object_width)

    cv2.putText(img, "Width {}".format(round(object_width,3)), (int(x),int(y)-20), cv2.FONT_HERSHEY_PLAIN,1,(255,216,235),2)
    cv2.putText(img, "Height {}".format(round(object_height,3)), (int(x),int(y)+20), cv2.FONT_HERSHEY_PLAIN,1,(255,216,235),2)

cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.imshow("Image", img)
cv2.waitKey(0)