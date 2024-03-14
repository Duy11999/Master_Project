# python3 length_detection.py
import cv2
import numpy as np

def detect_aruco_corners(img, aruco_type):
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

    print("[INFO] Loading image...")
    image = cv2.imread(img)

    if ARUCO_DICT.get(aruco_type, None) is None:
        print("[INFO] ArUCo tag of '{}' is not supported".format(aruco_type))
        return None

    print("[INFO] Detecting '{}' tags...".format(aruco_type))
    arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
    arucoParams = cv2.aruco.DetectorParameters()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

    if corners is not None and len(corners) >= 2:
        return corners

    return None

def calculate_distance_between_markers(marker1_corners, marker2_corners):
    # Calculate the centroids of each marker
    center1 = np.mean(marker1_corners, axis=0).astype(int)  # Convert to integer
    center2 = np.mean(marker2_corners, axis=0).astype(int)  # Convert to integer
    # Calculate the Euclidean distance between the centroids
    distance = np.linalg.norm(center1 - center2)
    return distance

# Initialize sum of distances
total_distance_mm = 0

# Example usage
img_paths = ["../Images/3.jpg", "../Images/2.jpg", "../Images/1.jpg", "../Images/4.jpg","../Images/5.jpg"]
aruco_type = "DICT_APRILTAG_36h11"
tag_size = 150

for img_path in img_paths:
    corners = detect_aruco_corners(img_path, aruco_type)

    if corners is not None and len(corners) >= 2:
        # Extract corners of the two markers
        marker1_corners, marker2_corners = corners[:2]
        # Convert corners to a format suitable for drawContours
        marker1_corners = marker1_corners.reshape(-1, 1, 2).astype(np.int32)
        marker2_corners = marker2_corners.reshape(-1, 1, 2).astype(np.int32)
        # Calculate distance between markers
        distance = calculate_distance_between_markers(marker1_corners, marker2_corners)
        print("Distance between markers in pixel:", distance)

        # Calculate perimeter of each box
        perimeter_marker1 = cv2.arcLength(marker1_corners, closed=True)
        perimeter_marker2 = cv2.arcLength(marker2_corners, closed=True)
        print("Perimeter of marker 1:", perimeter_marker1)
        print("Perimeter of marker 2:", perimeter_marker2)
        scaling_factor1 = (tag_size*4)/perimeter_marker1
        scaling_factor2 = (tag_size*4)/perimeter_marker2
        a = (scaling_factor2+scaling_factor1)/2
        distance_mm = distance*a
        print("Distance between markers in mm:", distance_mm, "mm")

        # Add distance to the total
        total_distance_mm += distance_mm

        # Draw contours around the markers
        image = cv2.imread(img_path)
        image = cv2.drawContours(image, [marker1_corners], -1, (0, 255, 0), 2)
        image = cv2.drawContours(image, [marker2_corners], -1, (0, 255, 0), 2)

        # Draw line between markers
        center1 = np.mean(marker1_corners, axis=0).astype(int)
        center2 = np.mean(marker2_corners, axis=0).astype(int)
        center1 = center1.reshape((1, 2))
        center2 = center2.reshape((1, 2))
        image = cv2.line(image, tuple(center1[0]), tuple(center2[0]), (0, 0, 255), 2)

        # Calculate midpoint for text position
        text_position = ((center1[0][0] + center2[0][0]) // 2, (center1[0][1] + center2[0][1]) // 2)

        # Display distance as blue text near the line
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, f"Distance: {distance_mm:.2f} mm", text_position, font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.namedWindow("Detected ArUco Markers", cv2.WINDOW_NORMAL)
        cv2.imshow("Detected ArUco Markers", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Two markers not detected.")

# Print the sum of distances in all images
print("Total distance between markers in all images:", total_distance_mm, "mm")

