import cv2
import numpy as np
# Position 2 markers

# Test Position
def detect_aruco_corners(img_paths, aruco_type):
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

    for img_path in img_paths:
        print("[INFO] Loading image...")
        image = cv2.imread(img_path)

        if image.shape[0] > image.shape[1]:
            # Rotate the image 90 degrees counterclockwise
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if image is None:
            print("[ERROR] Image not found or cannot be loaded.")
            continue

        if ARUCO_DICT.get(aruco_type, None) is None:
            print("[INFO] ArUCo tag of '{}' is not supported".format(aruco_type))
            continue

        print("[INFO] Detecting '{}' tags in {}...".format(aruco_type, img_path))
        arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[aruco_type])
        arucoParams = cv2.aruco.DetectorParameters()
        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
        print(corners)

        if ids is not None:
            # Find marker 7
            marker7_index = np.where(ids == 7)[0]
            if len(marker7_index) == 0:
                print("Marker 7 not found in {}.".format(img_path))
                continue

            marker7_index = marker7_index[0]
            marker7_corners = corners[marker7_index][0]
            marker7_centroid = np.mean(marker7_corners, axis=0)

            # Calculate relative positions of other markers based on marker 7
            relative_positions = {}
            for i, corner in enumerate(corners):
                if i != marker7_index:
                    marker_corners = corner[0]
                    marker_centroid = np.mean(marker_corners, axis=0)
                    relative_position = marker_centroid - marker7_centroid
                    relative_positions[ids[i][0]] = relative_position

            print("Relative positions based on marker 7:", relative_positions)

            # Draw the detected markers and their centroids
            for i, corner in enumerate(corners):
                marker_corners = corner[0]
                marker_centroid = np.mean(marker_corners, axis=0)

                # Draw marker boundaries
                cv2.polylines(image, [marker_corners.astype(np.int32)], True, (0, 255, 0), 2)
                # Draw marker centroid
                cv2.circle(image, tuple(marker_centroid.astype(int)), 5, (0, 0, 255), -1)
                # Draw marker ID
                cv2.putText(image, f"ID: {ids[i][0]}", tuple(marker_centroid.astype(int) + np.array([0, -10])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Display the image
            cv2.namedWindow("Detected ArUco Markers", cv2.WINDOW_NORMAL)
            cv2.imshow("Detected ArUco Markers", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No markers detected in {}.".format(img_path))


# Example usage
img_paths = ["../Images/1.jpg", "../Images/4-av.jpg"]
aruco_type = "DICT_APRILTAG_36h11"

detect_aruco_corners(img_paths, aruco_type)