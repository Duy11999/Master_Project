import cv2
import numpy as np
import random
from collections import defaultdict
# Find the real edges by sort the most value in the edges according to x and y coordinates


# Load the images
img = cv2.imread("/home/d8/Work/Master_Project/Images/s1/P1h1.jpg")
img1 = cv2.imread("/home/d8/Work/Master_Project/Images/wt11.jpg")

# Convert the image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds for the color range you want to segment
l_b = np.array([0, 0, 138])  # Lower bound
u_b = np.array([255, 255, 255])  # Upper bound

# Create a mask using the defined lower and upper bounds
mask = cv2.inRange(hsv, l_b, u_b)

# Find connected components in the mask
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)


def get_adjacent_white_pixels(object_mask):
    adjacent_white_pixels = []
    for y in range(1, object_mask.shape[0] - 1):
        for x in range(1, object_mask.shape[1] - 1):
            if object_mask[y, x] == 255:
                if (object_mask[y - 1, x] == 0 or object_mask[y + 1, x] == 0 or
                        object_mask[y, x - 1] == 0 or object_mask[y, x + 1] == 0):
                    adjacent_white_pixels.append((x, y))
    return adjacent_white_pixels


def draw_lines_between_groups(img, group1, group2, same_coordinate='x', color=(0, 255, 255)):
    if group1 and group2:
        point1 = random.choice(group1)
        point2 = random.choice(group2)
        if same_coordinate == 'x':
            point2 = (point1[0], point2[1])
        else:
            point2 = (point2[0], point1[1])
        cv2.line(img, point1, point2, color, 2)
        # Calculate and return the length of the line
        length = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
        return point1, point2, length
    return None, None, None


# Process each object
for label in range(1, num_labels):  # Start from 1 to skip the background
    # Create a mask for the current object
    object_mask = np.zeros_like(mask)
    object_mask[labels == label] = 255

    # Get adjacent white pixels
    adjacent_white_pixels = get_adjacent_white_pixels(object_mask)

    # Sort the pixels by y coordinate
    adjacent_white_pixels_sorted = sorted(adjacent_white_pixels, key=lambda pixel: pixel[1], reverse=True)
    adjacent_white_pixels_sorted_smallest = sorted(adjacent_white_pixels, key=lambda pixel: pixel[1])

    # Sort the pixels by x coordinate
    adjacent_white_pixels_sorted_x = sorted(adjacent_white_pixels, key=lambda pixel: pixel[0])
    adjacent_white_pixels_sorted_x_smallest = sorted(adjacent_white_pixels, key=lambda pixel: pixel[0], reverse=True)

    # Select the first 300 pixels
    selected_pixels = adjacent_white_pixels_sorted[:300]
    selected_pixels1 = adjacent_white_pixels_sorted_smallest[:300]
    selected_pixels2 = adjacent_white_pixels_sorted_x[:150]
    selected_pixels3 = adjacent_white_pixels_sorted_x_smallest[:150]

    # Count occurrences of each y-coordinate
    y_counts = defaultdict(list)
    for (x, y) in selected_pixels:
        y_counts[y].append((x, y))

    y_counts1 = defaultdict(list)
    for (x, y) in selected_pixels1:
        y_counts1[y].append((x, y))

    x_counts = defaultdict(list)
    for (x, y) in selected_pixels2:
        x_counts[x].append((x, y))

    x_counts1 = defaultdict(list)
    for (x, y) in selected_pixels3:
        x_counts1[x].append((x, y))

    # Find the y-coordinate with the most occurrences
    most_common_y = max(y_counts, key=lambda y: len(y_counts[y]))
    most_common_group = y_counts[most_common_y]

    most_common_y1 = max(y_counts1, key=lambda y: len(y_counts1[y]))
    most_common_group1 = y_counts1[most_common_y1]

    most_common_x = max(list(x_counts.keys()), key=lambda x: len(x_counts[x]))
    most_common_group2 = x_counts[most_common_x]

    most_common_x1 = max(list(x_counts1.keys()), key=lambda x: len(x_counts1[x]))
    most_common_group3 = x_counts1[most_common_x1]

    # Draw lines between randomly selected points in the groups and print the lengths
    print(f"Object {label}:")
    point1, point2, length = draw_lines_between_groups(img, most_common_group, most_common_group1, same_coordinate='x',
                                                       color=(0, 0, 255))  # Red
    if point1 and point2:
        print(f"Length of the line from {point1} to {point2}: {length:.2f} pixels")

    point1, point2, length = draw_lines_between_groups(img, most_common_group2, most_common_group3, same_coordinate='y',
                                                       color=(0, 255, 0))  # Blue
    if point1 and point2:
        print(f"Length of the line from {point1} to {point2}: {length:.2f} pixels")

    # Draw the same lines on img1 without printing the lengths again
    draw_lines_between_groups(img1, most_common_group, most_common_group1, same_coordinate='x',
                              color=(0, 0, 255))  # Red
    draw_lines_between_groups(img1, most_common_group2, most_common_group3, same_coordinate='y',
                              color=(0, 255, 0))  # Blue

    # Mark the selected pixels on the original image
    for (x, y) in selected_pixels:
        img[y, x] = [255, 0, 255]  # Mark the pixel in red

    for (x, y) in selected_pixels1:
        img[y, x] = [255, 0, 255]  # Mark the pixel in yellow

# Display the images with marked pixels and the line
cv2.namedWindow("Marked Image", cv2.WINDOW_NORMAL)
cv2.imshow("Marked Image", img)
cv2.namedWindow("Marked Image1", cv2.WINDOW_NORMAL)
cv2.imshow("Marked Image1", img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
