import cv2
import numpy as np

def detect_sky_color(hsv_image):
    # Crop the image to the upper half, because we assume the sky is always on the upper half of the image
    height = hsv_image.shape[0]
    upper_half_image = hsv_image[:height//2, :]

    # Define color ranges in HSV
    blue_lower = np.array([46, 17, 148], np.uint8)
    blue_upper = np.array([154, 185, 249], np.uint8)
    orange_lower = np.array([10, 100, 100], np.uint8)
    orange_upper = np.array([25, 183, 254], np.uint8)
    pale_lower = np.array([0, 0, 129], np.uint8)
    pale_upper = np.array([171, 64, 225], np.uint8)

    # Create masks for colors
    blue_mask = cv2.inRange(upper_half_image, blue_lower, blue_upper)
    orange_mask = cv2.inRange(upper_half_image, orange_lower, orange_upper)
    pale_mask = cv2.inRange(upper_half_image, pale_lower, pale_upper)

    # Calculate the percentage of cropped image covered by each color
    blue_percentage = np.sum(blue_mask > 0) / (upper_half_image.shape[0] * upper_half_image.shape[1]) * 100
    orange_percentage = np.sum(orange_mask > 0) / (upper_half_image.shape[0] * upper_half_image.shape[1]) * 100
    pale_percentage = np.sum(pale_mask > 0) / (upper_half_image.shape[0] * upper_half_image.shape[1]) * 100

    # Determine the predominant color in the upper half
    max_color = max(blue_percentage, orange_percentage, pale_percentage)
    if max_color == blue_percentage:
        return blue_lower, blue_upper
    elif max_color == orange_percentage:
        return orange_lower, orange_upper
    else:
        return pale_lower, pale_upper
    
    
