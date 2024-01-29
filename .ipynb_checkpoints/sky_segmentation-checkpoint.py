import cv2
import numpy as np
from matplotlib import pyplot as plt
from helper_sky_color_detector import detect_sky_color

# Read the image
image = cv2.imread('pale1.jpeg')

# Convert to HSV image
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Determine HSV range based on helper function
(hsv_lower, hsv_upper) = detect_sky_color(hsv)

# Use hsv_lower and hsv_upper to create a mask, which isolates the sky region
mask_initial = cv2.inRange(hsv, hsv_lower, hsv_upper)

# Apply morphological operations to fine-tune the mask
kernel = np.ones((3,3), np.uint8)
mask_fine_tuned = cv2.erode(mask_initial, kernel, iterations=1)
mask_fine_tuned = cv2.dilate(mask_fine_tuned, kernel, iterations=1)


# Perform connected component analysis
num_labels, labels_im = cv2.connectedComponents(mask_fine_tuned)

# Create an array to hold the size of each component
sizes = np.bincount(labels_im.flatten())

# Set the size of the background (label 0) to zero
sizes[0] = 0

# Find the largest component
max_label = np.argmax(sizes)

# Create a mask with only the largest component
sky_mask = np.zeros_like(mask_fine_tuned)
sky_mask[labels_im == max_label] = 255

# Convert BGR image to RGB for matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Plotting
plt.figure(figsize=(18, 6))

# Original Image
plt.subplot(1, 4, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

# Initial Mask
plt.subplot(1, 4, 2)
plt.imshow(mask_initial, cmap='gray')
plt.title('Initial Sky Mask')
plt.axis('off')

# Fine-tuned Mask
plt.subplot(1, 4, 3)
plt.imshow(mask_fine_tuned, cmap='gray')
plt.title('Fine-tuned Sky Mask')
plt.axis('off')

# Sky Mask
plt.subplot(1, 4, 4)
plt.imshow(sky_mask, cmap='gray')
plt.title('Sky Mask')
plt.axis('off')

plt.show()
