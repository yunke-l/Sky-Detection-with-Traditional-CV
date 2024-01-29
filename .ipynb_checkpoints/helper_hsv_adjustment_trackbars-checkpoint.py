import cv2
import numpy as np

# Read the image
image = cv2.imread('bright-blue1.jpg')

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Defines a function nothing that does nothing and is used as a placeholder for the trackbar callback
def nothing(x):
    pass

# Create a window to display the image and the trackbars
cv2.namedWindow('image')

# Create trackbars for color change
cv2.createTrackbar('LowerH', 'image', 0, 179, nothing)
cv2.createTrackbar('LowerS', 'image', 0, 255, nothing)
cv2.createTrackbar('LowerV', 'image', 0, 255, nothing)
cv2.createTrackbar('UpperH', 'image', 179, 179, nothing)
cv2.createTrackbar('UpperS', 'image', 255, 255, nothing)
cv2.createTrackbar('UpperV', 'image', 255, 255, nothing)

while True:
    # Get current positions of the trackbars
    lower_h = cv2.getTrackbarPos('LowerH', 'image')
    lower_s = cv2.getTrackbarPos('LowerS', 'image')
    lower_v = cv2.getTrackbarPos('LowerV', 'image')
    upper_h = cv2.getTrackbarPos('UpperH', 'image')
    upper_s = cv2.getTrackbarPos('UpperS', 'image')
    upper_v = cv2.getTrackbarPos('UpperV', 'image')

    # Define the HSV range
    lower_blue = np.array([lower_h, lower_s, lower_v])
    upper_blue = np.array([upper_h, upper_s, upper_v])

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask=mask)

    # Display images
    cv2.imshow('image', res)

    # Break loop on ESC key press
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()