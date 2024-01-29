import cv2
import numpy as np
import gradio as gr

# Helper function to detect sky condition and get the HSV range
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


# Main function to process image and display sky masks
def sky_segmentation(uploaded_image):
    # Read the image
    image = cv2.imread(uploaded_image)

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
    
    return sky_mask


# Create a Gradio demo
demo = gr.Interface(
    fn=sky_segmentation, 
    inputs= gr.Image(type='filepath'), 
    outputs="image",
    title='Sky Segmentation', 
    description='Sky Pixel Identification in Images using Traditional Computer Vision Techniques',
    examples=["blue1.jpeg", "pink1.jpeg", "pale1.jpeg"]
)

demo.launch()
