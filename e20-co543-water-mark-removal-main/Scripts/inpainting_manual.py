import cv2 
import numpy as np


ref_point = []
cropping = False
temp_image = None  # Temporary image to display dynamic rectangle

def click_and_crop(event, x, y, flags, param):
    global ref_point, cropping, image, temp_image

    if event == cv2.EVENT_LBUTTONDOWN:
        # Start cropping and record the starting point
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:
        # If cropping, draw the rectangle dynamically
        if cropping:
            temp_image = image.copy()  # Reset temporary image
            cv2.rectangle(temp_image, ref_point[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("Image", temp_image)

    elif event == cv2.EVENT_LBUTTONUP:
        # Record the endpoint and finalize the rectangle
        ref_point.append((x, y))
        cropping = False

        # Draw the final rectangle on the original image
        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("Image", image)
# Load the image
image = cv2.imread('images/img5.jpg')
image = cv2.resize(image,None,fx=0.1,fy=0.1)

if image is None:
    print("Error: Unable to load image. Check the file path.")
    exit()

clone = image.copy()
temp_image = image.copy()  # Initialize the temp image for dynamic drawing
cv2.imshow("Image", image)

cv2.setMouseCallback("Image", click_and_crop)

print("Instructions:")
print("1. Left-click and drag to draw an adjustable rectangle over the watermark region.")
print("2. Release the left mouse button to finalize the selection.")
print("3. Press 'r' to reset the selection.")
print("4. Press 'c' to confirm the selection and remove the watermark.")

while True:
    key = cv2.waitKey(1) & 0xFF

    # Reset the selection
    if key == ord("r"):
        image = clone.copy()
        temp_image = image.copy()
        cv2.imshow("Image", image)
        ref_point = []
        print("Selection reset.")

    # Confirm the selection
    elif key == ord("c"):
        break

cv2.destroyAllWindows()

# Check if ROI is selected
if len(ref_point) == 2:
    # Extract the region of interest (ROI)
    roi = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]

    # Convert ROI to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Calculate the mean intensity of the selected region
    mean_intensity = np.mean(gray_roi)
    print(f"Mean Intensity: {mean_intensity}")

    # Thresholding based on intensity to detect watermark pixels
    _, mask = cv2.threshold(gray_roi, mean_intensity + 10, 255, cv2.THRESH_BINARY)

    # Create a mask for the selected region (for inpainting)
    full_mask = np.zeros(clone.shape[:2], dtype=np.uint8)
    full_mask[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]] = mask

    # Inpainting using the mask
    inpainted_image = cv2.inpaint(clone, full_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # Save the inpainted image to a file
    cv2.imwrite('images/inpainted_image.jpg', inpainted_image)

    # Display the results
    cv2.imshow("Original Image", clone)
    cv2.imshow("Mask", full_mask)
    cv2.imshow("Watermark Removed", inpainted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No region selected. Exiting.")
