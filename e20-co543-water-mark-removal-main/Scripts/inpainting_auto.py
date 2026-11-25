import cv2
import numpy as np
import os

def detect_watermark(image):
    """
    Automatically detect the entire watermark region using edge detection and contour analysis.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Canny edge detection to find edges
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for all detected watermark regions
    mask = np.zeros_like(gray)
    for contour in contours:
        # Filter out small contours (noise)
        if cv2.contourArea(contour) > 100:  # Adjust this threshold as needed
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    return mask

def remove_watermark(image, mask):
    """
    Remove the watermark using inpainting.
    """
    # Inpainting using the mask
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return inpainted_image

def main():
    # Load the image
    image = cv2.imread('images/img12.jpg')
    if image is None:
        print("Error: Unable to load image. Check the file path.")
        return

    # Resize the image for faster processing (optional)
    image = cv2.resize(image, None, fx=0.1, fy=0.1)

    # Automatically detect the watermark mask
    mask = detect_watermark(image)

    if np.any(mask):  # Check if the mask is not empty
        print("Watermark detected.")

        # Remove the watermark
        inpainted_image = remove_watermark(image, mask)

        # Ensure the 'images' directory exists
        save_path = "images"
        os.makedirs(save_path, exist_ok=True)

        # Save the inpainted image to a file
        cv2.imwrite(os.path.join(save_path, 'cleaned_image.jpg'), inpainted_image)

        # Display the results
        cv2.imshow("Original Image", image)
        cv2.imshow("Watermark Mask", mask)
        cv2.imshow("Watermark Removed", inpainted_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No watermark detected. Exiting.")

if __name__ == "__main__":
    main()