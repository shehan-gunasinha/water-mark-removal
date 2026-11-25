import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def load_pretrained_model(model_path):
    """
    Load the pre-trained model (.h5 format).
    """
    model = load_model(model_path)
    return model

def preprocess_image(image_path):
    """
    Preprocess the image for model input.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256))  # Adjust based on model input size
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def remove_watermark(model, image_tensor):
    """
    Use the model to remove watermark.
    """
    output = model.predict(image_tensor)
    output = np.squeeze(output, axis=0)  # Remove batch dimension
    output = (output * 255).astype(np.uint8)  # Convert to image format
    return cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

def main():
    model_path = "watermark_removal_model (1).h5"  
    image_path = "images/img7.jpg"  
    save_path = "images/clean_image.jpg"
    cleaned_image_path = "images/cleaned_img.jpg"

    model = load_pretrained_model(model_path)
    image_tensor = preprocess_image(image_path)
    clean_image = remove_watermark(model, image_tensor)
    
    # Load original image for comparison
    original_image = cv2.imread(image_path)
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Save the cleaned image
    cv2.imwrite(save_path, cv2.cvtColor(clean_image, cv2.COLOR_RGB2BGR))
    
    cleaned_image = cv2.imread(cleaned_image_path)
    cleaned_image = cv2.cvtColor(cleaned_image, cv2.COLOR_BGR2RGB)
   
    # Display the images as subplots
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cleaned_image) 
    plt.title("Cleaned Image")
    plt.axis("off")

    plt.show()

if __name__ == "__main__":
    main()
