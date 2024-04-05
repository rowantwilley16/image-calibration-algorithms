import cv2
import numpy as np

def reverse_conrad_brown_distortion(distorted_img, k1, k2, k3):
    """
    Reverse Conrad Brown distortion model to obtain a corrected image.
    
    Args:
    - distorted_img: Distorted image (numpy array).
    - k1: Distortion coefficient k1.
    - k2: Distortion coefficient k2.
    - k3: Distortion coefficient k3.
    
    Returns:
    - corrected_img: Corrected image (numpy array).
    """
    h, w = distorted_img.shape[:2]
    cx, cy = w // 2, h // 2
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Normalize image coordinates
    x = (x - cx) / cx
    y = (y - cy) / cy
    
    r2 = x**2 + y**2
    r4 = r2**2
    r6 = r2**3
    
    # Reverse distortion model
    x_corrected = x / (1 + k1 * r2 + k2 * r4 + k3 * r6)
    y_corrected = y / (1 + k1 * r2 + k2 * r4 + k3 * r6)
    
    # Denormalize coordinates
    x_corrected = x_corrected * cx + cx
    y_corrected = y_corrected * cy + cy
    
    # Remap corrected image
    corrected_img = cv2.remap(distorted_img, x_corrected.astype(np.float32), y_corrected.astype(np.float32), interpolation=cv2.INTER_LINEAR)
    
    return corrected_img

# Load distorted image
distorted_image_path = r"C:\Users\rowan\Documents\masters\sw_comparison\lens_distortion_model\output_images\checkerboard_dots_squares_distorted.png"
distorted_img = cv2.imread(distorted_image_path)

# Distortion coefficients used previously
k1 = 0.001
k2 = 0.01
k3 = 0.001

# Reverse distortion
corrected_image = reverse_conrad_brown_distortion(distorted_img, k1, k2, k3)

# Display original and corrected images
cv2.imshow('Distorted Image', distorted_img)
cv2.imshow('Corrected Image', corrected_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
