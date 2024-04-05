import cv2
import numpy as np
import matplotlib.pyplot as plt

def conrad_brown_distortion(img, k1, k2, k3):
    """
    Apply Conrad Brown distortion model to an image.
    
    Args:
    - img: Input image (numpy array).
    - k1: Distortion coefficient k1.
    - k2: Distortion coefficient k2.
    - k3: Distortion coefficient k3. 
    Returns:
    - distortion_profile: Distortion profile at different radial distances.
    """
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Calculate radial distance from center
    radial_distance = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Normalize radial distance
    normalized_distance = radial_distance / np.sqrt(cx**2 + cy**2)
    
    # Apply distortion model - 3 parameter distortion model
    distortion = k1 * normalized_distance**2 + k2 * normalized_distance**4 + k3 * normalized_distance**6
    
    return distortion

# Load image
filename = "checkerboard_dots_squares" 
image_path = r"C:\Users\rowan\Documents\masters\sw_comparison\lens_distortion_model\source_images\{:s}.png".format(filename)
original_img = cv2.imread(image_path)

# Distortion coefficients
k1 = 0.001
k2 = 0.01
k3 = 0.001

# Apply distortion
distortion_profile = conrad_brown_distortion(original_img, k1, k2, k3)

# Plot the distortion profile
plt.plot(distortion_profile)
plt.xlabel('Radial Distance')
plt.ylabel('Distortion Amount')
plt.title('Distortion Profile')
plt.grid(True)
plt.show()
