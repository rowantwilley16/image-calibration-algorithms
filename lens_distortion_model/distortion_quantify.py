import cv2
import numpy as np
import matplotlib.pyplot as plt
from rich.progress import track

def conrad_brown_distortion(img, k1, k2, k3):
    """
    Apply Conrad Brown distortion model to an image.
    
    Args:
    - img: Input image (numpy array).
    - k1: Distortion coefficient k1.
    - k2: Distortion coefficient k2.
    - k3: Distortion coefficient k3. 
    Returns:
    - radial_distances: Array of radial distances.
    - average_distortions: Array of average distortion amounts corresponding to each unique radial distance.
    """
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Calculate radial distance from center
    radial_distances = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    # Normalize radial distance
    normalized_distances = radial_distances / np.sqrt(cx**2 + cy**2)
    
    # Apply distortion model - 3 parameter distortion model
    distortions = k1 * normalized_distances**2 + k2 * normalized_distances**4 + k3 * normalized_distances**6
    
    # Group distortions by unique radial distances
    unique_distances = np.unique(normalized_distances)
    
    return unique_distances, 1 + k1 * unique_distances**2 + k2 * unique_distances**4 + k3 * unique_distances**6

# Load image
filename = "checkerboard_dots_squares" #change this if you want to use a different source image
image_path = r"C:\Users\rowan\Documents\masters\sw_comparison\lens_distortion_model\source_images\{:s}.png".format(filename)
original_img = cv2.imread(image_path)

# Distortion coefficients
k1 = 0.001
k2 = 0.01
k3 = 0.001

# Apply distortion
radial_distances, average_distortions = conrad_brown_distortion(original_img, k1, k2, k3)

# Plot the distortion profile
plt.plot(radial_distances, average_distortions)
plt.xlabel('Radial Distance')
plt.ylabel('Average Distortion Amount')
plt.title('Average Distortion Profile')
plt.grid(True)
plt.show()
