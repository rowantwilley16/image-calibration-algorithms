import cv2
import numpy as np

def conrad_brown_distortion(img, k1, k2, k3):
    """
    Apply Conrad Brown distortion model to an image.
    
    Args:
    - img: Input image (numpy array).
    - k1: Distortion coefficient k1.
    - k2: Distortion coefficient k2.
    - k3: Distortion coefficient k3. 
    Returns:
    - distorted_img: Distorted image (numpy array).
    """
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # Normalize image coordinates
    x_norm = (x - cx) / cx
    y_norm = (y - cy) / cy
    
    r2 = x_norm**2 + y_norm**2
    r4 = r2**2
    r6 = r2**3
    
    # Apply distortion model
    x_distorted_norm = x_norm * (1 + k1 * r2 + k2 * r4 + k3 * r6)
    y_distorted_norm = y_norm * (1 + k1 * r2 + k2 * r4 + k3 * r6)
    
    # Denormalize coordinates
    x_distorted = x_distorted_norm * cx + cx
    y_distorted = y_distorted_norm * cy + cy
    
    # Initialize distorted image
    distorted_img = np.zeros_like(img)
    
    # Iterate over each pixel in the distorted image
    for i in range(h):
        for j in range(w):
            # Use bilinear interpolation to find the pixel value at distorted coordinates
            x_int, y_int = int(x_distorted[i, j]), int(y_distorted[i, j])
            x_frac, y_frac = x_distorted[i, j] - x_int, y_distorted[i, j] - y_int
            
            if 0 <= x_int < w - 1 and 0 <= y_int < h - 1:
                # Perform bilinear interpolation
                top_left        = img[y_int, x_int] * (1 - x_frac) * (1 - y_frac)
                top_right       = img[y_int, x_int + 1] * x_frac * (1 - y_frac)
                bottom_left     = img[y_int + 1, x_int] * (1 - x_frac) * y_frac
                bottom_right    = img[y_int + 1, x_int + 1] * x_frac * y_frac
                
                # Assign interpolated pixel value to the distorted image
                distorted_img[i, j] = top_left + top_right + bottom_left + bottom_right
            
    return distorted_img.astype(np.uint8)

# Load image
filename = "checkerboard_dots_squares" #change this if you want to use a different source image
image_path = r"C:\Users\rowan\Documents\masters\sw_comparison\lens_distortion_model\source_images\{:s}.png".format(filename)
original_img = cv2.imread(image_path)

# Distortion coefficients
k1 = 0.1
k2 = 0.01
k3 = 0.01

# Apply distortion
distorted_image = conrad_brown_distortion(original_img, k1, k2, k3)

# Save the distorted image
output_path = r"C:\Users\rowan\Documents\masters\sw_comparison\lens_distortion_model\output_images\{:s}_distorted_first_principals.png".format(filename)
cv2.imwrite(output_path, distorted_image)

# Display original and distorted images
cv2.imshow('Original Image', original_img)
cv2.imshow('Distorted Image', distorted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
