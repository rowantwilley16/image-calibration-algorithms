import cv2
import numpy as np
import matplotlib.pyplot as plt

def conrad_brown_distortion(img, k1, k2,k3):
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
    
    print(x[0,0],y[0,0])
    # Normalize image coordinates
    x = (x - cx) / cx
    y = (y - cy) / cy
    
    print(x[0,0],y[0,0])

    r2 = x**2 + y**2
    r4 = r2**2
    r6 = r2**3
     
    # Apply distortion model - 3 parameter distortion model
    x_distorted = x / (1 + k1 * r2) # the reason we have divide here to to account for the remap fn. 
    y_distorted = y / (1 + k1 * r2)
    
    #x_distorted = x * (1 + k1 * r2 + k2 * r4 + k3 * r6)
    #y_distorted = y * (1 + k1 * r2 + k2 * r4 + k3 * r6)
    
    print(x_distorted[0,0], y_distorted[0,0])
    # Denormalize coordinates
    x_distorted = x_distorted * cx + cx
    y_distorted = y_distorted * cy + cy

    print(x_distorted[0,0], y_distorted[0,0])
    
    # Remap distorted image
    distorted_img = cv2.remap(img, x_distorted.astype(np.float32), y_distorted.astype(np.float32), interpolation=cv2.INTER_LINEAR)
    #print("x_distorted: ",x_distorted)
    #print("y_distorted: ",y_distorted)
    return distorted_img

# Load image
filename = "checkerboard_dots_squares" #change this if you want to use a different source image
image_path = r"C:\Users\rowan\Documents\masters\sw_comparison\lens_distortion_model\source_images\{:s}.png".format(filename)
original_img = cv2.imread(image_path)

# Distortion coefficients
k1 = 0.1
k2 = 0.01
k3 = 0.01

# Apply distortion
distorted_image = conrad_brown_distortion(original_img, k1, k2,k3)

# Save the distorted image
output_path = r"C:\Users\rowan\Documents\masters\sw_comparison\lens_distortion_model\output_images\{:s}_distorted.png".format(filename)
cv2.imwrite(output_path, distorted_image)

plt.imshow(distorted_image)
plt.axis('off')
plt.show()
# Display original and distorted images
#cv2.imshow('Original Image', original_img)
#cv2.imshow('Distorted Image', distorted_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
