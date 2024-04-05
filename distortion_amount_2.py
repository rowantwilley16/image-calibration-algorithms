import cv2
import numpy as np
import matplotlib.pyplot as plt

def brown_conrady_distortion(p, k1, k2,k3):
    # Brown-Conrady distortion model
    r = np.linalg.norm(p)
    distortion_factor = 1 + k1 * r**2 + k2 * r**4 + k3 * r**6
    return p * distortion_factor

def calculate_distortion_amount(ref_img, distorted_img, k1, k2,k3):
    # Get image dimensions
    height, width, _ = ref_img.shape

    # Initialize arrays to store distortion amounts and radii
    distortion_amounts = []
    radii = []

    # Iterate through each pixel in the image
    for y in range(height):
        for x in range(width):
            # Calculate distortion for the pixel using Brown-Conrady model
            p_ref = np.array([x, y])  # Reference image pixel location
            p_distorted = brown_conrady_distortion(p_ref, k1, k2, k3)  # Distorted image pixel location

            # Calculate distortion amount (distance between distorted and reference pixels)
            distortion_amount = np.linalg.norm(p_distorted - p_ref)

            # Calculate radius (distance from the center)
            radius = np.linalg.norm(p_ref - np.array([width/2, height/2]))

            # Append distortion amount and radius to respective arrays
            distortion_amounts.append(distortion_amount)
            radii.append(radius)

    return np.array(radii), np.array(distortion_amounts)

def plot_distortion(radius, distortion_amount):
    # Plot radius vs distortion amount
    plt.figure(figsize=(8, 6))
    plt.scatter(radius, distortion_amount, s=1, color='b')
    plt.title('Radius vs Distortion Amount')
    plt.xlabel('Radius')
    plt.ylabel('Distortion Amount')
    plt.grid(True)
    plt.show()

def main():
    # Load reference and distorted images
    ref_img = cv2.imread(r"C:\Users\rowan\Documents\masters\sw_comparison\lens_distortion_model\source_images\checkerboard_dots_squares.png")
    distorted_img = cv2.imread(r"C:\Users\rowan\Documents\masters\sw_comparison\lens_distortion_model\output_images\checkerboard_dots_squares_distorted.png")

    # Brown-Conrady distortion parameters
    k1 = 0.001
    k2 = 0.01
    k3 = 0.001

    # Calculate distortion amount
    radius, distortion_amount = calculate_distortion_amount(ref_img, distorted_img, k1, k2,k3)

    # Plot distortion
    plot_distortion(radius, distortion_amount)

if __name__ == "__main__":
    main()
