from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
def generate_checkerboard(width, height, square_size):
    img = Image.new('RGB', (width, height), color='white')
    pixels = img.load()
    for y in range(0, height, square_size):
        for x in range(0, width, square_size):
            if ((x // square_size) + (y // square_size)) % 2 == 0:
                for i in range(square_size):
                    for j in range(square_size):
                        if x + i < width and y + j < height:
                            pixels[x + i, y + j] = (0, 0, 0)
    
    
    # Draw a red dot with a circle at the center
    draw = ImageDraw.Draw(img)
    center_x = width // 2
    center_y = height // 2
    radius = min(width, height) // 100  # Adjust the radius as needed
    draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), fill='red')
    
    # Add reference points
    #reference_points = [(100, 100), (width - 100, 100), (width - 100, height - 100), (100, height - 100)]
    #for point in reference_points:
        #draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill='blue')
    
    # Add squares
    square_sizes = [1000, 2000, 3000]
    for size in square_sizes:
        draw.rectangle((center_x - size//2, center_y - size//2, center_x + size//2, center_y + size//2), outline='red',width=20)
    
    
    
    
    img.save(r"C:\Users\rowan\Documents\masters\sw_comparison\lens_distortion_model\source_images\checkerboard_dots_squares.png")
    return img

if __name__ == "__main__":
    width = 4000
    height = 4000
    square_size = 100  # Adjust the square size as needed
    checkerboard_img = generate_checkerboard(width, height, square_size)
    print("GENERATED : Checkerboard_with_dot_reference_points_and_squares.png ")
    plt.imshow(checkerboard_img)
    plt.axis('off')
    plt.show()