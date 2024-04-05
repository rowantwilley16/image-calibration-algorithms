from PIL import Image, ImageDraw

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
    radius = min(width, height) // 20  # Adjust the radius as needed
    draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), fill='red')
    
    # Add reference points
    reference_points = [(100, 100), (width - 100, 100), (width - 100, height - 100), (100, height - 100)]
    for point in reference_points:
        draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill='red')
    
    img.save("checkerboard_with_dot_and_reference_points.png")
    return img

if __name__ == "__main__":
    width = 4000
    height = 4000
    square_size = 100  # Adjust the square size as needed
    checkerboard_img = generate_checkerboard(width, height, square_size)
    print("Checkerboard image with dot and reference points generated and saved as 'checkerboard_with_dot_and_reference_points.png'")
