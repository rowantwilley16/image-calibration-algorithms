from PIL import Image, ImageDraw

# Define image dimensions
width, height = 4000, 4000

# Create a new image with white background
image = Image.new("RGB", (width, height), "white")

# Create a draw object
draw = ImageDraw.Draw(image)

# Define parameters
center_x, center_y = width // 2, height // 2
square_size = 200
num_squares = 20
intensity = 50
outline_width = 4

# Draw concentric squares
for i in range(num_squares):
    # Calculate square dimensions
    left    = center_x - square_size * (i + 1) // 2
    top     = center_y - square_size * (i + 1) // 2
    right   = left + square_size * (i + 1)
    bottom  = top + square_size * (i + 1)
    
    # Draw square outline
    draw.rectangle([left, top, right, bottom], outline=(intensity, intensity, intensity), width=outline_width)

# Draw diagonals
draw.line([(0, 0), (width, height)], fill=(0, 0, 0), width=outline_width)  # Top-left to bottom-right
draw.line([(0, height), (width, 0)], fill=(0, 0, 0), width=outline_width)  # Bottom-left to top-right

# Save the image as JPEG
image.save("concentric_squares_diagonals.jpg")

# Display image
image.show()
