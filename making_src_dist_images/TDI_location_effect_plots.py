import numpy as np 
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm # for the color maps
import math 

image_width     = 4000
image_height    = 4000

cx = image_width//2
cy = image_height//2

k1 = 0.5
k2 = 0.01
k3 = 0.01

TDI_stages = 4
filter_step_size = 50 #loop incrementor

x, y = np.meshgrid(np.arange(image_width), np.arange(image_height))

nx, ny = (x-cx)/cx, (y-cy)/cy
r = np.sqrt((nx)**2 + (ny)**2)

r2 = r**2
r4 = r2**2
r6 = r2**4

nx2, ny2 = nx * (1 + k1 * r**2 + k2 * r4 + k3 * r6), ny * (1 + k1 * r**2 + k2 * r4 + k3 * r6)
x2, y2 = nx2*cx + cx, ny2*cy + cy

plot_figures = True

if plot_figures: 
    plt.figure(figsize=(8,6)) 
    #set the grid interval to 1
    plt.grid()
    #plt.tight_layout()
    plt.title("M_vector(x-comp) for varying spectral filter locations")
    plt.tight_layout()
    c = cm.get_cmap('magma')

    #plot the M_vector x component for varying spectral filter locations
    for i in range(0, cy, filter_step_size):

        blue_filter_start_line  = i
        blue_filter_end_line    = blue_filter_start_line + TDI_stages

        r2_window_blue      = r2[blue_filter_start_line     :blue_filter_end_line,:]

        nx_window_blue      = nx[blue_filter_start_line     :blue_filter_end_line,:]
        ny_window_blue      = ny[blue_filter_start_line     :blue_filter_end_line,:]

        nx2_window_blue     = nx2[blue_filter_start_line    :blue_filter_end_line,:] 
        ny2_window_blue     = ny2[blue_filter_start_line    :blue_filter_end_line,:]

        M_vector_x_window_blue  = nx2_window_blue   - nx_window_blue
        M_vector_y_window_blue  = ny2_window_blue   - ny_window_blue

        plt.plot( M_vector_x_window_blue[0,:], label = "M_vector_x_window_blue", color = c(i/cy))
        
    plt.figure(figsize=(8,6))

    plt.grid()
    plt.title("M_vector(y-comp) for varying spectral filter locations")
    plt.tight_layout()

    #plot the M_vector y component for varying spectral filter locations
    for i in range(0, cy, filter_step_size):

        #blue filter is used her ein the variable names but it is just the example, it actually represents the a generic filter since it looks through all the spectral filter locations
        blue_filter_start_line  = i 
        blue_filter_end_line    = blue_filter_start_line + TDI_stages

        r2_window_blue      = r2[blue_filter_start_line     :blue_filter_end_line,:]

        nx_window_blue      = nx[blue_filter_start_line     :blue_filter_end_line,:]
        ny_window_blue      = ny[blue_filter_start_line     :blue_filter_end_line,:]

        nx2_window_blue     = nx2[blue_filter_start_line    :blue_filter_end_line,:] 
        ny2_window_blue     = ny2[blue_filter_start_line    :blue_filter_end_line,:]

        M_vector_x_window_blue  = nx2_window_blue   - nx_window_blue
        M_vector_y_window_blue  = ny2_window_blue   - ny_window_blue

        plt.plot( M_vector_y_window_blue[0,:], label = "M_vector_x_window_blue", color = c(i/cy))

    #make an array of size width and height of zeros 
    legend_for_M_vector = np.zeros((image_height, image_width))

    #creating the legend for the M-vector x and y plots 
    #for rows 2000 to 2048, set all the values in the column to linspace between 0 and 1
    for i in range(0, image_height//2):
        legend_for_M_vector[i,:] = (i/(image_height//2))
        #this makes it symmetrical about the y = 0 axis
        legend_for_M_vector[image_height - i -1 , :] = (i/(image_height//2))

    plt.figure(figsize=(8,6))
    plt.imshow(legend_for_M_vector, cmap = 'magma')
    plt.colorbar()
    plt.title("M_vector legend")
    plt.tight_layout()

    # Create a matplotlib figure showing the image sensor with the top half of the image sensor having 
    # the magma gradient and the bottom half having the reflection of the magma gradient around the x axis 
    plt.figure(figsize=(8,6))

    # plot the effect of the r values - create subplots for r2, r4, and r6
    plt.subplot(131)
    plt.imshow(r2, cmap='magma')
    plt.colorbar(shrink=0.366) # reduce the size of the colorbar
    plt.title("r2 values")
    plt.grid()

    plt.subplot(132)
    plt.imshow(r4, cmap='magma')
    plt.colorbar(shrink=0.366)
    plt.title("r4 values")
    plt.grid()

    plt.subplot(133)
    plt.imshow(r6, cmap='magma')
    plt.colorbar(shrink=0.366)
    plt.title("r6 values")
    plt.grid()

    plt.tight_layout()
    #plt.show()

    #getting the max pixel displacement for buffering purposes

    filter_location = 0

    #plot x and x2 for varying filter locations in the y direction 

    plt.figure(figsize=(8,6))
    plt.grid()
    plt.title("x and x2 for varying filter locations in the y direction")
    plt.tight_layout()

    for i in range(0, cy, filter_step_size):
        filter_location = i 

        #plot x and i - same profiles as the M vector but with different scaling
        plt.plot(x2[filter_location,:] - x[filter_location,:], label = "Pixel Movement x-axis", color = c(i/cy))
        plt.plot(y2[filter_location,:] - y[filter_location,:], label = "Pixel Movement y-axis", color = c(i/cy))

    #plt.show()

draw_output_image = True

if draw_output_image: 

    #read in images from split_ref_images 
    blue_image  = cv2.imread(r"C:\Users\rowan\OneDrive\Desktop\code\image-calibration-algorithms\making_src_dist_images\split_ref_images\blue_channel_4k.jpg", cv2.IMREAD_GRAYSCALE)
    green_image = cv2.imread(r"C:\Users\rowan\OneDrive\Desktop\code\image-calibration-algorithms\making_src_dist_images\split_ref_images\green_channel_4k.jpg",cv2.IMREAD_GRAYSCALE)
    red_image   = cv2.imread(r"C:\Users\rowan\OneDrive\Desktop\code\image-calibration-algorithms\making_src_dist_images\split_ref_images\red_channel_4k.jpg",  cv2.IMREAD_GRAYSCALE)
    
    if blue_image is None or green_image is None or red_image is None:
        print("Images not found")
        exit()
    
    #get the height and width of the images
    height, width = blue_image.shape
    print("image [width : height]: ", width, height)

    #SET THE FILTER LOCATIONS FOR THE BLUE, GREEN AND RED FILTERS HERE
    middle_of_sensor        = cy 
    blue_filter_location    = middle_of_sensor - 500
    green_filter_location   = middle_of_sensor 
    red_filter_location     = middle_of_sensor + 500

    #create padding layer for the images for new distorted image size
    padding_x = math.ceil(max(x2[0,:] - x[0,:]))
    padding_y = math.ceil(max(y2[image_height-1,:] - y[image_height-1,:]))

    dist_image_width    = width     + 2*padding_x
    dist_image_height   = height    + 2*padding_y

    #create new images with the new dimensions as the original images 
    blue_image_distorted    = np.zeros((dist_image_height, dist_image_width, 1), dtype = np.uint8)
    green_image_distorted   = np.zeros((dist_image_height, dist_image_width, 1), dtype = np.uint8)
    red_image_distorted     = np.zeros((dist_image_height, dist_image_width, 1), dtype = np.uint8)

    # =====

    # Managing the distortion follows two parts: 

    # 1. Setting up the spectral filter window and getting the M vector values for the window
    # 2. Distorting the image using the M vector values

    # =====
    
    #copied from the plotting section above but now only for a single TDI stage since the TDI doesnt affect it much (show plots why)
    
    #BLUE FILTER WINDOW SETUP 
    r2_window_blue      = r2[blue_filter_location,:]

    nx_window_blue      = nx[blue_filter_location,:]
    ny_window_blue      = ny[blue_filter_location,:]

    nx2_window_blue     = nx2[blue_filter_location,:] 
    ny2_window_blue     = ny2[blue_filter_location,:]

    M_vector_x_window_blue  = nx2_window_blue   - nx_window_blue
    M_vector_y_window_blue  = ny2_window_blue   - ny_window_blue

    #floating point movement values for the pixels(original to distorted image)
    pixel_x_movement_blue = M_vector_x_window_blue * cx 
    pixel_y_movement_blue = M_vector_y_window_blue * cy

    #BLUE FILTER DISTORTION 

    #loop throught the original image and apply the distortion to the new image 
    for y in range(height):
        for x in range(width):

            current_pixel_x = x
            current_pixel_y = y

            #get the pixel value from the original image
            pixel_value = blue_image[y,x]

            #get the new x and y coordinates -this needs to change!Wont be max distortion but based on the M vector instead.
            new_x = current_pixel_x + pixel_x_movement_blue[x] + padding_x #x is the column index
            new_y = current_pixel_y + pixel_y_movement_blue[x] + padding_y #x is the column index

            #set the pixel value in the new image
            blue_image_distorted[math.ceil(new_y), math.ceil(new_x)] = pixel_value #the math. ceil causes black lines in the iamge(interpolation issue)
        


    #GREEN FILTER WINDOW SETUP 
    r2_window_green      = r2[green_filter_location,:]

    nx_window_green      = nx[green_filter_location,:]
    ny_window_green      = ny[green_filter_location,:]

    nx2_window_green     = nx2[green_filter_location,:] 
    ny2_window_green     = ny2[green_filter_location,:]

    M_vector_x_window_green  = nx2_window_green   - nx_window_green
    M_vector_y_window_green  = ny2_window_green   - ny_window_green

    #floating point movement values for the pixels(original to distorted image)
    pixel_x_movement_green  = M_vector_x_window_green * cx 
    pixel_y_movement_green  = M_vector_y_window_green * cy

    #GREEN FILTER DISTORTION

        #loop throught the original image and apply the distortion to the new image 
    for y in range(height):
        for x in range(width):

            current_pixel_x = x
            current_pixel_y = y

            #get the pixel value from the original image
            pixel_value = green_image[y,x]

            #get the new x and y coordinates -this needs to change!Wont be max distortion but based on the M vector instead.
            new_x = current_pixel_x + pixel_x_movement_green[x] + padding_x #x is the column index
            new_y = current_pixel_y + pixel_y_movement_green[x] + padding_y #x is the column index

            #set the pixel value in the new image
            green_image_distorted[math.ceil(new_y), math.ceil(new_x)] = pixel_value #the math. ceil causes black lines in the iamge(interpolation issue)



    #RED FILTER WINDOW SETUP 
    r2_window_red      = r2[red_filter_location,:]

    nx_window_red      = nx[red_filter_location,:]
    ny_window_red      = ny[red_filter_location,:]

    nx2_window_red     = nx2[red_filter_location,:] 
    ny2_window_red     = ny2[red_filter_location,:]

    M_vector_x_window_red  = nx2_window_red   - nx_window_red
    M_vector_y_window_red  = ny2_window_red   - ny_window_red

    #floating point movement values for the pixels(original to distorted image)
    pixel_x_movement_red = M_vector_x_window_red * cx 
    pixel_y_movement_red = M_vector_y_window_red * cy

    #RED FILTER DISTORTION 

    #loop throught the original image and apply the distortion to the new image 
    for y in range(height):
        for x in range(width):

            current_pixel_x = x
            current_pixel_y = y

            #get the pixel value from the original image
            pixel_value = red_image[y,x]

            #get the new x and y coordinates -this needs to change!Wont be max distortion but based on the M vector instead.
            new_x = current_pixel_x + pixel_x_movement_red[x] + padding_x #x is the column index
            new_y = current_pixel_y + pixel_y_movement_red[x] + padding_y #x is the column index

            #set the pixel value in the new image
            red_image_distorted[math.ceil(new_y), math.ceil(new_x)] = pixel_value #the math. ceil causes black lines in the iamge(interpolation issue)



    #show the origirnal blue image and distorted blue image using side by side subplots 
    plt.figure(figsize=(8,6))
    plt.subplot(141)
    plt.imshow(blue_image, cmap='magma')
    plt.title("Original Blue Image")
    plt.grid()

    #change the display image here to the distorted image
    plt.subplot(142)
    plt.imshow(blue_image_distorted, cmap='magma')    
    plt.title("Distorted Blue Image")   
    plt.grid()
    plt.tight_layout()

    #change the display image here to the distorted image
    plt.subplot(143)
    plt.imshow(green_image_distorted, cmap='magma')    
    plt.title("Distorted green Image")   
    plt.grid()
    plt.tight_layout()

    #change the display image here to the distorted image
    plt.subplot(144)
    plt.imshow(red_image_distorted, cmap='magma')    
    plt.title("Distorted red Image")   
    plt.grid()
    plt.tight_layout()

    #plt.show()

    #stack the blue, green and red images into a single image
    stacked_image = cv2.merge([blue_image_distorted, green_image_distorted, red_image_distorted])

    plt.figure(figsize=(8,6))
    plt.imshow(stacked_image)
    plt.title("Simulated Distorted Image")
    plt.grid()
    plt.tight_layout()

    plt.show()