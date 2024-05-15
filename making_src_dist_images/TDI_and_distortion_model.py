import numpy as np 
import cv2
import matplotlib.pyplot as plt

#fig_width,fig_height = 6,3.3
plt.rcParams['font.family'] = 'Serif'
plt.rcParams['axes.labelsize']     = 12
plt.rcParams['axes.titlesize']     = 12
plt.rcParams['font.size']          = 12
plt.rcParams['legend.fontsize']    = 12
plt.rcParams['xtick.labelsize']    = 12
plt.rcParams['ytick.labelsize']    = 12

image_width     = 4000
image_height    = 4000

cx = image_width//2
cy = image_height//2

k1 = -0.1
TDI_stages = 4
TDI_start_line = 0
TDI_current_output_line = TDI_start_line + TDI_stages

x, y = np.meshgrid(np.arange(image_width), np.arange(image_height))

nx, ny = (x-cx)/cx, (y-cy)/cy
r = np.sqrt((nx)**2 + (ny)**2)
r2 = r**2
nx2, ny2 = nx * (1 + k1 * r**2), ny * (1 + k1 * r**2)
x2, y2 = nx2*cx + cx, ny2*cy + cy

#read a test image from a file rgb image

test_image_1 = r"C:\Users\rowan\OneDrive\Desktop\code\image-calibration-algorithms\making_src_dist_images\ref_sat_img_2.jpg"
test_image_2 = r"C:\Users\rowan\OneDrive\Desktop\code\image-calibration-algorithms\making_src_dist_images\ref_sat_img_4k.jpg"
test_image_3 = r"C:\Users\rowan\OneDrive\Desktop\code\image-calibration-algorithms\making_src_dist_images\rectangle_4k.jpg"
test_image_4 = r"C:\Users\rowan\OneDrive\Desktop\code\image-calibration-algorithms\making_src_dist_images\grey_square_diagonals_4k.jpg"

test_img = cv2.imread(test_image_4, cv2.IMREAD_COLOR)

#split the image into 3 color channels
red_array, green_array, blue_array = cv2.split(test_img)

#show the 3 color channels
cv2.imshow("b", blue_array)
cv2.imshow("g", green_array)
cv2.imshow("r", red_array)

#save the b,g,r channels to a file
cv2.imwrite(r"C:\Users\rowan\OneDrive\Desktop\code\image-calibration-algorithms\making_src_dist_images\split_ref_images\blue_channel_4k.jpg", blue_array)
cv2.imwrite(r"C:\Users\rowan\OneDrive\Desktop\code\image-calibration-algorithms\making_src_dist_images\split_ref_images\green_channel_4k.jpg", green_array)
cv2.imwrite(r"C:\Users\rowan\OneDrive\Desktop\code\image-calibration-algorithms\making_src_dist_images\split_ref_images\red_channel_4k.jpg", red_array)

cv2.waitKey(0)

TDI_stages = 4
blue_filter_start_line  = 1
blue_filter_end_line    = blue_filter_start_line + TDI_stages

green_filter_start_line = 148
green_filter_end_line   = green_filter_start_line + TDI_stages

red_filter_start_line   = 200
red_filter_end_line     = red_filter_start_line + TDI_stages

r2_window_blue      = r2[blue_filter_start_line     :blue_filter_end_line,:]
r2_window_green     = r2[green_filter_start_line    :green_filter_end_line,:]
r2_window_red       = r2[red_filter_start_line      :red_filter_end_line,:]

nx2_window_blue     = nx2[blue_filter_start_line    :blue_filter_end_line,:] 
nx2_window_green    = nx2[green_filter_start_line   :green_filter_end_line,:] 
nx2_window_red      = nx2[red_filter_start_line     :red_filter_end_line,:] 

ny2_window_blue     = ny2[blue_filter_start_line    :blue_filter_end_line,:]
ny2_window_green    = ny2[green_filter_start_line   :green_filter_end_line,:]
ny2_window_red      = ny2[red_filter_start_line     :red_filter_end_line,:]

nx_window_blue      = nx[blue_filter_start_line     :blue_filter_end_line,:]
nx_window_green     = nx[green_filter_start_line    :green_filter_end_line,:]
nx_window_red       = nx[red_filter_start_line      :red_filter_end_line,:]

ny_window_blue      = ny[blue_filter_start_line     :blue_filter_end_line,:]
ny_window_green     = ny[green_filter_start_line    :green_filter_end_line,:]
ny_window_red       = ny[red_filter_start_line      :red_filter_end_line,:]

M_vector_x_window_blue  = nx2_window_blue   - nx_window_blue
M_vector_x_window_green = nx2_window_green  - nx_window_green
M_vector_x_window_red   = nx2_window_red    - nx_window_red

M_vector_y_window_blue  = ny2_window_blue   - ny_window_blue
M_vector_y_window_green = ny2_window_green  - ny_window_green
M_vector_y_window_red   = ny2_window_red    - ny_window_red


print("nx2_window_blue size: "   , nx2_window_blue.shape)
print("nx2_window_green size: "  , nx2_window_green.shape)
print("nx2_window_red size: "    , nx2_window_red.shape)

print("nx2_window_blue: "    , nx2_window_blue)
print("nx2_window_green: "   , nx2_window_green)
print("nx2_window_red: "     , nx2_window_red)

#plot the blue window nx2 and nx values and M vector, the x axis should be the index and the y index should be the rows of the nx2, nx and M vector values
plt.figure()
plt.plot(nx2_window_blue[0,:], label = "nx2 tdi 1")
plt.plot(nx2_window_blue[1,:], label = "nx2 tdi 2")
plt.plot(nx2_window_blue[2,:], label = "nx2 tdi 3")
plt.plot(nx2_window_blue[3,:], label = "nx2 tdi 4")

plt.plot(nx_window_blue[0,:], label = "nx tdi 1")
plt.plot(nx_window_blue[1,:], label = "nx tdi 2")
plt.plot(nx_window_blue[2,:], label = "nx tdi 3")
plt.plot(nx_window_blue[3,:], label = "nx tdi 4")

plt.plot(M_vector_x_window_blue[0,:], label = "M vector tdi 1")
plt.plot(M_vector_x_window_blue[1,:], label = "M vector tdi 2")
plt.plot(M_vector_x_window_blue[2,:], label = "M vector tdi 3")
plt.plot(M_vector_x_window_blue[3,:], label = "M vector tdi 4")

plt.title("Blue Channel nx2, nx and M vector values")
plt.grid()
plt.legend()
plt.show()

#green channel nx2, nx and M vector values
plt.figure()
plt.title("Green Channel nx2, nx and M vector values")

plt.plot(nx2_window_green[0,:], label = "nx2 tdi 1")
plt.plot(nx2_window_green[1,:], label = "nx2 tdi 2")
plt.plot(nx2_window_green[2,:], label = "nx2 tdi 3")
plt.plot(nx2_window_green[3,:], label = "nx2 tdi 4")

plt.plot(nx_window_green[0,:], label = "nx tdi 1")
plt.plot(nx_window_green[1,:], label = "nx tdi 2")
plt.plot(nx_window_green[2,:], label = "nx tdi 3")
plt.plot(nx_window_green[3,:], label = "nx tdi 4")

plt.plot(M_vector_x_window_green[0,:], label = "M vector tdi 1")
plt.plot(M_vector_x_window_green[1,:], label = "M vector tdi 2")
plt.plot(M_vector_x_window_green[2,:], label = "M vector tdi 3")
plt.plot(M_vector_x_window_green[3,:], label = "M vector tdi 4")

plt.grid()
plt.legend()
plt.show()

#red channel nx2, nx and M vector values
plt.figure()
plt.title("Red Channel nx2, nx and M vector values")

plt.plot(nx2_window_red[0,:], label = "nx2 tdi 1")
plt.plot(nx2_window_red[1,:], label = "nx2 tdi 2")
plt.plot(nx2_window_red[2,:], label = "nx2 tdi 3")
plt.plot(nx2_window_red[3,:], label = "nx2 tdi 4")

plt.plot(nx_window_red[0,:], label = "nx tdi 1")
plt.plot(nx_window_red[1,:], label = "nx tdi 2")
plt.plot(nx_window_red[2,:], label = "nx tdi 3")
plt.plot(nx_window_red[3,:], label = "nx tdi 4")

plt.plot(M_vector_x_window_red[0,:], label = "M vector tdi 1")
plt.plot(M_vector_x_window_red[1,:], label = "M vector tdi 2")
plt.plot(M_vector_x_window_red[2,:], label = "M vector tdi 3")
plt.plot(M_vector_x_window_red[3,:], label = "M vector tdi 4")

plt.grid()
plt.legend()
plt.show()

plt.figure()
plt.title("M vector x values for color channels")
plt.plot(M_vector_x_window_blue[0,:], label = "M vector x blue")
plt.plot(M_vector_x_window_green[0,:], label = "M vector x green")
plt.plot(M_vector_x_window_red[0,:], label = "M vector x red")
plt.legend()
plt.grid()
plt.show()

























