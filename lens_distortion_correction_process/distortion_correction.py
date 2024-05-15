import numpy as np 
import cv2
import matplotlib.pyplot as plt
import math

#setup the following before run ========================================
#m vectors folder read location
m_vector_folder = "4000x4000"
dist_image_folder = "4000x4000_checkerboard"
corrected_image_filename = "4000x4000_checkerboard.jpg"
corrected_image_filename_pdf = "4000x4000_checkerboard.pdf"
corrected_figures_folder = dist_image_folder 

cx = 2000
cy = 2000
# =======================================================================

#default figsize
fig_width   = 6
fig_height  = 3.3

#matplotlib rcParams
plt.rcParams['font.family'] = 'Serif'
plt.rcParams['axes.labelsize']     = 12
plt.rcParams['axes.titlesize']     = 12
plt.rcParams['font.size']          = 12
plt.rcParams['legend.fontsize']    = 12
plt.rcParams['xtick.labelsize']    = 12
plt.rcParams['ytick.labelsize']    = 12

#read in 3 text files from distorted images folder into mumpy arrays 
r_dist = np.loadtxt(fr'C:\Users\rowan\OneDrive\Desktop\code\image-calibration-algorithms\lens_distortion_correction_process\distorted_images\{dist_image_folder}\red_image_distorted.txt')
g_dist = np.loadtxt(fr'C:\Users\rowan\OneDrive\Desktop\code\image-calibration-algorithms\lens_distortion_correction_process\distorted_images\{dist_image_folder}\green_image_distorted.txt')
b_dist = np.loadtxt(fr'C:\Users\rowan\OneDrive\Desktop\code\image-calibration-algorithms\lens_distortion_correction_process\distorted_images\{dist_image_folder}\blue_image_distorted.txt')

#centre of the image 
#cx = r_dist.shape[1] // 2
#cy = r_dist.shape[0] // 2 



print("Distorted Image Size : ", cx,cy)

#convert the numpy arrays to uint8
r_dist = r_dist.astype(np.uint8)
g_dist = g_dist.astype(np.uint8)
b_dist = b_dist.astype(np.uint8)

#display the distorted images
plt.figure(figsize=(fig_width,fig_height))
plt.imshow(cv2.merge([b_dist,g_dist,r_dist]))
plt.title('Distorted Image')
plt.axis()

#read in the M_vector text file into a numpy array
Mx_red    = np.loadtxt(fr'C:\Users\rowan\OneDrive\Desktop\code\image-calibration-algorithms\lens_distortion_correction_process\M_vectors\{m_vector_folder}\Mx_red.txt')
My_red    = np.loadtxt(fr'C:\Users\rowan\OneDrive\Desktop\code\image-calibration-algorithms\lens_distortion_correction_process\M_vectors\{m_vector_folder}\My_red.txt')
Mx_green  = np.loadtxt(fr'C:\Users\rowan\OneDrive\Desktop\code\image-calibration-algorithms\lens_distortion_correction_process\M_vectors\{m_vector_folder}\Mx_green.txt')
My_green  = np.loadtxt(fr'C:\Users\rowan\OneDrive\Desktop\code\image-calibration-algorithms\lens_distortion_correction_process\M_vectors\{m_vector_folder}\My_green.txt')
Mx_blue   = np.loadtxt(fr'C:\Users\rowan\OneDrive\Desktop\code\image-calibration-algorithms\lens_distortion_correction_process\M_vectors\{m_vector_folder}\Mx_blue.txt')
My_blue   = np.loadtxt(fr'C:\Users\rowan\OneDrive\Desktop\code\image-calibration-algorithms\lens_distortion_correction_process\M_vectors\{m_vector_folder}\My_blue.txt')

#loop through the red channel and apply the distortion correction
r_corrected_image_plane = np.zeros(r_dist.shape, dtype=np.uint8)
g_corrected_image_plane = np.zeros(g_dist.shape, dtype=np.uint8)
b_corrected_image_plane = np.zeros(b_dist.shape, dtype=np.uint8)

for i in range(r_dist.shape[0]): #loop through the rows
    for j in range(r_dist.shape[1]): #loop through the columns
        corr_x_loc = j + (-1)*(Mx_red[j]*cx) #-1 beacause its the reverse process
        corr_y_loc = i + (-1)*(My_red[j]*cy) #-1 beacause its the reverse process
        if corr_x_loc >= 0 and corr_x_loc < r_dist.shape[1] and corr_y_loc >= 0 and corr_y_loc < r_dist.shape[0]:
            r_corrected_image_plane[math.floor(corr_y_loc),math.floor(corr_x_loc)] = r_dist[i,j]

for i in range(g_dist.shape[0]): #loop through the rows
    for j in range(g_dist.shape[1]): #loop through the columns
        corr_x_loc = j + (-1)*(Mx_green[j]*cx) #-1 beacause its the reverse process
        corr_y_loc = i + (-1)*(My_green[j]*cy) #-1 beacause its the reverse process
        if corr_x_loc >= 0 and corr_x_loc < g_dist.shape[1] and corr_y_loc >= 0 and corr_y_loc < g_dist.shape[0]:
            g_corrected_image_plane[math.floor(corr_y_loc),math.floor(corr_x_loc)] = g_dist[i,j]

for i in range(b_dist.shape[0]): #loop through the rows
    for j in range(b_dist.shape[1]): #loop through the columns
        corr_x_loc = j + (-1)*(Mx_blue[j]*cx) #-1 beacause its the reverse process
        corr_y_loc = i + (-1)*(My_blue[j]*cy) #-1 beacause its the reverse process
        if corr_x_loc >= 0 and corr_x_loc < b_dist.shape[1] and corr_y_loc >= 0 and corr_y_loc < b_dist.shape[0]:
            b_corrected_image_plane[math.floor(corr_y_loc),math.floor(corr_x_loc)] = b_dist[i,j] 

# plot all 3 corrected images side by side 
plt.figure(figsize=(fig_width*2,fig_height*2))
plt.subplot(1,3,1)
plt.imshow(r_corrected_image_plane, cmap='gray')
plt.title('Corrected Red Image')
plt.axis()

plt.subplot(1,3,2)
plt.imshow(g_corrected_image_plane, cmap='gray')
plt.title('Corrected Green Image')
plt.axis()

plt.subplot(1,3,3)
plt.imshow(b_corrected_image_plane, cmap='gray')
plt.title('Corrected Blue Image')
plt.axis()

#save the figure to folder
print("figure saved to folder")
plt.savefig(fr'C:\Users\rowan\OneDrive\Desktop\code\image-calibration-algorithms\lens_distortion_correction_process\corrected_images\figures\{corrected_figures_folder}\split_channels_corrected_images.pdf')

#merge all 3 corrected images into one image
corrected_image = cv2.merge([b_corrected_image_plane,g_corrected_image_plane,r_corrected_image_plane])

plt.figure(figsize=(fig_width,fig_height))
plt.subplot(1,2,1)
plt.imshow(cv2.merge([b_dist,g_dist,r_dist]))
plt.title('Distorted Image')
plt.subplot(1,2,2)
plt.imshow(corrected_image)
plt.title('Corrected Image')
plt.axis()

plt.savefig(fr'C:\Users\rowan\OneDrive\Desktop\code\image-calibration-algorithms\lens_distortion_correction_process\corrected_images\{corrected_image_filename_pdf}')
print("Corrected Image Saved to Folder as matplotlib figure ...")
#save the figure to folder
plt.show()
#save the figure to folder


#save the corrected image to folder 
#cv2.imwrite(fr'C:\Users\rowan\OneDrive\Desktop\code\image-calibration-algorithms\lens_distortion_correction_process\corrected_images\{corrected_image_filename}',corrected_image)
#print("Corrected Image Saved to Folder as jpg ...")

