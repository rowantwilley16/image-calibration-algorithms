import numpy as np
import cv2 
import matplotlib.pyplot as plt

#read in an input image and seperate it intop its color channels and save them as images 
image_path = r"C:\Users\rowan\OneDrive\Desktop\code\image-calibration-algorithms\distortion_correction\input2.jpg"

test_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
b, g, r = cv2.split(test_img)
cv2.imwrite(r'C:\Users\rowan\OneDrive\Desktop\code\image-calibration-algorithms\distortion_correction\split_channels/r_channel.jpg', r)
cv2.imwrite(r'C:\Users\rowan\OneDrive\Desktop\code\image-calibration-algorithms\distortion_correction\split_channels/g_channel.jpg', g)
cv2.imwrite(r'C:\Users\rowan\OneDrive\Desktop\code\image-calibration-algorithms\distortion_correction\split_channels/b_channel.jpg', b)

#plot the original image and the 3 color channels 
plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.subplot(2,2,2)
plt.imshow(r, cmap='gray')
plt.title('Red Channel')
plt.subplot(2,2,3)
plt.imshow(g, cmap='gray')
plt.title('Green Channel')
plt.subplot(2,2,4)
plt.imshow(b, cmap='gray')
plt.title('Blue Channel')
plt.show()

b_for_calcs = np.copy(b)
g_for_calcs = np.copy(g)
r_for_calcs = np.copy(r)

#in [u,v] pairs 
M_vector_red    = [0,1] #up by one pixel
M_vector_green  = [0,-1] #down by one pixel 
M_vector_blue   = [1,0] #right by one pixel

#apply the distortion correction to the red channel, red moves up by one pixel, green moves down by one pixel, and blue moves right by one pixel
r_corrected = np.zeros(r.shape, dtype=np.uint8)
g_corrected = np.zeros(g.shape, dtype=np.uint8)
b_corrected = np.zeros(b.shape, dtype=np.uint8)

for i in range(r.shape[0]):
    for j in range(r.shape[1]): 
        #get the pixel value at the current location 
        r_val = r_for_calcs[i,j]
        
        #get the new location for the pixel value 
        new_i_red = i - M_vector_red[1]
        new_j_red = j + M_vector_red[0]
        
        #check if the new location is within the bounds of the image
        if new_i_red >= 0 and new_i_red < r.shape[0] and new_j_red >= 0 and new_j_red < r.shape[1]:
            r_corrected[new_i_red, new_j_red] = r_val
            r_for_calcs[i,j] = 0

for i in range(g.shape[0]):
    for j in range(g.shape[1]): 
        #get the pixel value at the current location 
        g_val = g_for_calcs[i,j]
        
        #get the new location for the pixel value 
        new_i_green = i - M_vector_green[1]
        new_j_green = j + M_vector_green[0]
        
        #check if the new location is within the bounds of the image
        if new_i_green >= 0 and new_i_green < g.shape[0] and new_j_green >= 0 and new_j_green < g.shape[1]:
            g_corrected[new_i_green, new_j_green] = g_val
            g_for_calcs[i,j] = 0

for i in range(b.shape[0]):
    for j in range(b.shape[1]): 
        #get the pixel value at the current location 
        b_val = b_for_calcs[i,j]
        
        #get the new location for the pixel value 
        new_i_blue = i - M_vector_blue[1]
        new_j_blue = j + M_vector_blue[0]
        
        #check if the new location is within the bounds of the image
        if new_i_blue >= 0 and new_i_blue < b.shape[0] and new_j_blue >= 0 and new_j_blue < b.shape[1]:
            b_corrected[new_i_blue, new_j_blue] = b_val
            b_for_calcs[i,j] = 0
        
#save the corrected images as images 
cv2.imwrite(r'C:\Users\rowan\OneDrive\Desktop\code\image-calibration-algorithms\distortion_correction\corrected_channels/r_channel_corrected.jpg', r_corrected)
cv2.imwrite(r'C:\Users\rowan\OneDrive\Desktop\code\image-calibration-algorithms\distortion_correction\corrected_channels/g_channel_corrected.jpg', g_corrected)
cv2.imwrite(r'C:\Users\rowan\OneDrive\Desktop\code\image-calibration-algorithms\distortion_correction\corrected_channels/b_channel_corrected.jpg', b_corrected)

#combine the corrected channels into a single image
corrected_image = cv2.merge((b_corrected, g_corrected, r_corrected))

#plot the original image and the corrected image
plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(corrected_image,cv2.COLOR_BGR2RGB))
plt.title('Corrected Image')
plt.show()

#plot the orginal 3 color channels and the 3 corrected color channels
plt.figure(figsize=(10,10))
plt.subplot(2,3,1)
plt.imshow(r, cmap='gray')
plt.title('Red Channel')
plt.subplot(2,3,2)
plt.imshow(g, cmap='gray')
plt.title('Green Channel')
plt.subplot(2,3,3)
plt.imshow(b, cmap='gray')
plt.title('Blue Channel')
plt.subplot(2,3,4)
plt.imshow(r_corrected, cmap='gray')
plt.title('Red Channel Corrected')
plt.subplot(2,3,5)
plt.imshow(g_corrected, cmap='gray')
plt.title('Green Channel Corrected')
plt.subplot(2,3,6)
plt.imshow(b_corrected, cmap='gray')
plt.title('Blue Channel Corrected')
plt.show()