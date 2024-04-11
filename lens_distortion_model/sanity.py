import numpy as np
import cv2

img_width       = 30
img_height      = 30
TDI_stages      = 5
TDI_start_line  = 0
TDI_current_output_line = TDI_start_line + TDI_stages

k1 = 0.1

x, y = np.meshgrid(np.arange(img_width), np.arange(img_height))
cx, cy = img_width//2, img_height//2
nx, ny = (x-cx)/cx, (y-cy)/cy
r = np.sqrt((nx)**2 + (ny)**2)
nx2, ny2 = nx * (1 + k1 * r**2), ny * (1 + k1 * r**2)
x2, y2 = nx2*cx + cx, ny2*cy + cy

test_img = np.zeros((img_width, img_height),dtype=np.uint8)
test_img[TDI_start_line:TDI_current_output_line, : ] = np.broadcast_to(np.random.randint(0,256, img_width), (TDI_stages, img_width)) #why does it overexpose if i use 255?
distorted_img = cv2.remap(test_img, x2.astype(np.float32), y2.astype(np.float32), cv2.INTER_LINEAR)

#display the test image 
cv2.imshow("test_image", test_img)
cv2.imshow("distorted_image", distorted_img)
cv2.waitKey(0)

print("test_img\n", test_img)
#save the test image to a file 
np.savetxt("test_image.txt", test_img, fmt='%d', delimiter='\t')
print("distorted_img\n", distorted_img)

#save the distorted image array to a text file with tabs between the values 
np.savetxt("distorted_image.txt", distorted_img, fmt='%d', delimiter='\t')





tdi = np.sum(distorted_img[TDI_start_line:TDI_current_output_line, :], axis=0)
#print("test_img\n", test_img)

norm_test = test_img[0, :]/np.sum(test_img[0, :])

