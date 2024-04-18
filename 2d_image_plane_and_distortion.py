import cv2
import numpy as np
import matplotlib.pyplot as plt

image_height    = 500
image_width     = 500

rectangle_size = 5

#create a list of image co-ords evenly spaced across the image
image_co_ords = []
num_points = 30

for i in range(num_points):
    for j in range(num_points-1):
        x = (i + 0.5) * (image_width / num_points)
        y = (j + 0.5) * (image_height / num_points)
        image_co_ords.append([x, y])

    #image_co_ords = np.array([[0,0],[0,image_height],[image_width,0],[image_width,image_height]], dtype=np.float32)

# center of the image   
cx = image_width//2
cy = image_height//2

offset = image_width//2

#plot the original points a blank image
blank_image = np.zeros((int(image_width*2),int(image_height*2),3), np.uint8)

#make all the pixels in the blank image 255
blank_image.fill(255)

#plt.figure(figsize=(8, 8))
#plt.imshow(blank_image)
#plt.title('Original and Distorted Points')

for coords in image_co_ords:
    x = coords[0]
    y = coords[1]

    nx = (x - cx) / cx
    ny = (y - cy) / cy 

    #normalized radial co-ords
    r2 = nx**2 + ny**2
    r4 = r2**2
    r6 = r2**3

#distrotion co-efficents 

    k1 = 0.1 
    k2 = 0.01 
    k3 = 0.01

    # Apply distortion model - 3 parameter distortion model
    stage_1_x = nx * (1 + k1 * r2)
    stage_1_y = ny * (1 + k1 * r2)

    stage_2_x = nx * (1 + k1 * r2 + k2 * r4)
    stage_2_y = ny * (1 + k1 * r2 + k2 * r4)

    stage_3_x = nx * (1 + k1 * r2 + k2 * r4 + k3 * r6)
    stage_3_y = ny * (1 + k1 * r2 + k2 * r4 + k3 * r6)

    stage_1_dn_x = stage_1_x * cx + cx + offset
    stage_1_dn_y = stage_1_y * cy + cy + offset

    stage_2_dn_x = stage_2_x * cx + cx + offset
    stage_2_dn_y = stage_2_y * cy + cy + offset

    stage_3_dn_x = stage_3_x * cx + cx + offset
    stage_3_dn_y = stage_3_y * cy + cy + offset

    nx_distorted = nx * (1 + k1 * r2 + k2 * r4 + k3 * r6)
    ny_distorted = ny * (1 + k1 * r2 + k2 * r4 + k3 * r6)

    x_dn = nx_distorted *cx + cx 
    y_dn = ny_distorted *cy + cy 

    print("FOR IMAGE CO-ORDS: ", x,y)
    print("actual co-ords:\t\t\t", x,y)

    if k1 > 0: 
        print("K is positive")
    else: 
        print("K is negative")

    print("normalized co-ords:\t\t", nx, ny)
    print("normalized distorted co-ords:\t", nx_distorted,ny_distorted)
    print("denormalized co-ords: \t\t",x_dn, y_dn )

    print("\n\n")

    #make an empty list for dn co-ords
    dn_coords       = []
    original_coords = []

    #append the dn co-ords in tuple pairs 
    x_offset = x + offset
    y_offset = y + offset

    x_dn_offset = x_dn + offset
    y_dn_offset = y_dn + offset

    cx_offset   = cx + offset
    cy_offset   = cy + offset

    #draw the vector lines from the orginal points to the distorted points
    #cv2.line(blank_image, (int(x_offset), int(y_offset)), (int(x_dn_offset), int(y_dn_offset)), (0,255,0), 1)

    #color is BGR
    stage_1_color = (0,100,255) #orange 
    stage_2_color = (50,150,50) #green
    stage_3_color = (200,50,0) 

    #plot stage 1dn, 2dn and 3dn on the image
    #cv2.circle(blank_image, (int(stage_1_dn_x), int(stage_1_dn_y)), 1, stage_1_color, -1)
    #cv2.circle(blank_image, (int(stage_2_dn_x), int(stage_2_dn_y)), 1, stage_2_color, -1)
    cv2.circle(blank_image, (int(stage_3_dn_x), int(stage_3_dn_y)), 1, stage_3_color, -1)

    #draw vector lines from original points to distorted points

    #cv2.line(blank_image, (int(x_offset), int(y_offset)), (int(stage_1_dn_x), int(stage_1_dn_y)), stage_1_color, 1)
    #cv2.line(blank_image, (int(stage_1_dn_x), int(stage_1_dn_y)), (int(stage_2_dn_x), int(stage_2_dn_y)), stage_2_color, 1)
    #cv2.line(blank_image, (int(stage_2_dn_x), int(stage_2_dn_y)), (int(stage_3_dn_x), int(stage_3_dn_y)), stage_3_color, 1)

    #draw the concentric circles for r2 
    #cv2.circle(blank_image, (int(x_offset), int(y_offset)), int((r2*10)), (0,255,255), 1)
    
    #cv2.line(blank_image, (int(cx_offset), int(cy_offset)), (int(x_offset), int(y_offset)), (255,255,255), 1)

    cv2.circle(blank_image, (int(cx_offset),    int(cy_offset)), 2, (0,0,255),   -1)
    cv2.circle(blank_image, (int(x_offset),     int(y_offset)) , 1, (0,0,0), -1)

    #draw small squares at each orgirnal and final point 

    cv2.rectangle(blank_image, (int(x_offset)-rectangle_size, int(y_offset)-rectangle_size), (int(x_offset)+rectangle_size, int(y_offset)+rectangle_size), (185,185,255), -1)
    cv2.rectangle(blank_image, (int(x_dn_offset)-rectangle_size, int(y_dn_offset)-rectangle_size), (int(x_dn_offset)+rectangle_size, int(y_dn_offset)+rectangle_size), (0,0,0), 1)

    #plot the vector line from original to final point 
    cv2.line(blank_image, (int(x_offset), int(y_offset)), (int(x_dn_offset), int(y_dn_offset)), (100,0,255), 1)
    #cv2.circle(blank_image, (int(x_dn_offset), int(y_dn_offset)), 1, (0,255,255), -1)

    #plt.plot([x + offset, x_dn + offset], [y + offset, y_dn + offset], color='blue')
    #plt.plot(x + offset, y + offset, '.')  # Original points
    #plt.plot(x_dn + offset, y_dn + offset, 'x')  # Distorted points

#plt.gca().invert_yaxis()  # Invert y-axis to match the image coordinate system
#plt.show()

#write the cv2 image to pdf file
cv2.imwrite("distortion_stages.png", blank_image)

cv2.imshow("Original Points", blank_image)
cv2.waitKey(0)
cv2.destroyAllWindows()





