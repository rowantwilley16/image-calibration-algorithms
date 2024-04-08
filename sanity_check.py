import cv2
import numpy as np

image_height = 500
image_width = 500

#create a list of image co-ords evenly spaced across the image
image_co_ords = []
num_points = 20

for i in range(num_points):
    for j in range(num_points):
        x = (i + 0.5) * (image_width / num_points)
        y = (j + 0.5) * (image_height / num_points)
        image_co_ords.append([x, y])

    #image_co_ords = np.array([[0,0],[0,image_height],[image_width,0],[image_width,image_height]], dtype=np.float32)

# center of the image
cx = image_width//2
cy = image_height//2

#plot the original points a blank image
blank_image = np.zeros((image_width*2,image_height*2,3), np.uint8)

offset = 250

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
    nx_distorted = nx * (1 + k1 * r2)
    ny_distorted = ny * (1 + k1 * r2)

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
    dn_coords = []
    original_coords = []

    #append the dn co-ords in tuple pairs 
    x_offset = x + offset
    y_offset = y + offset

    x_dn_offset = x_dn + offset
    y_dn_offset = y_dn + offset

    cx_offset = cx + offset
    cy_offset = cy + offset

    #draw the vector lines from the orginal points to the distorted points
    cv2.line(blank_image, (int(x_offset), int(y_offset)), (int(x_dn_offset), int(y_dn_offset)), (0,255,0), 1)

    #draw the concentric circles for r2 r4 and r6
    cv2.circle(blank_image, (int(x_offset), int(y_offset)), int(r2*10), (255,0,0), 1)
    
    #cv2.line(blank_image, (int(cx_offset), int(cy_offset)), (int(x_offset), int(y_offset)), (255,255,255), 1)

    cv2.circle(blank_image, (int(cx_offset), int(cy_offset)), 1, (255,0,255), -1)
    cv2.circle(blank_image, (int(x_offset), int(y_offset)), 1, (255,255,255), -1)
    cv2.circle(blank_image, (int(x_dn_offset), int(y_dn_offset)), 1, (0,255,255), -1)

cv2.imshow("Original Points", blank_image)
cv2.waitKey(0)
cv2.destroyAllWindows()





