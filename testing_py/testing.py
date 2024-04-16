import numpy as np
import matplotlib.pyplot as plt

#set the debug level
# 0 - no debug
# 1 - print the values to the console
# 2 - save the values to textfiles 

#set the debug level before you run the program, the debug level is a global variable
debug = 2

def main():

    #set the image size
    image_height    = 50
    image_width     = 50

    x, y = np.meshgrid(np.arange(image_width), np.arange(image_height))

    cx = image_width    // 2
    cy = image_height   // 2

    #set distortion coefficient
    k1 = 0.1

    #normalize the coordinates matrix
    nx, ny = (x-cx)/cx, (y-cy)/cy

    #get the radial distance matrix
    r = np.sqrt((nx)**2 + (ny)**2)
    r2 = r**2

    #distort the coordinates 
    dist_nx, dist_ny = nx * (1 + k1 * r**2), ny * (1 + k1 * r**2)

    #convert the distorted coordinates back to image coordinates
    x2, y2 = dist_nx*cx + cx, dist_ny*cy + cy

    #run the debug processes
    if debug == 1:
        debug_level_one(cx,cy,k1,nx,ny,r2,dist_nx,dist_ny,x2,y2)
    elif debug == 2:
        debug_level_two(cx,cy,k1,nx,ny,r2,dist_nx,dist_ny,x2,y2)
    
#debug level 1 - print the values to the console
def debug_level_one(cx,cy,k1,nx,ny,r2,dist_nx,dist_ny,x2,y2): 

    #print the image centre
    print("Image Centre:\n", cx,cy)
    
    #print the k value 
    print("k1 : \n", k1)
    
    #print the normalized coordinates
    print("nx : \n", nx)
    print("ny : \n", ny)

    #print the radial distance
    print("r2 : \n", r2)

    #print the distorted coordinates
    print("dist_nx : \n", dist_nx)
    print("ddist_ny : \n", dist_ny)
    
    #print the new image coordinates
    print("x2 : \n", x2)
    print("y2 : \n", y2)

#debug level 2 - save the values to textfiles and print the values to the console
def debug_level_two(cx,cy,k1,nx,ny,r2,dist_nx,dist_ny,x2,y2):

    #call the debug level 1 function
    debug_level_one(cx,cy,k1,nx,ny,r2,dist_nx,dist_ny,x2,y2)

    #save the image centre to a textfile 
    np.savetxt("testing_py/image_centre.txt", [cx,cy,k1], fmt="%f")

    #save the radial values to a textfile 
    np.savetxt("testing_py/r_values.txt", r2, fmt="%f")

    #save the normalized coordinates to a textfiles
    np.savetxt("testing_py/nx_values.txt", nx, fmt="%f")
    np.savetxt("testing_py/ny_values.txt", ny, fmt="%f")

    #save the distorted coordinates to a textfiles
    np.savetxt("testing_py/dist_nx_values.txt", dist_nx, fmt="%f")
    np.savetxt("testing_py/dist_ny_values.txt", dist_ny, fmt="%f")

if __name__ == "__main__":
    main()

    #plot the original and distorted coordinates
    #plt.figure()
    #plt.plot(x,y,'ro')
    #plt.plot(x2,y2,'bo')
    #plt.show()
