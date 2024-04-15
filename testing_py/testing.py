import numpy as np
import matplotlib.pyplot as plt

image_height    = 18
image_width     = 18

x, y = np.meshgrid(np.arange(image_width), np.arange(image_height))

cx = image_width    // 2
cy = image_height   // 2
print("Image Centre: ", cx,cy)

#set distortion coefficient
k1 = 0.1

#normalize the coordinates
nx, ny = (x-cx)/cx, (y-cy)/cy

print("nx : ", nx)
print("ny : ", ny)

#get the radial distance
r = np.sqrt((nx)**2 + (ny)**2)

print("r : ", r)

#distort the coordinates
nx2, ny2 = nx * (1 + k1 * r**2), ny * (1 + k1 * r**2)

print("nx2 : ", nx2)
print("ny2 : ", ny2)

#convert the distorted coordinates back to image coordinates
x2, y2 = nx2*cx + cx, ny2*cy + cy

print("x2 : ", x2)
print("y2 : ", y2)

