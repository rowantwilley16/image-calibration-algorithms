import cv2
import numpy as np

x = 0
y = 0

cx = 2000
cy = 2000 

x_norm = (x - cx) / cx
y_norm = (y - cy) / cy 

r2 = x_norm**2 + y_norm**2
r4 = r2**2
r6 = r2**3

#distrotion co-efficents 

k1 = 0.1 
k2 = 0.01 
k3 = 0.01

# Apply distortion model - 3 parameter distortion model
x_distorted = x_norm * (1 + k1 * r2)
y_distorted = y_norm * (1 + k1 * r2)

x_final = x_distorted *cx + cx 
y_final = y_distorted *cy + cy 

print("actual co-ords:\t\t\t", x,y)
if k1 > 0: 
    print("K is positive")
else: 
    print("K is negative")
print("normalized co-ords:\t\t", x_norm, y_norm)
print("normalized distorted co-ords:\t", x_distorted,y_distorted)
print("denormalized co-ords: \t\t",x_final, y_final )




