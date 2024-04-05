import numpy as np
import cv2
import matplotlib.pyplot as plt

k1 = 0.1
x, y = np.meshgrid(np.arange(300), np.arange(300))
cx, cy = 150, 150
nx, ny = (x-cx)/cx, (y-cy)/cy
r = np.sqrt((nx)**2 + (ny)**2)
nx2, ny2 = nx * (1 + k1 * r**2), ny * (1 + k1 * r**2)
x2, y2 = nx2*cx + cx, ny2*cy + cy

#single test case for how the a straight line maps to the distorted image
res     = np.zeros((300, 300))
test_img = np.zeros((300, 300))
test_img[10:110, 50] = 1
distorted_img = cv2.remap(test_img, x2.astype(np.float32), y2.astype(np.float32), cv2.INTER_LINEAR)
res[50,:] = np.sum(distorted_img[:100, :], axis=0)/300

cv2.imshow("input_image"        , cv2.resize(test_img, (300,300)            , interpolation=cv2.INTER_NEAREST))
cv2.imshow("distorted_image"    , cv2.resize(distorted_img, (300, 300)      , interpolation=cv2.INTER_NEAREST))
cv2.waitKey(0)

test_img = np.zeros((300, 300))
test_img[10:110, 50] = 1
distorted_img = cv2.remap(test_img, x2.astype(np.float32), y2.astype(np.float32), cv2.INTER_LINEAR)
res[50,:] = np.sum(distorted_img[:100, :], axis=0)/300 #100 is the tdi stages!!!!!

#Iterates over different positions of the straight line in the test image, 
#computes the distorted images, and calculates the average intensity along the lines, storing the results in res.

cv2.imshow("res", cv2.resize(distorted_img, (300,300), interpolation=cv2.INTER_NEAREST))
cv2.imshow("distorted", cv2.resize(res/np.max(res), (600, 600), interpolation=cv2.INTER_NEAREST))
cv2.waitKey(0)

for i in range(300):
    test_img = np.zeros((300, 300))
    test_img[10:110, i] = 255 #this is the image moving!!!!!
    distorted_img = cv2.remap(test_img, x2.astype(np.float32), y2.astype(np.float32), cv2.INTER_LINEAR)
    res[i,:] = np.sum(distorted_img[:100, :], axis=0) #/300

   # test_img = np.zeros((300, 300))
    #test_img[100:200, i] = 1
   # distorted_img = cv2.remap(test_img, x2.astype(np.float32), y2.astype(np.float32), cv2.INTER_LINEAR)
    #res2[i,:] = np.sum(distorted_img[100:200, :], axis=0)/300

    #test_img = np.zeros((300, 300))
    #test_img[200:, i] = 1
    #distorted_img = cv2.remap(test_img, x2.astype(np.float32), y2.astype(np.float32), cv2.INTER_LINEAR)
    #res3[i,:] = np.sum(distorted_img[200:, :], axis=0)/300

#show testimage and distorted image
cv2.imshow("res", cv2.resize(test_img, (300,300), interpolation=cv2.INTER_NEAREST))
cv2.imshow("distorted", cv2.resize(res/np.max(res), (600, 600), interpolation=cv2.INTER_NEAREST))
cv2.waitKey(0)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(res.mean(axis=1), color='blue')  # Plot the average intensity along the line
plt.title('Average Intensity Along the Straight Line')
plt.xlabel('Position along the Line')
plt.ylabel('Average Intensity')
plt.grid(True)
plt.show()

test_img = np.zeros((300, 300))
test_img[:100, :] = np.broadcast_to(np.random.rand(300), (100, 300))

cv2.imshow("res", cv2.resize(test_img, (200, 200), interpolation=cv2.INTER_NEAREST))
cv2.waitKey(0)

distorted_img = cv2.remap(test_img, x2.astype(np.float32), y2.astype(np.float32), cv2.INTER_LINEAR)

cv2.imshow("res", cv2.resize(distorted_img, (200, 200), interpolation=cv2.INTER_NEAREST))
cv2.waitKey(0)

tdi = np.sum(distorted_img[:100, :], axis=0)

norm_test = test_img[0, :]/np.sum(test_img[0, :])
norm_tdi = tdi/np.sum(tdi)

# IMPORTANT: np.linalg.pinv(res.T)
norm_corr = (np.linalg.pinv(res.T)@tdi)/np.sum(np.linalg.pinv(res.T)@tdi)

#norm_corr = np.linalg.lstsq(res.T, tdi, rcond=None)[0]/np.sum(np.linalg.lstsq(res.T, tdi, rcond=None)[0])

print(norm_test)
print(norm_tdi)
test_img[100:200,:] = np.broadcast_to(norm_tdi, (100, 300))/np.amax(norm_tdi)

print(norm_corr)
test_img[200:,:] = np.broadcast_to(norm_corr, (100, 300))/np.amax(norm_corr)

cv2.imshow("res", cv2.resize(np.broadcast_to(test_img,(300,300)), (600, 600), interpolation=cv2.INTER_NEAREST))
cv2.waitKey(0)

# print rms norm_test to norm_tdi
print(np.sqrt(np.mean((norm_test - norm_tdi)**2)))
# print rms norm_test to norm_corr
print(np.sqrt(np.mean((norm_test - norm_corr)**2)))

out_image = np.zeros((800, 300, 3))

img = cv2.imread(r"C:\Users\rowan\Documents\masters\sw_comparison\lens_distortion_model\source_images\checkerboard_4000x4000.png")

for offset in range(800-300-1):
    # read test.png into numpy array RGB
    
    # take a 300x300 crop of the original image at y=offset
    crop = img[offset:offset+300, :, :]

    # distort this
    distorted_img = cv2.remap(crop, x2.astype(np.float32), y2.astype(np.float32), cv2.INTER_LINEAR)

    # isolate red in top 1/3, green in middle, blue in bottom with vstack as if using filters in a scanning sattelite
    distorted_img[:100,:,1:] = 0
    distorted_img[100:200,:,0] = 0
    distorted_img[100:200,:,2] = 0
    distorted_img[200:,:,0:2] = 0

    #cv2.imshow("res", cv2.resize(distorted_img, (600, 600), interpolation=cv2.INTER_NEAREST))
    #cv2.waitKey(0)

    # add distorted image to out image at offset
    out_image[offset:offset+300, :, :] += distorted_img

distorted_test = out_image[200:600,:, :]/25500.0

cv2.imshow("res", cv2.resize(distorted_test, (600, 800), interpolation=cv2.INTER_NEAREST))
cv2.waitKey(0)


fixed_image = np.zeros((400, 300, 3))

print(np.average(distorted_test[:, :, 0]))

mat = np.linalg.pinv(res.T)
mat2 = np.linalg.pinv(res2.T)
mat3 = np.linalg.pinv(res3.T)
print(mat)
for i in range(400):
    fixed_image[i, :, 0] = mat@distorted_test[i, :, 0]
    fixed_image[i, :, 1] = mat2@distorted_test[i, :, 1]
    fixed_image[i, :, 2] = mat3@distorted_test[i, :, 2]

print(np.average(fixed_image[:, :, 0]))

fixed_image[:,:,0] = fixed_image[:,:,0]/np.average(fixed_image[:,:,0])*np.average(distorted_test[:,:,0])
fixed_image[:,:,1] = fixed_image[:,:,1]/np.average(fixed_image[:,:,1])*np.average(distorted_test[:,:,1])
fixed_image[:,:,2] = fixed_image[:,:,2]/np.average(fixed_image[:,:,2])*np.average(distorted_test[:,:,2])

cv2.imshow("res", cv2.resize(fixed_image, (600, 800), interpolation=cv2.INTER_NEAREST))
cv2.waitKey(0)


