    # -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread('img/test.jpg')
print( " Image Type",type(image),
      "Image dimensions", image.shape)

ysize = image.shape[0]
xsize = image.shape[1]

color_select = np.copy(image)
line_image =np.copy(image)
test_image =np.copy(image)

red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

left_bottom = [0, 539]
right_bottom = [900, 539]
apex = [475, 320]

#making line functions
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# Mask pixels below the threshold
color_thresholds = ((image[:,:,0] < rgb_threshold[0])\
              |(image[:,:,1] < rgb_threshold[1])\
                  |(image[:,:,2] < rgb_threshold[2]))
    
# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))

#Region Masking
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))
# Mask color selection
color_select[color_thresholds | ~region_thresholds] = [0,0,0]
# Find where image is both colored right and in the region
line_image[~color_thresholds & region_thresholds] = [255,0,0]

                  
# Display the image                 
plt.imshow(image)
x = [left_bottom[0], right_bottom[0], apex[0], left_bottom[0]]
y = [left_bottom[1], right_bottom[1], apex[1], left_bottom[1]]
plt.plot(x, y)
plt.imshow(color_select)
plt.imshow(line_image)

cv2.imwrite("img/test-after.png", color_select)
cv2.imwrite("img/test-after-line.png", line_image)

#image_after= cv2.imread('img/test-after.png')




gray_image=cv2.cvtColor(test_image, cv2.COLOR_RGB2GRAY) #grayscale conversion
plt.imshow(gray_image, cmap='gray')

#Gaussian Blur
kernel_size =3
blur_image = cv2.GaussianBlur(gray_image,(kernel_size,kernel_size),0)


#Canny Edge Detection
low_threshold = 50
high_threshold = 150
edges = cv2.Canny(blur_image, low_threshold, high_threshold )

#Display the image
plt.imshow(edges, cmap='Greys_r')

# Next we'll create a masked edges image using cv2.fillPoly()
mask = np.zeros_like(edges)   
ignore_mask_color = 255  

# This time we are defining a four sided polygon to mask
imshape = image.shape
vertices = np.array([[(0,imshape[0]),(475, 335), (480, 335), (imshape[1],imshape[0])]], dtype=np.int32)
mask = cv2.fillPoly(mask, vertices, ignore_mask_color)
masked_edges = cv2.bitwise_and(edges, mask)

plt.imshow(masked_edges, cmap='Greys_r')

#Hough Transforam to find lane lines

rho = 2
theta = np.pi/180
threshold = 15
min_line_length = 10
max_line_gap = 20
line_image = np.copy(image)*0 #creating a blank to draw lines on

# Run Hough on edge detected image
lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

# Iterate over the output "lines" and draw lines on a blank image
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
        
        
# Draw the lines on the edge image
masked_rgb = np.dstack((masked_edges,masked_edges,masked_edges))
lines_edges = cv2.addWeighted(masked_rgb, 0.8, line_image, 1, 0) 
plt.imshow(lines_edges)

