### refer : https://www.moonbooks.org/Articles/Implementing-a-simple-python-code-to-detect-straight-lines-using-Hough-transform/#myGallery

from turtle import color
import cv2 as cv

import matplotlib.pyplot as plt

file_name = './datas/images/lines.png'
grayImage = cv.imread(file_name, cv.IMREAD_GRAYSCALE)

# 검출 높이기 위해 black And White 변환
(thresh, blackAndWhiteImage) = cv.threshold(grayImage, 127, 255, cv.THRESH_BINARY)
print('blackAndWhiteImage shape: ', blackAndWhiteImage.shape)

plt.imshow(blackAndWhiteImage, cmap=plt.get_cmap('gray'))
plt.show()

# plt.savefig("save_save_source",bbox_inches='tight')
plt.close()

"""##Hough space"""

import numpy as np
import math

img_shape = blackAndWhiteImage.shape

x_max = img_shape[0]
y_max = img_shape[1]

theta_max = 1.0 * math.pi 
theta_min = 0.0

r_min = 0.0
r_max = math.hypot(x_max, y_max)

rows = 200 
cols = 300

hough_space = np.zeros((rows,cols))

for x_list in range(x_max):
    for y_list in range(y_max):
        if blackAndWhiteImage[x_list,y_list] == 255: continue
        for itheta in range(cols):
            theta = 1.0 * itheta * theta_max / cols
            r = x_list * math.cos(theta) + y_list * math.sin(theta)
            ir = round(rows * ( 1.0 * r ) / r_max)
            hough_space[ir,itheta] = hough_space[ir,itheta] + 1

# plt.imshow(hough_space, origin='lower')
plt.xlim(0,cols)
plt.ylim(0,rows)

tick_locs = [y for y in range(0,cols,40)]
tick_lbls = [round( (1.0 * y * theta_max) / cols,1) for y in range(0,cols,40)]
plt.xticks(tick_locs, tick_lbls)

tick_locs = [y for y in range(0,rows,20)]
tick_lbls = [round( (1.0 * y * r_max ) / rows,1) for y in range(0,rows,20)]
plt.yticks(tick_locs, tick_lbls)

plt.xlabel(r'cols')
plt.ylabel(r'rows')
plt.title('Hough Space')

# plt.show()

# plt.savefig("save_hough_space_r_theta.png",bbox_inches='tight')
# plt.close()

"""## Find the maximums"""

neighborhood_size = 10
threshold = 400

import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage

data_max = filters.maximum_filter(hough_space, neighborhood_size)
maxima = (hough_space == data_max)

data_min = filters.minimum_filter(hough_space, neighborhood_size)
diff = ((data_max - data_min) > threshold)
maxima[diff == 0] = 0

labeled, num_objects = ndimage.label(maxima)
slices = ndimage.find_objects(labeled)

x_list, y_list = [], []
for dy,dx in slices:
    x_center = (dx.start + dx.stop - 1)/2
    x_list.append(x_center)
    y_center = (dy.start + dy.stop - 1)/2    
    y_list.append(y_center)

print('coordinatioin - x_list : {}, y_list : {}'.format(x_list,y_list))

plt.imshow(hough_space, origin='lower')

plt.autoscale(False)
plt.plot(x_list,y_list, 'ro')
# plt.savefig('hough_space_maximas.png', bbox_inches = 'tight')

plt.xlabel(r'cols')
plt.ylabel(r'rows')
plt.title('Hough Space')

plt.show()
plt.close()

"""##Plot straight lines"""

line_index = 1
line_extend = 50
for y,x in zip(y_list, x_list):

    r = round( (1.0 * y * r_max ) / rows,1)
    theta = round( (1.0 * x * theta_max) / cols,1)

    fig, ax = plt.subplots()

    ax.imshow(blackAndWhiteImage)

    ax.autoscale(False)

    px = []
    py = []
    for y in range(-y_max-line_extend,y_max+line_extend,1):
        px.append( math.cos(-theta) * y - math.sin(-theta) * r ) 
        py.append( math.sin(-theta) * y + math.cos(-theta) * r )

    ax.plot(px,py, linewidth=5, color="red")

    # plt.savefig("save_image_line_"+ "%02d" % line_index +".png",bbox_inches='tight')

    plt.show()
    # plt.close()

    line_index = line_index + 1