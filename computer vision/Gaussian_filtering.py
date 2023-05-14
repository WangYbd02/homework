import numpy as np
import math as m
import cv2.cv2 as cv
from matplotlib import pyplot as plt
from filtering_func import generate_distance_matrix_2d, generate_gaussian_kernel_2d, padding, conv

# set parameters
sigma = 1.2  # standard deviation
size = 7  # size of filtering kernel

# generate distance matrix
distance_matrix = generate_distance_matrix_2d(size)
# generate Gaussian filtering kernel
Gaussian_kernel = generate_gaussian_kernel_2d(distance_matrix, size, sigma)

# get image
img = cv.imread('C:/Users/Admin/Desktop/test_pic4.jpg', cv.IMREAD_COLOR)
img = cv.resize(img, (100,100))

# filtering
dst = padding(img, size, 1)  # padding
size_of_padding = int((size-1)/2)
res = np.zeros(img.shape, dtype=np.uint8)
# convolution
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        res[i][j] = conv(dst[i:i+2*size_of_padding+1, j:j+2*size_of_padding+1, :], Gaussian_kernel, size)

# display
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
res = cv.cvtColor(res, cv.COLOR_BGR2RGB)

plt.figure('result')
plt.subplot(231), plt.imshow(img)
plt.xticks([]), plt.yticks([])
plt.title('source image')
plt.subplot(232), plt.imshow(res)
plt.xticks([]), plt.yticks([])
plt.title('clip filter')

plt.show()
