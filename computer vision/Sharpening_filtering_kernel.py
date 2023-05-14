import matplotlib.pyplot as plt
import numpy as np
import math as m
import cv2.cv2 as cv
from filtering_func import generate_distance_matrix_2d, generate_gaussian_kernel_2d, padding, conv


# set parameters
sigma = 0.8  # standard deviation
size = 5  # size of filtering kernel
alpha = 0.6  # coefficient alpha
sigma2 = 0.5
size2 = 3

# get image
img = cv.imread('C:/Users/Admin/Desktop/linghua.jpg', cv.IMREAD_GRAYSCALE)

# generate distance matrix
distance_matrix = generate_distance_matrix_2d(size)
# generate Gaussian filtering kernel
Gaussian_kernel = generate_gaussian_kernel_2d(distance_matrix, size, sigma)
# generate sharpening filtering kernel based on Gaussian filtering kernel
all_pass_filter = np.zeros((size, size))  # all-pass filter w
center = int((size-1)/2)
all_pass_filter[center][center] = 1
sharpening_kernel = (1+alpha)*all_pass_filter - alpha*Gaussian_kernel

size_of_padding = int((size-1)/2)
w, h = img.shape
dst = np.zeros((w+2*size_of_padding, h+2*size_of_padding), dtype=np.uint8)
# padding
dst[size_of_padding:w+size_of_padding, size_of_padding:h+size_of_padding] = img[:, :]
# filtering
res = np.zeros(img.shape, dtype=np.uint8)
for i in range(w):
    for j in range(h):
        res[i][j] = np.sum(dst[i:i+size, j:j+size]*sharpening_kernel)
        if res[i][j] > 255:
            res[i][j] = 255
        elif res[i][j] < 0:
            res[i][j] = 0


distance_matrix2 = generate_distance_matrix_2d(size2)
Gaussian_kernel2 = generate_gaussian_kernel_2d(distance_matrix2, size2, sigma2)
all_pass_filter2 = np.zeros((size2, size2))  # all-pass filter w
center2 = int((size2-1)/2)
all_pass_filter2[center2][center2] = 1
sharpening_kernel2 = (1+alpha)*all_pass_filter2 - alpha*Gaussian_kernel2

size_of_padding2 = int((size2-1)/2)
w, h = img.shape
dst2 = np.zeros((w+2*size_of_padding2, h+2*size_of_padding2), dtype=np.uint8)
# padding
dst2[size_of_padding2:w+size_of_padding2, size_of_padding2:h+size_of_padding2] = img[:, :]
# filtering
res2 = np.zeros(img.shape, dtype=np.uint8)
for i in range(w):
    for j in range(h):
        res2[i][j] = np.sum(dst2[i:i+size2, j:j+size2]*sharpening_kernel2)
        if res2[i][j] > 255:
            res2[i][j] = 255
        elif res2[i][j] < 0:
            res2[i][j] = 0






# display
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
res = cv.cvtColor(res, cv.COLOR_BGR2RGB)
res2 = cv.cvtColor(res2, cv.COLOR_BGR2RGB)
plt.figure('result')
plt.subplot(131), plt.imshow(img)
plt.xticks([]), plt.yticks([])
plt.title('source image')
plt.subplot(132), plt.imshow(res2)
plt.xticks([]), plt.yticks([])
plt.title('sigma=0.5 size=3')
plt.subplot(133), plt.imshow(res)
plt.xticks([]), plt.yticks([])
plt.title('sigma=0.8 size=5')
plt.show()
