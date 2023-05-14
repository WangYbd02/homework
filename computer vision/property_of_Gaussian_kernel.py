import cv2.cv2 as cv
import math as m
import numpy as np
from matplotlib import pyplot as plt
from filtering_func import generate_distance_matrix_2d, generate_gaussian_kernel_2d
from filtering_func import generate_distance_vector, generate_gaussian_kernel_1d
from filtering_func import clip_filter_padding, conv


# one-dimension Gaussian function
def one_dim_gaussian_func(x, sigma):
    return m.exp(-x**2/(2*sigma**2)) / (sigma*(2*m.pi)**(1/2))


# two-dimension Gaussian function
def two_dim_gaussian_func(x, y, sigma):
    return m.exp(-(x**2 + y**2)/(2*sigma**2)) / (2*m.pi*sigma**2)


# # experiment 1
# # set parameters
# sigma1 = 0.75
# sigma2 = 1.0
# sigma3 = 1.25
# size1 = 5  # size of filtering kernel
#
# # generate distance matrix
# distance_matrix = generate_distance_matrix_2d(size1)
# # generate Gaussian filtering kernel
# # sigma1 = 0.75
# Gaussian_kernel1 = generate_gaussian_kernel_2d(distance_matrix, size1, sigma1)
# # sigma2 = 1.0
# Gaussian_kernel2 = generate_gaussian_kernel_2d(distance_matrix, size1, sigma2)
# # sigma3 = 1.25
# Gaussian_kernel3 = generate_gaussian_kernel_2d(distance_matrix, size1, sigma3)
#
# # get image
# img = cv.imread('C:/Users/Admin/Desktop/test_pic4.jpg', cv.IMREAD_COLOR)
# img = cv.resize(img, (100, 100))
# size_of_padding = int((size1-1)/2)
# # filtering
# dst1 = clip_filter_padding(img, size1)
# # sigma1 = 0.75 & sigma2 = 1.00
# res1 = np.zeros(img.shape, dtype=np.uint8)
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         res1[i][j] = conv(dst1[i:i+2*size_of_padding+1, j:j+2*size_of_padding+1, :], Gaussian_kernel1, size1)
# dst2 = clip_filter_padding(res1, size1)
# res2 = np.zeros(res1.shape, dtype=np.uint8)
# for i in range(res1.shape[0]):
#     for j in range(res1.shape[1]):
#         res2[i][j] = conv(dst2[i:i+2*size_of_padding+1, j:j+2*size_of_padding+1, :], Gaussian_kernel2, size1)
# # sigma3 = 1.25
# res3 = np.zeros(img.shape, dtype=np.uint8)
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         res3[i][j] = conv(dst1[i:i+2*size_of_padding+1, j:j+2*size_of_padding+1, :], Gaussian_kernel3, size1)
#
# # display
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# res1 = cv.cvtColor(res1, cv.COLOR_BGR2RGB)
# res2 = cv.cvtColor(res2, cv.COLOR_BGR2RGB)
# res3 = cv.cvtColor(res3, cv.COLOR_BGR2RGB)
# plt.figure('result')
# plt.subplot(221), plt.imshow(img)
# plt.xticks([]), plt.yticks([])
# plt.title('source image')
# plt.subplot(222), plt.imshow(res3)
# plt.xticks([]), plt.yticks([])
# plt.title('sigma = 1.25')
# plt.subplot(223), plt.imshow(res1)
# plt.xticks([]), plt.yticks([])
# plt.title('sigma = 0.75')
# plt.subplot(224), plt.imshow(res2)
# plt.xticks([]), plt.yticks([])
# plt.title('sigma = 0.75 & sigma = 1.00')
# plt.show()


# # experiment 2
# # set parameters
# sigma = 0.8
# size = 5  # size of filtering kernel
# # get image
# img = cv.imread('C:/Users/Admin/Desktop/test_pic4.jpg', cv.IMREAD_COLOR)
# img = cv.resize(img, (100, 100))
#
# # 2-dimension Gaussian kernel
# # generate distance matrix
# distance_matrix = generate_distance_matrix_2d(size)
# # generate Gaussian filtering kernel
# Gaussian_kernel1 = generate_gaussian_kernel_2d(distance_matrix, size, sigma)
# # filtering
# size_of_padding = int((size-1)/2)
# dst1 = clip_filter_padding(img, size)
# res1 = np.zeros(img.shape, dtype=np.uint8)
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         res1[i][j] = conv(dst1[i:i+2*size_of_padding+1, j:j+2*size_of_padding+1, :], Gaussian_kernel1, size)
#
# # 1-dimension Gaussian kernel
# # generate distance vector
# distance_vector = generate_distance_vector(size)
# # generate Gaussian filtering kernel
# Gaussian_kernel2 = generate_gaussian_kernel_1d(distance_vector, size, sigma)  # row filtering kernel
# Gaussian_kernel3 = generate_gaussian_kernel_1d(distance_vector, size, sigma)  # column filtering kernel
# # filtering
# size_of_padding3 = int((size-1)/2)
# dst2 = clip_filter_padding(img, size)
# res2 = np.zeros((img.shape[0], dst2.shape[1], img.shape[2]), dtype=np.uint8)
# for i in range(dst2.shape[1]):  # row filtering
#     for j in range(img.shape[0]):
#         conv_res = [0, 0, 0]
#         for k in range(size):
#             conv_res[0] = conv_res[0] + dst2[j+k][i][0]*Gaussian_kernel2[k]
#             conv_res[1] = conv_res[1] + dst2[j+k][i][1]*Gaussian_kernel2[k]
#             conv_res[2] = conv_res[2] + dst2[j+k][i][2]*Gaussian_kernel2[k]
#         res2[j][i] = conv_res
# res3 = np.zeros(img.shape, dtype=np.uint8)
# for i in range(img.shape[0]):  # column filtering
#     for j in range(img.shape[1]):
#         conv_res = [0, 0, 0]
#         for k in range(size):
#             conv_res[0] = conv_res[0] + res2[i][j+k][0]*Gaussian_kernel3[k]
#             conv_res[1] = conv_res[1] + res2[i][j+k][1]*Gaussian_kernel3[k]
#             conv_res[2] = conv_res[2] + res2[i][j+k][2]*Gaussian_kernel3[k]
#         res3[i][j] = conv_res
#
# # display
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# res1 = cv.cvtColor(res1, cv.COLOR_BGR2RGB)
# res2 = cv.cvtColor(res2, cv.COLOR_BGR2RGB)
# res3 = cv.cvtColor(res3, cv.COLOR_BGR2RGB)
# plt.figure('result')
# plt.subplot(221), plt.imshow(img)
# plt.xticks([]), plt.yticks([])
# plt.title('source image')
# plt.subplot(222), plt.imshow(res1)
# plt.xticks([]), plt.yticks([])
# plt.title('5*5 kernel')
# plt.subplot(223), plt.imshow(res2)
# plt.xticks([]), plt.yticks([])
# plt.title('1*5 kernel')
# plt.subplot(224), plt.imshow(res3)
# plt.xticks([]), plt.yticks([])
# plt.title('1*5 kernel ane 5*1 kernel')
# plt.show()


# experiment 3
# set parameters
sigma1 = 0.7
sigma2 = 0.8
size = 5  # size of filtering kernel
# get image
img = cv.imread('C:/Users/Admin/Desktop/linghua.jpg', cv.IMREAD_GRAYSCALE)

# generate distance matrix
distance_matrix = generate_distance_matrix_2d(size)
# generate Gaussian filtering kernel
Gaussian_kernel1 = generate_gaussian_kernel_2d(distance_matrix, size, sigma1)
Gaussian_kernel2 = generate_gaussian_kernel_2d(distance_matrix, size, sigma2)
kernel = Gaussian_kernel2 - Gaussian_kernel1  # DOG
size_of_padding = int((size-1)/2)
w, h = img.shape
dst = np.zeros((w+2*size_of_padding, h+2*size_of_padding), dtype=np.uint8)
# padding
dst[size_of_padding:w+size_of_padding, size_of_padding:h+size_of_padding] = img[:, :]
# filtering
res = np.zeros(img.shape, dtype=np.uint8)
for i in range(w):
    for j in range(h):
        res[i][j] = np.sum(dst[i:i+size, j:j+size]*kernel)

# display
plt.figure('result')
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title('source image')
plt.subplot(132), plt.imshow(res, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title('sigma2(0.8)-sigma1(0.7)')
plt.show()

