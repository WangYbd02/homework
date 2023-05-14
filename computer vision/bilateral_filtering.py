import matplotlib.pyplot as plt
import numpy as np
import math as m
import cv2.cv2 as cv
from filtering_func import generate_distance_matrix_2d, generate_gaussian_kernel_2d, padding, func_gaussian_1d

# set parameters
sigma_s = 2  # sigma of space operator
sigma_r = 30  # sigma of pixel operator
size = 5  # size of filtering kernel
sigma_s1 = 2
sigma_r1 = 5
sigma_s2 = 10
sigma_r2 = 30

# get image
img = cv.imread('C:/Users/Admin/Desktop/test.jpg', cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (100, 100))

# # define bilateral filtering convolution function
# def bilateral_conv(src, gaussian_kernel, size, sigma_r):
#     """
#     @ bilateral filtering fundamental convolution function
#     :param src: ndarray. input image
#     :param gaussian_kernel: ndarray. a two dimension Gaussian filtering kernel
#     :param size: int. kernel size.
#     :param sigma_r: float. sigma of pixel operator
#     :return: float. convolution result
#     """
#     conv_res = [0, 0, 0]  # convolution result
#     kernel = [0, 0, 0]  # Gaussian kernel * pixel kernel
#     w = [0, 0, 0]  # sum of weights
#     center = int((size-1)/2)  # center position of kernel
#     for i in range(size):
#         for j in range(size):
#             kernel[0] = gaussian_kernel[i][j]*func_gaussian_1d(src[i][j][0]-src[center][center][0], sigma_r)
#             kernel[1] = gaussian_kernel[i][j]*func_gaussian_1d(src[i][j][1]-src[center][center][1], sigma_r)
#             kernel[2] = gaussian_kernel[i][j]*func_gaussian_1d(src[i][j][2]-src[center][center][2], sigma_r)
#             conv_res[0] = conv_res[0] + kernel[0]*src[i][j][0]
#             conv_res[1] = conv_res[1] + kernel[1]*src[i][j][1]
#             conv_res[2] = conv_res[2] + kernel[2]*src[i][j][2]
#             w[0] = w[0] + kernel[0]
#             w[1] = w[1] + kernel[1]
#             w[2] = w[2] + kernel[2]
#     conv_res[0] = conv_res[0] / w[0]
#     conv_res[1] = conv_res[1] / w[1]
#     conv_res[2] = conv_res[2] / w[2]
#     return conv_res

# define bilateral filtering convolution function
def bilateral_conv(src, gaussian_kernel, size, sigma_r):
    """
    @ bilateral filtering fundamental convolution function
    :param src: ndarray. input image
    :param gaussian_kernel: ndarray. a two dimension Gaussian filtering kernel
    :param size: int. kernel size.
    :param sigma_r: float. sigma of pixel operator
    :return: float. convolution result
    """
    conv_res = 0
    kernel = 0
    w = 0
    center = int((size-1)/2)
    for i in range(size):
        for j in range(size):
            kernel = gaussian_kernel[i][j]*func_gaussian_1d(src[i][j]-src[center][center], sigma_r)
            conv_res = conv_res + kernel*src[i][j]
            w = w + kernel
    conv_res = conv_res / w
    return conv_res


# generate distance matrix
distance_matrix = generate_distance_matrix_2d(size)
# generate Gaussian filtering kernel
Gaussian_kernel = generate_gaussian_kernel_2d(distance_matrix, size, sigma_s)
kernel = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        kernel[i][j] = Gaussian_kernel[i][j]*func_gaussian_1d(img[35+i][63+j]-img[37][65], sigma_r)
sumsss = np.sum(kernel)
kernel = kernel /sumsss
print(np.sum(kernel))
print(kernel)


# generate distance matrix
distance_matrix = generate_distance_matrix_2d(size)
# generate Gaussian filtering kernel
Gaussian_kernel = generate_gaussian_kernel_2d(distance_matrix, size, sigma_s)
# filtering
# dst = padding(img, size, 1)
w, h = img.shape
dst = np.zeros((w+size-1, h+size-1), dtype=np.uint8)
size_of_padding = int((size-1)/2)
dst[size_of_padding:w+size_of_padding, size_of_padding:h+size_of_padding] = img[:, :]
res = np.zeros(img.shape, dtype=np.uint8)
# convolution
for i in range(w):
    for j in range(h):
        res[i][j] = bilateral_conv(dst[i:i+size, j:j+size], Gaussian_kernel, size, sigma_r)


# distance_matrix1 = generate_distance_matrix_2d(size)
# Gaussian_kernel1 = generate_gaussian_kernel_2d(distance_matrix1, size, sigma_s)
# dst1 = np.zeros((img.shape[0]+size-1, img.shape[1]+size-1), dtype=np.uint8)
# size_of_padding = int((size-1)/2)
# w, h =img.shape
# dst1[size_of_padding:w+size_of_padding, size_of_padding:h+size_of_padding] = img[:, :]
# res1 = np.zeros(img.shape, dtype=np.uint8)
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         res1[i][j] = bil(dst[i:i+size, j:j+size], Gaussian_kernel1, size, sigma_r1)
#
# distance_matrix2 = generate_distance_matrix_2d(size)
# Gaussian_kernel2 = generate_gaussian_kernel_2d(distance_matrix2, size, sigma_s1)
# dst2 = np.zeros((img.shape[0]+size-1, img.shape[1]+size-1), dtype=np.uint8)
# size_of_padding = int((size-1)/2)
# w, h =img.shape
# dst2[size_of_padding:w+size_of_padding, size_of_padding:h+size_of_padding] = img[:, :]
# res2 = np.zeros(img.shape, dtype=np.uint8)
# for i in range(img.shape[0]):
#     for j in range(img.shape[1]):
#         res2[i][j] = bil(dst2[i:i+size, j:j+size], Gaussian_kernel2, size, sigma_r2)




# display
plt.figure('result')
plt.subplot(221), plt.imshow(img, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title('source image')
plt.subplot(222), plt.imshow(res, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title('sigma_s=2 sigma_r=30')
# plt.subplot(223), plt.imshow(res1, cmap='gray')
# plt.xticks([]), plt.yticks([])
# plt.title('sigma_s=2 sigma_r=5')
# plt.subplot(224), plt.imshow(res2, cmap='gray')
# plt.xticks([]), plt.yticks([])
# plt.title('sigma_s=10 sigma_r=30')
plt.show()
