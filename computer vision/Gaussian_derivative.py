import numpy as np
import cv2.cv2 as cv
import matplotlib.pyplot as plt
import math as m


# 高斯一阶对x偏微分
def G_x(x, y, sigma):
    g_x = -np.exp(-(x**2 + y**2) / (2*sigma**2)) * x / (2*np.pi*sigma**4)
    return g_x

# 高斯一阶对y偏微分
def G_y(x, y, sigma):
    g_y = -np.exp(-(x**2 + y**2) / (2*sigma**2)) * y / (2*np.pi*sigma ** 4)
    return g_y

# 生成二维距离矩阵
def generate_distance_matrix_2d(size):
    distance_matrix = []
    for i in range(size):
        distance_matrix.append([])
        for j in range(size):
            distance_matrix[i].append(((j - (size-1)/2), (((size-1)/2) - i)))
    return distance_matrix

# 生成二维高斯一阶微分模板
def generate_gaussian_kernel_2d(distance_matrix, size, sigma):
    gaussian_x_kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            tmp = distance_matrix[i][j]
            gaussian_x_kernel[i][j] = G_x(tmp[0], tmp[1], sigma)

    gaussian_y_kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            tmp = distance_matrix[i][j]
            gaussian_y_kernel[i][j] = G_y(tmp[0], tmp[1], sigma)
    return gaussian_x_kernel, gaussian_y_kernel

# sigma1 = 0.8
# kernel_size1 = 5
# sigma2 = 3.5
# kernel_size2 = 21
# sigma3 = 6.8
# kernel_size3 = 41
# distance_matrix1 = generate_distance_matrix_2d(kernel_size1)
# distance_matrix2 = generate_distance_matrix_2d(kernel_size2)
# distance_matrix3 = generate_distance_matrix_2d(kernel_size3)
# g_x_kernel1, g_y_kernel1 = generate_gaussian_kernel_2d(distance_matrix1, kernel_size1, sigma1)
# g_x_kernel2, g_y_kernel2 = generate_gaussian_kernel_2d(distance_matrix2, kernel_size2, sigma2)
# g_x_kernel3, g_y_kernel3 = generate_gaussian_kernel_2d(distance_matrix3, kernel_size3, sigma3)
# plt.figure('kernel')
# plt.xticks([]), plt.yticks([])
# plt.subplot(231), plt.imshow(g_x_kernel1, cmap='gray')
# plt.title('sigma=0.8,x')
# plt.subplot(234), plt.imshow(g_y_kernel1, cmap='gray')
# plt.title('sigma=0.8,y')
# plt.subplot(232), plt.imshow(g_x_kernel2, cmap='gray')
# plt.title('sigma=3.5,x')
# plt.subplot(235), plt.imshow(g_y_kernel2, cmap='gray')
# plt.title('sigma=3.5,y')
# plt.subplot(233), plt.imshow(g_x_kernel3, cmap='gray')
# plt.title('sigma=6.8,x')
# plt.subplot(236), plt.imshow(g_y_kernel3, cmap='gray')
# plt.title('sigma=6.8,y')
# plt.show()


# 获取图像
img = cv.imread('C:/Users/Admin/Desktop/keqing.png', cv.IMREAD_GRAYSCALE)

# 设置参数
sigma = 0.5
kernel_size = 5

# 生成高斯一阶微分模板
distance_matrix = generate_distance_matrix_2d(kernel_size)
g_x_kernel, g_y_kernel = generate_gaussian_kernel_2d(distance_matrix, kernel_size, sigma)
# padding
padding_size = int((kernel_size-1)/2)
h, w = img.shape
dst = np.zeros((h+kernel_size-1, w+kernel_size-1), dtype=np.uint8)
dst[padding_size:h+padding_size, padding_size:w+padding_size] = img[:, :]

# 生成水平梯度图和垂直梯度图
res_x = np.zeros((h, w), dtype=np.uint8)
res_y = np.zeros((h, w), dtype=np.uint8)
for y in range(h):
    for x in range(w):
        res_x[y, x] = np.sum(dst[y:y+kernel_size, x:x+kernel_size] * g_x_kernel)
        res_y[y, x] = np.sum(dst[y:y+kernel_size, x:x+kernel_size] * g_y_kernel)

# 生成幅度图和相位图
res_magnitude = np.zeros((h, w), dtype=np.float32)
res_phase = np.zeros((h, w), dtype=np.float32)
for y1 in range(h):
    for x1 in range(w):
        res_magnitude[y1, x1] = np.sqrt(res_x[y1, x1]**2 + res_y[y1, x1]**2)
        res_phase[y1, x1] = np.degrees(np.arctan2(res_y[y1, x1], res_x[y1, x1]))

# 归一化
min_magnitude = np.min(res_magnitude)
max_magnitude = np.max(res_magnitude)
delta_magnitude = max_magnitude - min_magnitude
min_phase = np.min(res_phase)
max_phase = np.max(res_phase)
delta_phase = max_phase - min_phase
res_magnitude = (res_magnitude - min_magnitude) / delta_magnitude
res_phase = (res_phase - min_phase) / delta_phase



plt.subplot(121), plt.imshow(res_x, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title('Ix')
plt.subplot(122), plt.imshow(res_y, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title('Iy')
plt.show()


# # display
# plt.figure('res')
# plt.subplot(131), plt.imshow(img, cmap='gray')
# plt.xticks([]), plt.yticks([])
# plt.title('source image')
# plt.subplot(132), plt.imshow(res_magnitude, cmap='gray')
# plt.xticks([]), plt.yticks([])
# plt.title('magnitude spectrum')
# plt.subplot(133), plt.imshow(res_phase, cmap='gray')
# plt.xticks([]), plt.yticks([])
# plt.title('phase spectrum')
# plt.show()