import numpy as np
import cv2.cv2 as cv
import matplotlib.pyplot as plt


# get image
img = cv.imread('C:/Users/Admin/Desktop/test.jpg', cv.IMREAD_GRAYSCALE)

# 2维高斯滤波核
gaussian_kernel_1d = np.array([1, 4, 6, 4, 1])/16
gaussian_kernel_2d = gaussian_kernel_1d*gaussian_kernel_1d.reshape(-1,1)

# 计算得到高斯金字塔
layers = 4
gaussian_pyramid = []
gaussian_pyramid.append(img)
for i in range(layers):
    h, w = gaussian_pyramid[i].shape
    # padding
    tmp = np.zeros((h+4, w+4), dtype=np.uint8)
    tmp[2:h+2, 2:w+2] = gaussian_pyramid[i][:, :]
    dst = np.zeros((h, w), dtype=np.uint8)
    # 高斯平滑
    for y in range(h):
        for x in range(w):
            dst[y, x] = np.sum(tmp[y:y+5, x:x+5]*gaussian_kernel_2d)
    # 下采样
    down_h = np.ceil(h/2).astype(int)
    down_w = np.ceil(w/2).astype(int)
    res = np.zeros((down_h, down_w), dtype=np.uint8)
    for y1 in range(down_h):
        for x1 in range(down_w):
            res[y1, x1] = dst[2*y1, 2*x1]
    gaussian_pyramid.append(res)


# 计算得到拉普拉斯金字塔
laplace_pyramid = []
upsample_pyramid = []
for i in range(layers):
    h0, w0 = gaussian_pyramid[i].shape
    h1, w1 = gaussian_pyramid[i+1].shape
    # 插入0
    dst = np.zeros((h0, w0), dtype=np.uint8)
    for y in range(h1):
        for x in range(w1):
            dst[2*y, 2*x] = gaussian_pyramid[i+1][y, x]
    # padding
    tmp = np.zeros((h0+4, w0+4), dtype=np.uint8)
    tmp[2:h0+2, 2:w0+2] = dst[:, :]
    # 上采样
    res = np.zeros((h0, w0), dtype=np.uint8)
    for y1 in range(h0):
        for x1 in range(w0):
            res[y1, x1] = 4*np.sum(tmp[y1:y1+5, x1:x1+5]*gaussian_kernel_2d)
    upsample_pyramid.append(res)
    result = gaussian_pyramid[i] - res
    laplace_pyramid.append(result)

# display
plt.figure('res')
plt.subplot(3,4,1), plt.imshow(gaussian_pyramid[0], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title('G0')
plt.subplot(3,4,2), plt.imshow(gaussian_pyramid[1], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title('G1')
plt.subplot(3,4,3), plt.imshow(gaussian_pyramid[2], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title('G2')
plt.subplot(3,4,4), plt.imshow(gaussian_pyramid[3], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title('G3')
plt.subplot(3,4,5), plt.imshow(upsample_pyramid[0], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title('Upsample0')
plt.subplot(3,4,6), plt.imshow(upsample_pyramid[1], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title('Upsample1')
plt.subplot(3,4,7), plt.imshow(upsample_pyramid[2], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title('Upsample2')
plt.subplot(3,4,8), plt.imshow(upsample_pyramid[3], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title('Upsample3')
plt.subplot(3,4,9), plt.imshow(laplace_pyramid[0], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title('L0')
plt.subplot(3,4,10), plt.imshow(laplace_pyramid[1], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title('L1')
plt.subplot(3,4,11), plt.imshow(laplace_pyramid[2], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title('L2')
plt.subplot(3,4,12), plt.imshow(laplace_pyramid[3], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title('L3')
plt.show()
