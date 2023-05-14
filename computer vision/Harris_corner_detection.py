import numpy as np
import cv2.cv2 as cv
import matplotlib.pyplot as plt
from parametric_geometric_transformation import backward_transformation
from parametric_geometric_transformation import translation, rotation, euclidean, affine

# two-dimension Gaussian function
def func_gaussian_2d(x, y, sigma):
    return np.exp(-(x**2 + y**2)/(2*sigma**2)) / (2*np.pi*sigma**2)

# generate 2-dimension distance matrix
def generate_distance_matrix_2d(size):
    distance_matrix = []
    for i in range(size):
        distance_matrix.append([])
        for j in range(size):
            distance_matrix[i].append(((j - (size-1)/2), (((size-1)/2) - i)))
    return distance_matrix

# generate 2-dimension Gaussian filtering kernel
def generate_gaussian_kernel_2d(distance_matrix, size, sigma):
    gaussian_kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            tmp = distance_matrix[i][j]
            gaussian_kernel[i][j] = func_gaussian_2d(tmp[0], tmp[1], sigma)
    total_value = np.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel / total_value
    return gaussian_kernel

# 高斯滤波
def gaussian_filetring(src, kernel_size, gaussian_kernel):
    padding_size = int((kernel_size-1)/2)
    h, w = src.shape
    dst = np.zeros((h+kernel_size-1, w+kernel_size-1), dtype=np.float64)
    dst[padding_size:h+padding_size, padding_size:w+padding_size] = src[:, :]
    res = np.zeros((h, w), dtype=np.float64)
    for y in range(h):
        for x in range(w):
            res[y, x] = np.sum(dst[y:y+kernel_size, x:x+kernel_size]*gaussian_kernel)
    return res

# 遍历移动函数
def moving(g, gg, j, k, m, n):
    """
    :param g: 待遍历的图
    :param gg: 遍历与否的记录图，0为已遍历，1为未遍历
    :param j: 遍历到的点的行坐标
    :param k: 遍历到的点的列坐标
    :param m: 图的行数
    :param n:  图的列数
    :return: 当前点所在连通域的所有点
    """
    q_row = [j]  # 遍历到的点的行坐标
    q_col = [k]  # 遍历到的点的列坐标
    site_list = []
    while len(q_row) != 0:
        j = q_row[0]; k = q_col[0]
        site_list.append((j, k))
        gg[j, k] = 0
        if k-1 >= 0:
            if g[j, k-1] != 0 and gg[j, k-1] == 1:
                q_row.append(j); q_col.append(k-1); gg[j, k-1] = 0
        if k+1 < n:
            if g[j, k+1] != 0 and gg[j, k+1] == 1:
                q_row.append(j); q_col.append(k+1); gg[j, k+1] = 0
        if j-1 >= 0:
            if g[j-1, k] != 0 and gg[j-1, k] == 1:
                q_row.append(j-1); q_col.append(k); gg[j-1, k] = 0
        if j+1 < m:
            if g[j+1, k] != 0 and gg[j+1, k] == 1:
                q_row.append(j+1); q_col.append(k); gg[j+1, k] = 0
        del q_row[0]
        del q_col[0]
    return site_list


# get image
img = cv.imread('C:/Users/Admin/Desktop/test.png', cv.IMREAD_GRAYSCALE)
h, w = img.shape
img1 = np.zeros((h, w), dtype=np.uint8)
img1[:, :] = img[:, :]


# translation_matrix = euclidean(60, 100, 200)
# img2 = backward_transformation(img, translation_matrix, 1)




# set parameters
sigma = 0.6
kernel_size = 3

# 生成高斯滤波核
distance_matrix = generate_distance_matrix_2d(kernel_size)
g_kernel= generate_gaussian_kernel_2d(distance_matrix, kernel_size, sigma)
# 高斯滤波
res1 = gaussian_filetring(img1, kernel_size, g_kernel)
res = res1.astype(np.uint8)

t1 = cv.getTickCount()


# 梯度计算
grad_x = cv.Sobel(res, cv.CV_64F, 1, 0)  # 计算水平梯度，Sobel算子：[[-1 0 1],[-2 0 2],[-1 0 1]]
grad_y = cv.Sobel(res, cv.CV_64F, 0, 1)  # 计算垂直梯度，Sobel算子：[[-1 -2 -1],[0 0 0],[1 2 1]]
grad_xx = grad_x * grad_x
grad_xy = grad_x * grad_y
grad_yy = grad_y * grad_y
# 对二阶差分梯度矩阵做高斯滤波
gradient_xx = gaussian_filetring(grad_xx, kernel_size, g_kernel)
gradient_xy = gaussian_filetring(grad_xy, kernel_size, g_kernel)
gradient_yy = gaussian_filetring(grad_yy, kernel_size, g_kernel)


# 构造二阶矩矩阵M
alpha = 0.05
win_size = 5  # 移动窗口大小
h, w = img.shape
win_func = np.ones((win_size, win_size), dtype=np.float64)  # 窗口函数
M = np.zeros((h, w, 3), dtype=np.float64)  # [Ixx, Ixy, Iyy]
det_M = np.zeros((h, w), dtype=np.float64)  # 矩阵M的行列式
trace_M = np.zeros((h, w), dtype=np.float64)  # 矩阵M的迹
R = np.zeros((h, w), dtype=np.float64)
s = int((win_size-1)/2)
mark_points = np.zeros((h, w))  # 标记角点
for y in range(s, h-s):
    for x in range(s, w-s):
        M[y, x, 0] = np.sum(win_func*gradient_xx[y-s:y+s+1, x-s:x+s+1])
        M[y, x, 1] = np.sum(win_func*gradient_xy[y-s:y+s+1, x-s:x+s+1])
        M[y, x, 2] = np.sum(win_func*gradient_yy[y-s:y+s+1, x-s:x+s+1])
        det_M[y, x] = M[y, x, 0]*M[y, x, 2] - M[y, x, 1]*M[y, x, 1]
        trace_M[y, x] = M[y, x, 0] + M[y, x, 2]
        R[y, x] = det_M[y, x] - alpha*trace_M[y, x]*trace_M[y, x]

# 设置正阈值分割角点
T = np.max(R) * 0.01  # 正阈值
for y1 in range(h):
    for x1 in range(w):
        if R[y1, x1] > T:
            mark_points[y1, x1] = 1

# 非极大值抑制
record_map = np.ones((h, w))  # 遍历记录图
mark_list = []  # 某一连通域的标记点集合
value_list = []  # 对应连通域的标记点值的集合
for m in range(h):
    for n in range(w):
        print((m, n))
        if record_map[m, n] == 0:
            continue
        elif mark_points[m, n] == 0:
            record_map[m, n] = 0
        else:
            p = m
            q = n
            mark_list = []
            value_list = []
            mark_list = moving(mark_points, record_map, m, n, h, w)  # 步进到连通域中
            for z1 in range(len(mark_list)):
                value_list.append(R[mark_list[z1][0], mark_list[z1][1]])
            max_value = max(value_list)
            for z2 in range(len(mark_list)):
                # 选取连通域中值最大的点为角点
                if R[mark_list[z2][0], mark_list[z2][1]] != max_value:
                    mark_points[mark_list[z2][0], mark_list[z2][1]] = 0
            m = p
            n = q



# 在原图中标记角点
img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
for y2 in range(h):
    for x2 in range(w):
        if mark_points[y2, x2] == 1:
            img[y2, x2] = [0, 0, 255]


# t1 = cv.getTickCount()

# opencv cornerHarris
res = res.astype(np.float32)
res1 = cv.cornerHarris(res, 3, 3, k=0.04)
res1 = cv.dilate(res1, None)
img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
img[res1 > 0.01*res1.max()] = [0, 0, 255]

# t2 = cv.getTickCount()
# print((t2-t1)/cv.getTickFrequency())









# # 生成高斯滤波核
# distance_matrix = generate_distance_matrix_2d(kernel_size)
# g_kernel= generate_gaussian_kernel_2d(distance_matrix, kernel_size, sigma)
# # 高斯滤波
# res2 = gaussian_filetring(img2, kernel_size, g_kernel)
# res2 = res2.astype(np.uint8)
#
# # 梯度计算
# grad_x2 = cv.Sobel(res2, cv.CV_64F, 1, 0)  # 计算水平梯度，Sobel算子：[[-1 0 1],[-2 0 2],[-1 0 1]]
# grad_y2 = cv.Sobel(res2, cv.CV_64F, 0, 1)  # 计算垂直梯度，Sobel算子：[[-1 -2 -1],[0 0 0],[1 2 1]]
# grad_xx2 = grad_x2 * grad_x2
# grad_xy2 = grad_x2 * grad_y2
# grad_yy2 = grad_y2 * grad_y2
#
# # 对二阶差分梯度矩阵做高斯滤波
# gradient_xx2 = gaussian_filetring(grad_xx2, kernel_size, g_kernel)
# gradient_xy2 = gaussian_filetring(grad_xy2, kernel_size, g_kernel)
# gradient_yy2 = gaussian_filetring(grad_yy2, kernel_size, g_kernel)
#
# # 构造二阶矩矩阵M
# alpha = 0.05
# win_size = 3  # 移动窗口大小
# h, w = img2.shape
# win_func = np.ones((win_size, win_size), dtype=np.float64)  # 窗口函数
# M2 = np.zeros((h, w, 3), dtype=np.float64)  # [Ixx, Ixy, Iyy]
# det_M2 = np.zeros((h, w), dtype=np.float64)  # 矩阵M的行列式
# trace_M2 = np.zeros((h, w), dtype=np.float64)  # 矩阵M的迹
# R2 = np.zeros((h, w), dtype=np.float64)
# s = int((win_size-1)/2)
# mark_points2 = np.zeros((h, w))  # 标记角点
# for y in range(s, h-s):
#     for x in range(s, w-s):
#         M2[y, x, 0] = np.sum(win_func*gradient_xx2[y-s:y+s+1, x-s:x+s+1])
#         M2[y, x, 1] = np.sum(win_func*gradient_xy2[y-s:y+s+1, x-s:x+s+1])
#         M2[y, x, 2] = np.sum(win_func*gradient_yy2[y-s:y+s+1, x-s:x+s+1])
#         det_M2[y, x] = M2[y, x, 0]*M2[y, x, 2] - M2[y, x, 1]*M2[y, x, 1]
#         trace_M2[y, x] = M2[y, x, 0] + M2[y, x, 2]
#         R2[y, x] = det_M2[y, x] - alpha*trace_M2[y, x]*trace_M2[y, x]
#
# # 设置正阈值分割角点
# T = np.max(R2) * 0.01  # 正阈值
# mark_list = []
# img2 = cv.cvtColor(img2, cv.COLOR_GRAY2RGB)
# for y1 in range(h):
#     for x1 in range(w):
#         if R2[y1, x1] > T:
#             mark_points2[y1, x1] = 1
#             img2[y1, x1] = [0, 0, 255]








img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
# img2 = cv.cvtColor(img2, cv.COLOR_BGR2RGB)
plt.subplot(121), plt.imshow(img1)
plt.xticks([]), plt.yticks([])
plt.title('source')
# plt.subplot(122), plt.imshow(img2)
# plt.xticks([]), plt.yticks([])
# plt.title('euclidean transformation')
plt.show()

