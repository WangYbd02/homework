import numpy as np
import cv2.cv2 as cv
import matplotlib.pyplot as plt


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

def bilinear_interpolation(src, x, y):
    max_x = np.ceil(x).astype(int)
    min_x = np.floor(x).astype(int)
    max_y = np.ceil(y).astype(int)
    min_y = np.floor(y).astype(int)
    if min_x != max_x and min_y != max_y:
        x1 = max_x - x;  x2 = x - min_x
        y1 = max_y - y;  y2 = y - min_y
        value = np.round(x1*y1*src[min_y][min_x]+x1*y2*src[max_y][min_x]+
                         x2*y1*src[min_y][max_x]+x2*y2*src[max_y][max_x])
    elif min_x == max_x and min_y == max_y:
        value = src[min_y][min_x]
    elif min_x == max_x and min_y != max_y:
        y1 = max_y - y; y2 = y - min_y
        value = np.round(y1*src[min_y][min_x]+y2*src[max_y][min_x])
    else:
        x1 = max_x - x; x2 = x - min_x
        value = np.round(x1*src[min_y][min_x]+x2*src[min_y][max_x])
    return value


# get image
img = cv.imread('C:/Users/Admin/Desktop/test.png', cv.IMREAD_GRAYSCALE)


# set parameters
sigma = 0.6
kernel_size = 3

# 生成高斯滤波核
distance_matrix = generate_distance_matrix_2d(kernel_size)
g_kernel= generate_gaussian_kernel_2d(distance_matrix, kernel_size, sigma)
# padding
padding_size = int((kernel_size-1)/2)
h, w = img.shape
dst = np.zeros((h+kernel_size-1, w+kernel_size-1), dtype=np.uint8)
dst[padding_size:h+padding_size, padding_size:w+padding_size] = img[:, :]
res = np.zeros((h, w), dtype=np.uint8)
# 高斯滤波
for y in range(h):
    for x in range(w):
        res[y, x] = np.sum(dst[y:y+kernel_size, x:x+kernel_size]*g_kernel)

t1 = cv.getTickCount()

# 梯度计算
grad_x = cv.Sobel(res, cv.CV_64F, 1, 0)  # 计算水平梯度，Sobel算子：[[-1 0 1],[-2 0 2],[-1 0 1]]
grad_y = cv.Sobel(res, cv.CV_64F, 0, 1)  # 计算垂直梯度，Sobel算子：[[-1 -2 -1],[0 0 0],[1 2 1]]
# grad_x = cv.Scharr(res, cv.CV_64F, 1, 0)  # 计算水平梯度，Scharr算子：[[-3 0 3],[-10 0 10],[-3 0 3]]
# grad_y = cv.Scharr(res, cv.CV_64F, 0, 1)  # 计算垂直梯度，Scharr算子：[[-3 -10 -3],[0 0 10],[3 10 3]]
grad_value = np.sqrt(grad_x*grad_x + grad_y*grad_y)  # 计算梯度大小
grad_phase = cv.phase(grad_x, grad_y, angleInDegrees=True)  # 计算梯度方向

# 非极大值抑制
NMS_grad = np.copy(grad_value)
grad_value_padding = np.zeros((h+2, w+2))
grad_value_padding[1:h+1, 1:w+1] = grad_value[:, :]

# 非极大值抑制
for y1 in range(1, h+1):
    for x1 in range(1, w+1):
        p = grad_value_padding[y1, x1]
        direct = grad_phase[y1-1, x1-1]
        delta_x = np.cos(np.radians(direct))
        delta_y = np.sin(np.radians(direct))
        r = bilinear_interpolation(grad_value_padding, x1+delta_x, y1+delta_y)
        q = bilinear_interpolation(grad_value_padding, x1-delta_x, y1-delta_y)
        if p < r or p < q:
            NMS_grad[y1-1, x1-1] = 0

# # 简化非极大值抑制
# for y1 in range(1, h+1):
#     for x1 in range(1, w+1):
#         p = grad_value_padding[y1, x1]
#         direct = grad_phase[y1-1, x1-1]
#         if direct <= 22.5 or direct >= 337.5 or (direct >= 157.5 and direct <= 202.5):
#             r = grad_value_padding[y1, x1-1]
#             q = grad_value_padding[y1, x1+1]
#         elif (direct > 22.5 and direct < 67.5) or (direct > 202.5 and direct < 247.5):
#             r = grad_value_padding[y1-1, x1+1]
#             q = grad_value_padding[y1+1, x1-1]
#         elif (direct >= 67.5 and direct <= 112.5) or (direct >= 247.5 and direct <= 292.5):
#             r = grad_value_padding[y1-1, x1]
#             q = grad_value_padding[y1+1, x1]
#         else:
#             r = grad_value_padding[y1-1, x1-1]
#             q = grad_value_padding[y1+1, x1+1]
#         if p < r or p < q:
#             NMS_grad[y1-1, x1-1] = 0





# 边缘连接
TH = 0.3*np.max(NMS_grad)
TL = 0.2*np.max(NMS_grad)
mark = np.zeros((h, w), dtype=np.float32)  # 保留点为1，抑制点为0
undetermined_list = []  # 待定点坐标
for y2 in range(h):
    for x2 in range(w):
        if NMS_grad[y2, x2] <= TL:
            NMS_grad[y2, x2] = 0
            mark[y2, x2] = 0
        elif NMS_grad[y2, x2] >= TH:
            mark[y2, x2] = 1
        else:
            mark[y2, x2] = 0
            undetermined_list.append((y2, x2))

count = 1  # 第i次遍历后的新增保留点的数量
# 当count=0时，表明一次遍历后不再增加新的保留点，停止循环
mark_padding = np.zeros((h+2, w+2), dtype=np.float32)
mark_padding[1:h+1, 1:w+1] = mark[:, :]
delete_list = []
while count != 0:
    count = 0
    delete_list = []
    for i in range(len(undetermined_list)):
        ele_y, ele_x = undetermined_list[i]
        ele_y = ele_y + 1
        ele_x = ele_x + 1
        if mark_padding[ele_y-1, ele_x] == 1 or mark_padding[ele_y+1, ele_x] == 1:
            mark_padding[ele_y, ele_x] = 1; count = count + 1; delete_list.append(i)
        elif mark_padding[ele_y, ele_x-1] == 1 or mark_padding[ele_y, ele_x+1] == 1:
            mark_padding[ele_y, ele_x] = 1; count = count + 1; delete_list.append(i)
        elif mark_padding[ele_y-1, ele_x-1] == 1 or mark_padding[ele_y-1, ele_x+1] == 1:
            mark_padding[ele_y, ele_x] = 1; count = count + 1; delete_list.append(i)
        elif mark_padding[ele_y+1, ele_x-1] == 1 or mark_padding[ele_y+1, ele_x+1] == 1:
            mark_padding[ele_y, ele_x] = 1; count = count + 1; delete_list.append(i)
    for j in delete_list[::-1]:
        del undetermined_list[j]
mark = mark_padding[1:h+1, 1:w+1]

# # # TH = 242
# # # TL = 162
t2 = cv.getTickCount()
print("my method={}s".format((t2-t1)/cv.getTickFrequency()))
canny = cv.Canny(res, TL, TH)
t3 = cv.getTickCount()
print("opencv method={}s".format((t3-t2)/cv.getTickFrequency()))

# mark1 = np.zeros((h, w), dtype=np.float32)  # 保留点为1，抑制点为0
# for y2 in range(h):
#     for x2 in range(w):
#         if NMS_grad[y2, x2] <= TL:
#             mark1[y2, x2] = 0
#         else:
#             mark1[y2, x2] = 1


# plt.imshow(mark1, cmap='gray')
# plt.xticks([]), plt.yticks([])
# plt.title('Scharr simplify_NMS non-EdgeLink')
# plt.show()



# display
plt.figure('res')
plt.subplot(131), plt.imshow(NMS_grad, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title('NMS result')
plt.subplot(132), plt.imshow(mark, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title('my result')
plt.subplot(133), plt.imshow(canny, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title('opencv Canny method result')
plt.show()
