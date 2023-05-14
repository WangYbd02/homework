import numpy as np
import cv2.cv2 as cv
import matplotlib.pyplot as plt

# get image
# img = cv.imread('C:/Users/Admin/Desktop/test_pic.jpg', cv.IMREAD_GRAYSCALE)
# h, w = img.shape
# img = cv.resize(img, (int(0.5*w), int(0.5*h)))

def translation(tx, ty):
    matrix = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])
    return matrix
def rotation(theta):  # 度为单位
    sin_t = np.sin(np.radians(theta))
    cos_t = np.cos(np.radians(theta))
    matrix = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])
    return matrix
def euclidean(theta, tx, ty):  # 度为单位
    sin_t = np.sin(np.radians(theta))
    cos_t = np.cos(np.radians(theta))
    matrix = np.array([[cos_t, -sin_t, tx], [sin_t, cos_t, ty], [0, 0, 1]])
    return matrix
def similarity(s, theta, tx, ty):  # 度为单位
    sin_t = np.sin(np.radians(theta))
    cos_t = np.cos(np.radians(theta))
    matrix = np.array([[s*cos_t, -sin_t, tx], [sin_t, s*cos_t, ty], [0, 0, 1]])
    return matrix
def affine(a, b, c, d, tx, ty):
    matrix = np.array([[a, b, tx], [c, d, ty], [0, 0, 1]])
    return matrix

def nearest_neighbor_interpolation(src, x, y):
    x_new = np.round(x).astype(int)
    y_new = np.round(y).astype(int)
    return src[y_new, x_new]
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

def forward_transformation(img, matrix):
    a, b, tx = matrix[0]
    c, d, ty = matrix[1]
    h, w = img.shape
    w_new = np.ceil(abs(a*w) + abs(b*h) + abs(tx)).astype(int)
    h_new = np.ceil(abs(c*w) + abs(d*h) + abs(ty)).astype(int)
    res = np.zeros((h_new, w_new), dtype=np.float32)
    w_transf = 0;  h_transf = 0
    tx_transf = 0;  ty_transf = 0
    w_tmp = np.floor(min(a*w, b*h, a*w+b*h))
    if w_tmp < 0:  w_transf = w_tmp
    h_tmp = np.floor(min(c*w, d*h, c*w+d*h))
    if h_tmp < 0:  h_transf = h_tmp
    if tx < 0:  tx_transf = tx
    if ty < 0:  ty_transf = ty
    for y in range(h):
        for x in range(w):
            x_new = a*x + b*y + tx - w_transf - tx_transf
            y_new = c*x + d*y + ty - h_transf - ty_transf
            res[int(y_new), int(x_new)] = img[y, x]
    res = res.astype(np.uint8)
    return res

def backward_transformation(img, matrix, interpolation):
    a0, b0, tx0 = matrix[0]
    c0, d0, ty0 = matrix[1]
    matrix_inv = np.linalg.inv(matrix)  # 逆变换矩阵
    a, b, tx = matrix_inv[0]
    c, d, ty = matrix_inv[1]
    h, w = img.shape
    w_new = np.ceil(abs(a0*w)+abs(b0*h)+abs(tx0)).astype(int)
    h_new = np.ceil(abs(c0*w)+abs(d0*h)+abs(ty0)).astype(int)
    res = np.zeros((h_new, w_new), dtype=np.float32)
    x_min = np.floor(min(0, a0*w, b0*h, a0*w+b0*h))
    y_min = np.floor(min(0, c0*w, d0*h, c0*w+d0*h))
    tx_transf = 0;  ty_transf = 0
    if tx0 < 0:  tx_transf = int(tx0)
    if ty0 < 0:  ty_transf = int(ty0)
    for y in range(ty_transf, h_new):
        for x in range(tx_transf, w_new):
            x_src = a*(x+x_min) + b*(y+y_min) + tx
            y_src = c*(x+x_min) + d*(y+y_min) + ty
            if (0 <= x_src and x_src <= w-1) and (0 <= y_src and y_src <= h-1):
                if interpolation == 0:
                    res[y-ty_transf, x-tx_transf] = \
                        nearest_neighbor_interpolation(img, x_src, y_src)
                else:
                    res[y-ty_transf, x-tx_transf] = \
                        bilinear_interpolation(img, x_src, y_src)
    res = res.astype(np.uint8)
    return res


# display
H_src = np.array([0.1013, 1.4139, 140.3365, 1.3958, 0.1580, 61.2217, 0.0008, 0.0004, 1.0000], dtype=np.float64)
H = H_src.reshape(3,3)
H_inv = np.linalg.inv(H)
img = cv.imread('D:/Projects/PyCharm/CV_experiment/experiment3/chessboard_image/Image14.jpg', cv.IMREAD_GRAYSCALE)
res = backward_transformation(img, H_inv, 0)
plt.imshow(res, cmap='gray')
plt.show()

# translation_matrix = translation(100, -200)
# rotation_matrix = rotation(45)
# euclidean_matrix = euclidean(69, -100, 200)
# similarity_matrix = similarity(0.8, -158, -189, -146)
# affine_matrix = affine(1, -3, -2, 1, -167, 140)
# res1 = forward_transformation(img, rotation_matrix)
# res2 = backward_transformation(img, rotation_matrix, 0)
# res3 = backward_transformation(img, rotation_matrix, 1)
#
# plt.figure('res')
# plt.subplot(131), plt.imshow(res1, cmap='gray')
# plt.xticks([]), plt.yticks([])
# plt.title('forward')
# plt.subplot(132), plt.imshow(res2, cmap='gray')
# plt.xticks([]), plt.yticks([])
# plt.title('backward-nearest')
# plt.subplot(133), plt.imshow(res3, cmap='gray')
# plt.xticks([]), plt.yticks([])
# plt.title('backward-bilinear')
# plt.show()
