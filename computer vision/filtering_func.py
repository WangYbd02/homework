import numpy as np
import math as m
import cv2.cv2 as cv


# Gaussian function
# one-dimension Gaussian function
def func_gaussian_1d(x, sigma):
    return m.exp(-x**2/(2*sigma**2)) / (sigma*(2*m.pi)**(1/2))


# two-dimension Gaussian function
def func_gaussian_2d(x, y, sigma):
    return m.exp(-(x**2 + y**2)/(2*sigma**2)) / (2*m.pi*sigma**2)


# generate 1-dimension distance vector
def generate_distance_vector(size):
    """
    @ distance vector
    :param size: int. kernel size
    :return: list(int). distance vector
    """
    distance_vector = []
    for i in range(size):
        distance_vector.append((i-(size-1)/2))
    return distance_vector


# generate 2-dimension distance matrix
def generate_distance_matrix_2d(size):
    """
    @ 2-dimension distance matrix
    :param size: int. kernel size
    :return: list(list(tuple)). distance matrix
    """
    distance_matrix = []
    for i in range(size):
        distance_matrix.append([])
        for j in range(size):
            distance_matrix[i].append(((j - (size-1)/2), (((size-1)/2) - i)))
    return distance_matrix


# generate 1-dimension Gaussian filtering kernel
def generate_gaussian_kernel_1d(distance_vector, size, sigma):
    """
    @ generate a 1-dimensa=ion filtering kernel based on distance vector
    :param distance_vector: list. a distance vector
    :param size: int. kernel size
    :param sigma: float. standard deviation of Gaussian function
    :return: ndarray. a 1-dimension Gaussian filtering kernel
    """
    gaussian_kernel = np.zeros(size)
    for i in range(size):
        gaussian_kernel[i] = func_gaussian_1d(distance_vector[i], sigma)
    total_value = np.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel / total_value
    return gaussian_kernel


# generate 2-dimension Gaussian filtering kernel
def generate_gaussian_kernel_2d(distance_matrix, size, sigma):
    """
    @ generate a 2-dimension Gaussian filtering kernel based on distance matrix
    :param distance_matrix: list. a distance matrix
    :param size: int. kernel size
    :param sigma: float. standard deviation of Gaussian function
    :return: ndarray. a 2-dimension Gaussian filtering kernel
    """
    gaussian_kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            tmp = distance_matrix[i][j]
            gaussian_kernel[i][j] = func_gaussian_2d(tmp[0], tmp[1], sigma)
    total_value = np.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel / total_value
    return gaussian_kernel


# padding method
# clip filter
def clip_filter_padding(src, size):
    src_size = src.shape
    width = src_size[0]
    height = src_size[1]
    depth = src_size[2]
    padding_size = int((size-1)/2)
    dst = np.zeros((width+size-1, height+size-1, depth), dtype=np.uint8)
    dst[padding_size:width+padding_size,
                    padding_size:height+padding_size, :] = src[:, :, :]
    return dst


# wrap around
def wrap_around_padding(src, size):
    src_size = src.shape
    width = src_size[0]
    height = src_size[1]
    depth = src_size[2]
    padding_size = int((size-1)/2)
    dst = np.zeros((width+size-1, height+size-1, depth), dtype=np.uint8)
    dst[padding_size:width+padding_size, padding_size:height+padding_size, :] = src[:, :, :]  # center
    dst[:padding_size, :padding_size, :] = src[width-padding_size:, height-padding_size:, :]  # top left
    dst[:padding_size, padding_size:height+padding_size, :] = src[width-padding_size:, :, :]  # top
    dst[:padding_size, height+padding_size:, :] = src[width-padding_size:, :padding_size, :]  # top right
    dst[padding_size:width+padding_size, :padding_size, :] = src[:, height-padding_size:, :]  # left
    dst[padding_size:width+padding_size, height+padding_size:, :] = src[:, :padding_size, :]  # right
    dst[width+padding_size:, :padding_size, :] = src[:padding_size, height-padding_size:, :]  # bottom left
    dst[width+padding_size:, padding_size:height+padding_size, :] = src[:padding_size, :, :]  # bottom
    dst[width+padding_size:, height+padding_size:, :] = src[:padding_size, :padding_size, :]  # bottom right
    return dst


# copy edge
def copy_edge_padding(src, size):
    src_size = src.shape
    width = src_size[0]
    height = src_size[1]
    depth = src_size[2]
    padding_size = int((size-1)/2)
    dst = np.zeros((width+size-1, height+size-1, depth), dtype=np.uint8)
    dst[padding_size:width+padding_size, padding_size:height+padding_size, :] = src[:, :, :]  # center
    dst[:padding_size, :padding_size, :] = src[0][0]  # top left
    dst[:padding_size, padding_size:height+padding_size, :] = src[0]  # top
    dst[:padding_size, height+padding_size:, :] = src[0][-1]  # top right
    for w1 in range(width):
        dst[padding_size+w1, :padding_size, :] = src[w1][0]  # left
    for w2 in range(width):
        dst[padding_size+w2, height+padding_size:, :] = src[w2][-1]  # right
    dst[width+padding_size:, :padding_size, :] = src[-1][0]  # bottom left
    dst[width+padding_size:, padding_size:height+padding_size, :] = src[-1]  # bottom
    dst[width+padding_size:, height+padding_size:, :] = src[-1][-1]  # bottom right
    return dst


# reflect across edge
def reflect_across_edge_padding(src, size):
    src_size = src.shape
    width = src_size[0]
    height = src_size[1]
    depth = src_size[2]
    padding_size = int((size-1)/2)
    dst = np.zeros((width+size-1, height+size-1, depth), dtype=np.uint8)
    dst[padding_size:width+padding_size, padding_size:height+padding_size, :] = src[:, :, :]  # center
    dst[:padding_size, :padding_size, :] = src[padding_size-1::-1, padding_size-1::-1, :]  # top left
    dst[:padding_size, padding_size:height+padding_size, :] = src[padding_size-1::-1, :, :]  # top
    dst[:padding_size, height+padding_size:, :] = src[padding_size-1::-1, height-1:height-padding_size-1:-1, :]  # top right
    dst[padding_size:width+padding_size, :padding_size, :] = src[:, padding_size-1::-1, :]  # left
    dst[padding_size:width+padding_size, height+padding_size:, :] = src[:, height-1:height-padding_size-1:-1, :]  # right
    dst[width+padding_size:, :padding_size, :] = src[width-1:width-padding_size-1:-1, padding_size-1::-1, :]  # bottom left
    dst[width+padding_size:, padding_size:height+padding_size, :] = src[width-1:width-padding_size-1:-1, :, :]  # bottom
    dst[width+padding_size:, height+padding_size:, :] = src[width-1:width-padding_size-1:-1,
                                                            height-1:height-padding_size-1:-1, :]  # bottom right
    return dst


# padding method packaging
def padding(src, size, method):
    """
    @ same padding
    :param src: ndarray. input image
    :param size: int. kernel size
    :param method: str or int. four padding methods:clip_filter(1), wrap around(2),
                   copy edge(3), reflect_across_edge(4)
    :return: ndarray. array after padding
    """
    if method == 'clip_filter' or method == 1:
        return clip_filter_padding(src, size)
    elif method == 'wrap_around' or method == 2:
        return wrap_around_padding(src, size)
    elif method == 'copy_edge' or method == 3:
        return copy_edge_padding(src, size)
    elif method == 'reflect_across_edge' or method == 4:
        return reflect_across_edge_padding(src, size)
    else:
        return clip_filter_padding(src, size)


# convolution function
def conv(matrix, kernel, kernel_size):
    """
    @ convolution
    :param matrix: ndarray. array
    :param kernel: list. kernel
    :param kernel_size: int. kernel size
    :return:  int. convolution result
    """
    conv_res = [0, 0, 0]
    for m1 in range(kernel_size):
        for m2 in range(kernel_size):
            conv_res[0] = conv_res[0] + matrix[m1][m2][0]*kernel[m1][m2]
            conv_res[1] = conv_res[1] + matrix[m1][m2][1]*kernel[m1][m2]
            conv_res[2] = conv_res[2] + matrix[m1][m2][2]*kernel[m1][m2]
    return conv_res

