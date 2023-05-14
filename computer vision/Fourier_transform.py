import numpy as np
import cv2.cv2 as cv
from matplotlib import pyplot as plt
from filtering_func import generate_distance_matrix_2d, generate_gaussian_kernel_2d, padding, conv


# img = cv.imread('C:/Users/Admin/Desktop/linghua.jpg', cv.IMREAD_COLOR)
# img2 = cv.cvtColor(img, cv.COLOR_BGR2RGB)
# b, g, r = cv.split(img)
# b_dft = cv.dft(np.float32(b), flags=cv.DFT_COMPLEX_OUTPUT)
# g_dft = cv.dft(np.float32(g), flags=cv.DFT_COMPLEX_OUTPUT)
# r_dft = cv.dft(np.float32(r), flags=cv.DFT_COMPLEX_OUTPUT)
# b_dft_shift = np.fft.fftshift(b_dft)
# g_dft_shift = np.fft.fftshift(g_dft)
# r_dft_shift = np.fft.fftshift(r_dft)
# dst = cv.merge((r_dft_shift, g_dft_shift, b_dft_shift))
# # dft = cv.dft(np.float32(img), flags=cv.DFT_COMPLEX_OUTPUT)
# # dst = np.fft.fftshift(dft)
# magnitude_spectrum = 20*np.log(cv.magnitude(dst[:, :, 0], dst[:, :, 1]))
# phase_spectrum = 20*np.log(cv.phase(dst[:, :, 0], dst[:, :, 1]))
# plt.subplot(221), plt.imshow(img2)
# plt.subplot(222), plt.imshow(magnitude_spectrum)
# plt.subplot(223), plt.imshow(phase_spectrum)
# plt.show()

# set parameters
sigma = 0.5  # standard deviation
size = 3  # size of filtering kernel

# get image
img = cv.imread('C:/Users/Admin/Desktop/linghua.jpg', cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (100, 100))

# generate distance matrix
distance_matrix = generate_distance_matrix_2d(size)
# generate Gaussian filtering kernel
Gaussian_kernel = generate_gaussian_kernel_2d(distance_matrix, size, sigma)
# padding
w, h = img.shape
size_of_padding = int((size-1)/2)
dst = np.zeros((w+size-1, h+size-1), dtype=np.uint8)
dst[size_of_padding:w+size_of_padding, size_of_padding:h+size_of_padding] = img[:, :]
# filtering
res = np.zeros(img.shape, dtype=np.uint8)
for i in range(w):
    for j in range(h):
        res[i][j] = np.sum(dst[i:i+size, j:j+size]*Gaussian_kernel)


# # Fourier transform
# dft = cv.dft(np.float32(res), flags=cv.DFT_COMPLEX_OUTPUT)
# # spectrum centralization
# dft_shift = np.fft.fftshift(dft)
# # get spectrum
# amplitude_spectrum, phase_spectrum = cv.cartToPolar(dft_shift[:, :, 0], dft_shift[:, :, 1], angleInDegrees=True)
# amplitude_spectrum = 20*np.log(amplitude_spectrum)

# # display
# plt.figure('result')
# plt.subplot(131), plt.imshow(img, cmap='gray')
# plt.xticks([]), plt.yticks([])
# plt.title('source image')
# plt.subplot(132), plt.imshow(amplitude_spectrum, cmap='gray')
# plt.xticks([]), plt.yticks([])
# plt.title('amplitude spectrum')
# plt.subplot(133), plt.imshow(phase_spectrum, cmap='gray')
# plt.xticks([]), plt.yticks([])
# plt.title('phase spectrum')
# plt.show()


# Fourier transform Gaussian kernel
dft_gaussian = abs(np.fft.fft2(Gaussian_kernel, img.shape))
# Fourier transform image
dft = np.fft.fft2(img)
# filtering
freq_res = dft*dft_gaussian
# inverse Fourier transform
result = np.fft.ifft2(freq_res)
# get filtering result
result = np.abs(result)

count = 0
for i in range(w):
    for j in range(h):
        if abs(res[i][j] - result[i][j]) > 1:
            print('no!!!')
            count += 1
print(count)

print(res.shape)
print(resul.shape)
plt.figure('res')
plt.subplot(131), plt.imshow(img, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title('source image')
plt.subplot(132), plt.imshow(res, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title('direct filtering')
plt.subplot(133), plt.imshow(result, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.title('frequency domain filtering')
plt.show()


# plt.figure('result')
# plt.subplot(221), plt.imshow(img, cmap='gray')
# plt.xticks([]), plt.yticks([])
# plt.title('source image')
# plt.subplot(222), plt.imshow(amplitude_spectrum, cmap='gray')
# plt.xticks([]), plt.yticks([])
# plt.title('amplitude spectrum')
# plt.subplot(223), plt.imshow(phase_spectrum, cmap='gray')
# plt.xticks([]), plt.yticks([])
# plt.title('phase spectrum')
# plt.show()