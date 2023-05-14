"""
@ 提取MFCC参数
"""


import wave
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import read_data, preEmphasis, enframed
from scipy.fftpack import dct
from VAD import point_check


def extractMFCC(frames, framerate, NFFT, n_filters=24, num_ceps=12):
    """
    提取MFCC系数
    :param frames: 2-dim ndarray. 经过预加重、分帧、加窗处理后的信号数据，大小为(帧数，帧长)
    :param framerate: int. 信号的采样频率
    :param NFFT: int. 作fft的点数，一般取值为帧长且为2的幂次
    :param n_filters: int. mel滤波器的数目
    :param num_ceps: int. 选取的MFCC系数的阶数
    :return: 2-dim ndarrat. 每一帧的MFCC系数，大小为(帧数，num_ceps)
    """
    num_frames = frames.shape[0]
    mag = np.zeros(frames.shape, dtype=complex)
    power = np.zeros(frames.shape)
    for i in range(num_frames):
        mag[i, :] = np.fft.fft(frames[i, :], NFFT)
        power[i, :] = np.square(abs(mag[i, :]))/(NFFT/2)**2
        power[i, 0] = power[i, 0]/4
    # plt.plot(np.arange(256), power[3, :], color='r')
    # plt.show()
    low_freq_mel = 0  # mel最低频率
    high_freq_mel = 2595*np.log10(1+(framerate/2)/700)  # mel最高频率
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filters+2)  # 将mel频率等距分点
    hz_points = 700*(10**(mel_points/2595)-1)  # 将等距分好的mel频率转换为实际频率
    w2 = int(NFFT/2+1)  # 有效点数
    df = framerate/(NFFT+1)  # fs/N
    s = np.zeros((num_frames, n_filters))  # mel滤波器能量
    color = ['b', 'g', 'r', 'm', 'orange', 'k']
    freq = []  # 频率
    for n in range(0, w2):
        freqs = int(n*df)
        freq.append(freqs)
    bank = np.zeros((n_filters, w2))  # mel滤波器的频率响应
    for m in range(1, n_filters+1):
        m0 = int(np.floor(hz_points[m]/df))  # 第m个mel滤波器的中心频率点
        m1 = int(np.floor(hz_points[m-1]/df))  # 第m-1个mel滤波器的中心频率点
        m2 = int(np.floor(hz_points[m+1]/df)+1)  # 第m+1个mel滤波器的中心频率点
        for k in range(m1, m0):
            bank[m-1, k] = (k-m1)/(m0-m1)
        for k in range(m0, m2):
            bank[m-1, k] = (m2-k)/(m2-m0)
    #     plt.plot(freq, bank[m-1, :], color=color[(m-1)%6])  # 画出mel滤波器组
    #     # plt.plot((0, 4000), (0, 0), 'k')
    #     # plt.plot((0, 4000), (1, 1), 'k')
    #     # plt.xlabel('Frequency')
    #     # plt.ylabel('Amplitude')
    #     # plt.show()
    s = np.dot(power[:, :w2], bank.T)  # mel滤波器能量
    # plt.imshow(np.flip(s.transpose(1, 0), axis=0), cmap='hot')
    # plt.xlabel('Frames')
    # plt.ylabel('Frequency/kHz')
    # plt.yticks(np.arange(0, 45/40*26, 5/40*26), np.arange(40, -1, -5)/10)
    # plt.show()
    logs = np.log(s)  # mel滤波器能量取e为底对数
    mfcc0 = dct(logs, type=2, axis=1, norm='ortho')[:, 1:(num_ceps+1)]  # 离散余弦变换后得MFCC并取低频的num_ceps项
    # plt.imshow(np.flip(mfcc0.transpose(1, 0), axis=0), cmap='hsv')
    # plt.xlabel('Frames')
    # plt.ylabel('MFCC Coefficients')
    # plt.yticks(np.arange(0, 12, 11), np.arange(13, 1, -11))
    # plt.show()
    mfcc1_start = mfcc0[:2, :].copy()
    mfcc1_end = mfcc0[-2:, :].copy()
    mfcc1_mid = (2*mfcc0[4:, :]+mfcc0[3:-1, :]-2*mfcc0[:-4, :]-mfcc0[1:-3, :])/10
    mfcc1 = np.concatenate((mfcc1_start, mfcc1_mid, mfcc1_end), axis=0)  # 一阶差分
    mfcc2_start = mfcc1[:2, :].copy()
    mfcc2_end = mfcc1[-2:, :].copy()
    mfcc2_mid = (2*mfcc1[4:, :]+mfcc1[3:-1, :]-2*mfcc1[:-4, :]-mfcc1[1:-3, :])/10
    mfcc2 = np.concatenate((mfcc2_start, mfcc2_mid, mfcc2_end), axis=0)  # 二阶差分
    mfcc = np.concatenate((mfcc0, mfcc1, mfcc2), axis=1)
    # plt.imshow(np.flip(mfcc.transpose(1, 0), axis=0), cmap='hsv')
    # plt.xlabel('Frames')
    # plt.ylabel('MFCC Coefficients')
    # plt.yticks(np.arange(0, 36, 12), np.arange(36, 0, -12))
    # plt.show()
    return mfcc


if __name__ == '__main__':
    data_path = 'D:/dataset/voice dataset/0/0-0-0.wav'
    win_len = 256
    step_len = 128
    n_filters = 26  # mel滤波器的个数
    wave_data, nframes, framerate = read_data(data_path)
    # wave_data = wave_data[:3*framerate]
    # nframes = 3*framerate
    # time_list = np.array(range(0, nframes)) * (1.0/framerate)
    emphasized_signal = preEmphasis(wave_data, 0.97)
    # plt.plot(time_list, emphasized_signal)
    # plt.show()
    frames = enframed(emphasized_signal, win_len, step_len, win_method=1)  # (186, 256)
    StartPoint, EndPoint = point_check(frames)  # 端点检测
    framed_voice = frames[StartPoint:EndPoint, :]  # 分帧后的语音段
    NFFT = win_len
    mfcc = extractMFCC(framed_voice, framerate, NFFT, n_filters)
    print(mfcc)
