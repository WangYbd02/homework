"""
@ 语音信号预处理
"""


import wave
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


def read_data(data_path):
    """读取语音信号"""
    f = wave.open(data_path)
    params = f.getparams()
    # 声道数，量化位数，采样率，采样点数
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = f.readframes(nframes)  # 读取音频，字符串格式
    f.close()
    wave_data = np.fromstring(str_data, dtype=np.short)  # 将字符串转换为浮点型数据
    wave_data = wave_data/max(abs(wave_data))  # wave幅值归一化
    return wave_data, nframes, framerate

def preEmphasis(wave_data, a):
    """语音信号预加重"""
    res = np.append(wave_data[0], wave_data[1:]-a*wave_data[:-1])
    return res

def win_func(win_len, n, method=0):
    """窗函数"""
    if method == 0:  # 矩形窗
        return 1
    elif method == 1:  # 汉明窗
        return 0.54-0.46*np.cos(2*np.pi*n/(win_len-1))
    elif method == 2:  # 海宁窗
        return 0.5*(1-np.cos(2*np.pi*n/(win_len-1)))

def enframed(data, win_len, step_len, win_method=0):
    """分帧和加窗"""
    data_len = len(data)
    num_frames = int(np.floor((data_len-win_len)/step_len)+1)  # 总帧数
    framed = np.zeros((num_frames, win_len))
    for i in range(num_frames):
        for j in range(win_len):
            framed[i, j] = data[i*step_len+j] * win_func(win_len, j, win_method)
    return framed


if __name__ == '__main__':
    data_path = './voice_data/OSR_us_000_0010_8k.wav'
    win_len = 256
    step_len = 128
    wave_data, nframes, framerate = read_data(data_path)
    wave_data = wave_data[:3*framerate]
    nframes = 3*framerate
    time_list = np.array(range(0, nframes)) * (1.0 / framerate)
    wave_data_after = preEmphasis(wave_data, 0.97)
    plt.plot(time_list, wave_data_after, 'b')
    plt.grid()
    plt.show()
    framed = enframed(wave_data_after, win_len, step_len, 1)
    mag = np.zeros(framed.shape, dtype=complex)  # 频谱
    energy = np.zeros(framed.shape)  # 谱线能量
    for i in range(framed.shape[0]):
        mag[i, :] = np.fft.fft(framed[i, :], win_len)
        energy[i, :] = np.square(abs(mag[i, :]))
    print(energy.shape)
    # plt.plot(mag[0, :])
    # plt.plot(np.arange(win_len), energy[0, :])
    plt.plot(np.arange(256), framed[78])
    plt.grid()
    plt.show()
