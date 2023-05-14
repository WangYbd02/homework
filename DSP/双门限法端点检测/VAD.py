"""
@ 双门限法端点检测实现和测试
"""


import wave
import numpy as np
import matplotlib.pyplot as plt


def read(data_path):
    """读取语音信号"""
    wave_path = data_path
    f = wave.open(wave_path, 'rb')
    params = f.getparams()
    # 声道数，量化位数，采样频率，采样点数
    nchannels, sampwidth, framerate, nframes = params[:4]
    str_data = f.readframes(nframes)  # 读取音频，字符串格式
    f.close()
    wave_data = np.fromstring(str_data, dtype=np.short)  # 将字符串转化为浮点型数据
    wave_data = wave_data * 1.0 / (max(abs(wave_data)))  # wave幅值归一化
    return wave_data, nframes, framerate

def plot(data, time):
    """作图"""
    plt.plot(time, data)
    plt.grid('on')
    plt.show()

def rect_win(win_len, n):
    """矩形窗"""
    return 1

def hamming_win(win_len, n):
    """汉明窗"""
    return 0.54-0.46*np.cos(2*np.pi*n/(win_len-1))

def hanning_win(win_len, n):
    """海宁窗"""
    return 0.5*(1-np.cos(2*np.pi*n/(win_len-1)))

def win_func(win_len, n, method):
    """窗函数"""
    if method == 0:
        return rect_win(win_len, n)
    elif method == 1:
        return hamming_win(win_len, n)
    else:
        return hanning_win(win_len, n)

def sign(x):
    """符号函数"""
    if x >= 0.0: return 1
    else: return -1

def enframe(data, win_len, step_len, win_method=0):
    """
    @ 对语音数据进行分帧和加窗处理
    :param data: ndarray(1-dimension).语音信号
    :param win_len: int.滑动窗口长度
    :param step_len: int.窗口每次移动的长度
    :param win_method: int.窗函数选择
    :return: framed: ndarray(2-dimension).分帧和加窗结果
    """
    data_len = len(data)
    num_slide = int(np.floor((data_len-win_len)/step_len)+1)  # 窗口移动次数+1，即窗口总数
    framed = np.zeros((num_slide, win_len))  # 初始化二维数组(分帧加窗结果)
    for i in range(num_slide):
        for j in range(win_len):
            framed[i, j] = data[i*step_len+j] * win_func(win_len, j, win_method)
    return framed

def point_check(framed):
    """
    @ 语音信号端点检测
    :param framed: ndarray(2-dimension).分帧和加窗结果
    :return: StartPoint: int.语音段起始点(起始帧数)
             EndPoint: int.语音段终止点(终止帧数)
    """
    """--------------------------------------------------------------"""
    """1.计算短时过零率"""
    zeroCrossingRate = []  # 记录每帧的过零率
    for i in range(framed.shape[0]):
        sum_zcr = 0
        for j in range(framed.shape[1]-1):
            sum_zcr += (sign(framed[i][j]) != sign(framed[i][j+1]))
        zeroCrossingRate.append(sum_zcr)
    """--------------------------------------------------------------"""
    """2.计算短时能量"""
    energy = []  # 记录每帧的能量
    for i in range(framed.shape[0]):
        sum_energy = 0
        for j in range(framed.shape[1]):
            sum_energy += framed[i][j]*framed[i][j]
        energy.append(sum_energy)
    """--------------------------------------------------------------"""
    """3.设置门限"""
    zcrLow = 3  # 过零率低门限
    zcrHigh = np.max([np.round(max(zeroCrossingRate)*0.1), 6])  # 过零率高门限
    energyLow = max(energy)*0.01  # 能量低门限
    energyHigh = max(energy)*0.1  # 能量高门限
    """--------------------------------------------------------------"""
    """4.端点检测"""
    Status = 0  # 状态  0:静音段，1:语音段，2:结束段
    maxSilenceTime = 4  # 语音段内容许静默的最长时间(帧数为单位)
    minAudioTime = 12  # 语音段持续的最短时间(帧数为单位)
    holdTime = 0  # 语音段持续时间(帧数为单位)
    silenceTime = 0  # 语音段内静默的时间(帧数为单位)
    StartPoint = 0  # 语音段起始点(起始帧数)
    for n in range(framed.shape[0]):
        if Status == 0:
            if energy[n] > energyHigh:  # 高于能量高门限则认为是语音段的起点
                StartPoint = n
                Status = 1
                holdTime = holdTime + 1
                silenceTime = 0
            else:
                Status = 0
                holdTime = 0
        elif Status == 1:
            if energy[n] > energyLow and zeroCrossingRate[n] > zcrLow:  # 高于低门限持续认为是语音段
                holdTime = holdTime + silenceTime + 1
                silenceTime = 0
            else:
                silenceTime = silenceTime + 1
                if silenceTime <= maxSilenceTime:  # 静默时间小于语音段内容许的静默时间则继续
                    continue
                elif holdTime < minAudioTime:  # 语音段不能过短
                    Status = 0
                    holdTime = 0
                    silenceTime = 0
                else:  # 静默时间大于语音段内容许的静默时间则判断为语音段终止点(语音段持续时间满足要求)
                    Status = 2
        elif Status == 2:
            break
    EndPoint = StartPoint + holdTime  # 语音段终止点(终止帧数)

    return StartPoint, EndPoint


if __name__ == '__main__':
    """
    例子：3-0-0：噪声非常大
         4-0-0：奇怪的声音
    """
    data_path = 'D:/dataset/voice dataset/0/3-0-0.wav'
    win_len = 240
    step_len = 80
    wave_data, nframes, framerate = read(data_path)
    time_list = np.array(range(0, nframes)) * (1.0 / framerate)
    framed = enframe(wave_data, win_len, step_len, win_method=1)
    StartPoint, EndPoint = point_check(framed)
    print(StartPoint, EndPoint)
    start = time_list[StartPoint*step_len-1]
    end = time_list[(EndPoint-1)*step_len+win_len-1]
    plt.plot(time_list, wave_data)
    plt.vlines([start, end], -1, 1, linestyles="dashed", colors="red")
    plt.grid('on')
    plt.show()
