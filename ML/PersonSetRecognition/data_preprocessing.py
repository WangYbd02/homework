"""
@ 数据读取及预处理
"""


import numpy as np
import matplotlib.pyplot as plt


def load_data(filepath):
    """数据预处理
        返回的数据的格式为：allpoints=[time_1, time_2, ..., time_n]
        time_x表示第10*x的time step的所有点的信息，格式为：time_x=[point_1, point_2, ..., point_m]
        point_x表示第x个点的信息，格式为point_x=(id, x, y)"""
    n_frames = 0
    allpoints = []
    current_frame = -1.0
    with open(filepath, 'r') as f:
        while True:
            str_line = f.readline()
            if str_line == '':
                break
            str_ele = str_line.split('\t')
            frame = float(str_ele[0])
            id = float(str_ele[1])
            x = float(str_ele[2])
            y = float(str_ele[3].split('\n')[0])
            if current_frame != frame:
                current_frame = frame
                n_frames = n_frames + 1
                allpoints.append([])
            allpoints[n_frames-1].append((id, x, y))
        f.close()
    return allpoints


if __name__ == '__main__':
    allpoints = load_data('./students003.txt')
    # 这里计算了所有时间步的所有点的坐标中，横纵坐标的最大、最小值，为结果可视化时画出边框固定窗口提供了依据。，
    max_x = -1.0
    min_x = 100.0
    max_y = -1.0
    min_y = 100.0
    for i in range(len(allpoints)):
        for _, x, y in allpoints[i]:
            if x > max_x:  max_x = x
            if x < min_x:  min_x = x
            if y > max_y:  max_y = y
            if y < min_y:  min_y = y
    print(max_x)
    print(min_x)
    print(max_y)
    print(min_y)
