"""
@ 根据实验结果作miss rate vs FPPW曲线，计算AUC
"""

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


# iouThreshold = 0.3

# svm data
# scoreThreshold = [0.999, 0.995, 0.99, 0.95, 0.9, 0.5]
# TP = [101, 122, 133, 157, 166, 184]
# FN = [103, 82, 71, 47, 38, 20]
# FP = [235, 367, 570, 1320, 1830, 3664]
# winNum = [348, 508, 736, 1550, 2085, 4000]

scoreThreshold = [1, 0.999, 0.995, 0.99, 0.95, 0.9, 0.5, 0]
TP = [0, 101, 122, 133, 157, 166, 184, 1]
FN = [1, 103, 82, 71, 47, 38, 20, 0]
FP = [0, 235, 367, 570, 1320, 1830, 3664, 1]
winNum = [1, 348, 508, 736, 1550, 2085, 4000, 1]


# 计算miss rate和FPPW
miss_rate = []
fppw = []
for i in range(len(scoreThreshold)):
    miss_rate.append(FN[i]/(TP[i]+FN[i]))
    fppw.append(FP[i]/winNum[i])

miss_rate = np.array(miss_rate)
fppw = np.array(fppw)

# 三次样条插值
fppw_new = np.linspace(fppw.min(), fppw.max(), 300)
func = interp1d(fppw, miss_rate, kind='cubic')
miss_rate_new = func(fppw_new)

# 计算AUC
AUC = 0.0
for j in range(len(scoreThreshold)-1):
    delta_x = fppw[j+1]- fppw[j]
    AUC += (miss_rate[j]+miss_rate[j+1])*delta_x/2

# 作曲线图
# plt.plot(fppw_new, miss_rate_new)

# 作折线图并计算AUC
plt.plot(fppw, miss_rate)
print(AUC)
plt.title('miss rate vs FPPW')
plt.show()
