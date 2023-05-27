"""
@ 滑窗法实现
"""

import numpy as np
import cv2.cv2 as cv


def slidingWindows(img, winSize, winStride):
    h, w, _ = img.shape
    rects = []
    if w < winSize[0] or h < winSize[1]:
        return rects
    wStrideNum = int((w-winSize[0])/winStride[1])+1
    hStrideNum = int((h-winSize[1])/winStride[1])+1
    for h0 in range(hStrideNum):
        for w0 in range(wStrideNum):
            x = w0*winStride[0]
            y = h0*winStride[1]
            rects.append((x, y, winSize[0], winSize[1]))
    return rects

def downSample(src, scale):
    w = np.round(src.shape[1]/scale).astype(int)
    h = np.round(src.shape[0]/scale).astype(int)
    dst = cv.resize(src, (w, h), interpolation=cv.INTER_CUBIC)
    return dst
