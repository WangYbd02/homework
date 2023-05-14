"""
@ 制作视频
"""

import os
import numpy as np
import cv2.cv2 as cv


if __name__ == '__main__':
    # DBSCAN.py文件中第115行可以保存每一帧的聚类信息，请注意保存路径
    pic_path = './frames/'
    pic_nums = [str(k) for k in range(541)]
    fps = 10  # 帧率
    pic_size = (640, 480)
    videoWriter = cv.VideoWriter('./PersonSet.avi', cv.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, pic_size)
    for pic_num in pic_nums:
        pic_name = pic_path + 'frame' + pic_num + '.png'
        img = cv.imread(pic_name)
        videoWriter.write(img)

    videoWriter.release()
