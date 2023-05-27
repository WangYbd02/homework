"""
@ 测试数据读取方法的测试
"""

import os
from skimage.io import imread
import cv2.cv2 as cv
import numpy as np


"""
@ test list 的结构：
test list 是一个列表，列表中的每一项代表一个训练样本ele，test list = [ele1, ele2, ele3, ...]
每一个训练样本ele都是一个列表，ele = [img, boxes]，其中img就是图像，数据类型为ndarray
ele中的boxes是annotation中给定的矩形框的列表，boxes = [box1, box2, ...]，具体有几个box就看对应annotation中给了几个框
boxes中的每一个box都是一个四元tuple，即(x1, y1, x2, y2)，(x1, y1)是矩形框左上角的坐标，(x2, y2)是矩形框右下角的坐标
"""

test_image_path = "D:/dataset/INRIAPerson dataset/original_images/test/pos"
annotation_path = "D:/dataset/INRIAPerson dataset/original_images/test/annotations"

images = os.listdir(test_image_path)
test_list = []
for image in images:
    # print(image)
    img = imread(os.path.join(test_image_path, image))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    name = image.split('.')[0]  # 获取图片名(去掉后缀名.png)
    boxes = []  # 边框信息列表
    with open(os.path.join(annotation_path, name+'.txt')) as f:  # 打开标注文件
        iter_f = iter(f)  # 创建迭代器
        for line in iter_f:  # 逐行遍历文件，读取文本
            str_XY = "(Xmax, Ymax)"
            if str_XY in line:
                str_list = line.split(str_XY)
                data_in_txt = "".join(str_list[1:])
                data_in_txt = data_in_txt.replace(':', " ")
                data_in_txt = data_in_txt.replace(',', " ")
                data_in_txt = data_in_txt.replace('-', " ")
                data_in_txt = data_in_txt.replace('(', " ")
                data_in_txt = data_in_txt.replace(')', " ")
                data = data_in_txt.split()
                anno_rect = (int(data[0]), int(data[1]), int(data[2]), int(data[3]))
                boxes.append(anno_rect)
            else: continue
    test_list.append((img, boxes))
    # break
