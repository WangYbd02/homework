"""
@ svm性能测试
"""

import os
import numpy as np
import cv2.cv2 as cv
from skimage.io import imread
from sklearn import svm
import pickle
from sliding_window import slidingWindows, downSample
from NMS import non_max_suppression_func, judge_TP_FN_FP


"""------------------------------------------------------------------------------------------------------------------"""
"""单张图像测试"""

annotation_path = "D:/dataset/INRIAPerson dataset/original_images/test/annotations"
with open('Model/model1.pickle', 'rb') as f:
    clf2 = pickle.load(f)
print('ok')

t1 = cv.getTickCount()
img = imread('D:/dataset/INRIAPerson dataset/original_images/test/pos/crop001521.png')
res = img.copy()
res = cv.cvtColor(res, cv.COLOR_BGR2RGB)
hog = cv.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)

# 设置参数
num_layers = 4
scale = 1.3
scoreThreshold = 0.999
overlapThreshold = 0.3
iouThreshold = 0.3

dst = res.copy()
boxes = []
for i in range(num_layers):
    rects = slidingWindows(dst, (64, 128), (16, 16))
    print(rects)
    for r in rects:
        x, y, w, h = r
        tmp = np.zeros((128, 64, 3), dtype=np.uint8)
        tmp[:, :] = dst[y:y+h, x:x+w]
        tmp1 = cv.cvtColor(tmp, cv.COLOR_BGR2GRAY)
        gradient = hog.compute(tmp1)
        gradient = gradient.reshape(1, -1)
        predict_score = clf2.predict_proba(gradient)[0][-1]
        if predict_score > scoreThreshold:
            x0 = int(x*np.power(scale, i))
            y0 = int(y*np.power(scale, i))
            w0 = int(w*np.power(scale, i))
            h0 = int(h*np.power(scale, i))
            boxes.append((x0, y0, w0, h0, predict_score))
    dst = downSample(dst, scale)


boxes = np.array(boxes)
pick_boxes = non_max_suppression_func(boxes, overlapThreshold)
print(type(pick_boxes))
for r in pick_boxes:
    x, y, w, h, _ = r
    x0 = int(x); y0 = int(y); w0 = int(w); h0 = int(h)
    cv.rectangle(res, (x0, y0), (x0+w0, y0+h0), (0, 255, 0), 2)

t2 = cv.getTickCount()
time = (t2-t1)/cv.getTickFrequency()
print(time)

with open(os.path.join(annotation_path+'/crop001521.txt')) as f:
    iter_f = iter(f)
    boxess = []
    for line in iter_f:
        str_XY = "(Xmax, Ymax)"
        if str_XY in line:
            strlist = line.split(str_XY)
            strlist1 = "".join(strlist[1:])
            strlist1 = strlist1.replace(':', ' ')
            strlist1 = strlist1.replace('-', ' ')
            strlist1 = strlist1.replace(')', ' ')
            strlist1 = strlist1.replace('(', ' ')
            strlist1 = strlist1.replace(',', ' ')
            b = strlist1.split()
            bnd = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
            boxess.append(bnd)
for r in boxess:
    x, y, w, h = r
    x0 = int(x); y0 = int(y); x1 = int(w); y1 = int(h)
    cv.rectangle(res, (x0, y0), (x1, y1), (255, 0, 0), 2)

TP, FN, FP, winNum = judge_TP_FN_FP(pick_boxes, boxess, iouThreshold)
print("TP={},FN={},FP={}，winNUM={}".format(TP, FN, FP, winNum))

cv.namedWindow('res', cv.WINDOW_NORMAL)
cv.imshow('res', res)
cv.waitKey(0)
cv.destroyAllWindows()


"""------------------------------------------------------------------------------------------------------------------"""
"""100张图像测试"""

# test_image_path = "D:/dataset/INRIAPerson dataset/original_images/test/pos"
# annotation_path = "D:/dataset/INRIAPerson dataset/original_images/test/annotations"
#
# # 获取数据，test_list格式[img, boxes(box1(x1, y1, x2, y2), box2, ...)]
# images = os.listdir(test_image_path)[:100]
# test_list = []
# for image in images:
#     imgs = imread(os.path.join(test_image_path, image))
#     imgs = cv.cvtColor(imgs, cv.COLOR_BGR2RGB)
#     name = image.split('.')[0]  # 获取图片名(去掉后缀m名.png)
#     boxes = []  # 边框信息列表
#     with open(os.path.join(annotation_path, name+'.txt')) as f:  # 打开标注文件
#         iter_f = iter(f)  # 创建迭代器
#         for line in iter_f:  # 逐行遍历文件，读取文本
#             str_XY = "(Xmax, Ymax)"
#             if str_XY in line:
#                 str_list = line.split(str_XY)
#                 data_in_txt = "".join(str_list[1:])
#                 data_in_txt = data_in_txt.replace(':', " ")
#                 data_in_txt = data_in_txt.replace(',', " ")
#                 data_in_txt = data_in_txt.replace('-', " ")
#                 data_in_txt = data_in_txt.replace('(', " ")
#                 data_in_txt = data_in_txt.replace(')', " ")
#                 data = data_in_txt.split()
#                 anno_rect = (int(data[0]), int(data[1]), int(data[2]), int(data[3]))
#                 boxes.append(anno_rect)
#             else: continue
#     test_list.append((imgs, boxes))
# print('data preparing finished.')
#
# # 载入训练好的模型
# with open('Model/model1.pickle', 'rb') as f:
#     clf2 = pickle.load(f)
# print('svm model preparing finished')
#
# # 创建hog对象
# hog = cv.HOGDescriptor((64, 128), (16, 16), (8, 8), (8, 8), 9)
# # 设置超参数
# num_layers = 4  # 高斯金字塔的层数
# scale = 1.3  # 高斯金字塔下采样率
# scoreThreshold = 0.5  # svm判断为正样本的概率阈值
# overlapThreshold = 0.3  # 非极大值抑制作抑制时交并比的阈值
# iouThreshold = 0.3  # 判断为正样本的交并比阈值
# TP = 0
# FN = 0
# FP = 0
# winNum = 0
# # 开始识别
# print('test start!')
# for i in range(len(test_list)):
#     current_list = test_list[i]
#     img = current_list[0]
#     anno_boxes = current_list[1]
#     dst = img.copy()
#     predict_boxes = []  # 预测的框
#     for j in range(num_layers):
#         rects = slidingWindows(dst, (64, 128), (8, 8))
#         for r in rects:
#             x, y, w, h = r
#             tmp = np.zeros((h, w, 3), dtype=np.uint8)
#             tmp[:, :] = dst[y:y+h, x:x+w]
#             tmp_gray = cv.cvtColor(tmp, cv.COLOR_BGR2GRAY)
#             gradient = hog.compute(tmp_gray)
#             gradient = gradient.reshape(1, -1)
#             predict_score = clf2.predict_proba(gradient)[0][-1]
#             if predict_score >= scoreThreshold:
#                 x0 = int(x*np.power(scale, j))
#                 y0 = int(y*np.power(scale, j))
#                 w0 = int(w*np.power(scale, j))
#                 h0 = int(h*np.power(scale, j))
#                 predict_boxes.append((x0, y0, w0, h0, predict_score))
#         dst = downSample(dst, scale)
#     predict_boxes = np.array(predict_boxes)
#     pick_boxes = non_max_suppression_func(predict_boxes, overlapThreshold)
#     TP0, FN0, FP0, winNum0 = judge_TP_FN_FP(pick_boxes, anno_boxes, iouThreshold)
#     TP = TP+TP0; FN = FN+FN0; FP = FP+FP0; winNum = winNum+winNum0
#     print("epoch {} finished.TP={} FN={} FP={} winNum={}".format(i+1, TP, FN, FP, winNum))
#
# print('finish testing')
# print("final result:TP={} FN={} FP={} winNum={}".format(TP, FN, FP, winNum))

"""------------------------------------------------------------------------------------------------------------------"""
