"""
@ 读取数据，数据预处理，训练SVM
"""

import numpy as np
import cv2.cv2 as cv
from skimage.io import imread
from sklearn import svm
from sklearn.model_selection import train_test_split
import pickle
import os
import random


# 读取数据
def load_images(dirname):
    img_list = []
    images = os.listdir(dirname)
    count = 0
    for image_name in images:
        img = imread(os.path.join(dirname, image_name))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_list.append(img)
        count += 1
        print(count)
    return img_list

# 将归一化后用于训练的正样本(96x160px)crop出中间的64x128px
def pos_sample_crop(pos_list, crop_pos_list, crop_size=(64,128)):
    w, h = crop_size
    for i in range(len(pos_list)):
        if pos_list[i].shape[1] >= w and pos_list[i].shape[0] >= h:
            crop_left = int((pos_list[i].shape[1]-w)/2)
            crop_top = int((pos_list[i].shape[0]-h)/2)
            crop_pos_list.append(pos_list[i][crop_top:h+crop_top, crop_left:w+crop_left])
    return

# 从每个负样本中随机crop出10张64x128px的图像
def neg_sample_crop(neg_list, crop_neg_list, crop_size=(64, 128)):
    random.seed(1)
    w, h = crop_size
    for i in range(len(neg_list)):
        if neg_list[i].shape[1] >= w and neg_list[i].shape[0] >= h:
            for j in range(2):
                crop_left = int(random.random()*(neg_list[i].shape[1]-w))
                crop_top = int(random.random()*(neg_list[i].shape[0]-h))
                crop_neg_list.append(neg_list[i][crop_top:h+crop_top, crop_left:w+crop_left])
    return

# 计算HOG特征
def computeHOGs(img_list, gradient_list, winSize=(64,128), blockSize=(16,16), blockStride=(8,8), cellSize=(8,8), nbins=9):
    hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    for image in img_list:
        gray_img = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gradient_list.append(hog.compute(gray_img))
    return


# 设置参数
winSize = (64, 128)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9


# 读取数据并预处理
pos_list = load_images('D:/dataset/INRIAPerson dataset/normalized_images/train/pos')
neg_list = load_images('D:/dataset/INRIAPerson dataset/normalized_images/train/neg')
crop_pos_list = []
crop_neg_list = []
labels = []
pos_sample_crop(pos_list, crop_pos_list, (64, 128))
[labels.append(+1) for _ in range(len(crop_pos_list))]
neg_sample_crop(neg_list, crop_neg_list, (64, 128))
[labels.append(-1) for _ in range(len(crop_neg_list))]

# 计算HOG特征
gradient_list = []
computeHOGs(crop_pos_list, gradient_list, winSize, blockSize, blockStride, cellSize, nbins)
computeHOGs(crop_neg_list, gradient_list, winSize, blockSize, blockStride, cellSize, nbins)

# 将HOG特征数据划分为训练集和测试集
gradient_train_list, gradient_test_list, labels_train, labels_test = train_test_split(gradient_list, labels, test_size=0.3)

print("start train")
# 训练SVM
clf = svm.SVC(kernel='rbf', class_weight='balanced', probability=True)
clf.fit(gradient_train_list, labels_train)
score = clf.score(gradient_test_list, labels_test)
print("score:{}".format(score))
# 保存模型
# print("start to save")
# with open('Model/model1.pickle', 'wb') as f:
#     pickle.dump(clf, f)
