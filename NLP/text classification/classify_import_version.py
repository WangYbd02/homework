"""
@ 调包实现词向量转化和分类器的版本
"""


import os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from preprocessing import preprocess
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import cv2.cv2 as cv


if __name__ == '__main__':
    """--------------------------------------------------------------------------"""
    """1.读取文本数据并预处理"""
    time1 = cv.getTickCount()
    rootdir = 'D:/dataset/mini_newsgroups'  # 数据集根目录
    files = os.listdir(rootdir)
    num_files = len(files)  # 文本类别数
    train_text_list = []  # 训练集数据列表
    test_text_list = []  # 测试集数据列表
    train_label_list = []  # 训练集标签列表
    test_label_list = []  # 测试集标签列表
    train_set_ratio = 0.8  # 训练集分割比例
    for i, filename in enumerate(files):
        texts = os.listdir(rootdir + '/' + filename)
        data_set1 = texts[:int(0.2*len(texts))]
        data_set2 = texts[int(0.2*len(texts)):int(0.4*len(texts))]
        data_set3 = texts[int(0.4*len(texts)):int(0.6*len(texts))]
        data_set4 = texts[int(0.6*len(texts)):int(0.8*len(texts))]
        data_set5 = texts[int(0.8*len(texts)):]
        train_set = data_set1+data_set2+data_set3+data_set4
        test_set = data_set5
        # texts中有九个文本不含'Lines:'，有一个文本除了标注信息以外全为空白，有一个文本除了标注信息以外全部是停用词，经过预处理都变成了空字符串
        for textname in train_set:
            text = preprocess(rootdir + '/' + filename + '/' + textname)
            if text != '':
                train_text_list.append(text)
                train_label_list.append(i)
        for textname in test_set:
            text = preprocess(rootdir + '/' + filename + '/' + textname)
            if text != '':
                test_text_list.append(text)
                test_label_list.append(i)
    # print(len(train_text_list))
    # print(len(test_text_list))
    print('data prepare finished.')
    """--------------------------------------------------------------------------"""
    """2.转化为词向量"""
    tv = TfidfVectorizer(use_idf=True, smooth_idf=True, norm='l2')  # 实例化基于tf-idf的词向量转化对象
    x_train_tf = tv.fit_transform(train_text_list)  # 训练集文本转化为词向量(稀疏矩阵)
    x_train = x_train_tf.todense()  # 稀疏矩阵表示转变为普通矩阵表示
    x_test_tf = tv.transform(test_text_list)  # 测试集文本转化为词向量(稀疏矩阵)
    x_test = x_test_tf.todense()  # 稀疏矩阵表示转变为普通矩阵表示
    """--------------------------------------------------------------------------"""
    """3.训练分类器"""
    # clf = MultinomialNB()  # 多项式朴素贝叶斯
    # clf = KNeighborsClassifier(n_neighbors=11)  # K近邻分类器
    clf = DecisionTreeClassifier()  # 决策树
    clf.fit(x_train, train_label_list)  # 训练分类器
    """--------------------------------------------------------------------------"""
    """4.测试并输出结果"""
    y_predicted = clf.predict(x_test)  # 分类器预测结果
    total = len(y_predicted)  # 测试集数据总数
    correct = 0  # 所有类别的正确数量
    totals = [0]*num_files  # 每个类别的测试数据的数量
    corrects = [0]*num_files  # 每个类别的预测正确的数量
    for i in range(total):
        totals[test_label_list[i]] += 1
        if y_predicted[i] == test_label_list[i]:
            correct += 1
            corrects[test_label_list[i]] += 1
    print("acc:{:.3f}".format(correct/total))  # 所有类别的准确率
    # top-5准确率
    acc = [corrects[i]/totals[i] for i in range(num_files)]
    indx = np.argsort(np.array(acc))[::-1]  # 由大到小排序
    name_dict = {}  # 标签:文本类别名  字典
    for i in range(num_files):
        name_dict[i] = files[i]
    print('top-5 classes:{}'.format([name_dict[i] for i in indx[:5]]))
    print('top-5 accuracy:{}'.format([acc[i] for i in indx[:5]]))
    time2 = cv.getTickCount()
    print('time:{}s'.format((time2-time1)/cv.getTickFrequency()))
