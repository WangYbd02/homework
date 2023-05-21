"""
@ K近邻算法
"""


import numpy as np


class KNeighbors():
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors  # k值
        self.is_fit = 0  # 是否已经进行fit
        self.data_list = []  # 训练数据列表
        self.labels_list = []  # 训练数据的标签列表

    def fit(self, data, labels, is_clear: bool = True):
        """训练"""
        if is_clear:  # 清空已经被存储的训练数据
            self.data_list = []
            self.labels_list = []
        data = np.array(data)
        num_data = data.shape[0]
        labels = np.array(labels)
        num_labels = labels.shape[0]
        if num_data != num_labels:
            raise "data and labels don't match"
        for indx in range(num_data):
            self.data_list.append(data[indx])
            self.labels_list.append(labels[indx])
        self.is_fit = 1
        return self

    def sortedList(self, dist_list, labels_list):
        """排序，传入列表长度均为n，dist_list的前n-1项已经按照由小到大排序，
           需要把第n项插入到第i位使dist_list的全部n项按照由小到大排序，并将labels_list的第n项插入到第i位
        """
        new_ele = dist_list[-1]
        new_label = labels_list[-1]
        len_list = len(dist_list)
        for i in range(len_list-1):
            if dist_list[i] > new_ele:
                break
        for j in range(len_list-1, i, -1):
            dist_list[j] = dist_list[j-1]
            labels_list[j] = labels_list[j-1]
        dist_list[i] = new_ele
        labels_list[i] = new_label
        return dist_list, labels_list

    def predict(self, test_data):
        if self.is_fit == 0:
            raise "can't use predict function before fit."
        labels = []  # 返回的预测标签列表
        for data in test_data:  # 对逐个数据进行预测
            data = np.array(data)
            knearest_dist = [np.inf] * self.n_neighbors  # 初始化最近邻的n_neighbors项的列表
            knearest_labels = [-1] * self.n_neighbors  # 初始化对应的标签列表
            for indx, c in enumerate(self.data_list):
                dist = np.linalg.norm(data-c)  # 计算L2距离
                if dist < knearest_dist[-1]:  # 计算到的距离小于列表中的最大值，则进行替换
                    knearest_dist[-1] = dist
                    knearest_labels[-1] = self.labels_list[indx]
                    if self.n_neighbors > 1:  # neighbors数量大于1才需要进行排序
                        knearest_dist, knearest_labels = self.sortedList(knearest_dist, knearest_labels)
            label = max(knearest_labels, key=knearest_labels.count)  # 进行投票，选择投票数最多的标签
            # print(label)  # 输出选择的标签
            labels.append(label)
        return labels
