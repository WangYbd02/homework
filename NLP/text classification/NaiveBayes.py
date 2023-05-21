"""
@ 多项式朴素贝叶斯
"""


import numpy as np


class MultinomialNaiveBayes():
    def __init__(self, alpha=1.0, fit_prior: bool = True, if_ratio: bool = False):
        self.alpha = alpha  # 平滑因子
        self.fit_prior = fit_prior  # 是否需要从测试数据中学习先验概率
        self.if_ratio = if_ratio  # 计算条件概率和后验概率的方式，不会描述，请看具体函数
        self.is_fit = 0  # 是否已经进行fit
        self.labels_set = None  # 标签类别集合
        self.P_y = None  # 先验概率
        self.P_x_y = None  # 条件概率
        self.P_x_y_yindex = None  # 对于标签为y的条件概率存储位置的索引

    def fit(self, data, labels):
        """训练"""
        data = np.array(data)
        num_data = data.shape[0]
        labels = np.array(labels)
        num_labels = labels.shape[0]
        v_len = data[0].shape[0]  # 词向量长度
        if num_data != num_labels:  # 训练样本数与标签数不匹配则抛出异常
            raise "data and labels don't match"
        """
        朴素贝叶斯方法基于贝叶斯定理: p(y|X) = (p(X|y)p(y))/p(X)
        p(X)已知，下面要做的就是表示出p(y)，p(X|y)
        先表示出p(y)，即每个类别的先验概率
        """
        self.labels_set = set()  # 利用集合元素不能重复的特性，提取出每一个类别对应的标签
        for label in labels:
            self.labels_set.add(label)
        self.P_y = {}  # 以字典形式表示每个类别出现的概率，标签:先验概率
        num_classes = len(self.labels_set)  # 类别数量
        if not self.fit_prior:  # 不学习先验概率则使用均匀分布
            for label in self.labels_set:
                self.P_y[label] = 1.0/num_classes
        else:  # 从训练数据中学习先验概率
            for label in self.labels_set:
                self.P_y[label] = np.sum(np.equal(label, labels))/float(num_labels)
        """
        朴素贝叶斯方法中，“朴素”为独立性假设，则有p(X|y)=p(x1|y)p(x2|y)...p(xn|y)
        通过统计的方式得到各类别下各特征属性的条件概率，即p(xi|y)
        """
        # 条件概率p(xi|y)，存储层级为P_x_y = [p(x|y1), p(x|y2), ..., p(x|yn)], p(x|yn) = [p(x1|yn), p(x2|yn), ..., p(xm|yn)]
        self.P_x_y = []
        self.P_x_y_yindex = []  # 对于标签为y的条件概率存储位置的索引
        labels_list = list(labels)
        for y in self.P_y.keys():  # 对于每种标签y，计算p(xi|y)
            y_index = [j for j, label in enumerate(labels_list) if label==y]  # 获取所有标签为y的数据的索引
            x_y_counts = np.zeros(v_len)
            if not self.if_ratio:
                # 计算出标签为y的所有数据(向量)中每一维不为0的个数，即每个单词在标签为y的文本中出现的次数，每个文本最多统计一次
                for indx in y_index:
                    x_y_counts = x_y_counts + np.ones(data[indx].shape) - np.equal(0, data[indx])
            else:
                # 计算出标签为y的所有数据(向量)中每一维的和，即每个单词在标签为y的文本中出现的总次数
                for indx in y_index:
                    x_y_counts = x_y_counts + data[indx]
            num_words = np.sum(x_y_counts)  # 单词总数
            p_x_y = [0]*v_len  # p(xi|y)
            # 加入平滑因子进行平滑，并对p(xi|y)取对数，因为词向量太长，p(xi|y)会很小
            # 经过试验发现，计算后验概率时不取对数连乘后得到的结果都为0.0
            if not self.if_ratio:
                for i in range(v_len):
                    p_x_y[i] = np.log((x_y_counts[i]+self.alpha)/float(num_words+self.alpha*v_len))
            else:
                for i in range(v_len):
                    p_x_y[i] = np.log((x_y_counts[i]+self.alpha/v_len)/float(num_words+self.alpha))
            self.P_x_y.append(p_x_y)
            self.P_x_y_yindex.append(str(y))
        self.is_fit = 1
        return self

    def predict(self, test_data):
        """预测"""
        if self.is_fit == 0:
            raise "can't use predict function before fit."
        test_data = np.array(test_data)
        labels = []  # 返回的预测标签列表
        for i in range(test_data.shape[0]):  # 对逐个数据进行预测
            data = test_data[i]
            P_y_x = {}  # 创建记录后验概率的字典，标签y:p(y|X)
            for label in self.labels_set:  # 对每一个可能的标签y，计算后验概率，取后验概率最大的作为预测结果
                P_y_x[label] = np.log(self.P_y[label])
                loca = self.P_x_y_yindex.index(str(label))  # 找到标签y对于的条件概率存储位置的索引
                for indx in range(data.shape[0]):
                    if data[indx] != 0:
                        # p(X)为已知的定值，无需计算在内
                        if not self.if_ratio:
                            # p(y|X) = p(y)p(x1|y)p(x2|y)...p(xt|y)/p(X)
                            P_y_x[label] = P_y_x[label]+self.P_x_y[loca][indx]
                        else:
                            # p(y|X) = p(y)(p(x1|y))^d1(p(x2|y))^d2...(p(xt|y))^dt/p(X)，di为测试数据(向量)第i项的值
                            P_y_x[label] = P_y_x[label]+self.P_x_y[loca][indx]*data[indx]
            choose_label = max(P_y_x, key=P_y_x.get)  # 选择后验概率最大的标签
            # print(choose_label)  # 输出选择的标签
            labels.append(choose_label)
        return labels
