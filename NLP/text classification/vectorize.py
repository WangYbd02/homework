"""
@ 词向量转化
"""


import numpy as np


class TfIdfVectorizer():
    def __init__(self, norm='l2'):
        self.is_fit = 0  # 是否已经进行fit
        self.norm = norm  # 正则化方式
        self.words_dict = None  # fit操作得到的词典， vector_words_list[i]:vector_times_list[i]
        self.vector_words_list = None  # fit操作得到的词表
        self.vector_times_list = None  # 进行fit操作时，对于词表中每个词，包含该词的文本的数量，用于计算逆向文件频率
        self.vector_len = 0  # fit操作得到的词向量的长度
        self.idf_vector = None  # 逆向文件频率向量

    def fit_transform(self, text_list):
        """fit做出词表，计算逆向文件频率，并将用于fit的文本转化为词向量"""
        self.is_fit = 1
        self.words_dict = {}  # 初始化词典
        for text in text_list:
            words_list = []
            words = text.split(' ')[:-1]  # 因为经过预处理后，text的最后一个字符是空格，split操作得到的列表的最后一项是空字符串，需要删去
            # words_dict的每一个键是一个词表中的单词，对应的值即为该词的文件频率
            for word in words:
                if word not in words_list:
                    words_list.append(word)
                    if word not in self.words_dict.keys():
                        self.words_dict[word] = 1
                    else:
                        self.words_dict[word] += 1
        self.vector_words_list = list(self.words_dict.keys())  # 所有单词
        self.vector_times_list = list(self.words_dict.values())  # 每个单词出现在不同文档中的次数，即文件频率
        self.vector_len = len(self.vector_words_list)  # 词向量长度
        tf_vectors = []  # 用于fit的所有文本的词频统计
        for text in text_list:
            words = text.split(' ')[:-1]  # 同25行
            tf_vector = [0]*self.vector_len
            for word in words:  # 统计每个文本的词频
                loca = self.vector_words_list.index(word)
                tf_vector[loca] += 1
            tf_vectors.append(tf_vector)
        tf_vectors = np.array(tf_vectors)
        num_texts = len(text_list)
        # 计算逆向文件频率，计算方式为 ln((1+文本总数)/(1+文件频率))+1，即做加1平滑并取对数
        self.idf_vector = np.array([np.log((1+num_texts)/(1+self.vector_times_list[i]))+1
                                                for i in range(self.vector_len)])
        tfidf_vectors = tf_vectors[:]*self.idf_vector  # if-idf词向量
        if self.norm == 'l2':  # L2正则化
            l2_norm = np.linalg.norm(tfidf_vectors, axis=1)
            normed_tfidf_vectors = tfidf_vectors/l2_norm.reshape(-1, 1)
            return normed_tfidf_vectors
        else:  # 不做正则化
            return tfidf_vectors

    def transform(self, text_list):
        """将文本转化为词向量"""
        if self.is_fit == 0:
            raise "can't use transform function before fit_transform."
        tf_vectors = []  # 所有文本的词频统计
        for text in text_list:
            words = text.split(' ')[:-1]
            tf_vector = [0]*self.vector_len
            for word in words:  # 统计每个文本的词频
                if word in self.vector_words_list:
                    loca = self.vector_words_list.index(word)  # 如果当前的单词word在词表中，找到word在词表中的索引
                    tf_vector[loca] += 1
            tf_vectors.append(tf_vector)
        tf_vectors = np.array(tf_vectors)
        tfidf_vectors = tf_vectors[:]*self.idf_vector  # tf-idf词向量
        if self.norm == 'l2':  # L2正则化
            l2_norm = np.linalg.norm(tfidf_vectors, axis=1)
            normed_tfidf_vectors = tfidf_vectors/l2_norm.reshape(-1, 1)
            return normed_tfidf_vectors
        else:  # 不做正则化
            return tfidf_vectors

    def get_words_names(self):
        """获取词表"""
        if self.is_fit == 0:
            raise "can't use get_words_name function before fit_transform"
        return self.vector_words_list
