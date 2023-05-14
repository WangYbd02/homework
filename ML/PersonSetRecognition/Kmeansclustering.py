"""
@ K-means聚类算法
"""


import numpy as np
import random
import matplotlib.pyplot as plt
from data_preprocessing import load_data


class KmeansClustering():
    def __init__(self, n_clusters, points, old_centroids=[]):
        self.n_clusters = n_clusters  # 聚类中心数量
        self.points = points  # 待聚类的点信息列表
        self.num_points = len(points)  # 点的数量
        self.old_centroids = old_centroids  # 历史聚类中心坐标列表
        if n_clusters > self.num_points: assert "n_clusters bigger than num_points"

    def computeCluster(self, if_centroids=0):
        """计算出聚类中心和对应的簇，并可视化"""
        centroids = self.initCentroids()
        pre_centroids = []
        while pre_centroids != centroids:
            pre_centroids = centroids
            clusterSet = self.computeMinDistance(centroids)
            centroids = self.updateCentroids(clusterSet)
        self.showCluster(centroids, clusterSet)
        if if_centroids == 0:
            return None
        else: return centroids

    def initCentroids(self):
        """初始化k个聚类中心"""
        random.seed(1)  # 设置随机数种子
        initpoints = random.sample(self.points, self.n_clusters)  # 随机选取n_clusters个点初始化聚类中心
        initcentroids = []
        for i in range(len(initpoints)):
            id, x, y = initpoints[i]
            initcentroids.append((x, y))
        return initcentroids

    def computeMinDistance(self, centroids):
        """对每个点计算距离最近的聚类中心"""
        clusterSet = [[] for k in range(self.n_clusters)]  # 每个列表对应一个簇，每个列表在最外层列表中的索引对应聚类中心的索引
        for ele in self.points:
            min_dis = float("inf")  # 初始化最小距离
            id, x, y = ele
            for i in range(len(centroids)):
                c_x, c_y = centroids[i]
                dis = np.sqrt((x-c_x)**2+(y-c_y)**2)  # 计算L2距离
                if dis < min_dis:
                    min_dis = dis
                    flag = i
            clusterSet[flag].append(ele)  # 将点加入距离最近的聚类中心对应的簇
        return clusterSet

    def updateCentroids(self, clusterSet):
        """根据簇更新聚类中心"""
        centroids = []
        for cluster in clusterSet:  # 如果簇不为空，计算簇中所有点坐标的均值来更新聚类中心
            if cluster != []:
                x_mean = 0
                y_mean = 0
                ele_num = len(cluster)
                for _, x, y in cluster:
                    x_mean += x
                    y_mean += y
                x_mean = x_mean / ele_num
                y_mean = y_mean / ele_num
                centroids.append((x_mean, y_mean))
            else:  # 如果簇为空，则随机选取一个点作为聚类中心更新的替代
                a = random.sample(self.initCentroids(), 1)[0]
                centroids.append(a)

        return centroids

    def showCluster(self, centroids, clusterSet):
        """可视化"""
        colorMark = ['or', 'ob', 'og', 'ok', 'oy', 'oc']
        centroidMark = ['dr', 'db', 'dg', 'dk', 'dy', 'dc']
        """因为相邻帧之间的信息是比较连续的，为了使得相邻帧之间的簇的颜色分布变化较小，
           考虑了通过计算相邻帧之间的每个聚类中心的聚类来决定簇的颜色分配，
           即当前帧的某个簇的颜色 期望选择与上一帧中 二者的聚类中心的L2距离最小的簇 相同的颜色
           即若当前帧的某个簇的聚类中心坐标为(1, 1)，上一帧中，蓝色的簇的聚类中心坐标为(2, 2)，红色的簇的聚类中心坐标为(3, 4)，
           则该簇更期望选择红色。
           但做来的效果也并不好，因为假如当前帧还有一个簇的聚类中心为(1, 2)，则匹配方式会为：(1, 2)-(2, 2)，(1, 1)-(3, 4)
           即优先匹配距离最小的。"""
        if self.old_centroids != []:
            distance_matrix = np.zeros((self.n_clusters, self.n_clusters))
            for m in range(self.n_clusters):
                for n in range(self.n_clusters):
                    x_curr, y_curr = centroids[m]
                    x_old, y_old = self.old_centroids[n]
                    distance_matrix[m, n] = (x_curr-x_old)**2 + (y_curr-y_old)**2
            m_indxs = np.argsort(distance_matrix.reshape(-1))  # 从小到大排序
            match = 0
            indxs = [-1 for _ in range(self.n_clusters)]
            new_get = []
            old_get = []
            for j in range(len(m_indxs)):
                if match == self.n_clusters:
                    break
                m = m_indxs[j] // self.n_clusters
                n = m_indxs[j] % self.n_clusters
                if m not in new_get and n not in old_get:
                    indxs[n] = m
                    match += 1
                    new_get.append(m)
                    old_get.append(n)
        else:
            indxs = [t for t in range(self.n_clusters)]
        for i in range(self.n_clusters):
            indx = indxs[i]
            plt.plot(centroids[indx][0], centroids[indx][1], centroidMark[i])
            for _, x, y in clusterSet[indx]:
                plt.plot(x, y, colorMark[i])
        plt.draw()
        plt.pause(0.1)
        plt.cla()
        return


if __name__ == '__main__':
    allpoints = load_data('./students003.txt')
    centroids = []
    for i in range(len(allpoints)):
        clusterObject = KmeansClustering(6, allpoints[i], old_centroids=centroids)
        centroids = clusterObject.computeCluster(if_centroids=1)
