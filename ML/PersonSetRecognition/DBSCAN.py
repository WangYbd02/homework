"""
@ DBSCAN算法
"""


import numpy as np
import random
import matplotlib.pyplot as plt
from data_preprocessing import load_data


"""
DBSCAN算法，基于给定点集和epsilon进行密度聚类，由于该任务的每个点都代表一个行人，没有噪点，因而不设置minPts
"""
class DBSCAN():
    def __init__(self, points, e):
        self.points = points  # 初始点集
        self.e = e  # epsilon
        self.n_clusters = 0  # 形成的簇的数量
        self.num_points = len(points)  # 点的数量

    def computeCluster(self, frametime=-1.0):
        """计算聚类集合并可视化聚类结果"""
        clusterSet = self.generateClusterSet()  # 生成聚类中的每一个簇，得到聚类结果
        self.showCluster(clusterSet, frametime)  # 聚类结果可视化
        return

    def generateClusterSet(self):
        """
        基于初始化对象时给定的点，计算出每一个簇，得到簇集合(聚类结果)
        :return: list.簇集合
        """
        sourceSet = self.points[:]  # 拷贝初始点集
        item = sourceSet[0]  # 选择一个核心对象
        del sourceSet[0]
        clusterSet = [] # 初始化簇集合(聚类结果)，每一项是一个簇
        while sourceSet != []:
            # 选取核心对象item后要做del
            clusterSet.append(self.generateCluster(item, sourceSet))  # 计算出一个簇并加入簇集合
            self.n_clusters += 1 # 簇的数量加一
            if sourceSet == []:
                break
            item = sourceSet[0]  # 选择一个核心对象
            del sourceSet[0]
            if sourceSet == []:  # 选择核心对象后若未归簇点集成为空集，则该核心对象为孤立点，单独成为一个簇
                clusterSet.append([item])
        return clusterSet

    def getNeighbor(self, item, sourceSet):
        """
        计算出点集sourceSet中与当前点item密度直达的点，然后从sourceSet中删除这些点
        :param item: tuple.当前点
        :param sourceSet: list.未归簇点集
        :return: list.与点item密度直达的点集
        """
        neighbors = []  # 与点item密度直达的点的集合
        deleteIndexList = []  # 需要从sourceSet中删除的点在sourceSet中的索引
        for i in range(len(sourceSet)):
            ele = sourceSet[i]
            dis = np.sqrt((item[1]-ele[1])**2 + (item[2]-ele[2])**2)  # 计算L2距离
            if dis < self.e:  # 密度直达的点加入neighbors中并需要从sourceSet中删除
                neighbors.append(ele)
                deleteIndexList.append(i)
        deleteIndexList = deleteIndexList[::-1]
        for j in deleteIndexList:
            del sourceSet[j]
        return neighbors

    def generateCluster(self, item, sourceSet):
        """
        生成一个簇，给定一个核心对象点item，基于该核心对象生成一个簇
        :param item: tuple.给定的核心对象
        :param sourceSet: list.未归簇点集
        :return: list.一个簇
        """
        cluster = []
        q = [item]  # 生成队列，通过队列得到簇
        while q != []:
            head = q[0]
            cluster.append(head)
            del q[0]
            q.extend(self.getNeighbor(head, sourceSet))
        return cluster

    def showCluster(self, clusterSet, frametime=-1.0):
        """
        可视化聚类结果，黑色菱形表示孤立点(单个行人)，不同颜色的圆点表示不同的簇(不同的行人组)
        :param clusterSet: list.簇集合
        :param frametime: float.在窗口中显示时间步，默认为-1.0时不显示
        :return: None.
        """
        colorMark = ['k', 'b', 'r', 'g', 'y', 'c', 'm', 'gray', 'coral', 'navy', 'orange', 'brown',
                     'lime', 'tan', 'aquamarine', 'blueviolet', 'indigo', 'pink', 'deeppink', 'teal',
                     'gold', 'darkgoldenrod', 'silver', 'darkslategray', 'mediumslateblue']
        color_num = 1
        for i in range(len(clusterSet)):
            if len(clusterSet[i]) == 1:  color_indx = 0; marker= 'd'  # 对于孤立点，用黑色菱形表示
            else:  color_indx = color_num;  color_num += 1; marker = 'o'  # 对于非孤立点，用不同颜色的圆点表示各个不同的簇
            """作出每个点"""
            for _, x, y in clusterSet[i]:
                if color_indx < 25:
                    plt.plot(x, y, color=colorMark[color_indx], marker=marker)
                else:
                    color_indx = color_indx % 25 + 1
                    plt.plot(x, y, color=colorMark[color_indx], marker=marker)
        """作出边界"""
        plt.plot((-0.18, -0.18), (-0.23, 14), color='r', linestyle='dashed')
        plt.plot((-0.18, 15.5), (-0.23, -0.23), color='r', linestyle='dashed')
        plt.plot((15.5, 15.5), (-0.23, 14), color='r', linestyle='dashed')
        plt.plot((-0.18, 15.5), (14, 14), color='r', linestyle='dashed')
        """显示时间步"""
        if frametime != -1.0:
            plt.title("time step:"+str(frametime))
        plt.draw()
        # plt.savefig('./frames/frame'+str(int(frametime/10)))  # 以png图片形式保存当前帧的聚类结果
        plt.pause(0.1)  # 每一个时间步的结果的展示时间
        plt.cla()
        return


if __name__ == '__main__':
    allpoints = load_data('./students003.txt')
    for i in range(len(allpoints)):
        clusterObject = DBSCAN(allpoints[i], 1.0)
        clusterObject.computeCluster(i*10.0)
