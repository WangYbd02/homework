"""
@ 基于时间信息和历史信息的行人组识别
"""


import cv2.cv2 as cv
import numpy as np
from DBSCAN import DBSCAN
from data_preprocessing import load_data


def findPersonSet(idList,
                  stepRecordingList,
                  points,
                  clusterSet,
                  mark_clusters,
                  method=0,
                  lowThreshold=0.5,
                  highThreshold=0.7,
                  holdstep=4,
                  tolerationstep=2,
                  separationstep=6):
    """
    基于时间信息和历史聚类信息的行人组识别
    :param idList: list.记录行人的信息，列表中的每一项是一个二元元组(id, state)，id记录行人的id，state记录当前帧是否出现对应id的行人.
    :param stepRecordingList: list.记录行人的信息，列表中的每一项是一个列表，记录对应id的行人从第一次出现到当前帧的聚类信息(即簇标记)，
                                   id通过在idList和stepRecordingList相同的索引来对应.
    :param points: list.出现在当前帧的所有点的信息.
    :param clusterSet: list.当前帧的聚类信息，通过DBSCAN类的generateCluster方法获取.
    :param mark_clusters: int.簇的标记，不同帧中的簇标记信息不同，同一帧不同簇的标记信息不同，同一帧同一簇的标记信息相同.
    :param method: int.利用时间信息的方式选择.
    :param lowThreshold: float.低阈值.
    :param highThreshold: float.高阈值，为了在方式三中修正tolerationstep选取不恰当的影响所设定的阈值.
    :param holdstep: int.保持时间.
    :param tolerationstep: int.容差时间.
    :param separationstep: int.分隔时间.
    :return: new_clusterSet: list.当前帧的行人组信息.结构同clusterSet.
             mark_clusters: int.簇的标记.
    """
    for i in range(len(idList)):
        idList[i][1] = 0  # 将idList中所有id对应的state置0
    # 记录当前时间步的每个行人的情况
    for i in range(len(clusterSet)):
        mark_clusters = (mark_clusters + 1) % 1000
        for id, x, y in clusterSet[i]:
            idArray = np.array(idList)
            # 判断当前id是否在idList中，若不在则加入idList，且将对应的state置为1，若在则将对应的state置为1即可，
            # 同时添加相应的簇标记到stepRecordingList中
            if idList == [] or id not in idArray[:, 0]:
                idList.append([id, 1])
                stepRecordingList.append([])
                stepRecordingList[-1].append(mark_clusters)
            else:
                loca = np.where(idArray[:, 0]==id)[0][0]
                idList[loca][1] = 1
                stepRecordingList[loca].append(mark_clusters)
    """--------------------------------------------------------------------------------------------------------------"""
    # 删除idList和stepRecordingList中在当前时间步消失的行人的信息(id对应的state为0)
    deleteList = []
    for j in range(len(idList)):
        id, state = idList[j]
        if state == 0:
            deleteList.append(j)
    deleteList = deleteList[::-1]
    for i in deleteList:
        del idList[i]
        del stepRecordingList[i]
    """--------------------------------------------------------------------------------------------------------------"""
    # 返回添加了时间记录信息的clusterSet，即最终的行人组信息
    new_clusterSet = []
    """-------------------------------------------方式一，基于同一簇占据的百分比-------------------------------------------"""
    if method == 0:
        walked = []
        for i in range(len(idList)):
            id, _ = idList[i]
            point = findxy(id, points)  # 找到id对应的点在当前帧的信息(即获取坐标)
            cluster = [point]
            if id not in walked:
                for j in range(i+1, len(idList)):
                    # 如果idList[i]项和idList[j]项对应的簇标记相同的数量与idList[i]的簇标记总数的比值大于lowThreshold，
                    # 则被判断为同一行人组
                    # idList[i]一定比idList[j]早出现
                    tmp = [k for k in stepRecordingList[i] if k in stepRecordingList[j]]
                    if len(tmp) >= len(stepRecordingList[i])*lowThreshold:
                        point = findxy(idList[j][0], points)
                        cluster.append(point)
                        walked.append(idList[j][0])
                new_clusterSet.append(cluster)
        """------------------------------------------方式二，简单考虑开始与结束-------------------------------------------"""
    elif method == 1:
        walked = []
        for i in range(len(idList)):
            id, _ = idList[i]
            point = findxy(id, points)
            cluster = [point]
            if id not in walked:
                for j in range(i+1, len(idList)):
                    # tmp1为stepRecordingList[i]与stepRecordingList[j]的
                    # 前min(len(stepRecordingList[i]), holdstep+tolerationstep)中相同的簇标记对应的在前者中的索引
                    tmp1 = [k for k in range(len(stepRecordingList[i]))
                           if stepRecordingList[i][k] in stepRecordingList[j] and k < (holdstep+tolerationstep)]
                    tmp2 = findTailDifferent(stepRecordingList[i], stepRecordingList[j])
                    # 如果两个行人出现的前holdstep+tolerationstep帧中有holdstep帧在一起，
                    # 且分离时间小于separationstep，则会被认为是同一行人组
                    if len(tmp1) >= holdstep and tmp2 < separationstep:
                        point = findxy(idList[j][0], points)
                        cluster.append(point)
                        walked.append(idList[j][0])
                new_clusterSet.append(cluster)
        """---------------------------------------------方式三，前两种的结合---------------------------------------------"""
    else:
        walked = []
        for i in range(len(idList)):
            id, _ = idList[i]
            point = findxy(id, points)
            cluster = [point]
            if id not in walked:
                for j in range(i+1, len(idList)):
                    # idList[i]一定比idList[j]早出现
                    tmp1 = [k for k in stepRecordingList[i] if k in stepRecordingList[j]]
                    tmp2 = [k for k in range(len(stepRecordingList[i]))
                           if stepRecordingList[i][k] in stepRecordingList[j] and k < (holdstep+tolerationstep)]
                    tmp3 = findTailDifferent(stepRecordingList[i], stepRecordingList[j])
                    if len(tmp1) >= len(stepRecordingList[i])*lowThreshold and len(tmp2) >= holdstep and tmp3 < separationstep:
                        point = findxy(idList[j][0], points)
                        cluster.append(point)
                        walked.append(idList[j][0])
                    elif len(tmp1) >= len(stepRecordingList[i])*highThreshold and tmp3 < separationstep:
                        point = findxy(idList[j][0], points)
                        cluster.append(point)
                        walked.append(idList[j][0])
                new_clusterSet.append(cluster)

    return new_clusterSet, mark_clusters

def findxy(id, points):
    """通过id寻找当前帧的对应点的信息"""
    for i in range(len(points)):
        if id == points[i][0]:
            return points[i]

def findTailDifferent(ls1, ls2):
    """计算两个列表尾部有多少项不同"""
    diff_tail_len = 0
    for i in range(min(len(ls1), len(ls2))):
        if ls1[-1-i] != ls2[-1-i]:
            diff_tail_len += 1
        else:
            break
    return diff_tail_len

if __name__ == '__main__':
    allpoints = load_data('./students003.txt')
    idList = []
    stepRecordingList = []
    mark_clusters = 0
    t_sum = 0
    for i in range(len(allpoints)):
        t1 = cv.getTickCount()
        clusterObject = DBSCAN(allpoints[i], 1)
        clusterSet = clusterObject.generateClusterSet()
        new_clusterSet, mark_clusters = findPersonSet(idList, stepRecordingList, allpoints[i], clusterSet, mark_clusters,
                                                     method=2)
        t2 = cv.getTickCount()
        t_sum += (t2-t1)
        clusterObject.showCluster(new_clusterSet, i*10.0)
        # print((t2-t1)/cv.getTickFrequency())
    print("avg_time:", t_sum/cv.getTickFrequency()/len(allpoints))
