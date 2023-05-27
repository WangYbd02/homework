"""
@ 非极大值抑制实现，根据iou判断TP，FN，FP的实现
"""

import numpy as np


def computeIOU(rect1, rect2):
    """
    @ 计算两个矩形的交并比
    :param rect1: 矩形1(x1, y1, x2, y2) x1 < x2, y1 < y2
    :param rect2: 矩形2(x1, y1, x2, y2) x1 < x2, y1 < y2
    :return: 交并比
    """
    iou_left = max(rect1[0], rect2[0])
    iou_top = max(rect1[1], rect2[1])
    iou_right = min(rect1[2], rect2[2])
    iou_bottom = min(rect1[3], rect2[3])
    if iou_left > iou_right or iou_top > iou_bottom:  # 不相交
        return 0
    else:
        overlap = (iou_right-iou_left)*(iou_bottom-iou_top)
        area1 = (rect1[2]-rect1[0])*(rect1[3]-rect1[1])
        area2 = (rect2[2]-rect2[0])*(rect2[3]-rect2[1])
        return overlap/(area1+area2-overlap)

def non_max_suppression_func(boxes, overlapThreshold):
    """
    @ 非极大值抑制
    :param boxes: 预测框列表(x1, y1, w, h, predict_score)
    :param overlapThreshold:  抑制或保留的iou阈值
    :return: 经过非极大值抑制的预测框列表
    """
    # 如果列表为空，直接返回空列表
    if len(boxes) == 0:
        return []
    # 获取边框位置
    x1 = boxes[:, 0]; y1 = boxes[:, 1]
    w = boxes[:, 2]; h = boxes[:, 3]
    x2 = x1+w; y2 = y1+h
    scores = boxes[:, 4]
    # 根据score从小到大排序，返回按照下标排序的索引列表
    idxs = list(np.argsort(scores))
    pick_idxs = []  # 最终选取的检测框的索引列表
    while len(idxs) > 0:
        # 获取索引列表中的最后一个索引，并将索引值添加到pick_idxs中
        last = len(idxs) - 1
        i = idxs[last]
        rect1 = (x1[i], y1[i], x2[i], y2[i])
        pick_idxs.append(i)
        supress = []  # 需要被抑制的检测框的索引列表
        # 遍历索引列表中的所有索引
        for ele in range(0, last):
            j = idxs[ele]  # 获取当前索引
            rect2 = (x1[j], y1[j], x2[j], y2[j])
            overlap = computeIOU(rect1, rect2)  # 计算交并比，即重叠度
            if overlap > overlapThreshold:  # 重叠度高于阈值则抑制
                # if scores[i] > scores[j]:
                supress.append(ele)
        del idxs[last]
        supress1 = supress[::-1]
        for k in range(len(supress1)):
            del idxs[supress1[k]]
    pick_boxes = []  # 最终选取的检测框的列表
    for z in pick_idxs:
        pick_boxes.append((x1[z], y1[z], w[z], h[z], scores[z]))
    return pick_boxes

def judge_TP_FN_FP(pred_boxes, anno_boxes, iouThreshold):
    """
    @ 计算一张图片中TP，FN，FP的数量
      TP: ground true为有行人，detection检测到了该行人
      FN: ground true为有行人，detection没有检测到该行人
      FP: ground true为无行人，detection检测为有行人
    :param pred_boxes: 预测的检测框列表[boxes1(x, y, w, h), boxes2, ...]
    :param anno_boxes: 标记的检测框列表[boxes1(x1, y1, x2, y2), boxes2, ...]
    :param iouThreshold: 判断为正例的iou阈值
    :return: TP，FN，FP的数量
    """
    anno_num = len(anno_boxes)  # 标记的窗口数，即标记的人的数量
    anno_boxes = np.array(anno_boxes)
    anno_x1 = anno_boxes[:, 0]; anno_y1 = anno_boxes[:, 1]
    anno_x2 = anno_boxes[:, 2]; anno_y2 = anno_boxes[:, 3]
    anno_mark = np.zeros(anno_num)  # 标记该人是否被识别出，0为未识别出，1为识别出
    pred_num = len(pred_boxes)  # 预测生成的窗口数
    if len(pred_boxes) == 0:
        TP = 0; FN = anno_num; FP = 0; winNum = 0
        return TP, FN, FP, winNum
    pred_boxes = np.array(pred_boxes)
    pred_x1 = pred_boxes[:, 0]; pred_y1 = pred_boxes[:, 1]
    pred_w = pred_boxes[:, 2]; pred_h = pred_boxes[:, 3]
    pred_x2 = pred_x1+pred_w; pred_y2 = pred_y1+pred_h
    pred_mark = np.zeros(pred_num)  # 标记该检测框检测到的是否为人，0不为人，1为人
    for i in range(pred_num):
        pred_rect = (pred_x1[i], pred_y1[i], pred_x2[i], pred_y2[i])
        for j in range(anno_num):
            anno_rect = (anno_x1[j], anno_y1[j], anno_x2[j], anno_y2[j])
            iou = computeIOU(pred_rect, anno_rect)  # 计算iou
            if iou >= iouThreshold:
                anno_mark[j] = 1
                pred_mark[i] = 1
    TP = int(np.sum(anno_mark))
    FN = int(anno_num - np.sum(anno_mark))
    FP = int(pred_num - np.sum(pred_mark))
    winNum = pred_num
    return TP, FN, FP, winNum
