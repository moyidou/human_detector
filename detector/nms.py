"""在基于HOG的分类器中，很多不是行人的边界框被识别，且置信度甚至超过真正有人的区域，所以，此处选择使用非最大抑制，尝试对此进行改善"""

# 基本库
import numpy as np


def cal_overlap_area(detection_1, detection_2):
    """
    计算两个边界框之间的重叠比例，用于排除重叠区域很高的一些边界框
    由于保证接口的统一性以及置信度的比较，此处的边界框还包含了置信度的信息，在bbox[1]中，有关边框的信息在bbox[0]
    :param detection_1: 第一个边界框
    :param detection_2: 第二个边界框
    :return:
    """
    # 计算x与y方向上的重叠长度
    bbox_1 = detection_1[0]
    bbox_2 = detection_2[0]
    x_start_1 = bbox_1[0][0]
    x_stop_1 = bbox_1[1][0]
    y_start_1 = bbox_1[0][1]
    y_stop_1 = bbox_1[1][1]
    x_start_2 = bbox_2[0][0]
    x_stop_2 = bbox_2[1][0]
    y_start_2 = bbox_2[0][1]
    y_stop_2 = bbox_2[1][1]
    x_overlap = max(0, min(x_stop_1, x_stop_2) - max(x_start_1, x_start_2))
    y_overlap = max(0, min(y_stop_1, y_stop_2) - max(y_start_1, y_start_2))
    # 计算重叠区域的面积
    overlap_area = x_overlap * y_overlap
    # 计算并集区域的面积
    total_area = (x_stop_1 - x_start_1) * (y_stop_1 - y_start_1) + \
                 (x_stop_2 - x_start_2) * (y_stop_2 - y_start_2) - overlap_area
    return overlap_area / total_area


def nms(detections, threshold=.5):
    """
    排除置信度较低的边界框
    :param detections: 所有可能的边界框
    :param threshold: 阈值
    :return:
    """
    # 根据置信度对边界框进行排序
    detections = sorted(detections, key=lambda detection: detection[1], reverse=True)
    # 先添加置信度最高的边界框，保证循环的进行
    new_detections = [detections[0]]
    del detections[0]
    # 对边界框进行排除
    for index, detection in enumerate(detections):
        for new_detection in new_detections:
            if cal_overlap_area(detection, new_detection) > threshold:
                break
        else:
            new_detections.append(detection)
    return new_detections

