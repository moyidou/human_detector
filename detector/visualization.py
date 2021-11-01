"""此文件用于可视化svm分类与识别的结果"""

# 组件库
from detector.extract_feature import *
from detector.preprocess import *
# 图像处理
import cv2 as cv
# 基本库
import copy as copy


def slide_windows(img, x_start=None, x_stop=None, y_start=None, y_stop=None,
                  window_size=(32, 64), window_overlap=(0.5, 0.5)):
    """
    在整个图片上进行滑动窗口的计算，并保存到列表中，由于后续的搜索
    :param img: 单张图片
    :param x_start: x轴开始滑动的区域
    :param x_stop: x轴结束滑动的区域
    :param y_start: y轴开始滑动的区域
    :param y_stop: y轴结束滑动的区域
    :param window_size: 滑动窗口的大小
    :param window_overlap: 滑动窗口之间重叠的比例，用于计算步长，但是为什么用这个我也想不明白
    :return:
    """
    # 由于横纵坐标和图片的尺寸其实是相反的，需要转换，下面转化为画图的常规思维
    x_max = img.shape[1]
    y_max = img.shape[0]
    # 初始化滑动开始与结束的坐标
    if x_start is None:
        x_start = 0
    if x_stop is None:
        x_stop = x_max
    if y_start is None:
        y_start = 0
    if y_stop is None:
        y_stop = y_max
    # 计算滑动步长
    x_stride = int(window_size[0] * (1 - window_overlap[0]))
    y_stride = int(window_size[1] * (1 - window_overlap[1]))
    # 计算两个方向上的窗口数量
    x_n_windows = int(((x_stop - x_start) - window_size[0]) / x_stride) + 1
    y_n_windows = int(((y_stop - y_start) - window_size[1]) / y_stride) + 1
    # 计算所有滑动窗口的坐标
    windows = []
    for i in range(x_n_windows):
        for j in range(y_n_windows):
            start_x = i * x_stride + x_start
            end_x = start_x + window_size[0]
            start_y = j * y_stride + y_start
            end_y = start_y + window_size[1]
            windows.append(((start_x, start_y), (end_x, end_y)))
    return windows


def search_possible_bboxes(img, windows, size,
                           clf, scaler,
                           orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False,
                           transform_sqrt=True,
                           spatial_size=(32, 32),
                           bins=32,
                           hog_feat=True, hog_channel='ALL', spatial_feat=True, histogram_feat=True):
    """
    使用滑动窗口法，搜索图片中所有可能被分类为行人的边界框，需要注意的是：
    1.边界框之间可以存在重叠，需要进行筛选，此处先不做清理
    2.此函数默认不对图片进心颜色格式的修改，使用RGB格式进行处理
    3.此函数针对图片数据
    :param img: 单张图片
    :param windows: 所有可能的滑动窗口
    :param size: 特征提取函数使用的统一图片尺寸
    :param clf: 训练好的分类器
    :param scaler: 用于上方分类器的训练好的重塑器
    :param orientations: HOG的区间数量
    :param pixels_per_cell: 每个cell的像素边长
    :param cells_per_block: 每个block的cell边长
    :param visualize: 是否保存hog图
    :param transform_sqrt: 是否进行gamma标准化
    :param spatial_size: 空间特征的压缩大小
    :param bins: 颜色直方图的区间数量
    :param hog_feat: 是否提取HOG特征
    :param hog_channel: HOG通道
    :param spatial_feat: 是否提取空间特征
    :param histogram_feat: 是否提取颜色直方图
    :return:
    """
    on_windows = []
    # 循环对每一个窗口进行尺寸统一——特征提取——特征标准化——分类预测
    for window in windows:
        # 由于窗口的数据为x、y，而数组的读取为m、n，因此需要注意数组切片的使用
        cut = img[window[0][1]:window[1][1], window[0][0]:window[1][0]]
        # 统一特征提取的像素尺寸, 原函数是对多个图片进行处理，所以加上[0]，后续需要修改
        uniform_cut = uniform_imgs_size([cut], size)[0]
        # 提取预测特征向量
        features = extract_img_features(uniform_cut,
                                        orientations=orientations, pixels_per_cell=pixels_per_cell,
                                        cells_per_block=cells_per_block, visualize=visualize,
                                        transform_sqrt=transform_sqrt,
                                        size=spatial_size,
                                        bins=bins,
                                        hog_feat=hog_feat,
                                        spatial_feat=spatial_feat,
                                        histogram_feat=histogram_feat)
        features = np.concatenate(features)
        # 标准化特征向量
        features_handled = scaler.transform(features.reshape((1, -1)))
        # 预测窗口是否是边界框
        predict_target = clf.predict(features_handled)
        # 计算边界框的置信度
        predict_c = clf.decision_function(features_handled)
        # 如果窗口被预测包含行人，则将窗口信息保留
        if predict_target == 1 and predict_c > 1:
            on_windows.append((window, predict_c))
    return on_windows


def draw_boxes(img, bboxes, color=(255, 0, 0), thick=2):
    """
    将边界框在图片上画出来，请注意，这个函数作用与已经被选择出来的边界框
    :param img: 单张图片
    :param bboxes: 边界框列表
    :param color: 边界框颜色
    :param thick: 边界框厚度
    :return:
    """
    # 复制一份图片，不修改原图
    img_copy = copy.copy(img)
    # 迭代绘制所有边界框
    for bbox in bboxes:
        # 在副本图片上绘制边界框
        cv.rectangle(img_copy, bbox[0], bbox[1], color, thick)
    return img_copy
