"""此文件用于训练模型，且所有图片直接转化尺寸，不进行填充"""
# 组件库
from detector.extract_feature import *
from detector.preprocess import *
from detector.config import *
# 第三方库
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
# 基本库
import os
import numpy as np
import time
import pickle

# 读取数据---------------------------------------------------------------------------------------------------------------
print('开始读取训练数据与测试数据...')
t1_1 = time.time()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, r'data\Person')
train_pos_data, train_neg_data, test_pos_data, test_neg_data = train_test_pos_neg(file_path)
t1_2 = time.time()
print('数据读取完毕，耗时' + str(t1_2 - t1_1) + 's')
# 预处理数据--------------------------------------------------------------------------------------------------------------
train_pos_data = uniform_imgs_size_simple(train_pos_data, size)
train_neg_data = uniform_imgs_size_simple(train_neg_data, size)
train_pos_target = np.ones(len(train_pos_data))
train_neg_target = np.zeros(len(train_neg_data))
test_pos_data = uniform_imgs_size_simple(test_pos_data, size)
test_neg_data = uniform_imgs_size_simple(test_neg_data, size)
test_pos_target = np.ones(len(test_pos_data))
test_neg_target = np.zeros(len(test_neg_data))
pos_data = train_pos_data + test_pos_data
neg_data = train_neg_data + test_neg_data
pos_targets = np.concatenate((train_pos_target, test_pos_target))
neg_targets = np.concatenate((train_neg_target, test_neg_target))
# 提取训练图片特征---------------------------------------------------------------------------------------------------------
# 由于特征是直接用于模型训练的数据，所以在extract_imgs_features的最后进行了哪np.array的处理转化为数组
print('开始提取训练图片特征...')
pos_features = extract_imgs_features(pos_data,
                                     orientations=orientations, pixels_per_cell=pixels_per_cell,
                                     cells_per_block=cells_per_block, visualize=visualize,
                                     transform_sqrt=transform_sqrt,
                                     size=tuple(spatial_size),
                                     bins=bins,
                                     hog_feat=hog_feat,
                                     spatial_feat=spatial_feat,
                                     histogram_feat=histogram_feat)
neg_features = extract_imgs_features(neg_data,
                                     orientations=orientations, pixels_per_cell=pixels_per_cell,
                                     cells_per_block=cells_per_block, visualize=visualize,
                                     transform_sqrt=transform_sqrt,
                                     size=tuple(spatial_size),
                                     bins=bins,
                                     hog_feat=hog_feat,
                                     spatial_feat=spatial_feat,
                                     histogram_feat=histogram_feat)
# 设置训练数据------------------------------------------------------------------------------------------------------------
# 数据归一化
features = np.vstack((pos_features, neg_features))
scaler = StandardScaler()
features = scaler.fit_transform(features)
targets = np.concatenate((pos_targets, neg_targets))
# 设置线性SVM为训练模型----------------------------------------------------------------------------------------------------
# 由于LinearSVC使用hinge函数作为损失函数，且使用线性核，计算较快。在本项目中，此模型表现已经很好
clf = LinearSVC()
# 正式训练模型
print('开始训练模型，训练规模为' + str(features.shape) + '...')
clf.fit(features, targets)
# 保存模型信息------------------------------------------------------------------------------------------------------------
clf_pickle = {"complete": True,
              "clf": clf,
              "X_scaler": scaler,
              "orientations": orientations,
              "pixels_per_cell": pixels_per_cell,
              "cells_per_block": cells_per_block,
              "visualize": visualize,
              "transform_sqrt": transform_sqrt,
              "spatial_size": spatial_size,
              "bins": bins, "size": size}
clf_destination = clf_path
pickle.dump(clf_pickle, open(clf_destination, 'wb'))
print('训练完成的LinearSVC分类器保存至{}'.format(os.path.join(BASE_DIR, os.path.join('detector', clf_path))))
