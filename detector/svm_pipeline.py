"""此文件用于训练模型，可以认为是主文件"""
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
# 因为这些数据并不是直接用于计算，所以用list存储就可以了
print('开始转化数据格式为64*128*3的格式...')
t2_1 = time.time()
train_pos_data = uniform_imgs_size(train_pos_data, size)
train_neg_data = uniform_imgs_size(train_neg_data, size)
train_pos_target = np.ones(len(train_pos_data))
train_neg_target = np.zeros(len(train_neg_data))
t2_2 = time.time()
print('数据格式转化完毕，耗时' + str(t2_2 - t2_1) + 's')
# 提取训练图片特征---------------------------------------------------------------------------------------------------------
# 由于特征是直接用于模型训练的数据，所以在extract_imgs_features的最后进行了哪np.array的处理转化为数组
print('开始提取训练图片特征...')
t3_1 = time.time()
train_pos_features = extract_imgs_features(train_pos_data,
                                           orientations=orientations, pixels_per_cell=pixels_per_cell,
                                           cells_per_block=cells_per_block, visualize=visualize,
                                           transform_sqrt=transform_sqrt,
                                           size=tuple(spatial_size),
                                           bins=bins,
                                           hog_feat=hog_feat,
                                           spatial_feat=spatial_feat,
                                           histogram_feat=histogram_feat)
train_neg_features = extract_imgs_features(train_neg_data,
                                           orientations=orientations, pixels_per_cell=pixels_per_cell,
                                           cells_per_block=cells_per_block, visualize=visualize,
                                           transform_sqrt=transform_sqrt,
                                           size=tuple(spatial_size),
                                           bins=bins,
                                           hog_feat=hog_feat,
                                           spatial_feat=spatial_feat,
                                           histogram_feat=histogram_feat)
t3_2 = time.time()
print('训练图片特征提取完毕，耗时' + str(t3_2 - t3_1) + 's')
# 设置训练数据------------------------------------------------------------------------------------------------------------
print('开始设置训练数据...')
t4_1 = time.time()
train_X = np.vstack((train_pos_features, train_neg_features))
train_y = np.concatenate((train_pos_target, train_neg_target))
# SVM模型对输入数据的格式要求很高，需要标准化的数据
X_scaler = StandardScaler()
train_X = X_scaler.fit_transform(train_X)
t4_2 = time.time()
print('训练数据设置完毕，耗时' + str(t4_2 - t4_1) + 's')
# 设置线性SVM为训练模型----------------------------------------------------------------------------------------------------
# 由于LinearSVC使用hinge函数作为损失函数，且使用线性核，计算较快。在本项目中，此模型表现已经很好
print('开始训练模型，训练规模为' + str(train_X.shape) + '...')
t5_1 = time.time()
clf = LinearSVC(C=100)
clf.fit(train_X, train_y)
t5_2 = time.time()
print('模型训练完毕，耗时' + str(t5_2 - t5_1) + 's')
# 测试模型性能------------------------------------------------------------------------------------------------------------
# 性能测试只需要一次，后续使用的时候就不会用到这个函数了，但还是需要说明一下测试一个样本的性能
# 提取测试样本特征
print('开始测试模型...')
t6_1 = time.time()
test_pos_data = uniform_imgs_size(test_pos_data, size)
test_neg_data = uniform_imgs_size(test_neg_data, size)
test_pos_target = np.ones(len(test_pos_data))
test_neg_target = np.zeros(len(test_neg_data))
test_pos_features = extract_imgs_features(test_pos_data,
                                          orientations=orientations, pixels_per_cell=pixels_per_cell,
                                          cells_per_block=cells_per_block, visualize=visualize,
                                          transform_sqrt=transform_sqrt,
                                          size=tuple(spatial_size),
                                          bins=bins,
                                          hog_feat=hog_feat,
                                          spatial_feat=spatial_feat,
                                          histogram_feat=histogram_feat)
test_neg_features = extract_imgs_features(test_neg_data,
                                          orientations=orientations, pixels_per_cell=pixels_per_cell,
                                          cells_per_block=cells_per_block, visualize=visualize,
                                          transform_sqrt=transform_sqrt,
                                          size=tuple(spatial_size),
                                          bins=bins,
                                          hog_feat=hog_feat,
                                          spatial_feat=spatial_feat,
                                          histogram_feat=histogram_feat)
test_X = np.vstack((test_pos_features, test_neg_features))
test_y = np.concatenate((test_pos_target, test_neg_target))
predict_y = clf.predict(test_X)
t6_2 = time.time()
print('模型测试完毕，耗时' + str(t6_2 - t6_1) + 's，模型在测试集上的直接准确率为', end='')
print(np.mean(test_y == predict_y))
t7_1 = time.time()
predict_y_example = clf.predict(test_X[0].reshape([1, -1]))
t7_2 = time.time()
print('测试一个图片的时间为' + str(t7_2 - t7_1) + 's.')
# 保存模型信息------------------------------------------------------------------------------------------------------------
clf_pickle = {"complete": True,
              "clf": clf,
              "X_scaler": X_scaler,
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
