from skimage.io import imread
from skimage.transform import resize
import cv2 as cv
from sklearn.svm import LinearSVC, SVC
import numpy as np
import matplotlib.pyplot as plt
from detector.config import *
import os
from itertools import product
from detector.extract_feature import extract_hog

# 从config读取hog模型参数
params_hog = {'orientations': orientations,
              'pixels_per_cell': pixels_per_cell,
              'cells_per_block': cells_per_block,
              'visualize': visualize,
              'transform_sqrt': transform_sqrt}

# 提前构建后续使用的变量，清除Pycharm报错
train_pos_num = 1
train_neg_num = 1
test_pos_num = 1
test_neg_num = 1
train_pos_data = None
train_neg_data = None
test_pos_data = None
test_neg_data = None
train_pos_hog = None
train_neg_hog = None
test_pos_hog = None
test_neg_hog = None

# 读取所有图片数据，并保存
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for mode, category in product(('train', 'test'), ('pos', 'neg')):
    # 此变量为图片父目录，例：train_pos_path
    locals()[mode + '_' + category + '_path'] = BASE_DIR + r'\data\INRIAPerson\INRIAPerson\\'[:-1] + mode \
                                                 + r'_64x128_H96\\'[:-1] + category
    # 此变量为图片名称，例：train_pos_paths
    locals()[mode + '_' + category + '_paths'] = os.listdir(locals()[mode + '_' + category + '_path'])
    # 此变量为图片数量，例：train_pos_num
    locals()[mode + '_' + category + '_num'] = len(locals()[mode + '_' + category + '_paths'])
    print(category + '类的' + mode + '数据集有' + str(locals()[mode + '_' + category + '_num']) + '个样本')
    # 读取图片的灰度图，转换为128*64大小，并保存在numpy数组中
    # 此变量为图片数组，例：train_pos_data
    locals()[mode + '_' + category + '_data'] = np.empty((locals()[mode + '_' + category + '_num'], 128, 64))
    for i, path in enumerate(locals()[mode + '_' + category + '_paths']):
        img = imread(locals()[mode + '_' + category + '_path'] + r'\\'[:-1] + path, as_gray=True)
        if img.shape[0] > img.shape[1]:
            img = resize(img, (128, 64))
        else:
            img = resize(img, (int(img.shape[0]*64/img.shape[1]), 64))
            top = int((128 - img.shape[0])/2)
            bottom = 128 - top - img.shape[0]
            img = cv.copyMakeBorder(img, top, bottom, 0, 0, cv.BORDER_CONSTANT, value=(255, 255, 255))
        locals()[mode + '_' + category + '_data'][i] = img

# 保存数据
data = {
    'train_pos_data': train_pos_data,
    'train_neg_data': train_neg_data,
    'test_pos_data': test_pos_data,
    'test_neg_data': test_neg_data,
    'train_pos_target': np.ones(train_pos_num),
    'train_neg_target': np.zeros(train_neg_num),
    'test_pos_target': np.ones(test_pos_num),
    'test_neg_target': np.zeros(test_neg_num)
}

# 提取图片的hog特征，保存到数组中
for mode, category in product(('train', 'test'), ('pos', 'neg')):
    example_hog, _ = extract_hog(locals()[mode + '_' + category + '_data'][0], params_hog['orientations'],
                                 params_hog['pixels_per_cell'], params_hog['cells_per_block'],
                                 params_hog['visualize'], params_hog['transform_sqrt'])
    locals()[mode + '_' + category + '_hog'] = np.empty((locals()[mode + '_' + category + '_data'].shape[0],
                                                         len(example_hog)))
    for i in range(locals()[mode + '_' + category + '_data'].shape[0]):
        hog_feature = extract_hog(locals()[mode + '_' + category + '_data'][i], params_hog)
        locals()[mode + '_' + category + '_hog'][i] = hog_feature
data['train_pos_hog'] = train_pos_hog
data['train_neg_hog'] = train_neg_hog
data['test_pos_hog'] = test_pos_hog
data['test_neg_hog'] = test_neg_hog

# 训练SVM模型
# 1.设置训练数据
train_hog = np.vstack((data['train_pos_data'], data['train_neg_data'])).reshape((-1, 128*64))
train_target = np.vstack((data['train_pos_target'], data['train_neg_target']))

# 2.1径向基支持向量机
svc_1 = SVC(C=0.01, kernel='rbf', class_weight={0:1, 1:10})
svc_1.fit(train_hog, train_target)

# #2.2线性核支持向量机（损失函数为Hinge）
svc_2 = LinearSVC(C=1, class_weight={0:1, 1:10})
svc_2.fit(train_hog, train_target)

# 3.测试的结果显示，线性核的结果已经足够好了
test_hog = np.vstack((data['test_pos_hog'], data['test_neg_hog'])).reshape((test_pos_num + test_neg_num, -1))
pred_target = svc_2.predict(test_hog)
test_target = np.hstack((data['test_pos_target'], data['test_neg_target']))
print(np.mean(test_target == pred_target))
print(pred_target)

# 下一步行动
# 1.目标检测