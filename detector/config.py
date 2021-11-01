"""
设置配置变量
"""

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import configparser
import json

# 读取配置文件
config = configparser.ConfigParser()
config.read(BASE_DIR + r'\data\config.cfg')
# HOG参数
hog_feat = config.getboolean('hog', 'hog_feat')
orientations = json.loads(config.get('hog', 'orientations'))
pixels_per_cell = json.loads(config.get('hog', 'pixels_per_cell'))
cells_per_block = json.loads(config.get('hog', 'cells_per_block'))
visualize = config.getboolean('hog', 'visualize')
transform_sqrt = config.getboolean('hog', 'transform_sqrt')
channel = config.get('hog', 'channel')
# 空间特征参数
spatial_feat = config.getboolean('spatial', 'spatial_feat')
spatial_size = json.loads(config.get('spatial', 'size'))
# 颜色直方图参数
histogram_feat = config.getboolean('histogram', 'histogram_feat')
bins = json.loads(config.get('histogram', 'bins'))
# 图像尺寸统一参数
size = json.loads(config.get('preprocess', 'size'))
# 分类器保存目录
clf_path = config.get('path', 'clf')
