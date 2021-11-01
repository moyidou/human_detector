"""此文件用于在图片中识别行人"""

# 组件库
from detector.visualization import *
from detector.config import *
from detector.nms import *
# 图片处理
from skimage.io import imread
import matplotlib.pyplot as plt
# 基本库
import os
import pickle

# 读取训练完成的SVM分类器及相关组件
print('正在读取SVM分类器及其相关组件...')
with open('clf_linear_svc_simple.p', 'rb') as clf_file:
    clf_pickle = pickle.load(clf_file)
    clf = clf_pickle['clf']
    X_scaler = clf_pickle['X_scaler']
    size = clf_pickle['size']
    orientations = clf_pickle['orientations']
    pixels_per_cell = clf_pickle['pixels_per_cell']
    cells_per_block = clf_pickle['cells_per_block']
    visualize = clf_pickle['visualize']
    transform_sqrt = clf_pickle['transform_sqrt']
    spatial_size = clf_pickle['spatial_size']
    bins = clf_pickle['bins']
print('SVM分类器及其相关组件读取完毕.')
# 设置图片读取路径
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, 'data', 'predict')
# 读取目标图片
print('读取目标图片...')
img = imread(os.path.join(file_path, '3_persons.jpg'))
print('目标图片读取完毕.')
# 计算所有用于检测的窗口，常规情况下只使用一种大小，后续可以修改
print('计算滑动窗口...')
windows = slide_windows(img, window_size=(48, 128), window_overlap=(0.9, 0.9))
# 检测图片中的行人
print('检测可能的边界框...')
possible_bboxes = search_possible_bboxes(img, windows=windows, size=size,
                                         clf=clf, scaler=X_scaler,
                                         orientations=orientations, pixels_per_cell=pixels_per_cell,
                                         cells_per_block=cells_per_block, visualize=visualize,
                                         transform_sqrt=transform_sqrt,
                                         spatial_size=spatial_size,
                                         bins=bins,
                                         hog_feat=hog_feat,
                                         spatial_feat=spatial_feat,
                                         histogram_feat=histogram_feat)
# 将不同大小的边界框放在一起进行非最大抑制
new_bboxes = nms(possible_bboxes, 0.05)
new_bboxes = list(map(lambda bboxes: bboxes[0], new_bboxes))
# 在图片上将所有可能的边界框绘制出来
print('绘制边界框...')
detected_img = draw_boxes(img, new_bboxes)
# 展示图片
print('检测结果.')
plt.imshow(detected_img)
plt.show()
