"""此文件作用于已经读取且格式化的图片"""

# 特征提取
from skimage.feature import hog
# 图像处理
from skimage.transform import resize
# 基本库
import numpy as np


def extract_hog(img, orientations, pixels_per_cell, cells_per_block, visualize, transform_sqrt):
    """
    提取图片的HOG特征，以及对应的HOG图
    :param img: 单张图片
    :param orientations: 180°分成多少个区间，决定了HOG特征的维度
    :param pixels_per_cell: 每个cell中的像素个数，(x, y)
    :param cells_per_block: 每个block中的cell个数，(x, y)
    :param visualize: 是否可视化，如果True，保留hog_img
    :param transform_sqrt: 是否进行gamma标准化，如果True，进行标准化
    :return:
    """
    if visualize:
        fd, hog_img = hog(image=img,
                          orientations=orientations,
                          pixels_per_cell=pixels_per_cell,
                          cells_per_block=cells_per_block,
                          visualize=visualize,
                          transform_sqrt=transform_sqrt)
        return fd, hog_img
    else:
        fd = hog(image=img,
                 orientations=orientations,
                 pixels_per_cell=pixels_per_cell,
                 cells_per_block=cells_per_block,
                 visualize=visualize,
                 transform_sqrt=transform_sqrt)
        return fd


def extract_spatial(img, size):
    """
    压缩图片大小，并提取空间信息，这是很纯粹的信息，其实没有经过提取
    :param img: 单张图片
    :param size: 图片的压缩大小
    :return:
    """
    channel_1 = resize(img[:, :, 0], size).ravel()
    channel_2 = resize(img[:, :, 1], size).ravel()
    channel_3 = resize(img[:, :, 1], size).ravel()
    spatial_feature = np.concatenate((channel_1, channel_2, channel_3))
    return spatial_feature


def extract_color_histogram(img, bins):
    """
    提取图片的颜色直方图
    :param img: 单张图片
    :param bins: 直方图划分区间个数
    :return:
    """
    # 计算三个通道的颜色直方图特征
    channel_1 = np.histogram(img[:, :, 0], bins=bins)
    channel_2 = np.histogram(img[:, :, 1], bins=bins)
    channel_3 = np.histogram(img[:, :, 2], bins=bins)
    # 计算总直方图特征
    color_histogram = np.concatenate((channel_1[0], channel_2[0], channel_3[0]))
    return color_histogram


def extract_img_features(img, orientations, pixels_per_cell, cells_per_block, visualize, transform_sqrt, size, bins,
                         hog_feat=True, hog_channel='ALL', spatial_feat=True, histogram_feat=True):
    """
    提取单张图片的HOG、空间特征与颜色直方图
    :param img: 单张图片
    :param orientations: HOG区间个数
    :param pixels_per_cell: HOG每个cell包含像素个数
    :param cells_per_block: HOG每个block包含像素个数
    :param visualize: HOG是否可视化，本函数目前只支持不可视化
    :param transform_sqrt: HOG是否进行gamma转换
    :param size: 空间特征的压缩大小
    :param bins: 颜色直方图区间个数
    :param hog_feat: 是否提取HOG特征
    :param hog_channel: 提取HOG的通道
    :param spatial_feat: 是否提取空间特征
    :param histogram_feat: 是否提取颜色直方图
    :return:
    """
    # 在进行特征提取之前，需要对图片的格式进行转换，原项目中列举了几种格式，这里省略，先简单处理，直接使用RGB格式进行处理
    img_features = []
    # 提取HOG特征
    if hog_feat:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(img.shape[2]):
                hog_features.append(extract_hog(img[:, :, channel],
                                                orientations=orientations,
                                                pixels_per_cell=pixels_per_cell,
                                                cells_per_block=cells_per_block,
                                                visualize=visualize,
                                                transform_sqrt=transform_sqrt))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = extract_hog(img[:, :, hog_channel],
                                       orientations=orientations,
                                       pixels_per_cell=pixels_per_cell,
                                       cells_per_block=cells_per_block,
                                       visualize=visualize,
                                       transform_sqrt=transform_sqrt)
        img_features.append(hog_features)
    # 提取空间特征
    if spatial_feat:
        img_features.append(extract_spatial(img, size))
    # 提取颜色直方图
    if histogram_feat:
        img_features.append(extract_color_histogram(img, bins))
    return img_features


def extract_imgs_features(imgs, orientations, pixels_per_cell, cells_per_block, visualize, transform_sqrt, size, bins,
                          hog_feat=True, hog_channel='ALL', spatial_feat=True, histogram_feat=True):
    """
    提取一系列图片的三种特征，并保存在数组中
    :param imgs: 一系列图片
    :param orientations: 同extract_img_feature
    :param pixels_per_cell: 同extract_img_feature
    :param cells_per_block: 同extract_img_feature
    :param visualize: 同extract_img_feature
    :param transform_sqrt: 同extract_img_feature
    :param size: 同extract_img_feature
    :param bins: 同extract_img_feature
    :param hog_feat: 同extract_img_feature
    :param hog_channel: 同extract_img_feature
    :param spatial_feat: 同extract_img_feature
    :param histogram_feat: 同extract_img_feature
    :return:
    """
    imgs_features = []
    for img in imgs:
        img_features = extract_img_features(img,
                                            hog_feat=hog_feat, hog_channel=hog_channel, orientations=orientations,
                                            pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
                                            visualize=visualize, transform_sqrt=transform_sqrt,
                                            spatial_feat=spatial_feat, size=size,
                                            histogram_feat=histogram_feat, bins=bins)
        imgs_features.append(np.concatenate(img_features))
    # 最后需要转换成np数组，因为这是直接用于计算的
    return np.array(imgs_features)
