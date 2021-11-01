"""此文件用于读取与预处理图片"""

# 图片读取
from skimage.io import imread
from os import walk, path
# 图片处理
import cv2 as cv
from skimage.transform import resize
# 基本库
import os
from itertools import product


def train_test_pos_neg(root_dir):
    # 读取按照train、test与pos、neg分类好的文件夹
    data = []
    for mode, category in product(('train', 'test'), ('pos', 'neg')):
        print('正在读取' + os.path.join(os.path.join(root_dir, mode), category))
        data.append(file2data(os.path.join(os.path.join(root_dir, mode), category)))
        print('共读取到' + str(len(data[-1])))
    return data[0], data[1], data[2], data[3]


def file2data(root_dir):
    # 读取目标文件夹内的所有图片，仅读取，不做格式处理，后续根据其它函数进行转换
    # 此处剔除透明度信息
    data = []
    for root, dirs, files in walk(root_dir, topdown=True):
        for file in files:
            # 拆解文件名，检查是否是图片文件
            _, ending = path.splitext(file)
            if ending != '.jpg' and ending != '.jpeg' and ending != '.png':
                continue
            else:
                data.append(imread(path.join(root, file))[:, :, :3])
    return data


def uniform_imgs_size(imgs, size):
    """
    将所有图片转化为统一的格式，这个格式可以由使用者自定，反正cv2.resize都能做到，考虑到现实使用场景的图片为扁行，对于高大于宽的图片，拉升后进行黑
    色填充
    :param imgs: 所有图片（单张也可以）
    :param size: 目标尺寸
    :return:
    """
    # 不改变原数据
    imgs_handled = []
    for img in imgs:
        if img.shape[0] > img.shape[1]:
            img_handled = resize(img, size)
            imgs_handled.append(img_handled)
        elif img.shape[0] <= img.shape[1]:
            img_next = resize(img, (int(img.shape[0] * size[1] / img.shape[1]), size[1]))
            top = int((size[0] - img_next.shape[0]) / 2)
            bottom = int(size[0] - img_next.shape[0] - top)
            img_handled = cv.copyMakeBorder(img_next, top=top, bottom=bottom, left=0, right=0,
                                            borderType=cv.BORDER_CONSTANT, value=(255, 255, 255))
            imgs_handled.append(img_handled)
    return imgs_handled


def uniform_imgs_size_simple(imgs, size):
    """
    将所有图片转化为统一的格式，这个格式可以由使用者自定，反正cv2.resize都能做到，考虑到现实使用场景的图片为扁行，对于高大于宽的图片，拉升后进行黑
    色填充
    :param imgs: 所有图片（单张也可以）
    :param size: 目标尺寸
    :return:
    """
    # 不改变原数据
    imgs_handled = []
    for img in imgs:
        img_handled = resize(img, size)
        imgs_handled.append(img_handled)
    return imgs_handled
