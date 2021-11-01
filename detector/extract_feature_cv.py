"""说实话，基于skimage的hog特征效果并不好，但是网上的案例，使用hog都能实现基本的效果，所以化成cv试一试"""


# 第三方库
import cv2 as cv
# 基本库
import numpy as np


def extract_hog_feature(img, winSize, blockSize, blockStride, cellSize, nbins):
    hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    hog_feature = hog.compute(img)
    return hog_feature


def extract_img_features(img, winSize, blockSize, blockStride, cellSize, nbins):
    hog_feature = extract_hog_feature(img, winSize, blockSize, blockStride, cellSize, nbins)
    img_features = hog_feature
    return img_features


def extract_imgs_features(imgs, winSize, blockSize, blockStride, cellSize, nbins):
    imgs_features = []
    for img in imgs:
        imgs_features.append(extract_img_features(img, winSize, blockSize, blockStride, cellSize, nbins))
    imgs_features = np.array(imgs_features)
    return imgs_features
