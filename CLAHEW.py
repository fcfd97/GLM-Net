import cv2 as cv


def equalHist(image):
    """直方图均衡化，图像增强的一个方法"""
    # 彩色图片转换为灰度图片

    # 直方图均衡化，自动调整图像的对比度，让图像变得清晰
    dst = cv.equalizeHist(image)
    return dst


def clahe(image): # 二维切片

    """
    局部直方图均衡化
    把整个图像分成许多小块（比如按8*8作为一个小块），
    那么对每个小块进行均衡化。
    这种方法主要对于图像直方图不是那么单一的（比如存在多峰情况）图像比较实用
    """
    # 彩色图片转换为灰度图片
    # cliplimit：灰度值
    # tilegridsize：图像切割成块，每块的大小
    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    dst = clahe.apply(image)
    return dst

