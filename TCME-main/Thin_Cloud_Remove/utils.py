import numpy as np
import os
import cv2
import math
import rasterio
from rasterio.transform import Affine

def calculate_psnr(img1, img2, border=0):
    # img1 and img2 have range [0, 255]
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


# --------------------------------------------
# SSIM
# --------------------------------------------
def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w,  = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
        elif img1.shape[2] == 4:
            ssims = []
            for i in range(4):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


# def load_img(filepath):
#     return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGBA)

# def save_img(filepath, img):
#     cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
def save_img(filepath, img):
    converted_img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)  # 将图像颜色空间转换为RGBA
    cv2.imwrite(filepath, converted_img)


def load_tif(filepath):
    with rasterio.open(filepath) as tif:
        img = tif.read()  # Shape: (bands, height, width)
        img = np.moveaxis(img, 0, -1)  # 转为hwc
        return img


def save_RGB_tif(filepath, img):
    # 获取图像数组的形状信息
    height, width, num_channels = img.shape

    # 创建保存TIFF文件所需的元数据
    transform = Affine(1, 0, 0, 0, -1, height)
    count = 3  # 保留前三个通道
    dtype = img.dtype

    # 打开TIFF文件进行写入
    with rasterio.open(filepath, 'w', driver='GTiff', height=height, width=width, count=count, dtype=dtype,
                       transform=transform) as dst:
        # 交换通道顺序为321并写入TIFF文件
        dst.write(img[:, :, 2], indexes=1)  #R  # 通道索引以0为基准，索引为1表示第一个通道
        dst.write(img[:, :, 1], indexes=2)  #G
        dst.write(img[:, :, 0], indexes=3)  #B



def save_tif(filepath, img):
    # 获取数组的形状信息
    height, width, num_channels = img.shape

    # 创建保存TIF文件的元数据
    transform = Affine(1, 0, 0, 0, -1, height)  # 为了保持像素与空间坐标的对应关系，这里使用单位矩阵
    count = num_channels  # 通道数
    dtype = img.dtype  # 数据类型

    # 打开TIF文件进行写入
    with rasterio.open(filepath, 'w', driver='GTiff', height=height, width=width, count=count, dtype=dtype,
                        transform=transform) as dst:
        # 逐个通道写入TIF文件
        for i in range(num_channels):
            dst.write(img[:, :, i], indexes=i + 1)

def load_gray_img(filepath):
    return np.expand_dims(cv2.imread(filepath, cv2.IMREAD_GRAYSCALE), axis=2)

def save_gray_img(filepath, img):
    cv2.imwrite(filepath, img)
