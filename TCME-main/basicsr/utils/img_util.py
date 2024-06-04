import cv2
import math
import numpy as np
import os
import torch
from torchvision.utils import make_grid
import rasterio
from rasterio.transform import from_origin

def save_image_with_rasterio(img_np, output_path):
    # 获取图像的高度和宽度
    height, width = img_np.shape[:2]

    # 创建一个新的带有指定高度和宽度的空白图像
    output_image = np.zeros((height, width), dtype=np.uint8)

    # 将图像数据复制到空白图像中
    output_image[:, :] = img_np[:, :]

    # 通过 raterio 创建一个新的空白栅格图像文件
    with rasterio.open(output_path, 'w', driver='GTiff', height=height, width=width, count=1,
                       dtype=output_image.dtype) as dst:

        # 设置栅格图像的变换参数（原点、水平和垂直分辨率）
        dst.transform = from_origin(0, 0, 1, 1)

        # 将图像数据写入栅格图像文件
        dst.write(output_image, indexes=1)

def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img.shape[2] == 4 and bgr2rgb:
            img = img
        img = torch.from_numpy(img.transpose(2, 0, 1)) #CHW
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list)
             and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')
    #检查输入参数 tensor 是否为张量或张量列表，如果不是，则会引发 TypeError。

    if torch.is_tensor(tensor):  #将单个张量转换为张量列表（如果输入为单个张量）
        tensor = [tensor]
    result = []   #初始化结果列表
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)   #通过调用 squeeze(0) 方法去除张量的第一个维度（如果存在），这将移除一个维度为 1 的批次维度。
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])
        n_dim = _tensor.dim()
        if n_dim == 4:                #4D
            # img_np = make_grid(
            #     _tensor, nrow=int(math.sqrt(_tensor.size(0))),
            #     normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGRA if img_np.shape[2] == 4 else cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()   ###转为numpy数组
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            elif img_np.shape[2] == 4:
                img_np = img_np
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. '
                            f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


import rasterio


def tiffrombytes(filepath, float32=False):
    """读取四波段TIF图像。

    参数：
        filepath
        float32（bool）：是否将图像转换为float32类型。如果为True，还会将图像归一化到[0, 1]范围内。默认为False。
    返回：
        ndarray：加载的图像数组。
    """
    with rasterio.open(filepath) as dataset:
        img = dataset.read()

    if float32:
        img = img.astype(np.float32)
        img = img / 255
    img = np.moveaxis(img, 0, -1)   #调整原始（通道，高，宽）顺序为（高，宽，通道）使其与CV2读取方式一致
    return img



def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.
    参数:
        content (bytes):从文件或其他流获得的图像字节。
        flag (str):指定已加载图像的颜色类型的标志;
        候选项是“彩色”、“灰度”和“不变”。
        float32 (bool):是否转换为float32。，若为真，也将规范
        到[0,1]。默认值:False。
    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {
        'color': cv2.IMREAD_COLOR,
        'grayscale': cv2.IMREAD_GRAYSCALE,
        'unchanged': cv2.IMREAD_UNCHANGED    #不改变通道
    }
    if img_np is None:
        raise Exception('None .. !!!')
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.
    return img

def imfrombytesDP(content, flag='color', float32=False):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    if img_np is None:
        raise Exception('None .. !!!')
    img = cv2.imdecode(img_np, cv2.IMREAD_UNCHANGED)
    if float32:
        img = img.astype(np.float32) / 65535.
    return img

def padding(img_lq, img_gt, gt_size):
    h, w, _, = img_lq.shape[0], img_lq.shape[1], img_lq.shape[2]

    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)
    
    if h_pad == 0 and w_pad == 0:
        return img_lq, img_gt

    img_lq = cv2.copyMakeBorder(img_lq, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_gt = cv2.copyMakeBorder(img_gt, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    # print('img_lq', img_lq.shape, img_gt.shape)
    if img_lq.ndim == 2:
        img_lq = np.expand_dims(img_lq, axis=2)
    if img_gt.ndim == 2:
        img_gt = np.expand_dims(img_gt, axis=2)
    return img_lq, img_gt

def padding_DP(img_lqL, img_lqR, img_gt, gt_size):
    h, w, c = img_gt.shape

    h_pad = max(0, gt_size - h)
    w_pad = max(0, gt_size - w)
    
    if h_pad == 0 and w_pad == 0:
        return img_lqL, img_lqR, img_gt

    img_lqL = cv2.copyMakeBorder(img_lqL, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_lqR = cv2.copyMakeBorder(img_lqR, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    img_gt  = cv2.copyMakeBorder(img_gt,  0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    # print('img_lq', img_lq.shape, img_gt.shape)
    return img_lqL, img_lqR, img_gt

def imwrite(img, file_path, params=None, auto_mkdir=True):
    """Write image to file.

    Args:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)


def crop_border(imgs, crop_border):
    """Crop borders of images.

    Args:
        imgs (list[ndarray] | ndarray): Images with shape (h, w, c).
        crop_border (int): Crop border for each end of height and weight.

    Returns:
        list[ndarray]: Cropped images.
    """
    if crop_border == 0:
        return imgs
    else:
        if isinstance(imgs, list):
            return [
                v[crop_border:-crop_border, crop_border:-crop_border, ...]
                for v in imgs
            ]
        else:
            return imgs[crop_border:-crop_border, crop_border:-crop_border,
                        ...]
