import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import torchvision.models as models
from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']

# 使用装饰器定义带有权重的 L1 损失函数
@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')

# 使用装饰器定义带有权重的 MSE 损失函数
@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')

#####原始注释
# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss



def angle(a, b):
    vector = torch.mul(a, b)
    up = torch.sum(vector)
    norm_a = torch.norm(a)
    norm_b = torch.norm(b)
    down = norm_a * norm_b
    theta = torch.acos(up / down)  # 弧度制
    return theta

class ColorAngleLoss (nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(ColorAngleLoss, self).__init__ ( )
        self.loss_weight = loss_weight
        self.reduction = reduction
    def forward(self, pred, target):
        # If = get_if_tensor(pred, target)
        c = pred.shape[1]
        channel_losses = [angle(pred[:, i, :, :], target[:, i, :, :]) for i in range(c)]
        color_angle_losses = torch.mean(torch.stack(channel_losses))

        if self.loss_weight is not None:
            color_angle_losses = color_angle_losses * self.loss_weight
        if self.reduction == 'mean':
            color_angle_losses = color_angle_losses.mean ( )
        elif self.reduction == 'sum':
            color_angle_losses = color_angle_losses.sum ( )

        return color_angle_losses


# --- Edge loss function  --- #
class Edg_Capture(nn.Module):
    def __init__(self):
        super(Edg_Capture, self).__init__()

    def forward(self, x):
        c= x.shape[1]
        x_diffx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])  # 计算水平方向上的差异
        x_diffy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])  # 计算垂直方向上的差异
        y = x.new(x.size())  # 创建一个和输入张量 x 大小相同的新张量 y，并将其填充为0
        y.fill_(0)
        # 在每个通道上累加水平方向上的边缘信息
        y[:, :, :, 1:] += x_diffx
        y[:, :, :, :-1] += x_diffx
        # 在每个通道上累加垂直方向上的边缘信息
        y[:, :, 1:, :] += x_diffy
        y[:, :, :-1, :] += x_diffy
        y = torch.sum(y, 1, keepdim=True) / c
        y /= 4  # 归一化边缘信息的值, 水平垂直共四个方向
        return y

class Edge_Loss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super (Edge_Loss, self).__init__ ( )
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, x, y):
        laplace = Edg_Capture()
        L1 = nn.L1Loss()
        out = L1(laplace(x), laplace(y))
        if self.loss_weight is not None:
            edge_losses = out * self.loss_weight
        if self.reduction == 'mean':
            edge_losses = out.mean()
        elif self.reduction == 'sum':
            edge_losses = out.sum()

        return  edge_losses




