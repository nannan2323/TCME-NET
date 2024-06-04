## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import math
from einops import rearrange



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange (x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange (x, 'b (h w) c -> b c h w', h=h, w=w)

def imgtowindows(img, H_crop, W_crop):
    """
    Input: Image (B, num, C, H, W)
    Output: Window Partition (B',num, N, C)
    """
    B, num, C, H, W = img.shape
    img_reshape = img.view(B, num,  C, H // H_crop, H_crop, W // W_crop, W_crop)
    img_perm = img_reshape.permute(0, 1, 3, 5, 4, 6, 2).contiguous().reshape(-1, num, H_crop * W_crop, C)  #B', num,N,C
    return img_perm

def windowstoimg(window_tensor, B, H, W):
    """
    Input: Window Partition (B',num, N, C)
    Output: Reconstructed Image (B, num, C, H, W)
    """
    B_ , num, N, C = window_tensor.shape

    # N = window_tensor.size(3)  # 窗口数量

    # 将窗口张量重新变形为原始图像形状
    window_reshape = window_tensor.view((B_ * N)//(H*W), num, H, W, C)
    window_perm =  window_reshape.reshape(B, num, H, W, C)    #
    # window_reshape = window_tensor.reshape(B, window_tensor.shape[1], H, W, C)
    img_reconstructed = window_perm.permute(0, 1, 4, 2, 3).contiguous()


    return img_reconstructed


class BiasFree_LayerNorm (nn.Module):
    def __init__(self, normalized_shape):
        super (BiasFree_LayerNorm, self).__init__ ( )
        if isinstance (normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size (normalized_shape)

        assert len (normalized_shape) == 1

        self.weight = nn.Parameter (torch.ones (normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var (-1, keepdim=True, unbiased=False)
        return x / torch.sqrt (sigma + 1e-5) * self.weight


class WithBias_LayerNorm (nn.Module):
    def __init__(self, normalized_shape):
        super (WithBias_LayerNorm, self).__init__ ( )
        if isinstance (normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size (normalized_shape)

        assert len (normalized_shape) == 1

        self.weight = nn.Parameter (torch.ones (normalized_shape))
        self.bias = nn.Parameter (torch.zeros (normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean (-1, keepdim=True)
        sigma = x.var (-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt (sigma + 1e-5) * self.weight + self.bias


class LayerNorm (nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super (LayerNorm, self).__init__ ( )
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm (dim)
        else:
            self.body = WithBias_LayerNorm (dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d (self.body (to_3d (x)), h, w)


##  Top-K Mixed dimension Attention (TKMDA)
class TKMDAttention(nn.Module):
    def __init__(self, dim, num_heads, bias, cropsize = 8):
        super(TKMDAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.H_crop = cropsize
        self.W_crop = cropsize
        # self.norm = nn.LayerNorm (dim)
        self.conv = nn.Conv2d (dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)  # DW Conv
    # def to_crop_space(self, x ):
    #     B, num,  C, _  , _ = x.shape
    #     x = imgtowindows(x, self.H_crop, self.W_crop)  #B', num, N,C
    #     # x = x.reshape(-1, self.H_crop * self.W_crop, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()  #B',heand,N,C
    #     return x

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))    #Cx3
        q, k, v = qkv.chunk(3, dim=1)     #C

        qc = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        kc = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        vc = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        qs = rearrange(q, 'b (head c) h w -> b head c h w', head=self.num_heads)
        ks = rearrange(k, 'b (head c) h w -> b head c h w', head=self.num_heads)
        vs = rearrange(v, 'b (head c) h w -> b head c h w', head=self.num_heads)

        qs = imgtowindows(qs, self.H_crop, self.W_crop)    #B',num, N,C
        ks = imgtowindows(ks, self.H_crop, self.W_crop)
        vs = imgtowindows(vs, self.H_crop, self.W_crop)

        qc = torch.nn.functional.normalize(qc, dim=-1)
        kc = torch.nn.functional.normalize(kc, dim=-1)
        qs = torch.nn.functional.normalize (qs, dim=-1)
        ks = torch.nn.functional.normalize (ks, dim=-1)

        #对 q 沿着最后一个维度（通常是特征维度或类似通道维度）进行标准化，即将每个向量的模长（L2范数）归一化为1。
        # 这样可以确保每个向量的长度相同，便于后续的计算和应用，尤其是在多头注意力机制等模型中

        _, _, C, _ = qc.shape
        B, _, N, _ = qs.shape

        channel_mask1 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        channel_mask2 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        channel_mask3 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        channel_mask4 = torch.zeros(b, self.num_heads, C, C, device=x.device, requires_grad=False)
        spatial_mask1 = torch.zeros(B, self.num_heads, N, N, device=x.device, requires_grad=False)
        spatial_mask2 = torch.zeros(B, self.num_heads, N, N, device=x.device, requires_grad=False)
        spatial_mask3 = torch.zeros(B, self.num_heads, N, N, device=x.device, requires_grad=False)
        spatial_mask4 = torch.zeros(B, self.num_heads, N, N, device=x.device, requires_grad=False)
        channel_attn = (qc @ kc.transpose(-2, -1)) * self.temperature    #@表示矩阵乘法，transpose(-2, -1) 表示对 k 进行转置操作，交换倒数第二维和倒数第一维，这样可以满足乘法的维度要求。
        spatial_attn = (qs @ ks.transpose(-2, -1)) * self.temperature

        channel_index = torch.topk(channel_attn, k=int(C/2), dim=-1, largest=True)[1]
        spatial_index = torch.topk(spatial_attn, k=int(N/2), dim=-1, largest=True)[1]

        #torch.topk函数是PyTorch中用于在张量中找到最大的 k 个元素及其对应的索引的函数。
        #dim：指定在哪个维度上寻找最大值
        #.scatter_(-1, channel_index, 1.)
        # 1.: 是要填充的值，这里是一个浮点数1.0。
        # -1: 表示要在张量的最后一个维度（即通道维度）上进行填充
        # 它将1.0填充到 channel_mask1 的最后一个维度上的指定位置，这些位置由 index 张量给出

        channel_mask1.scatter_(-1, channel_index, 1.)
        spatial_mask1.scatter_(-1, spatial_index, 1.)
        channel_attn1 = torch.where(channel_mask1 > 0, channel_attn, torch.full_like(channel_attn, float('-inf')))
        spatial_attn1 = torch.where (spatial_mask1 > 0, spatial_attn, torch.full_like (spatial_attn, float ('-inf')))

        channel_index = torch.topk(channel_attn, k=int(C*2/3), dim=-1, largest=True)[1]
        spatial_index = torch.topk(spatial_attn, k=int(N*2/3), dim=-1, largest=True)[1]
        channel_mask2.scatter_(-1, channel_index, 1.)
        spatial_mask2.scatter_(-1, spatial_index, 1.)
        channel_attn2 = torch.where(channel_mask2 > 0, channel_attn, torch.full_like(channel_attn, float('-inf')))
        spatial_attn2 = torch.where (spatial_mask2 > 0, spatial_attn, torch.full_like(spatial_attn, float ('-inf')))

        channel_index = torch.topk(channel_attn, k=int(C*3/4), dim=-1, largest=True)[1]
        spatial_index = torch.topk(spatial_attn, k=int(N * 3/4), dim=-1, largest=True)[1]
        channel_mask3.scatter_(-1, channel_index, 1.)
        spatial_mask3.scatter_(-1, spatial_index, 1.)
        channel_attn3 = torch.where(channel_mask3 > 0, channel_attn, torch.full_like(channel_attn, float('-inf')))
        spatial_attn3 = torch.where(spatial_mask3 > 0, spatial_attn, torch.full_like(spatial_attn, float('-inf')))


        channel_index = torch.topk(channel_attn, k=int(C*4/5), dim=-1, largest=True)[1]
        spatial_index = torch.topk(spatial_attn, k=int(N * 4 / 5), dim=-1, largest=True)[1]
        channel_mask4.scatter_(-1, channel_index, 1.)
        spatial_mask4.scatter_(-1, spatial_index, 1.)
        channel_attn4 = torch.where(channel_mask4 > 0, channel_attn, torch.full_like(channel_attn, float('-inf')))
        spatial_attn4 = torch.where(spatial_mask4 > 0, spatial_attn, torch.full_like(spatial_attn, float('-inf')))

        channel_attn1 = channel_attn1.softmax(dim=-1)
        channel_attn2 = channel_attn2.softmax(dim=-1)
        channel_attn3 = channel_attn3.softmax(dim=-1)
        channel_attn4 = channel_attn4.softmax(dim=-1)
        spatial_attn1 = spatial_attn1.softmax(dim=-1)
        spatial_attn2 = spatial_attn2.softmax (dim=-1)
        spatial_attn3 = spatial_attn3.softmax (dim=-1)
        spatial_attn4 = spatial_attn4.softmax (dim=-1)

        channel_out1 = (channel_attn1 @ vc)
        channel_out2 = (channel_attn2 @ vc)
        channel_out3 = (channel_attn3 @ vc)
        channel_out4 = (channel_attn4 @ vc)
        spatial_out1 = (spatial_attn1 @ vs)
        spatial_out2 = (spatial_attn2 @ vs)
        spatial_out3 = (spatial_attn3 @ vs)
        spatial_out4 = (spatial_attn4 @ vs)

        channel_out = channel_out1 * self.attn1 + channel_out2 * self.attn2 + channel_out3 * self.attn3 + channel_out4 * self.attn4
        spatial_out = spatial_out1 * self.attn1 + spatial_out2 * self.attn2 + spatial_out3 * self.attn3 + spatial_out4 * self.attn4   #B', numhead, N, C
        spatial_out = windowstoimg(spatial_out, b, h, w)   #b
        spatial_out = rearrange(spatial_out, 'b head c h w -> b head c (h w)', head=self.num_heads)
        out = spatial_out + channel_out
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

#######################################自定义的新模块
#######多通道卷积
class OdconvAttention (nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super (OdconvAttention, self).__init__ ( )
        attention_channel = max (int (in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d (1)
        self.fc = nn.Conv2d (in_planes, attention_channel, 1, bias=False)
        self.bn = nn.BatchNorm2d (attention_channel)
        self.relu = nn.ReLU (inplace=True)

        self.channel_fc = nn.Conv2d (attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d (attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d (attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d (attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights ( )

    def _initialize_weights(self):
        for m in self.modules ( ):
            if isinstance (m, nn.Conv2d):
                nn.init.kaiming_normal_ (m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_ (m.bias, 0)
            if isinstance (m, nn.BatchNorm2d):
                nn.init.constant_ (m.weight, 1)
                nn.init.constant_ (m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid (self.channel_fc (x).view (x.size (0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid (self.filter_fc (x).view (x.size (0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc (x).view (x.size (0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid (spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc (x).view (x.size (0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax (kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool (x)
        x = self.fc (x)
        x = self.bn (x)
        x = self.relu (x)
        return self.func_channel (x), self.func_filter (x), self.func_spatial (x), self.func_kernel (x)

class ODConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 reduction=0.0625, kernel_num=4):
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = OdconvAttention(in_planes, out_planes, kernel_size, groups=groups,
                                   reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size),
                                   requires_grad=True)
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        # Multiplying channel attention (or filter attention) to weights and feature maps are equivalent,
        # while we observe that when using the latter method the models will run faster with less gpu memory cost.
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view(
            [-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding,
                          dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        return self._forward_impl(x)



class Double_convolutional_gate(nn.Module):
    """Spatial-Gate with channel information integration.
    Args:
        dim (int): Half of input channels.
    """
    def __init__(self, dim):
        super(Double_convolutional_gate, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.spatial_conv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)  # Depthwise Conv for spatial info
        self.channel_conv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)  # 1x1 Conv for channel info

    def forward(self, x, h, w):
        # Split
        x1, x2 = x.chunk(2, dim=-1)
        B, N, C = x2.shape
        # Process spatial information
        x2_spatial = self.spatial_conv(self.norm(x2).transpose(1, 2).contiguous().view(-1, C, h, w))
        x2_spatial = x2_spatial.flatten(2).transpose(-1, -2).contiguous()

        # Process channel information
        x2_channel = self.channel_conv(self.norm(x2).transpose(1, 2).contiguous().view(-1, C, h, w)).flatten(2).transpose(-1, -2).contiguous()

        # Combine spatial and channel information
        x2_combined = x2_spatial * x2_channel

        return x1 * x2_combined



class DCLF (nn.Module):
    def __init__(self, dim, hidden_dim=16, act_layer=nn.GELU, drop=0., use_eca=False):
        super (DCLF, self).__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim), act_layer())
        # self.sg = Double_convolutional_gate (hidden_dim // 2)
        self.DCG = Double_convolutional_gate(hidden_dim//2)
        # self.dwconv = nn.Sequential(
        #     nn.Conv2d (hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1), act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim//2, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim
        # self.eca = eca_layer_1d(dim) if use_eca else nn.Identity()
        self.act_layer = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # bs x hw x c
        b, c, h, w = x.size()
        # x 的形状：(batch_size, channels, height, width)
        # 将 x 的形状调整为 (batch_size * h * w, c)，为了适应 linear1 的输入
        x = x.view(b, h * w, c)  # 将高度和宽度维度展平，x 的形状：(batch_size, height * width, channels)
        x = self.linear1(x)
        x = self.act_layer(x)
        x = self.DCG(x, h, w)
        x = self.drop(x)
        x = self.linear2(x)
        x = self.drop(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        return x

class OCFE_Block(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1,
                 hidden_dim=16, act_layer=nn.GELU, drop=0., use_eca=False, bias=False,
                 LayerNorm_type = 'WithBias' ):
        super(OCFE_Block, self).__init__()

        self.layer_norm1 = LayerNorm(dim, LayerNorm_type)
        self.conv1 = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.odconv1 = ODConv2d(dim, dim, kernel_size, stride, padding, groups=dim)
        self.odconv2 = ODConv2d(dim, dim, kernel_size=5, stride=1, padding=2, groups=dim)
        self.layer_norm2 = LayerNorm(dim, LayerNorm_type)
        self.DCLF = DCLF(dim, hidden_dim, act_layer, drop, use_eca)
        self.dwconv3x3 = nn.Conv2d(dim*2, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

    def forward(self, x):
        residual1 = x
        x = self.layer_norm1(x)
        x = self.conv1(x)   #2c
        x1, x2 = x.chunk(2, dim=1)
        x1 = self.odconv1(x1)
        x1 = self.conv2(x1)
        x2 = self.odconv2(x2)
        x2 = self.conv2(x2)
        x_ = torch.cat([x1, x2], dim=1)
        x_ = self.conv2(self.dwconv3x3(x_))
        x_ = x_ + residual1

        residual2 = x_
        x3 = self.layer_norm2(x_)
        x3 = self.DCLF(x3)
        x3 = x3 + residual2
        return x3


class DCGFN(nn.Module):
    """ Spatial-Gate Feed-Forward Network.
    Args:
        in_features (int): Number of input channels.
        hidden_features (int | None): Number of hidden channels. Default: None
        out_features (int | None): Number of output channels. Default: None
        act_layer (nn.Module): Activation layer. Default: nn.GELU
        drop (float): Dropout rate. Default: 0.0
    """
    def __init__(self, dim, ffn_expansion_factor, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or dim
        # hidden_features = hidden_features or in_features
        hidden_features = int(dim * ffn_expansion_factor)  ##127
        if hidden_features % 2 !=0:
            hidden_features = hidden_features + 1
        self.fc1 = nn.Linear(dim, hidden_features)
        self.act = act_layer()
        # self.sg = SpatialGate(hidden_features//2)
        self.DCG = Double_convolutional_gate(hidden_features//2)
        self.fc2 = nn.Linear(hidden_features//2, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Input: x: (B, H*W, C), H, W
        Output: x: (B, H*W, C)
        """
        b, c, h, w = x.size ()
        x = x.permute(0, 2, 3, 1)  # 将 C 移动到最后一个维度
        x = x.view(b, h * w, c)  # 将高度和宽度维度展平，x 的形状：(batch_size, height * width, channels)
        # x = x.reshape(b, h * w, c)
        hh = int(math.sqrt(h * w))  # 计算新的 hh
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.DCG(x, h, w)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=hh, w=hh)
        return x



##########################################################################
class TransformerBlock (nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super (TransformerBlock, self).__init__ ( )

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        # self.attn = Attention (dim, num_heads, bias)
        self.attn = TKMDAttention(dim, num_heads, bias)
        self.norm2 = LayerNorm (dim, LayerNorm_type)
        # self.ffn = FeedForward (dim, ffn_expansion_factor, bias)
        self.ffn = DCGFN(dim, ffn_expansion_factor, out_features=dim)

    def forward(self, x):
        x = x + self.attn(self.norm1 (x))
        x = x + self.ffn (self.norm2 (x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed (nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super (OverlapPatchEmbed, self).__init__ ( )

        self.proj = nn.Conv2d (in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj (x)

        return x


##########################################################################
## Resizing modules
class Downsample (nn.Module):
    def __init__(self, n_feat):
        super (Downsample, self).__init__ ( )

        self.body = nn.Sequential (nn.Conv2d (n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.PixelUnshuffle (2))

    def forward(self, x):
        return self.body (x)


class Upsample (nn.Module):
    def __init__(self, n_feat):
        super (Upsample, self).__init__ ( )

        self.body = nn.Sequential (nn.Conv2d (n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.PixelShuffle (2))

    def forward(self, x):
        return self.body (x)


##########################################################################
##---------- Restormer -----------------------
class TCME (nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 ):

        super (TCME, self).__init__ ( )

        self.patch_embed = OverlapPatchEmbed (inp_channels, dim)
        self.encoder_level0 = OCFE_Block(dim=dim, kernel_size=3, stride=1,bias=bias)

        self.encoder_level1 = nn.Sequential (*[
            TransformerBlock (dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias,
                              LayerNorm_type=LayerNorm_type) for i in range (num_blocks[0])])

        self.down1_2 = Downsample (dim)  ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential (*[
            TransformerBlock (dim=int (dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, LayerNorm_type=LayerNorm_type) for i in range (num_blocks[1])])

        self.down2_3 = Downsample (int (dim * 2 ** 1))  ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential (*[
            TransformerBlock (dim=int (dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, LayerNorm_type=LayerNorm_type) for i in range (num_blocks[2])])

        self.down3_4 = Downsample (int (dim * 2 ** 2))  ## From Level 3 to Level 4
        self.latent = nn.Sequential (*[
            TransformerBlock (dim=int (dim * 2 ** 3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, LayerNorm_type=LayerNorm_type) for i in range (num_blocks[3])])

        self.up4_3 = Upsample (int (dim * 2 ** 3))  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d (int (dim * 2 ** 3), int (dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential (*[
            TransformerBlock (dim=int (dim * 2 ** 2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, LayerNorm_type=LayerNorm_type) for i in range (num_blocks[2])])

        self.up3_2 = Upsample (int (dim * 2 ** 2))  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d (int (dim * 2 ** 2), int (dim * 2 ** 1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential (*[
            TransformerBlock (dim=int (dim * 2 ** 1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, LayerNorm_type=LayerNorm_type) for i in range (num_blocks[1])])

        self.up2_1 = Upsample (int (dim * 2 ** 1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential (*[
            TransformerBlock (dim=int(dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, LayerNorm_type=LayerNorm_type) for i in range (num_blocks[0])])
        self.refinement = nn.Sequential (*[
            TransformerBlock (dim=int (dim * 2 ** 1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                              bias=bias, LayerNorm_type=LayerNorm_type) for i in range (num_refinement_blocks)])
        #########################
        self.decoder_level0 = OCFE_Block(dim=int(dim * 2 ** 1), kernel_size=3, stride=1, bias=bias)

        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d (dim, int (dim * 2 ** 1), kernel_size=1, bias=bias)

        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level0 = self.patch_embed (inp_img)
        out_enc_level0 = self.encoder_level0(inp_enc_level0)

        out_enc_level1 = self.encoder_level1 (out_enc_level0)

        inp_enc_level2 = self.down1_2 (out_enc_level1)
        out_enc_level2 = self.encoder_level2 (inp_enc_level2)

        inp_enc_level3 = self.down2_3 (out_enc_level2)
        out_enc_level3 = self.encoder_level3 (inp_enc_level3)

        inp_enc_level4 = self.down3_4 (out_enc_level3)
        latent = self.latent (inp_enc_level4)

        inp_dec_level3 = self.up4_3 (latent)
        inp_dec_level3 = torch.cat ([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3 (inp_dec_level3)
        out_dec_level3 = self.decoder_level3 (inp_dec_level3)

        inp_dec_level2 = self.up3_2 (out_dec_level3)
        inp_dec_level2 = torch.cat ([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2 (inp_dec_level2)
        out_dec_level2 = self.decoder_level2 (inp_dec_level2)

        inp_dec_level1 = self.up2_1 (out_dec_level2)
        inp_dec_level1 = torch.cat ([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1 (inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level0 = self.decoder_level0(out_dec_level1)

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv (inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)   ##3×3卷积
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level0) + inp_img

        return out_dec_level1

