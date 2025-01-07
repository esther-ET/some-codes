import math

import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d


class DeformableConv2d(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        *,
        offset_groups=1,
        with_mask=False
    ):
        super().__init__()
        assert in_dim % groups == 0
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim // groups, kernel_size, kernel_size))
        # 用kaiming初始化权重！！！！！！！！！！！！！！！
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_dim))
        else:
            self.bias = None

        self.with_mask = with_mask
        if with_mask:
            # batch_size, (2+1) * offset_groups * kernel_height * kernel_width, out_height, out_width
            self.param_generator = nn.Conv2d(in_dim, 3 * offset_groups * kernel_size * kernel_size, 3, 1, 1)
        else:
            self.param_generator = nn.Conv2d(in_dim, 2 * offset_groups * kernel_size * kernel_size, 3, 1, 1)

    def forward(self, x):
        if self.with_mask:
            oh, ow, mask = self.param_generator(x).chunk(3, dim=1)
            offset = torch.cat([oh, ow], dim=1)
            mask = mask.sigmoid()
        else:
            offset = self.param_generator(x)
            mask = None

        print("weight", self.weight)
        x = deform_conv2d(
            x,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=mask,
        )
        return x

tensor = torch.randn(4, 3, 3, 3)
print("input tensor", tensor)
conv = DeformableConv2d(3, 2, 3, stride=1,padding=1, offset_groups=1, with_mask=False, bias=False)
output = conv(tensor)
print("output tensor", output)



# import torch
# from torchvision.ops import deform_conv2d
#
# h = w = 3
#
# # batch_size, num_channels, out_height, out_width B C H W
# x = torch.arange(h * w * 3, dtype=torch.float32).reshape(1, 3, h, w)
#
# # to show the effect of offset more intuitively, only the case of kh=kw=1 is considered here
# offset = torch.FloatTensor(
#     [  # create our predefined offset with offset_groups = 3
#         0, -1,  # sample the left pixel of the centroid pixel
#         0, 1,  # sample the right pixel of the centroid pixel
#         -1, 0,  # sample the top pixel of the centroid pixel
#     ]  # here, we divide the input channels into offset_groups groups with different offsets.
# ).reshape(1, 2 * 3 * 1 * 1, 1, 1)
# # here we use the same offset for each local neighborhood in the single channel
# # so we repeat the offset to the whole space: batch_size, 2 * offset_groups * kh * kw, out_height, out_width
# offset = offset.repeat(1, 1, h, w)
#
# weight = torch.FloatTensor(
#     [
#         [1, 0, 0],  # only extract the first channel of the input tensor
#         [0, 1, 0],  # only extract the second channel of the input tensor
#         [1, 1, 0],  # add the first and the second channels of the input tensor
#         [0, 0, 1],  # only extract the third channel of the input tensor
#         [0, 1, 0],  # only extract the second channel of the input tensor
#     ]
# ).reshape(5, 3, 1, 1)
# deconv_shift = deform_conv2d(x, offset=offset, weight=weight)
# print(deconv_shift)
#
# """
# tensor([[
# [[ 0.,  0.,  1.],  # offset=(0, -1) the first channel of the input tensor
# [ 0.,  3.,  4.],  # output hw indices (1, 2) => (1, 2-1) => input indices (1, 1)
# [ 0.,  6.,  7.]], # output hw indices (2, 1) => (2, 1-1) => input indices (2, 0)
#
# [[10., 11.,  0.],  # offset=(0, 1) the second channel of the input tensor
# [13., 14.,  0.],  # output hw indices (1, 1) => (1, 1+1) => input indices (1, 2)
# [16., 17.,  0.]], # output hw indices (2, 0) => (2, 0+1) => input indices (2, 1)
#
# [[10., 11.,  1.],  # offset=[(0, -1), (0, 1)], accumulate the first and second channels after being sampled with an offset.
# [13., 17.,  4.],
# [16., 23.,  7.]],
#
# [[ 0.,  0.,  0.],  # offset=(-1, 0) the third channel of the input tensor
# [18., 19., 20.],  # output hw indices (1, 1) => (1-1, 1) => input indices (0, 1)
# [21., 22., 23.]], # output hw indices (2, 2) => (2-1, 2) => input indices (1, 2)
#
# [[10., 11.,  0.],  # offset=(0, 1) the second channel of the input tensor
# [13., 14.,  0.],  # output hw indices (1, 1) => (1, 1+1) => input indices (1, 2)
# [16., 17.,  0.]]  # output hw indices (2, 0) => (2, 0+1) => input indices (2, 1)
# ]])
# """