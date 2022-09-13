from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch
from torch import nn
from .DCNv2.dcn_v2 import DCN

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class MOC_Branch(nn.Module):
    def __init__(self, input_channel, head_conv, branch_info, K):
        super(MOC_Branch, self).__init__()
        assert head_conv > 0
        self.K = K

        # hm head只处理中间帧的fmap
        self.hm = nn.Sequential(
            nn.Conv2d(input_channel, head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, branch_info['hm'],
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        self.hm[-1].bias.data.fill_(-2.19)

        # wh head 分别处理每一帧的检测框高宽
        self.wh = nn.Sequential(
            nn.Conv2d(input_channel, head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            # 通道数为所有帧×2(高宽)
            # K是使用的视频总帧数
            nn.Conv2d(head_conv, branch_info['wh'],
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        fill_fc_weights(self.wh)

    def forward(self, input_chunk):
        output = {}

        output['hm'] = self.hm(input_chunk)
        output['wh'] = self.wh(input_chunk)
        return output
