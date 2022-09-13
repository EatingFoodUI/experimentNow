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


# 模型的分支
class MOC_Branch(nn.Module):
    def __init__(self, input_channel, head_conv, branch_info, K):
        super(MOC_Branch, self).__init__()
        assert head_conv > 0
        # k是在bezier曲线绘制时使用帧的间隔
        self.K = K

        # 三个branch都是对应单个video的某个行人轨迹来定义的

        # hm head只处理中间帧的fmap
        # 中心帧（关键帧）的被检测的行人特征
        self.hm = nn.Sequential(
            nn.Conv2d(input_channel, head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_conv, branch_info['hm'],
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        # conv2 初始化设置bias=-2.19 ？？
        self.hm[-1].bias.data.fill_(-2.19)

        # mov head一次性处理所有帧的fmap
        BN_MOMENTUM = 0.1
        self.bezier_ctp = nn.Sequential(
            # nn.Conv2d(K * input_channel, head_conv,
            #           kernel_size=3, padding=1, bias=True),
            # 使用deformable卷积层
            # 把间隔帧之间的k帧的移动信息全部获取
            DCN(K * input_channel, head_conv, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=K),
            nn.BatchNorm2d(head_conv, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            # 输出通道数12，代表移动的12个方向
            nn.Conv2d(head_conv, branch_info['bezier_ctp'],
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        # 设置全连接层的初始权重为0
        fill_fc_weights(self.bezier_ctp)

        # ID head一次性处理所有帧的fmap
        # 跟hm head只有k的差距，间隔了k帧
        self.ID = nn.Sequential(
            nn.Conv2d(K * input_channel, head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            # ?? branch_info['id']
            nn.Conv2d(head_conv, branch_info['id'],
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        fill_fc_weights(self.ID)

        # wh head 分别处理所有帧的fmap（要处理每个帧的检测对象的高宽）
        self.wh = nn.Sequential(
            nn.Conv2d(input_channel, head_conv,
                      kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            # K是使用的视频总帧数
            nn.Conv2d(head_conv, branch_info['wh'] // K,
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        fill_fc_weights(self.wh)

    # 分支的执行流程
    # input_chunk：backbone的输出
    def forward(self, input_chunk):
        output = {}
        output_wh = []
        # feature：每一帧的特征
        for feature in input_chunk:
            # 将每一帧的特征放入wh branch得到wh预测，再以此放入output_wh中
            output_wh.append(self.wh(feature))

        # 将中间帧的特征放入hm branch得到key frame的heatmap
        output['hm'] = self.hm(input_chunk[self.K // 2])

        # 多帧特征合并到一个张量上执行
        input_chunk = torch.cat(input_chunk, dim=1)
        output_wh = torch.cat(output_wh, dim=1)

        # 将所有帧的特征一并放入bezier_ctp branch得到movement prediction
        output['bezier_ctp'] = self.bezier_ctp(input_chunk)
        # id branch干什么？？（128个物体类别的概率）
        # 将所有帧的特征一并放入id branch得到id prediction
        output['id'] = self.ID(input_chunk)
        output['wh'] = output_wh
        return output

