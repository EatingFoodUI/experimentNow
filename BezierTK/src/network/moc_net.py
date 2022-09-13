from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
from .branch import MOC_Branch
from .dla import MOC_DLA
from .resnet import MOC_ResNet
backbone = {
    'dla': MOC_DLA,
    'resnet': MOC_ResNet
}


# 整个网络的模型
class MOC_Net(nn.Module):
    def __init__(self, arch, num_layers, branch_info, head_conv, K):
        super(MOC_Net, self).__init__()
        # k是在bezier曲线绘制时使用帧的间隔(一个序列帧的数量)
        self.K = K
        # 模型的backbone
        self.backbone = backbone[arch](num_layers)
        # for param in self.backbone.parameters():
        #     param.requires_grad = False

        # 模型的分支（三个分支）
        self.branch = MOC_Branch(self.backbone.output_channel, head_conv, branch_info, K)
        # ？？最后Key frame的feature map
        self.last_K_fmap = {}
        # for name, layer in self.branch.named_parameters():
        #     if 'hm' not in name:
        #         layer.requires_grad = False

    def forward(self, input):
        if isinstance(input, dict):
            to_del = []
            for img_id in self.last_K_fmap:
                if img_id not in input:
                    to_del.append(img_id)

            for k in to_del:
                del self.last_K_fmap[k]

            chunk = []
            for img_id in input:
                if img_id in self.last_K_fmap:
                    chunk.append(self.last_K_fmap[img_id])
                else:
                    f = self.backbone(input[img_id])
                    chunk.append(f)
                    self.last_K_fmap[img_id] = f
        else:
            # 先放入backbone
            chunk = [self.backbone(input[i]) for i in range(self.K)]

        # 然后放入分支
        return [self.branch(chunk)]
