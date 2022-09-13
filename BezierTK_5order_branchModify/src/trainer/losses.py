# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
import torch
from MOC_utils.utils import _tranpose_and_gather_feature
import torch.nn.functional as F

# hm在使用 pred:[1,7,152,272]  gt:[1,256,7]
def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
      Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    '''
    if gt.ndim != 4:
        gt = gt.unsqueeze(1)
    pos_inds = gt.eq(1).float()
    # print(pos_inds.shape) #(1, 128, 72)
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    # loss = 0
    loss = torch.Tensor([0.0]).float().cuda()
    
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(torch.nn.Module):
    '''torch.nn.Module warpper for focal loss'''

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


class RegL1Loss(torch.nn.Module):
    def __init__(self):
        super(RegL1Loss, self).__init__()

#        mov_loss = self.crit_mov(output['bezier_ctp'], batch['mask'],
#                                 batch['index'], batch['bezier_ctp'])
    def forward(self, output, mask, index, target, index_all=None):
        pred = _tranpose_and_gather_feature(output, index, index_all=index_all)
        # pred --> b, N, 2*K
        # mask --> b, N ---> b, N, 2*K
        mask = mask.unsqueeze(2).expand_as(pred).float()
        # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
        # loss = F.l1_loss(pred * mask, target * mask, size_average=False)
        loss = F.l1_loss(pred * mask, target * mask, reduction='sum')
        '''
        if index_all is None:
            # first_ctp(t) >= last_ctp(t)
            mask_flargerl = (pred[..., 2] >= pred[..., 11]).unsqueeze(2).expand_as(pred).float()
            # first_ctp(t) < last_ctp(t)
            mask_fsmallerl = (pred[..., 2] < pred[..., 11]).unsqueeze(2).expand_as(pred).float()

            ctp_mul_sign = torch.sign(pred[..., 2] * pred[..., 11])
            mask_neg = (ctp_mul_sign <= 0).unsqueeze(2).expand_as(pred).float()
            mask_pos = (ctp_mul_sign > 0).unsqueeze(2).expand_as(pred).float()

            neg_smaller_loss = F.l1_loss(pred * mask * mask_neg * mask_fsmallerl,
                                         target * mask * mask_neg * mask_fsmallerl, reduction='sum')
            pos_smaller_loss = 4 * F.l1_loss(pred * mask * mask_pos * mask_fsmallerl,
                                             target * mask * mask_pos * mask_fsmallerl, reduction='sum')
            neg_larger_loss = 4 * F.l1_loss(pred * mask * mask_neg * mask_flargerl,
                                            target * mask * mask_neg * mask_flargerl, reduction='sum')
            pos_larger_loss = 8 * F.l1_loss(pred * mask * mask_pos * mask_flargerl,
                                            target * mask * mask_pos * mask_flargerl, reduction='sum')
            loss = neg_smaller_loss + pos_smaller_loss + neg_larger_loss + pos_larger_loss
            # print(loss)
        '''
        # print('-'*80)
        loss = loss / (mask.sum() + 1e-4)
        return loss

# 专门处理movement branch新添加的通道
class RegL1Loss2(torch.nn.Module):
    def __init__(self):
        super(RegL1Loss2, self).__init__()

#        mov_loss = self.crit_mov(output['bezier_ctp'], batch['mask'],
#                                 batch['index'], batch['bezier_ctp'])
    # 先使用center_index, then centerFrame_index
    def forward(self, output, centerFrame_index, center_index, mask):
        output = output.unsqueeze(0)
        pred = _tranpose_and_gather_feature(output, center_index, index_all=None)

        centerFrame_index = centerFrame_index.permute(1, 0).contiguous().unsqueeze(0)
        mask = mask.unsqueeze(2)
        loss = F.l1_loss(pred * mask, centerFrame_index * mask, reduction='sum')

        # print('-'*80)
        loss = loss / (mask.sum() + 1e-4)
        return loss

# 修改的hm branch使用的loss
class RegL1Loss3(torch.nn.Module):
    def __init__(self):
        super(RegL1Loss3, self).__init__()

#        mov_loss = self.crit_mov(output['bezier_ctp'], batch['mask'],
#                                 batch['index'], batch['bezier_ctp'])

    # 先使用center_index, then centerFrame_index
    def forward(self, output, tube_id_similarity, all_center_index, mask):
        # index_all,赋值没有什么意义，主要是为了标记，且不会引起后续冲突
        pred = _tranpose_and_gather_feature(output, all_center_index, index_all=torch.ones(1).sum())

        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = F.l1_loss(pred * mask, tube_id_similarity * mask, reduction='sum')

        # print('-'*80)
        loss = loss / (mask.sum() + 1e-4)
        return loss
