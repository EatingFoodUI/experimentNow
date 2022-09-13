import torch

from .losses import FocalLoss, FocalLoss2, RegL1Loss, RegL1Loss2, RegL1Loss3, RegL1Loss4

from progress.bar import Bar
from MOC_utils.data_parallel import DataParallel
from MOC_utils.utils import AverageMeter, _tranpose_and_gather_feature

import torch.nn as nn
import torch.nn.functional as F
import math

import numpy as np


class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, batch):
        output = self.model(batch['input'])[0]

        loss, loss_stats = self.loss(output, batch)
        return output, loss, loss_stats


class MOCTrainLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MOCTrainLoss, self).__init__()
        self.crit_hm = FocalLoss()
        self.crit_mov = RegL1Loss()
        self.crit_wh = RegL1Loss()

        self.crit_mov2 = RegL1Loss2()
        self.crit_hm2 = RegL1Loss3()
        self.crit_mov3 = RegL1Loss4()

        self.opt = opt

        if opt.ID_head:
            self.emb_dim = opt.reid_dim
            self.nID = opt.nID
            self.classifier = nn.Linear(self.emb_dim, self.nID)
            self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
            # self.TriLoss = TripletLoss()
            self.emb_scale = math.sqrt(2) * math.log(self.nID - 1)
            self.s_det = nn.Parameter(-1.85 * torch.ones(1))
            self.s_id = nn.Parameter(-1.05 * torch.ones(1))

    def forward(self, output, batch):
        opt = self.opt

        def _sigmoid(x):
            y = torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)
            return y

        output['hm'] = _sigmoid(output['hm'])

        # 修改为所有帧进行处理 heatmap 检测物体的所有轨迹是否都是一个id
        # hm_loss = self.crit_hm(output['hm'], batch['hm'])
        # hm_loss = self.crit_hm2(output['hm'], batch['tube_id_similarity'],
                               # batch['center_index'], batch['mask'])

        # mov_loss 增加 轨迹中心帧在那个位置
        # output['bezier_ctp']有两个部分
        mov_loss1 = self.crit_mov(output['bezier_ctp'][:,:12,:,:], batch['mask'],
                                 batch['index'], batch['bezier_ctp'])
        # 中心帧在那个位置
        mov_loss2 = self.crit_mov2(output['bezier_ctp'][:,12,:,:], batch['centerFrame_index'],
                                  batch['center_index'], batch['mask'])

        # 中心帧位置的置信度 使用facalLoss
        # dimension equal
        # 非零的位置的值全部变成1
        target = output['bezier_ctp'][:,12,:,:].unsqueeze(1)
        target[target!=0] = 1
        # mov_loss3 = self.crit_hm(output['bezier_ctp'][:,12,:,:], batch['hm'])
        # 使用L1 loss
        mov_loss3 = self.crit_mov3(target, batch['hm'])
        # mov_loss3 = self.crit_mov2(target, batch['mask'],
        #                            batch['center_index'], batch['mask'])

        # 比例是多少
        mov_loss = opt.mov1_weight * mov_loss1 + opt.mov2_weight * mov_loss2 + opt.mov3_weight * mov_loss3

        # 轨迹中心帧在那个位置
        # mov_loss2 = self.crit_mov(output['bezier_ctp'][:,18,:,:], batch['mask'],
        #                          batch['index'], batch['center_index'])

        wh_loss = self.crit_wh(output['wh'], batch['mask'],
                               batch['index'], batch['wh'],
                               index_all=batch['index_all'])

        if opt.ID_head and opt.id_weight > 0:
            loss_id = 0
            id_head = _tranpose_and_gather_feature(output['id'], batch['index'])
            id_head = id_head[batch['mask'] > 0].contiguous()
            id_head = self.emb_scale * F.normalize(id_head)
            id_target = batch['id'][batch['mask'] > 0]
            id_output = self.classifier(id_head).contiguous()
            loss_id += self.IDLoss(id_output, id_target)

        det_loss = opt.wh_weight * wh_loss + opt.mov_weight * mov_loss
        if opt.ID_head:
            loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * loss_id + (self.s_det + self.s_id)
            loss *= 0.5
        else:
            loss = det_loss

        # MODIFY for pytorch 0.4.0
        loss = loss.unsqueeze(0)
        # hm_loss = hm_loss.unsqueeze(0)
        wh_loss = wh_loss.unsqueeze(0)
        mov_loss = mov_loss.unsqueeze(0)

        loss_stats = {'loss': loss, 'loss_bezier_ctp': mov_loss, 'loss_wh': wh_loss}
        if opt.ID_head:
            loss_id = loss_id.unsqueeze(0)
            loss_stats['loss_id'] = loss_id

        return loss, loss_stats


class MOCTrainer(object):
    def __init__(self, opt, model, optimizer=None):
        self.opt = opt
        self.optimizer = optimizer
        self.loss_stats, self.loss = self._get_losses(opt)
        self.model_with_loss = ModelWithLoss(model, self.loss)
        if opt.ID_head:
            self.optimizer.add_param_group({'params': self.loss.parameters()})

    def _get_losses(self, opt):
        loss_stats = ['loss', 'loss_bezier_ctp', 'loss_wh']
        if opt.ID_head:
            loss_stats.append('loss_id')
        loss = MOCTrainLoss(opt)
        return loss_stats, loss

    def train(self, epoch, data_loader, writer):
        return self.run_epoch('train', epoch, data_loader, writer)

    def val(self, epoch, data_loader, writer):
        return self.run_epoch('val', epoch, data_loader, writer)

    def run_epoch(self, phase, epoch, data_loader, writer):
        model_with_loss = self.model_with_loss
        if phase == 'train':
            model_with_loss.train()
        else:
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader)
        bar = Bar(opt.exp_id, max=num_iters)
        for iter, batch in enumerate(data_loader):
            if iter >= num_iters:
                break

            for k in batch:
                if k == 'input':
                    assert len(batch[k]) == self.opt.forward_frames
                    for i in range(len(batch[k])):
                        # MODIFY for pytorch 0.4.0
                        # batch[k][i] = batch[k][i].to(device=opt.device)
                        batch[k][i] = batch[k][i].type(torch.FloatTensor)
                        batch[k][i] = batch[k][i].to(device=opt.device, non_blocking=True)
                else:
                    # MODIFY for pytorch 0.4.0
                    # batch[k] = batch[k].to(device=opt.device)
                    batch[k] = batch[k].to(device=opt.device, non_blocking=True)

            output, loss, loss_stats = model_with_loss(batch)
            loss = loss.mean()

            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)

            step = iter // opt.visual_per_inter + num_iters // opt.visual_per_inter * (epoch - 1)

            for l in self.loss_stats:
                avg_loss_stats[l].update(
                    loss_stats[l].mean().item(), batch['input'][0].size(0))

                if phase == 'train' and iter % opt.visual_per_inter == 0 and iter != 0:
                    writer.add_scalar('train/{}'.format(l), avg_loss_stats[l].avg, step)
                    writer.flush()
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)
            bar.next()
            del output, loss, loss_stats

        bar.finish()
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        ret['time'] = bar.elapsed_td.total_seconds() / 60.

        return ret

    def set_device(self, gpus, chunk_sizes, device):
        if len(gpus) > 1:
            self.model_with_loss = DataParallel(
                self.model_with_loss, device_ids=gpus,
                chunk_sizes=chunk_sizes).to(device)
        else:
            self.model_with_loss = self.model_with_loss.to(device)

        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    # MODIFY for pytorch 0.4.0
                    state[k] = v.to(device=device, non_blocking=True)
                    # state[k] = v.to(device=device)
