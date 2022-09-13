import torch

from .losses import FocalLoss, RegL1Loss

from progress.bar import Bar
from MOC_utils.data_parallel import DataParallel
from MOC_utils.utils import AverageMeter, _tranpose_and_gather_feature

import torch.nn as nn
import torch.nn.functional as F
import math

# 有损失函数记录的模型
class ModelWithLoss(torch.nn.Module):
    def __init__(self, model, loss):
        super(ModelWithLoss, self).__init__()
        self.model = model
        self.loss = loss

    # 记录这个批次训练的loss
    def forward(self, batch):
        # 得到输出
        output = self.model(batch['input'])[0]

        # 计算损失，调用MOCTrainLoss
        loss, loss_stats = self.loss(output, batch)
        return output, loss, loss_stats

# 训练的loss函数
class MOCTrainLoss(torch.nn.Module):
    def __init__(self, opt):
        super(MOCTrainLoss, self).__init__()
        self.crit_hm = FocalLoss()
        self.crit_mov = RegL1Loss()
        self.crit_wh = RegL1Loss()

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

        hm_loss = self.crit_hm(output['hm'], batch['hm'])
        # print(batch['hm'])

        mov_loss = self.crit_mov(output['bezier_ctp'], batch['mask'],
                                 batch['index'], batch['bezier_ctp'])

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

        det_loss = opt.hm_weight * hm_loss + opt.wh_weight * wh_loss + opt.mov_weight * mov_loss
        if opt.ID_head:
            loss = torch.exp(-self.s_det) * det_loss + torch.exp(-self.s_id) * loss_id + (self.s_det + self.s_id)
            loss *= 0.5
        else:
            loss = det_loss

        # MODIFY for pytorch 0.4.0
        loss = loss.unsqueeze(0)
        hm_loss = hm_loss.unsqueeze(0)
        wh_loss = wh_loss.unsqueeze(0)
        mov_loss = mov_loss.unsqueeze(0)

        loss_stats = {'loss': loss, 'loss_hm': hm_loss,
                      'loss_bezier_ctp': mov_loss, 'loss_wh': wh_loss}
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
        loss_stats = ['loss', 'loss_hm', 'loss_bezier_ctp', 'loss_wh']
        if opt.ID_head:
            loss_stats.append('loss_id')
        loss = MOCTrainLoss(opt)
        return loss_stats, loss

    # 训练一轮
    def train(self, epoch, data_loader, writer):
        return self.run_epoch('train', epoch, data_loader, writer)

    def val(self, epoch, data_loader, writer):
        return self.run_epoch('val', epoch, data_loader, writer)

    # 执行训练/验证一轮
    def run_epoch(self, phase, epoch, data_loader, writer):
        # 初始化模型损失
        model_with_loss = self.model_with_loss
        # 是训练还是验证
        if phase == 'train':
            model_with_loss.train()
        else:
            model_with_loss.eval()
            torch.cuda.empty_cache()

        opt = self.opt
        avg_loss_stats = {l: AverageMeter() for l in self.loss_stats}
        num_iters = len(data_loader)
        bar = Bar(opt.exp_id, max=num_iters)
        # iter：index；batch：当前训练用的数据
        # 共533轮
        for iter, batch in enumerate(data_loader):
            if iter >= num_iters:
                break

            for k in batch:
                # 对当前批次输入的7帧进行数据类型的修改，将当前批次数据装载到GPU上
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

            # 将当前采样的数据放入模型中训练并计算损失
            # 执行 model_with_loss：ModelWithLoss.forward()
            output, loss, loss_stats = model_with_loss(batch)
            # 损失均值
            loss = loss.mean()

            # 清除梯度，后向传播
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # ??
            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                epoch, iter, num_iters, phase=phase,
                total=bar.elapsed_td, eta=bar.eta_td)

            # ??
            step = iter // opt.visual_per_inter + num_iters // opt.visual_per_inter * (epoch - 1)

            # 四个损失
            for l in self.loss_stats:
                # 更新平均损失
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
