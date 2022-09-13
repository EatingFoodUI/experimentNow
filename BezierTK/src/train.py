from opts import opts
import torch
import random
import numpy as np
from datasets.init_dataset import get_dataset
from tracking_utils.timer import Timer
from trainer.logger import Logger
from MOC_utils.model import create_model, load_model, save_model, load_coco_pretrained_model, \
    load_imagenet_pretrained_model, load_fairmot_pretrained_model
import os
from trainer.moc_trainer import MOCTrainer
from inference.normal_inference import normal_inference
from ACT import frameAP
import tensorboardX
import time


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


GLOBAL_SEED = 317

# 设置随机数
def worker_init_fn(dump):
    set_seed(GLOBAL_SEED)


def main(opt):
    set_seed(opt.seed)
    torch.backends.cudnn.benchmark = True
    print()
    # 默认MOT17（mot16，mot17，mot20），MOT_train
    print('dataset: ' + opt.dataset + '   task:  ' + opt.task)
    Dataset = get_dataset(opt.dataset)
    # 分支通道数设计
    opt = opts().update_dataset(opt, Dataset)

    train_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(opt.log_dir, 'train'))
    epoch_train_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(opt.log_dir, 'train_epoch'))
    val_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(opt.log_dir, 'val'))
    epoch_val_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(opt.log_dir, 'val_epoch'))

    logger = Logger(opt, epoch_train_writer, epoch_val_writer)

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
    # 创建模型
    model = create_model(opt.arch, opt.branch_info, opt.head_conv, opt.forward_frames)

    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)
    # optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad is not False, model.parameters()), opt.lr)
    # 使用MOC的训练方法进行训练（初始化MOC训练器和MOC损失函数）
    trainer = MOCTrainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
    start_epoch = opt.start_epoch

    # coco|imagenet|fairmot
    # 将预训练模型的参数导入到建立的模型中
    if opt.pretrain_model == 'coco':
        model = load_coco_pretrained_model(opt, model)
    elif opt.pretrain_model == 'fairmot':
        model = load_fairmot_pretrained_model(opt, model)
    else:
        model = load_imagenet_pretrained_model(opt, model)

    # 也可以直接使用训练过的模型，在opts.py的opt.load_model=中设置
    # 这样start_epoch会更新成导入模型的训练轮次
    if opt.load_model != '':
        model, optimizer, start_epoch, _ = load_model(model, opt.load_model, trainer.optimizer, opt.lr)

    # 数据装载
    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
        drop_last=True,
        worker_init_fn=worker_init_fn
    )

    print('training...')
    print('GPU allocate:', opt.chunk_sizes)
    best_ap = 0
    # opt.num_epochs = 20
    best_epoch = start_epoch

    # 使用loss来判断最好模型
    best_loss = 1

    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        timer = Timer()
        timer.tic()

        # 模型训练一轮
        log_dict_train = trainer.train(epoch, train_loader, train_writer)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('epoch/{}'.format(k), v, epoch, 'train')
            logger.write('train: {} {:8f} | '.format(k, v))
        logger.write('\n')

        # 保存每轮训练参数
        # 临时设置为false
        # if opt.save_all != false:
        if opt.save_all:
            # time_str = time.strftime('%Y-%m-%d-%H-%M')
            # model_name = 'model_[{}]_{}.pth'.format(epoch, time_str)
            model_name = 'model_{}.pth'.format(epoch)
            save_model(os.path.join(opt.save_dir, model_name),
                       model, optimizer, epoch, log_dict_train['loss'])
        else:
            model_name = 'model_last.pth'
            save_model(os.path.join(opt.save_dir, model_name),
                       model, optimizer, epoch, log_dict_train['loss'])  # save the last one

        # 使用loss来判断最好模型
        if best_loss > log_dict_train['loss']:
            best_loss = log_dict_train['loss']
            best_epoch = epoch

        # if opt.auto_stop:
        #     if opt.rgb_model != '':
        #         opt.rgb_model = os.path.join(opt.save_dir, model_name)
        #     normal_inference(opt)
        #     ap = frameAP(opt, print_info=opt.print_log, epoch=epoch)
        #     os.system("rm -rf tmp")
        #     if ap > best_ap:
        #         best_ap = ap
        #         best_epoch = epoch
        #         saved1 = os.path.join(opt.save_dir, model_name)
        #         saved2 = os.path.join(opt.save_dir, 'model_best.pth')
        #         os.system("cp " + str(saved1) + " " + str(saved2))

        # 修改学习率
        if epoch in opt.lr_step:
            lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
            logger.write('Drop LR to ' + str(lr) + '\n')
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        timer.toc()
        print('epoch is {}, use time: {}'.format(epoch, timer.average_time))

    if opt.auto_stop:
        print('best epoch is ', best_epoch)

    logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
