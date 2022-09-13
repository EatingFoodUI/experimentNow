import argparse
import os


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # basic experiment settings
        self.parser.add_argument('--task', default='MOT_train',
                                 help='current task')
        self.parser.add_argument('--exp_id', default='')
        # self.parser.add_argument('--load_model', default='../experiment/result_model/model_best.pth',
        #                          help='path to load model')
        self.parser.add_argument('--load_model', default='',
                                 help='path to load model')
        # self.parser.add_argument('--save_model', default='../experiment/result_model',
        #                          help='path to save model')
        self.parser.add_argument('--model_name', default='Train_f7s7_coco',
                                 help='current model name')
        self.parser.add_argument('--rgb_model', default='../../experiment/result_model/model_20.pth',
                                 help='path to rgb model')
        self.parser.add_argument('--seed', type=int, default=317,
                                 help='random seed')

        # model setting
        self.parser.add_argument('--arch', default='dla_34',
                                 help='model architecture. Currently tested'
                                      'resnet_18 | resnet_101 | dla_34')
        self.parser.add_argument('--head_conv', type=int, default=256,
                                 help='conv layer channels for output head'
                                      'default setting is 256 ')
        self.parser.add_argument('--down_ratio', type=int, default=4,
                                 help='output stride. Currently only supports 4.')
        self.parser.add_argument('--forward_frames', type=int, default=7,
                                 help='length of sparse track tublet')
        self.parser.add_argument('--clip_len', default=0.5, type=float, help="sparse clip time interval")
        # self.parser.add_argument('--warp_iterate', default=False, type=bool, help="align method")

        # system settings
        self.parser.add_argument('--gpus', default='0',
                                 help='visible gpu list, use comma for multiple gpus')
        self.parser.add_argument('--num_workers', type=int, default=0,
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--batch_size', type=int, default=1,
                                 help='batch size')
        self.parser.add_argument('--master_batch_size', type=int, default=-1,
                                 help='batch size on the master gpu.')

        # learning rate settings
        self.parser.add_argument('--lr', type=float, default=1e-5,
                                 help='learning rate for batch size 32.')
        self.parser.add_argument('--lr_step', type=str, default='8,16',
                                 help='drop learning rate by 10.')
        self.parser.add_argument('--num_epochs', type=int, default=30,
                                 help='total training epochs.')

        # dataset settings
        self.parser.add_argument('--dataset', default='mot17',
                                 help='mot16 | mot17 | mot20')
        self.parser.add_argument('--input_h', type=int, default=608,
                                 help='input image height 512')
        self.parser.add_argument('--input_w', type=int, default=1088,
                                 help='input image width 288')

        # training settings
        self.parser.add_argument('--pretrain_model', default='fairmot',
                                 help='training pretrain_model, coco|imagenet|fairmot')
        self.parser.add_argument('--auto_stop', default=True, action='store_true',
                                 help='auto_stop when training')
        self.parser.add_argument('--save_all', default=True, action='store_true',
                                 help='save each epoch training model')
        self.parser.add_argument('--val_epoch', default=True, action='store_true',
                                 help='val after each epoch')
        self.parser.add_argument('--visual_per_inter', type=int, default=100,
                                 help='iter for draw loss by tensorboardX')
        self.parser.add_argument('--start_epoch', type=int, default=0,
                                 help='strat epoch, used for recover experiment')
        self.parser.add_argument('--pin_memory', default=True, action='store_true',
                                 help='set pin_memory True')

        # augmentation
        self.parser.add_argument('--augment', default=True, action='store_true',
                                 help='data augmentation during training')
        self.parser.add_argument('--flip_aug', default=True, action='store_true',
                                 help='data flip_aug during training')
        self.parser.add_argument('--hsv_aug', default=True, action='store_true',
                                 help='data hsv_aug during training')
        self.parser.add_argument('--affine_aug', default=True, action='store_true',
                                 help='data affine_aug during training')

        # loss ratio settings
        self.parser.add_argument('--hm_weight', type=float, default=1,
                                 help='loss weight for center heatmaps.')
        self.parser.add_argument('--mov_weight', type=float, default=1,
                                 help='loss weight for moving offsets.')
        self.parser.add_argument('--wh_weight', type=float, default=1,
                                 help='loss weight for bbox regression.')
        self.parser.add_argument('--id_weight', type=float, default=1,
                                 help='loss weight for id')
        self.parser.add_argument('--reid_dim', type=int, default=128,
                                 help='feature dim for reid')
        self.parser.add_argument('--ID_head', type=bool, default=False,
                                 help='Add id head or not')

        # inference settings
        self.parser.add_argument('--redo_detection', action='store_true',
                                 help='redo for count APs')
        self.parser.add_argument('--max_objs', type=int, default=256,
                                 help='max number of output objects.')
        self.parser.add_argument('--inference_dir', default='tmp',
                                 help='directory for inferencing')
        self.parser.add_argument('--threshold', type=float, default=0.5,
                                 help='threshold for ACT.py')
        # log
        self.parser.add_argument('--print_log', default=True, action='store_true',
                                 help='print log info')

        # results visualize setting
        self.parser.add_argument('--vis_det', type=bool, default=False,
                                 help='print log info')
        self.parser.add_argument('--area_thre', type=float, default=0, help='bbox area threshold')
        self.parser.add_argument('--det_thres', type=float, default=0.5, help='bbox score threshold')

        # tracking
        self.parser.add_argument('--test_mot17', default=True, help='test mot20')
        self.parser.add_argument('--val_mot20', default=False, help='val mot20')
        self.parser.add_argument('--test_mot20', default=False, help='test mot20')

        self.parser.add_argument('--sliding_stride', type=int, default=2,
                                 help='sliding stride')
        self.parser.add_argument('--overlap_dist', type=float, default=0.8, help='confidence thresh for tracking')
        self.parser.add_argument('--min-box-area', type=float, default=400, help='filter out tiny boxes')

        # byteTrack
        self.parser.add_argument('--high_detection_score', type=float, default=0.6,
                                 help='high detection score threshold')
        self.parser.add_argument('--low_detection_score', type=float, default=0.1,
                                 help='low detection score threshold')

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]
        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]

        opt.mean = [0.40789654, 0.44719302, 0.47026115]
        opt.std = [0.28863828, 0.27408164, 0.27809835]
        if opt.master_batch_size == -1:
            opt.master_batch_size = opt.batch_size // len(opt.gpus)
        rest_batch_size = (opt.batch_size - opt.master_batch_size)
        opt.chunk_sizes = [opt.master_batch_size]
        for i in range(len(opt.gpus) - 1):
            slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
            if i < rest_batch_size % (len(opt.gpus) - 1):
                slave_chunk_size += 1
            opt.chunk_sizes.append(slave_chunk_size)

        opt.exp_id = '{}_{}_f{}cl{}'.format(opt.arch, opt.dataset, opt.forward_frames, opt.clip_len)
        opt.root_dir = os.path.join(os.path.dirname(__file__), '..')
        opt.exp_dir = os.path.join(opt.root_dir, 'experiment', 'result_model')
        opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
        opt.log_dir = opt.save_dir + '/logs_tensorboardX'

        # opt.rgb_model = os.path.join(opt.save_dir, 'model_13_valhalf.pth')
        opt.rgb_model = os.path.join(opt.save_dir, 'model_30.pth')
        opt.load_model = ""
        # print(opt.root_dir) # /home/vdai/peng/code/MOT/BezierTK/src/..
        # print(opt.exp_dir)
        # print(opt.save_dir)
        # print(opt.log_dir)
        return opt

    def update_dataset(self, opt, dataset):
        opt.nID = dataset.nID
        opt.num_classes = dataset.num_classes
        opt.branch_info = {'hm': opt.num_classes,  # one for all / one for one
                           'bezier_ctp': 18+1,
                           'wh': 2 * opt.forward_frames,
                           'id': opt.reid_dim}
        return opt
