import os
import numpy as np
from progress.bar import Bar
import torch
import pickle
import sys

sys.path.append('../')
from opts import opts
from datasets.init_dataset import switch_dataset
from detector.normal_bezier_det import BezierDetector
import random
import cv2
import math

from ACT import frameAP
from MOC_utils.gaussian_hm import gaussian_radius, draw_umich_gaussian
from datasets.sample.sampler import findIndex

import copy

GLOBAL_SEED = 317


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def worker_init_fn(dump):
    set_seed(GLOBAL_SEED)


class PrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, opt, dataset, pre_process_func):
        self.pre_process_func = pre_process_func
        self.opt = opt
        self.gt_object = dataset._gt_object
        self.nframes = dataset._nframes
        self.tubletImgFiles = dataset.tubletImgFiles
        self.resolution = dataset._resolution
        self._gt_object = dataset._gt_object
        self.mot_augment = dataset.mot_augment

        self.input_h = dataset.height
        self.input_w = dataset.width
        self.output_h = self.input_h // self.opt.down_ratio
        self.output_w = self.input_w // self.opt.down_ratio

        self.indices = dataset._indices
        self.mot_root = dataset.mot_root

        self.max_objs = opt.max_objs

    def __getitem__(self, index):

        vname, frame_id = self.indices[index]

        meta_info = open(os.path.join(self.mot_root, vname, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])

        input_h = self.input_h
        input_w = self.input_w
        output_h = self.output_h
        output_w = self.output_w

        vdir = self.mot_root
        img_file = os.path.join(vdir, vname, 'img1/{:0>6}.jpg'.format(int(frame_id)))
        # images = [cv2.imread(i).astype(np.float32) for i in img_files]
        images = cv2.imread(img_file)
        # img_path = img_files[3]
        data = np.empty((3, input_h, input_w), dtype=np.float32)
        h, w = self.resolution[vname]
        tube_info = self._gt_object[vname].copy()

        # gt_sparse_tublet = {}
        gt_mask = {}

        """x1 y1 x2 y2"""
        object = copy.deepcopy(tube_info['gt_object'][vname][frame_id])

        '''
            {tube_id:
                    {
                       tube:{array([[frame_id,x1,y1,x2,y2]...])}
                       curves_forward_7_stride_10:{
                                start_frame_id:{
                                            curve:[x1,y1,t1...x4,y4,t4],
                                            mask:[0 1 1 1 1 1 0]
                                }
                                ...
                        }
                    }
           }
        '''

        if frame_id == 169 and vname == 'MOT17-13-FRCNN': # 13
            print(169)

        images, object = self.mot_augment(images, object, only_letter_box=True)

        if isinstance(object[0], list):
            pass

        original_h, original_w = input_h, input_w

        data = np.transpose(images, (2, 0, 1))  # (c,h,w)
        #     data[i] = ((data[i] / 255.) - mean) / std
        data = data / 255.
        # data[i] = images[i]

        # draw ground truth
        hm = np.zeros((output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        # bezier_curves = np.zeros((self.max_objs, 18), dtype=np.float32)
        ID = np.zeros((self.max_objs,), dtype=np.int64)

        index = np.zeros((self.max_objs), dtype=np.int64)
        # index_all = np.zeros((self.max_objs, forward_frames * 2), dtype=np.int64)
        mask = np.zeros((self.max_objs), dtype=np.uint8)

        num_objs = 0

        for ind in range(len(object)):
            try:
                object[ind][0] = object[ind][0] / original_w * output_w
            except (IndexError):
                print(object)
                print(frame_id) # 169

            object[ind][1] = object[ind][1] / original_h * output_h
            object[ind][2] = object[ind][2] / original_w * output_w
            object[ind][3] = object[ind][3] / original_h * output_h

            obj_h, obj_w = object[ind][3] - object[ind][1], \
                           object[ind][2] - object[ind][0]

            # create gaussian heatmap
            radius = gaussian_radius((math.ceil(obj_h), math.ceil(obj_w)))
            radius = max(0, int(radius))

            # ground truth bbox's center in key frame
            center = np.array([(object[ind][0] + object[ind][2]) / 2,
                               (object[ind][1] + object[ind][3]) / 2],
                              dtype=np.float32)
            center_int = center.astype(np.int32)

            center_int[0] = np.clip(center_int[0], 0, output_w - 1)
            center_int[1] = np.clip(center_int[1], 0, output_h - 1)
            assert 0 <= center_int[0] <= output_w and 0 <= center_int[1] <= output_h
            # draw ground truth gaussian heatmap at each center location
            draw_umich_gaussian(hm, center_int, radius)

            wh[ind, 0:2] = int(object[ind][2] - object[ind][0]), \
                           int(object[ind][3] - object[ind][1])

            # v1 Fit bezier just using the sparse tublet center
            index[num_objs] = center_int[1] * output_w + center_int[0]

            # mask indicate how many objects in this tube
            mask[num_objs] = 1
            # ID[num_objs] = int(tube_id)
            num_objs = num_objs + 1

        if self.opt.rgb_model != '':
            images = self.pre_process_func(images)

        outfile = self.outfile(vname, frame_id)
        if not os.path.isdir(os.path.dirname(outfile)):
            os.system("mkdir -p '" + os.path.dirname(outfile) + "'")

        return {'outfile': outfile,
                'images': images,
                'input': data, 'hm': hm, 'wh': wh,
                'id': ID,
                'index': index,
                'mask': mask,
                'meta': {'height': h, 'width': w,
                         'input_h': self.input_h, 'input_w': self.input_w,
                         'output_h': self.output_h, 'output_w': self.output_w},
                }

    def outfile(self, v, i):
        return os.path.join(self.opt.inference_dir, v, "{:0>6}.pkl".format(i))

    def __len__(self):
        return len(self.indices)


def normal_inference(opt, mode='train'):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    torch.backends.cudnn.benchmark = True

    Dataset = switch_dataset[opt.dataset]
    opt = opts().update_dataset(opt, Dataset)

    detector = BezierDetector(opt)

    if mode == 'train':
        ##########################################################################
        train_dataset = Dataset(opt, 'train')
        train_prefetch_dataset = PrefetchDataset(opt, train_dataset, detector.pre_process)
        train_data_loader = torch.utils.data.DataLoader(
            train_prefetch_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=opt.pin_memory,
            drop_last=False,
            worker_init_fn=worker_init_fn)

        num_iters = len(train_data_loader)
        bar = Bar(opt.exp_id, max=num_iters)

        print()
        print('inference batch_size:', opt.batch_size)

        for iter, data in enumerate(train_data_loader):
            outfile = data['outfile']
            print(outfile)
            detections = detector.run(data)
            opt.redo_detection = True
            for i in range(len(outfile)):
                if opt.redo_detection:
                    with open(outfile[i], 'wb') as file:
                        pickle.dump(detections[i], file)
                    continue
                if os.path.exists(outfile[i]):
                    continue
                else:
                    with open(outfile[i], 'wb') as file:
                        pickle.dump(detections[i], file)

            Bar.suffix = 'inference: [{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                iter, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()
        bar.finish()
    ##########################################################################
    elif mode == 'val':
        val_dataset = Dataset(opt, 'val')
        val_prefetch_dataset = PrefetchDataset(opt, val_dataset, detector.pre_process)
        val_data_loader = torch.utils.data.DataLoader(
            val_prefetch_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=opt.pin_memory,
            drop_last=False,
            worker_init_fn=worker_init_fn)

        num_iters = len(val_data_loader)
        bar = Bar(opt.exp_id, max=num_iters)
        print()
        print('inference batch_size:', opt.batch_size)

        for iter, data in enumerate(val_data_loader):
            outfile = data['outfile']
            print(outfile)
            detections = detector.run(data)
            opt.redo_detection = True
            for i in range(len(outfile)):
                if opt.redo_detection:
                    with open(outfile[i], 'wb') as file:
                        pickle.dump(detections[i], file)
                    continue
                if os.path.exists(outfile[i]):
                    continue
                else:
                    with open(outfile[i], 'wb') as file:
                        pickle.dump(detections[i], file)

            Bar.suffix = 'inference: [{0}/{1}]|Tot: {total:} |ET {eta:} '.format(
                iter, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()
        bar.finish()


if __name__ == '__main__':
    opt = opts().parse()
    opt.rgb_model = os.path.join(opt.save_dir, 'model_39.pth')
    opt.batch_size = 1
    opt.num_workers = 0

    normal_inference(opt)
    frameAP(opt, mode='train')
