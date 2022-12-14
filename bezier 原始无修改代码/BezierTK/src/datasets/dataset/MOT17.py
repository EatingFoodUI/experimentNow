import os
from .base_dataset import BaseDataset
import sys
import cv2
import numpy as np

sys.path.append('../')
import math
from MOC_utils.utils import savePickle, readPickle, letterbox
from MOC_utils.gaussian_hm import gaussian_radius, draw_umich_gaussian


class MOT17(BaseDataset):
    num_classes = 1
    nID = 546

    def __init__(self, opt, mode, det='FRCNN'):
        ROOT_DATASET_PATH = os.path.join(opt.root_dir, 'data')
        super(MOT17, self).__init__(opt)

        if mode == 'test':
            self.mot_root = os.path.join(ROOT_DATASET_PATH, mode)
        else:
            self.mot_root = os.path.join(ROOT_DATASET_PATH, 'train')

        '''
        output:
        videos:['MOT17-02-FRCNN']
        nframes:{'MOT17-02-FRCNN':int}
        resolution:{'MOT17-02-FRCNN':(h,w)}
        gt_curves:{
                    'MOT17-02-FRCNN':
                            {id:
                                {tube:array([[frame_id,x1,y1,x2,y2]...])}
                                {
                                        curves_forward_7_stride_10:{
                                            frame_id:{
                                                        curve:[x1,y1,t1...x4,y4,t4],
                                                        mask:[0 1 1 1 1 1 0]
                                                    }
                                        }
                                }
                           }
                    }
        '''
        # curve_pkl_file = os.path.join(self.mot_root, det + '_tracks.pkl')
        # curve_pkl = readPickle(curve_pkl_file)
        # for k in curve_pkl:
        #     setattr(self, ('_' if k != 'labels' else '') + k, curve_pkl[k])
        #
        self._videos = []

        try:
            curve_pkl_file = os.path.join(self.mot_root, det + '_tracks.pkl')
            curve_pkl = readPickle(curve_pkl_file)
            for k in curve_pkl:
                setattr(self, ('_' if k != 'labels' else '') + k, curve_pkl[k])
        except FileNotFoundError:
            pass

        self.train_ratio = 1.0
        self.mode = mode
        self.forward_frames, self.clip_len = opt.forward_frames, opt.clip_len
        # ????????????
        # self._videos = ['MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-09-FRCNN',
        #                 'MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05', ]
        # self._videos = ['MOT20-03']
        # ['MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-09-FRCNN', 'MOT17-13-FRCNN']
        # self._videos = ['MOT17-02-FRCNN']

        self._indices = []
        # training or validation
        for v in self._videos:
            meta_info = open(os.path.join(self.mot_root, v, 'seqinfo.ini')).read()
            frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
            self.frame_stride = round((self.clip_len * frame_rate - 1) / (self.forward_frames - 1))
            nframes = self._nframes[v]
            if self.mode == 'train':
                MAX_IDX = int(nframes * self.train_ratio)
                _range = range(1, int(nframes * self.train_ratio))

                # check how the det goes on train dataset
                # del this line before training
                # _range = range(1, MAX_IDX,
                #                (self.forward_frames - 1) * self.frame_stride + 1)
            elif self.mode == 'val':
                MAX_IDX = nframes + 1
                # no overlap between clips
                _range = range(int(nframes * self.train_ratio), MAX_IDX,
                               (self.forward_frames - 1) * self.frame_stride + 1)

            for frame_id in _range:
                max_index = frame_id + (self.forward_frames - 1) * self.frame_stride
                if max_index >= MAX_IDX:
                    break
                self._indices.append((v, frame_id))

    def tubletImgFiles(self, vdir, vname, frame_ids):
        # img1 / {0: 06}.jpg
        return [os.path.join(vdir, vname, 'img1/{:0>6}.jpg'.format(int(i))) for i in frame_ids]


def pre_process(opt, images):
    forward_frames = opt.forward_frames
    images, ratio, padw, padh = letterbox(images, opt.input_h, opt.input_w)

    data = [np.empty((3, opt.input_h, opt.input_w), dtype=np.float32) for i in
            range(forward_frames)]
    for i in range(forward_frames):
        data[i] = np.transpose(images[i], (2, 0, 1))
        # data[i] = ((data[i] / 255.) - mean) / std
        data[i] = data[i] / 255.

    return data, ratio, padw, padh


class eval_MOT17_public_det(MOT17):
    def __init__(self, opt, mode='val', vname='MOT20-01'):
        super(eval_MOT17_public_det, self).__init__(opt, mode=mode)
        self.input_h = self.height
        self.input_w = self.width
        self.output_h = self.input_h // self.opt.down_ratio
        self.output_w = self.input_w // self.opt.down_ratio

        self.pre_process_func = pre_process
        self._indices = []

        v = vname
        meta_info = open(os.path.join(self.mot_root, v, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        self.frame_stride = round((self.clip_len * frame_rate - 1) / (self.forward_frames - 1))

        nframes = int(meta_info[meta_info.find('seqLength') + 10:meta_info.find('\nimWidth')])
        self.train_ratio = 0.5
        if self.mode == 'val':
            MAX_IDX = nframes + 1
            # ??????????????????
            _range = range(int(nframes * self.train_ratio), MAX_IDX, opt.sliding_stride)
        elif self.mode == 'test':
            MAX_IDX = nframes + 1
            _range = range(1, MAX_IDX, opt.sliding_stride)

        for frame_id in _range:
            max_index = frame_id + (self.forward_frames - 1) * self.frame_stride
            if max_index >= MAX_IDX:
                break

            self._indices.append((v, frame_id))

        last_frame_id = MAX_IDX - (self.forward_frames - 1) * self.frame_stride - 1
        if last_frame_id != self._indices[-1][1]:
            self._indices.append((v, last_frame_id))

        # read the public det file.txt
        if self.mode == 'val':
            mode = 'train'
            det_path = os.path.join(opt.root_dir, 'data', mode, vname, 'det/moter.txt')
        elif self.mode == 'test':
            mode = 'test'
            det_path = os.path.join(opt.root_dir, 'data', mode, vname, 'det/moter.txt')

        det_arr = np.loadtxt(det_path, dtype=np.float64, delimiter=',')
        self.det = {}
        for fid, tid, bx, by, bw, bh, mark, _, _, _ in det_arr:
            fid = int(fid)
            if fid not in self.det:
                self.det[fid] = []

            self.det[fid].append((bx, by, bw, bh))

    def __getitem__(self, index):
        vname, start_frame_id = self._indices[index]
        forward_frames, clip_len = self.opt.forward_frames, self.opt.clip_len
        meta_info = open(os.path.join(self.mot_root, vname, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        frame_stride = round((clip_len * frame_rate - 1) / (forward_frames - 1))

        max_index = start_frame_id + (forward_frames - 1) * frame_stride
        frame_ids = list(range(start_frame_id, max_index + 1, frame_stride))
        vdir = self.mot_root
        img_files = self.tubletImgFiles(vdir, vname, frame_ids)
        images0 = [cv2.imread(i) for i in img_files]
        height, width, _ = images0[0].shape

        images, ratio, padw, padh = self.pre_process_func(self.opt, images0)

        frame_ids = list(range(start_frame_id, max_index + 1))
        img_files = self.tubletImgFiles(vdir, vname, frame_ids)
        images0 = [cv2.imread(i) for i in img_files]

        cen_frame_id = (start_frame_id + max_index) // 2
        hm = np.zeros((self.output_h, self.output_w), dtype=np.float32)

        if cen_frame_id not in self.det:
            return {
                'frame_id': start_frame_id,
            }

        l = self.det[cen_frame_id]

        for (bx, by, bw, bh) in l:
            x1, y1, x2, y2 = bx, by, bx + bw, by + bh

            x1 = ratio * x1 + padw
            y1 = ratio * y1 + padh
            x2 = ratio * x2 + padw
            y2 = ratio * y2 + padh

            x1 = x1 / self.input_w * self.output_w
            y1 = y1 / self.input_h * self.output_h
            x2 = x2 / self.input_w * self.output_w
            y2 = y2 / self.input_h * self.output_h

            h, w = y2 - y1, x2 - x1
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))

            # ground truth bbox's center in key frame
            center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
            center_int = center.astype(np.int32)

            center_int[0] = np.clip(center_int[0], 0, self.output_w - 1)
            center_int[1] = np.clip(center_int[1], 0, self.output_h - 1)
            assert 0 <= center_int[0] <= self.output_w and 0 <= center_int[1] <= self.output_h

            # draw ground truth gaussian heatmap at each center location
            draw_umich_gaussian(hm, center_int, radius)

        return {
            'frame_id': start_frame_id,
            'images0': images0,
            'images': images,
            'meta': {
                'forward_frames': forward_frames, 'frame_stride': frame_stride,
                'height': height, 'width': width,
                'input_h': self.input_h, 'input_w': self.input_w,
                'output_h': self.output_h, 'output_w': self.output_w,
                'hm': hm},
            'start_frame_id': start_frame_id,
        }

    def __len__(self):
        return len(self._indices)
