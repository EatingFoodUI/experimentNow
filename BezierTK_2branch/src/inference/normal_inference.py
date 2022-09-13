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
        self.gt_curves = dataset._gt_curves
        self.nframes = dataset._nframes
        self.tubletImgFiles = dataset.tubletImgFiles
        self.resolution = dataset._resolution
        self._gt_curves = dataset._gt_curves
        self.mot_augment = dataset.mot_augment

        self.input_h = dataset.height
        self.input_w = dataset.width
        self.output_h = self.input_h // self.opt.down_ratio
        self.output_w = self.input_w // self.opt.down_ratio

        self.indices = dataset._indices
        self.mot_root = dataset.mot_root

    def __getitem__(self, index):
        vname, start_frame_id = self.indices[index]
        tube_info = self._gt_curves[vname]

        h, w = self.resolution[vname]
        forward_frames, clip_len = self.opt.forward_frames, self.opt.clip_len
        meta_info = open(os.path.join(self.mot_root, vname, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        frame_stride = round((clip_len * frame_rate - 1) / (forward_frames - 1))
        sampling_mode = 'forward_{}_stride_{}'.format(forward_frames, frame_stride)
        self.curve_type = 'curves_' + sampling_mode

        max_index = start_frame_id + (forward_frames - 1) * frame_stride
        frame_ids = list(range(start_frame_id, max_index + 1, frame_stride))
        vdir = self.mot_root
        img_files = self.tubletImgFiles(vdir, vname, frame_ids)
        images = [cv2.imread(i).astype(np.float32) for i in img_files]

        output_h = self.input_h // self.opt.down_ratio
        output_w = self.input_w // self.opt.down_ratio
        hm = np.zeros((output_h, output_w), dtype=np.float32)

        max_index = start_frame_id + (forward_frames - 1) * frame_stride
        center_frame_id = (max_index + start_frame_id) // 2
        frame_ids_clip = np.arange(start_frame_id, max_index + 1, frame_stride)

        gt_sparse_tublet = {}
        gt_mask = {}

        for tube_id in tube_info[vname].keys():
            tube = tube_info[vname][tube_id]['tube']
            frame_ids_in_tube = tube[:, 0]
            curves = tube_info[vname][tube_id][self.curve_type]
            if start_frame_id not in curves:
                continue

            mask = curves[start_frame_id]['mask']
            frame_ids_clip = np.arange(start_frame_id, max_index + 1, frame_stride)
            c = mask * frame_ids_clip
            tube_frame_id_in_clip = c.ravel()[np.flatnonzero(c)]  # 中间帧和其他帧
            true_ids = findIndex(tube_frame_id_in_clip, frame_ids_in_tube)  # 在tube中的index
            sparse_tublet = tube[true_ids]  # 我还是想保留index

            if tube_id not in gt_sparse_tublet:
                gt_sparse_tublet[tube_id] = sparse_tublet
                gt_mask[tube_id] = mask

        images, gt_sparse_tublet = self.mot_augment(images, gt_sparse_tublet, only_letter_box=True)
        original_h, original_w = self.input_h, self.input_w
        for tube_id in gt_sparse_tublet:
            gt_sparse_tublet[tube_id][:, 1] = gt_sparse_tublet[tube_id][:, 1] / original_w * output_w
            gt_sparse_tublet[tube_id][:, 2] = gt_sparse_tublet[tube_id][:, 2] / original_h * output_h
            gt_sparse_tublet[tube_id][:, 3] = gt_sparse_tublet[tube_id][:, 3] / original_w * output_w
            gt_sparse_tublet[tube_id][:, 4] = gt_sparse_tublet[tube_id][:, 4] / original_h * output_h

        for tube_id in gt_sparse_tublet:
            tube = tube_info[vname][tube_id]['tube']
            frame_ids_in_tube = tube[:, 0]
            _mask = gt_mask[tube_id]
            m = _mask * frame_ids_clip
            tube_frame_id_in_clip = m.ravel()[np.flatnonzero(m)]  # tube在clip中的部分的frame_id
            true_ids = findIndex(tube_frame_id_in_clip, frame_ids_in_tube)  # tube在clip中的部分的frame_id在tube中的index
            c = true_ids.tolist().index(frame_ids_in_tube.tolist().index(center_frame_id))

            key_h, key_w = gt_sparse_tublet[tube_id][c, 4] - gt_sparse_tublet[tube_id][c, 2], \
                           gt_sparse_tublet[tube_id][c, 3] - gt_sparse_tublet[tube_id][c, 1],

            # create gaussian heatmap
            radius = gaussian_radius((math.ceil(key_h), math.ceil(key_w)))
            radius = max(0, int(radius))

            # ground truth bbox's center in key frame
            center = np.array([(gt_sparse_tublet[tube_id][c, 1] + gt_sparse_tublet[tube_id][c, 3]) / 2,
                               (gt_sparse_tublet[tube_id][c, 2] + gt_sparse_tublet[tube_id][c, 4]) / 2],
                              dtype=np.float32)
            center_int = center.astype(np.int32)

            center_int[0] = np.clip(center_int[0], 0, output_w - 1)
            center_int[1] = np.clip(center_int[1], 0, output_h - 1)
            assert 0 <= center_int[0] <= output_w and 0 <= center_int[1] <= output_h
            # draw ground truth gaussian heatmap at each center location

            draw_umich_gaussian(hm, center_int, radius)

        if self.opt.rgb_model != '':
            images = self.pre_process_func(images)

        outfile = self.outfile(vname, start_frame_id)
        if not os.path.isdir(os.path.dirname(outfile)):
            os.system("mkdir -p '" + os.path.dirname(outfile) + "'")

        return {'outfile': outfile, 'images': images, 'forward_frames': forward_frames, 'frame_stride': frame_stride,
                'meta': {'height': h, 'width': w,
                         'input_h': self.input_h, 'input_w': self.input_w,
                         'output_h': self.output_h, 'output_w': self.output_w},
                'start_frame_id': start_frame_id,
                'hm': hm}

    def outfile(self, v, i):
        return os.path.join(self.opt.inference_dir, v, "{:0>6}.pkl".format(i))

    def __len__(self):
        return len(self.indices)


def normal_inference(opt, mode='val'):
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

            Bar.suffix = 'inference: [{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
                iter, num_iters, total=bar.elapsed_td, eta=bar.eta_td)
            bar.next()
        bar.finish()


if __name__ == '__main__':
    opt = opts().parse()
    opt.rgb_model = os.path.join(opt.save_dir, 'model_40.pth')
    opt.batch_size = 1
    opt.num_workers = 4

    normal_inference(opt)
    frameAP(opt, mode='val')
