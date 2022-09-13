import torch.utils.data as data
import math
import random
import numpy as np
import cv2
from ACT_utils.ACT_aug import apply_distort, apply_expand, crop_image
from ACT_utils.ACT_utils import curveFromTube
from MOC_utils.gaussian_hm import gaussian_radius, draw_umich_gaussian
import time
import os
import torch
from MOC_utils.utils import warp_bbox_iterate, warp_bbox_direct_v2


def findIndex(tube_frame_id_in_clip, frame_ids_in_tube):
    frame_ids_in_tube = frame_ids_in_tube.tolist()
    l = []
    for frame_id in tube_frame_id_in_clip:
        l.append(frame_ids_in_tube.index(frame_id))
    return np.array(l)


class Sampler(data.Dataset):
    def __getitem__(self, id):
        vname, start_frame_id = self._indices[id]

        num_classes = self.num_classes
        meta_info = open(os.path.join(self.mot_root, vname, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        self.frame_stride = round((self.clip_len * frame_rate - 1) / (self.forward_frames - 1))
        # print("clip_len:{},frame_rate:{},forward_frames:{}".format(self.clip_len,frame_rate,self.forward_frames))

        sampling_mode = 'forward_{}_stride_{}'.format(self.forward_frames, self.frame_stride)
        self.curve_type = 'curves_' + sampling_mode
        forward_frames, frame_stride = self.forward_frames, self.frame_stride

        # print("frame_stride:{}|forward_frames:{}|frame_rate:{}".format(self.frame_stride, forward_frames, frame_rate))

        input_h = self.height
        input_w = self.width
        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio

        max_index = start_frame_id + (forward_frames - 1) * frame_stride
        center_frame_id = (max_index + start_frame_id) // 2

        frame_ids = list(range(start_frame_id, max_index + 1, frame_stride))
        vdir = self.mot_root
        img_files = self.tubletImgFiles(vdir, vname, frame_ids)
        # images = [cv2.imread(i).astype(np.float32) for i in img_files]
        images = [cv2.imread(i) for i in img_files]
        # img_path = img_files[3]
        data = [np.empty((3, input_h, input_w), dtype=np.float32) for i in
                range(forward_frames)]
        h, w = self._resolution[vname]
        tube_info = self._gt_curves[vname]

        gt_sparse_tublet = {}
        gt_mask = {}

        # if vname in self.MC_video:
        #     if self.warp_iterate:
        #         warpmats = self.iterate_warp_mat[vname]
        #         warp_func = warp_bbox_iterate
        #     else:
        #         warpmats = self.direct_warp_mat[vname]
        #         warp_func = warp_bbox_direct_v2
        #     cen_idx = forward_frames // 2
        #     for i in range(len(images)):
        #         if i == cen_idx:
        #             continue
        #         cur_frame_id = center_frame_id - (cen_idx - i) * frame_stride
        #         _, images[i] = warp_func(warpmats, center_frame_id, cur_frame_id, np.array([0, 0, 0, 0]),
        #                                  images[i])

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
        for tube_id in tube_info[vname].keys():
            # 当前frame id所生成的小视频段
            tube = tube_info[vname][tube_id]['tube']
            # 小视频段的所有frame id
            frame_ids_in_tube = tube[:, 0]

            #print(tube_info.keys())
            #print(tube_info[vname].keys())
            #print(tube_info[vname][tube_id].keys())

            # 轨迹类型
            curves = tube_info[vname][tube_id][self.curve_type]
            # print(curves)
            if start_frame_id not in curves:
                continue

            curve = curves[start_frame_id]['curve']
            mask = curves[start_frame_id]['mask']
            frame_ids_clip = np.arange(start_frame_id, max_index + 1, frame_stride)
            c = mask * frame_ids_clip
            tube_frame_id_in_clip = c.ravel()[np.flatnonzero(c)]  # 中间帧和其他帧
            true_ids = findIndex(tube_frame_id_in_clip, frame_ids_in_tube)  # 在tube中的index
            # sparse_tublet = tube[true_ids][:, 1:]  # 我还是想保留index
            sparse_tublet = tube[true_ids]  # 我还是想保留index

            # if vname in self.MC_video:
            #     for gt_bbox in sparse_tublet:
            #         cur_frame_id = gt_bbox[0]
            #         if cur_frame_id == center_frame_id:
            #             continue
            #
            #         gt_bbox[1:] = warp_func(warpmats, center_frame_id, cur_frame_id, gt_bbox[1:])

            if tube_id not in gt_sparse_tublet:
                gt_sparse_tublet[tube_id] = sparse_tublet
                gt_mask[tube_id] = mask

        '''
        for tube_id in gt_sparse_tublet:
            tublet_frame_ids = gt_sparse_tublet[tube_id][:, 0]
            s = gt_sparse_tublet[tube_id][0, 0]
            boxes = gt_sparse_tublet[tube_id][:, 1:]
            if vname == 'MOT17-05-FRCNN':
                for ii, fid in enumerate(tublet_frame_ids):
                    fid = int(fid)
                    jj = (fid - s) // frame_stride
                    # img = images[jj][:, :, ::-1]
                    img = images[jj]
                    box = boxes[jj].astype(np.int32)

                    box[0] = np.clip(box[0], 0, w - 1)
                    box[1] = np.clip(box[1], 0, h - 1)
                    box[2] = np.clip(box[2], 0, w - 1)
                    box[3] = np.clip(box[3], 0, h - 1)

                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                    print(os.path.join(vdir, vname, 'img2/{:0>6}.jpg'.format(fid)))
                    cv2.imwrite(os.path.join(vdir, vname, 'img2/{:0>6}.jpg'.format(fid)), img)
        '''
        if self.mode == 'train':
            images, gt_sparse_tublet = self.mot_augment(images, gt_sparse_tublet)
        else:
            images, gt_sparse_tublet = self.mot_augment(images, gt_sparse_tublet, only_letter_box=True)

        original_h, original_w = input_h, input_w
        # for tube_id in gt_sparse_tublet:
        #     tublet_frame_ids = gt_sparse_tublet[tube_id][:, 0]
        #     s = gt_sparse_tublet[tube_id][0, 0]
        #     boxes = gt_sparse_tublet[tube_id][:, 1:]
        #     if vname == 'MOT17-11-FRCNN':
        #         for ii, fid in enumerate(tublet_frame_ids):
        #             fid = int(fid)
        #             jj = (fid - s) // frame_stride
        #             img = images[jj]
        #             box = boxes[jj].astype(np.int32)
        #             box[0] = np.clip(box[0], 0, original_w - 1)
        #             box[1] = np.clip(box[1], 0, original_h - 1)
        #             box[2] = np.clip(box[2], 0, original_w - 1)
        #             box[3] = np.clip(box[3], 0, original_h - 1)
        #             cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
        #             print(os.path.join(vdir, vname, 'img2/{:0>6}.jpg'.format(fid)))
        #             cv2.imwrite(os.path.join(vdir, vname, 'img2/{:0>6}.jpg'.format(fid)), img)
        for tube_id in gt_sparse_tublet:
            gt_sparse_tublet[tube_id][:, 1] = gt_sparse_tublet[tube_id][:, 1] / original_w * output_w
            gt_sparse_tublet[tube_id][:, 2] = gt_sparse_tublet[tube_id][:, 2] / original_h * output_h
            gt_sparse_tublet[tube_id][:, 3] = gt_sparse_tublet[tube_id][:, 3] / original_w * output_w
            gt_sparse_tublet[tube_id][:, 4] = gt_sparse_tublet[tube_id][:, 4] / original_h * output_h

        # transpose image channel and normalize
        # mean = np.tile(np.array(self.opt.mean, dtype=np.float32)[:, None, None], (1, 1, 1))
        # std = np.tile(np.array(self.opt.std, dtype=np.float32)[:, None, None], (1, 1, 1))
        for i in range(forward_frames):
            data[i] = np.transpose(images[i], (2, 0, 1))  # (c,h,w)
            #     data[i] = ((data[i] / 255.) - mean) / std
            data[i] = data[i] / 255.
            # data[i] = images[i]

        # draw ground truth
        hm = np.zeros((output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, forward_frames * 2), dtype=np.float32)
        bezier_curves = np.zeros((self.max_objs, 12), dtype=np.float32)
        ID = np.zeros((self.max_objs,), dtype=np.int64)

        index = np.zeros((self.max_objs), dtype=np.int64)
        index_all = np.zeros((self.max_objs, forward_frames * 2), dtype=np.int64)
        mask = np.zeros((self.max_objs), dtype=np.uint8)

        num_objs = 0
        frame_ids_clip = np.arange(start_frame_id, max_index + 1, frame_stride)

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

            for j in range(forward_frames):
                cur_frame_id = start_frame_id + j * frame_stride
                if cur_frame_id not in tube_frame_id_in_clip:
                    index_all[num_objs, 2 * j: 2 * j + 2] = -1, -1
                    wh[num_objs, 2 * j: 2 * j + 2] = 0, 0
                else:
                    cur_true_idx = tube_frame_id_in_clip.tolist().index(cur_frame_id)
                    center_all = np.array(
                        [(gt_sparse_tublet[tube_id][cur_true_idx, 1] + gt_sparse_tublet[tube_id][
                            cur_true_idx, 3]) / 2,
                         (gt_sparse_tublet[tube_id][cur_true_idx, 2] + gt_sparse_tublet[tube_id][
                             cur_true_idx, 4]) / 2],
                        dtype=np.float32)
                    center_all_int = center_all.astype(np.int32)
                    center_all_int[0] = np.clip(center_all_int[0], 0, output_w - 1)
                    center_all_int[1] = np.clip(center_all_int[1], 0, output_h - 1)
                    wh[num_objs, 2 * j: 2 * j + 2] = gt_sparse_tublet[tube_id][cur_true_idx, 3] - \
                                                     gt_sparse_tublet[tube_id][cur_true_idx, 1], \
                                                     gt_sparse_tublet[tube_id][cur_true_idx, 4] - \
                                                     gt_sparse_tublet[tube_id][cur_true_idx, 2]
                    index_all[num_objs, 2 * j: 2 * j + 2] = center_all_int[1] * output_w + center_all_int[0], \
                                                            center_all_int[1] * output_w + center_all_int[0]

            # v1 Fit bezier just using the sparse tublet center
            index[num_objs] = center_int[1] * output_w + center_int[0]

            center_true_idx = gt_sparse_tublet[tube_id][:, 0].tolist().index(center_frame_id)
            meta = (w, h, frame_stride)
            ctp, relative_ctp = curveFromTube(gt_sparse_tublet[tube_id], center_true_idx, meta)
            bezier_curves[num_objs] = relative_ctp.flatten()

            # mask indicate how many objects in this tube
            mask[num_objs] = 1
            ID[num_objs] = int(tube_id)
            num_objs = num_objs + 1
        gt = {'input': data, 'hm': hm, 'bezier_ctp': bezier_curves, 'wh': wh,
              'id': ID,
              'index': index, 'index_all': index_all,
              'mask': mask}
        
        # print(hm) 
        return gt
