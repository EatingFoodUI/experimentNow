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
import copy


def findIndex(tube_frame_id_in_clip, frame_ids_in_tube):
    frame_ids_in_tube = frame_ids_in_tube.tolist()
    l = []
    for frame_id in tube_frame_id_in_clip:
        l.append(frame_ids_in_tube.index(frame_id))
    return np.array(l)


class Sampler(data.Dataset):
    def __getitem__(self, id):
        vname, frame_id = self._indices[id]

        num_classes = self.num_classes
        meta_info = open(os.path.join(self.mot_root, vname, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])

        input_h = self.height
        input_w = self.width
        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio

        vdir = self.mot_root
        img_file = os.path.join(vdir, vname, 'img1/{:0>6}.jpg'.format(int(frame_id)))
        # images = [cv2.imread(i).astype(np.float32) for i in img_files]
        images = cv2.imread(img_file)
        # img_path = img_files[3]
        data = np.empty((3, input_h, input_w), dtype=np.float32)
        h, w = self._resolution[vname]
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

        if self.mode == 'train':
            images, object = self.mot_augment(images, object)
        else:
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
                object[ind][0][0] = object[ind][0][0] / original_w * output_w
            except (IndexError):
                print(object)
                print(frame_id) # 169

            object[ind][0][1] = object[ind][0][1] / original_h * output_h
            object[ind][0][2] = object[ind][0][2] / original_w * output_w
            object[ind][0][3] = object[ind][0][3] / original_h * output_h

            obj_h, obj_w = object[ind][0][3] - object[ind][0][1], \
                           object[ind][0][2] - object[ind][0][0]

            # create gaussian heatmap
            radius = gaussian_radius((math.ceil(obj_h), math.ceil(obj_w)))
            radius = max(0, int(radius))

            # ground truth bbox's center in key frame
            center = np.array([(object[ind][0][0] + object[ind][0][2]) / 2,
                               (object[ind][0][1] + object[ind][0][3]) / 2],
                              dtype=np.float32)
            center_int = center.astype(np.int32)

            center_int[0] = np.clip(center_int[0], 0, output_w - 1)
            center_int[1] = np.clip(center_int[1], 0, output_h - 1)
            assert 0 <= center_int[0] <= output_w and 0 <= center_int[1] <= output_h
            # draw ground truth gaussian heatmap at each center location
            draw_umich_gaussian(hm, center_int, radius)

            wh[ind, 0:2] = int(object[ind][0][2] - object[ind][0][0]), \
                           int(object[ind][0][3] - object[ind][0][1])

            # v1 Fit bezier just using the sparse tublet center
            index[num_objs] = center_int[1] * output_w + center_int[0]

            # mask indicate how many objects in this tube
            mask[num_objs] = 1
            # ID[num_objs] = int(tube_id)
            num_objs = num_objs + 1

        gt = {'input': data, 'hm': hm, 'wh': wh,
              'id': ID,
              'index': index,
              'mask': mask}

        return gt
