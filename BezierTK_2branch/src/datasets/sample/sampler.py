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
    # id: get a sampler randomly
    def __getitem__(self, id):
        # 获取当前视频段的一个物体的小轨迹tube
        # vname: 当前视频段的名称eg:MOT17-02-FRCNN
        # start_frame_id 小视频段可以开始的帧序列号
        # _indices：datasets返回的数据结构[vname, start_frame_id]
        vname, start_frame_id = self._indices[id]

        # 没用过
        num_classes = self.num_classes
        meta_info = open(os.path.join(self.mot_root, vname, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        # 采样帧与帧之间的间隔帧
        self.frame_stride = round((self.clip_len * frame_rate - 1) / (self.forward_frames - 1))
        # print("clip_len:{},frame_rate:{},forward_frames:{}".format(self.clip_len,frame_rate,self.forward_frames))

        # 采样的小视屏段的类型：由forward_frames和frame_stride决定
        sampling_mode = 'forward_{}_stride_{}'.format(self.forward_frames, self.frame_stride)
        self.curve_type = 'curves_' + sampling_mode
        forward_frames, frame_stride = self.forward_frames, self.frame_stride

        # print("frame_stride:{}|forward_frames:{}|frame_rate:{}".format(self.frame_stride, forward_frames, frame_rate))

        # 输入图片的高宽
        input_h = self.height
        input_w = self.width
        # 输出feature map的高宽，经过了下采样
        # 这里使用的目的是让gt的feature map和训练得到的feature map大小相同
        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio

        # 当前小视频段可能的最大帧是哪一帧
        max_index = start_frame_id + (forward_frames - 1) * frame_stride
        # 当前小视频段的中心帧
        center_frame_id = (max_index + start_frame_id) // 2

        # 可能的小视频段的帧序列
        frame_ids = list(range(start_frame_id, max_index + 1, frame_stride))
        vdir = self.mot_root
        # 这个tublet视频段的所有图片文件路径
        img_files = self.tubletImgFiles(vdir, vname, frame_ids)
        # images = [cv2.imread(i).astype(np.float32) for i in img_files]
        # 所有图片文件的二进制格式
        images = [cv2.imread(i) for i in img_files]
        # img_path = img_files[3]
        # 初始化当前小视频段的每个帧的feature map
        data = [np.empty((3, input_h, input_w), dtype=np.float32) for i in
                range(forward_frames)]
        h, w = self._resolution[vname] # _resolution datasets的数据结构，负责存当前视频段的帧高宽
        # _gt_curves：由数据集类得到的视频段的曲线轨迹信息
        # 当前视频段生成的曲线轨迹信息
        tube_info = self._gt_curves[vname]

        # gt sparse：稀疏的gt tublet小轨迹段
        gt_sparse_tublet = {}
        # 0 1 负责判断小轨迹中那些帧不使用
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
        # tube_id 获取其中小轨迹的id号； tube：小轨迹； id号是每个小轨迹的编号
        for tube_id in tube_info[vname].keys():
            # 获取一个物体的小轨迹的每一个bbox info
            # 此时帧号是相对帧号0-5
            tube = tube_info[vname][tube_id]['tube']
            # 对应的相对帧号
            frame_ids_in_tube = tube[:, 0]

            #print(tube_info.keys())
            #print(tube_info[vname].keys())
            #print(tube_info[vname][tube_id].keys())

            curves = tube_info[vname][tube_id][self.curve_type] # 一个轨迹
            # print(curves)
            # 一个筛选方式
            if start_frame_id not in curves:
                continue

            # 物体有这个开始帧的曲线轨迹->找到当前物体id对应于当前帧的轨迹
            curve = curves[start_frame_id]['curve']
            mask = curves[start_frame_id]['mask'] # 那些帧要用，那些帧不用
            frame_ids_clip = np.arange(start_frame_id, max_index + 1, frame_stride) # 实际帧id，绝对帧id
            c = mask * frame_ids_clip # c 绝对帧要用的帧
            tube_frame_id_in_clip = c.ravel()[np.flatnonzero(c)]  # 删除不用的帧后的帧序列
            true_ids = findIndex(tube_frame_id_in_clip, frame_ids_in_tube)  # 在整个tube视频段中的帧index
            # sparse_tublet = tube[true_ids][:, 1:]  # 我还是想保留index
            sparse_tublet = tube[true_ids]  # 我还是想保留index 获取整个视频段上的当前物体的轨迹【帧id，bbox】

            # if vname in self.MC_video:
            #     for gt_bbox in sparse_tublet:
            #         cur_frame_id = gt_bbox[0]
            #         if cur_frame_id == center_frame_id:
            #             continue
            #
            #         gt_bbox[1:] = warp_func(warpmats, center_frame_id, cur_frame_id, gt_bbox[1:])

            # 保存这个物体在当前小视频段下的轨迹（即是否在这个小视频段下）
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
            # 数据增强
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

        # 四个物体轨迹，tube_id每个小轨迹的id
        # 目的：将真实轨迹位置转变为下采样后的feature map的轨迹位置
        for tube_id in gt_sparse_tublet:
            gt_sparse_tublet[tube_id][:, 1] = gt_sparse_tublet[tube_id][:, 1] / original_w * output_w
            gt_sparse_tublet[tube_id][:, 2] = gt_sparse_tublet[tube_id][:, 2] / original_h * output_h
            gt_sparse_tublet[tube_id][:, 3] = gt_sparse_tublet[tube_id][:, 3] / original_w * output_w
            gt_sparse_tublet[tube_id][:, 4] = gt_sparse_tublet[tube_id][:, 4] / original_h * output_h

        # transpose image channel and normalize
        # mean = np.tile(np.array(self.opt.mean, dtype=np.float32)[:, None, None], (1, 1, 1))
        # std = np.tile(np.array(self.opt.std, dtype=np.float32)[:, None, None], (1, 1, 1))
        # 将帧维度改变，并且归一化。
        for i in range(forward_frames):
            data[i] = np.transpose(images[i], (2, 0, 1))  # (c,h,w)
            #     data[i] = ((data[i] / 255.) - mean) / std
            data[i] = data[i] / 255.
            # data[i] = images[i]

        # draw ground truth, max_objs: object numbers，假定物体一共有256个
        # gt全部初始化 修改为轨迹中心帧所在位置
        hm = np.zeros((output_h, output_w), dtype=np.float32) # 152 * 272
        wh = np.zeros((self.max_objs, forward_frames * 2), dtype=np.float32) # 256 * 14
        bezier_curves = np.zeros((self.max_objs, 12), dtype=np.float32) # 256 * 18
        ID = np.zeros((self.max_objs,), dtype=np.int64)  # 256 * 1

        centerFrame_index = np.zeros((self.max_objs,), dtype=np.int64) # 256 * 1
        center_index = np.zeros((self.max_objs,), dtype=np.int64) # 256 * 1

        # 每帧对应检测物体的所在轨迹是否是同一个id，每帧的index用index_all
        # tube_id_similarity = np.ones((self.max_objs, forward_frames), dtype=np.float32) # 256 * 7
        # 修改为在单帧上显示
        tube_id_similarity = np.zeros((self.max_objs, 1), dtype=np.float32) # 256 * 1

        # 物体每个轨迹的中心点索引
        # all_center_index = np.zeros((self.max_objs, forward_frames), dtype=np.int64)
        # 修改物体中心帧的中心点索引（不同物体中心帧位置不太一样）偏移程度
        all_center_index = np.zeros((self.max_objs, 1), dtype=np.int64)

        index = np.zeros((self.max_objs), dtype=np.int64) # 256 * 1
        index_all = np.zeros((self.max_objs, forward_frames * 2), dtype=np.int64) # 256 * 14
        mask = np.zeros((self.max_objs), dtype=np.uint8) # 256 * 1

        num_objs = 0
        frame_ids_clip = np.arange(start_frame_id, max_index + 1, frame_stride) # 当前视频段的帧id

        # 处理得到 每一个小视频段的物体的gt轨迹
        for tube_id in gt_sparse_tublet:
            tube = tube_info[vname][tube_id]['tube']
            frame_ids_in_tube = tube[:, 0]
            _mask = gt_mask[tube_id]
            m = _mask * frame_ids_clip

            # 改动，设置为轨迹所在center frame
            gt_oneObject = np.array(_mask)
            nonzero_index = gt_oneObject.nonzero()
            # 取第一个1和最后一个1的位置，然后取中间为中心帧index
            OneCenterFrame_index = int((nonzero_index[0][0] + nonzero_index[0][-1]) / 2)
            # centerFrame_index = int((gt_oneObject.shape[0] / 2) - OneCenterFrame_index)

            tube_frame_id_in_clip = m.ravel()[np.flatnonzero(m)]
            true_ids = findIndex(tube_frame_id_in_clip, frame_ids_in_tube)
            # c = true_ids.tolist().index(frame_ids_in_tube.tolist().index(center_frame_id)) # ?
            c = true_ids.tolist().index(frame_ids_in_tube.tolist().index(frame_ids_clip[OneCenterFrame_index]))

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

            # 对每帧的对象进行处理？
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
            # num_objs 当前tube上有多少个物体
            mask[num_objs] = 1
            ID[num_objs] = int(tube_id)
            num_objs = num_objs + 1

        # centerFrame_index：物体在当前轨迹的中心帧所在帧 [256, 1] 通过gt_mask得到
        # center_index：物体所在中心帧的位置 [256, 1] 通过index_all得到
        for i in range(0, num_objs):
            gt_id = ID[i]
            gt_oneObject = np.array(gt_mask[gt_id])
            nonzero_index = gt_oneObject.nonzero()
            # 取第一个1和最后一个1的位置，然后取中间为中心帧index
            OneCenterFrame_index = int((nonzero_index[0][0] + nonzero_index[0][-1]) / 2)
            centerFrame_index[i] = int((gt_oneObject.shape[0] / 2) - OneCenterFrame_index)

            # 看提取错index没有
            center_index[i] = index_all[i][OneCenterFrame_index * 2]

            # 初始化tube_id_similarity [0, 1]范围
            similarity_rate = (nonzero_index[0][-1] - nonzero_index[0][0] + 1) / forward_frames
            tube_id_similarity[i] = np.full((1), similarity_rate ,dtype=np.float32)

            # 切片 取中心帧索引就行 偏移程度
            all_center_index[i] = center_index[i]

            if i+1 == num_objs:
                break

        # data：当前tube帧的输入归一化
        # hm： 帧经过hm branch的输出，只初始化了
        # bezier_curves：贝塞尔曲线控制点 [256 18] 每个物体对应
        # wh：检测框高宽 [256 14] 一共256个物体，每个物体在所有帧上的高宽。如果没有的物体为0，如果不在的维度为0
        # id：物体id，一共256个物体，值为id值，如果不在为0
        # index：物体在hm上每个通道的索引(位置)
        # index_all：物体在wh上每个通道的索引（位置）
        # mask：有哪些物体在轨迹里面，应该是和index_all对应使用的

        # centerFrame_index：物体在当前轨迹的中心帧所在帧 [256, 1] 通过wh处理
        # tube_id_similarity：物体所在轨迹是否是同一个物体，没有发生id switch [256 1] 每个特征图上这个物体当前帧轨迹的位置设置为1,是下采样之后的特征图

        # 1. 将帧的点位置定位出来，设置1
        # 2. reshape as downraito feature map, motify point loaction

        # 修改使用的轨迹，交换相距近的轨迹，修改轨迹的一致性，修改其他参数。
        # 跟着修改的参数：bezier_ctp,wh,index_all,tube_id_similarity,

        # index_all和centerFrame_index不一样
        # all_center_index 和 center_index含义一样
        # index center_index含义一样
        gt = {'input': data, 'hm': hm, 'bezier_ctp': bezier_curves, 'wh': wh,
              'id': ID,
              'index': index, 'index_all': index_all,
              'mask': mask,
              # 轨迹中心帧在哪一帧 [1,256,1] problem 学的是绝对[相对]位置
              'centerFrame_index': centerFrame_index,
              # 物体在中心帧的的哪个位置 [256,1]
              'center_index': center_index,
              # 一个轨迹同一个id比例 [256 1]
              'tube_id_similarity': tube_id_similarity
              }

        return gt
