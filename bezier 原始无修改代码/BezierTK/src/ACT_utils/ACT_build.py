import sys
import os
import pickle

import numpy as np

from progress.bar import Bar

from datasets.init_dataset import get_dataset

from .ACT_utils import nms2d, nms_tubelets, iou2d, area2d


def load_frame_detections(opt, dataset, vlist, inference_dir):
    alldets_video = {}
    alldets = []  # list of numpy array with <video_index> <frame_index> <score> <x1> <y1> <x2> <y2>
    bar = Bar('{}'.format('FrameAP'), max=len(vlist))

    forward_frames, clip_len = dataset.opt.forward_frames, dataset.opt.clip_len
    for iv, vname in enumerate(vlist):
        if vname not in alldets_video:
            alldets_video[vname] = []
        # aggregate the results for each frame # x1, y1, x2, y2, score
        meta_info = open(os.path.join(dataset.mot_root, vname, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        frame_stride = round((dataset.clip_len * frame_rate - 1) / (forward_frames - 1))

        _range = None
        MAX_IDX = None
        nframes = dataset._nframes[vname]
        if dataset.mode == 'train':
            MAX_IDX = int(nframes * dataset.train_ratio)
            # _range = range(1, int(nframes * dataset.train_ratio))
            _range = range(1, MAX_IDX, (forward_frames - 1) * frame_stride + 1)
        elif dataset.mode == 'val':
            MAX_IDX = nframes + 1
            _range = range(int(nframes * dataset.train_ratio), MAX_IDX)
            _range = range(int(nframes * dataset.train_ratio), MAX_IDX, (forward_frames - 1) * frame_stride + 1)

        vdets = {i: np.empty((0, 6), dtype=np.float32) for start_frame in _range for i in
                 range(start_frame, start_frame + (forward_frames - 1) * frame_stride + 1, 1)
                 if start_frame + (forward_frames - 1) * frame_stride + 1 <= MAX_IDX}

        for start_frame_id in _range:
            max_index = start_frame_id + (forward_frames - 1) * frame_stride
            if max_index >= MAX_IDX:
                break

            pkl = os.path.join(inference_dir, vname, "{:0>6}.pkl".format(start_frame_id))
            if not os.path.isfile(pkl):
                print("\nERROR: Missing extracted tubelets " + pkl)
                sys.exit()

            with open(pkl, 'rb') as fid:
                dets = pickle.load(fid)
            # dets:(N, 4k+1)
            actual_forward_frames = frame_stride * (forward_frames - 1) + 1
            for k in range(actual_forward_frames):
            # for k in range(forward_frames):
                for bbox_id in range(dets.shape[0]):
                    # vdets:x1,y1,x2,y2,score,id
                    # np.concatenate((vdets[start_frame_id + t],
                    #                 dets[:, np.array([4 * k, 1 + 4 * k, 2 + 4 * k, 3 + 4 * k, -1])]),
                    #                axis=0)

                    xyxys = dets[bbox_id, np.array([4 * k, 1 + 4 * k, 2 + 4 * k, 3 + 4 * k, -1])]
                    vdets[start_frame_id + k] = np.concatenate(
                        (vdets[start_frame_id + k], np.append(xyxys, [bbox_id])[np.newaxis]),
                        axis=0)

        # 1. filter invalid boxes
        # opt.det_thres = 0.4
        # opt.area_thre = 0.
        nms_thres = 0.4
        h, w = dataset._resolution[vname]
        img_box = np.array([0, 0, w - 1, h - 1])
        # 第i张图像
        # vdets : {frame_id:  K*N, 6} ---- x1,y1,x2,y2,score,id
        for i in vdets:
            # vdets[i][:, :5] = nms2d(vdets[i][:, :5], nms_thres)
            # remain_inds = vdets[i][:, 4] > opt.det_thres
            # vdets[i] = vdets[i][remain_inds]

            # TODO rm invalid bbox(bbox out of the img)
            ious = iou2d(vdets[i][:, :-2], img_box)
            remain_inds = ious > 0
            vdets[i] = vdets[i][remain_inds]

            areas = area2d(vdets[i][:, :-2])
            remain_inds = areas > opt.area_thre
            vdets[i] = vdets[i][remain_inds]

        # 插值
        # for start_frame_id in _range:
        #     max_index = start_frame_id + (forward_frames - 1) * frame_stride
        #     if max_index >= MAX_IDX:
        #         break
        #
        #     for k in range(forward_frames):
        #         if k > 0:
        #             last_key_frame = start_frame_id + (k - 1) * frame_stride
        #             cur_key_frame = start_frame_id + k * frame_stride
        #             for frame_id in range(last_key_frame + 1, cur_key_frame):
        #                 for bbox_id in range(dets.shape[0]):
        #                     # 差值
        #                     if bbox_id in vdets[last_key_frame][:, 5] and bbox_id in vdets[cur_key_frame][:, 5]:
        #                         idx1 = vdets[last_key_frame][:, 5].tolist().index(bbox_id)
        #                         idx2 = vdets[cur_key_frame][:, 5].tolist().index(bbox_id)
        #
        #                         m = (frame_id - last_key_frame) / (cur_key_frame - last_key_frame)
        #                         vdets[frame_id] = np.concatenate(
        #                             (vdets[frame_id],
        #                              np.expand_dims(
        #                                  (1 - m) * vdets[last_key_frame][idx1] + m * vdets[cur_key_frame][idx2],
        #                                  axis=0),
        #                              ),
        #                             axis=0)
        #                         # vdets[frame_id] = (1 - m) * vdets[last_key_frame] + m * vdets[cur_key_frame]
        # 第i张图像
        for i in vdets:
            num_objs = vdets[i].shape[0]
            # alldets: N,7 --------> ith_video, ith_frame, score, x1, x2, y1, y2
            alldets.append(np.concatenate((iv * np.ones((num_objs, 1), dtype=np.float32),
                                           i * np.ones((num_objs, 1), dtype=np.float32),
                                           vdets[i][:, np.array([4, 0, 1, 2, 3], dtype=np.int32)]), axis=1))
            alldets_video[vname].append(np.concatenate((iv * np.ones((num_objs, 1), dtype=np.float32),
                                                        i * np.ones((num_objs, 1), dtype=np.float32),
                                                        vdets[i][:, np.array([4, 0, 1, 2, 3], dtype=np.int32)]),
                                                       axis=1))
        Bar.suffix = '[{0}/{1}]:{2}|Tot: {total:} |ETA: {eta:} '.format(iv + 1, len(vlist), vname, total=bar.elapsed_td,
                                                                        eta=bar.eta_td)
        bar.next()
    bar.finish()
    for vname in alldets_video:
        alldets_video[vname] = np.concatenate(alldets_video[vname], axis=0)
    alldets = np.concatenate(alldets, axis=0)
    return alldets, alldets_video
