import torch
import torch.nn as nn
from MOC_utils.utils import _gather_feature, _tranpose_and_gather_feature
import numpy as np
import time


def _nms(heatmap, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heatmap, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heatmap).float()

    return heatmap * keep


def _topk(heatmap, max_objs):
    batch, c, height, width = heatmap.size()
    topk_scores, topk_index = torch.topk(heatmap.view(batch, -1), max_objs)

    topk_index = topk_index % (height * width)
    topk_ys = (topk_index // width).int().float()
    topk_xs = (topk_index % width).int().float()

    return topk_scores, topk_index, topk_xs, topk_ys


def takeClosest(num, frame_stride, forward_frames):
    collection = [i * frame_stride for i in range(-1 * (forward_frames // 2), forward_frames // 2 + 1)]
    res = min(collection, key=lambda x: abs(x - num))

    return res


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def decode(heatmap, wh, bezier_ctp, max_objs, frame_stride, forward_frames):
    # 1.heatmap
    batch, c, height, width = heatmap.size()
    # perform 'nms' on heatmaps
    heatmap = _nms(heatmap)

    # (b, max_objs)
    scores, index, xs, ys = _topk(heatmap, max_objs)

    # 2.bezier curve for center points
    # 中心帧通道
    center_frames_feature = bezier_ctp[:,-1,:,:].unsqueeze(0)
    # 每个检测结果的中心帧是哪一帧 绝对位置[0,forward_frame-1]
    center_frames_eachFrames = _tranpose_and_gather_feature(center_frames_feature, index)
    center_frames_eachFrames = center_frames_eachFrames.view(batch, max_objs, 1)

    rctp = _tranpose_and_gather_feature(bezier_ctp[:,:18,:,:], index)
    rctp = rctp.view(batch, max_objs, 18)

    # 改动
    expanded_center_xs = xs.clone().unsqueeze(2).expand(batch, max_objs, 6)
    expanded_center_ys = ys.clone().unsqueeze(2).expand(batch, max_objs, 6)
    # 添加mv
    expanded_center_ts = center_frames_eachFrames.clone().expand(batch, max_objs, 6)
    ctp = rctp.clone()
    ctp[..., 0::3] += expanded_center_xs
    ctp[..., 1::3] += expanded_center_ys
    # 添加mv 作为真正中心的修正(相对于真正中心的距离差多少)
    ctp[..., 2::3] += (expanded_center_ts - 3)
    ctp_x = ctp[..., 0::3]
    ctp_y = ctp[..., 1::3]
    ctp_t = ctp[..., 2::3]

    for bt in range(ctp.shape[0]):
        # modify for normalize
        ctp_t[bt] *= frame_stride[bt]
        for obj in range(ctp.shape[1]):
            ctp_t[bt][obj][0] = takeClosest(ctp_t[bt][obj][0], frame_stride[bt], forward_frames)
            ctp_t[bt][obj][-1] = takeClosest(ctp_t[bt][obj][-1], frame_stride[bt], forward_frames)

    # now the t_dim is normalized back
    ctp_stacked = (torch.cat((ctp_x, ctp_y, ctp_t), dim=1)).cpu().numpy()

    # here we assume all video have the same frame rate(MOT20)
    # xs_all = torch.zeros((batch, max_objs, forward_frames)).cuda()
    # ys_all = torch.zeros((batch, max_objs, forward_frames)).cuda()
    stride = frame_stride[0]
    actual_forward_frames = stride * (forward_frames - 1) + 1
    xs_all = torch.zeros((batch, max_objs, actual_forward_frames)).cuda()
    ys_all = torch.zeros((batch, max_objs, actual_forward_frames)).cuda()

    # t = k ; a = x; 下标 = 列
    func = lambda a, k: \
        a[..., 0] * (1 - k) ** 5 + \
        a[..., 1] * 5 * k * (1 - k) ** 4 + \
        a[..., 2] * 10 * k ** 2 * (1 - k) ** 3 + \
        a[..., 3] * 10 * k ** 3 * (1 - k) ** 2 + \
        a[..., 4] * 5 * k ** 4 * (1 - k) + \
        a[..., 5] * k ** 5
    N = 1000
    delta_k = 1. / N
    # (1,300,1000)
    cxyt_samples = np.array([func(ctp_stacked, i * delta_k) for i in range(1, N + 1)]).transpose((1, 2, 0))
    cxyt_samples = torch.from_numpy(cxyt_samples)
    obj_num = cxyt_samples.shape[1] // 3
    cx_samples = cxyt_samples[:, :obj_num]
    cy_samples = cxyt_samples[:, obj_num:2 * obj_num]
    ct_samples = cxyt_samples[:, 2 * obj_num:]

    for bt in range(xs_all.shape[0]):
        for obj in range(xs_all.shape[1]):
            '''
            if not (ctp_t[bt][obj][-1] - ctp_t[bt][obj][0] >= frame_stride[bt] - 1):
                collection = [i * frame_stride[bt] for i in
                              range(-1 * (forward_frames // 2), forward_frames // 2 + 1)]
                print('2 ctp gap too small')
                print('frame_stride:', frame_stride[bt])
                print(collection)
                print(ctp_t[bt][obj][0], ctp_t[bt][obj][-1])
                print('-' * 80)
                for i in range(forward_frames):
                    xs_all[bt][obj][i], ys_all[bt][obj][i] = -1, -1
                continue

            if not (ctp_t[bt][obj][0] <= 0 <= ctp_t[bt][obj][-1]):
                collection = [i for i in range(-1 * (forward_frames // 2), forward_frames // 2 + 1)]
                print('center frame not between 2 ctp')
                print('frame_stride:', frame_stride[bt])
                print(collection)
                print(ctp_t[bt][obj][0] / frame_stride[bt], ctp_t[bt][obj][-1] / frame_stride[bt])
                print('-' * 80)
                for i in range(forward_frames):
                    xs_all[bt][obj][i], ys_all[bt][obj][i] = -1, -1
                continue
            '''
            for i in range(actual_forward_frames):
                frame_idx = (i - actual_forward_frames // 2)

                if frame_idx == 0:
                    xs_all[bt][obj][i] = xs[bt][obj]
                    ys_all[bt][obj][i] = ys[bt][obj]

                elif ctp_t[bt][obj][0] < frame_idx < ctp_t[bt][obj][-1]:
                    # precise
                    idx = find_nearest(ct_samples[bt][obj], frame_idx)
                    xs_all[bt][obj][i] = cx_samples[bt][obj][idx]
                    ys_all[bt][obj][i] = cy_samples[bt][obj][idx]

                elif frame_idx == ctp_t[bt][obj][0]:
                    xs_all[bt][obj][i] = ctp_x[bt][obj][0]
                    ys_all[bt][obj][i] = ctp_y[bt][obj][0]

                elif frame_idx == ctp_t[bt][obj][-1]:
                    xs_all[bt][obj][i] = ctp_x[bt][obj][-1]
                    ys_all[bt][obj][i] = ctp_y[bt][obj][-1]

                else:
                    xs_all[bt][obj][i], ys_all[bt][obj][i] = -1, -1

    xs_all_int = xs_all[..., 0::stride].long()
    ys_all_int = ys_all[..., 0::stride].long()

    index_all = torch.zeros((batch, max_objs, forward_frames, 2)).cuda()
    index_all[:, :, :, 0] = xs_all_int + ys_all_int * width
    index_all[:, :, :, 1] = xs_all_int + ys_all_int * width
    index_all[index_all < 0] = -1
    index_all[index_all > width * height - 1] = -1
    index_all = index_all.view(batch, max_objs, forward_frames * 2).long()

    # gather wh in each location after movement
    wh = _tranpose_and_gather_feature(wh, index, index_all=index_all)
    wh = wh.view(batch, max_objs, forward_frames * 2)
    w = wh[..., 0::2]
    h = wh[..., 1::2]

    actual_wh = torch.zeros((batch, max_objs, actual_forward_frames * 2)).cuda()
    actual_w = actual_wh[..., 0::2]
    actual_h = actual_wh[..., 1::2]
    for k in range(forward_frames):
        if k > 0:
            last_key_frame = (k - 1) * stride
            cur_key_frame = k * stride
            for frame_id in range(last_key_frame, cur_key_frame+1):
                m = (frame_id - last_key_frame) / (cur_key_frame - last_key_frame)
                actual_w[..., frame_id] = (1 - m) * w[..., k-1] + m * w[..., k]
                actual_h[..., frame_id] = (1 - m) * h[..., k-1] + m * h[..., k]

    actual_wh[..., 0::2] = actual_w
    actual_wh[..., 1::2] = actual_h

    # TODO
    # wh为0的情况

    scores = scores.view(batch, max_objs, 1)

    xs_all = torch.unsqueeze(xs_all, 2)
    ys_all = torch.unsqueeze(ys_all, 2)
    bboxes = []
    for i in range(actual_forward_frames):
        bboxes.extend([
            xs_all[..., i] - actual_wh[..., 2 * i:2 * i + 1] / 2,
            ys_all[..., i] - actual_wh[..., 2 * i + 1:2 * i + 2] / 2,
            xs_all[..., i] + actual_wh[..., 2 * i:2 * i + 1] / 2,
            ys_all[..., i] + actual_wh[..., 2 * i + 1:2 * i + 2] / 2,
        ])

    bboxes = torch.cat(bboxes, dim=2)

    detections = torch.cat([bboxes, scores], dim=2)

    return detections
