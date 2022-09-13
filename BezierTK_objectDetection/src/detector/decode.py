import torch
import torch.nn as nn
from MOC_utils.utils import _gather_feature, _tranpose_and_gather_feature_inference
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


def decode(heatmap, wh, max_objs, forward_frames):
    # 1.heatmap
    batch, c, height, width = heatmap.size()
    # perform 'nms' on heatmaps
    heatmap = _nms(heatmap)

    # (b, max_objs)
    scores, index, xs, ys = _topk(heatmap, max_objs)

    xs_all = xs.long()
    ys_all = ys.long()

    index_all = torch.zeros((max_objs, 2)).cuda()
    index_all[:, 0] = xs_all + ys_all * width
    index_all[:, 1] = xs_all + ys_all * width
    index_all[index_all < 0] = -1
    index_all[index_all > width * height - 1] = -1
    # index_all = index_all.view(max_objs).long()

    # gather wh in each location after movement
    wh = _tranpose_and_gather_feature_inference(wh, index, index_all=index_all)
    # wh = wh.view(max_objs)
    w = wh[..., 0::2].squeeze(2)
    h = wh[..., 1::2].squeeze(2)

    # TODO
    # wh为0的情况

    scores = scores.view(max_objs, 1)

    bboxes = []
    # ?
    bboxes.extend([
        xs_all[...] - w[...] / 2,
        ys_all[...] - h[...] / 2,
        xs_all[...] + w[...] / 2,
        ys_all[...] + h[...] / 2,
    ])

    bboxes = torch.cat(bboxes, dim=0).permute(1, 0)

    detections = torch.cat([bboxes, scores], dim=1).unsqueeze(0)

    return detections
