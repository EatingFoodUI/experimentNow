import numpy as np
from scipy.special import comb as n_over_k
import cv2

def tublet_in_clip(tube, clip_frame_ids):
    return all([j in tube[:, 0] for j in clip_frame_ids])


def tublet_out_clip(tube, clip_frame_ids):
    return all([not j in tube[:, 0] for j in clip_frame_ids])


def tubelet_in_out_clip(tubes, clip_frame_ids):
    return all([tublet_in_clip(tube, clip_frame_ids) or tublet_out_clip(tube, clip_frame_ids) for tube in tubes])


def clip_has_tublet(tubes, clip_frame_ids):
    return any([tublet_in_clip(tube, clip_frame_ids) for tube in tubes])


def area2d(b):
    """Compute the areas for a set of 2D boxes"""

    return (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)


def overlap2d(b1, b2):
    """Compute the overlaps between a set of boxes b1 and one box b2"""
    if b1.ndim == 1:
        b1 = b1[None, :]
    if b2.ndim == 1:
        b2 = b2[None, :]

    xmin = np.maximum(b1[:, 0], b2[:, 0])
    ymin = np.maximum(b1[:, 1], b2[:, 1])
    xmax = np.minimum(b1[:, 2] + 1, b2[:, 2] + 1)
    ymax = np.minimum(b1[:, 3] + 1, b2[:, 3] + 1)

    width = np.maximum(0, xmax - xmin)
    height = np.maximum(0, ymax - ymin)

    return width * height


def iou2d(b1, b2):
    """Compute the IoU between a set of boxes b1 and 1 box b2"""

    if b1.ndim == 1:
        b1 = b1[None, :]
    if b2.ndim == 1:
        b2 = b2[None, :]

    assert b2.shape[0] == 1

    ov = overlap2d(b1, b2)

    return ov / (area2d(b1) + area2d(b2) - ov)


def nms2d(boxes, overlap=0.6):
    """Compute the soft nms given a set of scored boxes,
    as numpy array with 5 columns <x1> <y1> <x2> <y2> <score>
    return the indices of the tubelets to keep
    """
    if boxes.size == 0:
        return np.array([], dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    scores = boxes[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(scores)[::-1]
    weight = np.zeros_like(scores) + 1

    while order.size > 0:
        i = order[0]

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = inter / (1e-5 + areas[i] + areas[order[1:]] - inter)

        index = np.where(iou > overlap)[0]
        weight[order[index + 1]] = 1 - iou[index]
        weight[order[index + 1]] = 0

        index2 = np.where(iou <= overlap)[0]
        order = order[index2 + 1]

    boxes[:, 4] = boxes[:, 4] * weight

    return boxes


# TUBELETS
""" tubelets of length K are represented using numpy array with 4K columns """


def nms_tubelets(dets, overlapThresh=0.3, top_k=None):
    """Compute the NMS for a set of scored tubelets
    scored tubelets are numpy array with 4K+1 columns, last one being the score
    return the indices of the tubelets to keep
    """

    # If there are no detections, return an empty list
    if len(dets) == 0:
        dets
    if top_k is None:
        top_k = len(dets)

    K = int((dets.shape[1] - 1) / 4)

    # Coordinates of bounding boxes
    x1 = [dets[:, 4 * k] for k in range(K)]
    y1 = [dets[:, 4 * k + 1] for k in range(K)]
    x2 = [dets[:, 4 * k + 2] for k in range(K)]
    y2 = [dets[:, 4 * k + 3] for k in range(K)]

    # Compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    # area = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores = dets[:, -1]
    area = [(x2[k] - x1[k] + 1) * (y2[k] - y1[k] + 1) for k in range(K)]
    order = np.argsort(scores)[::-1]
    weight = np.zeros_like(scores) + 1
    counter = 0

    while order.size > 0:
        i = order[0]
        counter += 1

        # Compute overlap
        xx1 = [np.maximum(x1[k][i], x1[k][order[1:]]) for k in range(K)]
        yy1 = [np.maximum(y1[k][i], y1[k][order[1:]]) for k in range(K)]
        xx2 = [np.minimum(x2[k][i], x2[k][order[1:]]) for k in range(K)]
        yy2 = [np.minimum(y2[k][i], y2[k][order[1:]]) for k in range(K)]

        w = [np.maximum(0, xx2[k] - xx1[k] + 1) for k in range(K)]
        h = [np.maximum(0, yy2[k] - yy1[k] + 1) for k in range(K)]

        inter_area = [w[k] * h[k] for k in range(K)]
        ious = sum([inter_area[k] / (area[k][order[1:]] + area[k][i] - inter_area[k]) for k in range(K)])
        index = np.where(ious > overlapThresh * K)[0]
        weight[order[index + 1]] = 1 - ious[index]

        index2 = np.where(ious <= overlapThresh * K)[0]
        order = order[index2 + 1]

    dets[:, -1] = dets[:, -1] * weight

    new_scores = dets[:, -1]
    new_order = np.argsort(new_scores)[::-1]
    dets = dets[new_order, :]

    return dets[:top_k, :]


# TUBES
""" tubes are represented as a numpy array with nframes rows and 5 columns (frame, x1, y1, x2, y2). frame number are 1-indexed, coordinates are 0-indexed """


def iou3d(b1, b2):
    """Compute the IoU between two tubes with same temporal extent"""

    assert b1.shape[0] == b2.shape[0]
    assert np.all(b1[:, 0] == b2[:, 0])

    ov = overlap2d(b1[:, 1:5], b2[:, 1:5])

    return np.mean(ov / (area2d(b1[:, 1:5]) + area2d(b2[:, 1:5]) - ov))


def iou3dt(b1, b2, spatialonly=False):
    """Compute the spatio-temporal IoU between two tubes"""

    tmin = max(b1[0, 0], b2[0, 0])
    tmax = min(b1[-1, 0], b2[-1, 0])

    if tmax < tmin:
        return 0.0

    temporal_inter = tmax - tmin + 1
    temporal_union = max(b1[-1, 0], b2[-1, 0]) - min(b1[0, 0], b2[0, 0]) + 1

    tube1 = b1[int(np.where(b1[:, 0] == tmin)[0]): int(np.where(b1[:, 0] == tmax)[0]) + 1, :]
    tube2 = b2[int(np.where(b2[:, 0] == tmin)[0]): int(np.where(b2[:, 0] == tmax)[0]) + 1, :]

    return iou3d(tube1, tube2) * (1. if spatialonly else temporal_inter / temporal_union)


def nms3dt(tubes, overlap=0.5):
    """Compute NMS of scored tubes. Tubes are given as list of (tube, score)
    return the list of indices to keep
    """

    if not tubes:
        return np.array([], dtype=np.int32)

    I = np.argsort([t[1] for t in tubes])
    indices = np.zeros(I.size, dtype=np.int32)
    counter = 0

    while I.size > 0:
        i = I[-1]
        indices[counter] = i
        counter += 1
        ious = np.array([iou3dt(tubes[ii][0], tubes[i][0]) for ii in I[:-1]])
        I = I[np.where(ious <= overlap)[0]]

    return indices[:counter]


# AP


def pr_to_ap(pr):
    """Compute AP given precision-recall
    pr is a Nx2 array with first row being precision and second row being recall
    """

    prdif = pr[1:, 1] - pr[:-1, 1]
    prsum = pr[1:, 0] + pr[:-1, 0]

    return np.sum(prdif * prsum * 0.5)


# Fit bezier curve
Mtk = lambda n, t, k: t ** k * (1 - t) ** (n - k) * n_over_k(n, k)
BezierCoeff = lambda ts: [[Mtk(5, t, k) for k in range(6)] for t in ts]


def bezierFit(x, y, frame_ids, c, meta):
    dy = y[1:] - y[:-1]
    dx = x[1:] - x[:-1]
    dframe_id = frame_ids[1:] - frame_ids[:-1]
    dt = (dx ** 2 + dy ** 2 + dframe_id ** 2) ** 0.5
    t = dt / dt.sum()
    t = np.hstack(([0], t))
    t = t.cumsum()
    data = np.column_stack((x, y, frame_ids))
    # print('data:', data)
    Pseudoinverse = np.linalg.pinv(BezierCoeff(t))  # (7,4) -> (7,9)
    control_points = Pseudoinverse.dot(data)  # (4,7)*(7,4) -> (4,3)
    control_points[0] = data[0]
    control_points[-1] = data[-1]
    # print('data:')
    # print(data)
    # print('abs:')
    # print(control_points)
    relative_control_points = control_points - data[c]  # 中间帧的(cx,cy,ct)
    # print('relative:')
    # print(relative_control_points)
    w_normal, h_normal, t_normal = meta
    # relative_control_points[:, 0] /= w_normal
    # relative_control_points[:, 1] /= h_normal
    relative_control_points[:, 2] /= t_normal
    # print('normalized relative:')
    # print(relative_control_points)
    # exit()
    return control_points, relative_control_points


# Fit bezier just using the sparse tublet center
def curveFromTube(sparse_tublet, center_idx, meta):
    frame_id = sparse_tublet[:, 0]
    center_xs = (sparse_tublet[:, 1] + sparse_tublet[:, 3]) / 2
    center_ys = (sparse_tublet[:, 2] + sparse_tublet[:, 4]) / 2
    ctp, relative_ctp = bezierFit(center_xs, center_ys, frame_id, center_idx, meta)
    return ctp, relative_ctp


