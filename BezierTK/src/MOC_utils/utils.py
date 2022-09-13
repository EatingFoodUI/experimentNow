from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import cv2
import numpy as np
import random
import math
import pickle
import time


def bbox_iou(box1, box2, x1y1x2y2=False):
    """
    Returns the IoU of two bounding boxes
    """
    N, M = len(box1), len(box2)
    if x1y1x2y2:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    # get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1.unsqueeze(1), b2_x1)
    inter_rect_y1 = torch.max(b1_y1.unsqueeze(1), b2_y1)
    inter_rect_x2 = torch.min(b1_x2.unsqueeze(1), b2_x2)
    inter_rect_y2 = torch.min(b1_y2.unsqueeze(1), b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)
    # Union Area
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
    b1_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1)).view(-1,1).expand(N,M)
    b2_area = ((b2_x2 - b2_x1) * (b2_y2 - b2_y1)).view(1,-1).expand(N,M)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)

def warp_pos(warp_matrix, pos):
    p1 = np.array([pos[0], pos[1], 1]).reshape(3, 1)
    p2 = np.array([pos[2], pos[3], 1]).reshape(3, 1)
    p1_n = np.dot(warp_matrix, p1)
    p2_n = np.dot(warp_matrix, p2)
    pos = np.concatenate([p1_n, p2_n], 0).reshape(-1).astype(np.int64)

    return pos


def warp_bbox_direct(pos, warp_matrix):
    res = warp_pos(warp_matrix, pos)
    return res


def warp_bbox_iterate(mats, base_frame_id, cur_frame_id, pos, cur_frame=None, decode=False):
    if decode:
        # 我把两帧之间的warp mat存在了后一帧
        if cur_frame_id > base_frame_id:
            base_frame_id += 1
            s = 1
        else:
            s = -1
        for fid in range(base_frame_id, cur_frame_id, s):
            warp_matrix = mats[base_frame_id][s]
            pos = warp_pos(warp_matrix, pos)

    else:
        # 我把两帧之间的warp mat存在了后一帧
        if cur_frame_id > base_frame_id:
            s = -1
        else:
            cur_frame_id += 1
            s = 1

        for fid in range(cur_frame_id, base_frame_id, s):
            warp_matrix = mats[cur_frame_id][s]
            pos = warp_pos(warp_matrix, pos)

            if cur_frame is not None:
                sz = cur_frame.shape[:-1]
                cur_frame = cv2.warpAffine(cur_frame, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR)

    if cur_frame is not None:
        return pos, cur_frame
    else:
        return pos


def warp_bbox_direct_v2(mats, cen_frame_id, cur_frame_id, pos, cur_frame=None, decode=False):
    if decode:
        k = (cen_frame_id, cur_frame_id)
    else:
        k = (cur_frame_id, cen_frame_id)

    # print(k)
    if k in mats:
        warp_matrix = mats[k]
        pos = warp_pos(warp_matrix, pos)
        if cur_frame is not None:
            sz = cur_frame.shape[:-1]
            # print('direct')
            time1 = time.time()
            cur_frame_aligned = cv2.warpAffine(cur_frame, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR)
            # print(time.time() - time1)

    else:
        if cur_frame is not None:
            print('no direct, has to indirect')
            time1 = time.time()
            pos, cur_frame_aligned, = warp_bbox_iterate(mats, cen_frame_id, cur_frame_id, pos, cur_frame, decode=decode)
            # print(time.time() - time1)
        else:
            pos = warp_bbox_iterate(mats, cen_frame_id, cur_frame_id, pos, decode=decode)

    if cur_frame is not None:
        return pos, cur_frame_aligned
    return pos


def savePickle(gt_curves, path):
    with open(path, 'wb') as f:
        pickle.dump(gt_curves, f)
        f.close()


def readPickle(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
        f.close()
        return d


def ECC(src, dst, warp_mode=cv2.MOTION_EUCLIDEAN, max_iter=100, eps=1e-5, scale=None, align=False):
    """Compute the warp matrix from src to dst.
    Parameters
    ----------
    src : ndarray
        An NxM matrix of source img(BGR or Gray), it must be the same format as dst.
    dst : ndarray
        An NxM matrix of target img(BGR or Gray).
    warp_mode: flags of opencv
        translation: cv2.MOTION_TRANSLATION
        rotated and shifted: cv2.MOTION_EUCLIDEAN
        affine(shift,rotated,shear): cv2.MOTION_AFFINE
        homography(3d): cv2.MOTION_HOMOGRAPHY
    eps: float
        the threshold of the increment in the correlation coefficient between two iterations
    max_iter: int
        the number of iterations.
    scale: float or [int, int]
        scale_ratio: float
        scale_size: [W, H]
    Returns
    -------
    warp matrix : ndarray
        Returns the warp matrix from src to dst.
        if motion model is homography, the warp matrix will be 3x3, otherwise 2x3
    src_aligned: ndarray
        aligned source image of gray
    """
    assert src.shape == dst.shape, "the source image must be the same format to the target image!"
    # BGR2GRAY
    if src.ndim == 3:
        # Convert images to grayscale
        src = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    # make the imgs smaller to speed up
    if scale is not None:
        if isinstance(scale, float) or isinstance(scale, int):
            if scale != 1:
                src_r = cv2.resize(src, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                dst_r = cv2.resize(dst, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                scale = [scale, scale]
            else:
                src_r, dst_r = src, dst
                scale = None
        else:
            if scale[0] != src.shape[1] and scale[1] != src.shape[0]:
                src_r = cv2.resize(src, (scale[0], scale[1]), interpolation=cv2.INTER_LINEAR)
                dst_r = cv2.resize(dst, (scale[0], scale[1]), interpolation=cv2.INTER_LINEAR)
                scale = [scale[0] / src.shape[1], scale[1] / src.shape[0]]
            else:
                src_r, dst_r = src, dst
                scale = None
    else:
        src_r, dst_r = src, dst
    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iter, eps)

    (cc, warp_matrix) = cv2.findTransformECC(src_r, dst_r, warp_matrix, warp_mode, criteria)
    if scale is not None:
        warp_matrix[0, 2] = warp_matrix[0, 2] / scale[0]
        warp_matrix[1, 2] = warp_matrix[1, 2] / scale[1]

    if align:
        sz = src.shape
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use warpPerspective for Homography
            src_aligned = cv2.warpPerspective(src, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR)
        else:
            # Use warpAffine for Translation, Euclidean and Affine
            src_aligned = cv2.warpAffine(src, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR)
        return warp_matrix, src_aligned
    else:
        return warp_matrix


def random_affine(images, gt_sparse_tublet=None, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2),
                  borderValue=(127.5, 127.5, 127.5)):
    border = 0  # width of added border (optional)
    N, height, width, _ = images.shape
    # Rotation and Scale
    R = np.eye(3)
    a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
    # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
    s = random.random() * (scale[1] - scale[0]) + scale[0]
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(width / 2, height / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = (random.random() * 2 - 1) * translate[0] * height + border  # x translation (pixels)
    T[1, 2] = (random.random() * 2 - 1) * translate[1] * width + border  # y translation (pixels)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

    M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    images_warp = images.copy()
    for i in range(N):
        images_warp[i] = cv2.warpPerspective(images[i], M, dsize=(width, height), flags=cv2.INTER_LINEAR,
                                             borderValue=borderValue)  # BGR order borderValue

    if gt_sparse_tublet is not None:
        # tublet in the clip
        for tube_id in gt_sparse_tublet:
            targets = gt_sparse_tublet[tube_id]
            n = targets.shape[0]
            points = targets[:, 1:].copy()
            area0 = (points[:, 2] - points[:, 0]) * (points[:, 3] - points[:, 1])

            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = points[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # apply angle-based reduction
            radians = a * math.pi / 180
            reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
            x = (xy[:, 2] + xy[:, 0]) / 2
            y = (xy[:, 3] + xy[:, 1]) / 2
            w = (xy[:, 2] - xy[:, 0]) * reduction
            h = (xy[:, 3] - xy[:, 1]) * reduction
            xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

            # reject warped points outside of image
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))
            i = (w > 4) & (h > 4) & (area / (area0 + 1e-16) > 0.1) & (ar < 10)

            if not all(i):
                return images, gt_sparse_tublet, M

            targets = targets[i]
            targets[:, 1:] = xy[i]
            gt_sparse_tublet[tube_id] = targets

        return images_warp, gt_sparse_tublet, M
    else:
        return images_warp


def letterbox(images, height=608, width=1088,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    # 图片矩阵化
    images = np.array(images)
    ratio, dw, dh = 0, 0, 0
    N, src_h, src_w, _ = images.shape
    ratio = min(float(height) / src_h, float(width) / src_w)
    new_shape = (round(src_w * ratio), round(src_h * ratio))  # new_shape = [width, height]
    images_ret = np.zeros((N, height, width, 3), dtype=np.uint8)
    for i in range(N):
        dw = (width - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)

        img = images[i].copy()
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
        images_ret[i] = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                           value=color)  # padded rectangular

    return images_ret, ratio, dw, dh


def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=5):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        i = 0
        for p in pts:
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)
            i += 1


def drawpoly(img, pts, color, thickness=1, style='dotted', ):
    s = pts[0]
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        drawline(img, s, e, color, thickness, style)


def drawrect(img, pt1, pt2, color, thickness=1, style='dotted'):
    pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
    drawpoly(img, pts, color, thickness, style)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


# feat: b, h*w, 2*K
# ind: b, N

# feature：移动特征b,h*w,c
# index: 峰值点索引，一维
def _gather_feature(feature, index, index_all=None):
    # dim = channel = 2*K
    # feature b, h*w , c
    # index  b, N --> b, N, c
    # index_all b, N, 2*K  (mov:no, wh,yes)

    if index_all is not None:
        index0 = index_all
    else:
        # 获取第三个维度数值
        dim = feature.size(2)
        # index0为index的三维扩展，三维维度是feature的通道数。
        index0 = index.unsqueeze(2).expand(index.size(0), index.size(1), dim)
    # feature = feature.gather(1, index0)
    # 处理每个批次
    # 主要作用是将移动位置的特征转换成周围峰值点的移动特征，目的类似于nms，即gather_feature
    for bt in range(index0.shape[0]):
        # 当前批次的索引
        index = index0[bt]
        # 想象成对二维索引进行处理
        for i in range(index0.shape[1]):
            for j in range(index0.shape[2]):
                if index[i][j] == -1:
                    feature[bt][i][j] = 0
                # index[i][j]为峰值点在每个方向的copy（12维）的索引，看成峰值点索引
                # [j]是哪个方向
                # feature[bt][index[i][j]][j]峰值点在哪个方向上的移动特征
                # feature[bt][i][j]
                feature[bt][i][j] = feature[bt][index[i][j]][j]
    # feature --> b, N, 2*K
    feature = feature[:, :index0.shape[1], :]
    return feature



def _tranpose_and_gather_feature(feature, index, index_all=None):
    # 维度交换
    # b,c,h,w --> b,h,w,c
    feature = feature.permute(0, 2, 3, 1).contiguous()
    # 一维变二维
    # b,h,w,c --> b,h*w,c
    feature = feature.view(feature.size(0), -1, feature.size(3))
    # 变成分数最高的前256个特征点的移动方向
    # feature --> b, N, 2*K
    feature = _gather_feature(feature, index, index_all=index_all)
    return feature


def flip_tensor(x):
    return torch.flip(x, [3])
    # MODIFY for pytorch 0.4.0
    # tmp = x.detach().cpu().numpy()[..., ::-1].copy()
    # return torch.from_numpy(tmp).to(x.device)


def flip(img):
    return img[:, :, ::-1].copy()


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img,
                             trans,
                             (int(output_size[0]), int(output_size[1])),
                             flags=cv2.INTER_LINEAR)

    return dst_img
