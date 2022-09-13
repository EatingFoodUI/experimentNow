import numpy as np
import torch

from tracking_utils.log import logger
from MOC_utils.model import create_model, load_model
from MOC_utils.utils import transform_preds, _gather_feature, _tranpose_and_gather_feature
from ACT_utils.ACT_utils import area2d
from detector.decode import _nms, _topk, takeClosest, find_nearest
from .basetrack import BaseTrack, TrackState
from tracker import matching
import time


class STrack(BaseTrack):
    def __init__(self, tlbrs, score, start_frame):
        self.tlbrs = tlbrs.reshape(-1, 4)
        self.score = score
        self.is_activated = False

        self.start_frame = start_frame
        self.end_frame = start_frame + len(self.tlbrs) - 1

    def update(self, new_track, frame_id):
        # update strack bbox
        # 有雷
        w = self.score / (self.score + new_track.score)

        overlap = self.end_frame - new_track.start_frame + 1
        new_track.tlbrs[:overlap, :] = w * self.tlbrs[len(self.tlbrs) - overlap:, :] + (1 - w) * new_track.tlbrs[
                                                                                                 :overlap, :]
        # new_track.tlbrs[:overlap, :] = new_track.tlbrs[:overlap, :]
        self.tlbrs = new_track.tlbrs
        self.score = new_track.score
        self.end_frame = new_track.end_frame

        self.state = TrackState.Tracked
        self.is_activated = True

    def activate(self, frame_id):
        """Start a new tracklet"""
        self.track_id = self.next_id()
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True

    def re_activate(self, new_track, frame_id, new_id=False):
        # update strack bbox
        w = self.score / (self.score + new_track.score)

        overlap = self.end_frame - new_track.start_frame + 1
        new_track.tlbrs[:overlap, :] = w * self.tlbrs[len(self.tlbrs) - overlap:, :] + (1 - w) * new_track.tlbrs[
                                                                                                 :overlap, :]

        # new_track.tlbrs[:overlap, :] = new_track.tlbrs[:overlap, :]
        self.tlbrs = new_track.tlbrs
        self.score = new_track.score
        self.end_frame = new_track.end_frame

        self.state = TrackState.Tracked
        self.is_activated = True
        if new_id:
            self.track_id = self.next_id()

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        ret = self.tlbrs.copy()
        for i in range(len(ret)):
            ret[i][2:] -= ret[i][:2]
        return ret


# def takeClosest(num, frame_stride, forward_frames):
#     num = num.cpu()
#     collection = torch.tensor([i * frame_stride for i in range(-1 * (forward_frames // 2), forward_frames // 2 + 1)])
#     collection = collection[None, :]
#     collection = collection.repeat(num.shape[1], 1)
#     index = torch.argmin(torch.abs(collection - num.reshape(-1, 1)), dim=1).type(torch.LongTensor)
#     index = index.reshape(-1, 1)
#     # print(index.shape)
#     # print(collection.shape)
#     # print()
#     # res = torch.argmin(torch.abs(collection - num))
#     # print(res)
#     res = torch.gather(collection, 1, index)
#     res = res.reshape(1, -1)
#     # print(res.shape)
#     # print(res)
#     # exit()
#     # (3,4) (2) (3,2)
#     return res


# def find_nearest(twodarray, value):
#     index = torch.argmin(torch.abs(twodarray - value), dim=1)
#     # res = torch.gather(twodarray, 1, index)
#     # res = res.reshape(1, -1)
#     # print(index)
#     return index


def decode(heatmap, wh, bezier_ctp, max_objs, frame_stride, forward_frames):
    # 1.heatmap
    batch, c, height, width = heatmap.size()
    # perform 'nms' on heatmaps
    heatmap = _nms(heatmap)

    # (b, max_objs)
    scores, index, xs, ys = _topk(heatmap, max_objs)

    # 2.bezier curve for center points
    rctp = _tranpose_and_gather_feature(bezier_ctp, index)
    rctp = rctp.view(batch, max_objs, 12)

    expanded_center_xs = xs.clone().unsqueeze(2).expand(batch, max_objs, 4)
    expanded_center_ys = ys.clone().unsqueeze(2).expand(batch, max_objs, 4)
    ctp = rctp.clone()
    ctp[..., 0::3] += expanded_center_xs
    ctp[..., 1::3] += expanded_center_ys
    ctp_x = ctp[..., 0::3]
    ctp_y = ctp[..., 1::3]
    ctp_t = ctp[..., 2::3]

    # ctp_t *= frame_stride
    # ctp_t[..., 0] = takeClosest(ctp_t[..., 0], frame_stride, forward_frames)
    # ctp_t[..., -1] = takeClosest(ctp_t[..., -1], frame_stride, forward_frames)
    for bt in range(ctp.shape[0]):
        # modify for normalize
        ctp_t[bt] *= frame_stride
        for obj in range(ctp.shape[1]):
            ctp_t[bt][obj][0] = takeClosest(ctp_t[bt][obj][0], frame_stride, forward_frames)
            ctp_t[bt][obj][-1] = takeClosest(ctp_t[bt][obj][-1], frame_stride, forward_frames)

    # now the t_dim is normalized back
    ctp_stacked = (torch.cat((ctp_x, ctp_y, ctp_t), dim=1)).cpu().numpy()

    actual_forward_frames = frame_stride * (forward_frames - 1) + 1
    xs_all = torch.zeros((batch, max_objs, actual_forward_frames)).cuda()
    ys_all = torch.zeros((batch, max_objs, actual_forward_frames)).cuda()

    func = lambda a, k: \
        (1 - k) * ((1 - k) * ((1 - k) * a[..., 0] + k * a[..., 1]) + k * ((1 - k) * a[..., 1] + k * a[..., 2])) + \
        k * ((1 - k) * ((1 - k) * a[..., 1] + k * a[..., 2]) + k * ((1 - k) * a[..., 2] + k * a[..., 3]))
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
            for i in range(actual_forward_frames):
                frame_idx = (i - actual_forward_frames // 2)

                if frame_idx == 0:
                    xs_all[bt][obj][i] = xs[bt][obj]
                    ys_all[bt][obj][i] = ys[bt][obj]

                if ctp_t[bt][obj][0] < frame_idx < ctp_t[bt][obj][-1]:
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

        # for i in range(actual_forward_frames):
        #     frame_idx = (i - actual_forward_frames // 2)
        #
        #     if frame_idx == 0:
        #         xs_all[..., i] = xs
        #         ys_all[..., i] = ys
        #
        #     a = ctp_t[..., 0] < frame_idx
        #     b = frame_idx < ctp_t[..., -1]
        #     mask1 = a & b
        #     if mask1.any():
        #         idx = find_nearest(ct_samples[mask1], frame_idx)
        #         xs_all[mask1][:, i] = cx_samples[mask1, idx]
        #         ys_all[mask1][:, i] = cy_samples[mask1, idx]

        #
        #     mask2 = ctp_t[..., 0] == frame_idx
        #     xs_all[mask2][:, i] = ctp_x[mask2][:, 0]
        #     ys_all[mask2][:, i] = ctp_y[mask2][:, 0]
        #
        #     mask3 = ctp_t[..., -1] == frame_idx
        #     xs_all[mask3][:, i] = ctp_x[mask3][:, -1]
        #     ys_all[mask3][:, i] = ctp_y[mask3][:, -1]
        #
        #     mask = ~(mask1 | mask2 | mask3)
        #     xs_all[mask][:, i] = -1
        #     ys_all[mask][:, i] = -1

    xs_all_int = xs_all[..., 0::frame_stride].long()
    ys_all_int = ys_all[..., 0::frame_stride].long()

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
            last_key_frame = (k - 1) * frame_stride
            cur_key_frame = k * frame_stride
            for frame_id in range(last_key_frame, cur_key_frame + 1):
                m = (frame_id - last_key_frame) / (cur_key_frame - last_key_frame)
                actual_w[..., frame_id] = (1 - m) * w[..., k - 1] + m * w[..., k]
                actual_h[..., frame_id] = (1 - m) * h[..., k - 1] + m * h[..., k]

    actual_wh[..., 0::2] = actual_w
    actual_wh[..., 1::2] = actual_h

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

    return detections, ctp


class JDETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        print('Creating model...')
        self.model = create_model(opt.arch, opt.branch_info, opt.head_conv, opt.forward_frames)
        self.model = load_model(self.model, opt.rgb_model)
        self.model = self.model.to(opt.device)
        self.model.eval()

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = -1
        self.det_thresh = opt.det_thres
        self.max_time_lost = self.opt.forward_frames
        self.max_per_image = opt.max_objs

    def process(self, images, hm=None, frame_stride=1):
        # for i in range(len(images)):
        for img_id in images:
            images[img_id] = images[img_id].type(torch.FloatTensor)
            images[img_id] = images[img_id].to(self.opt.device)

        if hm is not None:
            hm = hm.cuda()

        with torch.no_grad():
            rgb_output = self.model(images)
            if hm is None:
                hm = rgb_output[0]['hm'].sigmoid_()
            wh = rgb_output[0]['wh']
            bezier_ctp = rgb_output[0]['bezier_ctp']

        if hm.ndim != 4:
            hm = hm.unsqueeze(1)
        detections, ctp = decode(hm, wh, bezier_ctp, self.opt.max_objs,
                                 frame_stride=frame_stride, forward_frames=self.opt.forward_frames)

        return detections, ctp

    def post_process(self, dets, meta):
        dets = dets.squeeze()
        dets = dets.detach().cpu().numpy()
        c = np.array([meta['width'] / 2., meta['height'] / 2.], dtype=np.float32)
        s = max(float(meta['input_w']) / float(meta['input_h']) * meta['height'], meta['width']) * 1.0

        mask = dets[:, -1] > self.det_thresh
        dets = dets[mask]
        for j in range((dets.shape[1] - 1) // 2):
            # dets[:, 2 * j] = np.clip(dets[:, 2 * j], 0, meta['output_w'] - 1)
            # dets[:, 2 * j + 1] = np.clip(dets[:, 2 * j + 1], 0, meta['output_h'] - 1)
            dets[:, 2 * j:2 * j + 2] = transform_preds(dets[:, 2 * j:2 * j + 2],
                                                       c, s, (meta['output_w'], meta['output_h']))
            dets[:, 2 * j] = np.clip(dets[:, 2 * j], 0, meta['width'] - 1)
            dets[:, 2 * j + 1] = np.clip(dets[:, 2 * j + 1], 0, meta['height'] - 1)
            # print(dets[:, 2 * j:2 * j + 2])

        for j in range((dets.shape[1] - 1) // 4):
            area = area2d(dets[:, 4 * j:4 * j + 4])
            mask = area <= 0
            if mask.any():
                dets[mask, 4 * j:4 * j + 4] = 0, 0, 0, 0

        k = ((dets.shape[1] - 1) // 4) // 2
        key_cx = (dets[:, 4 * k] + dets[:, 4 * k + 2]) / 2
        key_cy = (dets[:, 4 * k + 1] + dets[:, 4 * k + 3]) / 2
        root_wh = np.sqrt((dets[:, 4 * k + 2] - dets[:, 4 * k]) * (dets[:, 4 * k + 3] - dets[:, 4 * k + 1]))
        # min_wh = np.min(np.stack([(dets[:, 4 * k + 2] - dets[:, 4 * k]), (dets[:, 4 * k + 3] - dets[:, 4 * k + 1])], axis=1), axis=1)
        # print(key_cx)
        # print(key_cy)
        for j in range((dets.shape[1] - 1) // 4):
            if j == k:
                continue
            cx = (dets[:, 4 * j] + dets[:, 4 * j + 2]) / 2
            cy = (dets[:, 4 * j + 1] + dets[:, 4 * j + 3]) / 2
            mask = np.sqrt((key_cx - cx) ** 2 + (key_cy - cy) ** 2) >= root_wh * 2.0
            if mask.any():
                dets[mask, 4 * j:4 * j + 4] = 0, 0, 0, 0

        # e_ratio = 0.8
        # e_num = int(e_ratio * self.opt.forward_frames) * 4
        # mask = np.count_nonzero(dets[:, :-1], axis=1) > e_num
        # dets = dets[mask]
        return dets

    def update(self, images, meta, frame_id):
        overlap_dist = self.opt.overlap_dist
        self.frame_id = frame_id
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        hm = None if 'hm' not in meta else meta['hm']
        frame_stride = meta['frame_stride'].cpu().numpy().tolist()

        detections, ctp = self.process(images, hm, frame_stride=frame_stride[0])
        detections = self.post_process(detections, meta)

        if len(detections) > 0:
            detections = [STrack(tlbrs[:-1], tlbrs[-1], self.frame_id) for tlbrs in detections]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with IOU'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        dists = matching.tracklets_iou_distance(strack_pool, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=overlap_dist)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = strack_pool[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        dists = matching.tracklets_iou_distance(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=overlap_dist)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_stracks.append(unconfirmed[itracked])

        for it in u_unconfirmed:
            track = unconfirmed[it]
            # if not track.state == TrackState.Lost:
            #     track.mark_lost()
            #     lost_stracks.append(track)
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue

            track.activate(self.frame_id)
            activated_stracks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if track.end_frame - self.frame_id < 0:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        # self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        # output_stracks = [track for track in self.tracked_stracks]

        # logger.debug('===========Frame {}=========='.format(self.frame_id))
        # logger.debug('Activated: {}'.format([track.track_id for track in activated_stracks]))
        # logger.debug('Refind: {}'.format([track.track_id for track in refind_stracks]))
        # logger.debug('Lost: {}'.format([track.track_id for track in lost_stracks]))
        # logger.debug('Removed: {}'.format([track.track_id for track in removed_stracks]))

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.tracklets_iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
