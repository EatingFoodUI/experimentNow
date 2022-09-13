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

# 初始化轨迹
class STrack(BaseTrack):
    def __init__(self, tlbrs, score, start_frame):
        self.tlbrs = tlbrs.reshape(-1, 4)
        self.score = score
        self.is_activated = False

        # 轨迹开始帧
        self.start_frame = start_frame
        # 轨迹结束帧
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


# 通过特征解码检测框和控制点
def decode(heatmap, wh, bezier_ctp, max_objs, frame_stride, forward_frames):
    # 1.heatmap
    # 批次、通道数、特征图高、宽
    batch, c, height, width = heatmap.size()
    # perform 'nms' on heatmaps
    # 非极大值抑制，以提取峰值关键点。保留热图分数大于阈值的关键点的位置。
    heatmap = _nms(heatmap)

    # (b, max_objs)
    # 拿到分数最高的前k个峰值点（代表最多有256个物体的移动框）
    # 分数、索引、x、y
    scores, index, xs, ys = _topk(heatmap, max_objs)

    # byteTrack


    # 2.bezier curve for center points
    # bezier_ctp 中心帧对象移动特征图（bezier点移动），index 中心帧特征图峰值点索引
    # rctp：k个峰值点在12个方向上的移动特征
    rctp = _tranpose_and_gather_feature(bezier_ctp, index)
    rctp = rctp.view(batch, max_objs, 12)

    # 12指的是3阶贝塞尔曲线的4个点在x、y、t三维上的位置特征
    # 中心点的x坐标和y坐标增加第三维为4
    # 移动特征加上峰值点特征等于峰值点下一次的位置特征
    ## 贝塞尔曲线阶数修改的位置之一
    expanded_center_xs = xs.clone().unsqueeze(2).expand(batch, max_objs, 4)
    expanded_center_ys = ys.clone().unsqueeze(2).expand(batch, max_objs, 4)
    # ctp复制k个峰值点在12个方向上的移动特征
    ctp = rctp.clone()
    # x轴加四个点的特征得到四个点的位置特征
    ctp[..., 0::3] += expanded_center_xs
    # y轴加四个点的特征得到四个点的位置特征
    ctp[..., 1::3] += expanded_center_ys
    # 得到四个点在x、y、t上的特征
    ctp_x = ctp[..., 0::3]
    ctp_y = ctp[..., 1::3]
    ctp_t = ctp[..., 2::3]

    # ctp_t *= frame_stride
    # ctp_t[..., 0] = takeClosest(ctp_t[..., 0], frame_stride, forward_frames)
    # ctp_t[..., -1] = takeClosest(ctp_t[..., -1], frame_stride, forward_frames)
    # bt：贝塞尔曲线控制点特征
    for bt in range(ctp.shape[0]):
        # modify for normalize
        # 因为控制点之间间隔了frame_stride帧
        ctp_t[bt] *= frame_stride
        # 对每一个检测框的控制点进行操作
        # ？？
        for obj in range(ctp.shape[1]):
            ctp_t[bt][obj][0] = takeClosest(ctp_t[bt][obj][0], frame_stride, forward_frames)
            ctp_t[bt][obj][-1] = takeClosest(ctp_t[bt][obj][-1], frame_stride, forward_frames)

    # now the t_dim is normalized back
    # 四个控制点对于所有物体的x、y、t预测
    ctp_stacked = (torch.cat((ctp_x, ctp_y, ctp_t), dim=1)).cpu().numpy()

    # 真正前进的帧数
    actual_forward_frames = frame_stride * (forward_frames - 1) + 1
    # 初始化所有帧的控制点x轴
    xs_all = torch.zeros((batch, max_objs, actual_forward_frames)).cuda()
    # 初始化所有帧的控制点y轴
    ys_all = torch.zeros((batch, max_objs, actual_forward_frames)).cuda()

    # 三阶贝塞尔曲线函数
    # 贝塞尔曲线修改位置
    # k是要学的参数（范围是0.001到1），a是控制点位置
    func = lambda a, k: \
        (1 - k) * ((1 - k) * ((1 - k) * a[..., 0] + k * a[..., 1]) + k * ((1 - k) * a[..., 1] + k * a[..., 2])) + \
        k * ((1 - k) * ((1 - k) * a[..., 1] + k * a[..., 2]) + k * ((1 - k) * a[..., 2] + k * a[..., 3]))
    N = 1000
    delta_k = 1. / N
    # (1,300,1000)

    # 贝塞尔曲线
    cxyt_samples = np.array([func(ctp_stacked, i * delta_k) for i in range(1, N + 1)]).transpose((1, 2, 0))
    cxyt_samples = torch.from_numpy(cxyt_samples)
    # 256个物体
    obj_num = cxyt_samples.shape[1] // 3
    # 每个物体控制点的x、y、t坐标
    cx_samples = cxyt_samples[:, :obj_num]
    cy_samples = cxyt_samples[:, obj_num:2 * obj_num]
    ct_samples = cxyt_samples[:, 2 * obj_num:]

    # 处理每一个批次的每一个物体的每一帧
    # xs_all ys_all既是控制点坐标，也可以是物体位置坐标
    # 具体待看
    for bt in range(xs_all.shape[0]):
        for obj in range(xs_all.shape[1]):
            for i in range(actual_forward_frames):
                # 是相对中心帧位置
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

    # xs_all_int ys_all_int是采样帧序列的帧的控制点的坐标
    xs_all_int = xs_all[..., 0::frame_stride].long()
    ys_all_int = ys_all[..., 0::frame_stride].long()

    # 2是为了后面获得框长宽
    index_all = torch.zeros((batch, max_objs, forward_frames, 2)).cuda()
    index_all[:, :, :, 0] = xs_all_int + ys_all_int * width
    index_all[:, :, :, 1] = xs_all_int + ys_all_int * width
    index_all[index_all < 0] = -1
    index_all[index_all > width * height - 1] = -1
    index_all = index_all.view(batch, max_objs, forward_frames * 2).long()

    # gather wh in each location after movement
    # 获得移动后的框的大小特征
    wh = _tranpose_and_gather_feature(wh, index, index_all=index_all)
    wh = wh.view(batch, max_objs, forward_frames * 2)
    w = wh[..., 0::2]
    h = wh[..., 1::2]

    # 真正帧大小初始化
    actual_wh = torch.zeros((batch, max_objs, actual_forward_frames * 2)).cuda()
    actual_w = actual_wh[..., 0::2]
    actual_h = actual_wh[..., 1::2]

    # ？？
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

    # 每个物体存在的置信度
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

    # 获得置信度前125个bbox的左上角坐标和右下角坐标
    bboxes = torch.cat(bboxes, dim=2)
    # 得到带分数的bbox：detections
    detections = torch.cat([bboxes, scores], dim=2)

    return detections, ctp


# JDE追踪器
class JDETracker(object):
    def __init__(self, opt, frame_rate=30):
        # 使用cpu还是gpu，使用哪一块gpu
        self.opt = opt
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')

        # 为检测器添加使用的训练好的模型
        print('Creating model...')
        # 创建模型
        self.model = create_model(opt.arch, opt.branch_info, opt.head_conv, opt.forward_frames)
        # 为创建模型导入训练好的模型的参数
        self.model = load_model(self.model, opt.rgb_model)
        # 将模型分配给指定的GPU
        self.model = self.model.to(opt.device)
        # 设置模型为验证模式，不会更新模型参数
        self.model.eval()

        # stracks:每帧检测的对象框的集合/目标 应该是轨迹
        # tracked_stracks:激活的轨迹（正在追踪的物体的轨迹）
        # lost_stracks:丢失的轨迹（只是简单离开的物体的轨迹）
        # removed_stracks:移除的轨迹（已经不会出现的物体的轨迹）
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        # 当前帧id
        self.frame_id = -1
        # 检测阈值
        self.det_thresh = opt.det_thres
        # byteTrack
        # self.det_thresh = 0.2
        # 最大轨迹丢失时间
        self.max_time_lost = self.opt.forward_frames
        # 每帧最多有多少个物体
        self.max_per_image = opt.max_objs

    # 前处理
    def process(self, images, hm=None, frame_stride=1):
        # for i in range(len(images)):
        # 对当前视频段的每一帧进行处理，变成torch并添加到指定GPU上
        for img_id in images:
            images[img_id] = images[img_id].type(torch.FloatTensor)
            images[img_id] = images[img_id].to(self.opt.device)

        if hm is not None:
            hm = hm.cuda()

        with torch.no_grad():
            # 将当前序列的所有帧作为模型的输入，输出得到特征结果
            # 一共有四个输出结果
            # hm：一帧的特征图
            # bezier_ctp：一帧的移动轨迹（bezier曲线的移动轨迹）
            # id：一帧中的所有分类的位置？？
            # wh：7帧的对象的高宽特征
            rgb_output = self.model(images)
            # 得到第一帧的特征图、7帧的框大小、一帧的bezier移动轨迹
            if hm is None:
                hm = rgb_output[0]['hm'].sigmoid_()
            wh = rgb_output[0]['wh']
            bezier_ctp = rgb_output[0]['bezier_ctp']

        if hm.ndim != 4:
            hm = hm.unsqueeze(1)
        # 使用hm、wh、bezier_ctp等，通过解码获得检测框和bezier控制点
        detections, ctp = decode(hm, wh, bezier_ctp, self.opt.max_objs,
                                 frame_stride=frame_stride, forward_frames=self.opt.forward_frames)

        return detections, ctp

    # 可能是修改框大小
    def post_process(self, dets, meta):
        # 去除第一维通道
        dets = dets.squeeze()
        dets = dets.detach().cpu().numpy()
        # 用来修正图片大小用？？
        # c原图一半大小
        c = np.array([meta['width'] / 2., meta['height'] / 2.], dtype=np.float32)
        # max（宽高比 * 原图高，原图宽）
        s = max(float(meta['input_w']) / float(meta['input_h']) * meta['height'], meta['width']) * 1.0

        # 置信度高于det_thresh的检测框才会被使用
        mask = dets[:, -1] > self.det_thresh
        # print(dets[:, -1])
        # print(mask)
        # 去掉底置信度的框
        dets = dets[mask]
        # 对每个框进行处理
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
        # 检测框置信度的阈值（IOU的阈值？？）
        overlap_dist = self.opt.overlap_dist
        # 当前帧id
        self.frame_id = frame_id
        # 激活的轨迹
        activated_stracks = []
        # 重新找到的轨迹
        refind_stracks = []
        # 丢失的轨迹
        lost_stracks = []
        # 移除的轨迹
        removed_stracks = []

        # 当前帧的特征图(还没有生成)
        hm = None if 'hm' not in meta else meta['hm']
        # 帧之间的间隔
        frame_stride = meta['frame_stride'].cpu().numpy().tolist()

        # 前处理
        # 使用bezier曲线，得到检测结果和bezier控制点
        detections, ctp = self.process(images, hm, frame_stride=frame_stride[0])
        # 后处理，减少置信度低的物体的检测框
        detections = self.post_process(detections, meta)

        if len(detections) > 0:
            # byteTrack
            # detections_second = [STrack(tlbrs[:-1], tlbrs[-1], self.frame_id) for tlbrs in detections if tlbrs[-1] < 0.5]
            # detections = [STrack(tlbrs[:-1], tlbrs[-1], self.frame_id) for tlbrs in detections if tlbrs[-1] > 0.5]
            detections = [STrack(tlbrs[:-1], tlbrs[-1], self.frame_id) for tlbrs in detections]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        # 将检测到的轨迹分为正在检测的轨迹和未确认的轨迹
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with IOU'''
        # 使用IOU进行第一次数据关联（检测框与轨迹进行IOU）
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        dists = matching.tracklets_iou_distance(strack_pool, detections)
        # 获得匹配的轨迹、检测框 0.8
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=overlap_dist)

        # 更新轨迹
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # byteTrack
        '''association the untrack to the low score detections'''
        # r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        # if len(r_tracked_stracks) != 0:
        #     dists = matching.tracklets_iou_distance(r_tracked_stracks, detections_second)
        #     matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.4)
        #     if len(matches) != 0:
        #         pass
        #     for itracked, idet in matches:
        #         track = r_tracked_stracks[itracked]
        #         det = detections_second[idet]
        #         if track.state == TrackState.Tracked:
        #             track.update(det, self.frame_id)
        #             activated_stracks.append(track)
        #         else:
        #             track.re_activate(det, self.frame_id, new_id=False)
        #             refind_stracks.append(track)

        for it in u_track:
            track = strack_pool[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        # 处理没有和现有轨迹关联到的对象框，通常作为一个新的轨迹的开始帧
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
        # 更新轨迹的状态
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
