import os
import pickle

import numpy as np

from copy import deepcopy
from datasets.init_dataset import get_dataset

from opts import opts
from ACT_utils.ACT_utils import iou2d, pr_to_ap, nms3dt, iou3dt
from ACT_utils.ACT_build import load_frame_detections
from MOC_utils.utils import drawline, drawpoly, drawrect
import sys
import cv2
import shutil
import glob
import os.path as osp
from progress.bar import Bar

sys.path.append('inference')
root_dir = os.path.join(os.path.dirname(__file__), '..')


def frameAP(opt, print_info=True, mode='val', epoch=-1):
    IOU_thres_GT = opt.threshold
    model_name = opt.exp_id
    Dataset = get_dataset(opt.dataset)
    dataset = Dataset(opt, mode)
    vdir = os.path.join(root_dir, 'data/train')

    inference_dirname = opt.inference_dir
    print('inference_dirname is ', inference_dirname)
    print('IOU_thres_GT is ', IOU_thres_GT)

    vlist = dataset._videos
    # alldets: N,7 --------> ith_video, ith_frame, score, x1, y1, x2, y2
    alldets, alldets_video = load_frame_detections(opt, dataset, vlist, inference_dirname)

    # delete = False
    # delete = opt.vis_det
    delete = True
    if delete:
        for vname in vlist:
            imgs = os.path.join(vdir, vname, 'img2/*')
            for img in glob.glob(imgs):
                if os.path.exists(img):
                    os.remove(img)
    # load the ground truth
    forward_frames, clip_len = dataset.opt.forward_frames, dataset.opt.clip_len
    gt = {}
    gt_video = {}

    bar = Bar('{}'.format('FrameAP'), max=len(vlist))
    for iv, vname in enumerate(vlist):
        if vname not in gt_video:
            gt_video[vname] = {}
        h, w = dataset._resolution[vname]
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
            _range = range(1, MAX_IDX)
        elif dataset.mode == 'val':
            MAX_IDX = nframes + 1
            # _range = range(int(nframes * dataset.train_ratio), MAX_IDX)
            _range = range(int(nframes * dataset.train_ratio), MAX_IDX)

        gt_ped_txt = osp.join(dataset.mot_root, vname, 'gt', 'gt_ped.txt')
        gt_ped_arr = np.loadtxt(gt_ped_txt, dtype=np.float64, delimiter=',')

        gt_ped = {}
        for fid, tid, bx, by, bw, bh, mark, label, vis in gt_ped_arr:
            fid = int(fid)
            if fid not in gt_ped:
                gt_ped[fid] = {}

            gt_ped[fid][tid] = (bx, by, bw, bh, mark, label, vis)

        for start_frame_id in _range:
            cur_max_index = start_frame_id
            # if cur_max_index >= MAX_IDX:
            #     break

            # for k in range(forward_frames):
            cen_frame_id = (start_frame_id + cur_max_index) // 2
            frame_id = start_frame_id
            if frame_id not in gt_ped:
                continue

            for tid, (bx, by, bw, bh, mark, label, vis) in gt_ped[frame_id].items():
                bx, by, bw, bh = int(bx), int(by), int(bw), int(bh)

                _k = (iv, frame_id)
                if _k not in gt_video[vname]:
                    gt_video[vname][_k] = []
                if _k not in gt:
                    gt[_k] = []

                gt_box = np.array([bx, by, bx + bw, by + bh])

                gt[_k].append(gt_box)
                gt_video[vname][_k].append(gt_box)

                '''标记为center frame'''
                cur_img_path = os.path.join(vdir, vname, 'img1/{:0>6}.jpg'.format(frame_id))
                base_bak_path = os.path.join(vdir, vname, 'img2/{:0>6}.jpg'.format(frame_id))
                if opt.vis_det:
                    if not os.path.exists(base_bak_path):
                        base_img_bak = cv2.imread(cur_img_path)
                        cv2.putText(base_img_bak, '{}'.format(frame_id),
                                    (w - 150, h - 100),
                                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 255), 2)

                        cv2.imwrite(base_bak_path, base_img_bak)

        for _k in gt:
            gt[_k] = np.array(gt[_k])

        for _k in gt_video[vname]:
            gt_video[vname][_k] = np.array(gt_video[vname][_k])
        Bar.suffix = '[{0}/{1}]:{2}|Tot: {total:} |ETA: {eta:} '.format(iv + 1, len(vlist), vname, total=bar.elapsed_td,
                                                                        eta=bar.eta_td)
        bar.next()
    bar.finish()

    # pr will be an array containing precision-recall values
    pr_video = {}
    fn_video = {}
    fp_video = {}  # false positives
    tp_video = {}  # true positives
    for vname in vlist:
        pr_video[vname] = np.empty((alldets_video[vname].shape[0] + 1, 2), dtype=np.float32)
        pr_video[vname][0, 0] = 1.0
        pr_video[vname][0, 1] = 0.0
        fn_video[vname] = sum([g.shape[0] for g in gt_video[vname].values()])
        fp_video[vname] = 0
        tp_video[vname] = 0

    pr = np.empty((alldets.shape[0] + 1, 2), dtype=np.float32)  # precision, recall
    pr[0, 0] = 1.0
    pr[0, 1] = 0.0
    fn = sum([g.shape[0] for g in gt.values()])  # false negatives
    fp = 0  # false positives
    tp = 0  # true positives

    for vname in vlist:
        for i, j in enumerate(np.argsort(-alldets_video[vname][:, 2])):
            _k = (int(alldets_video[vname][j, 0]), int(alldets_video[vname][j, 1]))
            pred_box = alldets_video[vname][j, 3:7]
            ispositive = False
            if _k in gt_video[vname]:
                ious = iou2d(gt_video[vname][_k], pred_box)
                amax = np.argmax(ious)

                if ious[amax] >= IOU_thres_GT:
                    ispositive = True
                    gt_video[vname][_k] = np.delete(gt_video[vname][_k], amax, 0)

                    if gt_video[vname][_k].size == 0:
                        del gt_video[vname][_k]

            if ispositive:
                tp_video[vname] += 1
                fn_video[vname] -= 1
            else:
                fp_video[vname] += 1

            pr_video[vname][i + 1, 0] = float(tp_video[vname]) / float(tp_video[vname] + fp_video[vname])
            pr_video[vname][i + 1, 1] = float(tp_video[vname]) / float(tp_video[vname] + fn_video[vname])

    for i, j in enumerate(np.argsort(-alldets[:, 2])):
        _k = (int(alldets[j, 0]), int(alldets[j, 1]))

        pred_box = alldets[j, 3:7]
        score = alldets[j, 2]
        vid = int(alldets[j, 0])
        fid = int(alldets[j, 1])

        # to visualize the detection result#
        vdir = os.path.join(root_dir, 'data/train')
        vname = vlist[vid]

        h, w = dataset._resolution[vname]
        save_images = delete
        if save_images:
            img_path = os.path.join(vdir, vname, 'img2/{:0>6}.jpg'.format(fid))
            if not os.path.exists(img_path):
                img_path = os.path.join(vdir, vname, 'img1/{:0>6}.jpg'.format(fid))
            img = cv2.imread(img_path)

            # area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
            # if area < 100:
            #     print(pred_box)

            # plot detection
            pred_box_draw = pred_box.copy()
            pred_box_draw[0] = np.clip(pred_box[0], 0, w - 1)
            pred_box_draw[1] = np.clip(pred_box[1], 0, h - 1)
            pred_box_draw[2] = np.clip(pred_box[2], 0, w - 1)
            pred_box_draw[3] = np.clip(pred_box[3], 0, h - 1)
            cv2.rectangle(img,
                          (int(pred_box_draw[0]), int(pred_box_draw[1])),
                          (int(pred_box_draw[2]), int(pred_box_draw[3])),
                          (255, 0, 0), 2)
            cv2.putText(img, '{:.2f}'.format(score), (int(pred_box[0]), int(pred_box[1])),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            # print(os.path.join(vdir, vname, 'img2/{:0>6}.jpg'.format(fid)))
            # plot gt
            if (vid, fid) in gt:
                for gt_box in gt[(vid, fid)]:
                    gt_box_draw = gt_box.copy()
                    gt_box_draw[0] = np.clip(gt_box[0], 0, w - 1)
                    gt_box_draw[1] = np.clip(gt_box[1], 0, h - 1)
                    gt_box_draw[2] = np.clip(gt_box[2], 0, w - 1)
                    gt_box_draw[3] = np.clip(gt_box[3], 0, h - 1)
                    drawrect(img,
                             (int(gt_box_draw[0]), int(gt_box_draw[1])), (int(gt_box_draw[2]), int(gt_box_draw[3])),
                             (0, 255, 255), 2, 'dotted')
                    # cv2.imshow('win',img)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()

            cv2.imwrite(os.path.join(vdir, vname, 'img2/{:0>6}.jpg'.format(fid)), img)
            # to visualize the detection result#

        ispositive = False
        if _k in gt:
            ious = iou2d(gt[_k], pred_box)
            amax = np.argmax(ious)

            if ious[amax] >= IOU_thres_GT:
                ispositive = True
                gt[_k] = np.delete(gt[_k], amax, 0)

                if gt[_k].size == 0:
                    del gt[_k]

        if ispositive:
            tp += 1
            fn -= 1
        else:
            fp += 1

        pr[i + 1, 0] = float(tp) / float(tp + fp)
        pr[i + 1, 1] = float(tp) / float(tp + fn)

    ap_video = {}
    video_str = ''
    for vname in vlist:
        ap_video[vname] = 100 * pr_to_ap(pr_video[vname])
        video_str += "{:s} {:8.2f}{:18.2f}{:17.2f}{:13d}{:14d}\n".format(vname, ap_video[vname],
                                                                         pr_video[vname][-1, 0] * 100,
                                                                         pr_video[vname][-1, 1] * 100,
                                                                         fp_video[vname],
                                                                         fn_video[vname])
    ap = 100 * pr_to_ap(pr)
    ID_head = opt.ID_head

    total_video_str = "{:8s} {:8.2f} {:8s}{:8.2f} {:8s}{:8.2f} {:8s}{:d} {:8s}{:d}\n".format("AP", ap,
                                                                                              'Precision',
                                                                                              pr[-1, 0] * 100,
                                                                                              'Recall', pr[-1, 1] * 100,
                                                                                              'FP', fp, 'FN', fn)
    if print_info:
        log_file = open(os.path.join(opt.root_dir, 'result', opt.exp_id), 'a+')
        log_file.write('\nTask_{}ID_head({}) IOU_thres_GT_{}({})\n'.format(model_name, ID_head, IOU_thres_GT, epoch))
        print('\nTask_{}ID_head({}) IOU_thres_GT_{}({})\n'.format(model_name, ID_head, IOU_thres_GT, epoch))
        log_file.write(total_video_str)
        log_file.write(video_str)
        log_file.close()
        print(total_video_str)
        print(video_str)
    return ap


if __name__ == "__main__":
    opt = opts().parse()
    opt.task = 'frameAP'

    if not os.path.exists(os.path.join(opt.root_dir, 'result')):
        os.system("mkdir -p '" + os.path.join(opt.root_dir, 'result') + "'")
    # if opt.task == 'BuildTubes':
    #     BuildTubes(opt)
    elif opt.task == 'frameAP':
        frameAP(opt, mode='train')
        # frameAP(opt, mode='val')
