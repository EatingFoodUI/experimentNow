import os
import os.path as osp
import cv2
import motmetrics as mm
import numpy as np
import torch
import logging

from opts import opts
from tracking_utils.utils import mkdir_if_missing
from tracking_utils.evaluation import Evaluator

from tracking_utils.log import logger
from tracking_utils.timer import Timer

import datasets.dataset.MOT20 as MOT20
import datasets.dataset.MOT17 as MOT17

from tracker.multitracker import JDETracker
from tracking_utils import visualization as vis
from ACT_utils.ACT_utils import iou2d
import random


def write_results(filename, results, data_type):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},1,-1,-1,-1\n'
    else:
        raise ValueError(data_type)

    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids in results:
            for tlwh, track_id in zip(tlwhs, track_ids):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                x2, y2 = x1 + w, y1 + h
                line = save_format.format(frame=frame_id, id=track_id, x1=x1, y1=y1, x2=x2, y2=y2, w=w, h=h)
                f.write(line)
    logger.info('save results to {}'.format(filename))


def eval_seq(opt, dataloader, data_type, result_filename, save_dir=None, show_image=False, frame_rate=30):
    if save_dir:
        mkdir_if_missing(save_dir)

    tracker = JDETracker(opt, frame_rate=frame_rate)

    timer = Timer()
    results = []

    for iter, data in enumerate(dataloader):
        if len(data.keys()) == 1:
            continue

        frame_id = int(data['frame_id'][0])
        # print("start_id:{},iter:{}".format(frame_id,iter))
        # if frame_id % 10 == 0:
        if iter % 10 == 0:
            logger.info('iter {}, Processing frame {} ({:.2f} fps)'.format(iter, frame_id,
                                                                  opt.sliding_stride / max(1e-5, timer.average_time)))

        # run tracking
        images0 = data['images0']
        images = data['images']
        meta = data['meta']
        # blob = [torch.from_numpy(img).float().cuda().unsqueeze(0) for img in images]
        timer.tic()
        # blob = images
        sliding_stride = opt.sliding_stride
        start = 0

        last = frame_id + 2 * sliding_stride
        if iter == len(dataloader) - 2:
            last = frame_id + sliding_stride

        if iter == len(dataloader) - 1:
            frame_stride = int(data['meta']['frame_stride'][0])
            sliding_stride = frame_stride * (opt.forward_frames - 1) + 1
            start = last - frame_id

        online_tracklets = tracker.update(images, meta, frame_id)
        timer.toc()

        for i in range(start, sliding_stride):
            online_tlwhs = []
            online_ids = []
            # l = det[frame_id + i]
            # target_boxes = []
            # for (bx, by, bw, bh, mark, label, _) in l:
            #     x1, y1, x2, y2 = bx, by, bx + bw, by + bh
            #     target_boxes.append([x1, y1, x2, y2])
            # target_boxes = np.array(target_boxes)
            # print(len(online_tracklets))
            for t in online_tracklets:
                tlwh = t.tlwh[i]
                tid = t.track_id

                vertical = tlwh[2] / (tlwh[3] + 1e-6) > 1.6

                pred_box = np.array([tlwh[0], tlwh[1],
                                     tlwh[0] + tlwh[2], tlwh[1] + tlwh[3]], dtype=np.int16)

                if pred_box[0] != 0 and pred_box[1] != 0 and pred_box[2] != 0 and pred_box[3] != 0:
                    vertical = tlwh[2] / (tlwh[3] + 1e-6) > 1.2
                    area_legal = tlwh[2] * tlwh[3] > 900
                    if vertical or not area_legal:
                        continue

                opt.min_box_area = 400
                if tlwh[2] * tlwh[3] > opt.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)

            results.append((frame_id + i, online_tlwhs, online_ids))

            if save_dir is not None:
                online_im = vis.plot_tracking(images0[i][0], online_tlwhs, online_ids, frame_id=frame_id + i,
                                              fps=opt.sliding_stride / timer.average_time)
                cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id + i)), online_im)
                # print(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id+i)))

    # save results
    write_results(result_filename, results, data_type)
    return  timer.average_time, timer.calls


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def worker_init_fn(dump):
    set_seed(317)


def main(opt, data_root='/data/MOT20/train', det_root=None, seqs=('MOT20-01',), exp_name='demo', mode='test',
         save_images=False, save_videos=False, show_image=False):
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)

    data_type = 'mot'
    # run tracking
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []

    dataset = MOT20.eval_MOT20
    # dataset = MOT20.eval_MOT20_public_det
    # dataset = MOT17.eval_MOT17_public_det
    opt = opts().update_dataset(opt, dataset)
    for seq in seqs:
        # if mode == 'val':
        #     _mode = 'train'
        #     det_path = os.path.join(opt.root_dir, 'data', _mode, seq, 'det/det.txt')
        # elif mode == 'test':
        #     _mode = 'test'
        #     det_path = os.path.join(opt.root_dir, 'data', _mode, seq, 'det/det.txt')
        #
        # det_arr = np.loadtxt(det_path, dtype=np.float64, delimiter=',')
        # det = {}
        # for fid, tid, bx, by, bw, bh, mark, label, vis, _ in det_arr:
        #     fid = int(fid)
        #     if fid not in det:
        #         det[fid] = []
        #
        #     det[fid].append((bx, by, bw, bh, mark, label, vis))

        output_dir = os.path.join(data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))

        dataloader = dataset(opt, mode=mode, vname=seq)

        dataloader = torch.utils.data.DataLoader(
            dataloader,
            batch_size=1,
            shuffle=False,
            # 2
            num_workers=0,
            pin_memory=opt.pin_memory,
            drop_last=False,
            worker_init_fn=worker_init_fn)

        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        meta_info = open(os.path.join(data_root, seq, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        ta, tc = eval_seq(opt, dataloader, data_type, result_filename,
                              save_dir=output_dir, show_image=show_image, frame_rate=frame_rate)

        # n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)

        # eval
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))

    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(all_time, opt.sliding_stride / avg_time))

    # get summary
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(result_root, 'summary_{}.xlsx'.format(exp_name)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().parse()

    if opt.val_mot20:
        '''
                      MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
        '''
        seqs_str = '''
                      MOT20-01
                      MOT20-02
                      MOT20-03
                      MOT20-05
                    '''
        data_root = os.path.join(opt.root_dir, 'data/train')
        mode = 'val'
    elif opt.test_mot20:
        '''
        HIE20-25
        HIE20-26
        HIE20-27
        HIE20-28
        HIE20-29
        HIE20-30
        HIE20-31
        HIE20-32
        '''
        seqs_str = '''               
        MOT20-04
        MOT20-06
        MOT20-07
        MOT20-08
                '''
        data_root = os.path.join(opt.root_dir, 'data/test')
        mode = 'test'

    elif opt.test_mot17:
        '''
        MOT17-01-DPM
        MOT17-03-DPM
        MOT17-07-DPM
        MOT17-08-DPM
        MOT17-12-DPM
        MOT17-14-DPM 
        MOT17-06-DPM
        
        MOT17-03-FRCNN
        MOT17-07-FRCNN
        MOT17-08-FRCNN
        MOT17-12-FRCNN
        MOT17-14-FRCNN
        MOT17-06-FRCNN
        '''
        seqs_str = '''
            MOT17-13-FRCNN
                '''
        # seqs_str = '''
        #
        #     MOT17-02-FRCNN
        #
        # '''
        #        MOT17-06-FRCNN
        #        MOT17-03-FRCNN
        #        MOT17-01-FRCNN

        # data_root = os.path.join(opt.root_dir, '../../FairMOT-master/MOT17/images/test')
        # 使用训练集
        data_root = os.path.join(opt.root_dir, 'data/test')
        mode = 'test'

    seqs = [seq.strip() for seq in seqs_str.split()]

    main(opt,
         data_root=data_root,
         # seqs=('MOT20-07'),
         seqs=seqs,
         exp_name='MOT17_refine_public_track',
         mode=mode,
         show_image=False,
         save_images=True,
         save_videos=False)
