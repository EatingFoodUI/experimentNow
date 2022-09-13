import sys
import cv2
import shutil
import glob
import os.path as osp
import numpy as np
import os
import time
from opts import opts
from MOC_utils.utils import ECC, readPickle, warp_bbox_iterate, warp_bbox_direct

if __name__ == '__main__':
    opt = opts().parse()
    ID = 2
    img_dir_num = 5
    forward_frames, clip_len = opt.forward_frames, opt.clip_len

    video_root = '../data/train/MOT17-{:0>2}-FRCNN'.format(img_dir_num)
    meta_info = open(os.path.join(video_root, 'seqinfo.ini')).read()
    frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
    frame_stride = round((opt.clip_len * frame_rate - 1) / (opt.forward_frames - 1))
    iw = int(meta_info[meta_info.find('imWidth') + 8:meta_info.find('\nimHeight')])
    ih = int(meta_info[meta_info.find('imHeight') + 9:meta_info.find('\nimExt')])

    data = readPickle('../data/train/iterate_warp_mat.pkl')
    vname = video_root.split('/')[-1]
    nframes = data['nframes'][vname]
    warpmats = data['iterate_warp_mat'][vname]

    delete = True
    # delete = False
    if delete:
        imgs = osp.join(video_root, 'img2/*')
        for img in glob.glob(imgs):
            if os.path.exists(img):
                os.remove(img)

    gt_ped_txt = os.path.join(video_root, 'gt/gt.txt')
    gt = np.loadtxt(gt_ped_txt, dtype=np.float64, delimiter=',')
    least_fid = 100000
    max_fid = 0
    for fid, tid, x, y, w, h, mark, label, vis in gt:
        if mark == 0 or not label == 1:
            continue

        fid = int(fid)
        tid = int(tid)
        if not tid == ID:
            continue
        if fid < least_fid:
            least_fid = fid
        if fid > max_fid:
            max_fid = fid
        save_images = delete
        if save_images:
            img_path = os.path.join(video_root, 'img2/{:0>6}.jpg'.format(fid))
            if not osp.exists(img_path):
                img_path = os.path.join(video_root, 'img1/{:0>6}.jpg'.format(fid))
            img = cv2.imread(img_path)

            gt_bbox = np.array([x, y, x + w, y + h], dtype=np.int64)

            gt_bbox[0] = np.clip(gt_bbox[0], 0, iw - 1)
            gt_bbox[1] = np.clip(gt_bbox[1], 0, ih - 1)
            gt_bbox[2] = np.clip(gt_bbox[2], 0, iw - 1)
            gt_bbox[3] = np.clip(gt_bbox[3], 0, ih - 1)

            cv2.rectangle(img, (int(gt_bbox[0]), int(gt_bbox[1])), (int(gt_bbox[2]), int(gt_bbox[3])), (0, 255, 255), 2)
            # cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (255, 0, 0), 2)
            cv2.putText(img, '{:d}'.format(tid), (gt_bbox[0], gt_bbox[1]), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255),
                        1)
            cv2.line(img, (int((gt_bbox[0] + gt_bbox[2]) / 2), 0), (int((gt_bbox[0] + gt_bbox[2]) / 2), ih),
                     (0, 255, 255), 2, 4)
            cv2.line(img, (0, int((gt_bbox[1] + gt_bbox[3]) / 2)), (iw, int((gt_bbox[1] + gt_bbox[3]) / 2)),
                     (0, 255, 255), 2, 4)
            img_path = os.path.join(video_root, 'img2/{:0>6}.jpg'.format(fid))
            cv2.imwrite(img_path, img)

    # exit()
    # select a couple of images to eliminate CM
    # static person, moving person
    # big time gap, small time gap
    # 1.static person(517-597)
    print('prepare bbox...')
    bbox = {}
    MAX_IDX = nframes + 1
    for frame_id in range(least_fid, max_fid + 1):
        for fid, tid, x, y, w, h, mark, label, vis in gt:
            if mark == 0 or not label == 1:
                continue

            fid = int(fid)
            tid = int(tid)

            if fid == frame_id and tid == ID:
                k = frame_id
                if k not in bbox:
                    bbox[k] = np.array([x, y, x + w, y + h], dtype=np.int64)

    start_frame_range = range(1, nframes + 1, (forward_frames - 1) * frame_stride + 1)

    print('start aligning...')
    for start_frame_id in start_frame_range:
        max_index = start_frame_id + (forward_frames - 1) * frame_stride
        if not (least_fid <= start_frame_id <= max_fid):
            continue
        if not (least_fid <= max_index <= max_fid):
            continue

        base_frame_id = start_frame_id + 3 * frame_stride
        base_img_path = os.path.join(video_root, 'img1/{:0>6}.jpg'.format(base_frame_id))
        base_img = cv2.imread(base_img_path)
        for k in range(forward_frames):
            cur_frame_id = start_frame_id + k * frame_stride
            if base_frame_id == cur_frame_id:
                base_bak_path = os.path.join(video_root, 'img2/{:0>6}.jpg'.format(base_frame_id))
                base_img_bak = cv2.imread(base_bak_path)
                cv2.putText(base_img_bak, '{}'.format('C'), (iw // 2, ih // 2),
                            cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 255), 2)

                cv2.imwrite(base_bak_path, base_img_bak)

                continue
            if cur_frame_id in bbox:
                align_bbox = warp_bbox_iterate(warpmats, base_frame_id, cur_frame_id, bbox[cur_frame_id])
            else:
                continue

            cur_img_path = os.path.join(video_root, 'img1/{:0>6}.jpg'.format(cur_frame_id))
            cur_img = cv2.imread(cur_img_path)
            warp_mat = ECC(cur_img, base_img)
            # warp_mat = ECC(base_img, cur_img)

            align_bbox1 = warp_bbox_direct(bbox[cur_frame_id], warp_mat)
            print('gap:')
            print(cur_frame_id - base_frame_id)
            print('direct warp diff:')
            print(align_bbox1 - bbox[base_frame_id])
            print('indirect warp diff:')
            print(align_bbox - bbox[base_frame_id])
            print('-' * 80)

            cur_bak_img_path = os.path.join(video_root, 'img2/{:0>6}.jpg'.format(cur_frame_id))
            bak_img = cv2.imread(cur_bak_img_path)
            cv2.rectangle(bak_img, (align_bbox1[0], align_bbox1[1]), (align_bbox1[2], align_bbox1[3],),
                          (255, 255, 0), 2)
            bx1 = int((align_bbox1[0] + align_bbox1[2]) / 2)
            by1 = int((align_bbox1[1] + align_bbox1[3]) / 2)
            cv2.line(bak_img, (bx1, 0), (bx1, ih),
                     (255, 255, 0), 2, 4)
            cv2.line(bak_img, (0, by1), (iw, by1),
                     (255, 255, 0), 2, 4)
            cv2.putText(bak_img, '{}'.format('direct'), (int(align_bbox1[0]), int(align_bbox1[1])),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

            bx = int((align_bbox[0] + align_bbox[2]) / 2)
            by = int((align_bbox[1] + align_bbox[3]) / 2)
            cv2.rectangle(bak_img, (align_bbox[0], align_bbox[1]), (align_bbox[2], align_bbox[3]), (0, 0, 255), 2)
            cv2.line(bak_img, (bx, 0), (bx, ih),
                     (0, 0, 255), 2, 4)

            cv2.line(bak_img, (0, by), (iw, by),
                     (0, 0, 255), 2, 4)
            cv2.putText(bak_img, '{}'.format('indirect'), (int(align_bbox[0]), int(align_bbox[1])),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(bak_img, '{}'.format(cur_frame_id - base_frame_id), (iw-100, ih-100),
                        cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 255), 2)
            base_bak_path = os.path.join(video_root, 'img2/{:0>6}.jpg'.format(base_frame_id))
            base_img_bak = cv2.imread(base_bak_path)

            img_aligned = np.zeros((2 * ih, 2 * iw, 3), dtype=np.uint8)
            img_aligned[0:ih, 0:iw] = bak_img
            img_aligned[0:ih, iw:2 * iw] = base_img_bak
            img_aligned[ih:2 * ih, 0:iw] = base_img_bak
            cv2.imwrite(cur_bak_img_path, img_aligned)
