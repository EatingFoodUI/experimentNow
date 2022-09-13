import os.path as osp
import os
import numpy as np
from tqdm import tqdm

seq_root = '../../data/train'

seqs = [s for s in os.listdir(seq_root)]

tid_curr = 0
tid_last = -1
det = 'FRCNN'
for seq in tqdm(seqs, ncols=100, colour='RED'):
    if seq == 'FRCNN_tracks.pkl':
        continue
    # if det not in seq:
    #     continue
    print('\n', seq)
    gt_txt = osp.join(seq_root, seq, 'gt', 'gt.txt')
    gt_ped_txt = osp.join(seq_root, seq, 'gt', 'gt_ped.txt')
    if os.path.exists(gt_ped_txt):
        os.remove(gt_ped_txt)

    gt = np.loadtxt(gt_txt, dtype=np.float64, delimiter=',')
    for fid, tid, x, y, w, h, mark, label, vis in gt:
        if mark == 0 or not label == 1:
            continue

        fid = int(fid)
        tid = int(tid)
        if not tid == tid_last:
            tid_curr += 1
            tid_last = tid

        label_str = '{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:d},{:.5f}\n'.format(
            int(fid), int(tid_curr), int(x), int(y), int(w), int(h), int(mark), int(label), vis)

        with open(gt_ped_txt, 'a') as f:
            f.write(label_str)

print('mot17 has {} people'.format(tid_curr))
