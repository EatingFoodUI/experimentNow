import cv2
import argparse
import os
import sys

sys.path.append('../')

import numpy as np
from tqdm import tqdm
import pickle
import shutil
import multiprocessing
import time
from MOC_utils.utils import ECC, savePickle, readPickle
import sys


class GTSingleParser:
    def __init__(self, video_root, iters, eps):
        self.video_root = video_root
        meta_info = open(os.path.join(video_root, 'seqinfo.ini')).read()
        self.frame_num = int(meta_info[meta_info.find('seqLength') + 10:meta_info.find('\nimWidth')])

        self.iterate_mat_file = os.path.join(self.video_root, 'iterate_warp_mat.pkl')
        self.direct_mat_file = os.path.join(self.video_root, 'direct_warp_mat.pkl')
        self.iters = iters
        self.eps = eps
        self.iterate_warp_mat = {}
        self.direct_warp_mat = {}
        self.video = video_root.split('/')[-1]

        self.forward_frames = opt.forward_frames
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        self.frame_stride = round((opt.clip_len * frame_rate - 1) / (opt.forward_frames - 1))

    def getDirectWarpMat(self):
        if os.path.exists(self.direct_mat_file):
            self.direct_warp_mat = readPickle(self.direct_mat_file)

        for cen_frame_id in tqdm(range(1 + self.frame_stride * (self.forward_frames // 2),
                                       self.frame_num - self.frame_stride * (self.forward_frames // 2) + 1)):
            cen_img_path = os.path.join(self.video_root, 'img1/{:0>6}.jpg'.format(cen_frame_id))
            cen_img = cv2.imread(cen_img_path)

            for i in range(-1 * (self.forward_frames // 2), self.forward_frames // 2 + 1):
                if i == 0:
                    continue
                else:
                    cur_frame_id = cen_frame_id + i * self.frame_stride
                    cur_img_path = os.path.join(self.video_root, 'img1/{:0>6}.jpg'.format(cur_frame_id))
                    cur_img = cv2.imread(cur_img_path)

                    k = (cur_frame_id, cen_frame_id)
                    if k not in self.direct_warp_mat:
                        # 从cur往center align
                        try:
                            self.direct_warp_mat[k] = ECC(cur_img, cen_img, max_iter=self.iters, eps=self.eps)
                        except:
                            print(k)
                            print('Maybe the gap is too big!')
                            pass
                    k = (cen_frame_id, cur_frame_id)
                    if k not in self.direct_warp_mat:
                        # 从center往cur align
                        try:
                            self.direct_warp_mat[k] = ECC(cen_img, cur_img, max_iter=self.iters, eps=self.eps)
                        except:
                            print(k)
                            print('Maybe the gap is too big!')
                            pass
        savePickle(self.direct_warp_mat, self.direct_mat_file)

    def getIterateWarpMat(self):
        # [2,frame_num]
        if os.path.exists(self.iterate_mat_file):
            self.iterate_warp_mat = readPickle(self.iterate_mat_file)
        for frame_id in tqdm(range(2, self.frame_num + 1)):
            if frame_id not in self.iterate_warp_mat:
                last_img_path = os.path.join(self.video_root, 'img1/{:0>6}.jpg'.format(frame_id - 1))
                cur_img_path = os.path.join(self.video_root, 'img1/{:0>6}.jpg'.format(frame_id))
                last_img = cv2.imread(last_img_path)
                cur_img = cv2.imread(cur_img_path)

                # 从后往前align
                time1 = time.time()
                self.iterate_warp_mat[frame_id] = {}
                warp_mat = ECC(cur_img, last_img, max_iter=self.iters, eps=self.eps)
                self.iterate_warp_mat[frame_id][-1] = warp_mat
                # 从前往后align
                warp_mat = ECC(last_img, cur_img, max_iter=self.iters, eps=self.eps)
                self.iterate_warp_mat[frame_id][1] = warp_mat

        savePickle(self.iterate_warp_mat, self.iterate_mat_file)

    def __len__(self):
        l = self.frame_num
        return l

    def clear(self):
        try:
            path = self.iterate_mat_file
            if os.path.exists(path):
                os.system('echo 1 | sudo -S rm -f {}'.format(path))
                print('rm -f ', path)
                shutil.rmtree(path)
        except:
            pass

    def collect_iterate_Info(self):
        self.iterate_warp_mat = readPickle(self.iterate_mat_file)
        return self.video, self.frame_num, self.iterate_warp_mat

    def collect_direct_Info(self):
        self.direct_warp_mat = readPickle(self.direct_mat_file)
        return self.video, self.frame_num, self.direct_warp_mat


class GTParser:
    def __init__(self, mot_root,
                 arg,
                 mode='train',
                 det='FRCNN',
                 ):
        # analsis all the folder in mot_root
        # 1. get all the folders
        mot_root = os.path.join(mot_root, mode)
        vlist = [5, 10, 11, 13]
        MC_video = [os.path.join(mot_root, 'MOT17-{:0>2}-{}'.format(i, det)) for i in vlist]

        # 2. create single parser
        self.parsers = [GTSingleParser(folder,
                                       iters=arg.number_of_iterations,
                                       eps=arg.termination_eps) for folder in MC_video]
        # 3. collect info
        self.collect_iterate_pkl = os.path.join(mot_root, 'iterate_warp_mat.pkl')
        self.collect_direct_pkl = os.path.join(mot_root, 'direct_warp_mat.pkl')

    def __len__(self):
        # 4. len
        self.lens = [len(p) for p in self.parsers]  # [frame num of a video]
        self.len = sum(self.lens)
        return self.len

    def runV2(self):
        print('Running')
        pool = multiprocessing.Pool(processes=len(self.parsers))
        pool_list = []
        for index in range(len(self.parsers)):
            self.parsers[index].getDirectWarpMat()
            # pool_list.append(pool.apply_async(self.parsers[index].getDirectWarpMat, ()))
        # for p in tqdm(pool_list, ncols=20):
        #     p.get()
        # pool.close()
        # pool.join()

        # collect info to a file
        if True:
            print('Save to ' + self.collect_direct_pkl)
            data = {
                'videos': [],
                'nframes': {},
                'direct_warp_mat': {},
            }
            for parser in self.parsers:
                video, nframes, direct_warp_mat = parser.collect_direct_Info()
                _, _, iterate_warp_mat = parser.collect_iterate_Info()
                for k in iterate_warp_mat:
                    if k not in direct_warp_mat:
                        direct_warp_mat[k] = iterate_warp_mat[k]

                savePickle(direct_warp_mat, parser.direct_mat_file)
                data['videos'].append(video)
                data['nframes'][video] = nframes
                data['direct_warp_mat'][video] = direct_warp_mat

            savePickle(data, self.collect_direct_pkl)

    def run(self):
        print('Running')
        pool = multiprocessing.Pool(processes=len(self.parsers))
        pool_list = []
        clear = False
        # clear = True
        if clear:
            for parser in self.parsers:
                parser.clear()
        for index in range(len(self.parsers)):
            self.parsers[index].getIterateWarpMat()
            # pool_list.append(pool.apply_async(self.parsers[index].getIterateWarpMat, ()))
        # for p in tqdm(pool_list, ncols=20):
        #     p.get()
        # pool.close()
        # pool.join()

        # collect info to a file
        if True:
            print('Save to ' + self.collect_iterate_pkl)
            data = {
                'videos': [],
                'nframes': {},
                'iterate_warp_mat': {},
            }
            for parser in self.parsers:
                video, nframes, iterate_warp_mat = parser.collect_iterate_Info()
                data['videos'].append(video)
                data['nframes'][video] = nframes
                data['iterate_warp_mat'][video] = iterate_warp_mat

            savePickle(data, self.collect_iterate_pkl)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--mot_root', default='../../data', type=str, help="mot data root")
    arg_parser.add_argument('--number_of_iterations', default=1000, type=int,
                            help="frame number to extract bezier curve")
    arg_parser.add_argument('--termination_eps', default=1e-6, type=float, help="frame number to extract bezier curve")
    arg_parser.add_argument('--forward_frames', type=int, default=7,
                            help='length of sparse track tublet')
    arg_parser.add_argument('--clip_len', default=1.5, type=float, help="sparse clip time interval")
    opt = arg_parser.parse_args()
    opt.clip_len = 1.5
    parser = GTParser(mot_root=opt.mot_root, arg=opt)
    # parser.run()
    parser.runV2()
