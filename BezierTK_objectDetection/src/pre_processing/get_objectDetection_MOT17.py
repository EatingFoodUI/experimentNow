# 将mot17数据集转换成目标检测格式的数据集
import argparse
import os
import pandas as pd
import sys

sys.path.append('../')
from datasets.Parsers.structures import Node, Tracks
import numpy as np
import glob
import cv2
from scipy.special import comb as n_over_k
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.text import Annotation
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from numpy.random import rand
import pickle
import shutil
import multiprocessing
from tqdm import tqdm
from ACT_utils.ACT_utils import tubelet_in_out_clip, clip_has_tublet
import time


def savePickle(gt_curves, path):
    with open(path, 'wb') as f:
        pickle.dump(gt_curves, f)
        f.close()


def readPickle(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
        f.close()
        return d


def findIndex(tube_frame_id_in_clip, frame_id_in_tube):
    frame_id_in_tube = frame_id_in_tube.tolist()
    l = []
    for frame_id in tube_frame_id_in_clip:
        l.append(frame_id_in_tube.index(frame_id))
    return np.array(l)

def drawBboxCurve(fig, ctp, tublet):
    '''
    [all tublets]draw only tublet and bezier curve
    :param fig:
    :param ctp:
    :param tublet: sparse tublet, [fid, x1,y1,x2,y2]
    :return:
    '''
    # create figure
    # fig = plt.figure(figsize=(16, 12))
    # print(tublet[:,0])
    ax = fig.gca(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('y')
    # draw the tublet
    toPoints = lambda xn, franme_idn, yn: list(zip(xn, franme_idn, yn))
    for i in range(tublet.shape[0]):
        xn = [tublet[i, 1], tublet[i, 3], tublet[i, 3], tublet[i, 1], ]
        yn = [tublet[i, 2], tublet[i, 2], tublet[i, 4], tublet[i, 4], ]
        franme_idn = [tublet[i, 0]] * 4
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        # xyzn = zip(xn,yn,franme_idn)
        p = toPoints(xn, franme_idn, yn)
        # print(p)
        # exit()
        segments = [(p[s], p[e]) for s, e in edges]
        ax.scatter(xn, franme_idn, yn, marker='o', c='k', s=1)
        # center
        cx = [(tublet[i, 1] + tublet[i, 3]) / 2]
        cy = [(tublet[i, 2] + tublet[i, 4]) / 2]
        cz = [tublet[i, 0]]
        ax.scatter(cx, cz, cy, marker='+', c='k', s=64, linewidths=20)

        # plot edges
        edge_col = Line3DCollection(segments, colors='k', lw=1.0)
        ax.add_collection3d(edge_col)

    # draw the curve
    x0, x1, x2, x3 = ctp[0, 0], ctp[1, 0], ctp[2, 0], ctp[3, 0]
    y0, y1, y2, y3 = ctp[0, 1], ctp[1, 1], ctp[2, 1], ctp[3, 1]
    frame0, frame1, frame2, frame3 = ctp[0, 2], ctp[1, 2], ctp[2, 2], ctp[3, 2]
    func = lambda x0, x1, x2, x3: [(1 - t) * ((1 - t) * ((1 - t) * x0 + t * x1) + t * ((1 - t) * x1 + t * x2)) + t * (
            (1 - t) * ((1 - t) * x1 + t * x2) + t * ((1 - t) * x2 + t * x3)) for t in np.linspace(0, 1, 128).tolist()]
    bezier_x = func(x0, x1, x2, x3)
    bezier_y = func(y0, y1, y2, y3)
    bezier_frame_id = func(frame0, frame1, frame2, frame3)
    p = toPoints(bezier_x, bezier_frame_id, bezier_y)
    edges = [(i, i + 1) for i in range(len(bezier_x) - 1)]
    segments = [(p[s], p[e]) for s, e in edges]
    ax.scatter(bezier_x, bezier_frame_id, bezier_y, marker='o', s=0.1)
    edge_col = Line3DCollection(segments, colors='r', lw=2.0)
    ax.add_collection3d(edge_col)

def drawBboxCurveV2(fig, ctp, tube, s, e, frame_stride):
    '''
    [single tublet]
    :param fig:
    :param ctp:
    :param tube:
    :param s:
    :param e:
    :param frame_stride:
    :return:
    '''
    e += 1
    _slice = slice(s, e, frame_stride)
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('y')
    sparse_tubelet = tube[_slice]
    # sparse_tubelet - bboxes
    toPoints = lambda xn, franme_idn, yn: list(zip(xn, franme_idn, yn))
    for i in range(sparse_tubelet.shape[0]):
        xn = [sparse_tubelet[i, 1], sparse_tubelet[i, 3], sparse_tubelet[i, 3], sparse_tubelet[i, 1], ]
        yn = [sparse_tubelet[i, 2], sparse_tubelet[i, 2], sparse_tubelet[i, 4], sparse_tubelet[i, 4], ]
        franme_idn = [sparse_tubelet[i, 0]] * 4
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        # edges = [(0, 1), (1, 2), (2, 3),(3,4), (4, 0)]
        # xyzn = zip(xn,yn,franme_idn)
        p = toPoints(xn, franme_idn, yn)
        # print(p)
        # exit()
        segments = [(p[s], p[e]) for s, e in edges]
        ax.scatter(xn, franme_idn, yn, marker='o', c='k', s=1)
        # center
        cx = [(sparse_tubelet[i, 1] + sparse_tubelet[i, 3]) / 2]
        cy = [(sparse_tubelet[i, 2] + sparse_tubelet[i, 4]) / 2]
        cz = [sparse_tubelet[i, 0]]
        ax.scatter(cx, cz, cy, marker='+', c='k', s=64, linewidths=1)

        # plot edges
        edge_col = Line3DCollection(segments, colors='k', lw=1.0)
        ax.add_collection3d(edge_col)

    # ctp
    xn = ctp[:, 0].tolist()
    yn = ctp[:, 1].tolist()
    franme_idn = ctp[:, 2].tolist()
    # edges = [(0, 1), (1, 2), (2, 3)]
    # 改动
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    p = toPoints(xn, franme_idn, yn)
    segments = [(p[s], p[e]) for s, e in edges]
    ax.scatter(xn, franme_idn, yn, marker='o', c='b', s=16)
    edge_col = Line3DCollection(segments, linestyles='dashed', colors='b', lw=2.0)
    ax.add_collection3d(edge_col)

    # center track
    cx = ((tube[s:e, 1] + tube[s:e, 3]) / 2).tolist()
    cy = ((tube[s:e, 2] + tube[s:e, 4]) / 2).tolist()
    cz = tube[s:e, 0].tolist()
    p = toPoints(cx, cz, cy)
    edges = [(i, i + 1) for i in range(len(cz) - 1)]
    segments = [(p[s], p[e]) for s, e in edges]
    ax.scatter(cx, cz, cy, marker='o', s=0.1)
    edge_col = Line3DCollection(segments, colors='k', lw=2.0)
    ax.add_collection3d(edge_col)

    # 改动
    # t0 = time.time()
    # bezier_curve
    # x0, x1, x2, x3 = ctp[0, 0], ctp[1, 0], ctp[2, 0], ctp[3, 0]
    # y0, y1, y2, y3 = ctp[0, 1], ctp[1, 1], ctp[2, 1], ctp[3, 1]
    # frame0, frame1, frame2, frame3 = ctp[0, 2], ctp[1, 2], ctp[2, 2], ctp[3, 2]
    x0, x1, x2, x3, x4, x5 = ctp[0, 0], ctp[1, 0], ctp[2, 0], ctp[3, 0], ctp[4, 0], ctp[5, 0]
    y0, y1, y2, y3, y4, y5 = ctp[0, 1], ctp[1, 1], ctp[2, 1], ctp[3, 1], ctp[4, 1], ctp[5, 1]
    frame0, frame1, frame2, frame3, frame4, frame5 = ctp[0, 2], ctp[1, 2], ctp[2, 2], ctp[3, 2], ctp[4, 2], ctp[5, 2]

    # 改动
    # func = lambda x0, x1, x2, x3: [(1 - t) * ((1 - t) * ((1 - t) * x0 + t * x1) + t * ((1 - t) * x1 + t * x2)) + t * (
    #         (1 - t) * ((1 - t) * x1 + t * x2) + t * ((1 - t) * x2 + t * x3)) for t in np.linspace(0, 1, 128).tolist()]
    # func = lambda x0, x1, x2, x3, x4, x5: [(bezierCurve(x0, t, 0, 5)
    #                                         + bezierCurve(x1, t, 1, 5)
    #                                         + bezierCurve(x2, t, 2, 5)
    #                                         + bezierCurve(x3, t, 3, 5)
    #                                         + bezierCurve(x4, t, 4, 5)
    #                                         + bezierCurve(x5, t, 5, 5)) for t in np.linspace(0, 1, 128).tolist()]
    func = lambda x0, x1, x2, x3, x4, x5: [
        x0 * (1 - t) ** 5 +
        x1 * 5 * t * (1 - t) ** 4 +
        x2 * 10 * t ** 2 * (1 - t) ** 3 +
        x3 * 10 * t ** 3 * (1 - t) ** 2 +
        x4 * 5 * t ** 4 * (1 - t) +
        x5 * t ** 5
        for t in np.linspace(0, 1, 128).tolist()]

    # 改动
    bezier_x = func(x0, x1, x2, x3, x4, x5)
    bezier_y = func(y0, y1, y2, y3, y4, y5)
    bezier_frame_id = func(frame0, frame1, frame2, frame3, frame4, frame5)

    p = toPoints(bezier_x, bezier_frame_id, bezier_y)
    edges = [(i, i + 1) for i in range(len(bezier_x) - 1)]
    segments = [(p[s], p[e]) for s, e in edges]
    ax.scatter(bezier_x, bezier_frame_id, bezier_y, marker='o', s=0.1)
    edge_col = Line3DCollection(segments, colors='r', lw=2.0)
    ax.add_collection3d(edge_col)

def drawBboxCurveV2(fig, ctp, tube, s, e, frame_stride):
    '''
    [single tublet]
    :param fig:
    :param ctp:
    :param tube:
    :param s:
    :param e:
    :param frame_stride:
    :return:
    '''
    e += 1
    _slice = slice(s, e, frame_stride)
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('y')
    sparse_tubelet = tube[_slice]
    # sparse_tubelet - bboxes
    toPoints = lambda xn, franme_idn, yn: list(zip(xn, franme_idn, yn))
    for i in range(sparse_tubelet.shape[0]):
        xn = [sparse_tubelet[i, 1], sparse_tubelet[i, 3], sparse_tubelet[i, 3], sparse_tubelet[i, 1], ]
        yn = [sparse_tubelet[i, 2], sparse_tubelet[i, 2], sparse_tubelet[i, 4], sparse_tubelet[i, 4], ]
        franme_idn = [sparse_tubelet[i, 0]] * 4
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        # edges = [(0, 1), (1, 2), (2, 3),(3,4), (4, 0)]
        # xyzn = zip(xn,yn,franme_idn)
        p = toPoints(xn, franme_idn, yn)
        # print(p)
        # exit()
        segments = [(p[s], p[e]) for s, e in edges]
        ax.scatter(xn, franme_idn, yn, marker='o', c='k', s=1)
        # center
        cx = [(sparse_tubelet[i, 1] + sparse_tubelet[i, 3]) / 2]
        cy = [(sparse_tubelet[i, 2] + sparse_tubelet[i, 4]) / 2]
        cz = [sparse_tubelet[i, 0]]
        ax.scatter(cx, cz, cy, marker='+', c='k', s=64, linewidths=1)

        # plot edges
        edge_col = Line3DCollection(segments, colors='k', lw=1.0)
        ax.add_collection3d(edge_col)

    # ctp
    xn = ctp[:, 0].tolist()
    yn = ctp[:, 1].tolist()
    franme_idn = ctp[:, 2].tolist()
    # edges = [(0, 1), (1, 2), (2, 3)]
    # 改动
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    p = toPoints(xn, franme_idn, yn)
    segments = [(p[s], p[e]) for s, e in edges]
    ax.scatter(xn, franme_idn, yn, marker='o', c='b', s=16)
    edge_col = Line3DCollection(segments, linestyles='dashed', colors='b', lw=2.0)
    ax.add_collection3d(edge_col)

    # center track
    cx = ((tube[s:e, 1] + tube[s:e, 3]) / 2).tolist()
    cy = ((tube[s:e, 2] + tube[s:e, 4]) / 2).tolist()
    cz = tube[s:e, 0].tolist()
    p = toPoints(cx, cz, cy)
    edges = [(i, i + 1) for i in range(len(cz) - 1)]
    segments = [(p[s], p[e]) for s, e in edges]
    ax.scatter(cx, cz, cy, marker='o', s=0.1)
    edge_col = Line3DCollection(segments, colors='k', lw=2.0)
    ax.add_collection3d(edge_col)

    # 改动
    # t0 = time.time()
    # bezier_curve
    # x0, x1, x2, x3 = ctp[0, 0], ctp[1, 0], ctp[2, 0], ctp[3, 0]
    # y0, y1, y2, y3 = ctp[0, 1], ctp[1, 1], ctp[2, 1], ctp[3, 1]
    # frame0, frame1, frame2, frame3 = ctp[0, 2], ctp[1, 2], ctp[2, 2], ctp[3, 2]
    x0, x1, x2, x3, x4, x5 = ctp[0, 0], ctp[1, 0], ctp[2, 0], ctp[3, 0], ctp[4, 0], ctp[5, 0]
    y0, y1, y2, y3, y4, y5 = ctp[0, 1], ctp[1, 1], ctp[2, 1], ctp[3, 1], ctp[4, 1], ctp[5, 1]
    frame0, frame1, frame2, frame3, frame4, frame5 = ctp[0, 2], ctp[1, 2], ctp[2, 2], ctp[3, 2], ctp[4, 2], ctp[5, 2]

    # 改动
    # func = lambda x0, x1, x2, x3: [(1 - t) * ((1 - t) * ((1 - t) * x0 + t * x1) + t * ((1 - t) * x1 + t * x2)) + t * (
    #         (1 - t) * ((1 - t) * x1 + t * x2) + t * ((1 - t) * x2 + t * x3)) for t in np.linspace(0, 1, 128).tolist()]
    # func = lambda x0, x1, x2, x3, x4, x5: [(bezierCurve(x0, t, 0, 5)
    #                                         + bezierCurve(x1, t, 1, 5)
    #                                         + bezierCurve(x2, t, 2, 5)
    #                                         + bezierCurve(x3, t, 3, 5)
    #                                         + bezierCurve(x4, t, 4, 5)
    #                                         + bezierCurve(x5, t, 5, 5)) for t in np.linspace(0, 1, 128).tolist()]
    func = lambda x0, x1, x2, x3, x4, x5: [
        x0 * (1 - t) ** 5 +
        x1 * 5 * t * (1 - t) ** 4 +
        x2 * 10 * t ** 2 * (1 - t) ** 3 +
        x3 * 10 * t ** 3 * (1 - t) ** 2 +
        x4 * 5 * t ** 4 * (1 - t) +
        x5 * t ** 5
        for t in np.linspace(0, 1, 128).tolist()]

    # 改动
    bezier_x = func(x0, x1, x2, x3, x4, x5)
    bezier_y = func(y0, y1, y2, y3, y4, y5)
    bezier_frame_id = func(frame0, frame1, frame2, frame3, frame4, frame5)

    p = toPoints(bezier_x, bezier_frame_id, bezier_y)
    edges = [(i, i + 1) for i in range(len(bezier_x) - 1)]
    segments = [(p[s], p[e]) for s, e in edges]
    ax.scatter(bezier_x, bezier_frame_id, bezier_y, marker='o', s=0.1)
    edge_col = Line3DCollection(segments, colors='r', lw=2.0)
    ax.add_collection3d(edge_col)


class GTSingleParser:
    def __init__(self, folder,
                 min_visibility):
        # 1. get the gt path and image folder
        gt_file_path = os.path.join(folder, 'gt/gt.txt')
        self.folder = folder
        self.min_visibility = min_visibility

        self.tracks_file = os.path.join(self.folder, 'tracks.pkl')

        # 2. read the gt data
        gt_file = pd.read_csv(gt_file_path, header=None)
        gt_file = gt_file[gt_file[6] == 1]  # human class
        gt_file = gt_file[gt_file[8] > min_visibility]
        gt_group = gt_file.groupby(0)
        gt_group_keys = gt_group.indices.keys()
        self.max_frame_index = max(gt_group_keys)

        # important info
        # print(folder) # ../../data/train/MOT17-02-FRCNN
        meta_info = open(os.path.join(folder, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        imWidth = int(meta_info[meta_info.find('imWidth') + 8:meta_info.find('\nimHeight')])
        imHeight = int(meta_info[meta_info.find('imHeight') + 9:meta_info.find('\nimExt')])

        self.video = folder.split('/')[-1]
        print(self.video)
        print('frame_rate:', frame_rate)
        self.nframes = self.max_frame_index
        self.resolution = (imHeight, imWidth)

        self.gt_object = {}

        '''
        output:
        video:['MOT17-02-FRCNN']
        nframes:int
        resolution:{'MOT17-02-FRCNN':(h,w)}
        gt_object:{'MOT17-02-FRCNN':
                    {
                        frame_id:array([[id,x1,y1,x2,y2]...])
                    }
                }
        '''

        # key names
        '''
        0(frame)   1(id)      2           3       4        5       6(conf,7为1时为1)   7(cls)   8(vis)
        1          2          1338        418     167      379     1                  1        (0-1)
        '''
        # 3. update object
        self.tracks = Tracks()
        self.recorder = {}
        for frame_id in gt_group_keys:
            det = gt_group.get_group(frame_id).values
            person_ids = np.array(det[:, 1]).astype(int)
            bboxes = np.array(det[:, 2:6]).astype(int)
            bboxes[:, 2:4] += bboxes[:, :2]  # tl, br

            # if folder == '../../data/train/MOT17-02-FRCNN':
            #     boxes = bboxes
            #     for kk, b in enumerate(bboxes):
            #         fid = frame_id
            #         box = boxes[kk].astype(np.int32)
            #         print(box)
            #         img_path = os.path.join(folder, 'img2/{:0>6}.jpg'.format(fid))
            #         img = cv2.imread(img_path)
            #         cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            #         print(os.path.join(folder, 'img2/{:0>6}.jpg'.format(fid)))
            #         cv2.imwrite(os.path.join(folder, 'img2/{:0>6}.jpg'.format(fid)), img)
            # print(bboxes[0])
            # exit()
            self.recorder[frame_id - 1] = list()
            for person_id, bbox in zip(person_ids, bboxes):
                self.recorder[frame_id - 1].append(bbox)

        self.gt_object[self.video] = self.recorder

    def getDetection(self):
        tracks_file = self.tracks_file
        self.gt_curves = {}

        '''
        output:
        video:['MOT17-02-FRCNN']
        nframes:int
        resolution:{'MOT17-02-FRCNN':(h,w)}
        gt_curves:{'MOT17-02-FRCNN':
                    {id:
                        {track:array([frame_id,x1,y1,x2,y2])}
                        {stride_7_forward_10:[[x1,y1,t1...x4,y4,t4]]}
                    }
                }
        '''
        if os.path.exists(tracks_file):
            print('File already exists!!!')
            self.gt_curves = readPickle(tracks_file)
            self.gt_curves['video'] = self.video
            self.gt_curves['nframes'] = self.nframes
            self.gt_curves['resolution'] = self.resolution
            self.gt_curves['gt_object'] = self.gt_object
        else:
            print('File not exists')
            self.gt_curves['video'] = self.video
            self.gt_curves['nframes'] = self.nframes
            self.gt_curves['resolution'] = self.resolution
            self.gt_curves['gt_object'] = self.gt_object

        savePickle(self.gt_curves, self.tracks_file)

    def __len__(self):
        l = self.max_frame_index
        return l

    def val_curve_num(self):
        self.gt_curves = readPickle(self.tracks_file)
        valid_curve_num = 0
        for track in self.tracks.tracks:
            tube_id = track.id
            valid_curve_num += len(self.gt_curves[self.video][tube_id][self.curve_type])
        return valid_curve_num

    def clear(self):
        try:
            path = self.tracks_file
            if os.path.exists(path):
                os.system('echo 123123 | sudo -S rm -f {}'.format(path))
                print('rm -f ', path)
                shutil.rmtree(path)
        except:
            pass

    def collectInfo(self):
        '''
        output:
        video:['MOT17-02-FRCNN']
        nframes:int
        resolution:{'MOT17-02-FRCNN':(h,w)}
        gt_curves:{'MOT17-02-FRCNN':
                    {id:
                        {track:array([frame_id,x1,y1,x2,y2])}
                        {stride_7_forward_10:[[x1,y1,t1...x4,y4,t4]]}
                    }
                }
        '''
        self.gt_object = readPickle(self.tracks_file)
        return self.video, self.nframes, self.resolution, self.gt_object


class GTParser:
    def __init__(self, mot_root,
                 arg,
                 mode='train',
                 det='FRCNN',
                 ):
        # analsis all the folder in mot_root
        # 1. get all the folders
        mot_root = os.path.join(mot_root, mode)
        all_folders = sorted(
            [os.path.join(mot_root, i) for i in os.listdir(mot_root)
             if os.path.isdir(os.path.join(mot_root, i)) and i.find(det) != -1]
        )
        # all_folders = ['../../data/train/MOT20-01',
        #                '../../data/train/MOT20-02',
        #                '../../data/train/MOT20-03',
        #                '../../data/train/MOT20-05']
        all_folders = [# 移动 ￥
            '../../data/train/MOT17-02-FRCNN',
            '../../data/train/MOT17-04-FRCNN',
            '../../data/train/MOT17-05-FRCNN', # 5后面部分 ￥
            '../../data/train/MOT17-09-FRCNN',
            '../../data/train/MOT17-10-FRCNN',
            '../../data/train/MOT17-11-FRCNN', # 11 ￥
            '../../data/train/MOT17-13-FRCNN' # ￥
        ]
        # all_folders = [
            # '../../data/train/HIE20-01',
            # '../../data/train/HIE20-02',
            # '../../data/train/HIE20-03',
            # '../../data/train/HIE20-04',
            # '../../data/train/HIE20-05',
            # '../../data/train/HIE20-06',
            # '../../data/train/HIE20-07',
            # '../../data/train/HIE20-08',
            # '../../data/train/HIE20-09',
            # '../../data/train/HIE20-10',
            # '../../data/train/HIE20-11',
            # '../../data/train/HIE20-12',
            # '../../data/train/HIE20-13',
            # '../../data/train/HIE20-14',
            # '../../data/train/HIE20-15',
            # '../../data/train/HIE20-16',
            # '../../data/train/HIE20-17',
            # '../../data/train/HIE20-18',
            # '../../data/train/HIE20-19',
        # ]
        # print(all_folders)
        # 2. create single parser
        self.parsers = [GTSingleParser(folder,
                                       min_visibility=arg.min_visibility) for folder in all_folders]
        # 3. collect info
        self.collect_pkl = os.path.join(mot_root, det + '_tracks.pkl')

    def val_curve_num(self):
        self.vallens = [p.val_curve_num() for p in self.parsers]  # [valid clip num]
        self.vallen = sum(self.vallens)
        return self.vallen



    def __len__(self):
        # 4. len
        self.lens = [len(p) for p in self.parsers]  # [frame num of a video]
        self.len = sum(self.lens)
        return self.len

    def runV2(self):
        print('Running')
        for parser in self.parsers:
            parser.getBezierCurvesV2()

    def run(self):
        print('Running')
        pool = multiprocessing.Pool(processes=len(self.parsers))
        pool_list = []
        # clear = False
        clear = True
        if clear:
            for parser in self.parsers:
                parser.clear()
        for index in range(len(self.parsers)):
            self.parsers[index].getDetection()

        self.gt_curves = {}

        # collect info to a file
        if True:
            print('Save to ' + self.collect_pkl)
            data = {
                'videos': [],
                'nframes': {},
                'resolution': {},
                'gt_object': {},
            }
            for parser in self.parsers:
                video, nframes, resolution, gt_object = parser.collectInfo()
                data['videos'].append(video)
                data['nframes'][video] = nframes
                data['resolution'][video] = resolution
                data['gt_object'][video] = gt_object

            savePickle(data, self.collect_pkl)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--mot_root', default='../../data', type=str, help="mot data root")
    arg_parser.add_argument('--min_visibility', default=-0.1, type=float, help="minimum visibility of person")

    opt = arg_parser.parse_args()
    parser = GTParser(mot_root=opt.mot_root, arg=opt)
    parser.run()
    # parser.runV2()
