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

Mtk = lambda n, t, k: t ** k * (1 - t) ** (n - k) * n_over_k(n, k)
BezierCoeff = lambda ts: [[Mtk(3, t, k) for k in range(4)] for t in ts]


def savePickle(gt_curves, path):
    with open(path, 'wb') as f:
        pickle.dump(gt_curves, f)
        f.close()


def readPickle(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
        f.close()
        return d


def drawBboxCurve(fig, ctp, tubelet):
    # create figure
    # fig = plt.figure(figsize=(16, 12))
    # print(tubelet[:,0])
    ax = fig.gca(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('y')
    # btube
    toPoints = lambda xn, franme_idn, yn: list(zip(xn, franme_idn, yn))
    for i in range(tubelet.shape[0]):
        xn = [tubelet[i, 1], tubelet[i, 3], tubelet[i, 3], tubelet[i, 1], ]
        yn = [tubelet[i, 2], tubelet[i, 2], tubelet[i, 4], tubelet[i, 4], ]
        franme_idn = [tubelet[i, 0]] * 4
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        # xyzn = zip(xn,yn,franme_idn)
        p = toPoints(xn, franme_idn, yn)
        # print(p)
        # exit()
        segments = [(p[s], p[e]) for s, e in edges]
        ax.scatter(xn, franme_idn, yn, marker='o', c='k', s=1)
        # center
        cx = [(tubelet[i, 1] + tubelet[i, 3]) / 2]
        cy = [(tubelet[i, 2] + tubelet[i, 4]) / 2]
        cz = [tubelet[i, 0]]
        ax.scatter(cx, cz, cy, marker='+', c='k', s=64, linewidths=20)

        # plot edges
        edge_col = Line3DCollection(segments, colors='k', lw=1.0)
        ax.add_collection3d(edge_col)

    # ctp
    # xn = ctp[:, 0].tolist()
    # yn = ctp[:, 1].tolist()
    # franme_idn = ctp[:, 2].tolist()
    # edges = [(0, 1), (1, 2), (2, 3)]
    # p = toPoints(xn, franme_idn, yn)
    # segments = [(p[s], p[e]) for s, e in edges]
    # ax.scatter(xn, franme_idn, yn, marker='o', c='b', s=16)
    # edge_col = Line3DCollection(segments, linestyles='dashed', colors='b', lw=2.0)
    # ax.add_collection3d(edge_col)

    # track
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
    _slice = slice(s, e, frame_stride)
    ax = fig.gca(projection='3d')
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
        ax.scatter(cx, cz, cy, marker='+', c='k', s=64, linewidths=20)

        # plot edges
        edge_col = Line3DCollection(segments, colors='k', lw=1.0)
        ax.add_collection3d(edge_col)

    # ctp
    xn = ctp[:, 0].tolist()
    yn = ctp[:, 1].tolist()
    franme_idn = ctp[:, 2].tolist()
    edges = [(0, 1), (1, 2), (2, 3)]
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

    # bezier_curve
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


class GTSingleParser:
    def __init__(self, folder,
                 min_visibility,
                 forward_frames,
                 frame_stride, ):
        # 1. get the gt path and image folder
        gt_file_path = os.path.join(folder, 'gt/gt.txt')
        self.folder = folder
        self.forward_frames = forward_frames
        self.min_visibility = min_visibility
        self.frame_stride = frame_stride
        self.tracks_file = os.path.join(self.folder, 'tracks.pkl')
        # print(self.tracks_file)

        # 2. read the gt data
        gt_file = pd.read_csv(gt_file_path, header=None)
        gt_file = gt_file[gt_file[6] == 1]  # human class
        gt_file = gt_file[gt_file[8] > min_visibility]
        gt_group = gt_file.groupby(0)
        gt_group_keys = gt_group.indices.keys()
        self.max_frame_index = max(gt_group_keys)

        # important info
        self.video = folder.split('/')[-1]
        self.nframes = self.max_frame_index
        img_path = glob.glob(folder + '/img1/*')[0]
        img = cv2.imread(img_path)
        self.resolution = img.shape[:-1]
        self.gt_curves = None

        # key names
        forward_frames, frame_stride = self.forward_frames, self.frame_stride
        sampling_mode = 'forward_{}_stride_{}'.format(forward_frames, frame_stride)
        self.clip_is_complete = 'clip_complete_' + sampling_mode
        self.curve_type = 'curves_' + sampling_mode
        '''
        0(frame)   1(id)      2           3       4        5       6(conf,7为1时为1)   7(cls)   8(vis)
        1          2          1338        418     167      379     1                  1        (0-1)
        '''
        # 3. update tracks
        self.tracks = Tracks()
        self.recorder = {}
        for frame_id in gt_group_keys:
            det = gt_group.get_group(frame_id).values
            person_ids = np.array(det[:, 1]).astype(int)
            bboxes = np.array(det[:, 2:6])
            bboxes[:, 2:4] += bboxes[:, :2]  # tl, br
            # print(bboxes[0])
            # exit()
            self.recorder[frame_id - 1] = list()
            for person_id, bbox in zip(person_ids, bboxes):
                person_node = Node(bbox, frame_id - 1)
                track_idx, person_node_idx = self.tracks.add_node(person_node, person_id)
                self.recorder[frame_id - 1].append((track_idx, person_node_idx))

    def bezierFit(self, x, y, frame_id):
        dy = y[1:] - y[:-1]
        dx = x[1:] - x[:-1]
        dframe_id = frame_id[1:] - frame_id[:-1]
        dt = (dx ** 2 + dy ** 2 + dframe_id ** 2) ** 0.5
        t = dt / dt.sum()
        t = np.hstack(([0], t))
        t = t.cumsum()
        # print('x:', x)
        # print('y:', y)
        # print('frame_id:', frame_id)
        # print('dx:', dx)
        # print('dy:', dy)
        # print('dframe_id:', dframe_id)
        # print('delta_t:', t)
        # print('t:', t)
        data = np.column_stack((x, y, frame_id))
        # print('data:', data)
        Pseudoinverse = np.linalg.pinv(BezierCoeff(t))  # (7,4) -> (7,9)
        control_points = Pseudoinverse.dot(data)  # (4,7)*(7,4) -> (4,3)
        control_points[0] = data[0]
        control_points[-1] = data[-1]
        # print('abs:', control_points)
        relative_control_points = control_points - data[self.forward_frames // 2]
        # print('mid center:', data[self.forward_frames // 2])
        # print('relative:', control_points)
        # print('bezier vector:', control_points[1:] - control_points[:-1])
        return control_points, relative_control_points

    def curveFromTube(self, tube, s):
        # print(tube)
        # print(tube[s])
        # print('----------------')
        frame_id = tube[:, 0]
        center_x = (tube[:, 1] + tube[:, 3]) / 2
        center_y = (tube[:, 2] + tube[:, 4]) / 2
        ctp, relative_ctp = self.bezierFit(center_x[s], center_y[s], frame_id[s])
        return ctp, relative_ctp

    def getBezierCurves(self):
        '''
            gt_curves:{
                    'MOT17-02-FRCNN':
                            {id:
                                {tube:array([[frame_id,x1,y1,x2,y2]...])}
                                {curves_forward_7_stride_10:{ frame_id:[x1,y1,t1...x4,y4,t4] }}
                            }
                    clip_complete_forward_7_stride_10:[True False]
                }
        '''

        def findIndex(frame_id, frame_id_list):
            return frame_id_list.index(frame_id)

        forward_frames, frame_stride = self.forward_frames, self.frame_stride
        curve_type = self.curve_type
        clip_is_complete = self.clip_is_complete
        tracks_file = self.tracks_file
        # print(tracks_file)
        tubes = []
        if os.path.exists(tracks_file):
            print('File already exists!!!')
            self.gt_curves = readPickle(tracks_file)

            for track in self.tracks.tracks:
                tube_id = track.id
                tubes.append(self.gt_curves[self.video][tube_id]['tube'])
                if curve_type not in self.gt_curves[self.video][tube_id]:
                    self.gt_curves[self.video][tube_id][curve_type] = {}

            if clip_is_complete not in self.gt_curves:
                self.gt_curves[clip_is_complete] = []
                # check if clip (under the sampling mode) is complete
                for frame_id in range(1, self.nframes + 1):
                    max_index = frame_id + (forward_frames - 1) * frame_stride
                    if max_index >= self.nframes + 1:
                        break
                    clip_frame_ids = list(range(frame_id, max_index + 1, frame_stride))
                    # print(max_index)
                    # print(clip_frame_ids)
                    # exit()
                    if tubelet_in_out_clip(tubes, clip_frame_ids) and clip_has_tublet(tubes, clip_frame_ids):
                        self.gt_curves[clip_is_complete].append(True)
                    else:
                        self.gt_curves[clip_is_complete].append(False)

            # print(self.gt_curves[clip_is_complete])
            fig = plt.figure(figsize=(16, 12))
            for frame_id in range(1, self.nframes + 1):
                max_index = frame_id + (forward_frames - 1) * frame_stride
                if max_index >= self.nframes + 1:
                    break
                if not self.gt_curves[clip_is_complete][frame_id - 1]:
                    continue
                for track in self.tracks.tracks:
                    tube_id = track.id
                    if frame_id in self.gt_curves[self.video][tube_id]['tube'][:, 0].tolist():
                        true_index = findIndex(frame_id, self.gt_curves[self.video][tube_id]['tube'][:, 0].tolist())
                        s = slice(true_index, max_index + 1 - (frame_id - true_index), frame_stride)

                        ctp, relative_ctp = self.curveFromTube(self.gt_curves[self.video][tube_id]['tube'], s)
                        self.gt_curves[self.video][tube_id][curve_type][frame_id] = relative_ctp  # <----
                        tubelet = self.gt_curves[self.video][tube_id]['tube'][s]
                        # drawBboxCurve(fig, ctp, tubelet)
                # plt.pause(0.001)
                # fig.clf()
            plt.close()
        else:
            print('File not exists')
            # get the tracks first
            self.gt_curves = {}
            self.gt_curves[self.video] = {}
            self.gt_curves[clip_is_complete] = []
            for track in self.tracks.tracks:
                tube_id = track.id
                # print(type(tube_id), tube_id)
                person_nodes = track.nodes
                self.gt_curves[self.video][tube_id] = {}
                self.gt_curves[self.video][tube_id][curve_type] = {}
                l = []
                for person_node in person_nodes:
                    frame_id = person_node.frame_id
                    bbox = person_node.box
                    l.append(np.hstack(([frame_id], bbox)))
                if len(l) == 0:
                    continue
                self.gt_curves[self.video][tube_id]['tube'] = np.array(l).astype(np.int64)  # <----
                tubes.append(np.array(l).astype(np.int64))

            # check if clip (under the sampling mode) is complete
            # frame_stride = 1
            for frame_id in range(1, self.nframes + 1):
                max_index = frame_id + (forward_frames - 1) * frame_stride
                if max_index >= self.nframes + 1:
                    break
                clip_frame_ids = list(range(frame_id, max_index + 1, frame_stride))
                if tubelet_in_out_clip(tubes, clip_frame_ids) and clip_has_tublet(tubes, clip_frame_ids):
                    self.gt_curves[clip_is_complete].append(True)
                else:
                    self.gt_curves[clip_is_complete].append(False)

            fig = plt.figure(figsize=(16, 12))
            for frame_id in range(1, self.nframes + 1):
                max_index = frame_id + (forward_frames - 1) * frame_stride
                if max_index >= self.nframes + 1:
                    break
                if not self.gt_curves[clip_is_complete][frame_id - 1]:
                    continue
                for track in self.tracks.tracks:
                    tube_id = track.id
                    if frame_id in self.gt_curves[self.video][tube_id]['tube'][:, 0].tolist():
                        true_index = findIndex(frame_id, self.gt_curves[self.video][tube_id]['tube'][:, 0].tolist())
                        s = slice(true_index, max_index + 1 - (frame_id - true_index), frame_stride)

                        ctp, relative_ctp = self.curveFromTube(self.gt_curves[self.video][tube_id]['tube'], s)
                        self.gt_curves[self.video][tube_id][curve_type][frame_id] = relative_ctp  # <----
                        # tubelet = self.gt_curves[self.video][tube_id]['tube'][s]
                        # drawBboxCurve(fig, ctp, tubelet)
                # plt.pause(0.001)
                # fig.clf()
            plt.close()
        savePickle(self.gt_curves, tracks_file)

    def curveFromTubeV2(self, tube_id, tube):
        curves = []
        frame_id = tube[:, 0]
        center_x = (tube[:, 1] + tube[:, 3]) / 2
        center_y = (tube[:, 2] + tube[:, 4]) / 2
        fig = plt.figure(figsize=(10.8, 7.2))
        metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support!')
        writer = FFMpegWriter(fps=10, metadata=metadata)
        dir = '../../data/{}_{}'.format(self.forward_frames, self.frame_stride)
        if not os.path.exists(dir):
            print('making directory:', dir)
            os.makedirs(dir)
        with writer.saving(fig, os.path.join(dir, str(tube_id) + '.mp4'), 100):  # 100指的dpi，dot per inch，表示清晰度
            plt.ion()
            plt.tight_layout()
            for i in range(tube.shape[0]):
                max_index = i + (self.forward_frames - 1) * self.frame_stride
                if max_index >= tube.shape[0]:
                    break
                s = slice(i, max_index + 1, self.frame_stride)
                ctp, relative_ctp = self.bezierFit(center_x[s], center_y[s], frame_id[s])
                s = i
                e = max_index + 1
                drawBboxCurveV2(fig, ctp, tube, s, e, self.frame_stride)
                # plt.pause(0.001)
                writer.grab_frame()
                fig.clf()
                curves.append(relative_ctp)
            plt.ioff()
            plt.close()
        return curves

    def getBezierCurvesV2(self):
        self.gt_curves = {}
        self.gt_curves[self.video] = {}
        for track in self.tracks.tracks:
            track_id = track.id
            # print(type(track_id), track_id)
            person_nodes = track.nodes
            self.gt_curves[self.video][track_id] = {}
            l = []
            for person_node in person_nodes:
                frame_id = person_node.frame_id
                bbox = person_node.box
                l.append(np.hstack(([frame_id], bbox)))
            if len(l) == 0:
                continue
            self.gt_curves[self.video][track_id]['tube'] = np.array(l)
            # print(np.array(l))

            # curve_type = 'forward_{}_stride_{}'.format(self.forward_frames, self.frame_stride)
            curves = self.curveFromTubeV2(track_id, self.gt_curves[self.video][track_id]['tube'])
            if len(curves) == 0:
                continue
            # self.gt_curves[self.video][track_id][curve_type] = np.array(curves)
            # print(np.array(curves))

    def __len__(self):
        l = self.max_frame_index
        return l

    def val_l(self):
        self.gt_curves = readPickle(self.tracks_file)
        valid_l = 0
        for track in self.tracks.tracks:
            tube_id = track.id
            valid_l += len(self.gt_curves[self.video][tube_id][self.curve_type])
        return valid_l

    def clear(self):
        try:
            path = self.tracks_file
            if os.path.exists(path):
                os.system('echo 1 | sudo -S rm -f {}'.format(path))
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
        self.gt_curves = readPickle(self.tracks_file)
        return self.video, self.nframes, self.resolution, self.gt_curves


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
        # 2. create single parser
        self.parsers = [GTSingleParser(folder, forward_frames=arg.forward_frames,
                                       min_visibility=arg.min_visibility,
                                       frame_stride=arg.frame_stride, ) for folder in all_folders]
        # 3. collect info
        self.collect_pkl = os.path.join(mot_root, det + '_tracks.pkl')

    def val_l(self):
        self.vallens = [p.val_l() for p in self.parsers]  # [valid clip num]
        self.vallen = sum(self.vallens)
        return self.vallen

    def __len__(self):
        # 4. len
        self.lens = [len(p) for p in self.parsers]  # [valid clip num]
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
        clear = False
        # clear = True
        if clear:
            for parser in self.parsers:
                parser.clear()
                # parser.getBezierCurves()
            # exit()
        for index in range(len(self.parsers)):
            self.parsers[index].getBezierCurves()
            # pool_list.append(pool.apply_async(self.parsers[index].getBezierCurves, ()))
        # for p in tqdm(pool_list, ncols=20):
        #     for p in pool_list:
        # p.get()
        # pool.close()
        # pool.join()

        # collect info to a file
        if False:
            data = {
                'videos': [],
                'nframes': {},
                'resolution': {},
                'gt_curves': {},
            }
            for parser in self.parsers:
                video, nframes, resolution, gt_curves = parser.collectInfo()
                data['videos'].append(video)
                data['nframes'][video] = nframes
                data['resolution'][video] = resolution
                data['gt_curves'][video] = gt_curves

            savePickle(data, self.collect_pkl)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--mot_root', default='../../data', type=str, help="mot data root")
    arg_parser.add_argument('--min_visibility', default=-0.1, type=float, help="minimum visibility of person")
    arg_parser.add_argument('--forward_frames', default=7, type=int, help="frame number to extract bezier curve")
    arg_parser.add_argument('--frame_stride', default=15, type=int, help="frames to skip")

    opt = arg_parser.parse_args()
    parser = GTParser(mot_root=opt.mot_root, arg=opt)
    # parser.runV2()
    parser.run()
    print('forward frames:', opt.forward_frames, 'frame_stride:', opt.frame_stride)
    print('{} / {}'.format(parser.val_l(), len(parser)))
    # 3925 / 5316
    # 1244 / 5316
    # 0 / 5316
