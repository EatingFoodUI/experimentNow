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

Mtk = lambda n, t, k: t ** k * (1 - t) ** (n - k) * n_over_k(n, k)
# 改阶数
# 三次贝塞尔曲线的所有Bernstein basis polynomials的列表形式
BezierCoeff = lambda ts: [[Mtk(3, t, k) for k in range(4)] for t in ts]

# -------------------------------------
# 高阶贝塞尔曲线 没用
# def helper(n):
#     if n == 1:
#         return n
#     res = 1
#     for i in range(1, n + 1):
#         res *= i
#     return res
#
# def getValue(n, m):
#     first = helper(n)
#     second = helper(m)
#     third = helper(n - m)
#     return first // (second * third)
#
# def bezierCurve(x, t, n, size):
#     return x * getValue(n, size) * pow(t, n) * pow((1-t), size)

# -------------------------------------

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

# fig图框, ctp控制点坐标, tube当前行人所有轨迹信息, s起始点所在帧的索引, e终止点所在帧的索引, frame_stride帧跳步数
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
    # e终止点所在帧的索引
    e += 1
    # 设置每此跳两帧
    _slice = slice(s, e, frame_stride)
    # 绘图
    ax = fig.gca(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('y')
    # 设置使用哪些帧（sparse）
    sparse_tubelet = tube[_slice]
    # sparse_tubelet - bboxes
    # 三元组构成的list（x，帧id，y）
    toPoints = lambda xn, franme_idn, yn: list(zip(xn, franme_idn, yn))
    # 绘制真正的帧中心点组成的轨迹曲线
    # 画行人轨迹序列使用的所有帧
    for i in range(sparse_tubelet.shape[0]):
        # 当前帧的left top和right bottom的x轴和y轴
        # [lt, rb, lt, rb]
        xn = [sparse_tubelet[i, 1], sparse_tubelet[i, 3], sparse_tubelet[i, 3], sparse_tubelet[i, 1], ]
        # [lt, lt, rb, rb]
        yn = [sparse_tubelet[i, 2], sparse_tubelet[i, 2], sparse_tubelet[i, 4], sparse_tubelet[i, 4], ]
        # 当前帧id*4 分别对应当前帧的四个顶点
        franme_idn = [sparse_tubelet[i, 0]] * 4

        # 边
        edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
        # edges = [(0, 1), (1, 2), (2, 3),(3,4), (4, 0)]
        # xyzn = zip(xn,yn,franme_idn)

        # （x，帧id，y）[lt lt, rb lt, lt rb, rb rb]
        # p是帧的四个顶点
        p = toPoints(xn, franme_idn, yn)
        # print(p)
        # exit()

        # segments 是帧的四条线段
        segments = [(p[s], p[e]) for s, e in edges]
        # 绘制帧边的四个顶点
        ax.scatter(xn, franme_idn, yn, marker='o', c='k', s=1)
        # center
        cx = [(sparse_tubelet[i, 1] + sparse_tubelet[i, 3]) / 2]
        cy = [(sparse_tubelet[i, 2] + sparse_tubelet[i, 4]) / 2]
        cz = [sparse_tubelet[i, 0]]
        # 绘制中心点
        ax.scatter(cx, cz, cy, marker='+', c='k', s=64, linewidths=1)

        # plot edges
        edge_col = Line3DCollection(segments, colors='k', lw=1.0)
        ax.add_collection3d(edge_col)

    # ctp
    # ctp 绘制控制点
    xn = ctp[:, 0].tolist()
    yn = ctp[:, 1].tolist()
    franme_idn = ctp[:, 2].tolist()
    # 三次bezier曲线，四个控制点，三条线段
    edges = [(0, 1), (1, 2), (2, 3)]
    # edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    p = toPoints(xn, franme_idn, yn)
    segments = [(p[s], p[e]) for s, e in edges]
    ax.scatter(xn, franme_idn, yn, marker='o', c='b', s=16)
    edge_col = Line3DCollection(segments, linestyles='dashed', colors='b', lw=2.0)
    ax.add_collection3d(edge_col)

    # 绘制中心轨迹（true ground curve）
    # center track
    cx = ((tube[s:e, 1] + tube[s:e, 3]) / 2).tolist()
    cy = ((tube[s:e, 2] + tube[s:e, 4]) / 2).tolist()
    cz = tube[s:e, 0].tolist()
    p = toPoints(cx, cz, cy)
    edges = [(i, i + 1) for i in range(len(cz) - 1)]
    segments = [(p[s], p[e]) for s, e in edges]
    # 画出控制点控制轨迹
    ax.scatter(cx, cz, cy, marker='o', s=0.1)
    edge_col = Line3DCollection(segments, colors='k', lw=2.0)
    ax.add_collection3d(edge_col)

    # 绘制bezier曲线轨迹
    # t0 = time.time()
    # bezier_curve
    # 四个控制点
    x0, x1, x2, x3 = ctp[0, 0], ctp[1, 0], ctp[2, 0], ctp[3, 0]
    y0, y1, y2, y3 = ctp[0, 1], ctp[1, 1], ctp[2, 1], ctp[3, 1]
    frame0, frame1, frame2, frame3 = ctp[0, 2], ctp[1, 2], ctp[2, 2], ctp[3, 2]
    # x0, x1, x2, x3, x4, x5 = ctp[0, 0], ctp[1, 0], ctp[2, 0], ctp[3, 0], ctp[4, 0], ctp[5, 0]
    # y0, y1, y2, y3, y4, y5 = ctp[0, 1], ctp[1, 1], ctp[2, 1], ctp[3, 1], ctp[4, 1], ctp[5, 1]
    # frame0, frame1, frame2, frame3, frame4, frame5 = ctp[0, 2], ctp[1, 2], ctp[2, 2], ctp[3, 2], ctp[4, 2], ctp[5, 2]

    # 三次bezier曲线func（参数是四个控制点）
    func = lambda x0, x1, x2, x3: [(1 - t) * ((1 - t) * ((1 - t) * x0 + t * x1) + t * ((1 - t) * x1 + t * x2)) + t * (
             (1 - t) * ((1 - t) * x1 + t * x2) + t * ((1 - t) * x2 + t * x3)) for t in np.linspace(0, 1, 128).tolist()]

    # func = lambda x0, x1, x2, x3, x4, x5: [(bezierCurve(x0, t, 0, 5)
    #                                        + bezierCurve(x1, t, 1, 5)
    #                                        + bezierCurve(x2, t, 2, 5)
    #                                        + bezierCurve(x3, t, 3, 5)
    #                                        + bezierCurve(x4, t, 4, 5)
    #                                        + bezierCurve(x5, t, 5, 5)) for t in np.linspace(0, 1, 128).tolist()]

    # bezier_x = func(x0, x1, x2, x3, x4, x5)
    # bezier_y = func(y0, y1, y2, y3, y4, y5)
    # bezier_frame_id = func(frame0, frame1, frame2, frame3, frame4, frame5)

    # 分别求bezier点的x、y、k（帧）坐标
    bezier_x = func(x0, x1, x2, x3)
    bezier_y = func(y0, y1, y2, y3)
    bezier_frame_id = func(frame0, frame1, frame2, frame3)

    p = toPoints(bezier_x, bezier_frame_id, bezier_y)
    edges = [(i, i + 1) for i in range(len(bezier_x) - 1)]
    segments = [(p[s], p[e]) for s, e in edges]
    ax.scatter(bezier_x, bezier_frame_id, bezier_y, marker='o', s=0.1)
    edge_col = Line3DCollection(segments, colors='r', lw=2.0)
    ax.add_collection3d(edge_col)

# 对所有数据集下的单个子目录（部分数据）进行分析
class GTSingleParser:
    def __init__(self, folder,
                 min_visibility,
                 forward_frames,
                 clip_len):
        # 1. get the gt path and image folder
        # 得到这个子目录下的gt所在路径
        gt_file_path = os.path.join(folder, 'gt/gt.txt')
        # 这个子目录的路径
        self.folder = folder
        # 使用图片的最小遮挡率，大于此遮挡率的图片不使用
        self.min_visibility = min_visibility

        # 轨迹文件
        self.tracks_file = os.path.join(self.folder, 'tracks.pkl')
        # print(self.tracks_file)

        # 2. read the gt data
        # 读取gt文件的数据
        gt_file = pd.read_csv(gt_file_path, header=None)
        # gt_file = gt_file[gt_file[6] == 1]  # human class
        # gt_file = gt_file[gt_file[8] > min_visibility]

        # 读特定结构的数据
        # 获取没有遮挡的人的框
        # 使用gt_file[6] == 1的数据（框的是人）
        gt_file = gt_file[gt_file[6] == 1]  # human class
        # visibility 可视程度（物品被遮挡或者裁剪的程度）（mot17：0-1）
        # 获取可视程度>min_visibility-0.1
        gt_file = gt_file[gt_file[8] > min_visibility]
        # 对满足要求的bbox按所在帧进行分组
        gt_group = gt_file.groupby(0)
        # 获取所有帧
        gt_group_keys = gt_group.indices.keys()
        # 得到最后的帧id
        self.max_frame_index = max(gt_group_keys)

        # important info
        # print(folder) # ../../data/train/MOT17-02-FRCNN
        # 获取这个部分数据的元信息（帧数、图片长宽、格式、文件名等）
        meta_info = open(os.path.join(folder, 'seqinfo.ini')).read()
        frame_rate = int(meta_info[meta_info.find('frameRate') + 10:meta_info.find('\nseqLength')])
        imWidth = int(meta_info[meta_info.find('imWidth') + 8:meta_info.find('\nimHeight')])
        imHeight = int(meta_info[meta_info.find('imHeight') + 9:meta_info.find('\nimExt')])

        # 提取帧长度（绘制一个bezier curve所用帧数）
        self.forward_frames = forward_frames
        # 在半秒内取forward_frames数量的帧，间隔为frame_stride
        self.frame_stride = round((clip_len * frame_rate - 1) / (forward_frames - 1))
        # video名称
        self.video = folder.split('/')[-1]
        # print(self.video)
        print("clip_len:{},frame_rate:{},forward_frames:{}".format(clip_len,frame_rate,self.forward_frames))
        # print('frame_rate:', frame_rate)
        # print('forward_frames:', self.forward_frames, 'frame_stride:', self.frame_stride)
        self.nframes = self.max_frame_index
        self.resolution = (imHeight, imWidth)

        # gt对应的bezier curve还没有
        self.gt_curves = None

        # key names
        forward_frames, frame_stride = self.forward_frames, self.frame_stride
        # 采样模式
        sampling_mode = 'forward_{}_stride_{}'.format(forward_frames, frame_stride)
        # 记录curve的生成类型
        self.curve_type = 'curves_' + sampling_mode
        # print(self.curve_type)
        # 每个数据（每张图片）的格式（元数据）
        # 6 是行人的置信度
        # 7 类别
        # 8 可见率
        '''
        0(frame)   1(id)      2           3       4        5       6(conf,7为1时为1)   7(cls)   8(vis)
        1          2          1338        418     167      379     1                  1        (0-1)
        '''
        # 3. update tracks
        # 根据帧前进（时间向前推）更新tracks
        self.tracks = Tracks()
        # 记录每帧检测结果的数据
        self.recorder = {}
        # 循环每一个选中的帧,一帧中可能有多个物体
        for frame_id in gt_group_keys:
            # 获取当前帧检测到的结果（可能有多个）
            det = gt_group.get_group(frame_id).values
            # 结果（人）集合的id
            person_ids = np.array(det[:, 1]).astype(int)
            # bbox的值
            bboxes = np.array(det[:, 2:6])
            # ？？修改为[左上角x,左上角y,右下角x,右下角y]:可能算错，左上角y-height为右下角y
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

            # 记录当前帧的信息
            self.recorder[frame_id - 1] = list()
            # 对帧中的每个检测结果（结果id，bbox）进行处理
            for person_id, bbox in zip(person_ids, bboxes):
                # 创建Node数据结构来保存当前帧的检测结果的person_id和bbox和此检测结果person对应下一个检测结果的所在帧
                person_node = Node(bbox, frame_id - 1)
                # 如果该person还没有初始化，初始化并添加node作为第一个轨迹
                # 如果该person已经初始化，在该person的轨迹序列后面添加一个新的轨迹
                # track_idx：该轨迹序列在tracks中的位置
                # person_node_idx：是所在轨迹序列的第几个轨迹（node）
                track_idx, person_node_idx = self.tracks.add_node(person_node, person_id)
                # 在帧中存储当前帧的每个检测结果对应在哪个person的第几个位置
                self.recorder[frame_id - 1].append((track_idx, person_node_idx))

    # 要执行的帧的中心点坐标x和y，当前要执行的所有帧的id，中心帧id
    def bezierFit(self, x, y, frame_ids, center_frame_id):
        # 帧和帧之间的中心点在y轴上移动的距离
        dy = y[1:] - y[:-1]
        # x轴上移动的距离
        dx = x[1:] - x[:-1]
        # 帧和帧间隔的帧数
        dframe_id = frame_ids[1:] - frame_ids[:-1]
        # 帧和帧之间中心点的距离（距离公式）
        dt = (dx ** 2 + dy ** 2 + dframe_id ** 2) ** 0.5
        # ？？贝塞尔曲线的t参数
        t = dt / dt.sum()
        # t在第一列添加0
        t = np.hstack(([0], t))
        # 对t进行累加操作
        t = t.cumsum()
        # print('x:', x)
        # print('y:', y)
        # print('frame_id:', frame_id)
        # print('dx:', dx)
        # print('dy:', dy)
        # print('dframe_id:', dframe_id)
        # print('delta_t:', t)
        # print('t:', t)

        # data:{中心点x，中心点y，对应帧id}
        data = np.column_stack((x, y, frame_ids))
        # print('data:', data)
        # np.linalg.pinv 求伪逆
        Pseudoinverse = np.linalg.pinv(BezierCoeff(t))  # (4,13)
        # bB = 控制点，得到四个控制点
        control_points = Pseudoinverse.dot(data)  # (4,13)*(13,3) -> (4,3) | (5,13)*(13,3) -> (5,3)
        # print(Pseudoinverse.shape)
        # print(data.shape)
        # exit()

        # 第一个控制点改为数据的第一个点
        control_points[0] = data[0]
        # 最后一个控制点改为数据的最后一个点
        control_points[-1] = data[-1]
        # print('abs:', control_points)

        # 相对于中心帧的控制点位置（为了预测相对的位置）
        relative_control_points = control_points - data[center_frame_id]
        # print('mid center:', data[self.forward_frames // 2])
        # print('relative:', control_points)
        # print('bezier vector:', control_points[1:] - control_points[:-1])
        return control_points, relative_control_points

    # tube：当前行人的轨迹信息
    # true_ids：会执行的帧的索引（满足所有条件的帧）
    # 绘制当前行人在一段时间内的轨迹和bezier curve
    def curveFromTube(self, tube, true_ids, center_frame_id):
        '''

        :param tube:
        :param true_ids: tube在clip中的frame_id的在tube中的坐标
        :param center_frame_id: 当前clip的center frame id
        :return:
        '''
        # 获取true_ids范围中的所有帧
        true_ids = np.arange(true_ids[0], true_ids[-1] + 1)  # 利用上了clip内的所有的frame
        # 当前行人的所有轨迹的帧id、每帧中心点坐标
        frame_id = tube[:, 0]
        center_x = (tube[:, 1] + tube[:, 3]) / 2
        center_y = (tube[:, 2] + tube[:, 4]) / 2
        # 参数：要执行的帧的中心点坐标x和y，要执行的帧的id（true_ids范围中的所有帧），中心帧id
        # 返回：控制点位置，相对控制点位置
        ctp, relative_ctp = self.bezierFit(center_x[true_ids], center_y[true_ids], frame_id[true_ids],
                                           true_ids.tolist().index(frame_id.tolist().index(center_frame_id)))
        return ctp, relative_ctp

    # 获取bezierCurves
    def getBezierCurves(self):
        # 曲线数据的格式
        '''
            gt_curves:{
                    # 哪个目录（video）下的数据对应的曲线
                    'MOT17-02-FRCNN':
                            #
                            {id:
                                #
                                {tube:array([[frame_id,x1,y1,x2,y2]...])}
                                {
                                        # 曲线的类型，怎样生成的曲线：et. 一共使用7帧，每帧间隔10帧
                                        curves_forward_7_stride_10:{
                                            # 帧id
                                            frame_id:{
                                                        # 曲线的点，使用的是四个控制点
                                                        curve:[x1,y1,t1...x4,y4,t4],
                                                        #
                                                        mask:[0 1 1 1 1 1 0]
                                                    }
                                        }
                                }
                           }
                    }
        '''

        # 获取curve格式
        forward_frames, frame_stride = self.forward_frames, self.frame_stride
        curve_type = self.curve_type
        # 存放轨迹文件的位置（满足条件的轨迹的集合）
        tracks_file = self.tracks_file
        # print('forward_frames:{} | frame_stride:{} | curve_type:{} '.format(forward_frames, frame_stride, curve_type))
        # print(tracks_file)
        #
        if os.path.exists(tracks_file):
            print('File already exists!!!')
            self.gt_curves = readPickle(tracks_file)

            for track in self.tracks.tracks:
                tube_id = track.id
                print(curve_type)
                print(self.video)
                if curve_type not in self.gt_curves[self.video][tube_id]:
                    self.gt_curves[self.video][tube_id][curve_type] = {}

            fig = plt.figure(figsize=(10.8, 7.2))
            for frame_id in range(1, self.nframes + 1):
                max_index = frame_id + (forward_frames - 1) * frame_stride
                center_frame_id = (max_index + frame_id) // 2
                clip_frame_ids = np.arange(frame_id, max_index + 1, frame_stride)
                if max_index >= self.nframes + 1:
                    break
                for track in self.tracks.tracks:
                    tube_id = track.id
                    tube = self.gt_curves[self.video][tube_id]['tube']
                    frame_id_in_tube = tube[:, 0]

                    if frame_id in self.gt_curves[self.video][tube_id][curve_type]:
                        continue
                        relative_ctp = self.gt_curves[self.video][tube_id][curve_type][frame_id]['curve']
                        mask = self.gt_curves[self.video][tube_id][curve_type][frame_id]['mask']

                        max_index = frame_id + (forward_frames - 1) * frame_stride
                        center_frame_id = (max_index + frame_id) // 2
                        clip_frame_ids = np.arange(frame_id, max_index + 1, frame_stride)
                        c = mask * clip_frame_ids
                        tube_frame_id_in_clip = c.ravel()[np.flatnonzero(c)]
                        true_ids = findIndex(tube_frame_id_in_clip, frame_id_in_tube)
                        tublet = self.gt_curves[self.video][tube_id]['tube'][true_ids]
                        i = true_ids.tolist().index(frame_id_in_tube.tolist().index(center_frame_id))
                        data = np.column_stack(
                            ((tublet[:, 1] + tublet[:, 3]) / 2, (tublet[:, 2] + tublet[:, 4]) / 2, tublet[:, 0]))
                        ctp = relative_ctp + data[i]
                        tublet = self.gt_curves[self.video][tube_id]['tube'][true_ids]
                        # drawBboxCurve(fig, ctp, tublet)

                    elif frame_id not in self.gt_curves[self.video][tube_id][curve_type]:
                        # tube not in center frame
                        if center_frame_id not in frame_id_in_tube:
                            continue
                        mask = []
                        for clip_frame_idx in clip_frame_ids:
                            if clip_frame_idx in frame_id_in_tube:
                                mask.append(1)
                            else:
                                mask.append(0)

                        # at least 2 points to fit the curve
                        if sum(mask) <= 5:
                            continue
                        c = mask * clip_frame_ids
                        tube_frame_id_in_clip = c.ravel()[np.flatnonzero(c)]
                        true_ids = findIndex(tube_frame_id_in_clip, frame_id_in_tube)

                        ctp, relative_ctp = self.curveFromTube(tube, true_ids, center_frame_id)
                        self.gt_curves[self.video][tube_id][curve_type][frame_id] = {}
                        self.gt_curves[self.video][tube_id][curve_type][frame_id]['curve'] = relative_ctp
                        self.gt_curves[self.video][tube_id][curve_type][frame_id]['mask'] = mask
                        # tublet = tube[true_ids]
                        # drawBboxCurve(fig, ctp, tublet)
                # plt.pause(0.001)
                # fig.clf()
            plt.close()
        else:
            print('File not exists')
            # get the tracks first
            # 存放所有曲线点的信息
            self.gt_curves = {}
            # 是哪个video下的数据信息
            self.gt_curves[self.video] = {}
            # self.tracks.tracks：当前video中不同人的轨迹序列
            # 目的：在gt_curves里存放当前video所有人的轨迹序列，存放格式为np形式
            for track in self.tracks.tracks:
                # tube_id 是 person_id
                tube_id = track.id
                # print(type(tube_id), tube_id)
                # 当前person的轨迹链表
                person_nodes = track.nodes
                self.gt_curves[self.video][tube_id] = {}
                self.gt_curves[self.video][tube_id][curve_type] = {}
                # 存储这个人所有的帧id和对应的bbox信息
                tk = []
                # person_node：这个人的某帧信息
                for person_node in person_nodes:
                    frame_id = person_node.frame_id
                    bbox = person_node.box
                    tk.append(np.hstack(([frame_id], bbox)))
                if len(tk) == 0:
                    continue
                self.gt_curves[self.video][tube_id]['tube'] = np.array(tk).astype(np.int64)  # <----

            fig = plt.figure(figsize=(10.8, 7.2))
            # video所有帧
            # 目的：尽可能找多的帧序列来拟合bezier curve的曲线参数
            for frame_id in range(1, self.nframes + 1):
                # print('frame:', frame_id)
                # 当前帧持续到的最后一帧
                max_index = frame_id + (forward_frames - 1) * frame_stride
                # ？？目的是什么
                center_frame_id = (max_index + frame_id) // 2
                # 这段帧序列使用的帧
                clip_frame_ids = np.arange(frame_id, max_index + 1, frame_stride)
                if max_index >= self.nframes + 1:
                    break
                # self.tracks.tracks：当前video中不同人的轨迹序列（所有人）
                # track即为一个人的轨迹序列
                for track in self.tracks.tracks:
                    # id：行人id
                    tube_id = track.id
                    # tube not in center frame
                    # 这个人的轨迹信息
                    tube = self.gt_curves[self.video][tube_id]['tube']
                    # 这个人的轨迹的每帧id
                    frame_id_in_tube = tube[:, 0]

                    if center_frame_id not in frame_id_in_tube:
                        continue
                    # 存放当前选择人的轨迹在clip_frame_idx里面的帧（设为1，否则设为0）
                    # 目的：在半秒内使用可使用、且满足条件的帧
                    mask = []
                    # 这段帧序列使用的帧（待选帧：当前起始帧，帧步长，最大帧决定）
                    for clip_frame_idx in clip_frame_ids:
                        if clip_frame_idx in frame_id_in_tube:
                            mask.append(1)
                        else:
                            mask.append(0)

                    # ？？
                    # at least 2 points to fit the curve
                    if sum(mask) <= 5:
                        continue
                    # 将不执行的帧置为0
                    c = mask * clip_frame_ids
                    # 存放会执行的帧
                    tube_frame_id_in_clip = c.ravel()[np.flatnonzero(c)]
                    # 会执行的帧 对应 当前行人帧序列中的索引位置（方便获取帧信息）
                    true_ids = findIndex(tube_frame_id_in_clip, frame_id_in_tube)  # tube在clip中的frame_id的在tube中的坐标

                    # print('center:', center_frame_id)
                    # print('clip_frame_ids:', clip_frame_ids)
                    # print('mask:', mask)
                    # print('tube_frame_id_in_clip:',tube_frame_id_in_clip)
                    # print('true_ids', true_ids)
                    # print('----------------------')

                    # 返回：控制点位置，相对控制点位置
                    ctp, relative_ctp = self.curveFromTube(tube, true_ids, center_frame_id)
                    self.gt_curves[self.video][tube_id][curve_type][frame_id] = {}
                    # 存放当前行人当前使用帧序列所对应的曲线的相对控制点
                    self.gt_curves[self.video][tube_id][curve_type][frame_id]['curve'] = relative_ctp
                    self.gt_curves[self.video][tube_id][curve_type][frame_id]['mask'] = mask
                    tublet = self.gt_curves[self.video][tube_id]['tube'][true_ids]
                    # drawBboxCurve(fig, ctp, tublet)
                    # plt.pause(0.01)
                    # fig.clf()
                    # plt.pause(0.001)
                    # fig.clf()
            plt.close()
        # 保存所有人的轨迹信息和对应曲线信息
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
                e = max_index + 1
                drawBboxCurveV2(fig, ctp, tube, i, e, self.frame_stride)
                # plt.pause(0.001)
                writer.grab_frame()
                fig.clf()
                curves.append(relative_ctp)
            plt.ioff()
            plt.close()
        return curves

    def getBezierCurvesV2(self):
        '''
            gt_curves:{
                    'MOT17-02-FRCNN':
                            {id:
                                {tube:array([[frame_id,x1,y1,x2,y2]...])}
                                {
                                        curves_forward_7_stride_10:{
                                            frame_id:{
                                                        curve:[x1,y1,t1...x4,y4,t4],
                                                        mask:[0 1 1 1 1 1 0]
                                                    }
                                        }
                                }
                           }
                    }
        '''
        forward_frames, frame_stride = self.forward_frames, self.frame_stride
        curve_type = self.curve_type
        tracks_file = self.tracks_file
        self.gt_curves = readPickle(tracks_file)
        print(curve_type)

        # 当前自数据集下的每一个行人轨迹
        for track in self.tracks.tracks:
            tube_id = track.id
            # 这个行人轨迹是否有满足之前条件从而形成的bezier curve轨迹
            if curve_type not in self.gt_curves[self.video][tube_id]:
                self.gt_curves[self.video][tube_id][curve_type] = {}

        for track in self.tracks.tracks:
            # 开始画图
            fig = plt.figure(figsize=(10.8, 7.2))
            # 当前行人id
            tube_id = track.id
            tube = self.gt_curves[self.video][tube_id]['tube']
            frame_id_in_tube = tube[:, 0]
            # 当前人的轨迹序列中满足条件的轨迹段的信息和curve信息
            tublets = self.gt_curves[self.video][tube_id][curve_type]
            for frame_id in sorted(tublets.keys()):
                # 当前轨迹序列对应控制点的相对位移
                relative_ctp = self.gt_curves[self.video][tube_id][curve_type][frame_id]['curve']
                # 当前轨迹序列使用到的帧的索引（用1标记）
                mask = self.gt_curves[self.video][tube_id][curve_type][frame_id]['mask']

                max_index = frame_id + (forward_frames - 1) * frame_stride
                center_frame_id = (max_index + frame_id) // 2
                clip_frame_ids = np.arange(frame_id, max_index + 1, frame_stride)
                c = mask * clip_frame_ids
                tube_frame_id_in_clip = c.ravel()[np.flatnonzero(c)]
                # 在当前行人要使用的轨迹在轨迹序列中的索引位置
                true_ids = findIndex(tube_frame_id_in_clip, frame_id_in_tube)
                # 得到真正使用的帧信息
                tublet = self.gt_curves[self.video][tube_id]['tube'][true_ids]
                # 中心帧所在位置
                i = true_ids.tolist().index(frame_id_in_tube.tolist().index(center_frame_id))

                # tublet[:, 0] -= tublet[i, 0]
                # tublet[:, 1] -= (tublet[i, 1] + tublet[i, 3]) // 2
                # tublet[:, 2] -= (tublet[i, 2] + tublet[i, 4]) // 2
                # tublet[:, 3] -= (tublet[i, 1] + tublet[i, 3]) // 2
                # tublet[:, 4] -= (tublet[i, 2] + tublet[i, 4]) // 2

                # 每帧中心的x，y，哪一帧
                data = np.column_stack(
                    ((tublet[:, 1] + tublet[:, 3]) / 2, (tublet[:, 2] + tublet[:, 4]) / 2, tublet[:, 0]))
                # 中心点坐标+相对控制点坐标=控制点坐标
                ctp = relative_ctp + data[i]
                # 起始点所在帧的索引
                s = true_ids[0]
                # 终止点所在帧的索引
                e = true_ids[-1]
                # fig图框, ctp控制点坐标, tube当前行人所有轨迹信息, s起始点所在帧的索引, e终止点所在帧的索引, frame_stride帧跳步数
                drawBboxCurveV2(fig, ctp, tube, s, e, self.frame_stride)
                plt.pause(0.01)
                fig.clf()
            plt.close()

        plt.close()

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
                os.system('echo qwe123 | sudo -S rm -f {}'.format(path))
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

# ground truth分析器
class GTParser:
    def __init__(self, mot_root,
                 arg,
                 mode='train',
                 det='FRCNN',
                 ):
        # analsis all the folder in mot_root
        # 1. get all the folders
        # 获取mot训练集的根目录
        mot_root = os.path.join(mot_root, mode)
        # 获取mot17训练集的所有子路径
        all_folders = sorted(
            [os.path.join(mot_root, i) for i in os.listdir(mot_root)
             if os.path.isdir(os.path.join(mot_root, i)) and i.find(det) != -1]
        )
        # 使用训练集的部分数据
        # all_folders = ['../../data/train/MOT20-01',
        #                '../../data/train/MOT20-02',
        #                '../../data/train/MOT20-03',
        #                '../../data/train/MOT20-05']
        all_folders = [# 移动 ￥
            '../../data/train/MOT17-02-FRCNN',
            '../../data/train/MOT17-04-FRCNN',
            '../../data/train/MOT17-09-FRCNN', # 5后面部分 ￥
            '../../data/train/MOT17-05-FRCNN', # 9少部分,0.25不用
            '../../data/train/MOT17-10-FRCNN', # 10 ￥
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
        # 对数据集的单个子目录下的每一个数据（检测结果）进行分析
        self.parsers = [GTSingleParser(folder,
                                       min_visibility=arg.min_visibility,
                                       forward_frames=arg.forward_frames,
                                       clip_len=arg.clip_len) for folder in all_folders]
        # 3. collect info
        # 记录信息到_tracks.pkl中
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
        # 使用多线程
        pool = multiprocessing.Pool(processes=len(self.parsers))
        pool_list = []
        # clear = False
        # 重新生成曲线
        clear = True
        if clear:
            # 删除tracks_file文件，后面重新生成
            for parser in self.parsers:
                parser.clear()
        # 每一个数据集的子目录有一个分析器parsers，通过每一个分析器获取一个video下满足之前条件的行人对应的bezierCurves
        for index in range(len(self.parsers)):
            # 绘制每一个video的BezierCurves
            self.parsers[index].getBezierCurves()
            # pool_list.append(pool.apply_async(self.parsers[index].getBezierCurves, ()))
        # for p in tqdm(pool_list, ncols=20):
        #     p.get()
        #
        # pool.close()
        # pool.join()

        # collect info to a file
        if True:
            print('Save to ' + self.collect_pkl)
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

            # 存放数据集所有bezer curve信息
            savePickle(data, self.collect_pkl)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--mot_root', default='../../data', type=str, help="mot data root")
    arg_parser.add_argument('--min_visibility', default=-0.1, type=float, help="minimum visibility of person")
    # arg_parser.add_argument('--frame_stride', default=7, type=int, help="frames to skip")
    # 提取bezier curve的帧数量
    arg_parser.add_argument('--forward_frames', default=7, type=int, help="frame number to extract bezier curve")
    # 剪切时间的间隔(取半秒)
    arg_parser.add_argument('--clip_len', default=1.0, type=float, help="sparse clip time interval")

    opt = arg_parser.parse_args()
    parser = GTParser(mot_root=opt.mot_root, arg=opt)
    # 获取行人轨迹对应的curve并保存这些数据
    parser.run()
    # parser.runV2()
    # print('forward frames:', opt.forward_frames, 'clip_len:', opt.clip_len)
    # print('{} / {}'.format(parser.val_curve_num(), len(parser)))
