import os
import torch.utils.data as data
import cv2
import numpy as np
import random
import torch
from MOC_utils.utils import random_affine,letterbox


class BaseDataset(data.Dataset):
    # 数据集基类初始化
    def __init__(self, opt):
        '''
            定义输入图片高宽，单张图片存在最大物体数量
            定义数据增强操作
        '''
        super(BaseDataset, self).__init__()

        self.opt = opt
        # self._mean_values = [104.0136177, 114.0342201, 119.91659325]
        self.height = opt.input_h
        self.width = opt.input_w
        self.max_objs = opt.max_objs
        self.augment = opt.augment
        self.hsv_aug = opt.hsv_aug
        self.flip_aug = opt.flip_aug
        self.affine_aug = opt.affine_aug

    # 数据集的数据大小
    def __len__(self):
        return len(self._indices)

    # 返回带检测框的图片
    def tubletImgFiles(self, vdir, vname, frame_ids):
        raise NotImplementedError

    # 为测试集数据进行数据增强操作
    def mot_augment(self, images, gt_sparse_tublet, only_letter_box=False):
        '''
            对数据集进行了hsv、flip、affine的数据增强操作
        '''
        if only_letter_box:
            self.augment = False
        images = np.array(images)

        height = self.height
        width = self.width
        if self.augment and self.hsv_aug:
            images_hsv = images.copy()
            for i in range(images_hsv.shape[0]):
                images_hsv[i] = cv2.cvtColor(images_hsv[i], cv2.COLOR_BGR2HSV)
            fraction = 0.50
            S = images_hsv[:, :, :, 1].astype(np.float32)
            V = images_hsv[:, :, :, 2].astype(np.float32)
            random_hsv = random.random()
            a = (random_hsv * 2 - 1) * fraction + 1
            S *= a
            if a > 1:
                np.clip(S, a_min=0, a_max=255, out=S)

            random_hsv = random.random()
            a = (random_hsv * 2 - 1) * fraction + 1
            V *= a
            if a > 1:
                np.clip(V, a_min=0, a_max=255, out=V)

            images_hsv[:, :, :, 1] = S.astype(np.uint8)
            images_hsv[:, :, :, 2] = V.astype(np.uint8)

            for i in range(images_hsv.shape[0]):
                cv2.cvtColor(images_hsv[i], cv2.COLOR_HSV2BGR, dst=images[i])

        h, w = self.height, self.width
        images, ratio, padw, padh = letterbox(images, height=height, width=width)

        for tube_id in gt_sparse_tublet:
            gt_sparse_tublet[tube_id][:, 1] = ratio * gt_sparse_tublet[tube_id][:, 1] + padw
            gt_sparse_tublet[tube_id][:, 2] = ratio * gt_sparse_tublet[tube_id][:, 2] + padh
            gt_sparse_tublet[tube_id][:, 3] = ratio * gt_sparse_tublet[tube_id][:, 3] + padw
            gt_sparse_tublet[tube_id][:, 4] = ratio * gt_sparse_tublet[tube_id][:, 4] + padh

        if self.augment and self.affine_aug:
            images, gt_sparse_tublet, M = random_affine(images, gt_sparse_tublet,
                                                        degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.50, 1.20))

        random_flip = random.random()
        do_mirror = self.augment and self.flip_aug and (random_flip > 0.5)
        # do_mirror = True
        # filp the image
        if do_mirror:
            images = [im[:, ::-1, :] for im in images]

            # filp the gt bbox
            for tube_id in gt_sparse_tublet:
                tubelet = gt_sparse_tublet[tube_id]
                xmin = w - tubelet[:, 3]
                tubelet[:, 3] = w - tubelet[:, 1]
                tubelet[:, 1] = xmin

        images = [np.ascontiguousarray(im[:, :, ::-1]) for im in images]

        # from torchvision.transforms import transforms as T
        # transforms = T.Compose([T.ToTensor()])
        # images = [transforms(im) for im in images]

        return images, gt_sparse_tublet
