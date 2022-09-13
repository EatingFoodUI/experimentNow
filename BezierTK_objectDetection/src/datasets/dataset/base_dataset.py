import os
import torch.utils.data as data
import cv2
import numpy as np
import random
import torch
from MOC_utils.utils import random_affine,letterbox


class BaseDataset(data.Dataset):
    def __init__(self, opt):
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

    def __len__(self):
        return len(self._indices)

    def tubletImgFiles(self, vdir, vname, frame_ids):
        raise NotImplementedError

    def mot_augment(self, image, gt_sparse_tublet, only_letter_box=False):
        if only_letter_box:
            self.augment = False
        image = np.array(image)

        height = self.height
        width = self.width
        if self.augment and self.hsv_aug:
            image_hsv = image.copy()
            image_hsv = cv2.cvtColor(image_hsv, cv2.COLOR_BGR2HSV)
            fraction = 0.50
            S = image_hsv[:, :, 1].astype(np.float32)
            V = image_hsv[:, :, 2].astype(np.float32)
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

            image_hsv[:, :, 1] = S.astype(np.uint8)
            image_hsv[:, :, 2] = V.astype(np.uint8)

            cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR, dst=image)

        h, w = self.height, self.width
        images, ratio, padw, padh = letterbox(image, height=height, width=width)

        for ind in range(len(gt_sparse_tublet)):
            try:
                gt_sparse_tublet[ind][0] = ratio * gt_sparse_tublet[ind][0] + padw
            except (IndexError):
                print(gt_sparse_tublet)
            gt_sparse_tublet[ind][1] = ratio * gt_sparse_tublet[ind][1] + padh
            gt_sparse_tublet[ind][2] = ratio * gt_sparse_tublet[ind][2] + padw
            gt_sparse_tublet[ind][3] = ratio * gt_sparse_tublet[ind][3] + padh

        if self.augment and self.affine_aug:
            images, gt_sparse_tublet, M = random_affine(images, gt_sparse_tublet,
                                                        degrees=(-5, 5), translate=(0.10, 0.10), scale=(0.50, 1.20))

        random_flip = random.random()
        do_mirror = self.augment and self.flip_aug and (random_flip > 0.5)
        do_mirror = True
        # filp the image
        if do_mirror and only_letter_box==False:
            images = images[::-1, :]

            # filp the gt bbox
            for ind in range(len(gt_sparse_tublet)):
                tubelet = gt_sparse_tublet[ind][0]
                xmin = w - tubelet[2]
                tubelet[2] = w - tubelet[0]
                tubelet[0] = xmin

        images = np.ascontiguousarray(images[:, ::-1])

        # from torchvision.transforms import transforms as T
        # transforms = T.Compose([T.ToTensor()])
        # images = [transforms(im) for im in images]

        return images, gt_sparse_tublet
