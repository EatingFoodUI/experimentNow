import cv2
import numpy as np

from MOC_utils.model import create_model, load_model
from MOC_utils.data_parallel import DataParallel
from MOC_utils.utils import transform_preds, letterbox, warp_bbox_iterate, warp_bbox_direct_v2, readPickle

import torch
from .decode import decode


class BezierDetector(object):
    def __init__(self, opt):
        self.opt = opt
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        self.rgb_model = None
        if opt.rgb_model != '':
            print('create rgb model')
            self.rgb_model = create_model(opt.arch, opt.branch_info, opt.head_conv, opt.forward_frames)
            self.rgb_model = load_model(self.rgb_model, opt.rgb_model)
            self.rgb_model = DataParallel(
                self.rgb_model,
                device_ids=opt.gpus,
                chunk_sizes=opt.chunk_sizes).to(opt.device)
            self.rgb_model.eval()

    def pre_process(self, images):
        forward_frames = self.opt.forward_frames
        images, _, _, _ = letterbox(images, self.opt.input_h, self.opt.input_w)
        mean = np.tile(np.array(self.opt.mean, dtype=np.float32)[:, None, None], (1, 1, 1))
        std = np.tile(np.array(self.opt.std, dtype=np.float32)[:, None, None], (1, 1, 1))

        data = [np.empty((3, self.opt.input_h, self.opt.input_w), dtype=np.float32) for i in
                range(forward_frames)]
        for i in range(forward_frames):
            data[i] = np.transpose(images[i], (2, 0, 1))
            # data[i] = ((data[i] / 255.) - mean) / std
            data[i] = data[i] / 255.
            # data[i] = images[i]

        return data

    def process(self, images, hm, frame_stride):
        with torch.no_grad():
            if self.rgb_model is not None:
                rgb_output = self.rgb_model(images)
                # hm = rgb_output[0]['hm'].sigmoid_()
                wh = rgb_output[0]['wh']
                bezier_ctp = rgb_output[0]['bezier_ctp']

        if hm.ndim != 4:
            hm = hm.unsqueeze(1)

        detections = decode(hm, wh, bezier_ctp, self.opt.max_objs,
                            frame_stride=frame_stride, forward_frames=self.opt.forward_frames)
        # print(detections.shape) # (batch, max_objs, 4*7+1)
        return detections

    def post_process(self, detections,
                     input_h,
                     input_w,
                     height,
                     width,
                     output_height,
                     output_width):

        detections = detections.detach().cpu().numpy()
        results = []
        # the ith batch which means sampling start from a different frame
        for i in range(detections.shape[0]):
            mask = detections[i, :, -1] > self.opt.det_thres
            cur_detections = detections[i][mask]

            _input_h = input_h[i]
            _input_w = input_w[i]
            _height = height[i]
            _width = width[i]
            _output_height = output_height[i]
            _output_width = output_width[i]
            c = np.array([_width / 2., _height / 2.], dtype=np.float32)
            s = max(float(_input_w) / float(_input_h) * _height, _width) * 1.0
            for j in range((cur_detections.shape[1] - 1) // 2):
                # cur_detections[i, :, 2 * j] = np.clip(cur_detections[i, :, 2 * j], 0, _output_width - 1)
                # cur_detections[i, :, 2 * j + 1] = np.clip(cur_detections[i, :, 2 * j + 1], 0, _output_height - 1)
                cur_detections[:, 2 * j:2 * j + 2] = transform_preds(cur_detections[:, 2 * j:2 * j + 2],
                                                                    c, s, (_output_width, _output_height))

            results.append(cur_detections.astype(np.float32))

        return results

    def run(self, data):
            images = None

            if self.rgb_model is not None:
                images = data['images']
                for i in range(len(images)):
                    images[i] = images[i].type(torch.FloatTensor)
                    images[i] = images[i].to(self.opt.device)
            meta = data['meta']
            meta = {k: v.numpy() for k, v in meta.items()}

            frame_stride = data['frame_stride'].cpu().numpy().tolist()

            hm = data['hm'].cuda()
            detections = self.process(images, hm, frame_stride)

            start_frame_id = data['start_frame_id'].cpu().numpy().tolist()
            detections = self.post_process(detections,
                                           input_h=meta['input_h'],
                                           input_w=meta['input_w'],
                                           height=meta['height'],
                                           width=meta['width'],
                                           output_height=meta['output_h'],
                                           output_width=meta['output_w'])

            return detections
