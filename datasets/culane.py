import os
import shutil
import glob
import json
import argparse

import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils.affinity_fields import generateAFs
import datasets.transforms as tf


def coord_op_to_ip(x, y, scale):
    # (208*scale, 72*scale) --> (208*scale, 72*scale+14=590) --> (1664, 590) --> (1640, 590)
    if x is not None:
        x = int(scale*x)
        x = x*1640./1664.
    if y is not None:
        y = int(scale*y+14)
    return x, y

def coord_ip_to_op(x, y, scale):
    # (1640, 590) --> (1664, 590) --> (1664, 590-14=576) --> (1664/scale, 576/scale)
    if x is not None:
        x = x*1664./1640.
        x = int(x/scale)
    if y is not None:
        y = int((y-14)/scale)
    return x, y

def get_lanes_culane(seg_out, samp_factor):
    # fit cubic spline to each lane
    h_samples = range(589, 240, -10)
    cs = []
    lane_ids = np.unique(seg_out[seg_out > 0])
    for idx, t_id in enumerate(lane_ids):
        xs, ys = [], []
        for y_op in range(seg_out.shape[0]):
            x_op = np.where(seg_out[y_op, :] == t_id)[0]
            if x_op.size > 0:
                x_op = np.mean(x_op)
                x_ip, y_ip = coord_op_to_ip(x_op, y_op, samp_factor)
                xs.append(x_ip)
                ys.append(y_ip)
        if len(xs) >= 20:
            cs.append(CubicSpline(ys, xs, extrapolate=False))
        else:
            cs.append(None)
    # get x-coordinates from fitted spline
    lanes = []
    for idx, t_id in enumerate(lane_ids):
        lane = []
        if cs[idx] is not None:
            y_out = np.array(h_samples)
            x_out = cs[idx](y_out)
            for _x, _y in zip(x_out, y_out):
                if np.isnan(_x):
                    continue
                else:
                    lane += [_x, _y]
            lanes.append(lane)

    return lanes

class CULane(Dataset):
    def __init__(self, path, image_set='train', random_transforms=False):
        super(CULane, self).__init__()
        assert image_set in ('train', 'val', 'test'), "image_set is not valid!"
        self.input_size = (288, 832) # original image res: (590, 1640) -> (590-14, 1640+24)/2
        self.output_scale = 0.25
        self.samp_factor = 2./self.output_scale
        self.data_dir_path = path
        self.image_set = image_set
        self.random_transforms = random_transforms
        # normalization transform for input images
        self.mean = [0.485, 0.456, 0.406] #[103.939, 116.779, 123.68]
        self.std = [0.229, 0.224, 0.225] #[1, 1, 1]
        self.ignore_label = 255
        if self.random_transforms:
            self.transforms = transforms.Compose([
                tf.GroupRandomScale(size=(0.5, 0.6), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
                tf.GroupRandomCropRatio(size=(self.input_size[1], self.input_size[0])),
                tf.GroupRandomHorizontalFlip(),
                tf.GroupRandomRotation(degree=(-1, 1), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST), padding=(self.mean, (self.ignore_label, ))),
                tf.GroupNormalize(mean=(self.mean, (0, )), std=(self.std, (1, ))),
            ])
        else:
            self.transforms = transforms.Compose([
                tf.GroupRandomScale(size=(0.5, 0.5), interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST)),
                tf.GroupNormalize(mean=(self.mean, (0, )), std=(self.std, (1, ))),
            ])

        self.create_index()

    def create_index(self):
        self.img_list = []
        self.seg_list = []

        listfile = os.path.join(self.data_dir_path, "list", "{}.txt".format(self.image_set))
        if not os.path.exists(listfile):
            raise FileNotFoundError("List file doesn't exist. Label has to be generated! ...")

        with open(listfile) as f:
            for line in f:
                l = line.strip()
                self.img_list.append(os.path.join(self.data_dir_path, l[1:]))  # l[1:]  get rid of the first '/' so as for os.path.join
                if self.image_set == 'test':
                    self.seg_list.append(os.path.join(self.data_dir_path, 'laneseg_label_w16_test', l[1:-3] + 'png'))
                else:
                    self.seg_list.append(os.path.join(self.data_dir_path, 'laneseg_label_w16', l[1:-3] + 'png'))

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx]).astype(np.float32)/255. # (H, W, 3)
        img = cv2.resize(img[14:, :, :], (1664, 576), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if os.path.exists(self.seg_list[idx]):
            seg = cv2.imread(self.seg_list[idx], cv2.IMREAD_UNCHANGED) # (H, W)
            seg = np.tile(seg[..., np.newaxis], (1, 1, 3)) # (H, W, 3)
            seg = cv2.resize(seg[14:, :, :], (1664, 576), interpolation=cv2.INTER_NEAREST)
            img, seg = self.transforms((img, seg))
            seg = cv2.resize(seg, None, fx=self.output_scale, fy=self.output_scale, interpolation=cv2.INTER_NEAREST)
            # create binary mask
            mask = seg[:, :, 0].copy()
            mask[seg[:, :, 0] >= 1] = 1
            mask[seg[:, :, 0] == self.ignore_label] = self.ignore_label
            # create AFs
            seg_wo_ignore = seg[:, :, 0].copy()
            seg_wo_ignore[seg_wo_ignore == self.ignore_label] = 0
            vaf, haf = generateAFs(seg_wo_ignore.astype(np.long), viz=False)
            af = np.concatenate((vaf, haf[:, :, 0:1]), axis=2)

            # convert all outputs to torch tensors
            img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
            seg = torch.from_numpy(seg[:, :, 0]).contiguous().long().unsqueeze(0)
            mask = torch.from_numpy(mask).contiguous().float().unsqueeze(0)
            af = torch.from_numpy(af).permute(2, 0, 1).contiguous().float()
        else: # if labels not available, set ground truth tensors to nan values
            img, _ = self.transforms((img, img))
            # convert all outputs to torch tensors
            img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
            seg, mask, af = torch.tensor(float('nan')), torch.tensor(float('nan')), torch.tensor(float('nan'))

        return img, seg, mask, af

    def __len__(self):
        return len(self.img_list)
