import os
import shutil
import glob
import json
import argparse
from math import floor, ceil

import cv2
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils.affinity_fields import generateAFs
import datasets.transforms as tf
from datasets.third_party.spline_creator import get_horizontal_values_for_four_lanes


def coord_op_to_ip(x, y, scale):
    # (160*scale, 88*scale) --> (160*scale, 88*scale+13) --> (1280, 717) --> (1276, 717)
    if x is not None:
        x = scale*x
        x = int(x*1276./1280.)
    if y is not None:
        y = int(scale*y+13)
    return x, y

def coord_ip_to_op(x, y, scale):
    # (1276, 717) --> (1280, 717) --> (1280, 717-13=704) --> (1280/scale, 704/scale)
    if x is not None:
        x = x*1280./1276.
        x = int(x/scale)
    if y is not None:
        y = int((y-13)/scale)
    return x, y

def match_multi_class(pred):
    pred_ids = np.unique(pred[pred > 0]) # find unique pred ids
    pred_out = np.zeros_like(pred) # initialize output array

    # return input array if no lane points in prediction/target
    if pred_ids.size == 0:
        return pred

    # sort lanes based on their size
    lane_num_pixels = [np.sum(pred == ids) for ids in pred_ids]
    ret_lane_ids = pred_ids[np.argsort(lane_num_pixels)[::-1]]
    # retain a maximum of 4 lanes
    if ret_lane_ids.size > 4:
        print("Detected more than 4 lanes")
        ret_lane_ids = ret_lane_ids[:4]
    elif ret_lane_ids.size < 4:
        print("Detected fewer than 4 lanes")

    # sort lanes based on their location
    lane_max_x = [np.median(np.nonzero(np.sum(pred == ids, axis=0))[0]) for ids in ret_lane_ids]
    ret_lane_ids = ret_lane_ids[np.argsort(lane_max_x)]

    # assign new IDs to lanes
    for i, r_id in enumerate(ret_lane_ids):
        pred_out[pred == r_id] = i+1

    return pred_out

def get_lanes_culane(seg_out, samp_factor):
    # fit cubic spline to each lane
    h_samples = range(300, 717, 1)
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
        if len(xs) >= 10:
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

def get_lanes_llamas(seg_out, samp_factor):
    # fit cubic spline to each lane
    h_samples = range(300, 717, 1)
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
        if len(xs) >= 10:
            cs.append(CubicSpline(ys, xs, extrapolate=False))
        else:
            cs.append(None)
    # get x-coordinates from fitted spline
    lanes_str = ['l1', 'l0', 'r0', 'r1']
    lanes = dict()
    for idx, t_id in enumerate(lane_ids):
        lane = []
        if cs[idx] is not None:
            y_out = np.array(h_samples)
            x_out = cs[idx](y_out)
            for _x, _y in zip(x_out, y_out):
                if np.isnan(_x):
                    lane += [-1]
                else:
                    lane += [_x]
            lanes[lanes_str[idx]] = lane
    return lanes

def create_interp_segmentation_label(json_path):
    """ Creates pixel-level label of markings color coded by lane association
    Only for the for closest lane dividers, i.e. l1, l0, r0, r1

    Parameters
    ----------
    json_path: str
               path to label file

    Returns
    -------
    numpy.array
        pixel level segmentation with interpolated lanes (717, 1276)

    Notes
    -----
    Only draws 4 classes, can easily be extended for to a given number of lanes
    """
    seg = np.zeros((717, 1276, 3), dtype=np.uint8) # initialize output array
    lanes = get_horizontal_values_for_four_lanes(json_path) # get projected lane centers

    for r in range(716):
        for i, lane in enumerate(lanes):
            if lane[r] >= 0 and lane[r+1] >= 0:
                # similar to CULane, draw lines with 16 pixel width
                seg = cv2.line(seg, (round(lane[r]), r), (round(lane[r+1]), r+1), (i+1, i+1, i+1), thickness=16)

    return seg[:, :, 0]

class Llamas(Dataset):
    def __init__(self, path, image_set='train', random_transforms=False):
        super(Llamas, self).__init__()
        assert image_set in ('train', 'valid', 'test'), "image_set is incorrect!"
        self.input_size = (352, 640) # original image res: (717, 1276) -> (717-13, 1276+4)/2
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
        self.img_list = sorted(glob.glob(os.path.join(self.data_dir_path, 'color_images', self.image_set, '*', '*.png')))
        if self.image_set == 'test':
            self.seg_list = [None for x in self.img_list]
        else:
            self.seg_list = [x.replace('color_images', 'labels').replace('_color_rect.png', '.json') for x in self.img_list]

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx]).astype(np.float32)/255. # (H, W, 3)
        img = cv2.resize(img[13:, :, :], (1280, 704), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.seg_list[idx] is not None:
            seg = create_interp_segmentation_label(self.seg_list[idx]) # (H, W)
            seg = np.tile(seg[..., np.newaxis], (1, 1, 3)) # (H, W, 3)
            seg = cv2.resize(seg[13:, :, :], (1280, 704), interpolation=cv2.INTER_NEAREST)
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
        else: # if testing, set ground truth tensors to nan values
            img, _ = self.transforms((img, img))
            # convert all outputs to torch tensors
            img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
            seg, mask, af = torch.tensor(float('nan')), torch.tensor(float('nan')), torch.tensor(float('nan'))

        return img, seg, mask, af

    def __len__(self):
        return len(self.img_list)
