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
    # (160*scale, 88*scale) --> (160*scale, 88*scale+16=720) --> (1280, 720)
    if x is not None:
        x = int(scale*x)
    if y is not None:
        y = int(scale*y+16)
    return x, y

def coord_ip_to_op(x, y, scale):
    # (1280, 720) --> (1280, 720-16=704) --> (1280/scale, 704/scale)
    if x is not None:
        x = int(x/scale)
    if y is not None:
        y = int((y-16)/scale)
    return x, y

def get_lanes_tusimple(seg_out, h_samples, samp_factor):
    pred_ids = np.unique(seg_out[seg_out > 0]) # find unique pred ids
    # sort lanes based on their size
    lane_num_pixels = [np.sum(seg_out == ids) for ids in pred_ids]
    ret_lane_ids = pred_ids[np.argsort(lane_num_pixels)[::-1]]
    # retain a maximum of 4 lanes
    if ret_lane_ids.size > 4:
        print("Detected more than 4 lanes")
        for rem_id in ret_lane_ids[4:]:
            seg_out[seg_out == rem_id] = 0
        ret_lane_ids = ret_lane_ids[:4]

    # fit cubic spline to each lane
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
        if cs[idx] is not None:
            x_out = cs[idx](np.array(h_samples))
            x_out[np.isnan(x_out)] = -2
            lanes.append(x_out.tolist())
        else:
            print("Lane too small, discarding...")
    return lanes

class TuSimple(Dataset):
    def __init__(self, path, image_set='train', random_transforms=False):
        super(TuSimple, self).__init__()
        assert image_set in ('train', 'val', 'test'), "image_set is not valid!"
        self.input_size = (352, 640)
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

        listfile = os.path.join(self.data_dir_path, "seg_label", "list", "{}_gt.txt".format(self.image_set))
        if not os.path.exists(listfile):
            raise FileNotFoundError("List file doesn't exist. Label has to be generated! ...")

        with open(listfile) as f:
            for line in f:
                line = line.strip()
                l = line.split(" ")
                self.img_list.append(os.path.join(self.data_dir_path, l[0][1:]))  # l[0][1:]  get rid of the first '/' so as for os.path.join
                self.seg_list.append(os.path.join(self.data_dir_path, l[1][1:]))

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx]).astype(np.float32)/255. # (H, W, 3)
        img = cv2.cvtColor(img[16:, :, :], cv2.COLOR_BGR2RGB)
        if os.path.exists(self.seg_list[idx]):
            seg = cv2.imread(self.seg_list[idx], cv2.IMREAD_UNCHANGED) # (H, W, 3)        
            seg = seg[16:, :, :]
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

def _gen_label_for_json(data_dir_path, image_set):
    H, W = 720, 1280
    SEG_WIDTH = 30
    save_dir = "seg_label"

    os.makedirs(os.path.join(data_dir_path, save_dir, "list"), exist_ok=True)
    list_f = open(os.path.join(data_dir_path, save_dir, "list", "{}_gt.txt".format(image_set)), "w")

    json_path = os.path.join(data_dir_path, save_dir, "{}.json".format(image_set))
    with open(json_path) as f:
        for line in f:
            label = json.loads(line)

            # ---------- clean and sort lanes -------------
            lanes = []
            _lanes = []
            slope = [] # identify 1st, 2nd, 3rd, 4th lane through slope
            for i in range(len(label['lanes'])):
                l = [(x, y) for x, y in zip(label['lanes'][i], label['h_samples']) if x >= 0]
                if (len(l)>1):
                    _lanes.append(l)
                    slope.append(np.arctan2(l[-1][1]-l[0][1], l[0][0]-l[-1][0]) / np.pi * 180)
            _lanes = [_lanes[i] for i in np.argsort(slope)]
            slope = [slope[i] for i in np.argsort(slope)]

            idx_1 = None
            idx_2 = None
            idx_3 = None
            idx_4 = None
            for i in range(len(slope)):
                if slope[i]<=90:
                    idx_2 = i
                    idx_1 = i-1 if i>0 else None
                else:
                    idx_3 = i
                    idx_4 = i+1 if i+1 < len(slope) else None
                    break
            lanes.append([] if idx_1 is None else _lanes[idx_1])
            lanes.append([] if idx_2 is None else _lanes[idx_2])
            lanes.append([] if idx_3 is None else _lanes[idx_3])
            lanes.append([] if idx_4 is None else _lanes[idx_4])
            # ---------------------------------------------

            img_path = label['raw_file']
            seg_img = np.zeros((H, W, 3))
            list_str = []  # str to be written to list.txt
            for i in range(len(lanes)):
                coords = lanes[i]
                if len(coords) < 4:
                    list_str.append('0')
                    continue
                for j in range(len(coords)-1):
                    cv2.line(seg_img, coords[j], coords[j+1], (i+1, i+1, i+1), SEG_WIDTH//2)
                list_str.append('1')

            seg_path = img_path.split("/")
            seg_path, img_name = os.path.join(data_dir_path, save_dir, seg_path[1], seg_path[2]), seg_path[3]
            os.makedirs(seg_path, exist_ok=True)
            seg_path = os.path.join(seg_path, img_name[:-3]+"png")
            cv2.imwrite(seg_path, seg_img)

            seg_path = "/".join([save_dir, *img_path.split("/")[1:3], img_name[:-3]+"png"])
            if seg_path[0] != '/':
                seg_path = '/' + seg_path
            if img_path[0] != '/':
                img_path = '/' + img_path
            list_str.insert(0, seg_path)
            list_str.insert(0, img_path)
            list_str = " ".join(list_str) + "\n"
            list_f.write(list_str)

    list_f.close()

def generate_labels(dataset_dir):
    """
    image_set is split into three partitions: train, val, test.
    train includes label_data_0313.json, label_data_0601.json
    val includes label_data_0531.json
    test includes test_label.json
    """
    TRAIN_SET = ['label_data_0313.json', 'label_data_0601.json']
    VAL_SET = ['label_data_0531.json']
    TEST_SET = ['test_label.json']
    save_dir = os.path.join(dataset_dir, "seg_label")
    if os.path.exists(save_dir):
        print("Deleting existing label directory...")
        shutil.rmtree(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    # --------- merge json into one file ---------
    with open(os.path.join(save_dir, "train.json"), "w") as outfile:
        for json_name in TRAIN_SET:
            with open(os.path.join(dataset_dir, json_name)) as infile:
                for line in infile:
                    outfile.write(line)

    with open(os.path.join(save_dir, "val.json"), "w") as outfile:
        for json_name in VAL_SET:
            with open(os.path.join(dataset_dir, json_name)) as infile:
                for line in infile:
                    outfile.write(line)

    with open(os.path.join(save_dir, "test.json"), "w") as outfile:
        for json_name in TEST_SET:
            with open(os.path.join(dataset_dir, json_name)) as infile:
                for line in infile:
                    outfile.write(line)

    _gen_label_for_json(dataset_dir, 'train')
    print("Finished generating labels for train set")
    _gen_label_for_json(dataset_dir, 'val')
    print("Finished generating labels for val set")
    _gen_label_for_json(dataset_dir, 'test')
    print("Finished generating labels for test set")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate and store labels for entire dataset')
    parser.add_argument('-o', '--dataset-dir', default='/home/akshay/data/TuSimple',
                        help='The dataset directory ["/path/to/TuSimple"]')

    args = parser.parse_args()
    print('Creating labels...')
    generate_labels(args.dataset_dir)
