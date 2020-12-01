import json
import os

import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class TuSimple(Dataset):
    def __init__(self, path, image_set):
        super(TuSimple, self).__init__()
        assert image_set in ('train', 'val', 'test'), "image_set is not valid!"
        self.input_size = (768, 1280)
        self.output_size = (192, 320)
        self.data_dir_path = path
        self.image_set = image_set
        self.img_transforms = transforms.Compose([transforms.Resize(self.input_size), 
            transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        self.create_index()

    def create_index(self):
        self.img_list = []
        self.segLabel_list = []
        self.AFLabel_list = []

        listfile = os.path.join(self.data_dir_path, "seg_label", "list", "{}_gt.txt".format(self.image_set))
        if not os.path.exists(listfile):
            raise FileNotFoundError("List file doesn't exist. Label has to be generated! ...")

        with open(listfile) as f:
            for line in f:
                line = line.strip()
                l = line.split(" ")
                self.img_list.append(os.path.join(self.data_dir_path, l[0][1:]))  # l[0][1:]  get rid of the first '/' so as for os.path.join
                self.seg_list.append(os.path.join(self.data_dir_path, l[1][1:]))
                self.af_list.append(os.path.join(self.data_dir_path, l[1][1:-3] + 'npy'))

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx])
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        seg = cv2.resize(cv2.imread(self.seg_list[idx]), (self.output_size[1], self.output_size[0]))[:, :, 0]
        af = cv2.resize(np.load(self.af_list[idx]), (self.output_size[1], self.output_size[0]))
        sample = {'img': img,
                  'img_name': self.img_list[idx],
                  'seg': seg,
                  'vaf': af[:, :, :2],
                  'haf': af[:, :, 2]}

        if self.img_transforms is not None:
            sample['img'] = self.img_transforms(sample['img'])
            
        sample['mask'] = (sample['seg'] >= 1).astype(np.uint8)
        return sample

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate(batch):
        if isinstance(batch[0]['img'], torch.Tensor):
            img = torch.stack([b['img'] for b in batch])
        else:
            img = [b['img'] for b in batch]

        if batch[0]['seg'] is None:
            seg = None
        elif isinstance(batch[0]['seg'], torch.Tensor):
            seg = torch.stack([b['seg'] for b in batch])
        else:
            seg = [b['seg'] for b in batch]

        if batch[0]['mask'] is None:
            mask = None
        elif isinstance(batch[0]['mask'], torch.Tensor):
            mask = torch.stack([b['mask'] for b in batch])
        else:
            mask = [b['mask'] for b in batch]

        if batch[0]['vaf'] is None:
            vaf = None
        elif isinstance(batch[0]['vaf'], torch.Tensor):
            vaf = torch.stack([b['vaf'] for b in batch])
        else:
            vaf = [b['vaf'] for b in batch]

        if batch[0]['haf'] is None:
            haf = None
        elif isinstance(batch[0]['haf'], torch.Tensor):
            haf = torch.stack([b['haf'] for b in batch])
        else:
            haf = [b['haf'] for b in batch]

        samples = {'img': img,
                   'img_name': [x['img_name'] for x in batch],
                   'seg': seg,
                   'mask': mask,
                   'vaf': vaf,
                   'haf': haf}

        return samples
