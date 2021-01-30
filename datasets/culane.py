import os
import shutil
import glob
import json
import argparse

import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import affinity_fields as af


def preprocess_outputs(arr, samp_factor=8):
    arr = arr[14:, 20:-20, :]
    arr = arr[int(samp_factor/2)::samp_factor, int(samp_factor/2)::samp_factor, :]
    return arr

class CULane(Dataset):
    def __init__(self, path, image_set='train', random_transforms=False):
        super(CULane, self).__init__()
        assert image_set in ('train', 'val', 'test'), "image_set is not valid!"
        self.input_size = (288, 800) # original image res: (590, 1640) -> (590-14, 1640-40)/2
        self.samp_factor = 2*4
        self.data_dir_path = path
        self.image_set = image_set
        # convert numpy array (H, W, C), uint8 --> torch tensor (C, H, W), float32
        self.to_tensor = transforms.ToTensor()
        # normalization transform for input images
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize = transforms.Normalize(self.mean, self.std)
        # random transformations + resizing for inputs
        #self.output_size = (int(self.input_size[0]/4), int(self.input_size[1]/4))
        #if random_transforms:
        #    self.transform = transforms.Compose([transforms.RandomHorizontalFlip(),
        #        transforms.RandomRotation((-10, 10)), 
        #        transforms.RandomResizedCrop(self.input_size, scale=(0.7, 1.0))])
        #else:
        #    self.transform = transforms.Resize(self.input_size)
        # resizing for outputs
        #self.resize = transforms.Resize(self.output_size)
        self.transform = transforms.Resize(self.input_size)

        self.create_index()

    def create_index(self):
        self.img_list = []
        self.seg_list = []
        self.af_list = []

        listfile = os.path.join(self.data_dir_path, "list", "{}.txt".format(self.image_set))
        if not os.path.exists(listfile):
            raise FileNotFoundError("List file doesn't exist. Label has to be generated! ...")

        with open(listfile) as f:
            for line in f:
                l = line.strip()
                self.img_list.append(os.path.join(self.data_dir_path, l[1:]))  # l[1:]  get rid of the first '/' so as for os.path.join
                if self.image_set == 'test':
                    self.seg_list.append(os.path.join(self.data_dir_path, 'laneseg_label_w16_test', l[1:-3] + 'png'))
                    self.af_list.append(os.path.join(self.data_dir_path, 'laneseg_label_w16_test', l[1:-3] + 'npy'))
                else:
                    self.seg_list.append(os.path.join(self.data_dir_path, 'laneseg_label_w16', l[1:-3] + 'png'))
                    self.af_list.append(os.path.join(self.data_dir_path, 'laneseg_label_w16', l[1:-3] + 'npy'))

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.img_list[idx]), cv2.COLOR_BGR2RGB) # (H, W, 3)
        seg = cv2.imread(self.seg_list[idx]) # (H, W, 3)
        af = np.load(self.af_list[idx]) # (H, W, 3)
        img = img[14:, 20:-20, :]
        seg = preprocess_outputs(seg, self.samp_factor)

        # convert all outputs to float32 tensors of shape (C, H, W) in range [0, 1]
        sample = {'img': self.to_tensor(img), # (3, H, W)
                  'img_name': self.img_list[idx],
                  'seg': self.to_tensor(seg[:, :, 0:1].astype(np.float32)), # (1, H, W)
                  'mask': self.to_tensor((seg[:, :, 0:1] >= 1).astype(np.float32)),  # (1, H, W)
                  'vaf': self.to_tensor(af[:, :, :2].astype(np.float32)), # (2, H, W)
                  'haf': self.to_tensor(af[:, :, 2:3].astype(np.float32))} # (1, H, W)

        # apply normalization, transformations, and resizing to inputs and outputs
        sample['img'] = self.normalize(self.transform(sample['img']))
        sample['seg'] = sample['seg']
        sample['mask'] = sample['mask']
        sample['vaf'] = sample['vaf']
        sample['haf'] = sample['haf']
        #sample['img'] = self.normalize(self.transform(sample['img']))
        #sample['seg'] = self.resize(self.transform(sample['seg']))
        #sample['mask'] = self.resize(self.transform(sample['mask']))
        #sample['vaf'] = self.resize(self.transform(sample['vaf']))
        #sample['haf'] = self.resize(self.transform(sample['haf']))

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

def generate_affinity_fields(dataset_dir):
    glob_pattern = os.path.join(dataset_dir, 'laneseg_label_w16*', '*', '*', '*.png')
    im_paths = sorted(glob.glob(glob_pattern))
    for i, f in enumerate(im_paths):
        label = cv2.imread(f)
        label = preprocess_outputs(label)
        generatedVAFs, generatedHAFs = af.generateAFs(label[:, :, 0], viz=False)
        generatedAFs = np.dstack((generatedVAFs, generatedHAFs[:, :, 0]))
        np.save(f[:-3] + 'npy', generatedAFs)
        print('Generated affinity fields for image %d/%d...' % (i+1, len(im_paths)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate and store affinity fields for entire dataset')
    parser.add_argument('-o', '--dataset-dir', default='/home/akshay/data/CULane',
                        help='The dataset directory ["/path/to/CULane"]')

    args = parser.parse_args()
    print('Creating affinity fields...')
    generate_affinity_fields(args.dataset_dir)
