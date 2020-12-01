import os
import glob
import argparse

import cv2
import numpy as np

import affinity_fields as af


def generate_affinity_fields(dataset_dir):
    glob_pattern = os.path.join(dataset_dir, 'seg_label', '*', '*', '*.png')
    im_paths = sorted(glob.glob(glob_pattern))
    for i, f in enumerate(im_paths):
        label = cv2.imread(f)
        generatedVAFs, generatedHAFs = af.generateAFs(label[:, :, 0], viz=False)
        generatedAFs = np.dstack((generatedVAFs, generatedHAFs[:, :, 0]))
        np.save(f[:-3] + 'npy', generatedAFs)
        print('Generated affinity fields for image %d/%d...' % (i+1, len(im_paths)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate and store affinity fields for entire dataset')
    parser.add_argument('-o', '--dataset-dir', default='/home/akshay/data/tusimple',
                        help='The dataset directory [default "/home/akshay/data/tusimple"]')

    args = parser.parse_args()
    print('Creating affinity fields...')
    generate_affinity_fields(dataset_dir=args.dataset_dir)
