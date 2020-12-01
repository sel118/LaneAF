import os
import shutil
import glob
import json
import argparse

import cv2
import numpy as np

import affinity_fields as af


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
            paf_path = "/".join([save_dir, *img_path.split("/")[1:3], img_name[:-3]+"npy"])
            if paf_path[0] != '/':
                paf_path = '/' + paf_path
            if seg_path[0] != '/':
                seg_path = '/' + seg_path
            if img_path[0] != '/':
                img_path = '/' + img_path
            list_str.insert(0, paf_path)
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
                        help='The dataset directory ["/path/to/tusimple"]')

    args = parser.parse_args()
    print('Creating labels...')
    generate_labels(args.dataset_dir)
    print('Creating affinity fields...')
    generate_affinity_fields(args.dataset_dir)
