import json
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class TuSimple(Dataset):
    """
    image_set is splitted into three partitions: train, val, test.
    train includes label_data_0313.json, label_data_0601.json
    val includes label_data_0531.json
    test includes test_label.json
    """
    TRAIN_SET = ['label_data_0313.json', 'label_data_0601.json']
    VAL_SET = ['label_data_0531.json']
    TEST_SET = ['test_label.json']

    def __init__(self, path, image_set, img_transforms=None, generate_pafs = False):
        super(TuSimple, self).__init__()
        assert image_set in ('train', 'val', 'test'), "image_set is not valid!"
        self.data_dir_path = path
        self.image_set = image_set
        self.img_transforms = img_transforms
        self.generate_pafs = generate_pafs

        if not os.path.exists(os.path.join(path, "seg_label")):
            print("Label is going to get generated into dir: {} ...".format(os.path.join(path, "seg_label")))
            self.generate_label()
        self.createIndex()

    def createIndex(self):
        self.img_list = []
        self.segLabel_list = []

        listfile = os.path.join(self.data_dir_path, "seg_label", "list", "{}_gt.txt".format(self.image_set))
        if not os.path.exists(listfile):
            raise FileNotFoundError("List file doesn't exist. Label has to be generated! ...")

        with open(listfile) as f:
            for line in f:
                line = line.strip()
                l = line.split(" ")
                self.img_list.append(os.path.join(self.data_dir_path, l[0][1:]))  # l[0][1:]  get rid of the first '/' so as for os.path.join
                self.segLabel_list.append(os.path.join(self.data_dir_path, l[1][1:]))

    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        '''if self.image_set != 'test':
            segLabel = cv2.imread(self.segLabel_list[idx])[:,:,0]
        else:
            segLabel = None'''
        
        segLabel = cv2.imread(self.segLabel_list[idx])[:,:,0]
        sample = {'img': img,
                  'segLabel': segLabel,
                  'img_name': self.img_list[idx],
                  'binary_mask': segLabel}

        if self.img_transforms is not None:
            sample['img'] = self.img_transforms(sample['img'])
            
        sample['binary_mask'] = cv2.resize(sample['binary_mask'], (320, 192))
        sample['binary_mask'] = sample['binary_mask'][np.newaxis, ...]
        sample['segLabel'] = cv2.resize(sample['segLabel'], (320, 192))
        #call Generate PAF here before we add new axis to segLabel
        #if self.generate_pafs:
            #call generatePAFs here
        sample['segLabel'] = sample['segLabel'][np.newaxis, ...]
        #print(np.unique(sample['segLabel']))
        sample['binary_mask'][sample['binary_mask'] >= 1] = 1
        #sample['segLabel'] = sample['segLabel'][::4,::4]
        return sample

    def __len__(self):
        return len(self.img_list)

    def generate_label(self):
        save_dir = os.path.join(self.data_dir_path, "seg_label")
        os.makedirs(save_dir, exist_ok=True)

        # --------- merge json into one file ---------
        with open(os.path.join(save_dir, "train.json"), "w") as outfile:
            for json_name in self.TRAIN_SET:
                with open(os.path.join(self.data_dir_path, json_name)) as infile:
                    for line in infile:
                        outfile.write(line)

        with open(os.path.join(save_dir, "val.json"), "w") as outfile:
            for json_name in self.VAL_SET:
                with open(os.path.join(self.data_dir_path, json_name)) as infile:
                    for line in infile:
                        outfile.write(line)

        with open(os.path.join(save_dir, "test.json"), "w") as outfile:
            for json_name in self.TEST_SET:
                with open(os.path.join(self.data_dir_path, json_name)) as infile:
                    for line in infile:
                        outfile.write(line)

        self._gen_label_for_json('train')
        print("train set is done")
        self._gen_label_for_json('val')
        print("val set is done")
        self._gen_label_for_json('test')
        print("test set is done")

    def _gen_label_for_json(self, image_set):
        H, W = 720, 1280
        SEG_WIDTH = 30
        save_dir = "seg_label"

        os.makedirs(os.path.join(self.data_dir_path, save_dir, "list"), exist_ok=True)
        list_f = open(os.path.join(self.data_dir_path, save_dir, "list", "{}_gt.txt".format(image_set)), "w")

        json_path = os.path.join(self.data_dir_path, save_dir, "{}.json".format(image_set))
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
                seg_path, img_name = os.path.join(self.data_dir_path, save_dir, seg_path[1], seg_path[2]), seg_path[3]
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

    @staticmethod
    def collate(batch):
        if isinstance(batch[0]['img'], torch.Tensor):
            img = torch.stack([b['img'] for b in batch])
        else:
            img = [b['img'] for b in batch]

        if batch[0]['segLabel'] is None:
            segLabel = None
        elif isinstance(batch[0]['segLabel'], torch.Tensor):
            segLabel = torch.stack([b['segLabel'] for b in batch])
        else:
            segLabel = [b['segLabel'] for b in batch]

        samples = {'img': img,
                   'segLabel': segLabel,
                   'img_name': [x['img_name'] for x in batch]}

        return samples
    
    def generatePAFs(Label):
        LabelSize = Label.shape
    #Creating PAF array
        NumLanes = np.max(Label)
        PAFs = np.zeros((LabelSize[0], LabelSize[1], 2))

    '''prev_row = Label[0,:]
    prev_row_index = np.where(prev_row == 1)
    print(prev_row_index[0].size == 0)
    average_prev_row_index = np.average(prev_row_index)
    print(math.isnan(average_prev_row_index))'''

    #Generating PAFs
        for i in range(1, NumLanes + 1):
        #Searching for previous lane pixels that match with the lane we are looking for
            prev_row = Label[LabelSize[0] - 1,:]
            prev_row_index = np.where(prev_row == i)
            avg_prev_row_index = np.average(prev_row_index)

            for j in range(LabelSize[0] - 1, 0, -1):
                cur_row = Label[j - 1, :]
                cur_row_index = np.where(cur_row == i)
                avg_cur_row_index = np.average(cur_row_index)

            #if no lane pixels are found, then skip the PAF embedding
                if prev_row_index[0].size == 0 or cur_row_index[0].size == 0:
                    prev_row = cur_row
                    prev_row_index = cur_row_index
                    avg_prev_row_index = avg_cur_row_index
                    continue


            #We kept the convention of vec = [rows, cols] or [y, x] instead of [x, y]
                vec = np.array([1, avg_cur_row_index - avg_prev_row_index])
                unit_vec = vec / np.linalg.norm(vec)
                PAFs[j, prev_row_index, 0] = unit_vec[0]
                PAFs[j, prev_row_index,  1] = unit_vec[1]

            #Set previous row to current row to save computation time
                prev_row = cur_row
                prev_row_index = cur_row_index
                avg_prev_row_index = avg_cur_row_index
    
        return PAFs
    
    
def Preprocessing():
    img_transform = transforms.Compose([
      transforms.Resize((768, 1280)),
      transforms.ToTensor(),
      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    #directory of dataset
    root_dir = './data'
    batch_size = 3
    #initializing dataset and dataloader objects
    trainSet = TuSimple(path=root_dir, image_set='train', 
                                 img_transforms=img_transform)
    valSet = TuSimple(path=root_dir, image_set='val', 
                               img_transforms=img_transform)
    testSet = TuSimple(path=root_dir, image_set='test', 
                               img_transforms=img_transform)
    trainLoader = DataLoader(trainSet, batch_size=batch_size, shuffle=True, num_workers=4)
    ValLoader = DataLoader(valSet, batch_size=batch_size, shuffle=True, num_workers=4)
    testLoader = DataLoader(testSet, batch_size=batch_size, shuffle=True, num_workers=4)
    return trainLoader, ValLoader, testLoader
