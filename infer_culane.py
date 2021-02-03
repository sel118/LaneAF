import os
import json
from datetime import datetime
from statistics import mean
import argparse

import numpy as np
import cv2
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.utils.data import DataLoader

from datasets.culane import CULane
from models.dla.pose_dla_dcn import get_pose_net
from utils.affinity_fields import decodeAFs
from utils.metrics import match_multi_class
from utils.visualize import create_viz, tensor2image


parser = argparse.ArgumentParser('Options for inference with lane detection models in PyTorch...')
parser.add_argument('--dataset-dir', type=str, default=None, help='path to dataset')
parser.add_argument('--output-dir', type=str, default=None, help='output directory for model and logs')
parser.add_argument('--snapshot', type=str, default=None, help='path to pre-trained model snapshot')
parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--no-cuda', action='store_true', default=False, help='do not use cuda for training')
parser.add_argument('--save-viz', action='store_true', default=False, help='save visualization depicting intermediate and final results')


args = parser.parse_args()
# check args
if args.dataset_dir is None:
    assert False, 'Path to dataset not provided!'
if args.snapshot is None:
    assert False, 'Model snapshot not provided!'

# set batch size to 1 for visualization purposes
args.batch_size = 1

# setup args
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.output_dir is None:
    args.output_dir = datetime.now().strftime("%Y-%m-%d-%H:%M-infer")
    args.output_dir = os.path.join('.', 'experiments', 'culane', args.output_dir)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
else:
    assert False, 'Output directory already exists!'

# store config in output directory
with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
    json.dump(vars(args), f)

# set random seed
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 6}
test_loader = DataLoader(CULane(args.dataset_dir, 'test', False), **kwargs)

# create file handles
f_log = open(os.path.join(args.output_dir, "logs.txt"), "w")


# test function
def test(net):
    epoch_acc, epoch_multi_acc, epoch_f1 = list(), list(), list()
    net.eval()
    out_vid = None

    for idx, sample in enumerate(test_loader):
        if args.cuda:
            sample['img'] = sample['img'].cuda()
            sample['seg'] = sample['seg'].cuda()
            sample['mask'] = sample['mask'].cuda()
            sample['vaf'] = sample['vaf'].cuda()
            sample['haf'] = sample['haf'].cuda()

        # do the forward pass
        outputs = net(sample['img'])[-1]

        # convert to arrays
        img = tensor2image(sample['img'].detach(), np.array(test_loader.dataset.mean), 
            np.array(test_loader.dataset.std))
        mask_out = tensor2image(torch.sigmoid(outputs['hm']).repeat(1, 3, 1, 1).detach(), 
            np.array([0.0 for _ in range(3)], dtype='float32'), np.array([1.0 for _ in range(3)], dtype='float32'))
        vaf_out = np.transpose(outputs['vaf'][0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))
        haf_out = np.transpose(outputs['haf'][0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))
        #mask_out = tensor2image(sample['mask'].repeat(1, 3, 1, 1).detach(), np.array([0.0 for _ in range(3)], 
        #    dtype='float32'), np.array([1.0 for _ in range(3)], dtype='float32'))
        #vaf_out = np.transpose(sample['vaf'][0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))
        #haf_out = np.transpose(sample['haf'][0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))

        # decode AFs to get lane instances
        seg_out = decodeAFs(mask_out[:, :, 0], vaf_out, haf_out, threshold=0.5, viz=False)
        # re-assign lane IDs to match with ground truth
        seg_out = match_multi_class(seg_out.astype(np.int64), sample['seg'][0, 0, :, :].detach().cpu().numpy().astype(np.int64))

        if args.save_viz:
            img_out = create_viz(img, seg_out.astype(np.uint8), mask_out, vaf_out, haf_out)

            if out_vid is None:
                out_vid = cv2.VideoWriter(os.path.join(args.output_dir, 'out.avi'), 
                    cv2.VideoWriter_fourcc(*'XVID'), 5, (img_out.shape[1], img_out.shape[0]))
            out_vid.write(img_out)

        pred = torch.sigmoid(outputs['hm']).detach().cpu().numpy().ravel()
        target = sample['mask'].detach().cpu().numpy().ravel()
        test_acc = accuracy_score((pred > 0.5).astype(np.int64), (target > 0.5).astype(np.int64))
        test_f1 = f1_score((target > 0.5).astype(np.int64), (pred > 0.5).astype(np.int64))
        epoch_acc.append(test_acc)
        epoch_f1.append(test_f1)

        pred = seg_out.ravel().astype(np.int64)
        target = sample['seg'][0, 0, :, :].detach().cpu().numpy().ravel().astype(np.int64)
        test_acc = accuracy_score(pred, target)
        epoch_multi_acc.append(test_acc)

        print('Done with image {} out of {}...'.format(min(args.batch_size*(idx+1), len(test_loader.dataset)), len(test_loader.dataset)))

    # calculate statistics and store logs
    avg_acc = mean(epoch_acc)
    avg_multi_acc = mean(epoch_multi_acc)
    avg_f1 = mean(epoch_f1)
    print("\n------------------------ Test set metrics ------------------------")
    f_log.write("\n------------------------ Test set metrics ------------------------\n")
    print("Average accuracy = {:.4f}".format(avg_acc))
    f_log.write("Average accuracy = {:.4f}\n".format(avg_acc))
    print("Average multi-class accuracy = {:.4f}".format(avg_multi_acc))
    f_log.write("Average multi-class accuracy = {:.4f}\n".format(avg_multi_acc))
    print("Average F1 score = {:.4f}".format(avg_f1))
    f_log.write("Average F1 score = {:.4f}\n".format(avg_f1))
    print("--------------------------------------------------------------------\n")
    f_log.write("--------------------------------------------------------------------\n\n")

    if args.save_viz:
        out_vid.release()

    return avg_acc, avg_multi_acc, avg_f1
            
if __name__ == "__main__":
    heads = {'hm': 1, 'vaf': 2, 'haf': 1}
    model = get_pose_net(num_layers=34, heads=heads, head_conv=256, down_ratio=4)

    model.load_state_dict(torch.load(args.snapshot), strict=True)
    if args.cuda:
        model.cuda()
    print(model)

    avg_acc, avg_f1 = test(model)
    f_log.close()
