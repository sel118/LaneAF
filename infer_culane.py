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

from datasets.culane import CULane, get_lanes_culane
from models.dla.pose_dla_dcn import get_pose_net
from models.erfnet.erfnet import ERFNet
from models.enet.ENet import ENet
from utils.affinity_fields import decodeAFs
from utils.metrics import match_multi_class
from utils.visualize import tensor2image, create_viz


parser = argparse.ArgumentParser('Options for inference with LaneAF models in PyTorch...')
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

# load args used from training snapshot (if available)
if os.path.exists(os.path.join(os.path.dirname(args.snapshot), 'config.json')):
    with open(os.path.join(os.path.dirname(args.snapshot), 'config.json')) as f:
        json_args = json.load(f)
    # augment infer args with training args for model consistency
    if 'backbone' in json_args.keys():
        args.backbone = json_args['backbone']
    else:
        args.backbone = 'dla34'

# store config in output directory
with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
    json.dump(vars(args), f)

# set random seed
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 1}
test_loader = DataLoader(CULane(args.dataset_dir, 'test', False), **kwargs)

# create file handles
f_log = open(os.path.join(args.output_dir, "logs.txt"), "w")

# get test set filenames
with open(os.path.join(args.dataset_dir, "list", "test.txt")) as f:
    filenames = f.readlines()
filenames = [x.strip() for x in filenames] 


# test function
def test(net):
    net.eval()
    out_vid = None

    for b_idx, sample in enumerate(test_loader):
        input_img, input_seg, input_mask, input_af = sample
        if args.cuda:
            input_img = input_img.cuda()
            input_seg = input_seg.cuda()
            input_mask = input_mask.cuda()
            input_af = input_af.cuda()

        # do the forward pass
        outputs = net(input_img)[-1]

        # convert to arrays
        img = tensor2image(input_img.detach(), np.array(test_loader.dataset.mean), 
            np.array(test_loader.dataset.std))
        mask_out = tensor2image(torch.sigmoid(outputs['hm']).repeat(1, 3, 1, 1).detach(), 
            np.array([0.0 for _ in range(3)], dtype='float32'), np.array([1.0 for _ in range(3)], dtype='float32'))
        vaf_out = np.transpose(outputs['vaf'][0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))
        haf_out = np.transpose(outputs['haf'][0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))

        # decode AFs to get lane instances
        seg_out = decodeAFs(mask_out[:, :, 0], vaf_out, haf_out, fg_thresh=128, err_thresh=5)

        if torch.any(torch.isnan(input_seg)):
            # if labels are not available, skip this step
            pass
        else:
            # if test set labels are available
            # re-assign lane IDs to match with ground truth
            seg_out = match_multi_class(seg_out.astype(np.int64), input_seg[0, 0, :, :].detach().cpu().numpy().astype(np.int64))

        # get results in output structure
        xy_coords = get_lanes_culane(seg_out, test_loader.dataset.samp_factor)

        # write results to file
        if not os.path.exists(os.path.join(args.output_dir, 'outputs', os.path.dirname(filenames[b_idx][1:]))):
            os.makedirs(os.path.join(args.output_dir, 'outputs', os.path.dirname(filenames[b_idx][1:])))
        with open(os.path.join(args.output_dir, 'outputs', filenames[b_idx][1:-3]+'lines.txt'), 'w') as f:
            f.write('\n'.join(' '.join(map(str, _lane)) for _lane in xy_coords))

        # create video visualization
        if args.save_viz:
            img_out = create_viz(img, seg_out.astype(np.uint8), mask_out, vaf_out, haf_out)

            if out_vid is None:
                out_vid = cv2.VideoWriter(os.path.join(args.output_dir, 'out.mkv'), 
                    cv2.VideoWriter_fourcc(*'H264'), 5, (img_out.shape[1], img_out.shape[0]))
            out_vid.write(img_out)

        print('Done with image {} out of {}...'.format(min(args.batch_size*(b_idx+1), len(test_loader.dataset)), len(test_loader.dataset)))

    if args.save_viz:
        out_vid.release()

    return

if __name__ == "__main__":
    heads = {'hm': 1, 'vaf': 2, 'haf': 1}
    if args.backbone == 'dla34':
        model = get_pose_net(num_layers=34, heads=heads, head_conv=256, down_ratio=4)
    elif args.backbone == 'erfnet':
        model = ERFNet(heads=heads)
    elif args.backbone == 'enet':
        model = ENet(heads=heads)

    model.load_state_dict(torch.load(args.snapshot), strict=True)
    if args.cuda:
        model.cuda()
    print(model)

    test(model)
