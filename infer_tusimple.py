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

from datasets.tusimple import TuSimple
from models.dla.pose_dla_dcn import get_pose_net
from utils.affinity_fields import decodeAFs
from utils.metrics import match_multi_class, LaneEval
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
    args.output_dir = os.path.join('.', 'experiments', 'tusimple', args.output_dir)

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

kwargs = {'batch_size': args.batch_size, 'shuffle': False, 'num_workers': 1}
test_loader = DataLoader(TuSimple(args.dataset_dir, 'test', False), **kwargs)

# create file handles
f_log = open(os.path.join(args.output_dir, "logs.txt"), "w")


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

def get_lanes_tusimple(seg_out, h_samples, target_ids):
    lanes = [[] for t_id in target_ids]
    for y_ip in h_samples:
        _, y_op = coord_ip_to_op(None, y_ip, test_loader.dataset.samp_factor)
        for idx, t_id in enumerate(target_ids):
            x_op = np.where(seg_out[y_op, :] == t_id)[0]
            if x_op.size > 0:
                x_op = np.mean(x_op)
                x_ip, _ = coord_op_to_ip(x_op, None, test_loader.dataset.samp_factor)
            else:
                x_ip = -2
            lanes[idx].append(x_ip)
    return lanes

# test function
def test(net):
    net.eval()
    out_vid = None
    json_pred = [json.loads(line) for line in open(os.path.join(args.dataset_dir, 'test_baseline.json')).readlines()]

    for b_idx, sample in enumerate(test_loader):
        if args.cuda:
            sample['img'] = sample['img'].cuda()
            sample['seg'] = sample['seg'].cuda()
            sample['mask'] = sample['mask'].cuda()
            sample['vaf'] = sample['vaf'].cuda()
            sample['haf'] = sample['haf'].cuda()

        st_time = datetime.now()
        # do the forward pass
        outputs = net(sample['img'])[-1]

        # convert to arrays
        img = tensor2image(sample['img'].detach(), np.array(test_loader.dataset.mean), 
            np.array(test_loader.dataset.std))
        mask_out = tensor2image(torch.sigmoid(outputs['hm']).repeat(1, 3, 1, 1).detach(), 
            np.array([0.0 for _ in range(3)], dtype='float32'), np.array([1.0 for _ in range(3)], dtype='float32'))
        vaf_out = np.transpose(outputs['vaf'][0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))
        haf_out = np.transpose(outputs['haf'][0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))

        # decode AFs to get lane instances
        seg_out = decodeAFs(mask_out[:, :, 0], vaf_out, haf_out, fg_thresh=128, err_thresh=10)
        ed_time = datetime.now()

        # re-assign lane IDs to match with ground truth
        seg_out, target_ids = match_multi_class(seg_out.astype(np.int64), sample['seg'][0, 0, :, :].detach().cpu().numpy().astype(np.int64))

        # fill results in output structure
        json_pred[b_idx]['run_time'] = (ed_time - st_time).total_seconds()*1000.
        json_pred[b_idx]['lanes'] = get_lanes_tusimple(seg_out, json_pred[b_idx]['h_samples'], target_ids)

        # write results to file
        with open(os.path.join(args.output_dir, 'outputs.json'), 'a') as f:
            json.dump(json_pred[b_idx], f)
            f.write('\n')

        if args.save_viz:
            img_out = create_viz(img, seg_out.astype(np.uint8), mask_out, vaf_out, haf_out)

            if out_vid is None:
                out_vid = cv2.VideoWriter(os.path.join(args.output_dir, 'out.mkv'), 
                    cv2.VideoWriter_fourcc(*'H264'), 5, (img_out.shape[1], img_out.shape[0]))
            out_vid.write(img_out)

        print('Done with image {} out of {}...'.format(min(args.batch_size*(b_idx+1), len(test_loader.dataset)), len(test_loader.dataset)))

    # benchmark on TuSimple
    results = LaneEval.bench_one_submit(os.path.join(args.output_dir, 'outputs.json'), os.path.join(args.dataset_dir, 'test_label.json'))
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f)

    if args.save_viz:
        out_vid.release()

    return

if __name__ == "__main__":
    heads = {'hm': 1, 'vaf': 2, 'haf': 1}
    model = get_pose_net(num_layers=34, heads=heads, head_conv=256, down_ratio=4)

    model.load_state_dict(torch.load(args.snapshot), strict=True)
    if args.cuda:
        model.cuda()
    print(model)

    test(model)
