import os
import json
from datetime import datetime
from statistics import mean
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from scipy.interpolate import CubicSpline

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.tusimple import TuSimple
from models.dla.pose_dla_dcn import get_pose_net
from models.loss import FocalLoss, IoULoss, RegL1Loss
from utils.affinity_fields import decodeAFs
from utils.metrics import match_multi_class, LaneEval
from utils.visualize import tensor2image


parser = argparse.ArgumentParser('Options for training LaneAF models in PyTorch...')
parser.add_argument('--dataset-dir', type=str, default=None, help='path to dataset')
parser.add_argument('--output-dir', type=str, default=None, help='output directory for model and logs')
parser.add_argument('--snapshot', type=str, default=None, help='path to pre-trained model snapshot')
parser.add_argument('--split', type=str, default='val', help='Split to validate on (val/test)')
parser.add_argument('--batch-size', type=int, default=2, metavar='N', help='batch size for training')
parser.add_argument('--epochs', type=int, default=60, metavar='N', help='number of epochs to train for')
parser.add_argument('--learning-rate', type=float, default=1e-4, metavar='LR', help='learning rate')
parser.add_argument('--weight-decay', type=float, default=1e-3, metavar='WD', help='weight decay')
parser.add_argument('--loss-type', type=str, default='wbce', help='Type of classification loss to use (focal/bce/wbce)')
parser.add_argument('--log-schedule', type=int, default=10, metavar='N', help='number of iterations to print/save log after')
parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--no-cuda', action='store_true', default=False, help='do not use cuda for training')
parser.add_argument('--random-transforms', action='store_true', default=False, help='apply random transforms to input during training')

args = parser.parse_args()
# check args
if args.dataset_dir is None:
    assert False, 'Path to dataset not provided!'

# setup args
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.output_dir is None:
    args.output_dir = datetime.now().strftime("%Y-%m-%d-%H:%M")
    args.output_dir = os.path.join('.', 'experiments', 'tusimple', args.output_dir)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
else:
    assert False, 'Output directory already exists!'

if args.split not in ['val', 'test']:
    assert False, 'Incorrect split provided!'

# store config in output directory
with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
    json.dump(vars(args), f)

# set random seed
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 6}
train_loader = DataLoader(TuSimple(args.dataset_dir, 'train', args.random_transforms), **kwargs)
kwargs = {'batch_size': 1, 'shuffle': False, 'num_workers': 1}
val_loader = DataLoader(TuSimple(args.dataset_dir, args.split, False), **kwargs)

# global var to store best validation F1 score across all epochs
best_acc = 0.0
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

def get_lanes_tusimple(seg_out, h_samples):
    cs = []
    lane_ids = np.unique(seg_out[seg_out > 0])
    for idx, t_id in enumerate(lane_ids):
        xs, ys = [], []
        for y_op in range(seg_out.shape[0]):
            x_op = np.where(seg_out[y_op, :] == t_id)[0]
            if x_op.size > 0:
                x_op = np.mean(x_op)
                x_ip, y_ip = coord_op_to_ip(x_op, y_op, val_loader.dataset.samp_factor)
                xs.append(x_ip)
                ys.append(y_ip)
        if len(xs) >= 2:
            cs.append(CubicSpline(ys, xs, extrapolate=False))
        else:
            cs.append(None)
    lanes = [[] for t_id in lane_ids]
    for idx, t_id in enumerate(lane_ids):
        if cs[idx] is not None:
            x_out = cs[idx](np.array(h_samples))
            x_out[np.isnan(x_out)] = -2
            lanes[idx] = x_out.tolist()
        else:
            lanes[idx] = [-2 for _ in h_samples]
            print("Lane completely missed!")
    return lanes

# training function
def train(net, epoch):
    epoch_loss_seg, epoch_loss_vaf, epoch_loss_haf, epoch_loss, epoch_acc, epoch_f1 = list(), list(), list(), list(), list(), list()
    net.train()
    for b_idx, sample in enumerate(train_loader):
        input_img, input_seg, input_mask, input_af = sample
        if args.cuda:
            input_img = input_img.cuda()
            input_seg = input_seg.cuda()
            input_mask = input_mask.cuda()
            input_af = input_af.cuda()

        # zero gradients before forward pass
        optimizer.zero_grad()

        # do the forward pass
        outputs = net(input_img)[-1]

        # calculate losses and metrics
        _mask = (input_mask != train_loader.dataset.ignore_label).float()
        loss_seg = criterion_1(outputs['hm']*_mask, input_mask*_mask) + criterion_2(torch.sigmoid(outputs['hm']), input_mask)
        loss_vaf = 0.5*criterion_reg(outputs['vaf'], input_af[:, :2, :, :], input_mask)
        loss_haf = 0.5*criterion_reg(outputs['haf'], input_af[:, 2:3, :, :], input_mask)
        pred = torch.sigmoid(outputs['hm']).detach().cpu().numpy().ravel()
        target = input_mask.detach().cpu().numpy().ravel()
        pred[target == train_loader.dataset.ignore_label] = 0
        target[target == train_loader.dataset.ignore_label] = 0
        train_acc = accuracy_score((pred > 0.5).astype(np.int64), (target > 0.5).astype(np.int64))
        train_f1 = f1_score((target > 0.5).astype(np.int64), (pred > 0.5).astype(np.int64), zero_division=1)

        epoch_loss_seg.append(loss_seg.item())
        epoch_loss_vaf.append(loss_vaf.item())
        epoch_loss_haf.append(loss_haf.item())
        loss = loss_seg + loss_vaf + loss_haf
        epoch_loss.append(loss.item())
        epoch_acc.append(train_acc)
        epoch_f1.append(train_f1)

        loss.backward()
        optimizer.step()
        if b_idx % args.log_schedule == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tF1-score: {:.4f}'.format(
                epoch, (b_idx+1) * args.batch_size, len(train_loader.dataset),
                100. * (b_idx+1) * args.batch_size / len(train_loader.dataset), loss.item(), train_f1))
            f_log.write('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tF1-score: {:.4f}\n'.format(
                epoch, (b_idx+1) * args.batch_size, len(train_loader.dataset),
                100. * (b_idx+1) * args.batch_size / len(train_loader.dataset), loss.item(), train_f1))

    scheduler.step()
    # now that the epoch is completed calculate statistics and store logs
    avg_loss_seg = mean(epoch_loss_seg)
    avg_loss_vaf = mean(epoch_loss_vaf)
    avg_loss_haf = mean(epoch_loss_haf)
    avg_loss = mean(epoch_loss)
    avg_acc = mean(epoch_acc)
    avg_f1 = mean(epoch_f1)
    print("\n------------------------ Training metrics ------------------------")
    f_log.write("\n------------------------ Training metrics ------------------------\n")
    print("Average segmentation loss for epoch = {:.2f}".format(avg_loss_seg))
    f_log.write("Average segmentation loss for epoch = {:.2f}\n".format(avg_loss_seg))
    print("Average VAF loss for epoch = {:.2f}".format(avg_loss_vaf))
    f_log.write("Average VAF loss for epoch = {:.2f}\n".format(avg_loss_vaf))
    print("Average HAF loss for epoch = {:.2f}".format(avg_loss_haf))
    f_log.write("Average HAF loss for epoch = {:.2f}\n".format(avg_loss_haf))
    print("Average loss for epoch = {:.2f}".format(avg_loss))
    f_log.write("Average loss for epoch = {:.2f}\n".format(avg_loss))
    print("Average accuracy for epoch = {:.4f}".format(avg_acc))
    f_log.write("Average accuracy for epoch = {:.4f}\n".format(avg_acc))
    print("Average F1 score for epoch = {:.4f}".format(avg_f1))
    f_log.write("Average F1 score for epoch = {:.4f}\n".format(avg_f1))
    print("------------------------------------------------------------------\n")
    f_log.write("------------------------------------------------------------------\n\n")
    
    return net, avg_loss_seg, avg_loss_vaf, avg_loss_haf, avg_loss, avg_acc, avg_f1

# validation function
def val(net, epoch):
    global best_acc
    epoch_loss_seg, epoch_loss_vaf, epoch_loss_haf, epoch_loss, epoch_acc, epoch_f1 = list(), list(), list(), list(), list(), list()
    net.eval()
    json_pred = [json.loads(line) for line in open(os.path.join(args.dataset_dir, 'seg_label', args.split+'.json')).readlines()]
    
    for b_idx, sample in enumerate(val_loader):
        input_img, input_seg, input_mask, input_af = sample
        if args.cuda:
            input_img = input_img.cuda()
            input_seg = input_seg.cuda()
            input_mask = input_mask.cuda()
            input_af = input_af.cuda()

        st_time = datetime.now()
        # do the forward pass
        outputs = net(input_img)[-1]

        # convert to arrays
        mask_out = tensor2image(torch.sigmoid(outputs['hm']).repeat(1, 3, 1, 1).detach(), 
            np.array([0.0 for _ in range(3)], dtype='float32'), np.array([1.0 for _ in range(3)], dtype='float32'))
        vaf_out = np.transpose(outputs['vaf'][0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))
        haf_out = np.transpose(outputs['haf'][0, :, :, :].detach().cpu().float().numpy(), (1, 2, 0))

        # decode AFs to get lane instances
        seg_out = decodeAFs(mask_out[:, :, 0], vaf_out, haf_out, fg_thresh=128, err_thresh=10)
        ed_time = datetime.now()

        # re-assign lane IDs to match with ground truth
        seg_out = match_multi_class(seg_out.astype(np.int64), input_seg[0, 0, :, :].detach().cpu().numpy().astype(np.int64))

        # fill results in output structure
        json_pred[b_idx]['run_time'] = (ed_time - st_time).total_seconds()*1000.
        json_pred[b_idx]['lanes'] = get_lanes_tusimple(seg_out, json_pred[b_idx]['h_samples'])

        # write results to file
        with open(os.path.join(args.output_dir, 'outputs.json'), 'a') as f:
            json.dump(json_pred[b_idx], f)
            f.write('\n')

        # calculate losses and metrics
        _mask = (input_mask != val_loader.dataset.ignore_label).float()
        loss_seg = criterion_1(outputs['hm'], input_mask) + criterion_2(torch.sigmoid(outputs['hm']), input_mask)
        loss_vaf = 0.5*criterion_reg(outputs['vaf'], input_af[:, :2, :, :], input_mask)
        loss_haf = 0.5*criterion_reg(outputs['haf'], input_af[:, 2:3, :, :], input_mask)
        pred = torch.sigmoid(outputs['hm']).detach().cpu().numpy().ravel()
        target = input_mask.detach().cpu().numpy().ravel()
        pred[target == val_loader.dataset.ignore_label] = 0
        target[target == val_loader.dataset.ignore_label] = 0
        val_acc = accuracy_score((pred > 0.5).astype(np.int64), (target > 0.5).astype(np.int64))
        val_f1 = f1_score((target > 0.5).astype(np.int64), (pred > 0.5).astype(np.int64), zero_division=1)

        epoch_loss_seg.append(loss_seg.item())
        epoch_loss_vaf.append(loss_vaf.item())
        epoch_loss_haf.append(loss_haf.item())
        loss = loss_seg + loss_vaf + loss_haf
        epoch_loss.append(loss.item())
        epoch_acc.append(val_acc)
        epoch_f1.append(val_f1)

        print('Done with image {} out of {}...'.format(min(args.batch_size*(b_idx+1), len(val_loader.dataset)), len(val_loader.dataset)))

    # benchmark on TuSimple
    results = LaneEval.bench_one_submit(os.path.join(args.output_dir, 'outputs.json'), os.path.join(args.dataset_dir, 'seg_label', args.split+'.json'))
    bench_acc = json.loads(results)[0]['value']
    os.remove(os.path.join(args.output_dir, 'outputs.json'))

    # now that the epoch is completed calculate statistics and store logs
    avg_loss_seg = mean(epoch_loss_seg)
    avg_loss_vaf = mean(epoch_loss_vaf)
    avg_loss_haf = mean(epoch_loss_haf)
    avg_loss = mean(epoch_loss)
    avg_acc = mean(epoch_acc)
    avg_f1 = mean(epoch_f1)
    print("\n------------------------ Validation metrics ------------------------")
    f_log.write("\n------------------------ Validation metrics ------------------------\n")
    print("Average segmentation loss for epoch = {:.2f}".format(avg_loss_seg))
    f_log.write("Average segmentation loss for epoch = {:.2f}\n".format(avg_loss_seg))
    print("Average VAF loss for epoch = {:.2f}".format(avg_loss_vaf))
    f_log.write("Average VAF loss for epoch = {:.2f}\n".format(avg_loss_vaf))
    print("Average HAF loss for epoch = {:.2f}".format(avg_loss_haf))
    f_log.write("Average HAF loss for epoch = {:.2f}\n".format(avg_loss_haf))
    print("Average loss for epoch = {:.2f}".format(avg_loss))
    f_log.write("Average loss for epoch = {:.2f}\n".format(avg_loss))
    print("Average accuracy for epoch = {:.4f}".format(avg_acc))
    f_log.write("Average accuracy for epoch = {:.4f}\n".format(avg_acc))
    print("Average F1 score for epoch = {:.4f}".format(avg_f1))
    f_log.write("Average F1 score for epoch = {:.4f}\n".format(avg_f1))
    print("Benchmark accuracy for epoch = {:.4f}".format(bench_acc))
    f_log.write("Benchmark accuracy for epoch = {:.4f}\n".format(bench_acc))
    print("--------------------------------------------------------------------\n")
    f_log.write("--------------------------------------------------------------------\n\n")

    # now save the model if it has a better F1 score than the best model seen so forward
    if bench_acc > best_acc:
        # save the model
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'net_' + '%.4d' % (epoch,) + '.pth'))
        best_acc = bench_acc

    return avg_loss_seg, avg_loss_vaf, avg_loss_haf, avg_loss, avg_acc, avg_f1, bench_acc

if __name__ == "__main__":
    heads = {'hm': 1, 'vaf': 2, 'haf': 1}
    model = get_pose_net(num_layers=34, heads=heads, head_conv=256, down_ratio=4)

    if args.snapshot is not None:
        model.load_state_dict(torch.load(args.snapshot), strict=True)
    if args.cuda:
        model.cuda()
    print(model)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    # BCE(Focal) loss applied to each pixel individually
    model.hm[2].bias.data.uniform_(-4.595, -4.595) # bias towards negative class
    if args.loss_type == 'focal':
        criterion_1 = FocalLoss(gamma=2.0, alpha=0.25, size_average=True)
    elif args.loss_type == 'bce':
        ## BCE weight
        criterion_1 = torch.nn.BCEWithLogitsLoss()
    elif args.loss_type == 'wbce':
        ## BCE weight
        criterion_1 = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9.6]).cuda())
    criterion_2 = IoULoss()
    criterion_reg = RegL1Loss()

    # set up figures and axes
    fig1, ax1 = plt.subplots()
    plt.grid(True)
    ax1.plot([], 'r', label='Training segmentation loss')
    ax1.plot([], 'g', label='Training VAF loss')
    ax1.plot([], 'b', label='Training HAF loss')
    ax1.plot([], 'k', label='Training total loss')
    ax1.legend()
    train_loss_seg, train_loss_vaf, train_loss_haf, train_loss = list(), list(), list(), list()

    fig2, ax2 = plt.subplots()
    plt.grid(True)
    ax2.plot([], 'r', label='Validation segmentation loss')
    ax2.plot([], 'g', label='Validation VAF loss')
    ax2.plot([], 'b', label='Validation HAF loss')
    ax2.plot([], 'k', label='Validation total loss')
    ax2.legend()
    val_loss_seg, val_loss_vaf, val_loss_haf, val_loss = list(), list(), list(), list()

    fig3, ax3 = plt.subplots()
    plt.grid(True)
    ax3.plot([], 'r', label='Training accuracy')
    ax3.plot([], 'g', label='Validation accuracy')
    ax3.plot([], 'b', label='Training F1 score')
    ax3.plot([], 'k', label='Validation F1 score')
    ax3.legend()
    train_acc, val_acc, train_f1, val_f1 = list(), list(), list(), list()

    # trainval loop
    for i in range(1, args.epochs + 1):
        # training epoch
        model, avg_loss_seg, avg_loss_vaf, avg_loss_haf, avg_loss, avg_acc, avg_f1 = train(model, i)
        train_loss_seg.append(avg_loss_seg)
        train_loss_vaf.append(avg_loss_vaf)
        train_loss_haf.append(avg_loss_haf)
        train_loss.append(avg_loss)
        train_acc.append(avg_acc)
        train_f1.append(avg_f1)
        # plot training loss
        ax1.plot(train_loss_seg, 'r', label='Training segmentation loss')
        ax1.plot(train_loss_vaf, 'g', label='Training VAF loss')
        ax1.plot(train_loss_haf, 'b', label='Training HAF loss')
        ax1.plot(train_loss, 'k', label='Training total loss')
        fig1.savefig(os.path.join(args.output_dir, "train_loss.jpg"))

        # validation epoch
        avg_loss_seg, avg_loss_vaf, avg_loss_haf, avg_loss, avg_acc, avg_f1, bench_acc = val(model, i)
        val_loss_seg.append(avg_loss_seg)
        val_loss_vaf.append(avg_loss_vaf)
        val_loss_haf.append(avg_loss_haf)
        val_loss.append(avg_loss)
        val_acc.append(bench_acc)
        val_f1.append(avg_f1)
        # plot validation loss
        ax2.plot(val_loss_seg, 'r', label='Validation segmentation loss')
        ax2.plot(val_loss_vaf, 'g', label='Validation VAF loss')
        ax2.plot(val_loss_haf, 'b', label='Validation HAF loss')
        ax2.plot(val_loss, 'k', label='Validation total loss')
        fig2.savefig(os.path.join(args.output_dir, "val_loss.jpg"))

        # plot the train and val metrics
        ax3.plot(train_acc, 'r', label='Train accuracy')
        ax3.plot(val_acc, 'g', label='Validation accuracy')
        ax3.plot(train_f1, 'b', label='Train F1 score')
        ax3.plot(val_f1, 'k', label='Validation F1 score')
        fig3.savefig(os.path.join(args.output_dir, 'trainval_acc_f1.jpg'))

    plt.close('all')
    f_log.close()
