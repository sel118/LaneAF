import argparse
import json
import os
from datetime import datetime
from statistics import mean

import matplotlib
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.culane import CULane
from models.dla.pose_dla_dcn import get_pose_net
from models.loss import FocalLoss, IoULoss, RegL1Loss

parser = argparse.ArgumentParser('Options for training LaneAF models in PyTorch...')
parser.add_argument('--dataset-dir', type=str, default=None, help='path to dataset')
parser.add_argument('--output-dir', type=str, default=None, help='output directory for model and logs')
parser.add_argument('--snapshot', type=str, default=None, help='path to pre-trained model snapshot')
parser.add_argument('--batch-size', type=int, default=2, metavar='N', help='batch size for training')
parser.add_argument('--epochs', type=int, default=60, metavar='N', help='number of epochs to train for')
parser.add_argument('--learning-rate', type=float, default=1e-4, metavar='LR', help='learning rate')
parser.add_argument('--weight-decay', type=float, default=1e-3, metavar='WD', help='weight decay')
parser.add_argument('--loss-type', type=str, default='wbce', help='Type of classification loss to use (focal/bce/wbce)')
parser.add_argument('--log-schedule', type=int, default=10, metavar='N',
                    help='number of iterations to print/save log after')
parser.add_argument('--seed', type=int, default=1, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--random-transforms', action='store_true', default=False,
                    help='apply random transforms to input during training')
parser.add_argument('-j', '--workers', default=10, type=int, metavar='N',
                    help='number of data loading workers (default: 10)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:8848', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--backend', default='nccl', type=str,
                    help='distributed backend')

args = parser.parse_args()

best_f1 = 0.0

def dist_print(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)


# training function
def train(model, train_loader, criterions, optimizer, scheduler, f_log, epoch, gpu):
    epoch_loss_seg, epoch_loss_vaf, epoch_loss_haf, epoch_loss, epoch_acc, epoch_f1 = list(), list(), list(), list(), list(), list()
    model.train()
    criterion_1, criterion_2, criterion_reg = criterions
    metric_skips = 10

    for b_idx, sample in enumerate(train_loader):
        input_img, input_seg, input_mask, input_af = sample
        input_img = input_img.cuda(gpu, non_blocking=True)
        input_mask = input_mask.cuda(gpu, non_blocking=True)
        input_af = input_af.cuda(gpu, non_blocking=True)

        # zero gradients before forward pass
        optimizer.zero_grad()

        # do the forward pass
        outputs = model(input_img)[-1]

        # calculate losses and metrics
        _mask = (input_mask != train_loader.dataset.ignore_label).float()
        loss_seg = criterion_1(outputs['hm'] * _mask, input_mask * _mask) + criterion_2(torch.sigmoid(outputs['hm']),
                                                                                        input_mask)
        loss_vaf = 0.5 * criterion_reg(outputs['vaf'], input_af[:, :2, :, :], input_mask)
        loss_haf = 0.5 * criterion_reg(outputs['haf'], input_af[:, 2:3, :, :], input_mask)

        epoch_loss_seg.append(loss_seg.item())
        epoch_loss_vaf.append(loss_vaf.item())
        epoch_loss_haf.append(loss_haf.item())
        loss = loss_seg + loss_vaf + loss_haf
        epoch_loss.append(loss.item())
        # Make training faster
        if b_idx % metric_skips == 0:
            pred = torch.sigmoid(outputs['hm']).detach().cpu().numpy().ravel()
            target = input_mask.detach().cpu().numpy().ravel()
            pred[target == train_loader.dataset.ignore_label] = 0
            target[target == train_loader.dataset.ignore_label] = 0
            train_acc = accuracy_score((pred > 0.5).astype(np.int64), (target > 0.5).astype(np.int64))
            train_f1 = f1_score((target > 0.5).astype(np.int64), (pred > 0.5).astype(np.int64), zero_division=1)
            epoch_acc.append(train_acc)
            epoch_f1.append(train_f1)

        loss.backward()
        optimizer.step()
        if b_idx % args.log_schedule == 0 and f_log is not None:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tF1-score: {:.4f}'.format(
                epoch, (b_idx + 1) * args.batch_size, len(train_loader.dataset),
                       100. * (b_idx + 1) * args.batch_size / len(train_loader.dataset), loss.item(), train_f1))
            f_log.write('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tF1-score: {:.4f}\n'.format(
                epoch, (b_idx + 1) * args.batch_size, len(train_loader.dataset),
                       100. * (b_idx + 1) * args.batch_size / len(train_loader.dataset), loss.item(), train_f1))

    scheduler.step()
    # now that the epoch is completed calculate statistics and store logs
    avg_loss_seg = mean(epoch_loss_seg)
    avg_loss_vaf = mean(epoch_loss_vaf)
    avg_loss_haf = mean(epoch_loss_haf)
    avg_loss = mean(epoch_loss)
    avg_acc = mean(epoch_acc)
    avg_f1 = mean(epoch_f1)
    if dist.get_rank() == 0:
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

    return model, avg_loss_seg, avg_loss_vaf, avg_loss_haf, avg_loss, avg_acc, avg_f1


# validation function
def val(net, val_loader, criterions, f_log, epoch, gpu):
    global best_f1
    epoch_loss_seg, epoch_loss_vaf, epoch_loss_haf, epoch_loss, epoch_acc, epoch_f1 = list(), list(), list(), list(), list(), list()
    net.eval()

    criterion_1, criterion_2, criterion_reg = criterions
    for b_idx, sample in enumerate(val_loader):
        input_img, input_seg, input_mask, input_af = sample
        input_img = input_img.cuda(gpu, non_blocking=True)
        input_mask = input_mask.cuda(gpu, non_blocking=True)
        input_af = input_af.cuda(gpu, non_blocking=True)

        # do the forward pass
        outputs = net(input_img)[-1]

        # calculate losses and metrics
        _mask = (input_mask != val_loader.dataset.ignore_label).float()
        loss_seg = criterion_1(outputs['hm'], input_mask) + criterion_2(torch.sigmoid(outputs['hm']), input_mask)
        loss_vaf = 0.5 * criterion_reg(outputs['vaf'], input_af[:, :2, :, :], input_mask)
        loss_haf = 0.5 * criterion_reg(outputs['haf'], input_af[:, 2:3, :, :], input_mask)
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

    # now that the epoch is completed calculate statistics and store logs
    avg_loss_seg = torch.tensor(mean(epoch_loss_seg)).cuda(gpu)
    avg_loss_vaf = torch.tensor(mean(epoch_loss_vaf)).cuda(gpu)
    avg_loss_haf = torch.tensor(mean(epoch_loss_haf)).cuda(gpu)
    avg_loss = torch.tensor(mean(epoch_loss)).cuda(gpu)
    avg_acc = torch.tensor(mean(epoch_acc)).cuda(gpu)
    avg_f1 = torch.tensor(mean(epoch_f1)).cuda(gpu)

    # Sync whole dataset, no need this in training.
    dist.all_reduce(avg_f1)
    dist.all_reduce(avg_acc)
    dist.all_reduce(avg_loss)
    dist.all_reduce(avg_loss_haf)
    dist.all_reduce(avg_loss_vaf)
    dist.all_reduce(avg_loss_seg)
    avg_f1 /= dist.get_world_size()
    avg_acc /= dist.get_world_size()
    avg_loss /= dist.get_world_size()
    avg_loss_haf /= dist.get_world_size()
    avg_loss_vaf /= dist.get_world_size()
    avg_loss_seg /= dist.get_world_size()

    if dist.get_rank() == 0:
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
        print("--------------------------------------------------------------------\n")
        f_log.write("--------------------------------------------------------------------\n\n")

    # now save the model if it has a better F1 score than the best model seen so forward

    if dist.get_rank() == 0 and avg_f1 > best_f1:
        # save the model
        if isinstance(net, torch.nn.parallel.DistributedDataParallel):
            net = net.module
        torch.save(net.state_dict(), os.path.join(args.output_dir, 'net_' + '%.4d' % (epoch,) + '.pth'))
        best_f1 = avg_f1

    return avg_loss_seg, avg_loss_vaf, avg_loss_haf, avg_loss, avg_acc, avg_f1


def worker(gpu, gpu_num, args):
    args.gpu = gpu
    args.rank = gpu

    print("{} -> {}\t{}/{}".format(args.backend, args.dist_url, args.rank, gpu_num))
    dist.init_process_group(backend=args.backend, init_method=args.dist_url, world_size=gpu_num, rank=args.rank)
    dist_print(datetime.now().strftime('[%Y/%m/%d %H:%M:%S]') + ' start training...')

    f_log = open(os.path.join(args.output_dir, "logs.txt"), "w") if args.rank == 0 else None

    heads = {'hm': 1, 'vaf': 2, 'haf': 1}
    model = get_pose_net(num_layers=34, heads=heads, head_conv=256, down_ratio=4)
    torch.cuda.set_device(args.gpu)

    if args.snapshot is not None:
        model.load_state_dict(torch.load(args.snapshot), strict=True)

    if args.cuda:
        model.cuda(args.gpu)
    # TODO disable whiling loading snapshot
    model.hm[2].bias.data.uniform_(-4.595, -4.595)  # bias towards negative class

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    dist_print(model)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.2)

    # BCE(Focal) loss applied to each pixel individually

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
    if args.rank == 0:
        fig1, ax1 = plt.subplots()
        plt.grid(True)
        ax1.plot([], 'r', label='Training segmentation loss')
        ax1.plot([], 'g', label='Training VAF loss')
        ax1.plot([], 'b', label='Training HAF loss')
        ax1.plot([], 'k', label='Training total loss')
        ax1.legend()

        fig2, ax2 = plt.subplots()
        plt.grid(True)
        ax2.plot([], 'r', label='Validation segmentation loss')
        ax2.plot([], 'g', label='Validation VAF loss')
        ax2.plot([], 'b', label='Validation HAF loss')
        ax2.plot([], 'k', label='Validation total loss')
        ax2.legend()

        fig3, ax3 = plt.subplots()
        plt.grid(True)
        ax3.plot([], 'r', label='Training accuracy')
        ax3.plot([], 'g', label='Validation accuracy')
        ax3.plot([], 'b', label='Training F1 score')
        ax3.plot([], 'k', label='Validation F1 score')
        ax3.legend()

    train_loss_seg, train_loss_vaf, train_loss_haf, train_loss = list(), list(), list(), list()
    val_loss_seg, val_loss_vaf, val_loss_haf, val_loss = list(), list(), list(), list()
    train_acc, val_acc, train_f1, val_f1 = list(), list(), list(), list()

    train_dataset = CULane(args.dataset_dir, 'train', args.random_transforms)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    kwargs = {'batch_size': args.batch_size // gpu_num, 'sampler': train_sampler, 'num_workers': args.workers}
    train_loader = DataLoader(train_dataset, **kwargs)

    val_dataset = CULane(args.dataset_dir, 'val', False)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    kwargs = {'batch_size': args.batch_size // gpu_num, 'num_workers': args.workers, 'sampler': val_sampler}
    val_loader = DataLoader(val_dataset, **kwargs)

    # trainval loop
    for i in range(1, args.epochs + 1):
        train_sampler.set_epoch(i - 1)
        val_sampler.set_epoch(i - 1)  # TODO xxx

        # training epoch
        model, avg_loss_seg, avg_loss_vaf, avg_loss_haf, avg_loss, avg_acc, avg_f1 = train(model, train_loader,
                                                                                           [criterion_1, criterion_2,
                                                                                            criterion_reg], optimizer,
                                                                                           scheduler, f_log, i, gpu)
        train_loss_seg.append(avg_loss_seg)
        train_loss_vaf.append(avg_loss_vaf)
        train_loss_haf.append(avg_loss_haf)
        train_loss.append(avg_loss)
        train_acc.append(avg_acc)
        train_f1.append(avg_f1)

        # validation epoch
        avg_loss_seg, avg_loss_vaf, avg_loss_haf, avg_loss, avg_acc, avg_f1 = val(model, val_loader,
                                                                                  [criterion_1, criterion_2,
                                                                                   criterion_reg], f_log, i, gpu)
        val_loss_seg.append(avg_loss_seg)
        val_loss_vaf.append(avg_loss_vaf)
        val_loss_haf.append(avg_loss_haf)
        val_loss.append(avg_loss)
        val_acc.append(avg_acc)
        val_f1.append(avg_f1)

        # plot
        if args.rank == 0:
            # plot training loss
            ax1.plot(train_loss_seg, 'r', label='Training segmentation loss')
            ax1.plot(train_loss_vaf, 'g', label='Training VAF loss')
            ax1.plot(train_loss_haf, 'b', label='Training HAF loss')
            ax1.plot(train_loss, 'k', label='Training total loss')
            fig1.savefig(os.path.join(args.output_dir, "train_loss.jpg"))

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

    plt.close('all') if args.rank == 0 else None
    f_log.close() if args.rank == 0 else None


if __name__ == "__main__":
    # check args
    if args.dataset_dir is None:
        assert False, 'Path to dataset not provided!'

    # setup args
    args.cuda = torch.cuda.is_available()
    if args.output_dir is None:
        args.output_dir = datetime.now().strftime("FastTraining-%Y%m%d-%H%M%S")
        args.output_dir = os.path.join('.', 'experiments', 'culane', args.output_dir)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        assert False, 'Output directory already exists!'

    # store config in output directory
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f)

    # set random seed
    if args.seed:
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    assert args.world_size == 1, "Only world size == 1 multi-process training now"
    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

    # # check args
    # if args.dataset_dir is None:
    #     assert False, 'Path to dataset not provided!'
    #
    # # setup args
    # args.cuda = not args.no_cuda and torch.cuda.is_available()
    # if args.output_dir is None:
    #     args.output_dir = datetime.now().strftime("%Y-%m-%d-%H:%M")
    #     args.output_dir = os.path.join('.', 'experiments', 'culane', args.output_dir)
    #
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)
    # else:
    #     assert False, 'Output directory already exists!'
    #
    # # store config in output directory
    # with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
    #     json.dump(vars(args), f)
    #
    # # set random seed
    # torch.manual_seed(args.seed)
    # if args.cuda:
    #     torch.cuda.manual_seed(args.seed)
    #
    # kwargs = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 6}
    # train_loader = DataLoader(CULane(args.dataset_dir, 'train', args.random_transforms), **kwargs)
    # kwargs = {'batch_size': 1, 'shuffle': False, 'num_workers': 3}
    # val_loader = DataLoader(CULane(args.dataset_dir, 'val', False), **kwargs)
    #
    # # global var to store best validation F1 score across all epochs
    # best_f1 = 0.0
    # # create file handles
    # f_log = open(os.path.join(args.output_dir, "logs.txt"), "w")
