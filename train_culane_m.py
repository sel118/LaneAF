import argparse
import json
import os
from datetime import datetime
from statistics import mean

import matplotlib
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim.lr_scheduler import CosineAnnealingLR

from models.erf.encoder import ERFNet as Encoder
from models.raw_resnet import DLAFPNAF

matplotlib.use('Agg')
from sklearn.metrics import accuracy_score, f1_score

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.culane import CULane

from models.loss import FocalLoss, IoULoss, RegL1Loss

import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

parser = argparse.ArgumentParser('Options for training LaneAF models in PyTorch...')
parser.add_argument('--dataset-dir', type=str, default=None, help='path to dataset')
parser.add_argument('--output-dir', type=str, default=None, help='output directory for model and logs')
parser.add_argument('--snapshot', type=str, default=None, help='path to pre-trained model snapshot')
parser.add_argument('--batch-size', type=int, default=32 * 4, metavar='N', help='batch size for training')
parser.add_argument('--epochs', type=int, default=90, metavar='N', help='number of epochs to train for')
parser.add_argument('--learning-rate', type=float, default=0.02, metavar='LR', help='learning rate')
parser.add_argument('--weight-decay', type=float, default=1e-3, metavar='WD', help='weight decay')
parser.add_argument('--loss-type', type=str, default='wbce', help='Type of classification loss to use (focal/bce/wbce)')
parser.add_argument('--log-schedule', type=int, default=10, metavar='N',
                    help='number of iterations to print/save log after')
parser.add_argument('--seed', type=int, default=None, help='set seed to some constant value to reproduce experiments')
parser.add_argument('--no-cuda', action='store_true', default=False, help='do not use cuda for training')
parser.add_argument('--random-transforms', action='store_true', default=True,
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
parser.add_argument('--pretrained', type=str, default=None, help='path to dataset')
args = parser.parse_args()


def save_model(net, optimizer, epoch, save_path):
    if isinstance(net, torch.nn.parallel.DistributedDataParallel):
        net = net.module
    if dist.get_rank() == 0:
        model_state_dict = net.state_dict()
        state = {'model': model_state_dict, 'optimizer': optimizer.state_dict()}
        # state = {'model': model_state_dict}
        assert os.path.exists(save_path)
        model_path = os.path.join(save_path, 'ep%03d.pth' % epoch)
        torch.save(state, model_path)


# training function
def train(net, train_loader, criterions, optimizer, scheduler, f_log, epoch, gpu):
    epoch_loss_seg, epoch_loss_vaf, epoch_loss_haf, epoch_loss, epoch_acc, epoch_f1 = list(), list(), list(), list(), list(), list()
    net.train()
    criterion_1, criterion_2, criterion_reg = criterions
    metric_sampler_inter = 20
    for b_idx, sample in enumerate(train_loader):
        input_img, _, input_mask, input_af = sample
        input_img = input_img.cuda(gpu, non_blocking=True)
        input_mask = input_mask.cuda(gpu, non_blocking=True)
        input_af = input_af.cuda(gpu, non_blocking=True)
        # print(input_mask.shape, input_img.shape, input_af.shape)
        # zero gradients before forward pass

        optimizer.zero_grad()

        # do the forward pass
        outputs = net(input_img)[-1]

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
        if b_idx % metric_sampler_inter == 0:
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
        if b_idx % args.log_schedule == 0 and dist.get_rank() == 0:
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tF1-score: {:.4f} \tSegloss: {:.4f}'.format(
                epoch, (b_idx + 1) * args.batch_size, len(train_loader.dataset),
                       100. * (b_idx + 1) * args.batch_size / len(train_loader.dataset), loss.item(), train_f1,
                loss_seg.item()))
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

    return net, avg_loss_seg, avg_loss_vaf, avg_loss_haf, avg_loss, avg_acc, avg_f1


def dist_print(*args, **kwargs):
    if dist.get_rank() == 0:
        print(*args, **kwargs)


def worker(gpu, gpu_num, args):
    args.gpu = gpu
    args.rank = gpu
    # print("Use GPU {} for training...".format(gpu))
    print("{} -> {}\t{}/{}".format(args.backend, args.dist_url, args.rank, gpu_num))
    dist.init_process_group(backend=args.backend, init_method=args.dist_url, world_size=gpu_num, rank=args.rank)
    dist_print(datetime.now().strftime('[%Y/%m/%d %H:%M:%S]') + ' start training...')

    f_log = open(os.path.join(args.output_dir, "logs.txt"), "w") if args.rank == 0 else None
    # Model
    if args.pretrained is not None:
        print("Loading pretrained weights from file {} ...".format(args.pretrained))
        encoder = Encoder(1000)
        sd = torch.load(args.pretrained, map_location="cpu")
        sd = sd['state_dict']
        new_sd = {}
        for k, v in sd.items():
            new_sd[k.replace("module.", '')] = v
        encoder.load_state_dict(new_sd)
    else:
        encoder = None

    torch.set_num_threads(1)
    # model = ResNetAF({"hm": 1, "haf": 1, "vaf": 2}, pretrained=True)
    model = DLAFPNAF({"hm": 1, "haf": 1, "vaf": 2})
    torch.cuda.set_device(args.gpu)

    if args.snapshot is not None:
        # model = D4UNet()
        sd = torch.load(args.snapshot)
        sd = sd['model']
        new_sd = {}
        for k, v in sd.items():
            new_sd[k.replace("module.", '')] = v
        model.load_state_dict(new_sd, strict=True)

    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    print("Model ready...")
    dist_print(model)

    # Loss && Optimizer
    # BCE(Focal) loss applied to each pixel individually
    if args.loss_type == 'focal':
        criterion_1 = FocalLoss(gamma=2.0, alpha=0.25, size_average=True)
    elif args.loss_type == 'bce':
        ## BCE weight
        criterion_1 = torch.nn.BCEWithLogitsLoss()
    elif args.loss_type == 'wbce':
        ## BCE weight
        criterion_1 = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([9.6]).cuda())
    else:
        print("No such loss: {}".format(args.loss_type))
        exit()
    criterion_1.cuda(args.gpu)
    criterion_2 = IoULoss().cuda(args.gpu)
    criterion_reg = RegL1Loss().cuda(args.gpu)
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # Loss && Optimizer ready

    scheduler = CosineAnnealingLR(optimizer, args.epochs)
    import torchvision.transforms as transforms
    augs = transforms.Compose([transforms.RandomErasing(), transforms.RandomGrayscale(),
                               transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.1, saturation=0.2)])
    train_dataset = CULane(args.dataset_dir, 'train', args.random_transforms, img_transforms=augs)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    kwargs = {'batch_size': args.batch_size // gpu_num, 'num_workers': args.workers,
              'sampler': train_sampler}
    train_loader = DataLoader(train_dataset, **kwargs, pin_memory=True)
    # TODO resume && finetune
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        train(model, train_loader, [criterion_1, criterion_2, criterion_reg], optimizer, scheduler, f_log, epoch,
              args.gpu)
        # val(model, val_loader, f_log, args.gpu)
        save_model(model, optimizer, epoch, args.output_dir)


def mp_train():
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

    assert args.world_size == 1, "Only world size == 1 mp training now"
    ngpus_per_node = torch.cuda.device_count()
    mp.spawn(worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


if __name__ == "__main__":
    # MP training for hm
    mp_train()
