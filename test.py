import numpy as np
import torch
import json
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import dataset
import cv2
import sys
import losses
import utils
import visualize

from models.dla.pose_dla_dcn import get_pose_net


def Test(batchSize, testLoader, criterion):
    model = torch.load('parallel_model_final')
    model.eval()
    ts = time.time()
    rolling_loss = 0
    rolling_acc = 0
    rolling_FP = 0
    rolling_FN = 0
    sigmoid = nn.Sigmoid()
    test_losses = []
    test_accuracies = []
    test_FP = []
    test_FN = []
    for iter, sample in enumerate(testLoader):
        if use_gpu:
            inputs = sample['img'].to(device)# Move your inputs onto the gpu
            labels = sample['binary_mask'].to(device,dtype=torch.int)# Move your labels onto the gpu
            segLabel = sample['segLabel'].to(device,dtype=torch.int)
            
        else:
            inputs, labels, segLabel = (sample['img'], sample['binary_mask'], sample['segLabel'])# Unpack variables into inputs and labels
            
        detector_ops = model(inputs)[-1]
        outputs = detector_ops['hm']
        embed = detector_ops['emb']
        #print(outputs.shape)
        del inputs
        torch.cuda.empty_cache()
        loss = criterion(outputs, labels.type_as(outputs))
        outputs = sigmoid(outputs)
        output_cpu = outputs.to(cpu_device).detach().numpy()
        labels_cpu = labels.to(cpu_device).detach().numpy()
        segLabel_cpu = segLabel.to(cpu_device).detach().numpy()
        del outputs
        Acc, false_neg, false_pos = Accuracy(output_cpu, labels_cpu)
        mask = Comparison(output_cpu, segLabel_cpu)
        comp_matrix = torch.from_numpy(mask)
        comp_matrix = comp_matrix.to(device)
        mean = MeanValue(embed, comp_matrix)
        var_loss = VarLoss(embed, comp_matrix, mean)
        dist_loss = Distloss(mean)
        rolling_acc += Acc
        rolling_FP += false_pos
        rolling_FN += false_neg
        loss += var_loss + dist_loss
        del labels,embed, comp_matrix, mean, var_loss, dist_loss

        if iter% 10 == 0:
            print("iter{}, loss: {}, acc: {}, FP: {}, FN: {}".format(iter, loss.item(), Acc, false_pos, false_neg))
        
        rolling_loss += loss.item()
        del loss
        torch.cuda.empty_cache()

    print("time elapsed {}".format(time.time() - ts))
    Normalizing_Factor = len(testLoader) * batch_size
    rolling_acc /= Normalizing_Factor
    rolling_loss /= Normalizing_Factor
    rolling_FP /= Normalizing_Factor
    rolling_FN /= Normalizing_Factor
    return rolling_loss, rolling_acc, rolling_FN, rolling_FP

if __name__ == "__main__":
    weights = torch.tensor([9.6])
    #model = torch.load('parallel_model_earlyStop=loss')
    #optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = .005)
    use_gpu = torch.cuda.is_available()
    cpu_device = torch.device("cpu")
    #checking if GPU is available for use
    if use_gpu:
        device = torch.device("cuda:0")
        model = model.to(device)
        weights = weights.to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    del weights
    _, _, testLoader = Preprocessing()
    batch_size = 3
    torch.cuda.empty_cache()
    test_loss, test_acc, test_FN, test_FP = Test(batch_size, testLoader, criterion)
    print("test_loss: ", test_loss)
    print("test_acc: ", test_acc * 3)
    print("test_FN: ", test_FN * 3)
    print("test_FP", test_FP * 3)
