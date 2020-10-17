#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Remember to take out unnecessary import statements later
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
import cv2
import sys


# In[ ]:


def VarLoss(emb_output, Label, means):
    num_classes = len(means)
    if num_classes == 0:
        return 0
    
    delta_v = .5
    loss = 0
    for i in range(num_classes):
        indeces = torch.where(Label == (i + 1))
        pixels = emb_output[indeces[0], :, indeces[2], indeces[3]]
        #print("num of rows: ", rows)
        mean = means[i]
        normalized_pixels = mean - pixels
        #print("before shape: ", normalized_pixels.shape)
        norm_pixels = torch.norm(normalized_pixels, p = 2, dim = 1)
        #print("after shape: ", norm_pixels.shape)
        norm_delta_pixels = norm_pixels - delta_v
        summed_pixels = F.relu(norm_delta_pixels)
        summed_pixels = summed_pixels ** 2
        summed_pixels = torch.mean(summed_pixels)
        #print("summed value ", summed_pixels)
        #print(torch.sum(summed_pixels))
        loss += summed_pixels
               
    return loss / num_classes
    
    
def Distloss(mean):
    num_classes = len(mean)
    if  num_classes <= 1:
        return 0
    
    loss = 0
    delta_d = 3
    for i in range(len(mean)):
        for j in range(len(mean)):
            if i == j:
                continue
            mean1 = mean[i]
            mean2 = mean[j]
            mean_difference = mean1 - mean2
            norm_mean_diff = torch.norm(mean_difference, p = 2)
            delta_minus_mean = delta_d - norm_mean_diff
            delta_minus_mean_relu = F.relu(delta_minus_mean)
            loss += delta_minus_mean_relu ** 2            
    
    return loss / num_classes / (num_classes - 1)

