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
from kmeans_pytorch import kmeans


# In[ ]:


def Accuracy(output, label):
    output[output >= .6] = 1
    output[output < .6] = 0
    trueLabelIndeces = np.where(label == 1.0)
    falseLabelIndeces = np.where(label == 0.0)
    trueLabels = label[trueLabelIndeces]
    newTrueOutputs = output[trueLabelIndeces]
    falseLabels = label[falseLabelIndeces]
    newFalseOutputs = output[falseLabelIndeces]
    newTrueOutputs = np.reshape(newTrueOutputs, (-1))
    trueLabels = np.reshape(trueLabels, (-1))
    falseLabels = np.reshape(falseLabels, (-1))
    newFalseOutputs = np.reshape(newFalseOutputs, (-1))
    num_false_positives = sum((newFalseOutputs != falseLabels).astype(np.float))
    num_true_positives = sum((newTrueOutputs == trueLabels).astype(np.float))
    acc = float(num_true_positives / trueLabels.shape[0])
    FN = 1 - acc
    FP = float(num_false_positives / falseLabels.shape[0])
    return acc, FN, FP


def Comparison(output, segLabel):
    output[output >= .6] = 1
    output[output < .6] = 0
    #print(np.unique(segLabel))
    comparison_matrix = np.multiply(output, segLabel)
    #print(np.unique(comparison_matrix))
    return comparison_matrix


def MeanValue(emb_output, Label):
    Means = []
    classes = torch.unique(Label)
    for i in range(1, classes.shape[0]):
        indeces = (Label == i).nonzero() #torch.nonzero(Label == i)#, as_tuple=True) #torch.where(Label == i)
        #n x 4 matrix
        print(type(indeces))
        print(len(indeces))
        print(indeces.shape)
        pixels = emb_output[indeces[0], :, indeces[2], indeces[3]]
        mean_pixels = torch.mean(pixels, 0)
        #print(mean_pixels)
        Means.append(mean_pixels)

    return Means


def Cluster(embed, Label):
    outputs_gpu = torch.from_numpy(Label)
    indices = torch.nonzero(outputs_gpu == 1)#, as_tuple=True) #torch.where(outputs_gpu == 1)
    num_clusters = 4
    shape = Label.shape
    new_outputs = torch.zeros(shape[0], shape[1], shape[2], shape[3])
    cluster_ids_x, cluster_centers = kmeans(embed[indices[0], :, indices[2], indices[3]], 
                                            num_clusters=num_clusters, 
                                            distance='euclidean', 
                                            device=torch.device('cuda:0'))
    
    for i in range(len(cluster_ids_x)):
        laneval = cluster_ids_x[i]
        new_outputs[indices[0][i], :, indices[2][i], indices[3][i]] = laneval + 1

    output = new_outputs
    return output

