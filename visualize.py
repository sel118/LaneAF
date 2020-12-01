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
import dataset
import glob
import matplotlib.image as mpimg
import utils


Epoch = [i for i in range(1,31)]

def Visualize():
    #grabs image from test dataset
    _, _, testLoader = Preprocessing()
    it = iter(testLoader)
    first = next(it)
    heatmap = first['segLabel']
    img = first['img_name']
    image = []
    for i in range(len(img)):
        for img_path in glob.glob(img[i]):
            image.append(mpimg.imread(img_path))

    gray_img = cv2.cvtColor(image[0], cv2.COLOR_BGR2GRAY)
    plt.imshow(image[0])
    print(image[0].shape)
    plt.axis('off')
    plt.imshow(heatmap.numpy()[0, 0, :, :])
    print(np.unique(heatmap.numpy()))
    print(first['img_name'][0])
    
    #Load model to visualize data
    model = torch.load('parallel_model_final')
    testImage = first['img'].to(device)
    detector_ops = model(testImage)[-1]
    outputs = detector_ops['hm']
    sigmoid = nn.Sigmoid()
    outputs = sigmoid(outputs)
    outputs_cpu = outputs.to(cpu_device).detach().numpy()
    del outputs
    outputs_cpu[outputs_cpu >= .8] = 1
    outputs_cpu[outputs_cpu < .8] = 0
    #print(np.unique(outputs_cpu))
    plt.axis('off')
    plt.imshow(outputs_cpu[0, 0, :, :])
    
    #Getting outputs and moving to CPU
    embed = detector_ops['emb']
    embed_cpu = embed.to(cpu_device).detach().numpy()
    
    #print(np.unique(outputs_cpu))
    new_output = Cluster(embed,outputs_cpu)
    new_output = new_output.to(cpu_device).detach().numpy()
    cluster_img = new_output[0, 0, :, :]
    plt.imshow(cluster_img, interpolation='none', vmin=0, vmax=4)
    #print(np.unique(cluster_img))
    
    masked = np.ma.masked_where(new_output[0, 0, :, :] <= 0.4, new_output[0, 0, :, :])
    plt.imshow(new_image, 'gray')
    plt.imshow(masked)
    plt.axis('off')
    
    #To display binary segmentation of lane pixels (left as legacy code)
    '''new_image = cv2.resize(image[0], (320, 192))
    masked = np.ma.masked_where(outputs_cpu[0, 0, :, :] <= 0.4, outputs_cpu[0, 0, :, :])
    plt.imshow(new_image, 'gray')
    plt.imshow(masked)
    plt.axis('off')'''
    
    
def Plotlosses():
    train_losses = torch.load("parallel_model_final_decay=.003_train_loss")
    val_losses = torch.load("parallel_model_final_decay=.003_val_loss")
    plt.title("Losses vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Losses")
    plt.grid(which='major', axis='both')
    line1 = plt.plot(Epoch, train_losses, label = 'train loss')
    line2 = plt.plot(Epoch, val_losses, label = 'val loss')
    plt.legend(loc = 'upper right')
    plt.show()
    

def Plotaccuracy():
    train_acc = torch.load("parallel_model_final_decay=.003_train_acc")
    val_acc = torch.load("parallel_model_final_decay=.003_val_acc")
    train_acc = [element * 3 for element in train_acc]
    val_acc = [element * 3 for element in val_acc]
    plt.title("Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(which='major', axis='both')
    line1 = plt.plot(Epoch, train_acc, label = 'train accuracy')
    line2 = plt.plot(Epoch, val_acc, label = 'val accuracy')
    plt.legend(loc = 'upper right')
    plt.show()
    

def PlotFP():
    train_FP = torch.load("parallel_model_final_decay=.003_train_FP")
    val_FP = torch.load("parallel_model_final_decay=.003_val_FP")
    train_FP = [element * 3 for element in train_FP]
    val_FP = [element * 3 for element in val_FP]
    plt.title("False Positive vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("False Positive")
    plt.grid(which='major', axis='both')
    line1 = plt.plot(Epoch, train_FP, label = 'train False Positive')
    line2 = plt.plot(Epoch, val_FP, label = 'val False Positive')
    plt.legend(loc = 'upper right')
    plt.show()
