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
import dataset
import cv2
import sys
import losses
import utils
#There were setup steps that Akshay showed us for this pose_dla_dcn in linux command line
sys.path.append('./DCNv2/build/lib.linux-x86_64-3.7')
sys.path.append('./DCNv2/')
from DCNv2 import dcn_v2
from dcn_v2 import DCN
from pose_dla_dcn import get_pose_net


# In[ ]:


def train(batch_size, trainLoader, valLoader, model, check_num = 5):
    lr = 1e-4 
    num_epochs = 30
    weights = torch.tensor([9.6])
    counter = 0 
    #initializing containers to store accuracy and loss every epoch
    train_losses = []
    accuracies = []
    FP = []
    FN = []
    val_losses = []
    val_accuracies = []
    val_FP = []
    val_FN = []
    sigmoid = nn.Sigmoid()
    use_gpu = torch.cuda.is_available()
    cpu_device = torch.device("cpu")
    #checking if GPU is available for use
    if use_gpu:
        device = torch.device("cuda:0")
        model = model.to(device)
        weights = weights.to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = .003)
    del weights
    
    for epoch in range(num_epochs):
        if epoch == 3:
            optimizer = optim.Adam(model.parameters(), lr = 5e-5, weight_decay = .003) 
        elif epoch == 7:
            optimizer = optim.Adam(model.parameters(), lr = 1e-6, weight_decay = .003) 
        elif epoch == 11:
            optimizer = optim.Adam(model.parameters(), lr = 5e-7, weight_decay = .003)
        elif epoch == 16:
            optimizer = optim.Adam(model.parameters(), lr = 1e-7, weight_decay = .003)
            
        ts = time.time()
        #variables to sum loss and accuracy of each batch
        rolling_loss = 0
        rolling_acc = 0
        rolling_FP = 0
        rolling_FN = 0
        for iter, sample in enumerate(trainLoader):
            optimizer.zero_grad()
            if use_gpu:
                inputs = sample['img'].to(device)# Move your inputs onto the gpu
                labels = sample['binary_mask'].to(device,dtype=torch.int)# Move your labels onto the gpu
                segLabel = sample['segLabel'].to(device,dtype=torch.int)
                
            else:
                inputs, labels, segLabel = (sample['img'], sample['binary_mask'], sample['segLabel']) # Unpack variables into inputs and labels

            #print(torch.unique(segLabel))
            detector_ops = model(inputs)[-1]
            outputs = detector_ops['hm']
            emb_outputs = detector_ops['emb']
            #print(outputs.shape)
            del inputs
            torch.cuda.empty_cache()
            loss = criterion(outputs, labels.type_as(outputs))
            outputs = sigmoid(outputs)
            output_cpu = outputs.to(cpu_device).detach().numpy()
            labels_cpu = labels.to(cpu_device).detach().numpy()
            segLabel_cpu = segLabel.to(cpu_device).detach().numpy()
            del outputs
            Acc, false_neg, false_pos = utils.Accuracy(output_cpu, labels_cpu)
            comp_matrix = torch.from_numpy(utils.Comparison(output_cpu, segLabel_cpu))
            if use_gpu:
                comp_matrix = comp_matrix.to(device)
                
            mean = utils.MeanValue(emb_outputs, comp_matrix)
            var_loss = losses.VarLoss(emb_outputs, comp_matrix, mean)
            dist_loss = losses.Distloss(mean)
            rolling_acc += Acc
            loss += var_loss + dist_loss
            rolling_FP += false_pos
            rolling_FN += false_neg
            del labels,emb_outputs, comp_matrix, mean, var_loss, dist_loss
            torch.cuda.empty_cache()
            loss.backward()
            optimizer.step()

            if iter % 10 == 0:
                print("epoch{}, iter{}, loss: {}, acc: {}, FP: {}, FN: {}".format(epoch, iter, loss.item(), Acc, false_pos, false_neg))

            rolling_loss += loss.item()
            del loss
            torch.cuda.empty_cache()

        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        Normalizing_Factor = len(trainLoader) * batch_size
        train_losses.append(rolling_loss / Normalizing_Factor)
        accuracies.append(rolling_acc / Normalizing_Factor)
        FP.append(rolling_FP / Normalizing_Factor)
        FN.append(rolling_FN / Normalizing_Factor)
        loss_val, acc_val, Fn_val, Fp_val = Val(epoch, valLoader, batch_size, use_gpu)
        val_losses.append(loss_val)
        val_accuracies.append(acc_val)
        val_FP.append(Fp_val)
        val_FN.append(Fn_val)
        model.train()
        #Checking if current model is better than the previous best model
        if epoch == 0:
            torch.save(model, 'parallel_model_final_decay=.003')
        else:
            if torch.argmin(torch.Tensor(val_losses)) == epoch:
                print("current-model saved as best model")
                torch.save(model, 'parallel_model_final_decay=.003')
                
        #Early Stopping Not implemented (uncomment if needed)
        '''if counter == check_num:
            print("early stop achieved")
            torch.save(losses, "parallel_model_earlyStop=loss_rev2_train_loss")
            torch.save(accuracies, "parallel_model_earlyStop=loss_rev2_train_acc")
            torch.save(val_losses, "parallel_model_earlyStop=loss_rev2_val_loss")
            torch.save(val_accuracies, "parallel_model_earlyStop=loss_rev2_val_acc")
            break'''
        
        if epoch == (num_epochs - 1):
            print("training is finished")
            #torch.save(model, 'parallel_model')
            torch.save(train_losses, "parallel_model_final_decay=.003_train_loss")
            torch.save(accuracies, "parallel_model_final_decay=.003_train_acc")
            torch.save(FP, "parallel_model_final_decay=.003_train_FP")
            torch.save(FN, "parallel_model_final_decay=.003_train_FN")
            torch.save(val_losses, "parallel_model_final_decay=.003_val_loss")
            torch.save(val_accuracies, "parallel_model_final_decay=.003_val_acc")
            torch.save(val_FP, "parallel_model_final_decay=.003_val_FP")
            torch.save(val_FN, "parallel_model_final_decay=.003_val_FN")
            
            
def Val(epoch, ValLoader, batchSize, use_gpu, device):
    model.eval()
    ts = time.time()
    rolling_loss = 0
    rolling_acc = 0
    rolling_FP = 0
    rolling_FN = 0
    sigmoid = nn.Sigmoid()
    for iter, sample in enumerate(ValLoader):
        if use_gpu:
            inputs = sample['img'].to(device)# Move your inputs onto the gpu
            labels = sample['binary_mask'].to(device,dtype=torch.int)# Move your labels onto the gpu
            segLabel = sample['segLabel'].to(device,dtype=torch.int)
            
        else:
            inputs, labels, segLabel = (sample['img'], sample['binary_mask'], sample['segLabel'])# Unpack variables into inputs and labels
            
        detector_ops = model(inputs)[-1]
        outputs = detector_ops['hm']
        emb_outputs = detector_ops['emb']
        #print(outputs.shape)
        del inputs
        torch.cuda.empty_cache()
        loss = criterion(outputs, labels.type_as(outputs))
        outputs = sigmoid(outputs)
        output_cpu = outputs.to(cpu_device).detach().numpy()
        labels_cpu = labels.to(cpu_device).detach().numpy()
        segLabel_cpu = segLabel.to(cpu_device).detach().numpy()
        del outputs
        Acc, false_neg, false_pos = utils.Accuracy(output_cpu, labels_cpu)
        comp_matrix = torch.from_numpy(utils.Comparison(output_cpu, segLabel_cpu))
        if use_gpu:
            comp_matrix = comp_matrix.to(device)
            
        mean = utils.MeanValue(emb_outputs, comp_matrix)
        var_loss = losses.VarLoss(emb_outputs, comp_matrix, mean)
        dist_loss = losses.Distloss(mean)
        rolling_acc += Acc
        rolling_FP += false_pos
        rolling_FN += false_neg
        loss += var_loss + dist_loss
        del labels,emb_outputs, comp_matrix, mean, var_loss, dist_loss

        if iter% 10 == 0:
            print("epoch{}, iter{}, loss: {}, acc: {}, FP: {}, FN: {}".format(epoch, iter, loss.item(), Acc, false_pos, false_neg))
        
        rolling_loss += loss.item()
        del loss
        torch.cuda.empty_cache()

    print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
    Normalizing_Factor = len(ValLoader) * batch_size
    rolling_acc /= Normalizing_Factor
    rolling_loss /= Normalizing_Factor
    rolling_FP /= Normalizing_Factor
    rolling_FN /= Normalizing_Factor
    return rolling_loss, rolling_acc, rolling_FN, rolling_FP
            
if __name__ == "__main__":
    #defining initial parameters and model
    lr = 1e-4 
    num_epochs = 30
    weights = torch.tensor([9.6])
    heads = {'hm': 1, 'emb': 4}
    model = get_pose_net(num_layers=34, heads=heads, head_conv=256, down_ratio=4)
    #model = torch.load('parallel_model_earlyStop=loss')
    #optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = .005)
    
    #We moved this into the train function to test if the .to(device) is out of scope of the train function
    '''use_gpu = torch.cuda.is_available()
    cpu_device = torch.device("cpu")
    #checking if GPU is available for use
    if use_gpu:
        device = torch.device("cuda:0")
        model = model.to(device)
        weights = weights.to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    del weights'''
    trainLoader, valLoader, _ = dataset.Preprocessing()
    batch_size = 3
    train(batch_size, trainLoader, valLoader, model)

