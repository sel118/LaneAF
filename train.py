import json
import time
import sys

import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.tusimple import TuSimple
from models.dla.pose_dla_dcn import get_pose_net
from models.loss import VAFLoss, HAFLoss
import utils


def train(batch_size, lr, num_epochs, weights, trainLoader, valLoader, model, check_num = 5):
    #counter = 0 
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
            
            #Add PAF variable to gpu or cpu, depending on if the gpu is available
            if use_gpu:
                inputs = sample['img'].to(device)# Move your inputs onto the gpu
                labels = sample['mask'].to(device, dtype=torch.int)# Move your labels onto the gpu
                segLabel = sample['seg'].to(device, dtype=torch.int)
                vafLabel = sample['vaf'].to(device, dtype=torch.float32)
                hafLabel = sample['haf'].to(device, dtype=torch.float32)
             
            # Unpack variables into inputs and labels
            else:
                inputs, labels, segLabel, vafLabel, hafLabel = (sample['img'], sample['mask'], 
                                                      sample['seg'], sample['vaf'], sample['haf']) 

            #print(torch.unique(segLabel))
            detector_ops = model(inputs)[-1]
            outputs = detector_ops['hm']
            #emb_outputs = detector_ops['emb']
            vaf_outputs = detector_ops['vaf']
            haf_outputs = detector_ops['haf']
            vaf_outputs = vaf_outputs[0,:,:,:]
            vaf_outputs = torch.reshape(vaf_outputs, (320,192,2))
            haf_outputs = haf_outputs[0,:,:,:]
            haf_outputs = torch.reshape(haf_outputs, (320,192))
            del inputs
            torch.cuda.empty_cache()
            loss = criterion(outputs, labels.type_as(outputs))
            outputs = sigmoid(outputs)
            output_cpu = outputs.to(cpu_device).detach().numpy()
            labels_cpu = labels.to(cpu_device).detach().numpy()
            segLabel_cpu = segLabel.to(cpu_device).detach().numpy()
            del outputs
            Acc, false_neg, false_pos = utils.Accuracy(output_cpu, labels_cpu)
            '''comp_matrix = torch.from_numpy(utils.Comparison(output_cpu, segLabel_cpu))
            if use_gpu:
                comp_matrix = comp_matrix.to(device)
                
            mean = utils.MeanValue(emb_outputs, comp_matrix)
            var_loss = VarLoss(emb_outputs, comp_matrix, mean)
            dist_loss = Distloss(mean)'''
            
            #add variable name for input PAFs
            vaf_l2_loss = VAFLoss(vaf_outputs, vafLabel)
            haf_l2_loss = HAFLoss(haf_outputs, hafLabel)
            
            rolling_acc += Acc
            #loss += var_loss + dist_loss
            loss += vaf_l2_loss + haf_l2_loss
            
            rolling_FP += false_pos
            rolling_FN += false_neg
            
            #del labels, emb_outputs, comp_matrix, mean, var_loss, dist_loss
            
            del labels, segLabel, vafLabel, hafLabel, vaf_outputs, haf_outputs, vaf_l2_loss, haf_l2_loss
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
        loss_val, acc_val, Fn_val, Fp_val = val(epoch, valLoader, batch_size, use_gpu, device, criterion, cpu_device)
        val_losses.append(loss_val)
        val_accuracies.append(acc_val)
        val_FP.append(Fp_val)
        val_FN.append(Fn_val)
        model.train()
        #Checking if current model is better than the previous best model
        if epoch == 0:
            torch.save(model, 'PAF_Model_V1_Best')
        else:
            if torch.argmin(torch.Tensor(val_losses)) == epoch:
                print("current-model saved as best model")
                torch.save(model, 'PAF_Model_V1_Best')
                
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
            torch.save(train_losses, "PAF_Model_V1_train_loss")
            torch.save(accuracies, "PAF_Model_V1_train_acc")
            torch.save(FP, "PAF_Model_V1_train_FP")
            torch.save(FN, "PAF_Model_V1_train_FN")
            torch.save(val_losses, "PAF_Model_V1_val_loss")
            torch.save(val_accuracies, "PAF_Model_V1_val_acc")
            torch.save(val_FP, "PAF_Model_V1_val_FP")
            torch.save(val_FN, "PAF_Model_V1_val_FN")
            
            
def val(epoch, ValLoader, batchSize, use_gpu, device, criterion, cpu_device):
    model.eval()
    ts = time.time()
    rolling_loss = 0
    rolling_acc = 0
    rolling_FP = 0
    rolling_FN = 0
    sigmoid = nn.Sigmoid()
    
    for iter, sample in enumerate(ValLoader):
        
        #Add PAF variable to gpu or cpu, depending on if the gpu is available
        if use_gpu:
            inputs = sample['img'].to(device)# Move your inputs onto the gpu
            labels = sample['mask'].to(device,dtype=torch.int)# Move your labels onto the gpu
            segLabel = sample['seg'].to(device,dtype=torch.int)
            vafLabel = sample['vaf'].to(device, dtype=torch.float32)
            hafLabel = sample['haf'].to(device, dtype=torch.float32)
             
            # Unpack variables into inputs and labels
        else:
            inputs, labels, segLabel, vafLabel, hafLabel = (sample['img'], sample['mask'], 
                                                  sample['seg'], sample['vaf'], sample['haf']) 
            
        detector_ops = model(inputs)[-1]
        outputs = detector_ops['hm']
        #emb_outputs = detector_ops['emb']
        vaf_outputs = detector_ops['vaf']
        haf_outputs = detector_ops['haf']
        vaf_outputs = vaf_outputs[0,:,:,:]
        vaf_outputs = torch.reshape(vaf_outputs, (320,192,2))
        haf_outputs = haf_outputs[0,:,:,:]
        haf_outputs = torch.reshape(haf_outputs, (320,192))
        del inputs
        torch.cuda.empty_cache()
        loss = criterion(outputs, labels.type_as(outputs))
        outputs = sigmoid(outputs)
        output_cpu = outputs.to(cpu_device).detach().numpy()
        labels_cpu = labels.to(cpu_device).detach().numpy()
        segLabel_cpu = segLabel.to(cpu_device).detach().numpy()
        del outputs
        Acc, false_neg, false_pos = utils.Accuracy(output_cpu, labels_cpu)
        '''comp_matrix = torch.from_numpy(utils.Comparison(output_cpu, segLabel_cpu))
        if use_gpu:
            comp_matrix = comp_matrix.to(device)

        mean = utils.MeanValue(emb_outputs, comp_matrix)
        var_loss = losses.VarLoss(emb_outputs, comp_matrix, mean)
        dist_loss = losses.Distloss(mean)'''

        #add variable name for input PAFs
        vaf_l2_loss = losses.VAFLoss(vaf_outputs, vafLabel)
        haf_l2_loss = losses.HAFLoss(haf_outputs, hafLabel)

        rolling_acc += Acc
        #loss += var_loss + dist_loss
        loss += vaf_l2_loss + haf_l2_loss

        rolling_FP += false_pos
        rolling_FN += false_neg

        #del labels, emb_outputs, comp_matrix, mean, var_loss, dist_loss

        del labels, segLabel, vafLabel, hafLabel, vaf_outputs, haf_outputs, vaf_l2_loss, haf_l2_loss

        if iter% 10 == 0:
            print("epoch{}, iter{}, loss: {}, acc: {}, FP: {}, FN: {}".format(epoch, iter, loss.item(), Acc, false_pos, false_neg))
        
        rolling_loss += loss.item()
        del loss
        torch.cuda.empty_cache()

    print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
    Normalizing_Factor = len(ValLoader) * batchSize
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
    #heads = {'hm': 1, 'emb': 4}
    heads = {'hm': 1, 'vaf': 2, 'haf': 1}
    model = get_pose_net(num_layers=34, heads=heads, head_conv=256, down_ratio=4)

    train_loader = DataLoader(TuSimple(path=args.dataset_dir, image_set='train'), batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(TuSimple(path=args.dataset_dir, image_set='val'), batch_size=args.batch_size, shuffle=False, num_workers=4)

    batch_size = 3
    train(batch_size, lr, num_epochs, weights, train_loader, val_loader, model)
