from math import ceil

import numpy as np
import cv2

import torch


def tensor2image(tensor, mean, std):
    mean = mean[..., np.newaxis, np.newaxis] # (nc, 1, 1)
    mean = np.tile(mean, (1, tensor.size()[2], tensor.size()[3])) # (nc, H, W)
    std = std[..., np.newaxis, np.newaxis] # (nc, 1, 1)
    std = np.tile(std, (1, tensor.size()[2], tensor.size()[3])) # (nc, H, W)

    image = 255.0*(std*tensor[0].cpu().float().numpy() + mean) # (nc, H, W)
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    image = np.transpose(image, (1, 2, 0)) # (C, H, W) to (H, W, C)
    image = image[:, :, ::-1] # RGB to BGR
    return image.astype(np.uint8) # (H, W, C)

def create_viz(img, mask, VAF, HAF):
    im_out = [] #test
    print(img.shape)
    print(mask.shape)
    print(VAF.shape)
    print(HAF.shape)
    jdasifji
    #down_rate = 5 # downsample visualization by this factor
    #fig, (ax1, ax2, ax3, ax4) = plt.subplots(2, 2)
    
    # visualize VAF
    #q = ax3.quiver(np.arange(0, label.shape[1], down_rate), -np.arange(0, label.shape[0], down_rate), 
    #               VAF[::down_rate, ::down_rate, 0], -VAF[::down_rate, ::down_rate, 1], scale=120)
    # visualize HAF
    #q = ax4.quiver(np.arange(0, label.shape[1], down_rate), -np.arange(0, label.shape[0], down_rate), 
    #               HAF[::down_rate, ::down_rate, 0], -HAF[::down_rate, ::down_rate, 1], scale=120)
    #plt.show()
    return im_out
