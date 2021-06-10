import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float, int)): self.alpha = torch.Tensor([1 - alpha, alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.eps = 1e-10

    def forward(self, outputs, targets):
        outputs, targets = torch.sigmoid(outputs.view(-1, 1)), targets.view(-1, 1).long() # (N, 1)
        outputs = torch.cat((1 - outputs, outputs), dim=1) # (N, 2)

        pt = outputs.gather(1, targets).view(-1)
        logpt = torch.log(outputs + self.eps)
        logpt = logpt.gather(1, targets).view(-1)

        if self.alpha is not None:
            if self.alpha.type() != outputs.data.type():
                self.alpha = self.alpha.type_as(outputs.data)
            at = self.alpha.gather(0, targets.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()

class IoULoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(IoULoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        mask = (targets != self.ignore_index).float()
        targets = targets.float()
        num = torch.sum(outputs*targets*mask)
        den = torch.sum(outputs*mask + targets*mask - outputs*targets*mask)
        return 1 - num/den

# borrowed from https://github.com/xingyizhou/CenterNet/blob/master/src/lib/models/losses.py
def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos  = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss

class RegL1Loss(nn.Module):
    def __init__(self, ignore_index=255):
        super(RegL1Loss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, output, target, mask):
        _mask = mask.detach().clone()
        _mask[mask == self.ignore_index] = 0.
        loss = F.l1_loss(output * _mask, target * _mask, reduction='sum')
        loss = loss / (_mask.sum() + 1e-12)
        return loss

class SADLoss(nn.Module):
  def __init__(self):
    super(SADLoss, self).__init__()

  def forward(self, layer_ops):
    loss = 0.0
    upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    for l in range(len(layer_ops)-1):
        if layer_ops[l].size()[2] == layer_ops[l+1].size()[2]:
            layer_cur = torch.softmax(torch.sum(layer_ops[l]**2, dim=1).view(layer_ops[l].size()[0], -1), dim=1)
            layer_next = torch.softmax(torch.sum(layer_ops[l+1].detach()**2, dim=1).view(layer_ops[l+1].size()[0], -1), dim=1)
        elif layer_ops[l].size()[2] > layer_ops[l+1].size()[2]:
            layer_cur = torch.softmax(torch.sum(layer_ops[l]**2, dim=1).view(layer_ops[l].size()[0], -1), dim=1)
            layer_next = torch.softmax(torch.sum(upsample(layer_ops[l+1].detach()**2), dim=1).view(layer_ops[l+1].size()[0], -1), dim=1)
        elif layer_ops[l].size()[2] < layer_ops[l+1].size()[2]:
            layer_cur = torch.softmax(torch.sum(upsample(layer_ops[l]**2), dim=1).view(layer_ops[l].size()[0], -1), dim=1)
            layer_next = torch.softmax(torch.sum(layer_ops[l+1].detach()**2, dim=1).view(layer_ops[l+1].size()[0], -1), dim=1)

        loss += F.mse_loss(layer_cur, layer_next, reduction='mean')
    return loss
