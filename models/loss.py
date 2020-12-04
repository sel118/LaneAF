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
        outputs, targets = torch.sigmoid(outputs.view(-1, 1)), targets.view(-1, 1) # (N, 1)
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
