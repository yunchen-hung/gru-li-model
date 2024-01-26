from math import gamma
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import smooth_l1_loss, mse_loss
from torch.distributions import Categorical


class Criterion(nn.Module):
    def __init__(self, encoding_loss, recall_loss):
        """
        encoding_loss: loss for encoding phase
        recall_loss: loss for recall phase
        """
        super().__init__()
        self.encoding_loss = encoding_loss
        self.recall_loss = recall_loss
    
    def forward(self, output, gt, memory_num=8):
        loss = torch.tensor(0.0).to(output.device)
        if self.encoding_loss:
            loss += self.encoding_loss(output[:memory_num], gt[:memory_num])
        if self.recall_loss:
            loss += self.recall_loss(output[memory_num:], gt[memory_num:])
        return loss
