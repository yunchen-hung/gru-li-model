from math import gamma
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import smooth_l1_loss, mse_loss
from torch.distributions import Categorical


class MultiSupervisedLoss(nn.Module):
    def __init__(self, criteria, output_index=[0]):
        """
        a list of criterions for mutliple outputs, all for supervised learning
        
        criteria: list of criterion
        output_index: list of int, the index of the output that the criterion is applied to
        """
        super().__init__()
        assert len(criteria) == len(output_index)
        self.criteria = criteria
        self.output_index = output_index
    
    def forward(self, outputs, gts):
        """
        outputs: list of torch.tensor, overall size is (timesteps, batch_size, output_dim)
        gts: list of torch.tensor, overall size is (timesteps, batch_size, output_dim)
        """
        assert len(outputs) > max(self.output_index)
        # loss = torch.tensor(0.0)
        loss = None
        for i in range(len(self.criteria)):
            # print(i, outputs[self.output_index[i]].shape, gts.shape)
            if loss is None:
                loss = self.criteria[i](outputs[self.output_index[i]], gts)
            else:
                loss += self.criteria[i](outputs[self.output_index[i]], gts)
        return loss
    

class MultiRLLoss(nn.Module):
    def __init__(self, criteria, output_index=[0]):
        """
        a list of criterions for mutliple outputs, all for reinforcement learning
        
        criteria: list of criterion
        output_index: list of int, the index of the output that the criterion is applied to
        """
        super().__init__()
        assert len(criteria) == len(output_index)
        self.criteria = criteria
        self.output_index = output_index

    def forward(self, probs, values, rewards, entropys, print_info=False, device="cpu"):
        assert len(probs) == len(values) == len(entropys)
        assert len(probs) > max(self.output_index)
        # loss = torch.tensor(0.0).to(device)
        # policy_gradient = torch.tensor(0.0).to(device)
        # value_loss = torch.tensor(0.0).to(device)
        # pi_ent = torch.tensor(0.0).to(device)
        loss, policy_gradient, value_loss, pi_ent = None, None, None, None
        for i in range(len(self.criteria)):
            l, p, v, ent = self.criteria[i](probs[self.output_index[i]], values[self.output_index[i]], 
                                     rewards, entropys[self.output_index[i]],
                                     print_info, device=device)
            # print(i, l, p, v, ent)
            if loss is None:
                loss = l
                policy_gradient = p
                value_loss = v
                pi_ent = ent
            else:
                loss += l
                policy_gradient += p
                value_loss += v
                pi_ent += ent
        # print()
        return loss, policy_gradient, value_loss, pi_ent
