import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ..utils import softmax
from ..base_module import BasicModule


class TCM(BasicModule):
    def __init__(self, dim: int, threshold = 0.0, beta=0.5, gamma_fc=0.5, rand_mem=False, 
    start_recall_with_ith_item_init=0, softmax_temperature=1.0, device: str = 'cpu'):
        super().__init__()
        self.device = device

        self.encoding = False
        self.retrieving = False

        self.dim = dim
        self.beta = beta
        self.rho = 1 - beta
        self.gamma_fc = gamma_fc
        self.threshold = threshold
        self.rand_mem = rand_mem
        self.softmax_temperature = softmax_temperature

        self.W_cf = torch.zeros((dim, dim), device=device)      # W_fc = W_cf.T
        # self.W_fc = torch.eye(dim, device=device) * (1 - lr_fc)
        self.W_fc_pre = torch.eye(dim, device=device)
        # self.W_cf = torch.zeros((1, dim, dim), device=device, requires_grad=True)
        # self.W_fc = (torch.eye(dim, device=device, requires_grad=True) * (1 - lr_fc)).repeat(1, 1, 1)

        self.not_recalled = torch.zeros(dim, device=device)

        self.empty_parameter = nn.Parameter(torch.zeros(1, device=device))

        self.start_recall_with_ith_item_init = start_recall_with_ith_item_init
        self.ith_item_state = torch.zeros((1, self.dim), device=self.device, requires_grad=True)
    
    def init_state(self, batch_size, recall=False, flush_level=1.0, prev_state=None):
        self.current_timestep = 0
        if recall and self.start_recall_with_ith_item_init:
            return self.ith_item_state.clone()
        return torch.zeros((batch_size, self.dim), device=self.device)
    
    def forward(self, f_t, c_t):
        """
        f_t: input item
        c_t: state
        """
        if self.encoding:
            c_in_enc = torch.mv(self.W_fc_pre, f_t.squeeze())
            # c_in = torch.bmm(self.W_fc, torch.unsqueeze(inp, dim=2)).squeeze(2)
            # c_in_enc = c_in_enc / torch.norm(c_in_enc, p=2)
            c_t = c_t * self.rho + c_in_enc * self.beta
            c_t = F.normalize(c_t, p=2, dim=1)
            self.W_cf = self.W_cf + torch.outer(f_t.squeeze(), c_t.squeeze())   # each row is a state
            self.write(self.W_cf, 'W_cf')
            self.write(c_t, 'state')

            self.current_timestep += 1
            if self.current_timestep == self.start_recall_with_ith_item_init:
                self.ith_item_state = c_t.detach().clone()

            self.not_recalled[torch.argmax(f_t)] = 1

            # print(f_t.shape, c_t.shape)

            return [f_t], [torch.zeros(1)], c_t, None
        elif self.retrieving:
            f_in_raw = torch.mv(self.W_cf, c_t.squeeze())
            f_in = (softmax(f_in_raw.unsqueeze(0), self.softmax_temperature) * self.not_recalled).squeeze(0) 
            if torch.max(f_in) < self.threshold:
                retrieved_idx = -1
                retrieved_memory = torch.zeros(self.dim, device=self.device)
            else:
                f_in = f_in / torch.sum(f_in)
                if self.rand_mem:
                    retrieved_idx = Categorical(f_in).sample()
                else:
                    retrieved_idx = torch.argmax(f_in)
                retrieved_memory = torch.zeros(self.dim, device=self.device)
                retrieved_memory[retrieved_idx] = 1

            c_in_rec = (1 - self.gamma_fc) * torch.mv(self.W_fc_pre, retrieved_memory) + self.gamma_fc * torch.mv(self.W_cf.T, retrieved_memory)
            c_t = c_t * self.rho + c_in_rec * self.beta
            c_t = F.normalize(c_t, p=2, dim=1)

            self.not_recalled = self.not_recalled * (1 - retrieved_memory)

            self.write(f_in_raw, 'similarity')
            self.write(retrieved_idx, 'retrieved_idx')
            self.write(retrieved_memory, 'retrieved_memory')
            self.write(c_t, 'state')
            self.write(self.not_recalled, 'not_recalled')

            output_memory = torch.zeros(1, self.dim, device=self.device)
            output_memory[0, retrieved_idx+1] = 1
            
            return [output_memory], [torch.zeros(1)], c_t, None
    
    def set_encoding(self, status):
        self.encoding = status
    
    def set_retrieval(self, status):
        self.retrieving = status

    def reset_memory(self, flush=True):
        if flush:
            self.W_cf = torch.zeros((self.dim, self.dim), device=self.device)      # W_fc = W_cf.T

        self.W_fc_pre = torch.eye(self.dim, device=self.device)

        self.not_recalled = torch.zeros(self.dim, device=self.device)
