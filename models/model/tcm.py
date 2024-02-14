import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ..utils import softmax
from ..base_module import BasicModule


class TCM(BasicModule):
    def __init__(self, dim: int, lr_cf: float = 1.0, lr_fc: float = 0.9, alpha: float = 0.5, threshold = 0.0, 
    start_recall_with_ith_item_init=0, rand_mem=False, softmax_temperature=1.0, device: str = 'cpu'):
        super().__init__()
        self.device = device

        self.encoding = False
        self.retrieving = False

        self.lr_cf = lr_cf
        self.lr_fc = lr_fc
        self.alpha = alpha
        self.threshold = threshold
        self.dim = dim
        self.rand_mem = rand_mem
        self.softmax_temperature = softmax_temperature

        self.W_cf = torch.zeros((dim, dim), device=device)
        # self.W_fc = torch.eye(dim, device=device) * (1 - lr_fc)
        self.W_fc = torch.eye(dim, device=device)
        # self.W_cf = torch.zeros((1, dim, dim), device=device, requires_grad=True)
        # self.W_fc = (torch.eye(dim, device=device, requires_grad=True) * (1 - lr_fc)).repeat(1, 1, 1)

        self.not_recalled = torch.ones(dim, device=device)

        self.empty_parameter = nn.Parameter(torch.zeros(1, device=device))

        self.start_recall_with_ith_item_init = start_recall_with_ith_item_init
        self.ith_item_state = torch.zeros((1, self.dim), device=self.device, requires_grad=True)
    
    def init_state(self, batch_size, recall=False, flush_level=1.0, prev_state=None):
        self.current_timestep = 0
        if recall and self.start_recall_with_ith_item_init:
            return self.ith_item_state.clone()
        return torch.zeros((batch_size, self.dim), device=self.device)
    
    def forward(self, inp, state):
        if self.encoding:
            c_in = torch.mv(self.W_fc, inp.squeeze())
            # c_in = torch.bmm(self.W_fc, torch.unsqueeze(inp, dim=2)).squeeze(2)
            c_in = c_in / torch.norm(c_in, p=2)
            state = state * self.alpha + c_in * (1 - self.alpha)
            state = F.normalize(state, p=2, dim=1)
            self.W_fc = self.W_fc + self.lr_fc * torch.outer(state.squeeze(), inp.squeeze())
            self.W_cf = self.W_cf + self.lr_cf * torch.outer(inp.squeeze(), state.squeeze())
            # self.W_fc = self.W_fc + self.lr_fc * torch.einsum("ik,ij->ikj", [state, inp])    # each column is a state
            # self.W_cf = self.W_cf + self.lr_cf * torch.einsum("ik,ij->ikj", [inp, state])    # store memory, each row is a state
            # print(inp, c_in, state, self.W_fc, self.W_cf)
            self.write(self.W_fc, 'W_fc')
            self.write(self.W_cf, 'W_cf')
            self.write(state, 'state')

            self.current_timestep += 1
            if self.current_timestep == self.start_recall_with_ith_item_init:
                self.ith_item_state = state.detach().clone()

            self.not_recalled += inp.squeeze()

            return inp, torch.zeros((1, self.dim)), state, None
        elif self.retrieving:
            # print(state.shape, self.W_cf.shape)
            # f_in_raw = torch.bmm(F.normalize(self.W_cf, p=2, dim=2), F.normalize(torch.unsqueeze(state, dim=2), p=2)).squeeze(2)
            f_in_raw = torch.mv(self.W_cf, state.squeeze())
            # f_in_filtered = (F.relu(f_in_raw - torch.max(f_in_raw) * self.threshold)) * self.not_recalled
            f_in = (softmax(f_in_raw.unsqueeze(0), self.softmax_temperature) * self.not_recalled).squeeze(0) 
            f_in = f_in / torch.sum(f_in)
            # print(f_in, f_in_filtered)
            if self.rand_mem:
                retrieved_idx = Categorical(f_in).sample()
            else:
                retrieved_idx = torch.argmax(f_in)
            retrieved_memory = torch.zeros(self.dim, device=self.device)
            retrieved_memory[retrieved_idx] = 1
            c_in = torch.mv(self.W_fc, retrieved_memory)
            # c_in = torch.bmm(self.W_fc, torch.unsqueeze(retrieved_memory, dim=2)).squeeze(2)
            state = F.relu(state * self.alpha + c_in * (1 - self.alpha))
            state = F.normalize(state, p=2, dim=1)
            self.not_recalled = self.not_recalled * (1 - retrieved_memory)
            # print(f_in, retrieved_idx, retrieved_memory, c_in, state)
            self.write(f_in_raw, 'similarity')
            self.write(retrieved_idx, 'retrieved_idx')
            self.write(retrieved_memory, 'retrieved_memory')
            self.write(state, 'state')
            self.write(self.not_recalled, 'not_recalled')
            return retrieved_memory.unsqueeze(0), torch.zeros((1, self.dim)), state, None
    
    def set_encoding(self, status):
        self.encoding = status
    
    def set_retrieval(self, status):
        self.retrieving = status

    def reset_memory(self):
        self.W_cf = torch.zeros((self.dim, self.dim), device=self.device)
        self.W_fc = torch.eye(self.dim, device=self.device) * (1 - self.lr_fc)
        self.not_recalled = torch.zeros(self.dim, device=self.device)