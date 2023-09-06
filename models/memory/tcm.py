import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_module import BasicModule


class TCMMemory(BasicModule):
    def __init__(self, dim: int, lr_cf: float = 1.0, lr_fc: float = 0.9, alpha: float = 0.5, device: str = 'cpu'):
        super().__init__()
        self.device = device

        self.encoding = False
        self.retrieving = False

        self.lr_cf = lr_cf
        self.lr_fc = lr_fc
        self.alpha = alpha
        self.dim = dim

        self.W_cf = torch.zeros((dim, dim), device=device)
        self.W_fc = torch.eye(dim, device=device) * (1 - lr_fc)

        self.not_recalled = torch.ones(dim, device=device)

    def reset_memory(self):
        self.encoding = False
        self.retrieving = False
        self.W_cf = torch.zeros((self.dim, self.dim), device=self.device)
        self.W_fc = torch.eye(self.dim, device=self.device) * (1 - self.lr_fc)
        self.not_recalled = torch.ones(self.dim, device=self.device)

    def encode(self, value):
        c_in = torch.mv(self.W_fc, value)
        c_in = c_in / torch.norm(c_in, p=2)
        state = F.relu(state * self.alpha + value * (1 - self.alpha))
        self.W_fc = self.W_fc + self.lr_fc * torch.outer(state.squeeze(), value.squeeze())
        self.W_cf = self.W_cf + self.lr_cf * torch.outer(value.squeeze(), state.squeeze())
        self.write(self.W_fc, 'W_fc')
        self.write(self.W_cf, 'W_cf')
        self.write(state, 'state')

    def retrieve(self, query):
        f_in = torch.mv(self.W_cf, query.squeeze())
        retrieved_idx = torch.argmax(f_in * self.not_recalled)
        retrieved_memory = torch.zeros(self.dim, device=self.device)
        retrieved_memory[retrieved_idx] = 1
        c_in = torch.mv(self.W_fc, retrieved_memory)
        state = F.relu(state * self.alpha + c_in * (1 - self.alpha))
        self.not_recalled = self.not_recalled * (1 - retrieved_memory)
        self.write(f_in, 'f_in')
        self.write(retrieved_idx, 'retrieved_idx')
        self.write(retrieved_memory, 'retrieved_memory')
        self.write(state, 'state')
        return retrieved_memory

    def get_vals(self):
        return self.W_cf.detach().cpu().numpy()
