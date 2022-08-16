import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import load_act_fn
from ..basic_module import BasicModule


class TCM(BasicModule):
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

        self.empty_parameter = nn.Parameter(torch.zeros(1, device=device))
    
    def init_state(self, batch_size, recall=False, flush_level=1.0, prev_state=None):
        return torch.zeros((batch_size, self.dim), device=self.device)
    
    def forward(self, inp, state):
        if self.encoding:
            c_in = torch.mv(self.W_fc, inp)
            c_in = c_in / torch.norm(c_in, p=2)
            state = F.relu(state * self.alpha + c_in * (1 - self.alpha))
            self.W_fc = self.W_fc + self.lr_fc * torch.outer(state.squeeze(), inp.squeeze())
            self.W_cf = self.W_cf + self.lr_cf * torch.outer(inp.squeeze(), state.squeeze())
            # print(inp, c_in, state, self.W_fc, self.W_cf)
            self.write(self.W_fc, 'W_fc')
            self.write(self.W_cf, 'W_cf')
            self.write(state, 'state')
            return inp, torch.zeros(self.dim), state
        elif self.retrieving:
            # print(state.shape, self.W_cf.shape)
            f_in = torch.mv(self.W_cf, state.squeeze())
            retrieved_idx = torch.argmax(f_in * self.not_recalled)
            retrieved_memory = torch.zeros(self.dim, device=self.device)
            retrieved_memory[retrieved_idx] = 1
            c_in = torch.mv(self.W_fc, retrieved_memory)
            state = F.relu(state * self.alpha + c_in * (1 - self.alpha))
            self.not_recalled = self.not_recalled * (1 - retrieved_memory)
            # print(f_in, retrieved_idx, retrieved_memory, c_in, state)
            self.write(f_in, 'f_in')
            self.write(retrieved_idx, 'retrieved_idx')
            self.write(retrieved_memory, 'retrieved_memory')
            self.write(state, 'state')
            return retrieved_memory, torch.zeros(self.dim), state
    
    def set_encoding(self, status):
        self.encoding = status
    
    def set_retrieval(self, status):
        self.retrieving = status

    def reset_memory(self):
        self.W_cf = torch.zeros((self.dim, self.dim), device=self.device)
        self.W_fc = torch.eye(self.dim, device=self.device) * (1 - self.lr_fc)
        self.not_recalled = torch.ones(self.dim, device=self.device)


class TCMRNN(BasicModule):
    def __init__(self, hidden_dim: int, act_fn='ReLU', lr_cf: float = 1.0, lr_fc: float = 0.9, alpha: float = 0.5, device: str = 'cpu'):
        super().__init__()
        self.device = device

        self.encoding = False
        self.retrieving = False

        self.lr_cf = lr_cf
        self.lr_fc = lr_fc
        self.alpha = alpha
        self.hidden_dim = hidden_dim

        self.act_fn = load_act_fn(act_fn)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)

        self.W_cf = torch.zeros((hidden_dim, hidden_dim), device=device)
        self.W_fc = torch.eye(hidden_dim, device=device) * (1 - lr_fc)

        self.not_recalled = torch.ones(hidden_dim, device=device)
    
    def init_state(self, batch_size, recall=False, flush_level=1.0, prev_state=None):
        return torch.zeros((batch_size, self.hidden_dim), device=self.device)
    
    def forward(self, inp, state):
        if self.encoding:
            c_in = torch.mv(self.W_fc, inp)
            c_in = c_in / torch.norm(c_in, p=2)
            state = self.act_fn(self.fc_hidden(state) * self.alpha + c_in * (1 - self.alpha))
            self.W_fc = self.W_fc + self.lr_fc * torch.outer(state.squeeze(), inp.squeeze())
            self.W_cf = self.W_cf + self.lr_cf * torch.outer(inp.squeeze(), state.squeeze())
            # print(inp, c_in, state, self.W_fc, self.W_cf)
            self.write(self.W_fc, 'W_fc')
            self.write(self.W_cf, 'W_cf')
            self.write(state, 'state')
            return inp, torch.zeros(self.hidden_dim), state
        elif self.retrieving:
            # print(state.shape, self.W_cf.shape)
            f_in = torch.mv(self.W_cf, state.squeeze())
            f_in_inhibit_recall = f_in * self.not_recalled
            retrieved_idx = torch.argmax(f_in_inhibit_recall, requires_grad=True)
            retrieved_memory = F.one_hot(retrieved_idx, self.hidden_dim).float()
            c_in = torch.mv(self.W_fc, f_in)
            state = self.act_fn(self.fc_hidden(state) * self.alpha + c_in * (1 - self.alpha))
            self.not_recalled = self.not_recalled * (1 - retrieved_memory)
            # print(f_in, retrieved_idx, retrieved_memory, c_in, state)
            self.write(f_in, 'f_in')
            self.write(retrieved_idx, 'retrieved_idx')
            self.write(retrieved_memory, 'retrieved_memory')
            self.write(state, 'state')
            return retrieved_memory, torch.zeros(self.hidden_dim), state
    
    def set_encoding(self, status):
        self.encoding = status
    
    def set_retrieval(self, status):
        self.retrieving = status

    def reset_memory(self):
        self.W_cf = torch.zeros((self.hidden_dim, self.hidden_dim), device=self.device)
        self.W_fc = torch.eye(self.hidden_dim, device=self.device) * (1 - self.lr_fc)
        self.not_recalled = torch.ones(self.hidden_dim, device=self.device)
