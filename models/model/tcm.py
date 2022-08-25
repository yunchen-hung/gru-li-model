from importlib.metadata import requires
from tkinter import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ..utils import load_act_fn, softmax
from ..basic_module import BasicModule


class TCM(BasicModule):
    def __init__(self, dim: int, lr_cf: float = 1.0, lr_fc: float = 0.9, alpha: float = 0.5, threshold = 0.0, device: str = 'cpu'):
        super().__init__()
        self.device = device

        self.encoding = False
        self.retrieving = False

        self.lr_cf = lr_cf
        self.lr_fc = lr_fc
        self.alpha = alpha
        self.threshold = threshold
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
            f_in_filtered = (F.relu(f_in - torch.max(f_in) * self.threshold)) * self.not_recalled
            # print(f_in, f_in_filtered)
            # retrieved_idx = torch.argmax(f_in * self.not_recalled)
            retrieved_idx = Categorical(f_in_filtered).sample()
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
    def __init__(self, hidden_dim: int, act_fn='ReLU', lr_cf: float = 1.0, lr_fc: float = 0.9, dt: float = 10, tau: float = 20, record_recalled: bool = False, 
    mem_gate_type="constant", output_type="recalled_item", device: str = 'cpu'):
        super().__init__()
        self.device = device

        self.encoding = False
        self.retrieving = False

        self.lr_cf = lr_cf
        self.lr_fc = lr_fc
        self.alpha = float(dt) / float(tau)
        self.record_recalled = record_recalled

        self.hidden_dim = hidden_dim
        self.act_fn = load_act_fn(act_fn)

        self.mem_gate_type = mem_gate_type
        if mem_gate_type == "vector":
            self.mem_gate = nn.Linear(hidden_dim, hidden_dim)
        elif mem_gate_type == "scalar":
            self.mem_gate = nn.Linear(hidden_dim, 1)
        elif mem_gate_type == "constant":
            self.mem_gate = 0.5
        else:
            raise AttributeError("Invalid memory gate type, should be vector or scalar")
        self.output_type = output_type

        # self.fc_in = nn.Linear(hidden_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc_decision = nn.Linear(hidden_dim, hidden_dim)
        self.fc_critic = nn.Linear(hidden_dim, 1)

        self.W_cf = torch.zeros((hidden_dim, hidden_dim), device=device, requires_grad=True)
        self.W_fc = torch.eye(hidden_dim, device=device, requires_grad=True) * (1 - lr_fc)

        self.hidden_state = torch.zeros((1, self.hidden_dim), device=self.device, requires_grad=True)

        self.not_recalled = torch.ones(hidden_dim, device=device)
    
    def init_state(self, batch_size, recall=False, flush_level=1.0, prev_state=None):
        if recall:
            self.hidden_state = self.hidden_state + torch.normal(0.0, torch.mean(self.hidden_state) * flush_level)
            state = self.act_fn(self.hidden_state)
        else:
            self.hidden_state = torch.zeros((batch_size, self.hidden_dim), device=self.device, requires_grad=True)
            state = torch.zeros((batch_size, self.hidden_dim), device=self.device, requires_grad=True)
        return state
    
    def forward(self, inp, state):
        if self.encoding:
            c_in = torch.mv(self.W_fc, inp)
            c_in = c_in / torch.norm(c_in, p=2)
            if self.mem_gate_type == "constant":
                gate = self.mem_gate
            else:
                gate = self.mem_gate(state).sigmoid()
            self.hidden_state = self.hidden_state * (1 - self.alpha) + (self.fc_hidden(state) * (1 - gate) + c_in * gate) * self.alpha
            state = self.act_fn(self.hidden_state)
            self.W_fc = self.W_fc + self.lr_fc * torch.outer(state.squeeze(), inp.squeeze())    # each column is a state
            self.W_cf = self.W_cf + self.lr_cf * torch.outer(inp.squeeze(), state.squeeze())    # store memory, each row is a state
            # print(self.hidden_state)
            # print(state)
            decision = softmax(self.fc_decision(state))
            # print(inp, c_in, state, self.W_fc, self.W_cf)
            if self.output_type == "decision":
                output = decision
            elif self.output_type == "recalled_item":
                output = inp
            else:
                raise AttributeError("Invalid output type, should be decision or recalled_item")
            value = self.fc_critic(state)

            self.write(self.W_fc, 'W_fc')
            self.write(self.W_cf, 'W_cf')
            self.write(state, 'state')

            return output, value, state
        elif self.retrieving:
            # print(state.shape, self.W_cf.shape)
            f_in = softmax(torch.mv(self.W_cf, state.squeeze()))
            f_in_inhibit_recall = f_in * self.not_recalled
            retrieved_idx = torch.argmax(f_in_inhibit_recall)
            # retrieved_idx = Categorical(f_in_inhibit_recall).sample()
            retrieved_memory = F.one_hot(retrieved_idx, self.hidden_dim).float()
            retrieved_memory.requires_grad = True
            c_in = torch.mv(self.W_fc, retrieved_memory)
            c_in = c_in / torch.norm(c_in, p=2)
            if self.mem_gate_type == "constant":
                gate = self.mem_gate
            else:
                gate = self.mem_gate(state).sigmoid()
            self.hidden_state = self.hidden_state * (1 - self.alpha) + (self.fc_hidden(state) * (1 - gate) + c_in * gate) * self.alpha
            state = self.act_fn(self.hidden_state)
            decision = softmax(self.fc_decision(state))
            # self.W_fc = self.W_fc - self.lr_fc * torch.outer(state.squeeze(), retrieved_memory.squeeze())
            # self.not_recalled[retrieved_idx] = 0
            if self.record_recalled:
                self.not_recalled = self.not_recalled * (1 - retrieved_memory)
            if self.output_type == "decision":
                output = decision
            elif self.output_type == "recalled_item":
                output = f_in_inhibit_recall
            else:
                raise AttributeError("Invalid output type, should be decision or recalled_item")
            value = self.fc_critic(state)

            self.write(f_in, 'f_in')
            self.write(retrieved_idx, 'retrieved_idx')
            self.write(retrieved_memory, 'retrieved_memory')
            self.write(state, 'state')

            return output, value, state
    
    def set_encoding(self, status):
        self.encoding = status
    
    def set_retrieval(self, status):
        self.retrieving = status

    def reset_memory(self):
        self.W_cf = torch.zeros((self.hidden_dim, self.hidden_dim), device=self.device)
        self.W_fc = torch.eye(self.hidden_dim, device=self.device) * (1 - self.lr_fc)
        self.not_recalled = torch.ones(self.hidden_dim, device=self.device)


class TCMLSTM(BasicModule):
    def __init__(self, hidden_dim: int, act_fn='ReLU', lr_cf: float = 1.0, lr_fc: float = 0.9, record_recalled: bool = False, mem_gate_type="constant", 
    output_type="recalled_item", device: str = 'cpu'):
        super().__init__()
        self.device = device

        self.encoding = False
        self.retrieving = False

        self.lr_cf = lr_cf
        self.lr_fc = lr_fc
        self.record_recalled = record_recalled

        self.hidden_dim = hidden_dim
        self.act_fn = load_act_fn(act_fn)

        self.mem_gate_type = mem_gate_type
        if mem_gate_type == "vector":
            self.mem_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        elif mem_gate_type == "scalar":
            self.mem_gate = nn.Linear(hidden_dim * 2, 1)
        elif mem_gate_type == "constant":
            self.mem_gate = 0.5
        else:
            raise AttributeError("Invalid memory gate type, should be vector or scalar")
        self.output_type = output_type

        # self.fc_in = nn.Linear(hidden_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)
        self.fc_decision = nn.Linear(hidden_dim, hidden_dim)
        self.fc_critic = nn.Linear(hidden_dim, 1)

        self.W_cf = torch.zeros((hidden_dim, hidden_dim), device=device, requires_grad=True)
        self.W_fc = torch.eye(hidden_dim, device=device, requires_grad=True) * (1 - lr_fc)

        self.not_recalled = torch.ones(hidden_dim, device=device)
    
    def init_state(self, batch_size, recall=False, flush_level=1.0, prev_state=None):
        if recall:
            h_prev, c_prev = prev_state
            h = h_prev + torch.normal(0.0, torch.mean(h_prev) * flush_level)
            c = c_prev + torch.normal(0.0, torch.mean(c_prev) * flush_level)
        else:
            h = torch.zeros((batch_size, self.hidden_dim), device=self.device, requires_grad=True)
            c = torch.zeros((batch_size, self.hidden_dim), device=self.device, requires_grad=True)
        return (h, c)
    
    def forward(self, inp, state):
        h, c = state
        if self.encoding:
            c_in = torch.mv(self.W_fc, inp)
            c_in = c_in / torch.norm(c_in, p=2)

            if self.mem_gate_type == "constant":
                gate = self.mem_gate
            else:
                gate = self.mem_gate(torch.cat((c, h), 1)).sigmoid()
            
            preact = self.fc_hidden(h.to(self.device)) * (1 - gate) + c_in.repeat(4) * gate
            z = preact[:, :self.hidden_dim].tanh()
            z_i, z_f, z_o = preact[:, self.hidden_dim:].sigmoid().chunk(3, 1)
            c_next = torch.mul(z_f, c) + torch.mul(z_i, z)
            h_next = torch.mul(z_o, c_next.tanh())
            state = self.act_fn(h_next)

            self.W_fc = self.W_fc + self.lr_fc * torch.outer(h_next.squeeze(), inp.squeeze())    # each column is a state
            self.W_cf = self.W_cf + self.lr_cf * torch.outer(inp.squeeze(), h_next.squeeze())    # store memory, each row is a state

            # print(self.hidden_state)
            # print(state)
            decision = softmax(self.fc_decision(state))
            # print(inp, c_in, state, self.W_fc, self.W_cf)
            if self.output_type == "decision":
                output = decision
            elif self.output_type == "recalled_item":
                output = inp
            else:
                raise AttributeError("Invalid output type, should be decision or recalled_item")
            value = self.fc_critic(state)

            self.write(self.W_fc, 'W_fc')
            self.write(self.W_cf, 'W_cf')
            self.write(state, 'state')

            return output, value, (h_next, c_next)
        elif self.retrieving:
            # print(state.shape, self.W_cf.shape)
            f_in = softmax(torch.mv(self.W_cf, h.squeeze()))
            f_in_inhibit_recall = f_in * self.not_recalled
            retrieved_idx = torch.argmax(f_in_inhibit_recall)
            # retrieved_idx = Categorical(f_in_inhibit_recall).sample()
            retrieved_memory = F.one_hot(retrieved_idx, self.hidden_dim).float()
            retrieved_memory.requires_grad = True
            c_in = torch.mv(self.W_fc, retrieved_memory)
            c_in = c_in / torch.norm(c_in, p=2)

            if self.mem_gate_type == "constant":
                gate = self.mem_gate
            else:
                gate = self.mem_gate(torch.cat((c, h), 1)).sigmoid()
            
            preact = self.fc_hidden(h.to(self.device)) * (1 - gate) + c_in.repeat(4) * gate
            z = preact[:, :self.hidden_dim].tanh()
            z_i, z_f, z_o = preact[:, self.hidden_dim:].sigmoid().chunk(3, 1)
            c_next = torch.mul(z_f, c) + torch.mul(z_i, z)
            h_next = torch.mul(z_o, c_next.tanh())
            state = self.act_fn(h_next)

            decision = softmax(self.fc_decision(state))
            # self.W_fc = self.W_fc - self.lr_fc * torch.outer(state.squeeze(), retrieved_memory.squeeze())
            # self.not_recalled[retrieved_idx] = 0
            if self.record_recalled:
                self.not_recalled = self.not_recalled * (1 - retrieved_memory)
            if self.output_type == "decision":
                output = decision
            elif self.output_type == "recalled_item":
                output = f_in_inhibit_recall
            else:
                raise AttributeError("Invalid output type, should be decision or recalled_item")
            value = self.fc_critic(state)

            self.write(f_in, 'f_in')
            self.write(retrieved_idx, 'retrieved_idx')
            self.write(retrieved_memory, 'retrieved_memory')
            self.write(state, 'state')

            return output, value, (h_next, c_next)
    
    def set_encoding(self, status):
        self.encoding = status
    
    def set_retrieval(self, status):
        self.retrieving = status

    def reset_memory(self):
        self.W_cf = torch.zeros((self.hidden_dim, self.hidden_dim), device=self.device)
        self.W_fc = torch.eye(self.hidden_dim, device=self.device) * (1 - self.lr_fc)
        self.not_recalled = torch.ones(self.hidden_dim, device=self.device)
