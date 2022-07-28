from turtle import forward
import torch
import torch.nn as nn

from ..basic_module import BasicModule


class LSTM(BasicModule):
    def __init__(self, input_dim: int, hidden_dim: int, init_state_type='train', device: str = 'cpu') -> None:
        """
        init_state_type: 
            zero: initialize states with all zeros
            random: initialize states with random values
            train: train init states
            train_diff: train init states of stimuli presenting and 
        """
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.init_state_type = init_state_type

        self.fc_in = nn.Linear(input_dim, hidden_dim * 4)
        self.fc_rec = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)

        self.c0 = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)
        self.h0 = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)
        if self.init_state_type == 'train_diff':
            self.c0_recall = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)
            self.h0_recall = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)

    def init_state(self, batch_size, recall=False, flush_level=1.0, prev_state=None):
        if prev_state is None:
            h = torch.zeros(batch_size, self.hidden_dim).to(self.device)
            c = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        else:
            h, c = prev_state
        if self.init_state_type == 'zero':
            c0 = torch.zeros(batch_size, self.hidden_dim).to(self.device) * flush_level + c * (1 - flush_level)
            h0 = torch.zeros(batch_size, self.hidden_dim).to(self.device) * flush_level + h * (1 - flush_level)
        elif self.init_state_type == 'random':
            c0 = torch.randn(batch_size, self.hidden_dim).to(self.device) * flush_level + c * (1 - flush_level)
            h0 = torch.randn(batch_size, self.hidden_dim).to(self.device) * flush_level + h * (1 - flush_level)
        elif self.init_state_type == 'train':
            c0 = flush_level * self.c0.repeat(batch_size, 1) + c * (1 - flush_level)
            h0 = flush_level * self.h0.repeat(batch_size, 1) + h * (1 - flush_level)
        elif self.init_state_type == 'train_diff': 
            if recall:
                c0 = self.c0_recall.repeat(batch_size, 1) * flush_level + c * (1 - flush_level)
                h0 = self.h0_recall.repeat(batch_size, 1) * flush_level + h * (1 - flush_level)
            else:
                c0 = self.c0.repeat(batch_size, 1) * flush_level + c * (1 - flush_level)
                h0 = self.h0.repeat(batch_size, 1) * flush_level + h * (1 - flush_level)
        return h0, c0

    def forward(self, inp, state):
        h, c = state
        preact = self.fc_in(inp.to(self.device)) + self.fc_rec(h.to(self.device))
        z = preact[:, :self.hidden_dim].tanh()
        z_i, z_f, z_o = preact[:, self.hidden_dim:].sigmoid().chunk(3, 1)
        c_next = torch.mul(z_f, c) + torch.mul(z_i, z)
        h_next = torch.mul(z_o, c_next.tanh())

        self.write(z, 'z')
        self.write(z_i, 'z_i')
        self.write(z_f, 'z_f')
        self.write(z_o, 'z_o')
        self.write(c_next, 'c')
        self.write(h_next, 'h')
        self.write(torch.mul(z_f, c), "rec")
        self.write(torch.mul(z_i, z), "inp")
        self.write(c_next.tanh(), "c_next_act_fn")
        self.write(h_next, "h_next")

        return h_next, c_next, [z_i, z_f, z_o]
