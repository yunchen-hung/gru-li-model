from turtle import forward
import torch
import torch.nn as nn

from ..basic_module import BasicModule


class LSTM(BasicModule):
    def __init__(self, input_dim: int, hidden_dim: int, device: str = 'cpu') -> None:
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.fc_in = nn.Linear(input_dim, hidden_dim * 4)
        self.fc_rec = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)

        self.c0 = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)
        self.h0 = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)

    def init_state(self, batch_size):
        return (self.h0.repeat(batch_size, 1).to(self.device),
                self.c0.repeat(batch_size, 1).to(self.device))

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

        return h_next, c_next, [z_i, z_f, z_o]
