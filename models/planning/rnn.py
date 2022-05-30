import torch
import torch.nn as nn

from utils import import_attr
from ..basic_module import BasicModule

INIT_STATE_METHODS = ['zero', 'random', 'train', 'keep_last']


class RNN(BasicModule):
    def __init__(self, input_dim: int, hidden_dim: int, act_fn: str, init_state_method: str = 'zero', device: str = 'cpu') -> None:
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_rec = nn.Linear(hidden_dim, hidden_dim)

        if act_fn == "ReTanh":
            self.act_fn = lambda x: torch.max(torch.tanh(x), torch.zeros_like(x))
        else:
            self.act_fn = import_attr("torch.nn.{}".format(act_fn))()

    def init_state(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim).to(self.device)

    def forward(self, inp, state):
        next_state = self.act_fn(self.fc_in(inp.to(self.device)) + self.fc_rec(state.to(self.device)))
        return next_state


class CTRNN(BasicModule):
    def __init__(self, input_dim: int, hidden_dim: int, act_fn: str, dt: int, tau: int, device: str = 'cpu') -> None:
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_rec = nn.Linear(hidden_dim, hidden_dim)

        if act_fn == "ReTanh":
            self.act_fn = lambda x: torch.max(torch.tanh(x), torch.zeros_like(x))
        else:
            self.act_fn = import_attr("torch.nn.{}".format(act_fn))()

        self.dt = dt
        self.tau = tau
        self.alpha = self.dt / self.tau

    def init_state(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim).to(self.device)

    def forward(self, inp, state):
        self.ah = (1-self.alpha)*self.ah + self.alpha*(self.fc_rec(state.to(self.device)) + self.fc_in(inp.to(self.device)))
        next_state = self.act_fn(self.ah)
        return next_state


