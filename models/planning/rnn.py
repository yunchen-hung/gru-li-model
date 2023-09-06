import torch
import torch.nn as nn

from utils import import_attr
from ..base_module import BasicModule

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

    def init_state(self, batch_size, recall=False, flush_level=1.0, prev_state=None):
        if prev_state is None:
            prev_state = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        if self.init_state_type == 'zero':
            init_state = torch.zeros(batch_size, self.hidden_dim).to(self.device) * flush_level + prev_state * (1 - flush_level)
        elif self.init_state_type == 'random':
            init_state = torch.randn(batch_size, self.hidden_dim).to(self.device) * flush_level + prev_state * (1 - flush_level)
        elif self.init_state_type == 'train':
            init_state = flush_level * self.c0.repeat(batch_size, 1) + prev_state * (1 - flush_level)
        elif self.init_state_type == 'train_diff': 
            if recall:
                init_state = self.c0_recall.repeat(batch_size, 1) * flush_level + prev_state * (1 - flush_level)
            else:
                init_state = self.c0.repeat(batch_size, 1) * flush_level + prev_state * (1 - flush_level)
        return init_state

    def forward(self, inp, state):
        next_state = self.act_fn(self.fc_in(inp.to(self.device)) + self.fc_rec(state.to(self.device)))
        return next_state


class CTRNN(BasicModule):
    def __init__(self, input_dim: int, hidden_dim: int, act_fn: str = "Tanh", dt: int = 100, tau: int = 100, init_state_type='train', device: str = 'cpu') -> None:
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.init_state_type = init_state_type

        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.fc_rec = nn.Linear(hidden_dim, hidden_dim)

        if act_fn == "ReTanh":
            self.act_fn = lambda x: torch.max(torch.tanh(x), torch.zeros_like(x))
        else:
            self.act_fn = import_attr("torch.nn.{}".format(act_fn))()

        self.dt = dt
        self.tau = tau
        self.alpha = self.dt / self.tau

        self.h0 = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)
        if self.init_state_type == 'train_diff':
            self.c0_recall = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)
            self.h0_recall = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)

        self.ah = torch.zeros(hidden_dim).to(self.device)

    def init_state(self, batch_size, recall=False, flush_level=1.0, prev_state=None):
        if prev_state is None:
            prev_state = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        if self.init_state_type == 'zero':
            init_state = torch.zeros(batch_size, self.hidden_dim).to(self.device) * flush_level + prev_state * (1 - flush_level)
        elif self.init_state_type == 'random':
            init_state = torch.randn(batch_size, self.hidden_dim).to(self.device) * flush_level + prev_state * (1 - flush_level)
        elif self.init_state_type == 'train':
            init_state = flush_level * self.h0.repeat(batch_size, 1) + prev_state * (1 - flush_level)
        elif self.init_state_type == 'train_diff': 
            if recall:
                init_state = self.h0_recall.repeat(batch_size, 1) * flush_level + prev_state * (1 - flush_level)
            else:
                init_state = self.h0.repeat(batch_size, 1) * flush_level + prev_state * (1 - flush_level)
        self.ah = init_state
        return init_state

    def forward(self, inp, state):
        self.ah = (1-self.alpha)*self.ah + self.alpha*(self.fc_rec(state.to(self.device)) + self.fc_in(inp.to(self.device)))
        next_state = self.act_fn(self.ah)
        return next_state
