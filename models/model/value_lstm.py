import imp
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..planning.lstm import LSTM
from ..utils import load_act_fn, softmax
from ..memory import ValueMemory
from ..basic_module import BasicModule


class ValueMemoryLSTM(BasicModule):
    def __init__(self, memory_module: ValueMemory, input_dim: int, hidden_dim: int, decision_dim: int, output_dim: int, act_fn="ReLU", em_gate_act_fn="Sigmoid", 
    use_memory=True, init_state_type='train', device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.use_memory = use_memory

        self.lstm = LSTM(input_dim, hidden_dim, init_state_type, device)
        self.memory_module = memory_module

        self.decision_dim = decision_dim

        self.fc_decision = nn.Linear(hidden_dim, decision_dim)
        self.fc_actor = nn.Linear(decision_dim, output_dim)
        self.fc_critic = nn.Linear(decision_dim, 1)
        self.fc_em_gate_value = nn.Linear(hidden_dim + decision_dim, 1)

        self.act_fn = load_act_fn(act_fn)
        self.em_gate_act_fn = load_act_fn(em_gate_act_fn)

    def init_state(self, batch_size, recall=False, flush_level=1.0, prev_state=None):
        return self.lstm.init_state(batch_size, recall, flush_level, prev_state)

    def forward(self, inp, state, beta=1.0):
        h, c, z = self.lstm(inp, state)
        o = z[2]
        # print(inp, state)
        # print(h, self.fc_decision(h))
        dec_act = self.act_fn(self.fc_decision(h))
        # print(dec_act)
        # print()
        if self.use_memory:
            em_gate = self.em_gate_act_fn(self.fc_em_gate_value(torch.cat((c, dec_act), 1)))
            memory = self.memory_module.retrieve(c, em_gate)
            c = c + memory
            self.memory_module.encode(c)
            h = torch.mul(o, c.tanh())
            dec_act = self.act_fn(self.fc_decision(h))
            
            self.write(em_gate, 'em_gate')
            self.write(memory, 'memory')
        else:
            self.write(dec_act, 'dec_act')
        # print(dec_act)
        pi_a = softmax(self.fc_actor(dec_act), beta)
        value = self.fc_critic(dec_act)

        # record
        self.write(dec_act, 'dec_act')
        self.write(c, 'c')
        self.write(h, 'h')
        self.write(pi_a, 'pi_a')
        self.write(value, 'value')

        return pi_a, value, (h, c)

    def set_encoding(self, status):
        self.memory_module.encoding = status
    
    def set_retrieval(self, status):
        self.memory_module.retrieving = status
