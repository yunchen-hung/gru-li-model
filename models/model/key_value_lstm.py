import imp
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..planning.lstm import LSTM
from ..utils import load_act_fn, softmax
from ..memory import KeyValueMemory
from ..basic_module import BasicModule

class KeyValueLSTM(BasicModule):
    def __init__(self, memory_module: KeyValueMemory, input_dim: int, other_input_dim: int, output_dim: int, hidden_dim: int = 256, decision_dim: int = 128, 
        context_embedding_dim: int = 16, act_fn="ReLU", em_gate_act_fn="Sigmoid", device: str = 'cpu'):
        super().__init__()
        self.device = device

        self.lstm = LSTM(context_embedding_dim + other_input_dim, hidden_dim, device)
        self.memory_module = memory_module

        self.input_dim = input_dim
        self.decision_dim = decision_dim

        self.fc_in = nn.Linear(input_dim, context_embedding_dim)
        self.fc_decision = nn.Linear(hidden_dim, decision_dim)
        self.fc_actor = nn.Linear(decision_dim, output_dim)
        self.fc_critic = nn.Linear(decision_dim, 1)
        self.fc_em_gate_in = nn.Linear(context_embedding_dim, hidden_dim)
        self.fc_em_gate_rec = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.act_fn = load_act_fn(act_fn)
        self.em_gate_act_fn = load_act_fn(em_gate_act_fn)

    def init_state(self, batch_size):
        return self.lstm.init_state(batch_size)

    def forward(self, inp, state, beta=1.0):
        context_embedding = self.fc_in(inp[:self.input_dim])
        h_prev, c_prev = state
        lstm_input = torch.cat((context_embedding,inp[self.input_dim:]), -1)
        h, c, z = self.lstm(lstm_input, state)
        o = z[2]
        # decision = self.act_fn(self.fc_decision(h))
        em_gate = self.em_gate_act_fn(self.fc_em_gate_in(context_embedding) + self.fc_em_gate_rec(h_prev))
        memory = self.memory_module.retrieve(context_embedding, 1.0)
        c2 = c + memory * em_gate
        self.memory_module.encode((context_embedding, c2))
        h2 = torch.mul(o, c2.tanh())
        decision2 = self.act_fn(self.fc_decision(h2))
        pi_a = softmax(self.fc_actor(decision2), beta)
        value = self.fc_critic(decision2)

        # record
        self.write(context_embedding, 'context_embedding')
        self.write(em_gate, 'em_gate')
        self.write(memory, 'memory')
        self.write(c2, 'c')
        self.write(h, 'h')
        self.write(decision2, 'decision')
        self.write(pi_a, 'pi_a')
        self.write(value, 'value')

        return pi_a, value, (h2, c2)

    def set_encoding(self, status):
        self.memory_module.encoding = status
    
    def set_retrieval(self, status):
        self.memory_module.retrieving = status
