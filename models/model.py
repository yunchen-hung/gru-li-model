import imp
import torch
import torch.nn as nn
import torch.nn.functional as F

from .planning.lstm import LSTM
from .utils import load_act_fn
from .memory import ValueMemory, KeyValueMemory
from .basic_module import BasicModule


class ValueMemoryLSTM(BasicModule):
    def __init__(self, memory_module: ValueMemory, input_dim: int, hidden_dim: int, decision_dim: int, output_dim: int, act_fn="ReLU", em_gate_act_fn="Sigmoid", device: str = 'cpu'):
        super().__init__()
        self.device = device

        self.lstm = LSTM(input_dim, hidden_dim, device)
        self.memory_module = memory_module

        self.decision_dim = decision_dim

        self.fc_decision = nn.Linear(hidden_dim, decision_dim)
        self.fc_actor = nn.Linear(decision_dim, output_dim)
        self.fc_critic = nn.Linear(decision_dim, 1)
        self.fc_em_gate_value = nn.Linear(hidden_dim + decision_dim, 1)

        self.act_fn = load_act_fn(act_fn)
        self.em_gate_act_fn = load_act_fn(em_gate_act_fn)

    def init_state(self, batch_size):
        return self.lstm.init_state(batch_size)

    def forward(self, inp, state, beta=1.0):
        h, c, z = self.lstm(inp, state)
        o = z[2]
        dec_act = self.act_fn(self.fc_decision(h))
        em_gate = self.em_gate_act_fn(self.fc_em_gate_value(torch.cat((c, dec_act), 1)))
        memory = self.memory_module.retrieve(c, em_gate)
        c2 = c + memory
        self.memory_module.encode(c2)
        h2 = torch.mul(o, c2.tanh())
        dec_act2 = self.act_fn(self.fc_decision(h2))
        pi_a = _softmax(self.fc_actor(dec_act2), beta)
        value = self.fc_critic(dec_act2)

        # record
        self.write(em_gate, 'em_gate')
        self.write(memory, 'memory')
        self.write(c2, 'c')
        self.write(h, 'h')
        self.write(dec_act2, 'dec_act')
        self.write(pi_a, 'pi_a')
        self.write(value, 'value')

        return pi_a, value, (h2, c2)

    def set_encoding(self, status):
        self.memory_module.encoding = status
    
    def set_retrieval(self, status):
        self.memory_module.retrieving = status


class KeyValueLSTM(BasicModule):
    def __init__(self, memory_module: KeyValueMemory, input_dim: int, output_dim: int, hidden_dim: int = 256, decision_dim: int = 128, 
        context_embedding_dim: int = 16, act_fn="ReLU", em_gate_act_fn="Sigmoid", device: str = 'cpu'):
        super().__init__()
        self.device = device

        self.lstm = LSTM(context_embedding_dim, hidden_dim, device)
        self.memory_module = memory_module

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
        context_embedding = self.fc_in(inp)
        h, c, z = self.lstm(context_embedding, state)
        o = z[2]
        # decision = self.act_fn(self.fc_decision(h))
        em_gate = self.em_gate_act_fn(self.fc_em_gate_in(context_embedding) + self.fc_em_gate_rec(h))
        memory = self.memory_module.retrieve(context_embedding, em_gate)
        c2 = c + memory
        self.memory_module.encode((context_embedding, c2))
        h2 = torch.mul(o, c2.tanh())
        decision2 = self.act_fn(self.fc_decision(h2))
        pi_a = _softmax(self.fc_actor(decision2), beta)
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


def _softmax(z, beta):
    """helper function, softmax with beta

    Parameters
    ----------
    z : torch tensor, has 1d underlying structure after torch.squeeze
        the raw logits
    beta : float, >0
        softmax temp, big value -> more "randomness"

    Returns
    -------
    1d torch tensor
        a probability distribution | beta

    """
    assert beta > 0
    # softmax the input to a valid PMF
    pi_a = F.softmax(torch.squeeze(z / beta), dim=0)
    # make sure the output is valid
    if torch.any(torch.isnan(pi_a)):
        raise ValueError(f'Softmax produced nan: {z} -> {pi_a}')
    return pi_a
