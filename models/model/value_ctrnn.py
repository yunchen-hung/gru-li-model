from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import load_act_fn, softmax
from ..basic_module import BasicModule
from ..planning import CTRNN
from ..memory import ValueMemory


class ValueMemoryCTRNN(BasicModule):
    def __init__(self, planning_module: CTRNN, memory_module: ValueMemory, hidden_dim: int, decision_dim: int, output_dim: int, em_gate_type='vector', act_fn='Tanh', 
    em_gate_act_fn='Sigmoid', use_memory=True, decode_memory=False, encode_memory=False, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.use_memory = use_memory

        self.planning_module = planning_module
        self.memory_module = memory_module

        self.fc_decision = nn.Linear(hidden_dim, decision_dim)
        self.fc_actor = nn.Linear(decision_dim, output_dim)
        self.fc_critic = nn.Linear(decision_dim, 1)

        self.em_gate_type = em_gate_type
        if em_gate_type == "scalar":
            self.fc_em_gate = nn.Linear(hidden_dim + decision_dim, 1)
        elif em_gate_type == "vector":
            self.fc_em_gate = nn.Linear(hidden_dim + decision_dim, hidden_dim)
        else:
            raise ValueError(f"Invalid em_gate_type: {em_gate_type}")

        self.decode_memory = decode_memory
        if decode_memory:
            self.fc_memory_decoder = nn.Linear(hidden_dim, hidden_dim)

        self.encode_memory = encode_memory
        if encode_memory:
            self.fc_memory_encoder = nn.Linear(hidden_dim, hidden_dim)

        self.act_fn = load_act_fn(act_fn)
        self.em_gate_act_fn = load_act_fn(em_gate_act_fn)

    def init_state(self, batch_size, recall=False, flush_level=1.0, prev_state=None):
        return self.planning_module.init_state(batch_size, recall, flush_level, prev_state)

    def compute_memory(self, h, dec_act):
        if self.em_gate_type == "scalar":
            em_gate = self.em_gate_act_fn(self.fc_em_gate(torch.cat((h, dec_act), 1)))
            self.write(mem_gate, 'em_gate')
        elif self.em_gate_type == "vector":
            em_gate = 1.0
        else:
            raise ValueError(f"Invalid em_gate_type: {self.em_gate_type}")
        memory = self.memory_module.retrieve(h, em_gate)
        if self.em_gate_type == "vector":
            mem_gate = self.em_gate_act_fn(self.fc_em_gate(torch.cat((h, dec_act), 1)))
            self.write(memory, 'raw_memory')
            self.write(mem_gate, 'em_gate')
            memory = torch.mul(mem_gate, memory)
        return memory

    def forward(self, inp, state, beta=1.0):
        h = self.planning_module(inp, state)
        dec_act = self.act_fn(self.fc_decision(h))
        if self.use_memory:
            memory = self.compute_memory(h, dec_act)
            if self.decode_memory:
                memory = self.fc_memory_decoder(memory)
            self.write(memory, 'memory')
            h = h + memory
            dec_act = self.act_fn(self.fc_decision(h))
            if self.encode_memory:
                memory_to_store = self.fc_memory_encoder(memory)
                self.write(memory_to_store, "encoded_memory")
            else:
                memory_to_store = h
            self.memory_module.encode(memory_to_store)
            
        pi_a = softmax(self.fc_actor(dec_act), beta)
        value = self.fc_critic(dec_act)

        self.write(dec_act, 'dec_act')
        self.write(h, 'h')
        self.write(pi_a, 'pi_a')
        self.write(value, 'value')

        return pi_a, value, h

    def set_encoding(self, status):
        self.memory_module.encoding = status
    
    def set_retrieval(self, status):
        self.memory_module.retrieving = status
