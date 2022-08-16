import torch
import torch.nn as nn

from ..utils import load_act_fn, softmax
from ..basic_module import BasicModule
from ..planning.lstm import LSTM
from ..memory import ValueMemory


class ValueMemoryLSTM(BasicModule):
    def __init__(self, memory_module: ValueMemory, input_dim: int, hidden_dim: int, decision_dim: int, output_dim: int, em_gate_type='scalar', act_fn='ReLU', 
    em_gate_act_fn='Sigmoid', use_memory=True, init_state_type='train', use_mem_gate=False, device: str = 'cpu'):
        """
        init_state_type: see LSTM
        em_gate_type: 'scalar' or 'vector'
        """
        super().__init__()
        self.device = device
        self.use_memory = use_memory
        self.use_mem_gate = use_mem_gate

        self.lstm = LSTM(input_dim, hidden_dim, init_state_type, device)
        self.memory_module = memory_module

        self.decision_dim = decision_dim

        self.fc_decision = nn.Linear(hidden_dim, decision_dim)
        self.fc_actor = nn.Linear(decision_dim, output_dim)
        self.fc_critic = nn.Linear(decision_dim, 1)
        if em_gate_type == 'scalar':
            self.fc_em_gate_value = nn.Linear(hidden_dim + decision_dim, 1)
        else:
            self.fc_em_gate_value = nn.Linear(hidden_dim + decision_dim, memory_module.capacity)
        if self.use_mem_gate:
            self.fc_mem_gate = nn.Linear(hidden_dim + decision_dim, hidden_dim)
            # self.mem_gate = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)

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
            if self.use_mem_gate:
                mem_gate = self.fc_mem_gate(torch.cat((c, dec_act), 1)).sigmoid()
                self.write(memory, 'raw_memory')
                self.write(mem_gate, 'mem_gate')
                memory = torch.mul(mem_gate, memory)
            c = c + memory
            self.memory_module.encode(c)
            h = torch.mul(o, c.tanh())
            dec_act = self.act_fn(self.fc_decision(h))
            
            self.write(em_gate, 'em_gate')
            self.write(memory, 'memory')
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


class SimpleValueMemoryLSTM(BasicModule):
    def __init__(self, memory_module: ValueMemory, input_dim: int, hidden_dim: int, output_dim: int, em_gate_type='scalar', act_fn='ReLU', 
    em_gate_act_fn='Sigmoid', use_memory=True, init_state_type='train', use_mem_gate=False, device: str = 'cpu'):
        """
        no decision layer
        init_state_type: see LSTM
        em_gate_type: 'scalar' or 'vector'
        """
        super().__init__()
        self.device = device
        self.use_memory = use_memory
        self.use_mem_gate = use_mem_gate

        self.lstm = LSTM(input_dim, hidden_dim, init_state_type, device)
        self.memory_module = memory_module

        self.fc_actor = nn.Linear(hidden_dim, output_dim)
        self.fc_critic = nn.Linear(hidden_dim, 1)
        if em_gate_type == 'scalar':
            self.fc_em_gate_value = nn.Linear(hidden_dim * 2, 1)
        else:
            self.fc_em_gate_value = nn.Linear(hidden_dim * 2, memory_module.capacity)
        if self.use_mem_gate:
            self.fc_mem_gate = nn.Linear(hidden_dim * 2, hidden_dim)
            # self.mem_gate = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)

        self.act_fn = load_act_fn(act_fn)
        self.em_gate_act_fn = load_act_fn(em_gate_act_fn)

    def init_state(self, batch_size, recall=False, flush_level=1.0, prev_state=None):
        return self.lstm.init_state(batch_size, recall, flush_level, prev_state)

    def forward(self, inp, state, beta=1.0):
        h, c, z = self.lstm(inp, state)
        o = z[2]
        if self.use_memory:
            em_gate = self.em_gate_act_fn(self.fc_em_gate_value(torch.cat((c, h), 1)))
            memory = self.memory_module.retrieve(c, em_gate)
            if self.use_mem_gate:
                mem_gate = self.fc_mem_gate(torch.cat((c, h), 1)).sigmoid()
                self.write(memory, 'raw_memory')
                self.write(mem_gate, 'mem_gate')
                memory = torch.mul(mem_gate, memory)
            c = c + memory
            self.memory_module.encode(c)
            h = torch.mul(o, c.tanh())
            
            self.write(em_gate, 'em_gate')
            self.write(memory, 'memory')
        pi_a = softmax(self.fc_actor(h), beta)
        value = self.fc_critic(h)

        # record
        self.write(c, 'c')
        self.write(h, 'h')
        self.write(pi_a, 'pi_a')
        self.write(value, 'value')

        return pi_a, value, (h, c)

    def set_encoding(self, status):
        self.memory_module.encoding = status
    
    def set_retrieval(self, status):
        self.memory_module.retrieving = status


class ValueEncodeLSTM(BasicModule):
    def __init__(self, memory_module: ValueMemory, input_dim: int, hidden_dim: int, decision_dim: int, output_dim: int, mem_dim: int = 0, act_fn="ReLU", em_gate_act_fn="Sigmoid", 
    use_memory=True, init_state_type='train', encode_mem=True, decode_mem=True, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.use_memory = use_memory
        self.encode_mem = encode_mem
        self.decode_mem = decode_mem

        self.lstm = LSTM(input_dim, hidden_dim, init_state_type, device)
        self.memory_module = memory_module

        self.decision_dim = decision_dim

        self.fc_decision = nn.Linear(hidden_dim, decision_dim)
        self.fc_actor = nn.Linear(decision_dim, output_dim)
        self.fc_critic = nn.Linear(decision_dim, 1)
        self.fc_em_gate_value = nn.Linear(hidden_dim + decision_dim, 1)
        
        if self.encode_mem:
            self.fc_mem_encode = nn.Linear(hidden_dim, hidden_dim)
        if self.decode_mem:
            self.fc_mem_decode = nn.Linear(hidden_dim, hidden_dim)

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
            if self.decode_mem:
                self.write(memory, 'raw_memory')
                memory = self.fc_mem_decode(memory)
                self.write(memory, 'memory')
            else:
                self.write(memory, 'memory')
            c = c + memory
            if self.encode_mem:
                memory_to_store = self.fc_mem_encode(c)
                self.write(memory_to_store, "encoded_memory")
            else:
                memory_to_store = c
            self.memory_module.encode(memory_to_store)
            h = torch.mul(o, c.tanh())
            dec_act = self.act_fn(self.fc_decision(h))
            
            self.write(em_gate, 'em_gate')
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
