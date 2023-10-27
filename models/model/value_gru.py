import math
import torch
import torch.nn as nn

from ..utils import load_act_fn, softmax
from ..base_module import BasicModule
from ..memory import ValueMemory


class ValueMemoryGRU(BasicModule):
    def __init__(self, memory_module: ValueMemory, hidden_dim: int, input_dim: int, output_dim: int, em_gate_type='constant',
    init_state_type="zeros", evolve_state_between_phases=False, noise_std=0, softmax_beta=1.0, use_memory=True,
    start_recall_with_ith_item_init=0, reset_param=True, device: str = 'cpu'):
        super().__init__()
        self.device = device

        self.memory_module = memory_module      # memory module of the model, pre-instantiated
        self.use_memory = use_memory            # if false, do not use memory module in the forward pass

        # encoding and retrieval status, if true, do memory encoding or/and retrieval in the forward pass
        self.encoding = False
        self.retrieving = False

        self.noise_std = noise_std
        self.softmax_beta = softmax_beta        # 1/temperature for softmax function for computing final output decision
        try:
            # self.mem_beta = self.memory_module.similarity_measure.softmax_temperature   # TODO: make it more flexible with other kinds of memory
            self.mem_beta = torch.nn.Parameter(torch.tensor(self.memory_module.similarity_measure.softmax_temperature), requires_grad=False)
        except:
            self.mem_beta = None

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc_input = nn.Linear(input_dim, 3 * hidden_dim)
        # self.fc_memory = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.fc_decision = nn.Linear(hidden_dim, output_dim)
        self.fc_critic = nn.Linear(hidden_dim, 1)

        # gate when adding episodic memory to hidden state
        self.em_gate_type = em_gate_type
        if em_gate_type == "constant":
            self.em_gate = 1.0
        elif em_gate_type == "scalar" or em_gate_type == "scalar_sigmoid":
            self.em_gate = nn.Linear(hidden_dim, 1)
        elif em_gate_type == "vector":
            self.em_gate = nn.Linear(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Invalid em_gate_type: {em_gate_type}")

        # if true, compute forward pass for an extra timestep between encoding and retrieval phases
        self.evolve_state_between_phases = evolve_state_between_phases
        self.last_encoding = False

        self.start_recall_with_ith_item_init = start_recall_with_ith_item_init
        self.ith_item_state = torch.zeros((1, self.hidden_dim), device=self.device, requires_grad=True)
        self.current_timestep = 0

        self.init_state_type = init_state_type
        if init_state_type == "train":
            # train initial hidden state
            self.h0 = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)
        if self.init_state_type == 'train_diff':
            # train different initial hidden state for encoding and recall phase
            self.h0 = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)
            self.h0_recall = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)

        if reset_param:
            self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_dim)
        for w in self.parameters():
            if w.requires_grad:
                w.data.uniform_(-std, std)

    def init_state(self, batch_size, recall=False, flush_level=1.0, prev_state=None):
        if recall:
            # initialize hidden state for recall phase
            if self.start_recall_with_ith_item_init != 0:
                state = self.ith_item_state.clone()
            elif self.init_state_type == 'zeros':
                state = torch.zeros((batch_size, self.hidden_dim), device=self.device, requires_grad=True)
            elif self.init_state_type == 'train':
                state = self.h0.repeat(batch_size, 1)
            elif self.init_state_type == 'train_diff':
                state = self.h0_recall.repeat(batch_size, 1)
            else:
                raise AttributeError("Invalid init_state_type, should be zeros, train or train_diff")
            state = torch.tanh(state)
        else:
            # initialize hidden state for encoding phase
            if self.init_state_type == "zeros":
                state = torch.zeros((batch_size, self.hidden_dim), device=self.device, requires_grad=True)
            elif self.init_state_type == "train" or self.init_state_type == "train_diff":
                state = torch.tanh(self.h0.repeat(batch_size, 1))
            else:
                raise AttributeError("Invalid init_state_type, should be zeros, train or train_diff")
        
        self.write(state, 'init_state')
        return state

    def forward(self, inp, state, beta=None, mem_beta=None):
        if self.last_encoding and self.evolve_state_between_phases and self.retrieving:
            # do a timestep of forward pass between encoding and retrieval phases
            gate_h = self.fc_hidden(state)
            h_r, h_i, h_n = gate_h.chunk(3, 1)
            resetgate = torch.sigmoid(h_r)
            inputgate = torch.sigmoid(h_i)
            newgate = torch.tanh(resetgate * h_n)
            state = newgate + inputgate * (state - newgate)
            self.last_encoding = False
            self.write(state, 'state')

        # retrieve memory
        if self.use_memory and self.retrieving:
            mem_beta = self.mem_beta if mem_beta is None else mem_beta
            retrieved_memory = self.memory_module.retrieve(state, beta=mem_beta)
            if self.em_gate_type == "constant":
                mem_gate = self.em_gate
            elif self.em_gate_type == "scalar_sigmoid" or self.em_gate_type == "vector":
                mem_gate = self.em_gate(state).sigmoid()
            elif self.em_gate_type == "scalar":
                mem_gate = self.em_gate(state)
            else:
                raise ValueError(f"Invalid em_gate_type: {self.em_gate_type}")
            self.write(mem_gate, 'mem_gate_recall')
        else:
            retrieved_memory = torch.zeros(1, self.hidden_dim)
            mem_gate = 0.0

        # compute forward pass
        state = state
        gate_x = self.fc_input(inp)
        gate_h = self.fc_hidden(state)
        # gate_m = self.fc_memory(retrieved_memory)
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        # m_r, m_i, m_n = gate_m.chunk(3, 1)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        # resetgate = torch.sigmoid(i_r + h_r + m_r)
        # inputgate = torch.sigmoid(i_i + h_i + m_i)
        newgate = torch.tanh(i_n + resetgate * h_n + mem_gate * retrieved_memory)
        # newgate = torch.tanh(i_n + resetgate * h_n + m_n)
        state = newgate + inputgate * (state - newgate)
        self.write(state, 'state')

        # store memory
        if self.use_memory and self.encoding:
            self.memory_module.encode(state)
            self.last_encoding = True
            self.current_timestep += 1
            if self.current_timestep == self.start_recall_with_ith_item_init:
                self.ith_item_state = state.detach().clone()

        # compute output decision(s)
        beta = self.softmax_beta if beta is None else beta
        decision = softmax(self.fc_decision(state), beta)
        self.write(decision, 'decision')
        value = self.fc_critic(state)

        self.write(self.use_memory, 'use_memory')
        
        return decision, value, state

    def set_encoding(self, status):
        """
        set memory encoding status of the model
        """
        self.encoding = status
        self.memory_module.encoding = status
    
    def set_retrieval(self, status):
        """
        set memory retrieval status of the model
        """
        self.retrieving = status
        self.memory_module.retrieving = status

    def reset_memory(self):
        """
        reset memory of the memory module of the model
        """
        self.memory_module.reset_memory()
