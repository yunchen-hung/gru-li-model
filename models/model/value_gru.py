import math
import torch
import torch.nn as nn

from ..utils import load_act_fn, softmax
from ..base_module import BasicModule
from ..memory import ValueMemory


class ValueMemoryGRU(BasicModule):
    def __init__(self, memory_module: ValueMemory, hidden_dim: int, input_dim: int, output_dim: int, em_gate_type='constant',
    init_state_type="zeros", evolve_state_between_phases=False, noise_std=0, start_recall_with_ith_item_init=0, 
    softmax_beta=1.0, use_memory=True, device: str = 'cpu'):
        super().__init__()
        self.device = device

        self.memory_module = memory_module      # memory module of the model, pre-instantiated
        self.use_memory = use_memory            # if false, do not use memory module in the forward pass

        # encoding and retrieval status, if true, do memory encoding or/and retrieval in the forward pass
        self.encoding = False
        self.retrieving = False

        self.noise_std = noise_std
        self.softmax_beta = softmax_beta        # 1/temperature for softmax function for computing final output decision

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc_input = nn.Linear(input_dim, 3 * hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.fc_decision = nn.Linear(hidden_dim, output_dim)
        self.fc_critic = nn.Linear(hidden_dim, 1)

        self.init_state_type = init_state_type
        if init_state_type == "train":
            # train initial hidden state
            self.h0 = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)
        if self.init_state_type == 'train_diff':
            # train different initial hidden state for encoding and recall phase
            self.h0 = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)
            self.h0_recall = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)

    #     self.reset_parameters()

    # def reset_parameters(self):
    #     std = 1.0 / math.sqrt(self.hidden_dim)
    #     for w in self.parameters():
    #         w.data.uniform_(-std, std)

    def init_state(self, batch_size, recall=False, flush_level=1.0, prev_state=None):
        if recall:
            # initialize hidden state for recall phase
            if self.start_recall_with_ith_item_init != 0:
                self.hidden_state = self.ith_item_state.clone()
            elif self.init_state_type == 'zeros':
                self.hidden_state = torch.zeros((batch_size, self.hidden_dim), device=self.device, requires_grad=True)
                state = torch.zeros((batch_size, self.hidden_dim), device=self.device, requires_grad=True)
            elif self.init_state_type == 'train':
                self.hidden_state = self.h0.repeat(batch_size, 1)
            elif self.init_state_type == 'train_diff':
                self.hidden_state = self.h0_recall.repeat(batch_size, 1)
            else:
                raise AttributeError("Invalid init_state_type, should be zeros, train or train_diff")
            state = torch.tanh(self.hidden_state)
            if self.start_recall_with_ith_item_init != 0:
                self.write(state, 'state')
        else:
            # initialize hidden state for encoding phase
            if self.init_state_type == "zeros":
                self.hidden_state = torch.zeros((batch_size, self.hidden_dim), device=self.device, requires_grad=True)
                state = torch.zeros((batch_size, self.hidden_dim), device=self.device, requires_grad=True)
            elif self.init_state_type == "train" or self.init_state_type == "train_diff":
                self.hidden_state = self.h0.repeat(batch_size, 1)
                state = torch.tanh(self.hidden_state)
            else:
                raise AttributeError("Invalid init_state_type, should be zeros, train or train_diff")
        
        self.write(state, 'init_state')
        return state

    def forward(self, inp, state, beta=None):
        # compute forward pass
        gate_x = self.fc_input(inp)
        gate_h = self.fc_hidden(state)
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        state = newgate + inputgate * (state - newgate)
        self.write(state, 'state')

        # compute output decision(s)
        beta = self.softmax_beta if beta is None else beta
        decision = softmax(self.fc_decision(state), beta)
        self.write(decision, 'decision')
        value = self.fc_critic(state)
        
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
