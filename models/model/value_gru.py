from turtle import forward
import torch
import torch.nn as nn

from ..utils import load_act_fn, softmax
from ..base_module import BasicModule
from ..memory import ValueMemory


class ValueMemoryGRU(BasicModule):
    def __init__(self, memory_module: ValueMemory, hidden_dim: int, input_dim: int, output_dim: int, em_gate_type='constant', act_fn='Tanh', 
    init_state_type="zeros", evolve_state_between_phases=False, dt: float = 10, tau: float = 10, noise_std=0, start_recall_with_ith_item_init=0, 
    softmax_beta=1.0, use_memory=True, two_decisions=False, step_for_each_timestep=None, device: str = 'cpu'):
        super().__init__()
        self.device = device

        self.memory_module = memory_module      # memory module of the model, pre-instantiated
        self.use_memory = use_memory            # if false, do not use memory module in the forward pass

        # encoding and retrieval status, if true, do memory encoding or/and retrieval in the forward pass
        self.encoding = False
        self.retrieving = False

        # for CTRNN
        self.dt = dt
        self.alpha = float(dt) / float(tau)
        self.step_for_each_timestep = step_for_each_timestep if step_for_each_timestep is not None else int(tau / dt)

        self.noise_std = noise_std
        self.softmax_beta = softmax_beta        # 1/temperature for softmax function for computing final output decision

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc_input = nn.Linear(input_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc_decision = nn.Linear(hidden_dim, output_dim)
        self.fc_critic = nn.Linear(output_dim, 1)
        if two_decisions:           # if true, train to output both current timestep and last timestep
            self.fc_decision2 = nn.Linear(hidden_dim, output_dim)
        self.two_decisions = two_decisions

        # gate when adding episodic memory to hidden state
        self.em_gate_type = em_gate_type
        if em_gate_type == "constant":
            self.em_gate = 1.0
        elif em_gate_type == "scalar":
            self.em_gate = nn.Linear(hidden_dim, 1)
        elif em_gate_type == "vector":
            self.em_gate = nn.Linear(hidden_dim, hidden_dim)
        else:
            raise ValueError(f"Invalid em_gate_type: {em_gate_type}")

        self.act_fn = load_act_fn(act_fn)

        # if true, compute forward pass for an extra timestep between encoding and retrieval phases
        self.evolve_state_between_phases = evolve_state_between_phases
        self.last_encoding = False

        self.hidden_state = torch.zeros((1, self.hidden_dim), device=self.device, requires_grad=True)
        # initialize the hidden state at recall phase with the ith item's hidden state at encoding phase
        # start_recall_with_ith_item_init can take 1~mem_num, 0 means do not initialize hidden state at recall phase
        self.start_recall_with_ith_item_init = start_recall_with_ith_item_init
        self.ith_item_state = torch.zeros((1, self.hidden_dim), device=self.device, requires_grad=True)

        self.init_state_type = init_state_type
        if init_state_type == "train":
            # train initial hidden state
            self.h0 = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)
        if self.init_state_type == 'train_diff':
            # train different initial hidden state for encoding and recall phase
            self.h0 = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)
            self.h0_recall = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)

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
            state = self.act_fn(self.hidden_state)
            if self.start_recall_with_ith_item_init != 0:
                self.write(state, 'state')
        else:
            # initialize hidden state for encoding phase
            if self.init_state_type == "zeros":
                self.hidden_state = torch.zeros((batch_size, self.hidden_dim), device=self.device, requires_grad=True)
                state = torch.zeros((batch_size, self.hidden_dim), device=self.device, requires_grad=True)
            elif self.init_state_type == "train" or self.init_state_type == "train_diff":
                self.hidden_state = self.h0.repeat(batch_size, 1)
                state = self.act_fn(self.hidden_state)
            else:
                raise AttributeError("Invalid init_state_type, should be zeros, train or train_diff")
        
        self.write(state, 'init_state')
        self.current_timestep = 0
        return state

    def forward(self, inp, state, beta=1.0):
        if self.last_encoding and self.evolve_state_between_phases and self.retrieving:
            # do a timestep of forward pass between encoding and retrieval phases
            if self.noise_std > 0:
                noise = self.noise_std*torch.randn(self.hidden_state.size()).to(self.device)
            else:
                noise = 0
            self.hidden_state = self.hidden_state * (1 - self.alpha) + (self.fc_hidden(state)) * self.alpha + noise
            state = self.act_fn(self.hidden_state)
            self.write(state, 'state')
            self.last_encoding = False

        # encode input info
        c_in = self.fc_input(inp)

        # retrieve memory
        if self.use_memory and self.retrieving:
            print("retrieving")
            if self.em_gate_type == "constant":
                mem_gate = self.em_gate
            elif self.em_gate_type == "scalar":
                mem_gate = self.em_gate(state)
            else:
                mem_gate = self.em_gate(state).sigmoid()
            self.write(mem_gate, 'mem_gate_recall')

            # when use_memory is false, this function will return a zero tensor
            retrieved_memory = self.memory_module.retrieve(state)
        else:
            retrieved_memory = 0
            mem_gate = 0

        # compute forward pass
        for _ in range(self.step_for_each_timestep):
            if self.noise_std > 0:
                noise = self.noise_std*torch.randn(self.hidden_state.size()).to(self.device)
            else:
                noise = 0
            self.hidden_state = self.hidden_state * (1 - self.alpha) + (self.fc_hidden(state) + c_in + retrieved_memory * mem_gate) * self.alpha + noise
            state = self.act_fn(self.hidden_state)
            self.write(state, 'state')

        # store memory
        if self.use_memory and self.encoding:
            print("encoding")
            self.memory_module.encode(state)
            self.last_encoding = True       # flag for evolve_state_between_phases, indicating last timestep is in encoding phase
            self.current_timestep += 1
            if self.current_timestep == self.start_recall_with_ith_item_init:
                self.ith_item_state = self.hidden_state.detach().clone()

        # compute output decision(s)
        decision = softmax(self.fc_decision(state))
        self.write(decision, 'decision')
        if self.two_decisions:
            decision2 = softmax(self.fc_decision2(state))
        value = self.fc_critic(decision)
        decision = (decision, decision2) if self.two_decisions else decision
        
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
