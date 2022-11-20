from turtle import forward
import torch
import torch.nn as nn

from ..utils import load_act_fn, softmax
from ..basic_module import BasicModule
from ..memory import ValueMemory


class ValueMemoryCTRNN(BasicModule):
    def __init__(self, memory_module: ValueMemory, hidden_dim: int, input_dim: int, output_dim: int, em_gate_type='constant', act_fn='Tanh', 
    em_gate_act_fn='Sigmoid', init_state_type="zeros", evolve_state_between_phases=False, dt: float = 10, tau: float = 10, noise_std=0,
    start_recall_with_ith_item_init=0, softmax_beta=1.0, use_memory=True, input_gate=1.0, two_decisions=False, step_for_each_timestep=None, 
    device: str = 'cpu'):
        super().__init__()
        self.device = device

        self.memory_module = memory_module
        self.use_memory = use_memory
        self.encoding = False
        self.retrieving = False

        self.dt = dt
        self.alpha = float(dt) / float(tau)
        self.step_for_each_timestep = step_for_each_timestep if step_for_each_timestep is not None else int(tau / dt)
        self.noise_std = noise_std
        self.softmax_beta = softmax_beta
        self.input_gate = input_gate

        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc_decision = nn.Linear(hidden_dim, output_dim)
        self.fc_critic = nn.Linear(output_dim, 1)
        if two_decisions:
            self.fc_decision2 = nn.Linear(hidden_dim, output_dim)
        self.two_decisions = two_decisions

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
        self.em_gate_act_fn = load_act_fn(em_gate_act_fn)

        self.evolve_state_between_phases = evolve_state_between_phases

        self.hidden_state = torch.zeros((1, self.hidden_dim), device=self.device, requires_grad=True)
        self.start_recall_with_ith_item_init = start_recall_with_ith_item_init
        self.ith_item_state = torch.zeros((1, self.hidden_dim), device=self.device, requires_grad=True)

        if self.hidden_dim == self.input_dim:
            self.W_in = (torch.eye(self.hidden_dim, device=self.device, requires_grad=True)).repeat(1, 1, 1)
        else:
            self.W_in = (torch.cat((torch.eye(self.input_dim, device=self.device, requires_grad=True), \
                        torch.zeros((self.hidden_dim-self.input_dim, self.input_dim), device=self.device, requires_grad=True)), 0)).repeat(1, 1, 1)

        self.init_state_type = init_state_type
        if init_state_type == "train":
            self.h0 = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)
        if self.init_state_type == 'train_diff':
            self.h0 = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)
            self.h0_recall = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)

    def init_state(self, batch_size, recall=False, flush_level=1.0, prev_state=None):
        if recall:
            if self.start_recall_with_ith_item_init:
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
            self.write(state, 'state')
        else:
            if self.init_state_type == "zeros":
                self.hidden_state = torch.zeros((batch_size, self.hidden_dim), device=self.device, requires_grad=True)
                state = torch.zeros((batch_size, self.hidden_dim), device=self.device, requires_grad=True)
            elif self.init_state_type == "train" or self.init_state_type == "train_diff":
                self.hidden_state = self.h0.repeat(batch_size, 1)
                state = self.act_fn(self.hidden_state)
            else:
                raise AttributeError("Invalid init_state_type, should be zeros, train or train_diff")
        
        if self.hidden_dim == self.input_dim:
            self.W_in = (torch.eye(self.hidden_dim, device=self.device, requires_grad=True)).repeat(batch_size, 1, 1) * self.input_gate
        else:
            self.W_in = torch.cat((torch.eye(self.input_dim, device=self.device, requires_grad=True), torch.zeros((self.hidden_dim-self.input_dim, 
                self.input_dim), device=self.device, requires_grad=True)), 0).repeat(batch_size, 1, 1) * self.input_gate
        
        self.write(state, 'init_state')
        self.current_timestep = 0
        return state

    def forward(self, inp, state, beta=1.0):
        if self.encoding:
            c_in = torch.bmm(self.W_in, torch.unsqueeze(inp, dim=2)).squeeze(2)
            c_in = c_in / torch.norm(c_in, p=2, dim=1).reshape(-1, 1)

            for _ in range(self.step_for_each_timestep):
                if self.noise_std > 0:
                    noise = self.noise_std*torch.randn(self.hidden_state.size()).to(self.device)
                else:
                    noise = 0
                self.hidden_state = self.hidden_state * (1 - self.alpha) + (self.fc_hidden(state) + c_in) * self.alpha + noise
                self.write(self.fc_hidden(state), "half_state")
                state = self.act_fn(self.hidden_state)
                self.write(state, 'state')

            if self.use_memory:
                self.memory_module.encode(state)

            decision = softmax(self.fc_decision(state))
            self.write(decision, 'decision')
            if self.two_decisions:
                decision2 = softmax(self.fc_decision2(state))
            value = self.fc_critic(decision)
            decision = (decision, decision2) if self.two_decisions else decision

            self.current_timestep += 1
            if self.current_timestep == self.start_recall_with_ith_item_init:
                self.ith_item_state = self.hidden_state.detach().clone()
            self.last_encoding = True

        elif self.retrieving:
            if self.last_encoding and self.evolve_state_between_phases:
                if self.noise_std > 0:
                    noise = self.noise_std*torch.randn(self.hidden_state.size()).to(self.device)
                else:
                    noise = 0
                self.hidden_state = self.hidden_state * (1 - self.alpha) + (self.fc_hidden(state)) * self.alpha + noise
                state = self.act_fn(self.hidden_state)
                self.write(state, 'state')
            self.last_encoding = False

            if self.em_gate_type == "constant":
                mem_gate = self.em_gate
            else:
                mem_gate = self.em_gate(state).sigmoid()

            c_in = self.memory_module.retrieve(state)

            for _ in range(self.step_for_each_timestep):
                if self.noise_std > 0:
                    noise = self.noise_std*torch.randn(self.hidden_state.size()).to(self.device)
                else:
                    noise = 0
                self.hidden_state = self.hidden_state * (1 - self.alpha) + (self.fc_hidden(state) + c_in * mem_gate) * self.alpha + noise
                self.write(self.fc_hidden(state), "half_state")
                state = self.act_fn(self.hidden_state)
                self.write(state, 'state')
                
            decision = softmax(self.fc_decision(state), beta=self.softmax_beta)
            self.write(decision, 'decision')
            if self.two_decisions:
                decision2 = softmax(self.fc_decision2(state))
            value = self.fc_critic(decision)
            decision = (decision, decision2) if self.two_decisions else decision
            
            self.write(mem_gate, 'mem_gate_recall')

        # print(decision)

        return decision, value, state

    def set_encoding(self, status):
        self.encoding = status
        self.memory_module.encoding = status
    
    def set_retrieval(self, status):
        self.retrieving = status
        self.memory_module.retrieving = status

    def reset_memory(self):
        self.memory_module.reset_memory()
