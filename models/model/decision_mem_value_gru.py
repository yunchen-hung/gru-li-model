
import math
import torch
import torch.nn as nn

from ..utils import load_act_fn, softmax
from ..base_module import BasicModule
from ..memory import ValueMemory
from ..module.encoders import MLPEncoder
from ..module.decoders import ActorCriticMLPDecoder


class DecisionMemValueGRU(BasicModule):
    """
    Deep value memory GRU with a separate GRU for decision making
    Only the decision making module receives question input, and it generates the gates for encoding and retrieving memories
    """
    def __init__(self, 
                 memory_module: ValueMemory, 
                 hidden_dim: int,                   # same for the memory module GRU and decision making GRU
                 mem_input_dim: int,                # split the input into memory input and decision input, memory input at the front
                 decision_input_dim: int,
                 mem_output_dims: list, 
                 decision_output_dims: list,        # all outputs are returned in a list, decision outputs are in front of mem outputs
                 em_gate_type='constant',
                 init_state_type="zeros", 
                 evolve_state_between_phases=False, 
                 evolve_steps=1, 
                 softmax_beta=1.0, 
                 wm_noise_prop=0,                   # all the noise are added to the memory module
                 em_noise_prop=0, 
                 wm_enc_noise_prop=0,
                 wm_em_zero_noise=False,            # if true, make the noise for wm/em to be all zero (only cancel the original data but do not add noise)
                 start_recall_with_ith_item_init=0, 
                 reset_param=True, 
                 step_for_each_timestep=1, 
                 flush_noise=0.1, 
                 random_init_noise=0.1, 
                 use_memory=True, 
                 decision_to_mem_feedback=False,    # if true, feedback decision hidden state to memory module, otherwise only have connection from memory to decision
                 mem_beta_decay=False,              # whether to decay softmax beta for computing memory similarity loss
                 mem_beta_decay_rate=0.5,           # decay rate for softmax beta
                 device: str = 'cpu'):
        super().__init__()
        self.device = device

        self.memory_module = memory_module      # memory module of the model, pre-instantiated

        self.use_memory = use_memory            # if false, do not use memory module in the forward pass

        self.step_for_each_timestep = step_for_each_timestep

        # encoding and retrieval status, if true, do memory encoding or/and retrieval in the forward pass
        self.encoding = False
        self.retrieving = False

        self.wm_noise_prop = wm_noise_prop      # noise proportion for working memory
        self.em_noise_prop = em_noise_prop      # noise proportion for episodic memory
        self.wm_enc_noise_prop = wm_enc_noise_prop   # noise proportion for working memory only during encoding phase
        self.wm_em_zero_noise = 1 - float(wm_em_zero_noise)      # if true, make the noise for wm/em to be all zero (only cancel the original data but do not add noise)

        self.softmax_beta = softmax_beta        # 1/temperature for softmax function for computing final output decision
        if mem_beta_decay:
            self.mem_beta = self.memory_module.similarity_measure.softmax_temperature
            print("mem_beta initialized to {}".format(self.mem_beta))
        else:
            self.mem_beta = None
        self.mem_beta_decay = mem_beta_decay
        self.mem_beta_decay_rate = mem_beta_decay_rate
        
        self.flush_noise = flush_noise
        self.random_init_noise = random_init_noise

        self.hidden_dim = hidden_dim
        self.mem_input_dim = mem_input_dim
        self.decision_input_dim = decision_input_dim

        if isinstance(mem_output_dims, int):
            mem_output_dims = [mem_output_dims]
        self.mem_output_dims = mem_output_dims
        self.mem_outputs_num = len(mem_output_dims)
        if isinstance(decision_output_dims, int):
            decision_output_dims = [decision_output_dims]
        self.decision_output_dims = decision_output_dims
        self.decision_outputs_num = len(decision_output_dims)

        fc_hidden_dim = int(hidden_dim/2)

        self.mem_encoder = MLPEncoder(mem_input_dim, 3 * hidden_dim, hidden_dims=[])
        if decision_input_dim != 0:
            self.decision_encoder = MLPEncoder(decision_input_dim, 3 * hidden_dim, hidden_dims=[])
        else:
            self.decision_encoder = None

        self.fc_mem_hidden = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.fc_decision_hidden = nn.Linear(hidden_dim, 3 * hidden_dim)

        self.decoders = nn.ModuleList()
        for output_dim in decision_output_dims:
            self.decoders.append(ActorCriticMLPDecoder(hidden_dim, output_dim, hidden_dims=[]))
        for output_dim in mem_output_dims:
            self.decoders.append(ActorCriticMLPDecoder(hidden_dim, output_dim, hidden_dims=[]))

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
        self.evolve_steps = evolve_steps
        self.last_encoding = False

        self.start_recall_with_ith_item_init = start_recall_with_ith_item_init
        self.ith_item_state = torch.zeros((1, self.hidden_dim), device=self.device, requires_grad=True)
        self.current_timestep = 0

        self.init_state_type = init_state_type

        if reset_param:
            self.reset_parameters()
        else:
            self.reset_parameters2()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_dim)
        for w in self.parameters():
            if w.requires_grad:
                w.data.uniform_(-std, std)

    def reset_parameters2(self):
        std = 1.0 / math.sqrt(self.hidden_dim)
        for w in self.parameters():
            if w.requires_grad:
                w.data.normal_(0.0, std)

    def init_state(self, batch_size, recall=False, flush_level=1.0, prev_state=None, decay_mem_beta=False):
        if recall:
            # initialize hidden state for recall phase
            if self.start_recall_with_ith_item_init != 0:
                mem_state = self.ith_item_state.clone()
                # decision_state = self.ith_item_state.clone()
                decision_state = torch.zeros((batch_size, self.hidden_dim), device=self.device, requires_grad=True)
            elif self.init_state_type == 'zeros':
                mem_state = torch.zeros((batch_size, self.hidden_dim), device=self.device, requires_grad=True)
                decision_state = torch.zeros((batch_size, self.hidden_dim), device=self.device, requires_grad=True)
            elif self.init_state_type == 'noise' or self.init_state_type == 'noise_all':
                prev_mem_state, prev_decision_state = prev_state
                mem_state = (1 - self.flush_noise) * prev_mem_state + self.flush_noise * torch.randn_like(prev_mem_state) * torch.std(prev_mem_state)
                decision_state = (1 - self.flush_noise) * prev_decision_state + self.flush_noise * torch.randn_like(prev_decision_state) * torch.std(prev_decision_state)
            elif self.init_state_type == 'random':
                mem_state = torch.randn((batch_size, self.hidden_dim), device=self.device, requires_grad=True) * self.random_init_noise
                decision_state = torch.randn((batch_size, self.hidden_dim), device=self.device, requires_grad=True) * self.random_init_noise
            else:
                raise AttributeError("Invalid init_state_type, should be zeros, noise, noise_all or random")
            mem_state = torch.tanh(mem_state)
            decision_state = torch.tanh(decision_state)
        else:
            # initialize hidden state for encoding phase
            if self.init_state_type == "zeros" or self.init_state_type == "noise":
                mem_state = torch.zeros((batch_size, self.hidden_dim), device=self.device, requires_grad=True)
                decision_state = torch.zeros((batch_size, self.hidden_dim), device=self.device, requires_grad=True)
            elif self.init_state_type == "random" or self.init_state_type == "noise_all":
                mem_state = torch.randn((batch_size, self.hidden_dim), device=self.device, requires_grad=True) * self.random_init_noise
                decision_state = torch.randn((batch_size, self.hidden_dim), device=self.device, requires_grad=True) * self.random_init_noise
            else:
                raise AttributeError("Invalid init_state_type, should be zeros, train or train_diff")
        
        if self.mem_beta_decay and decay_mem_beta:
            self.mem_beta = self.mem_beta * self.mem_beta_decay_rate
            print("mem_beta decayed to {}".format(self.mem_beta))
        
        self.write(mem_state, 'init_mem_state')
        self.write(decision_state, 'init_decision_state')
        return (mem_state, decision_state)

    def forward(self, inp, state, beta=None):
        mem_state, decision_state = state

        batch_size = inp.shape[0]

        if self.decision_input_dim == 0:
            inp_mem = inp
            inp_decision = None
        else:
            inp_mem, inp_decision = inp[:, :self.mem_input_dim], inp[:, self.mem_input_dim:]

        if self.last_encoding and self.evolve_state_between_phases and self.retrieving:
            for _ in range(self.evolve_steps):
                inp0 = torch.zeros_like(inp_mem)
                # do a timestep of forward pass between encoding and retrieval phases
                mem_state = self.mem_gru(inp0, mem_state)
                self.last_encoding = False
                self.write(mem_state, 'mem_state')

        # retrieve memory
        if self.use_memory and self.retrieving:
            # mem_beta = self.mem_beta if mem_beta is None else mem_beta
            retrieved_memory, memory_similarity = self.memory_module.retrieve(mem_state, beta=self.mem_beta)
            if self.em_gate_type == "constant":
                mem_gate = self.em_gate
            elif self.em_gate_type == "scalar_sigmoid" or self.em_gate_type == "vector":
                mem_gate = self.em_gate(mem_state).sigmoid()
            elif self.em_gate_type == "scalar":
                mem_gate = self.em_gate(mem_state)
            else:
                raise ValueError(f"Invalid em_gate_type: {self.em_gate_type}")
            self.write(mem_gate, 'mem_gate_recall')
            self.write(memory_similarity, 'memory_similarity')
            self.write(retrieved_memory, 'retrieved_memory')
        else:
            retrieved_memory = torch.zeros(batch_size, self.hidden_dim)
            mem_gate = 0.0
            memory_similarity = torch.zeros(batch_size, self.memory_module.capacity)

        if self.use_memory and self.encoding:
            # add working memory noise to the memory state
            mem_state = (1 - self.wm_enc_noise_prop) * mem_state + \
                self.wm_enc_noise_prop * torch.randn_like(mem_state) * torch.std(mem_state) * self.wm_em_zero_noise

        # compute forward pass
        for i in range(self.step_for_each_timestep):
            mem_state = self.mem_gru(inp_mem, mem_state, mem_gate, retrieved_memory)
            decision_state = self.decision_gru(inp_decision, decision_state, mem_state)
        
        # store memory
        if self.use_memory and self.encoding:
            self.memory_module.encode(mem_state)
            self.last_encoding = True
            self.current_timestep += 1
            if self.current_timestep == self.start_recall_with_ith_item_init:
                self.ith_item_state = mem_state.detach().clone()

        self.write(mem_state, 'mem_state')
        self.write(decision_state, 'decision_state')

        beta = self.softmax_beta if beta is None else beta
        decisions, values = [], []
        for i in range(self.decision_outputs_num):
            decision, value = self.decoders[i](decision_state)
            decision = softmax(decision, beta)
            decisions.append(decision)
            values.append(value)
            self.write(decision, 'decision{}'.format(i+1))
            self.write(value, 'value{}'.format(i+1))
        for i in range(self.mem_outputs_num):
            decision, value = self.decoders[i+self.decision_outputs_num](mem_state)
            decision = softmax(decision, beta)
            decisions.append(decision)
            values.append(value)
            self.write(decision, 'decision{}'.format(i+1+self.decision_outputs_num))
            self.write(value, 'value{}'.format(i+1+self.decision_outputs_num))

        self.write(self.use_memory, 'use_memory')
        info = {
            "memory_similarity": memory_similarity
        }
        
        return decisions, values, state, info
    
    def mem_gru(self, inp, state, mem_gate=None, retrieved_memory=None, decision_state=None):
        gate_x = self.mem_encoder(inp)
        gate_h = self.fc_mem_hidden(state)

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)

        newgate_preact = i_n + resetgate * h_n
        if mem_gate is not None and retrieved_memory is not None:
            retrieved_memory = (1 - self.em_noise_prop) * retrieved_memory + \
                self.em_noise_prop * torch.randn_like(retrieved_memory) * torch.std(retrieved_memory) * self.wm_em_zero_noise
            newgate_preact += mem_gate * retrieved_memory
        if decision_state is not None:
            newgate_preact += decision_state
        newgate = torch.tanh(newgate_preact)
        state = newgate + inputgate * (state - newgate)
        state = (1 - self.wm_noise_prop) * state + self.wm_noise_prop * torch.randn_like(state) * torch.std(state) * self.wm_em_zero_noise
        return state
    
    def decision_gru(self, inp, state, mem_state):
        if self.decision_input_dim == 0:
            i_r, i_i, i_n = 0.0, 0.0, 0.0
        else:
            gate_x = self.decision_encoder(inp)
            i_r, i_i, i_n = gate_x.chunk(3, 1)
        
        gate_h = self.fc_decision_hidden(state)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)

        newgate_preact = i_n + resetgate * h_n + mem_state
        newgate = torch.tanh(newgate_preact)
        state = newgate + inputgate * (state - newgate)
        return state

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

    def reset_memory(self, flush=True):
        """
        reset memory of the memory module of the model
        """
        self.memory_module.reset_memory(flush=flush)
