import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ...utils import load_act_fn, softmax
from ...base_module import BasicModule


class TCM(BasicModule):
    def __init__(self, dim: int, lr_cf: float = 1.0, lr_fc: float = 0.9, alpha: float = 0.5, threshold = 0.0, 
    start_recall_with_ith_item_init=0, rand_mem=False, softmax_temperature=1.0, device: str = 'cpu'):
        super().__init__()
        self.device = device

        self.encoding = False
        self.retrieving = False

        self.lr_cf = lr_cf
        self.lr_fc = lr_fc
        self.alpha = alpha
        self.threshold = threshold
        self.dim = dim
        self.rand_mem = rand_mem
        self.softmax_temperature = softmax_temperature

        self.W_cf = torch.zeros((dim, dim), device=device)
        self.W_fc = torch.eye(dim, device=device) * (1 - lr_fc)
        # self.W_cf = torch.zeros((1, dim, dim), device=device, requires_grad=True)
        # self.W_fc = (torch.eye(dim, device=device, requires_grad=True) * (1 - lr_fc)).repeat(1, 1, 1)

        self.not_recalled = torch.ones(dim, device=device)

        self.empty_parameter = nn.Parameter(torch.zeros(1, device=device))

        self.start_recall_with_ith_item_init = start_recall_with_ith_item_init
        self.ith_item_state = torch.zeros((1, self.dim), device=self.device, requires_grad=True)
    
    def init_state(self, batch_size, recall=False, flush_level=1.0, prev_state=None):
        self.current_timestep = 0
        if recall and self.start_recall_with_ith_item_init:
            return self.ith_item_state.clone()
        return torch.zeros((batch_size, self.dim), device=self.device)
    
    def forward(self, inp, state):
        if self.encoding:
            c_in = torch.mv(self.W_fc, inp.squeeze())
            # c_in = torch.bmm(self.W_fc, torch.unsqueeze(inp, dim=2)).squeeze(2)
            c_in = c_in / torch.norm(c_in, p=2)
            state = state * self.alpha + c_in * (1 - self.alpha)
            state = F.normalize(state, p=2, dim=1)
            self.W_fc = self.W_fc + self.lr_fc * torch.outer(state.squeeze(), inp.squeeze())
            self.W_cf = self.W_cf + self.lr_cf * torch.outer(inp.squeeze(), state.squeeze())
            # self.W_fc = self.W_fc + self.lr_fc * torch.einsum("ik,ij->ikj", [state, inp])    # each column is a state
            # self.W_cf = self.W_cf + self.lr_cf * torch.einsum("ik,ij->ikj", [inp, state])    # store memory, each row is a state
            # print(inp, c_in, state, self.W_fc, self.W_cf)
            self.write(self.W_fc, 'W_fc')
            self.write(self.W_cf, 'W_cf')
            self.write(state, 'state')

            self.current_timestep += 1
            if self.current_timestep == self.start_recall_with_ith_item_init:
                self.ith_item_state = state.detach().clone()

            return inp, torch.zeros((1, self.dim)), state
        elif self.retrieving:
            # print(state.shape, self.W_cf.shape)
            # f_in_raw = torch.bmm(F.normalize(self.W_cf, p=2, dim=2), F.normalize(torch.unsqueeze(state, dim=2), p=2)).squeeze(2)
            f_in_raw = torch.mv(self.W_cf, state.squeeze())
            # f_in_filtered = (F.relu(f_in_raw - torch.max(f_in_raw) * self.threshold)) * self.not_recalled
            f_in = softmax(f_in_raw.unsqueeze(0) * self.not_recalled, self.softmax_temperature).squeeze(0)
            # print(f_in, f_in_filtered)
            if self.rand_mem:
                retrieved_idx = Categorical(f_in).sample()
            else:
                retrieved_idx = torch.argmax(f_in)
            retrieved_memory = torch.zeros(self.dim, device=self.device)
            retrieved_memory[retrieved_idx] = 1
            c_in = torch.mv(self.W_fc, retrieved_memory)
            # c_in = torch.bmm(self.W_fc, torch.unsqueeze(retrieved_memory, dim=2)).squeeze(2)
            state = F.relu(state * self.alpha + c_in * (1 - self.alpha))
            state = F.normalize(state, p=2, dim=1)
            self.not_recalled = self.not_recalled * (1 - retrieved_memory)
            # print(f_in, retrieved_idx, retrieved_memory, c_in, state)
            self.write(f_in, 'f_in')
            self.write(retrieved_idx, 'retrieved_idx')
            self.write(retrieved_memory, 'retrieved_memory')
            self.write(state, 'state')
            return retrieved_memory.unsqueeze(0), torch.zeros((1, self.dim)), state
    
    def set_encoding(self, status):
        self.encoding = status
    
    def set_retrieval(self, status):
        self.retrieving = status

    def reset_memory(self):
        self.W_cf = torch.zeros((self.dim, self.dim), device=self.device)
        self.W_fc = torch.eye(self.dim, device=self.device) * (1 - self.lr_fc)
        self.not_recalled = torch.ones(self.dim, device=self.device)


class TCMRNN(BasicModule):
    def __init__(self, hidden_dim: int, slot_num=21, act_fn='ReLU', lr_cf: float = 1.0, lr_fc: float = 0.9, dt: float = 10, tau: float = 20, 
    record_recalled: bool = False, mem_gate_type="constant", output_type="recalled_item", init_state_type="zeros", evolve_state_before_recall=False, 
    flush_weight=1.0, noise_std=0.0, evolve_state_between_phases=False, input_layer=False, start_recall_with_ith_item_init=0, use_input_gate=False, 
    recall_type="weight_sum", recurrence_after_adding_memory=False, rec_gate_type="constant", decision_time="after_recurrence", small_init_weight=False,
    normalize_state=False, recall_softmax_temperature=1.0, device: str = 'cpu'):
        super().__init__()
        self.device = device

        self.encoding = False
        self.retrieving = False

        self.lr_cf = lr_cf
        self.lr_fc = lr_fc
        self.dt = dt
        self.alpha = float(dt) / float(tau)
        self.record_recalled = record_recalled
        self.evolve_state_before_recall = evolve_state_before_recall
        self.flush_weight = flush_weight
        self.noise_std = noise_std
        self.evolve_state_between_phases = evolve_state_between_phases
        self.last_encoding = False  # last step is encoding, flag for one more iteration between encoding and retrieval
        self.start_recall_with_ith_item_init = start_recall_with_ith_item_init
        self.recall_type = recall_type
        self.recurrence_after_adding_memory = recurrence_after_adding_memory
        self.decision_time = decision_time      # after_recurrence or after_recall
        self.normalize_state = normalize_state
        self.small_init_weight = small_init_weight
        self.recall_softmax_temperature = recall_softmax_temperature

        self.current_timestep = 0
        
        self.hidden_dim = hidden_dim
        self.slot_num = slot_num
        self.act_fn = load_act_fn(act_fn)

        self.use_input_gate = use_input_gate
        if use_input_gate:
            self.input_gate = torch.nn.Parameter(torch.ones(1, device=device), requires_grad=True)

        self.mem_gate_type = mem_gate_type
        if mem_gate_type == "vector":
            self.mem_gate = nn.Linear(hidden_dim, hidden_dim)
        elif mem_gate_type == "scalar":
            self.mem_gate = nn.Linear(hidden_dim, 1)
        elif mem_gate_type == "constant":
            self.mem_gate = 0.5
        else:
            raise AttributeError("Invalid memory gate type, should be vector or scalar")

        self.rec_gate_type = rec_gate_type
        if rec_gate_type == "vector":
            self.rec_gate = nn.Linear(hidden_dim, hidden_dim)
        elif rec_gate_type == "scalar":
            self.rec_gate = nn.Linear(hidden_dim, 1)
        elif rec_gate_type == "constant":
            self.rec_gate = 1.0
        else:
            raise AttributeError("Invalid memory gate type, should be vector or scalar")

        self.output_type = output_type

        # self.fc_in = nn.Linear(hidden_dim, hidden_dim)
        self.input_layer = input_layer
        if input_layer:
            self.fc_in = nn.Linear(hidden_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc_decision = nn.Linear(hidden_dim, slot_num)
        self.fc_critic = nn.Linear(hidden_dim, 1)
        if self.output_type == "dec2" or self.output_type == "both_dec2":
            self.fc_decision2 = nn.Linear(hidden_dim, slot_num)

        self.W_cf = torch.zeros((1, slot_num, hidden_dim), device=device, requires_grad=True)
        self.W_fc = (torch.eye(slot_num, device=device, requires_grad=True) * (1 - lr_fc)).repeat(1, 1, 1)

        self.hidden_state = torch.zeros((1, self.hidden_dim), device=self.device, requires_grad=True)
        self.ith_item_state = torch.zeros((1, self.hidden_dim), device=self.device, requires_grad=True)

        self.init_state_type = init_state_type
        if init_state_type == "train":
            self.h0 = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)
        if self.init_state_type == 'train_diff':
            self.h0 = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)
            self.h0_recall = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)

        # self.not_recalled = torch.ones(hidden_dim, device=device)
        if self.small_init_weight:
            self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        self.fc_hidden.weight.data.uniform_(-stdv, stdv)
        self.fc_hidden.bias.data.zero_()

    def init_state(self, batch_size, recall=False, flush_level=1.0, prev_state=None):
        if recall:
            if self.start_recall_with_ith_item_init:
                self.hidden_state = self.ith_item_state.clone()
            elif self.init_state_type == 'zeros':
                # print(torch.mean(self.hidden_state))
                self.hidden_state = self.hidden_state + torch.randn(self.hidden_state.shape).to(self.device) * max(torch.mean(self.hidden_state) * flush_level * self.flush_weight, torch.tensor(0.1))
            elif self.init_state_type == 'all_zeros':
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
            if self.init_state_type == "zeros" or self.init_state_type == 'all_zeros':
                self.hidden_state = torch.zeros((batch_size, self.hidden_dim), device=self.device, requires_grad=True)
                state = torch.zeros((batch_size, self.hidden_dim), device=self.device, requires_grad=True)
            elif self.init_state_type == "train" or self.init_state_type == "train_diff":
                self.hidden_state = self.h0.repeat(batch_size, 1)
                state = self.act_fn(self.hidden_state)
            else:
                raise AttributeError("Invalid init_state_type, should be zeros, train or train_diff")
            self.W_cf = torch.zeros((batch_size, self.slot_num, self.hidden_dim), device=self.device, requires_grad=True)
        if self.hidden_dim == self.slot_num:
            self.W_fc = (torch.eye(self.hidden_dim, device=self.device, requires_grad=True) * (1 - self.lr_fc)).repeat(batch_size, 1, 1)
        else:
            self.W_fc = (torch.cat((torch.eye(self.slot_num, device=self.device, requires_grad=True) * (1 - self.lr_fc), \
                        torch.zeros((self.hidden_dim-self.slot_num, self.slot_num), device=self.device, requires_grad=True)), 0)).repeat(batch_size, 1, 1)
        self.write(state, 'init_state')
        self.current_timestep = 0
        return state
    
    def forward(self, inp, state):
        if self.encoding:
            c_in = torch.bmm(self.W_fc, torch.unsqueeze(inp, dim=2)).squeeze(2)
            c_in = c_in / torch.norm(c_in, p=2, dim=1).reshape(-1, 1)
            # if self.mem_gate_type == "constant":
            #     gate = self.mem_gate
            # else:
            #     gate = self.mem_gate(state).sigmoid()
            if self.noise_std > 0:
                noise = math.sqrt(2*self.dt)*self.noise_std*torch.randn(self.hidden_state.size()).to(self.device)
            else:
                noise = 0
            if self.input_layer:
                c_in = self.fc_in(c_in)
            if self.use_input_gate:
                c_in = c_in * torch.min(self.input_gate, torch.tensor(5.0))
            if self.decision_time == "after_recall":
                decision = softmax(self.fc_decision(state + c_in))
            if self.recurrence_after_adding_memory:
                self.hidden_state = self.hidden_state * (1 - self.alpha) + (self.fc_hidden(state + c_in)) * self.alpha + noise
                self.write(state + c_in, "half_state")
            else:
                self.hidden_state = self.hidden_state * (1 - self.alpha) + (self.fc_hidden(state) + c_in) * self.alpha + noise
                self.write(self.fc_hidden(state), "half_state")
            state = self.act_fn(self.hidden_state)
            if self.normalize_state:
                state = F.normalize(state, p=2)
            self.W_fc = self.W_fc + self.lr_fc * torch.einsum("ik,ij->ikj", [state, inp])    # each column is a state
            self.W_cf = self.W_cf + self.lr_cf * torch.einsum("ik,ij->ikj", [inp, state])    # store memory, each row is a state
            # print(self.hidden_state)
            # print(state)
            if self.decision_time == "after_recurrence":
                decision = softmax(self.fc_decision(state))
            # print(inp, c_in, state, self.W_fc, self.W_cf)
            if self.output_type == "decision":
                output = decision
            elif self.output_type == "recalled_item":
                output = inp
            elif self.output_type == "both":
                output = (inp, decision)
            elif self.output_type == "both_dec":
                output = (decision, inp)
            elif self.output_type == "dec2":
                decision2 = softmax(self.fc_decision2(state))
                output = (decision, decision2)
            elif self.output_type == "both_dec2":
                decision2 = softmax(self.fc_decision2(state))
                output = (decision, inp, decision2)
            else:
                raise AttributeError("Invalid output type, should be decision or recalled_item")
            value = self.fc_critic(state)

            self.current_timestep += 1
            if self.current_timestep == self.start_recall_with_ith_item_init:
                self.ith_item_state = self.hidden_state.detach().clone()
            self.last_encoding = True

            self.write(self.W_fc, 'W_fc')
            self.write(self.W_cf, 'W_cf')
            self.write(state, 'state')
            self.write(decision, 'decision')

            return output, value, state
        elif self.retrieving:
            if self.last_encoding and self.evolve_state_between_phases:
                if self.noise_std > 0:
                    noise = math.sqrt(2*self.dt)*self.noise_std*torch.randn(self.hidden_state.size()).to(self.device)
                else:
                    noise = 0
                self.hidden_state = self.hidden_state * (1 - self.alpha) + (self.fc_hidden(state)) * self.alpha + noise
                state = self.act_fn(self.hidden_state)
                self.write(state, 'state')
            self.last_encoding = False

            if self.evolve_state_before_recall:
                state_for_recall = self.act_fn(self.hidden_state * (1 - self.alpha) + self.fc_hidden(state) * self.alpha)
            # print(state.shape, self.W_cf.shape)
            else:
                state_for_recall = state

            if self.mem_gate_type == "constant":
                mem_gate = self.mem_gate
            else:
                mem_gate = self.mem_gate(state_for_recall).sigmoid()
            if self.rec_gate_type == "constant":
                rec_gate = self.rec_gate
            else:
                rec_gate = self.rec_gate(state_for_recall).sigmoid()

            f_in_raw = torch.bmm(F.normalize(self.W_cf, p=2, dim=2), F.normalize(torch.unsqueeze(state_for_recall, dim=2), p=2)).squeeze(2)
            f_in = softmax(f_in_raw, self.recall_softmax_temperature)
            # f_in_inhibit_recall = f_in * self.not_recalled
            if self.recall_type == "random":
                retrieved_idx = Categorical(f_in).sample()
                retrieved_memory = F.one_hot(retrieved_idx, self.slot_num).float()
                retrieved_memory.requires_grad = True
            elif self.recall_type == "weight_sum":
                retrieved_idx = torch.argmax(f_in)
                retrieved_memory = f_in
            else:
                retrieved_idx = torch.argmax(f_in, dim=1)
                retrieved_memory = F.one_hot(retrieved_idx, self.slot_num).float()
                retrieved_memory.requires_grad = True
            c_in = torch.bmm(self.W_fc, torch.unsqueeze(retrieved_memory, dim=2)).squeeze(2)
            c_in = c_in / torch.norm(c_in, p=2, dim=1).reshape(-1, 1)

            if self.noise_std > 0:
                noise = math.sqrt(2*self.dt)*self.noise_std*torch.randn(self.hidden_state.size()).to(self.device)
            else:
                noise = 0
            if self.decision_time == "after_recall":
                decision = softmax(self.fc_decision(state * rec_gate + c_in * mem_gate))
            if self.recurrence_after_adding_memory:
                self.hidden_state = self.hidden_state * (1 - self.alpha) + (self.fc_hidden(state * rec_gate + c_in * mem_gate)) * self.alpha + noise
                self.write(state * rec_gate + c_in * mem_gate, "half_state")
            else:
                self.hidden_state = self.hidden_state * (1 - self.alpha) + (self.fc_hidden(state) * rec_gate + c_in * mem_gate) * self.alpha + noise
                self.write(self.fc_hidden(state) * rec_gate, "half_state")
            state = self.act_fn(self.hidden_state)
            if self.normalize_state:
                state = F.normalize(state, p=2)
                
            if self.decision_time == "after_recurrence":
                decision = softmax(self.fc_decision(state))

            # self.W_fc = self.W_fc - self.lr_fc * torch.outer(state.squeeze(), retrieved_memory.squeeze())
            # self.not_recalled[retrieved_idx] = 0
            if self.record_recalled:
                self.not_recalled = self.not_recalled * (1 - retrieved_memory)
            if self.output_type == "decision":
                output = decision
            elif self.output_type == "recalled_item":
                output = f_in
            elif self.output_type == "both":
                output = (f_in, decision)
            elif self.output_type == "both_dec":
                output = (decision, f_in)
            elif self.output_type == "dec2":
                decision2 = softmax(self.fc_decision2(state))
                output = (decision, decision2)
            elif self.output_type == "both_dec2":
                decision2 = softmax(self.fc_decision2(state))
                output = (decision, f_in, decision2)
            else:
                raise AttributeError("Invalid output type, should be decision or recalled_item")
            value = self.fc_critic(state)

            self.write(f_in, 'f_in')
            self.write(f_in_raw, 'f_in_raw')
            self.write(retrieved_idx, 'retrieved_idx')
            self.write(retrieved_memory, 'retrieved_memory')
            self.write(state, 'state')
            self.write(decision, 'decision')
            self.write(mem_gate, 'mem_gate_recall')
            if self.rec_gate_type != "constant":
                self.write(rec_gate, 'rec_gate_recall')

            return output, value, state
    
    def set_encoding(self, status):
        self.encoding = status
    
    def set_retrieval(self, status):
        self.retrieving = status

    def reset_memory(self):
        self.W_cf = torch.zeros((self.hidden_dim, self.hidden_dim), device=self.device)
        self.W_fc = torch.eye(self.hidden_dim, device=self.device) * (1 - self.lr_fc)
        self.not_recalled = torch.ones(self.hidden_dim, device=self.device)


        # if self.encoding:
        #     c_in = torch.bmm(self.W_fc, torch.unsqueeze(inp, dim=2)).squeeze(2)
        #     c_in = c_in / torch.norm(c_in, p=2, dim=1).reshape(-1, 1)
        # elif self.retrieving:
        #     if self.mem_gate_type == "constant":
        #         gate = self.mem_gate
        #     else:
        #         gate = self.mem_gate(state).sigmoid()
        #     f_in_raw = torch.bmm(self.W_cf, torch.unsqueeze(state, dim=2)).squeeze(2)
        #     f_in = softmax(f_in_raw)
        #     retrieved_idx = torch.argmax(f_in, dim=1)
        #     retrieved_memory = F.one_hot(retrieved_idx, self.hidden_dim).float()
        #     retrieved_memory.requires_grad = True
        #     c_in = torch.bmm(self.W_fc, torch.unsqueeze(retrieved_memory, dim=2)).squeeze(2)
        #     c_in = c_in / torch.norm(c_in, p=2, dim=1).reshape(-1, 1) * gate
        # else:
        #     c_in = torch.zeros(inp.shape)
        # self.hidden_state = self.hidden_state * (1 - self.alpha) + (self.fc_hidden(state) + c_in) * self.alpha


class TCMSlotRNN(BasicModule):
    def __init__(self, hidden_dim: int, slot_num=21, act_fn='ReLU', lr_cf: float = 1.0, lr_fc: float = 0.9, dt: float = 10, tau: float = 20, 
    record_recalled: bool = False, mem_gate_type="constant", output_type="recalled_item", init_state_type="zeros", evolve_state_before_recall=False, 
    flush_weight=1.0, noise_std=0.0, evolve_state_between_phases=False, input_layer=False, start_recall_with_ith_item_init=0, use_input_gate=False, 
    random_recall=False, recurrence_after_adding_memory=False, rec_gate_type="constant", decision_time="after_recurrence", small_init_weight=False,
    normalize_state=False, device: str = 'cpu'):
        super().__init__()
        self.device = device

        self.encoding = False
        self.retrieving = False

        self.lr_cf = lr_cf
        self.lr_fc = lr_fc
        self.dt = dt
        self.alpha = float(dt) / float(tau)
        self.record_recalled = record_recalled
        self.evolve_state_before_recall = evolve_state_before_recall
        self.flush_weight = flush_weight
        self.noise_std = noise_std
        self.evolve_state_between_phases = evolve_state_between_phases
        self.last_encoding = False  # last step is encoding, flag for one more iteration between encoding and retrieval
        self.start_recall_with_ith_item_init = start_recall_with_ith_item_init
        self.random_recall = random_recall
        self.recurrence_after_adding_memory = recurrence_after_adding_memory
        self.decision_time = decision_time      # after_recurrence or after_recall
        self.normalize_state = normalize_state
        self.small_init_weight = small_init_weight

        self.current_timestep = 0
        
        self.hidden_dim = hidden_dim
        self.slot_num = slot_num
        self.act_fn = load_act_fn(act_fn)

        self.use_input_gate = use_input_gate
        if use_input_gate:
            self.input_gate = torch.nn.Parameter(torch.ones(1, device=device), requires_grad=True)

        self.mem_gate_type = mem_gate_type
        if mem_gate_type == "vector":
            self.mem_gate = nn.Linear(hidden_dim, hidden_dim)
        elif mem_gate_type == "scalar":
            self.mem_gate = nn.Linear(hidden_dim, 1)
        elif mem_gate_type == "constant":
            self.mem_gate = 0.5
        else:
            raise AttributeError("Invalid memory gate type, should be vector or scalar")

        self.rec_gate_type = rec_gate_type
        if rec_gate_type == "vector":
            self.rec_gate = nn.Linear(hidden_dim, hidden_dim)
        elif rec_gate_type == "scalar":
            self.rec_gate = nn.Linear(hidden_dim, 1)
        elif rec_gate_type == "constant":
            self.rec_gate = 1.0
        else:
            raise AttributeError("Invalid memory gate type, should be vector or scalar")

        self.output_type = output_type

        # self.fc_in = nn.Linear(hidden_dim, hidden_dim)
        self.input_layer = input_layer
        if input_layer:
            self.fc_in = nn.Linear(hidden_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.fc_decision = nn.Linear(hidden_dim, slot_num)
        self.fc_critic = nn.Linear(hidden_dim, 1)
        if self.output_type == "dec2" or self.output_type == "both_dec2":
            self.fc_decision2 = nn.Linear(hidden_dim, slot_num)

        self.W_cf = torch.zeros((1, slot_num, hidden_dim), device=device, requires_grad=True)
        self.W_fc = (torch.eye(slot_num, device=device, requires_grad=True) * (1 - lr_fc)).repeat(1, 1, 1)

        self.hidden_state = torch.zeros((1, self.hidden_dim), device=self.device, requires_grad=True)
        self.ith_item_state = torch.zeros((1, self.hidden_dim), device=self.device, requires_grad=True)

        self.init_state_type = init_state_type
        if init_state_type == "train":
            self.h0 = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)
        if self.init_state_type == 'train_diff':
            self.h0 = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)
            self.h0_recall = torch.nn.Parameter(torch.zeros(hidden_dim), requires_grad=True)

        # self.not_recalled = torch.ones(hidden_dim, device=device)
        if self.small_init_weight:
            self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        self.fc_hidden.weight.data.uniform_(-stdv, stdv)
        self.fc_hidden.bias.data.zero_()

    def init_state(self, batch_size, recall=False, flush_level=1.0, prev_state=None):
        if recall:
            if self.start_recall_with_ith_item_init:
                self.hidden_state = self.ith_item_state.clone()
            elif self.init_state_type == 'zeros':
                # print(torch.mean(self.hidden_state))
                self.hidden_state = self.hidden_state + torch.randn(self.hidden_state.shape).to(self.device) * max(torch.mean(self.hidden_state) * flush_level * self.flush_weight, torch.tensor(0.1))
            elif self.init_state_type == 'all_zeros':
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
            if self.init_state_type == "zeros" or self.init_state_type == 'all_zeros':
                self.hidden_state = torch.zeros((batch_size, self.hidden_dim), device=self.device, requires_grad=True)
                state = torch.zeros((batch_size, self.hidden_dim), device=self.device, requires_grad=True)
            elif self.init_state_type == "train" or self.init_state_type == "train_diff":
                self.hidden_state = self.h0.repeat(batch_size, 1)
                state = self.act_fn(self.hidden_state)
            else:
                raise AttributeError("Invalid init_state_type, should be zeros, train or train_diff")
            self.W_cf = torch.zeros((batch_size, self.slot_num, self.hidden_dim), device=self.device, requires_grad=True)
        if self.hidden_dim == self.slot_num:
            self.W_fc = (torch.eye(self.hidden_dim, device=self.device, requires_grad=True) * (1 - self.lr_fc)).repeat(batch_size, 1, 1)
        else:
            self.W_fc = (torch.cat((torch.eye(self.slot_num, device=self.device, requires_grad=True) * (1 - self.lr_fc), \
                        torch.zeros((self.hidden_dim-self.slot_num, self.slot_num), device=self.device, requires_grad=True)), 0)).repeat(batch_size, 1, 1)
        self.write(state, 'init_state')
        self.current_timestep = 0
        return state
    
    def forward(self, inp, state):
        if self.encoding:
            c_in = torch.bmm(self.W_fc, torch.unsqueeze(inp, dim=2)).squeeze(2)
            c_in = c_in / torch.norm(c_in, p=2, dim=1).reshape(-1, 1)
            # if self.mem_gate_type == "constant":
            #     gate = self.mem_gate
            # else:
            #     gate = self.mem_gate(state).sigmoid()
            if self.noise_std > 0:
                noise = math.sqrt(2*self.dt)*self.noise_std*torch.randn(self.hidden_state.size()).to(self.device)
            else:
                noise = 0
            if self.input_layer:
                c_in = self.fc_in(c_in)
            if self.use_input_gate:
                c_in = c_in * torch.min(self.input_gate, torch.tensor(5.0))
            if self.decision_time == "after_recall":
                decision = softmax(self.fc_decision(state + c_in))
            if self.recurrence_after_adding_memory:
                self.hidden_state = self.hidden_state * (1 - self.alpha) + (self.fc_hidden(state + c_in)) * self.alpha + noise
                self.write(state + c_in, "half_state")
            else:
                self.hidden_state = self.hidden_state * (1 - self.alpha) + (self.fc_hidden(state) + c_in) * self.alpha + noise
                self.write(self.fc_hidden(state), "half_state")
            state = self.act_fn(self.hidden_state)
            if self.normalize_state:
                state = F.normalize(state, p=2)
            self.W_fc = self.W_fc + self.lr_fc * torch.einsum("ik,ij->ikj", [state, inp])    # each column is a state
            self.W_cf = self.W_cf + self.lr_cf * torch.einsum("ik,ij->ikj", [inp, state])    # store memory, each row is a state
            # print(self.hidden_state)
            # print(state)
            if self.decision_time == "after_recurrence":
                decision = softmax(self.fc_decision(state))
            # print(inp, c_in, state, self.W_fc, self.W_cf)
            if self.output_type == "decision":
                output = decision
            elif self.output_type == "recalled_item":
                output = inp
            elif self.output_type == "both":
                output = (inp, decision)
            elif self.output_type == "both_dec":
                output = (decision, inp)
            elif self.output_type == "dec2":
                decision2 = softmax(self.fc_decision2(state))
                output = (decision, decision2)
            elif self.output_type == "both_dec2":
                decision2 = softmax(self.fc_decision2(state))
                output = (decision, inp, decision2)
            else:
                raise AttributeError("Invalid output type, should be decision or recalled_item")
            value = self.fc_critic(state)

            self.current_timestep += 1
            if self.current_timestep == self.start_recall_with_ith_item_init:
                self.ith_item_state = self.hidden_state.detach().clone()
            self.last_encoding = True

            self.write(self.W_fc, 'W_fc')
            self.write(self.W_cf, 'W_cf')
            self.write(state, 'state')
            self.write(decision, 'decision')

            return output, value, state
        elif self.retrieving:
            if self.last_encoding and self.evolve_state_between_phases:
                if self.noise_std > 0:
                    noise = math.sqrt(2*self.dt)*self.noise_std*torch.randn(self.hidden_state.size()).to(self.device)
                else:
                    noise = 0
                self.hidden_state = self.hidden_state * (1 - self.alpha) + (self.fc_hidden(state)) * self.alpha + noise
                state = self.act_fn(self.hidden_state)
                self.write(state, 'state')
            self.last_encoding = False

            if self.evolve_state_before_recall:
                state_for_recall = self.act_fn(self.hidden_state * (1 - self.alpha) + self.fc_hidden(state) * self.alpha)
            # print(state.shape, self.W_cf.shape)
            else:
                state_for_recall = state

            if self.mem_gate_type == "constant":
                mem_gate = self.mem_gate
            else:
                mem_gate = self.mem_gate(state_for_recall).sigmoid()
            if self.rec_gate_type == "constant":
                rec_gate = self.rec_gate
            else:
                rec_gate = self.rec_gate(state_for_recall).sigmoid()

            f_in_raw = torch.bmm(F.normalize(self.W_cf, p=2, dim=2), F.normalize(torch.unsqueeze(state_for_recall, dim=2), p=2)).squeeze(2)
            f_in = softmax(f_in_raw)
            # f_in_inhibit_recall = f_in * self.not_recalled
            if self.random_recall:
                retrieved_idx = Categorical(f_in).sample()
            else:
                retrieved_idx = torch.argmax(f_in, dim=1)
            retrieved_memory = F.one_hot(retrieved_idx, self.slot_num).float()
            retrieved_memory.requires_grad = True
            c_in = torch.bmm(self.W_fc, torch.unsqueeze(retrieved_memory, dim=2)).squeeze(2)
            c_in = c_in / torch.norm(c_in, p=2, dim=1).reshape(-1, 1)

            if self.noise_std > 0:
                noise = math.sqrt(2*self.dt)*self.noise_std*torch.randn(self.hidden_state.size()).to(self.device)
            else:
                noise = 0
            if self.decision_time == "after_recall":
                decision = softmax(self.fc_decision(state * rec_gate + c_in * mem_gate))
            if self.recurrence_after_adding_memory:
                self.hidden_state = self.hidden_state * (1 - self.alpha) + (self.fc_hidden(state * rec_gate + c_in * mem_gate)) * self.alpha + noise
                self.write(state * rec_gate + c_in * mem_gate, "half_state")
            else:
                self.hidden_state = self.hidden_state * (1 - self.alpha) + (self.fc_hidden(state) * rec_gate + c_in * mem_gate) * self.alpha + noise
                self.write(self.fc_hidden(state) * rec_gate, "half_state")
            state = self.act_fn(self.hidden_state)
            if self.normalize_state:
                state = F.normalize(state, p=2)
                
            if self.decision_time == "after_recurrence":
                decision = softmax(self.fc_decision(state))

            # self.W_fc = self.W_fc - self.lr_fc * torch.outer(state.squeeze(), retrieved_memory.squeeze())
            # self.not_recalled[retrieved_idx] = 0
            if self.record_recalled:
                self.not_recalled = self.not_recalled * (1 - retrieved_memory)
            if self.output_type == "decision":
                output = decision
            elif self.output_type == "recalled_item":
                output = f_in
            elif self.output_type == "both":
                output = (f_in, decision)
            elif self.output_type == "both_dec":
                output = (decision, f_in)
            elif self.output_type == "dec2":
                decision2 = softmax(self.fc_decision2(state))
                output = (decision, decision2)
            elif self.output_type == "both_dec2":
                decision2 = softmax(self.fc_decision2(state))
                output = (decision, f_in, decision2)
            else:
                raise AttributeError("Invalid output type, should be decision or recalled_item")
            value = self.fc_critic(state)

            self.write(f_in, 'f_in')
            self.write(f_in_raw, 'f_in_raw')
            self.write(retrieved_idx, 'retrieved_idx')
            self.write(retrieved_memory, 'retrieved_memory')
            self.write(state, 'state')
            self.write(decision, 'decision')
            self.write(mem_gate, 'mem_gate_recall')
            if self.rec_gate_type != "constant":
                self.write(rec_gate, 'rec_gate_recall')

            return output, value, state
    
    def set_encoding(self, status):
        self.encoding = status
    
    def set_retrieval(self, status):
        self.retrieving = status

    def reset_memory(self):
        self.W_cf = torch.zeros((self.hidden_dim, self.hidden_dim), device=self.device)
        self.W_fc = torch.eye(self.hidden_dim, device=self.device) * (1 - self.lr_fc)
        self.not_recalled = torch.ones(self.hidden_dim, device=self.device)


class TCMLSTM(BasicModule):
    def __init__(self, hidden_dim: int, act_fn='ReLU', lr_cf: float = 1.0, lr_fc: float = 0.9, record_recalled: bool = False, mem_gate_type="constant", 
    output_type="recalled_item", device: str = 'cpu'):
        super().__init__()
        self.device = device

        self.encoding = False
        self.retrieving = False

        self.lr_cf = lr_cf
        self.lr_fc = lr_fc
        self.record_recalled = record_recalled

        self.hidden_dim = hidden_dim
        self.act_fn = load_act_fn(act_fn)

        self.mem_gate_type = mem_gate_type
        if mem_gate_type == "vector":
            self.mem_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        elif mem_gate_type == "scalar":
            self.mem_gate = nn.Linear(hidden_dim * 2, 1)
        elif mem_gate_type == "constant":
            self.mem_gate = 0.5
        else:
            raise AttributeError("Invalid memory gate type, should be vector or scalar")
        self.output_type = output_type

        # self.fc_in = nn.Linear(hidden_dim, hidden_dim)
        self.fc_hidden = nn.Linear(hidden_dim, hidden_dim * 4, bias=False)
        self.fc_decision = nn.Linear(hidden_dim, hidden_dim)
        self.fc_critic = nn.Linear(hidden_dim, 1)

        self.W_cf = torch.zeros((hidden_dim, hidden_dim), device=device, requires_grad=True)
        self.W_fc = torch.eye(hidden_dim, device=device, requires_grad=True) * (1 - lr_fc)

        self.not_recalled = torch.ones(hidden_dim, device=device)
    
    def init_state(self, batch_size, recall=False, flush_level=1.0, prev_state=None):
        if recall:
            h_prev, c_prev = prev_state
            h = h_prev + torch.normal(0.0, torch.mean(h_prev) * flush_level)
            c = c_prev + torch.normal(0.0, torch.mean(c_prev) * flush_level)
        else:
            h = torch.zeros((batch_size, self.hidden_dim), device=self.device, requires_grad=True)
            c = torch.zeros((batch_size, self.hidden_dim), device=self.device, requires_grad=True)
        return (h, c)
    
    def forward(self, inp, state):
        h, c = state
        if self.encoding:
            c_in = torch.mv(self.W_fc, inp)
            c_in = c_in / torch.norm(c_in, p=2)

            if self.mem_gate_type == "constant":
                gate = self.mem_gate
            else:
                gate = self.mem_gate(torch.cat((c, h), 1)).sigmoid()
            
            preact = self.fc_hidden(h.to(self.device)) * (1 - gate) + c_in.repeat(4) * gate
            z = preact[:, :self.hidden_dim].tanh()
            z_i, z_f, z_o = preact[:, self.hidden_dim:].sigmoid().chunk(3, 1)
            c_next = torch.mul(z_f, c) + torch.mul(z_i, z)
            h_next = torch.mul(z_o, c_next.tanh())
            state = self.act_fn(h_next)

            self.W_fc = self.W_fc + self.lr_fc * torch.outer(h_next.squeeze(), inp.squeeze())    # each column is a state
            self.W_cf = self.W_cf + self.lr_cf * torch.outer(inp.squeeze(), h_next.squeeze())    # store memory, each row is a state

            # print(self.hidden_state)
            # print(state)
            decision = softmax(self.fc_decision(state))
            # print(inp, c_in, state, self.W_fc, self.W_cf)
            if self.output_type == "decision":
                output = decision
            elif self.output_type == "recalled_item":
                output = inp
            else:
                raise AttributeError("Invalid output type, should be decision or recalled_item")
            value = self.fc_critic(state)

            self.write(self.W_fc, 'W_fc')
            self.write(self.W_cf, 'W_cf')
            self.write(state, 'state')

            return output, value, (h_next, c_next)
        elif self.retrieving:
            # print(state.shape, self.W_cf.shape)
            f_in = softmax(torch.mv(self.W_cf, h.squeeze()))
            f_in_inhibit_recall = f_in * self.not_recalled
            retrieved_idx = torch.argmax(f_in_inhibit_recall)
            # retrieved_idx = Categorical(f_in_inhibit_recall).sample()
            retrieved_memory = F.one_hot(retrieved_idx, self.hidden_dim).float()
            retrieved_memory.requires_grad = True
            c_in = torch.mv(self.W_fc, retrieved_memory)
            c_in = c_in / torch.norm(c_in, p=2)

            if self.mem_gate_type == "constant":
                gate = self.mem_gate
            else:
                gate = self.mem_gate(torch.cat((c, h), 1)).sigmoid()
            
            preact = self.fc_hidden(h.to(self.device)) * (1 - gate) + c_in.repeat(4) * gate
            z = preact[:, :self.hidden_dim].tanh()
            z_i, z_f, z_o = preact[:, self.hidden_dim:].sigmoid().chunk(3, 1)
            c_next = torch.mul(z_f, c) + torch.mul(z_i, z)
            h_next = torch.mul(z_o, c_next.tanh())
            state = self.act_fn(h_next)

            decision = softmax(self.fc_decision(state))
            # self.W_fc = self.W_fc - self.lr_fc * torch.outer(state.squeeze(), retrieved_memory.squeeze())
            # self.not_recalled[retrieved_idx] = 0
            if self.record_recalled:
                self.not_recalled = self.not_recalled * (1 - retrieved_memory)
            if self.output_type == "decision":
                output = decision
            elif self.output_type == "recalled_item":
                output = f_in_inhibit_recall
            else:
                raise AttributeError("Invalid output type, should be decision or recalled_item")
            value = self.fc_critic(state)

            self.write(f_in, 'f_in')
            self.write(retrieved_idx, 'retrieved_idx')
            self.write(retrieved_memory, 'retrieved_memory')
            self.write(state, 'state')

            return output, value, (h_next, c_next)
    
    def set_encoding(self, status):
        self.encoding = status
    
    def set_retrieval(self, status):
        self.retrieving = status

    def reset_memory(self):
        self.W_cf = torch.zeros((self.hidden_dim, self.hidden_dim), device=self.device)
        self.W_fc = torch.eye(self.hidden_dim, device=self.device) * (1 - self.lr_fc)
        self.not_recalled = torch.ones(self.hidden_dim, device=self.device)
