import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ..base_module import BasicModule


class ValueMemory(BasicModule):
    def __init__(self, similarity_measure, value_dim: int, capacity: int, recall_method="argmax", 
                 batch_size=1, noise_std=0.0, device: str = 'cpu') -> None:
        super().__init__()
        self.device = device
        self.similarity_measure = similarity_measure
        self.value_dim = value_dim
        self.capacity = capacity
        self.noise_std = noise_std

        self.values = torch.zeros((batch_size, capacity, value_dim)).to(self.device)
        
        self.stored_memory = 0
        self.to_be_replaced = 0

        self.encoding = False
        self.retrieving = False

        self.recall_method = recall_method
        self.batch_size = batch_size

    def reset_memory(self, flush=True):
        if flush:
            self.flush()
        else:
            self.values.detach_()
        self.encoding = False   
        self.retrieving = False

    def flush(self):
        self.values = torch.zeros((self.batch_size, self.capacity, self.value_dim)).to(self.device)
        self.stored_memory = 0
        # self.to_be_replaced = 0

    def encode(self, value):
        if self.encoding:
            self.write(self.values, "memory_values")
            index = F.one_hot(torch.tensor(self.to_be_replaced), self.capacity).repeat(self.batch_size, 1).float().to(self.device)
            self.values = self.values + torch.einsum("ik,ij->ikj", [index, value])
            self.to_be_replaced = (self.to_be_replaced + 1) % self.capacity
            self.stored_memory = min(self.stored_memory + 1, self.capacity)
            # print(self.values[:, :10])
    
    def retrieve(self, query, input_weight=1.0, beta=None):
        if self.stored_memory == 0 or not self.retrieving:
            return torch.zeros(query.shape[0], self.value_dim).to(self.device)
        # values = self.values.detach().clone()
        similarity, raw_similarity = self.similarity_measure(query, self.values, input_weight, beta)
        similarity = similarity + torch.randn_like(similarity) * self.noise_std
        self.write(raw_similarity, "raw_similarity")
        if self.recall_method == "random":
            retrieved_idx = torch.tensor([Categorical(torch.abs(similarity[i])).sample() for i in range(similarity.shape[0])])
            similarity = F.one_hot(retrieved_idx, self.capacity).float()
        elif self.recall_method == "argmax":
            retrieved_idx = torch.argmax(similarity, dim=-1)
            similarity = F.one_hot(retrieved_idx, self.capacity).float()
        elif self.recall_method == "weight_sum":
            pass
        else:
            raise NotImplementedError
        self.write(similarity, "similarity")
        retrieved_memory = torch.bmm(torch.unsqueeze(similarity, dim=1), self.values).squeeze(1)
        self.write(retrieved_memory, "retrieved_memory")
        # print(retrieved_memory[:, :10], similarity)
        return retrieved_memory, raw_similarity

    def get_vals(self):
        return self.values.detach().cpu().numpy()

