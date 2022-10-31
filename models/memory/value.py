import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ..basic_module import BasicModule


class ValueMemory(BasicModule):
    def __init__(self, similarity_measure, value_dim: int, capacity: int, random_recall=False, batch_size=1, device: str = 'cpu') -> None:
        super().__init__()
        self.device = device
        self.similarity_measure = similarity_measure
        self.value_dim = value_dim
        self.capacity = capacity

        self.values = torch.zeros((batch_size, capacity, value_dim)).to(self.device)
        
        self.stored_memory = 0
        self.to_be_replaced = 0

        self.encoding = False
        self.retrieving = False

        self.random_recall = random_recall
        self.batch_size = batch_size

    def reset_memory(self):
        self.flush()
        self.encoding = False
        self.retrieving = False

    def flush(self):
        self.values = torch.zeros((self.batch_size, self.capacity, self.value_dim)).to(self.device)
        self.stored_memory = 0
        self.to_be_replaced = 0

    def encode(self, value):
        if self.encoding:
            self.write(self.values, "memory_values")
            index = F.one_hot(torch.tensor(self.to_be_replaced), self.capacity).repeat(self.batch_size, 1).float().to(self.device)
            self.values = self.values + torch.einsum("ik,ij->ikj", [index, value])
            self.to_be_replaced = (self.to_be_replaced + 1) % self.capacity
            self.stored_memory = min(self.stored_memory + 1, self.capacity)
    
    def retrieve(self, query, input_weight=1.0):
        if self.stored_memory == 0 or not self.retrieving:
            return torch.zeros(query.shape[0], self.value_dim).to(self.device)
        # values = self.values.detach().clone()
        similarity = self.similarity_measure(query, self.values, input_weight)
        self.write(similarity, "similarity")
        if self.random_recall:
            retrieved_idx = torch.tensor([Categorical(torch.abs(similarity[i])).sample() for i in range(similarity.shape[0])])
        else:
            retrieved_idx = torch.argmax(similarity, dim=-1)
        retrieved_memory = F.one_hot(retrieved_idx, self.capacity).float()
        # print(retrieved_memory.shape, self.values.shape)
        return torch.bmm(torch.unsqueeze(retrieved_memory, dim=1), self.values).squeeze(1)

    def get_vals(self):
        return self.values.detach().cpu().numpy()

