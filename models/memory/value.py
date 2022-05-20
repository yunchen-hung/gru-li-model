import torch
import torch.nn as nn


class ValueMemory(nn.Module):
    def __init__(self, similarity_measure, value_dim: int, capacity: int, device: str = 'cpu') -> None:
        super().__init__()
        self.device = device
        self.similarity_measure = similarity_measure
        self.value_dim = value_dim
        self.capacity = capacity

        self.values = torch.zeros((capacity, value_dim)).to(self.device)
        self.stored_memory = 0
        self.to_be_replaced = 0

        self.encoding = False
        self.retrieving = False

    def reset_memory(self):
        self.flush()
        self.encoding = False
        self.retrieving = False

    def flush(self):
        self.values = torch.zeros((self.capacity, self.value_dim)).to(self.device)
        self.stored_memory = 0
        self.to_be_replaced = 0

    def encode(self, value):
        if self.encoding:
            self.values[self.to_be_replaced] = value
            self.to_be_replaced = (self.to_be_replaced + 1) % self.capacity
            self.stored_memory = min(self.stored_memory + 1, self.capacity)
    
    def retrieve(self, query, input_weight=1.0):
        if self.stored_memory == 0 or not self.retrieving:
            return torch.zeros(self.value_dim)
        similarity = self.similarity_measure(query, self.values, input_weight)
        return torch.matmul(similarity, self.values)
        
    def get_vals(self):
        return self.values.detach().cpu().numpy()

