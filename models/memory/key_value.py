import torch
import torch.nn as nn

from ..basic_module import BasicModule


class KeyValueMemory(BasicModule):
    def __init__(self, key_dim: int, value_dim: int, capacity: int, device: str = 'cpu') -> None:
        super().__init__()
        self.device = device
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.capacity = capacity

        self.keys = torch.zeros((capacity, key_dim)).to(self.device)
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
        self.keys = torch.zeros((self.capacity, self.key_dim)).to(self.device)
        self.values = torch.zeros((self.capacity, self.value_dim)).to(self.device)
        self.stored_memory = 0
        self.to_be_replaced = 0

    def encode(self, key_value):
        if self.encoding:
            key, value = key_value
            self.write(self.keys, "memory_keys")
            self.write(self.values, "memory_values")
            self.keys[self.to_be_replaced] = key
            self.values[self.to_be_replaced] = value
            self.to_be_replaced = (self.to_be_replaced + 1) % self.capacity
            self.stored_memory = min(self.stored_memory + 1, self.capacity)
    
    def retrieve(self, query, input_weight=1.0):
        if self.stored_memory == 0 or not self.retrieving:
            return torch.zeros(self.value_dim).to(self.device)
        keys = self.keys.detach().clone()
        values = self.values.detach().clone()
        similarity = self.similarity_measure(query, keys, input_weight)
        return torch.matmul(similarity, values)
        
    def get_vals(self):
        return (self.keys.detach().cpu().numpy(), self.values.detach().cpu().numpy())
