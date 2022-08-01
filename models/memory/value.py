import torch
import torch.nn as nn

from ..basic_module import BasicModule


class ValueMemory(BasicModule):
    def __init__(self, similarity_measure, value_dim: int, capacity: int, device: str = 'cpu') -> None:
        super().__init__()
        self.device = device
        self.similarity_measure = similarity_measure
        self.value_dim = value_dim
        self.capacity = capacity

        self.values = torch.zeros((capacity, value_dim)).to(self.device)
        # self.values.detach_()
        # self.values = [torch.zeros((1, value_dim)) for _ in range(self.capacity)]
        
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
            # print("encode")
            # if len(self.values) > 0:
            #     values = torch.stack(self.values)
            #     self.write(values, "memory_values")
            # self.values.append(value)
            # if len(self.values) > self.capacity:
            #     self.values.pop(0)
            self.write(self.values, "memory_values")
            self.values[self.to_be_replaced] = value
            self.to_be_replaced = (self.to_be_replaced + 1) % self.capacity
            self.stored_memory = min(self.stored_memory + 1, self.capacity)
    
    def retrieve(self, query, input_weight=1.0):
        if self.stored_memory == 0 or not self.retrieving:
            return torch.zeros(query.shape[0], self.value_dim).to(self.device)
        # print("retrieve")
        # values = torch.stack(self.values).squeeze(dim=1)
        # print(values.shape)
        values = self.values.detach().clone()
        similarity = self.similarity_measure(query, values, input_weight)
        return torch.matmul(similarity, values)
        
    def get_vals(self):
        # return torch.stack(self.values).numpy()
        return self.values.detach().cpu().numpy()

