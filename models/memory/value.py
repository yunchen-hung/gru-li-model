import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from ..base_module import BasicModule


class ValueMemory(BasicModule):
    def __init__(self, similarity_measure, value_dim: int, capacity: int, remove_retrieved=False,
                 recall_method="argmax", batch_size=1, noise_std=0.0, force_false_prob=0.0, 
                 transform_memory=False, different_transform=False, device: str = 'cpu') -> None:
        """
        remove_retrieved: if True, the previously retrieved memory will not be retrieved again
        transform_memory: if True, the memory is transformed by a linear layer before encoding and retrieval
        different_transform: if True, the transform is different for encoding and retrieval
        """
        super().__init__()
        self.device = device
        self.similarity_measure = similarity_measure
        self.value_dim = value_dim
        self.capacity = capacity
        self.noise_std = noise_std
        self.force_false_prob = force_false_prob
        self.remove_retrieved = remove_retrieved

        self.values = torch.zeros((batch_size, capacity, value_dim)).to(self.device)
        
        self.stored_memory = 0
        self.to_be_replaced = 0

        self.encoding = False
        self.retrieving = False

        self.not_retrieved = torch.ones((batch_size, capacity), device=self.device)

        self.recall_method = recall_method
        self.batch_size = batch_size

        self.transform_memory = transform_memory
        self.different_transform = different_transform

        self.fc_enc_mem = nn.Linear(value_dim, value_dim)
        self.fc_rec_mem = nn.Linear(value_dim, value_dim)

    def reset_memory(self, flush=True):
        if flush:
            self.flush()
        else:
            self.values.detach_()
        self.encoding = False
        self.retrieving = False
        self.not_retrieved = torch.ones((self.batch_size, self.capacity), device=self.device)

    def flush(self):
        self.values = torch.zeros((self.batch_size, self.capacity, self.value_dim)).to(self.device)
        self.stored_memory = 0
        # self.to_be_replaced = 0

    def encode(self, value):
        if self.encoding:
            if self.transform_memory:
                value = self.fc_enc_mem(value)
            self.write(value, "encoded_memory")
            # self.write(self.values, "memory_values")
            index = F.one_hot(torch.tensor(self.to_be_replaced), self.capacity).repeat(self.batch_size, 1).float().to(self.device)
            self.values = self.values + torch.einsum("ik,ij->ikj", [index, value])
            self.to_be_replaced = (self.to_be_replaced + 1) % self.capacity
            self.stored_memory = min(self.stored_memory + 1, self.capacity)
            # print(self.values[:, :10])
    
    def retrieve(self, query, input_weight=1.0, beta=None):
        if self.stored_memory == 0 or not self.retrieving:
            return torch.zeros(query.shape[0], self.value_dim).to(self.device)
        if self.transform_memory:
            if self.different_transform:
                query = self.fc_rec_mem(query)
            else:
                query = self.fc_enc_mem(query)
        # values = self.values.detach().clone()
        similarity, raw_similarity = self.similarity_measure(query, self.values, input_weight, beta, similarity_mask=self.not_retrieved)
        similarity = similarity + torch.randn_like(similarity) * self.noise_std
        self.write(raw_similarity, "raw_similarity")
        if self.recall_method == "random":
            retrieved_idx = torch.tensor([Categorical(torch.abs(similarity[i])).sample() for i in range(similarity.shape[0])])
            similarity = F.one_hot(retrieved_idx, self.capacity).float()
        elif self.recall_method == "argmax":
            retrieved_idx = torch.argmax(similarity, dim=-1)
            similarity = F.one_hot(retrieved_idx, self.capacity).float()
        elif self.recall_method == "weight_sum":
            retrieved_idx = torch.argmax(similarity, dim=-1)
        else:
            raise NotImplementedError
        # print(similarity.shape)

        if self.remove_retrieved:
            not_retrieved = torch.ones_like(similarity)
            not_retrieved[torch.arange(self.batch_size), retrieved_idx] = 0
            self.not_retrieved = not_retrieved * self.not_retrieved

        if self.force_false_prob > 0:
            force_false = np.random.uniform() < self.force_false_prob
            if force_false:
                retrieved_idx = torch.randint(0, self.capacity, (1,))
                similarity = F.one_hot(retrieved_idx, self.capacity).float()
                # print("force false", similarity.shape)
        self.write(similarity, "similarity")
        retrieved_memory = torch.bmm(torch.unsqueeze(similarity, dim=1), self.values).squeeze(1)
        self.write(retrieved_memory, "retrieved_memory")
        # print(retrieved_memory[:, :10], similarity)
        return retrieved_memory, raw_similarity

    def get_vals(self):
        return self.values.detach().to(self.device).numpy()

