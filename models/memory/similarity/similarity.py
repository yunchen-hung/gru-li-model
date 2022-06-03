import torch
import torch.nn as nn
import torch.nn.functional as F

from .lca import LCA
from models.basic_module import BasicModule

SIMILARITY_MEASURES = ['cosine', 'l1', 'l2']


class BasicSimilarity(BasicModule):
    def __init__(self, similarity_measure='cosine', find_max=False, device: str = 'cpu'):
        super().__init__()
        self.measure = similarity_measure
        self.find_max = find_max
        self.device = device

    def forward(self, query, values, input_weight=1.0):
        if self.measure == 'cosine':
            similarities = F.cosine_similarity(query.to(self.device), values.to(self.device))
        elif self.measure == 'l1':
            similarities = - F.pairwise_distance(query.to(self.device), values.to(self.device), p=1)
        elif self.measure == 'l2':
            similarities = - F.pairwise_distance(query.to(self.device), values.to(self.device), p=2)
        else:
            raise Exception(f'Unrecognizable self.measure: {self.measure}')
        self.write(similarities, 'similarities')
        if self.find_max:
            similarities = F.one_hot(torch.argmax(similarities, dim=-1), num_classes=similarities.shape[-1]).to(self.device)
            self.write(similarities, 'max_similarities')
        return similarities * input_weight


class LCASimilarity(BasicModule):
    def __init__(self, lca_cycles, input_size, similarity_measure='cosine', leak=0, lateral_inhibition=0.8, dt_div_tau=0.6, self_excitation=0, input_weight=1, cross_weight=0, 
    offset=0, noise_std=0, threshold=1, device: str = 'cpu'):
        super().__init__()
        self.measure = BasicSimilarity(similarity_measure, device)
        self.lca = LCA(input_size, leak, lateral_inhibition, dt_div_tau, self_excitation, input_weight, cross_weight, offset, noise_std, threshold, device)
        self.lca_cycles = lca_cycles
        self.device = device

    def forward(self, query, values, input_weight=1.0):
        similarities = self.measure(query.to(self.device), values.to(self.device))
        lcas = self.lca(similarities.repeat(self.lca_cycles, 1), input_weight)
        
        self.write(similarities, 'similarities')
        self.write(lcas, 'lcas')

        return lcas[-1]
