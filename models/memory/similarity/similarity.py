import torch.nn as nn
import torch.nn.functional as F

from .lca import LCA

SIMILARITY_MEASURES = ['cosine', 'l1', 'l2']


class BasicSimilarity(nn.Module):
    def __init__(self, similarity_measure='cosine', device: str = 'cpu'):
        super().__init__()
        self.measure = similarity_measure
        self.device = device

    def forward(self, query, values):
        if self.measure == 'cosine':
            similarities = F.cosine_similarity(query.to(self.device), values.to(self.device))
        elif self.measure == 'l1':
            similarities = - F.pairwise_distance(query.to(self.device), values.to(self.device), p=1)
        elif self.measure == 'l2':
            similarities = - F.pairwise_distance(query.to(self.device), values.to(self.device), p=2)
        else:
            raise Exception(f'Unrecognizable self.measure: {self.measure}')
        return similarities


class LCASimilarity(nn.Module):
    def __init__(self, lca_cycles, input_size, similarity_measure='cosine', leak=0, lateral_inhibition=0.8, dt_div_tau=0.6, self_excitation=0, input_weight=1, cross_weight=0, 
    offset=0, noise_std=0, threshold=1, device: str = 'cpu'):
        super().__init__()
        self.measure = BasicSimilarity(similarity_measure, device)
        self.lca = LCA(input_size, leak, lateral_inhibition, dt_div_tau, self_excitation, input_weight, cross_weight, offset, noise_std, threshold, device)
        self.lca_cycles = lca_cycles
        self.device = device

    def forward(self, query, values, input_weight=1.0):
        similarities = self.measure(query, values)
        lcas = self.lca(similarities.repeat(self.lca_cycles, 1), input_weight)
        return lcas[-1]
