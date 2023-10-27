import torch
import torch.nn as nn
import torch.nn.functional as F

from .lca import LCA
from models.base_module import BasicModule
from models.utils import softmax

SIMILARITY_MEASURES = ['cosine', 'l1', 'l2']


class BasicSimilarity(BasicModule):
    def __init__(self, similarity_measure='cosine', process_similarity='normalize', softmax_temperature=1.0, device: str = 'cpu'):
        super().__init__()
        self.measure = similarity_measure
        self.process_similarity = process_similarity
        self.softmax_temperature = softmax_temperature
        self.device = device

    def forward(self, query, values, input_weight=1.0, beta=None):
        if self.measure == 'cosine':
            # print(query.shape, values.shape)
            similarities = F.cosine_similarity(query.to(self.device), values.permute(1, 0, 2).to(self.device), dim=-1).permute(1, 0)
        elif self.measure == 'l1':
            similarities = - F.pairwise_distance(query.to(self.device), values.permute(1, 0, 2).to(self.device), p=1, dim=-1).permute(1, 0)
        elif self.measure == 'l2':
            similarities = - F.pairwise_distance(query.to(self.device), values.permute(1, 0, 2).to(self.device), p=2, dim=-1).permute(1, 0)
        elif self.measure == 'inner_product':
            similarities = torch.bmm(F.normalize(values, p=2, dim=2), F.normalize(torch.unsqueeze(query, dim=2), p=2)).squeeze(2)
        else:
            raise Exception(f'Unrecognizable similarity measure: {self.measure}')
        self.write(similarities, 'raw_similarity')
        if self.process_similarity == 'normalize':
            # normalize
            similarities = similarities / torch.sum(similarities, dim=-1, keepdim=True)
        elif self.process_similarity == 'softmax':
            beta = self.softmax_temperature if beta is None else beta
            similarities = softmax(similarities, beta=beta)
        elif self.process_similarity == 'none':
            pass
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
