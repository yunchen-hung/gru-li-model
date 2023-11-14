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
            origin_similarities = F.cosine_similarity(query.to(self.device), values.permute(1, 0, 2).to(self.device), dim=-1).permute(1, 0)
        elif self.measure == 'l1':
            origin_similarities = - F.pairwise_distance(query.to(self.device), values.permute(1, 0, 2).to(self.device), p=1, dim=-1).permute(1, 0)
        elif self.measure == 'l2':
            origin_similarities = - F.pairwise_distance(query.to(self.device), values.permute(1, 0, 2).to(self.device), p=2, dim=-1).permute(1, 0)
        elif self.measure == 'inner_product':
            origin_similarities = torch.bmm(F.normalize(values, p=2, dim=2), F.normalize(torch.unsqueeze(query, dim=2), p=2)).squeeze(2)
        else:
            raise Exception(f'Unrecognizable similarity measure: {self.measure}')
        self.write(origin_similarities, 'raw_similarity')
        if self.process_similarity == 'normalize':
            # normalize
            similarities = origin_similarities / torch.sum(origin_similarities, dim=-1, keepdim=True)
        elif self.process_similarity == 'softmax':
            beta = self.softmax_temperature if beta is None else beta
            similarities = softmax(origin_similarities, beta=beta)
        elif self.process_similarity == 'none':
            pass
        return similarities * input_weight, softmax(origin_similarities, beta=1.0)


class LCASimilarity(BasicModule):
    def __init__(self, lca_cycles, input_size, similarity_measure='cosine', leak=0, lateral_inhibition=0.8, 
                 dt_div_tau=0.6, self_excitation=0, input_weight=1, cross_weight=0, process_similarity='softmax',
                 softmax_beta=0.2, offset=0, noise_std=0, threshold=1, device: str = 'cpu'):
        super().__init__()
        self.measure = BasicSimilarity(similarity_measure, device)
        self.lca = LCA(input_size, leak, lateral_inhibition, dt_div_tau, self_excitation, input_weight, cross_weight, offset, noise_std, threshold, device)
        self.lca_cycles = lca_cycles
        self.process_similarity = process_similarity
        self.softmax_beta = softmax_beta
        self.device = device

    def forward(self, query, values, input_weight=1.0, beta=None):
        # similarities = self.measure(query.to(self.device), values.to(self.device))
        similarities = F.cosine_similarity(query.to(self.device), values.permute(1, 0, 2).to(self.device), dim=-1).permute(1, 0)
        lcas = self.lca(similarities.repeat(self.lca_cycles, 1), input_weight)
        lca = lcas[-1]
        lca_out = lca.reshape(1, -1)
        if self.process_similarity == 'softmax':
            beta = self.softmax_beta if beta is None else beta
            # TODO: why softmax function cannot work here???
            if torch.sum(lca_out) != 0:
                lca_out = lca_out / torch.sum(lca_out)
            lca_out = torch.exp(lca_out / beta) / torch.sum(torch.exp(lca_out / beta))
        
        self.write(similarities, 'similarities')
        # self.write(lcas, 'lcas')
        self.write(lca_out, 'lca')

        return lca_out, torch.exp(similarities) / torch.sum(torch.exp(similarities))
