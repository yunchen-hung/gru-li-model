import math
import torch
import torch.nn as nn

from .utils import make_weights
from models.base_module import BasicModule


class LCA(BasicModule):
    """
    input_size : int
        the number of accumulators in the LCA
    leak : float
        the leak term
    lateral_inhibition : float
        the lateral inhibition across accumulators (i vs. j)
    dt_div_tau : float
        the dt / tao term, representing the time step size
    self_excitation : float
        the self excitation of a accumulator (i vs. i)
    input_weight : float
        input strengh of the feedforward weights
    cross_weight : float
        cross talk of the feedforward weights
    offset : float
        the additive drift term of the LCA process
    noise_std : float
        the sd of the noise term of the LCA process
    """
    def __init__(self, input_size, leak=0, lateral_inhibition=0.8, dt_div_tau=0.6, self_excitation=0, input_weight=1, cross_weight=0, 
    offset=0, noise_std=0, threshold=1, device: str = 'cpu'):
        super().__init__()
        self.device = device

        self.input_size = input_size
        self.leak = leak
        self.lateral_inhibition = lateral_inhibition
        self.alpha = dt_div_tau
        self.self_excitation = self_excitation
        self.input_weight = input_weight
        self.cross_weight = cross_weight
        self.offset = offset
        self.noise_std = noise_std
        self.threshold = threshold

        # the input / recurrent weights
        self.W_i = make_weights(input_weight, cross_weight, input_size, device)
        self.W_r = make_weights(self_excitation, -lateral_inhibition, input_size, device)
    
    def forward(self, inp, input_weight=1.0):
        """
        inp: timestep * memory_capacity(input_size)
        """
        self.W_i = make_weights(input_weight, self.cross_weight, self.input_size, self.device)

        inp = inp.to(self.device)
        timestep = inp.shape[0]
        inp = torch.matmul(inp, self.W_i)
        offset = self.offset * torch.ones(self.input_size, device=self.device)
        noise = (torch.randn(size=(timestep, self.input_size)) * self.noise_std * math.sqrt(self.alpha)).to(self.device)
        w = torch.zeros((timestep, self.input_size)).to(self.device)
        for t in range(timestep):
            w_prev = w[max(t-1, 0)]
            # print(offset.device, inp.device, self.W_r.device, w.device, noise.device)
            w[t] = w_prev + offset + (inp[t] - self.leak * w_prev + torch.matmul(self.W_r, w_prev)) * self.alpha + noise[t]
            w[t] = torch.min(w[t].relu(), torch.ones(self.input_size, device=self.device) * self.threshold)
        return w
