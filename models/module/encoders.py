import torch.nn as nn

from ..base_module import BasicModule


class MLPEncoder(BasicModule):
    def __init__(self, input_dim, output_dim, hidden_dims=[]):
        super(MLPEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        self.encoder = nn.Sequential()
        last_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            self.encoder.add_module(f'fc{i}', nn.Linear(last_dim, hidden_dim))
            self.encoder.add_module(f'relu{i}', nn.ReLU())
            last_dim = hidden_dim
        self.encoder.add_module('fc_out', nn.Linear(last_dim, output_dim))

    def forward(self, x):
        return self.encoder(x)
