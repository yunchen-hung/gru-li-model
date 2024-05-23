import torch.nn as nn

from ..base_module import BasicModule


class MLPDecoder(BasicModule):
    def __init__(self, input_dim, output_dim, hidden_dims=[]):
        super(MLPDecoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        self.decoder = nn.Sequential()
        last_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            self.decoder.add_module(f'fc{i}', nn.Linear(last_dim, hidden_dim))
            self.decoder.add_module(f'relu{i}', nn.ReLU())
            last_dim = hidden_dim
        self.decoder.add_module('fc_out', nn.Linear(last_dim, output_dim))

    def forward(self, x):
        return self.decoder(x)
    

class ActorCriticMLPDecoder(BasicModule):
    def __init__(self, input_dim, output_dim, hidden_dims=[]):
        super(ActorCriticMLPDecoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

        self.actor_decoder = nn.Sequential()
        self.critic_decoder = nn.Sequential()
        last_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            self.actor_decoder.add_module(f'fc{i}', nn.Linear(last_dim, hidden_dim))
            self.actor_decoder.add_module(f'relu{i}', nn.ReLU())
            self.critic_decoder.add_module(f'fc{i}', nn.Linear(last_dim, hidden_dim))
            self.critic_decoder.add_module(f'relu{i}', nn.ReLU())
            last_dim = hidden_dim
        self.actor_decoder.add_module('fc_out', nn.Linear(last_dim, output_dim))
        self.critic_decoder.add_module('fc_out', nn.Linear(last_dim, 1))

    def forward(self, x):
        return self.actor_decoder(x), self.critic_decoder(x)
    
