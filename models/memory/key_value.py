import torch
import torch.nn as nn


class KeyValueMemory(nn.Module):
    def __init__(self, key_dim: int, value_dim: int, device: str = 'cpu') -> None:
        super().__init__()
        self.device = device
        self.key_dim = key_dim
        self.value_dim = value_dim
