import numpy as np
import torch


class ReplayBuffer:
    def __init__(self):
        """
        Initialize the buffer.
        """
        self.reset()

    def reset(self):
        """
        Reset the rollout.
        """
        pass

    def push(self, **kwargs):
        """
        Push data.
        """
        for key, value in kwargs.items():
            if key in self.rollout:
                self.rollout[key].append(value)
            else:
                raise KeyError(f"Key {key} not found in replay buffer.")
    
    def pull(self, *keys):
        """
        Pull data according to keys.
        """
        return [self.rollout[key] for key in keys]
    
    def reformat(self):
        """
        Reformat rollout data.
        """
        pass
