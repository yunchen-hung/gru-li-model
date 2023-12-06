import math
import numpy as np
import gymnasium as gym


class BaseEMTask(gym.Env):
    def __init__(self, reset_state_before_test=True):
        super().__init__()
        self.reset_state_before_test = reset_state_before_test

    def compute_accuracy(self, actions):
        raise NotImplementedError
    
    def get_ground_truth(self, phase="recall"):
        raise NotImplementedError
    
