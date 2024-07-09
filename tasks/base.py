import math
import numpy as np
import gymnasium as gym


class BaseEMTask(gym.Env):
    def __init__(self, reset_state_before_test=True, seed=None):
        super().__init__()
        self.reset_state_before_test = reset_state_before_test
        if seed is not None:
            np.random.seed(seed)

    def compute_accuracy(self, actions):
        """
        given action sequence, compute the accuracy of current trial
        """
        raise NotImplementedError
    
    def get_ground_truth(self, phase="recall"):
        """
        return the ground truth for the current trial
        """
        raise NotImplementedError
    
    def get_trial_data(self):
        """
        used when recording data
        """
        raise NotImplementedError
    
    def convert_action_to_observation(self, action):
        """
        convert action to observation, default is one-hot vector
        """
        return np.eye(self.action_space.n)[action]
    
