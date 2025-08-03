import math
import itertools
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .base import BaseEMTask


class FreeRecall(BaseEMTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        # self._generate_task_condition()
        self.retrieved = np.zeros(self.vocabulary_size, dtype=bool)
        self.matched_item_indexes = np.arange(self.sequence_len)
        return obs, info

    
    def step(self, action):
        obs, reward, done, _, info = super().step(action)
        return obs, reward, done, False, info
    

    def compute_accuracy(self, actions):
        corrects, wrongs, not_knows = 0, 0, 0

        retrieved = np.ones(self.vocabulary_size, dtype=bool)
        for action in actions:
            action_item = self._convert_action_to_item(action)
            action_item_int = self._convert_item_to_int(action_item)
            if action_item_int < 0:
                not_knows += 1
            elif action_item in self.memory_sequence and not retrieved[action_item_int]:
                corrects += 1
                retrieved[action_item_int] = True
            else:
                wrongs += 1
        return corrects, wrongs, not_knows

    
    def _compute_reward_and_metrics(self, action):
        action_item = self._convert_action_to_item(action)
        action_item_int = self._convert_item_to_int(action_item.reshape(1, -1))
        correct, wrong, not_know = 0, 0, 0
        if action_item_int in self.memory_sequence_int and not self.retrieved[action_item_int]:
            correct += 1
            reward = self.correct_reward
            self.retrieved[action_item_int] = True
        elif action_item_int < 0:
            not_know += 1
            reward = self.no_action_reward
        else:
            wrong += 1
            reward = self.wrong_reward
        return reward, correct, wrong, not_know

        