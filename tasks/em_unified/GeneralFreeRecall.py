import math
import itertools
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .base import BaseGeneralEMTask


class GeneralFreeRecall(BaseGeneralEMTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def reset(self):
        obs, info = super().reset()
        self._generate_task_condition()
        return obs, info

    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        if self.phase == "recall":
            reward, correct, wrong, not_know = self._compute_reward_and_update_metrics(action)
            info["correct"] = correct
            info["wrong"] = wrong
            info["not_know"] = not_know
            info["done"] = done
        return obs, reward, done, False, info


        self.condition_feature = None
        self.condition_value = None
        self.query_feature = None
        self.answer = None
        self.num_matched_item = self.sequence_len

    
    def _compute_reward_and_update_metrics(self, action):
        action_item = self._convert_action_to_item(action)
        correct, wrong, not_know = 0, 0, 0
        if action_item in self.memory_sequence:
            correct += 1
            reward = self.correct_reward
        elif action_item == -1:
            not_know += 1
            reward = self.no_action_reward
        else:
            wrong += 1
            reward = self.wrong_reward
        return reward, correct, wrong, not_know


    def _generate_gt(self):
        if self.action_space_type == "feature_wise":
            return self.memory_sequence[self.timestep]
        elif self.action_space_type == "task_wise":
            return self._convert_item_to_int(self.memory_sequence[self.timestep])