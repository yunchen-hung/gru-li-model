import math
import itertools
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .base import BaseEMTask



class FreeRecall2(BaseEMTask):
    """
    repeat recall has no penalty
    there's a final penalty for not recalled items
    the trial ends when all items are recalled
    """
    def __init__(self, repeat_reward=0.0, final_penalty=-1.0, **kwargs):
        """
        recommend reward: correct = 1, wrong = -1, no_action = 0, repeat = 0
        """
        super().__init__(**kwargs)
        self.repeat_reward = repeat_reward if repeat_reward is not None else self.correct_reward
        self.final_penalty = final_penalty if final_penalty is not None else self.wrong_reward


    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        # self._generate_task_condition()
        self.retrieved = np.zeros(self.vocabulary_size, dtype=bool)
        self.matched_item_indexes = np.arange(self.sequence_len)
        self.num_retrieved = 0
        return obs, info

    
    def step(self, action):
        obs, reward, done, _, info = super().step(action)
        if self.num_retrieved == self.sequence_len:
            info['done'] = True
            self.done = True
        return obs, reward, done, False, info
    

    def compute_accuracy(self, actions):
        corrects, wrongs, not_knows = 0, 0, 0

        retrieved = np.ones(self.vocabulary_size, dtype=bool)
        for action in actions:
            action_item = self._convert_action_to_item(action)
            action_item_int = self._convert_item_to_int(action_item)
            if action_item_int < 0:
                not_knows += 1
            elif action_item in self.memory_sequence:
                if retrieved[action_item_int]:
                    not_knows += 1
                else:
                    corrects += 1
                    retrieved[action_item_int] = True
            else:
                wrongs += 1
        return corrects, wrongs, not_knows

    
    def _compute_reward_and_metrics(self, action):
        if self.done:
            return 0.0, 0, 0, 0
        action_item = self._convert_action_to_item(action)
        action_item_int = self._convert_item_to_int(action_item.reshape(1, -1))
        correct, wrong, not_know = 0, 0, 0
        if action_item_int in self.memory_sequence_int:
            if self.retrieved[action_item_int]:
                reward = self.repeat_reward
                not_know += 1
            else:
                correct += 1
                reward = self.correct_reward
                self.retrieved[action_item_int] = True
                self.num_retrieved += 1
        elif action_item_int < 0:
            not_know += 1
            reward = self.no_action_reward
        else:
            wrong += 1
            reward = self.wrong_reward

        if self.timestep >= self.retrieve_time_limit:
            reward += self.final_penalty * (self.sequence_len - self.num_retrieved)

        return reward, correct, wrong, not_know

        