import math
import itertools
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .base import BaseGeneralEMTask


class GeneralConditionalQA(BaseGeneralEMTask):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def reset(self):
        obs, info = super().reset()
        self._generate_task_condition()
        return obs, info


    def _generate_condition_features(self):
        iter_num = 0
        while True:
            rand_feature = np.random.choice(self.num_features, 2, replace=False)
            self.condition_feature = rand_feature[0]
            self.condition_value = np.random.choice(self.feature_dim)
            self.query_feature = rand_feature[1]
            if np.sum(self.memory_sequence[:, self.condition_feature] == self.condition_value) > 0 or iter_num>10:
                break
        
        self.answer = 0
        self.cnt = 0
        for i in range(self.sequence_len):
            if self.memory_sequence[i, self.query_feature] == self.condition_value:
                if self.question_type == "xor":
                    self.answer = np.logical_xor(self.answer, self.memory_sequence[i, self.condition_feature])
                else:
                    self.answer += self.memory_sequence[i, self.condition_feature]
                self.cnt += 1
        if self.cnt == 0:
            self.answer = 0
        else:
            self.answer = int(self.answer)


