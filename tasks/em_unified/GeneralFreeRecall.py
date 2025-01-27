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


    def _generate_condition_features(self):
        self.condition_feature = None
        self.condition_value = None
        self.query_feature = None
        self.answer = None
        self.num_matched_item = self.sequence_len



