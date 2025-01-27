import math
import itertools
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .base import BaseEMTask


class BaseGeneralEMTask(BaseEMTask):
    def __init__(self,
                 num_features=4,
                 feature_dim=2,
                 sequence_len=8,
                 retrieve_time_limit=None,
                 reset_state_before_test=True,

                 correct_reward=1.0,
                 wrong_reward=-1.0,
                 no_action_reward=0.0,

                 include_question_during_encode=False,  # whether to give the condition/question during encode
                 question_type="sum",                   # question_type: sum, xor
                 sum_reference=1,                       # sum_reference: the value to compare the sum with
                 answer_dim=3,                         # answer_dim: the dimension of the answer space
                 
                 one_hot_stimuli=True,
                 action_space_type="feature_wise",      # action_space_type: feature_wise (an action for each feature), task_wise (an action for each task)

                 seed=None):
        super().__init__(reset_state_before_test=reset_state_before_test, seed=seed)
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.sequence_len = sequence_len
        self.answer_dim = answer_dim
        self.retrieve_time_limit = retrieve_time_limit if retrieve_time_limit is not None else sequence_len
        
        self.include_question_during_encode = include_question_during_encode
        self.question_type = question_type
        self.sum_reference = sum_reference

        self.correct_reward = correct_reward
        self.wrong_reward = wrong_reward
        self.no_action_reward = no_action_reward

        self.action_space_type = action_space_type

        self.one_hot_stimuli = one_hot_stimuli
        if self.one_hot_stimuli:
            obs_space_dim = self.feature_dim ** self.num_features + self.num_features + self.feature_dim + self.num_features
        else:
            obs_space_dim = self.num_features * self.feature_dim + self.num_features + self.feature_dim + self.num_features
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_space_dim,), dtype=np.float32)

        action_space_dim = 0
        if self.action_space_type == "feature_wise":
            action_space_dim = self.num_features * self.feature_dim + self.answer_dim
        elif self.action_space_type == "task_wise":
            action_space_dim = self.feature_dim ** self.num_features + self.feature_dim + self.answer_dim # free recall, recall feature, answer question

        self.action_space = spaces.Box(low=0, high=1, shape=(action_space_dim,), dtype=np.float32)

        self.all_stimuli = self._generate_all_stimuli()


    def reset(self):
        self.memory_sequence = self.all_stimuli[np.random.choice(len(self.all_stimuli), self.sequence_len, replace=True)]

        if self.task_type == "free_recall":
            self.condition_feature = None
            self.condition_value = None
            self.query_feature = None
        elif self.task_type == "cond_free_recall":
            iter_num = 0
            while True:
                self.condition_feature = np.random.choice(self.num_features)
                self.condition_value = np.random.choice(self.feature_dim)
                if np.sum(self.memory_sequence[:, self.condition_feature] == self.condition_value) > 0 or iter_num>10:
                    break
            self.query_feature = None
        elif self.task_type == "cond_qa" or self.task_type == "cond_free_recall_feature":
            iter_num = 0
            while True:
                rand_feature = np.random.choice(self.num_features, 2, replace=False)
                self.condition_feature = rand_feature[0]
                self.condition_value = np.random.choice(self.feature_dim)
                self.query_feature = rand_feature[1]
                if np.sum(self.memory_sequence[:, self.condition_feature] == self.condition_value) > 0 or iter_num>10:
                    break
        else:
            raise ValueError(f"task_type {self.task_type} not supported")

        if self.task_type == "cond_qa":
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
        elif self.task_type == "cond_free_recall_feature" or self.task_type == "cond_free_recall":
            self.answer = None
            self.cnt = 0
            for i in range(self.sequence_len):
                if self.memory_sequence[i, self.query_feature] == self.condition_value:
                    self.cnt += 1
        elif self.task_type == "free_recall":
            self.answer = None
            self.cnt = self.sequence_len
        else:
            raise ValueError(f"task_type {self.task_type} not supported")

        self.phase = "encoding"     # encoding, recall
        self.timestep = 0
        self.answered = False

        # convert the first observation to concatenated one-hot vectors
        obs = self._generate_observation(self.memory_sequence[0], self.condition_feature, self.condition_value, 
                                         self.query_feature, include_question=self.include_question_during_encode)
        info = {"phase": "encoding"}
        return obs, info


    def step(self, action):
        try:
            action = action[0].item()
        except:
            pass

        self.timestep += 1
        if self.phase == "encoding":
            info = {"phase": "encoding", 
                    "gt": None, "gt_mask": False, 
                    "loss_mask": True, 
                    "correct": 0, "wrong": 0, "not_know": 0,
                    "done": False}
        elif self.phase == "recall":
            pass


    def _generate_all_stimuli(self):
        """
        generate all possible stimuli, in the format of a list of numbers with length num_features, the value of each number is within [0, feature_dim)
        """
        all_stimuli = list(itertools.product(range(self.feature_dim), repeat=self.num_features))
        all_stimuli = np.array([np.array(stimuli) for stimuli in all_stimuli])
        return all_stimuli


    def _generate_observation(self, stimuli, condition_feature, condition_value, query_feature, include_question=False):
        """
        generate the observation for the given stimulus
        """
        observation = np.zeros(self.obs_shape)
        if self.one_hot_stimuli:
            if stimuli is not None:
                stimuli_int = self.convert_stimuli_to_action(stimuli)
                observation[stimuli_int] = 1
            question_offset = self.feature_dim ** self.num_features
        else:
            if stimuli is not None:
                for i in range(self.num_features):
                    observation[i*self.feature_dim+stimuli[i]] = 1
            question_offset = self.num_features*self.feature_dim
        if include_question:
            if self.condition_feature is not None:
                observation[question_offset+condition_feature] = 1
            if self.condition_value is not None:
                observation[question_offset+self.num_features+condition_value] = 1
            if self.query_feature is not None:
                observation[question_offset+self.num_features+self.feature_dim+query_feature] = 1
        return observation


    def _convert_item_to_int(self, item):
        """
        convert the item to an integer
        item: shape (num_item, num_features)
        """
        return np.sum(item * (self.feature_dim ** np.arange(self.num_features)), axis=1).astype(int)


    def _generate_task_condition(self):
        """
        generate condition_feature, condition_value, query_feature, answer, num_matched_item
        a different method for different task
        """
        raise NotImplementedError

