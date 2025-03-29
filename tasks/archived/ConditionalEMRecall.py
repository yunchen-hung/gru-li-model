import math
import itertools
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .base import BaseEMTask


class ConditionalEMRecall(BaseEMTask):
    def __init__(self, num_features=2, feature_dim=2, vocabulary_num=50, value_dim=2,
                 sequence_len=8, retrieve_time_limit=None,
                 correct_reward=1.0, wrong_reward=-1.0, no_action_reward=0.0, 
                 include_question_during_encode=False, reset_state_before_test=True,
                 no_condition=False,
                 seed=None):
        """
        During encoding phase, give a sequence of stimuli, each stimuli contains a number of features and an identity, 
            each stimuli is different from each other.
        During recall phase, give a condition, e.g. given the value of the 1st feature as x,
            ask the agent to recall all identity of stimuli matching the question.

        Parameters:
            no_condition: if True, always use the first feature = 1 as the condition (the first feature is always 1),
                if False, randomly select a feature that's not the first feature as the condition
            
        Observation space:
            feature: each feature is a one-hot vector of length feature_dim
            identity: a one-hot vector of length vocabulary_num
            value: a one-hot vector of length value_dim
            question:
                feature: a one-hot vector of length num_features
                feature value: a one-hot vector of length feature_dim
        Action space:
            a one-hot vector of length vocabulary_num + 1, where the last action is "no action"
        """
        super().__init__(reset_state_before_test=reset_state_before_test, seed=seed)
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.vocabulary_num = vocabulary_num
        self.value_dim = value_dim
        self.sequence_len = sequence_len
        self.retrieve_time_limit = retrieve_time_limit if retrieve_time_limit is not None else sequence_len
        self.include_question_during_encode = include_question_during_encode
        self.no_condition = no_condition

        self.correct_reward = correct_reward
        self.wrong_reward = wrong_reward
        self.no_action_reward = no_action_reward

        obs_space_list = [feature_dim for _ in range(num_features)]
        obs_space_list.extend([vocabulary_num, value_dim])
        obs_space_list.extend([num_features, feature_dim])  # question
        self.obs_shape = np.sum(obs_space_list)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_shape,), dtype=np.float32)
        self.action_space = spaces.Discrete(vocabulary_num + 1)


    def reset(self):
        """
        question_during_encode: whether to give the question during encoding phase, default: False
        """
        self.features = np.zeros((self.sequence_len, self.num_features))
        self.features[:, 0] = 1
        self.features[:, 1:] = np.random.randint(0, self.feature_dim, (self.sequence_len, self.num_features - 1))

        self.items = np.random.choice(self.vocabulary_num, self.sequence_len, replace=False)

        self.values = np.random.randint(0, self.value_dim, self.sequence_len)

        if self.no_condition:
            self.question_feature = 0
            self.question_value = 1
        else:
            self.question_feature = np.random.randint(1, self.num_features)
            self.question_value = np.random.randint(0, self.feature_dim)

        self.matched_index = np.where(self.features[:, self.question_feature] == self.question_value)[0]
        self.matched_items = self.items[self.matched_index]
        self.matched_values = self.values[self.matched_index]

        self.timestep = 0
        self.phase = "encoding"
        self.recalled = np.zeros(self.vocabulary_num, dtype=bool)

        if self.include_question_during_encode:
            obs = self._generate_observation(self.features[self.timestep], self.items[self.timestep], self.values[self.timestep], 
                                             self.question_feature, self.question_value)
        else:
            obs = self._generate_observation(self.features[self.timestep], self.items[self.timestep], self.values[self.timestep])
        info = {"phase": "encoding", "done": False}
        return obs, info

        
    def step(self, action):
        """
        action: a MultiDiscrete vector of length num_features
        """
        try:
            action = action[0].item()
        except:
            pass

        self.timestep += 1
        if self.phase == "encoding":
            if self.timestep < self.sequence_len:
                if self.include_question_during_encode:
                    obs = self._generate_observation(self.features[self.timestep], self.items[self.timestep], self.values[self.timestep], 
                                                     self.question_feature, self.question_value)
                else:
                    obs = self._generate_observation(self.features[self.timestep], self.items[self.timestep], self.values[self.timestep])
                reward = 0
                done = False
                gt = self.items[self.timestep-1] if self.items[self.timestep-1] in self.matched_items else self.vocabulary_num
                info = {"phase": "encoding", "done": False,
                        "gt": gt, "gt_mask": True, "loss_mask": True, 
                        "correct": 0, "wrong": 0, "not_know": 0}
                return obs, reward, False, False, info
            else:
                self.phase = "recall"
                self.timestep = 0
        if self.phase == "recall":
            if self.timestep >= self.retrieve_time_limit:
                done = True
            else:
                done = False

            gt = self.items[self.timestep-1] if self.items[self.timestep-1] in self.matched_items else self.vocabulary_num

            obs = self._generate_observation(None, None, None, self.question_feature, self.question_value)

            info = {"phase": "recall", "done": done,
                    "gt": gt, "gt_mask": False, "loss_mask": True, 
                    "correct": 0, "wrong": 0, "not_know": 0}
            
            if action == self.vocabulary_num:
                reward = self.no_action_reward
                info["not_know"] = 1
            elif action in self.matched_items and not self.recalled[action]:
                reward = self.correct_reward
                self.recalled[action] = 1
                info["correct"] = 1
            else:
                reward = self.wrong_reward
                info["wrong"] = 1

            return obs, reward, False, False, info
        

    def render(self, mode='human'):
        print("features:", self.features)
        print("items:", self.items)
        print("values:", self.values)
        print("question_feature:", self.question_feature)
        print("question_value:", self.question_value)
        print("matched_index:", self.matched_index)


    def compute_accuracy(self, actions):
        """
        compute the accuracy of a sequence of actions during recall phase
        """
        recalled = np.zeros(self.vocabulary_num, dtype=bool)

        correct, wrong, not_know = 0, 0, 0

        for action in actions:
            if action == self.vocabulary_num:
                not_know += 1
            elif action in self.matched_items and not recalled[action]:
                recalled[action] = 1
                correct += 1
            else:
                wrong += 1
        return correct, wrong, not_know
    
    def get_ground_truth(self, phase='recall'):
        """
        get expected actions of a trial
        """
        if self.include_question_during_encode:
            gt_enc = np.array([self.items[i] if self.items[i] in self.matched_items else self.vocabulary_num for i in range(self.sequence_len)])
        else:
            gt_enc = np.array([self.vocabulary_num] * self.sequence_len)

        if self.retrieve_time_limit < self.sequence_len:
            gt_rec = np.array([self.items[i] if self.items[i] in self.matched_items else self.vocabulary_num for i in range(self.retrieve_time_limit)])
        else:
            gt_rec = np.array([self.items[i] if self.items[i] in self.matched_items else self.vocabulary_num for i in range(self.sequence_len)])
            gt_rec2 = np.ones(self.sequence_len-self.retrieve_time_limit) * self.vocabulary_num
            gt_rec = np.concatenate((gt_rec, gt_rec2))

        gt = np.concatenate((gt_enc, gt_rec))

        if phase == 'encoding':
            mask = np.concatenate((np.ones(self.sequence_len), np.zeros(self.retrieve_time_limit))).astype(int)
        elif phase == 'recall':
            mask = np.concatenate((np.zeros(self.sequence_len), np.ones(self.retrieve_time_limit))).astype(int)
        else:
            mask = np.ones(self.sequence_len+self.retrieve_time_limit).astype(int)

        return gt.reshape(1, -1), mask.reshape(1, -1)

    def get_trial_data(self):
        """
        get trial data, including memory sequence, question type, question value, and correct answers
        """
        return {"features": self.features, "items": self.items, "values": self.values, 
                "question_feature": self.question_feature, "question_value": self.question_value,
                "matched_index": self.matched_index, "matched_items": self.matched_items, 
                "matched_values": self.matched_values}

    
    def _generate_observation(self, features=None, item=None, value=None, question_feature=None, question_value=None):
        """
        if features, item and value are None, then do not include stimuli (set to all zeros)
        if question_feature and question_value are None, then do not include question (set to all zeros)
        """
        stimuli = np.zeros(self.obs_shape)
        offset = 0
        if features is not None:
            for i in range(self.num_features):
                stimuli[int(i * self.feature_dim + features[i])] = 1
        offset += self.num_features * self.feature_dim

        if item is not None:
            stimuli[int(offset + item)] = 1
        offset += self.vocabulary_num

        if value is not None:
            stimuli[int(offset + value)] = 1
        offset += self.value_dim

        if question_feature is not None and question_value is not None:
            stimuli[int(offset + question_feature)] = 1
            offset += self.num_features
            stimuli[int(offset + question_value)] = 1
        return stimuli
