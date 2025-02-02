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
                 answer_dim=2,                         # answer_dim: the dimension of the answer space
                 
                 one_hot_stimuli=True,
                 action_space_type="task_wise",      # action_space_type: feature_wise (an action for each feature), task_wise (an action for each task)

                 seed=None,
                 **kwargs):
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
            action_space_dim = self.num_features * self.feature_dim + self.answer_dim + 1 # +1 for no action
        elif self.action_space_type == "task_wise":
            action_space_dim = self.feature_dim ** self.num_features + 1 + self.feature_dim + 1 + self.answer_dim + 1 # free recall, recall feature, answer question

        self.action_space = spaces.Box(low=0, high=1, shape=(action_space_dim,), dtype=np.float32)

        self.all_stimuli = self._generate_all_stimuli()


    def reset(self):
        self.memory_sequence = self.all_stimuli[np.random.choice(len(self.all_stimuli), self.sequence_len, replace=True)]

        self.phase = "encoding"     # encoding, recall
        self.timestep = 0
        self.answered = False       # whether all matched items are recalled/the question is answered

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
            if self.timestep >= self.sequence_len:
                self.phase = "recall"
                self.timestep = 0
                info["phase"] = "recall"
                info["gt"] = self.memory_sequence[self.timestep]
                info["gt_mask"] = True
                obs = self._generate_observation(None, self.condition_feature, self.condition_value,
                                                    self.query_feature, include_question=True)
                return obs, 0.0, False, False, info
            else:
                obs = self._generate_observation(self.memory_sequence[self.timestep], self.condition_feature, self.condition_value,
                                                self.query_feature, include_question=self.include_question_during_encode)
                return obs, 0.0, False, False, info
        elif self.phase == "recall":
            obs = self._generate_observation(None, self.condition_feature, self.condition_value, self.query_feature, include_question=True)
            info = {"phase": "recall",
                    "gt": self.memory_sequence[self.timestep], "gt_mask": False,
                    "loss_mask": True,
                    "correct": 0, "wrong": 0, "not_know": 0,
                    "done": False}
            
            # for different task
            # compute the reward and the ground truth
            # update number of correct, wrong, and not_know
            # check if the trial is done

            done = False
            return obs, 0.0, done, False, info


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
        convert the item with multiple features to an integer
        item: shape (num_item, num_features)
        """
        return np.sum(item * (self.feature_dim ** np.arange(self.num_features)), axis=1).astype(int)

    
    def _convert_action_to_item(self, action):
        """
        convert the action to the item
        action: based on the action space type, the action is either a feature-wise action or a task-wise action
        item: shape (num_features), ignore the answer dimensions
        """
        if self.action_space_type == "feature_wise":
            item = np.zeros(self.num_features)
            for i in range(self.num_features):
                # get each feature value from the action
                item[i] = np.argmax(action[i*self.feature_dim:i*self.feature_dim+self.feature_dim])
        elif self.action_space_type == "task_wise":
            item = np.zeros(self.num_features)
            if action[self.feature_dim**self.num_features] == 1:
                # considered as no action
                item = -1
            else:
                action_int = np.argmax(action[:self.feature_dim**self.num_features])
                for i in range(self.num_features):
                    item[i] = action_int // (self.feature_dim**i) % self.feature_dim
        return item

    
    def _convert_action_to_feature(self, item):
        """
        convert the action to the query feature value
        action: based on the action space type, the action is either a feature-wise action or a task-wise action
        feature: the query feature
        """
        if self.action_space_type == "feature_wise":
            feature = np.argmax(action[self.query_feature*self.feature_dim:self.query_feature*self.feature_dim+self.feature_dim])
        elif self.action_space_type == "task_wise":
            feature = np.argmax(action[self.feature_dim**self.num_features:self.feature_dim**self.num_features+self.feature_dim])
        return feature

    
    def _convert_action_to_answer(self, action):
        """
        convert the action to the answer
        action: based on the action space type, the action is either a feature-wise action or a task-wise action
        answer: the answer
        """
        answer = np.argmax(action[-self.answer_dim:])
        return answer


    def _generate_task_condition(self):
        """
        generate condition_feature, condition_value, query_feature, answer, num_matched_item
        a different method for different task
        """
        raise NotImplementedError

    
    def _compute_reward(self, action):
        """
        check if the action is correct, return the reward
        """
        raise NotImplementedError

