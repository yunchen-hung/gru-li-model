import math
import itertools
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .base import BaseEMTask


class ConditionalQuestionAnswer(BaseEMTask):
    def __init__(self, num_features=4, feature_dim=2, sequence_len=8, retrieve_time_limit=None, 
                 correct_reward=1.0, wrong_reward=-1.0, no_action_reward=0.0, cumulated_gt=False,
                 include_question_during_encode=False, reset_state_before_test=True, one_hot_stimuli=False,
                 no_early_stop=False):
        """
        During encoding phase, give a sequence of stimuli, each stimuli contains a number of features, 
            each stimuli is different from each other.
        During recall phase, give a question that can be answered within 1 time step, 
            e.g. given the value of the 1st feature as x, or given the max value of all the features as x, 
                answer the sum of the 2nd feature of stimuli that match the condition.
            (Right now we only make the question to be "sum" here.)
            
        Parameters:
            num_features: number of features in one stimuli
            feature_dim: dimension of each feature
            sequence_len: length of the sequence (number of stimuli in one trial)
            rewards: correct, wrong, no_action
            retrieve_time_limit: maximum number of steps allowed in the recall phase
            reset_state_before_test: whether to reset the state of the network before testing
            include_question_during_encode: whether to give the question during encoding phase
            one_hot_stimuli: whether to convert stimuli to one-hot vector
        Observation space:
            stimuli: num_features * [feature_dim one-hot vector]
            question: 
                question_space1: the feature in the condition, dim = num_features
                question_space2: the feature to be summed up, dim = num_features
                question_value: the value within the question to be matched, dim = feature_dim
                    e.g. given the value of the 1st feature as x, x is the question_value
        Action space:
            feature_dim ^ seq_len (include all possible answers for sum) + 1 no_action dim, a one-hot vector overall
        """
        super().__init__(reset_state_before_test=reset_state_before_test)
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.sequence_len = sequence_len
        self.retrieve_time_limit = retrieve_time_limit if retrieve_time_limit is not None else sequence_len
        self.include_question_during_encode = include_question_during_encode
        self.one_hot_stimuli = one_hot_stimuli
        self.cumulated_gt = cumulated_gt
        self.no_early_stop = no_early_stop

        self.correct_reward = correct_reward
        self.wrong_reward = wrong_reward
        self.no_action_reward = no_action_reward

        self.question_space_dim = num_features

        if self.one_hot_stimuli:
            self.observation_space = spaces.MultiDiscrete([self.feature_dim ** self.num_features, 
                                                           self.question_space_dim, 
                                                           feature_dim, 
                                                           self.question_space_dim])
        else:
            self.observation_space = spaces.MultiDiscrete([feature_dim for _ in range(num_features)]
                                                        + [self.question_space_dim, 
                                                           feature_dim, 
                                                           self.question_space_dim])
        # self.action_space = spaces.Discrete((feature_dim-1) * sequence_len + 2)
        self.action_space = spaces.Discrete(3)
        
        self.all_stimuli = self._generate_all_stimuli()

    def reset(self, batch_size=1):
        """
        question_during_encode: whether to give the question during encoding phase, default: False
        """
        # generate a random sequence of stimuli from all_stimuli without replacement
        # memory_sequence: sequence_len * num_features, features represented as int
        # self.memory_sequence = self.all_stimuli[np.random.choice(len(self.all_stimuli), self.sequence_len, replace=False)]
        self.memory_sequence = self.all_stimuli[np.random.choice(len(self.all_stimuli), self.sequence_len, replace=True)]
        iter_num = 0
        while True:
            feature = np.random.choice(self.num_features, 2, replace=False)
            # the feature in the condition, the feature to be summed up
            self.question_feature, self.sum_feature = feature[0], feature[1]
            self.question_value = np.random.choice(self.feature_dim)
            iter_num += 1
            if np.sum(self.memory_sequence[:, self.question_feature] == self.question_value) > 0 or iter_num>10:
                break

        self.gt_by_timestep = np.zeros(self.sequence_len)
        cnt = 0
        self.answer = 0
        for i in range(self.sequence_len):
            if self.memory_sequence[i, self.question_feature] == self.question_value:
                if cnt == 0:
                    self.answer = self.memory_sequence[i, self.sum_feature]
                else:
                    self.answer = np.logical_xor(self.answer, self.memory_sequence[i, self.sum_feature])
                cnt += 1
            self.gt_by_timestep[i] = self.answer
        if cnt == 0:
            self.answer = 0
        else:
            self.answer = int(self.answer)
        self.cnt = cnt
        self.gt_by_timestep = self.gt_by_timestep.astype(int)

        # self.answer = np.logical_xor(self.memory_sequence[
        #     self.memory_sequence[:, self.question_feature] == self.question_value, self.sum_feature])

        self.phase = "encoding"     # encoding, recall
        self.timestep = 0

        # convert the first observation to concatenated one-hot vectors
        obs = self._generate_observation(self.memory_sequence[0], self.question_feature, self.question_value, 
                                         self.sum_feature, include_question=self.include_question_during_encode)
        info = {"phase": "encoding"}
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
            if self.timestep >= self.sequence_len:
                # first timestep of recall phase
                self.phase = "recall"
                self.timestep = 0
                obs = self._generate_observation(None, self.question_feature, self.question_value, self.sum_feature, 
                                                 include_question=True)
                info = {"phase": "recall", "reset_state": self.reset_state_before_test}
            else:
                # encoding phase
                obs = self._generate_observation(self.memory_sequence[self.timestep], self.question_feature, self.question_value,
                                                self.sum_feature, include_question=self.include_question_during_encode)
                info = {"phase": "encoding"}
            return obs, [0.0], np.array([False]), info
        elif self.phase == "recall":
            obs = self._generate_observation(None, self.question_feature, self.question_value, self.sum_feature, 
                                             include_question=True)
            info = {"phase": "recall"}

            if action == self.action_space.n - 1 or (self.no_early_stop and self.timestep < self.retrieve_time_limit):
                # not action
                reward = self.no_action_reward
            elif self.answer is None:
                reward = self.correct_reward / self.feature_dim
            elif action == self.answer:
                reward = self.correct_reward
            else:
                reward = self.wrong_reward

            if (not self.no_early_stop and action != self.action_space.n - 1) or self.timestep >= self.retrieve_time_limit:
                done = True
            else:
                done = False
            
            return obs, [reward], np.array([done]), info

    def render(self, mode='human'):
        print("memory sequence:", self.memory_sequence)
        print("question feature:", self.question_feature)
        print("question value:", self.question_value)
        print("sum feature:", self.sum_feature) 
        print("answer:", self.answer)
    
    def compute_accuracy(self, actions):
        """
        compute the accuracy of a sequence of actions during recall phase
        """
        correct_actions = 0
        wrong_actions = 0
        no_actions = 0

        if actions[-1] == self.action_space.n - 1:
            # didn't answer at the end
            no_actions += 1
        elif self.answer is None:
            correct_actions += 1
        elif actions[-1] == self.answer:
            correct_actions += 1
        else:
            wrong_actions += 1

        return correct_actions, wrong_actions, no_actions
    
    def get_ground_truth(self, phase='recall'):
        """
        get expected actions of a trial
        """
        answer = self.answer if self.answer is not None else np.random.choice(self.feature_dim)

        if self.include_question_during_encode:
            if self.cumulated_gt:
                gt_enc = self.gt_by_timestep
            else:
                gt_enc = np.array([self.memory_sequence[i, self.sum_feature] if self.memory_sequence[i, self.question_feature] == self.question_value \
                    else self.action_space.n-1 for i in range(self.sequence_len)])
        else:
            gt_enc = np.array([self.action_space.n-1]*(self.sequence_len))

        gt_rec = np.array([self.action_space.n-1]*(self.sequence_len-1)+[answer])

        gt = np.concatenate((gt_enc, gt_rec))

        if phase == 'encoding':
            mask = np.concatenate((np.ones(self.sequence_len), np.zeros(self.sequence_len))).astype(int)
        elif phase == 'recall':
            mask = np.concatenate((np.zeros(self.sequence_len), np.ones(self.sequence_len))).astype(int)
        elif phase == 'last':
            mask = np.concatenate((np.zeros(self.sequence_len*2-1), np.ones(1))).astype(int)
        else:
            mask = np.ones(self.sequence_len*2).astype(int)

        return gt.reshape(1, -1), mask.reshape(1, -1)

    def get_trial_data(self):
        """
        get trial data, including memory sequence, question type, question value, and correct answers
        """
        num_matched_stimuli, correct_answers_index = self.get_matched_stimuli()
        return {"memory_sequence": self.memory_sequence, "question_feature": self.question_feature, 
                "question_value": self.question_value, "sum_feature": self.sum_feature,
                "correct_answer": self.answer,
                "num_matched_stimuli": num_matched_stimuli, "matched_stimuli_index": correct_answers_index,
                "memory_sequence_int": np.array([self.convert_stimuli_to_int(m) for m in self.memory_sequence])}
    
    def get_matched_stimuli(self):
        """
        get the number and index of stimuli matching the question
        """
        num_matched_stimuli = np.sum(self.memory_sequence[:, self.question_feature] == self.question_value)
        correct_answers_index = np.where(self.memory_sequence[:, self.question_feature] == self.question_value)[0]
        return num_matched_stimuli, correct_answers_index
        
    def convert_stimuli_to_int(self, stimuli):
        """
        converte the multi-discrete stimuli to integer
        """
        return np.sum([stimuli[i] * self.feature_dim ** i for i in range(self.num_features)])
    
    def _generate_all_stimuli(self):
        """
        generate all possible stimuli, in the format of a list of numbers with length num_features, the value of each number is within [0, feature_dim)
        """
        all_stimuli = list(itertools.product(range(self.feature_dim), repeat=self.num_features))
        all_stimuli = np.array([np.array(stimuli) for stimuli in all_stimuli])
        return all_stimuli

    def _generate_observation(self, stimuli, question_feature, question_value, sum_feature, include_question=False):
        """
        convert observation vector to concatenated one-hot vector
        stimuli: a list of numbers with length num_features, the value of each number is within [0, feature_dim)
            if None, no stimuli is presented
        question_feature: the index of the question in the question space
        question_value: the value within the question, dim = feature_dim
            e.g. given the value of the 1st feature as x, x is the question_value
        sum_feature: the index of the feature to be summed up
        include_question: whether to include the question in the observation
        """
        observation = np.zeros(np.sum(self.observation_space.nvec))
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
            observation[question_offset+question_feature] = 1
            observation[question_offset+self.question_space_dim+question_value] = 1
            observation[question_offset+self.question_space_dim+self.feature_dim+sum_feature] = 1
        return observation.reshape(1, -1)
