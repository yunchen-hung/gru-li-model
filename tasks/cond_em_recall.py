import math
import itertools
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ConditionalEMRecall(gym.Env):
    def __init__(self, num_features=2, feature_dim=5, sequence_len=8, correct_reward=1.0, wrong_reward=-1.0, no_action_reward=0.0,
                 retrieve_time_limit=None, include_question_during_encode=False, question_space=("choice", "max", "min")):
        """
        During encoding phase, give a sequence of stimuli, each stimuli contains a number of features, 
            each stimuli is different from each other.
        During recall phase, give a question, e.g. given the value of the 1st feature as x, or given the max value of all the features, 
            ask the agent to recall all stimuli matching the question.

        Parameters:
            num_features: number of features in one stimuli
            feature_dim: dimension of each feature
            sequence_len: length of the sequence (number of stimuli in one trial)
            rewards: correct, wrong, no_action. When all possible stimuli have been recalled, 
                the agent will receive a correct reward for taking an extra timestep "stop".
            retrieve_time_limit: maximum number of steps allowed in the recall phase
            question_space:
                choice: given one of the feature equals to a particular value
                max: given the max value of all the features
                min: given the min value of all the features
        Observation space:
            stimuli: num_features * [feature_dim one-hot vector]
            question: 
                question_space: the number of possible questions
                question_value: the value within the question, dim = feature_dim
                    e.g. given the value of the 1st feature as x, x is the question_value
        Action space:
            feature_dim ^ num_features + 1 no_action dim + 1 stop dim, a one-hot vector overall
        """
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.sequence_len = sequence_len
        self.retrieve_time_limit = retrieve_time_limit if retrieve_time_limit is not None else sequence_len
        self.include_question_during_encode = include_question_during_encode

        self.correct_reward = correct_reward
        self.wrong_reward = wrong_reward
        self.no_action_reward = no_action_reward

        self.question_space_dim = len(question_space)
        if "choice" in question_space:
            self.question_space_dim += num_features - 1
        self.question_space = question_space
        self.question_type_dict = self._generate_question_type_dict()

        self.observation_space = spaces.MultiDiscrete([feature_dim for _ in range(num_features)]+[self.question_space_dim, feature_dim])
        self.action_space = spaces.Discrete(feature_dim ** num_features + 2)
        
        self.all_stimuli = self._generate_all_stimuli()

    def reset(self):
        """
        question_during_encode: whether to give the question during encoding phase, default: False
        """
        # generate a random sequence of stimuli from all_stimuli without replacement
        self.memory_sequence = self.all_stimuli[np.random.choice(len(self.all_stimuli), self.sequence_len, replace=False)]  # sequence_len * num_features
        self.question_type = np.random.choice(self.question_space_dim)
        # generate question value, make sure the question value exists in at least one of the stimuli
        while True:
            self.question_value = np.random.choice(self.feature_dim)
            if self.question_type_dict[self.question_type] == "choice":
                if self.question_value in self.memory_sequence[:, self.question_type]:
                    break
            elif self.question_type_dict[self.question_type] == "max":
                if self.question_value in np.max(self.memory_sequence, axis=1):
                    break
            elif self.question_type_dict[self.question_type] == "min":
                if self.question_value in np.min(self.memory_sequence, axis=1):
                    break
            else:
                raise NotImplementedError

        # count the number of stimuli matching the questionm, and record the index of the correct answers
        if self.question_type_dict[self.question_type] == "choice":
            self.num_answers = np.sum(self.memory_sequence[:, self.question_type] == self.question_value)
            self.correct_answers_index = np.where(self.memory_sequence[:, self.question_type] == self.question_value)[0]
        elif self.question_type_dict[self.question_type] == "max":
            self.num_answers = np.sum(np.max(self.memory_sequence, axis=1) == self.question_value)
            self.correct_answers_index = np.where(np.max(self.memory_sequence, axis=1) == self.question_value)[0]
        elif self.question_type_dict[self.question_type] == "min":
            self.num_answers = np.sum(np.min(self.memory_sequence, axis=1) == self.question_value)
            self.correct_answers_index = np.where(np.min(self.memory_sequence, axis=1) == self.question_value)[0]
        else:
            raise NotImplementedError

        self.phase = "encoding"     # encoding, recall
        self.timestep = 0
        self.correct_answer_num = 0

        # convert the first observation to concatenated one-hot vectors
        obs = self._generate_observation(self.memory_sequence[0], self.question_type, self.question_value, include_question=self.include_question_during_encode)
        info = {"phase": "encoding"}
        return obs, info

    def step(self, action):
        """
        action: a MultiDiscrete vector of length num_features
        """
        self.timestep += 1
        if self.phase == "encoding":
            if self.timestep >= self.sequence_len:
                # first timestep of recall phase
                self.phase = "recall"
                self.timestep = 0
                obs = self._generate_observation(None, self.question_type, self.question_value, include_question=True)
                info = {"phase": "recall"}
                return obs, self.no_action_reward, False, info
            else:
                # encoding phase
                obs = self._generate_observation(self.memory_sequence[self.timestep], self.question_type, self.question_value, 
                                                include_question=self.include_question_during_encode)
                info = {"phase": "encoding"}
                return obs, self.no_action_reward, False, info
        elif self.phase == "recall":
            obs = self._generate_observation(None, self.question_type, self.question_value, include_question=True)
            info = {"phase": "recall"}

            converted_action = self._convert_action(action)
            action_correct = self._check_action(converted_action)

            if action_correct or converted_action[0] == "stop" and self.correct_answer_num == self.num_answers:
                reward = self.correct_reward
                self.correct_answer_num += 1
            elif converted_action[0] == "no_action":
                reward = self.no_action_reward
            else:
                reward = self.wrong_reward

            if converted_action[0] == "stop" or self.timestep > self.retrieve_time_limit:
                done = True
            else:
                done = False
            
            return obs, reward, done, info

    def render(self, mode='human'):
        return

    def _generate_question_type_dict(self):
        """
        generate a dict with key being the index of the question in the question space and value being the question type
        e.g. there's two features, and question_space=("choice", "max", "min"), then the dict would be
        {
            0: "choice",
            1: "choice",
            2: "max",
            3: "min"
        }
        if there's "choice" in the question_space, the "choice[i]" question would be marked as the first num_feature questions
        """
        question_index = {}
        feature_index = 0
        if "choice" in self.question_space:
            question_space = [q for q in self.question_space if q != "choice"]
            for i in range(self.num_features):
                question_index[feature_index] = "choice"
                feature_index += 1
        else:
            question_space = self.question_space
        for q in question_space:
            question_index[feature_index] = q
            feature_index += 1
        return question_index
    
    def _generate_all_stimuli(self):
        """
        generate all possible stimuli, in the format of a list of numbers with length num_features, the value of each number is within [0, feature_dim)
        """
        all_stimuli = list(itertools.product(range(self.feature_dim), repeat=self.num_features))
        all_stimuli = np.array([np.array(stimuli) for stimuli in all_stimuli])
        return all_stimuli

    def _generate_observation(self, stimuli, question_type, question_value, include_question=False):
        """
        convert observation vector to concatenated one-hot vector
        stimuli: a list of numbers with length num_features, the value of each number is within [0, feature_dim)
            if None, no stimuli is presented
        question_type: the index of the question in the question space
        question_value: the value within the question, dim = feature_dim
            e.g. given the value of the 1st feature as x, x is the question_value
        include_question: whether to include the question in the observation
        """
        observation = np.zeros(np.sum(self.observation_space.nvec))
        if stimuli is not None:
            for i in range(self.num_features):
                observation[i*self.feature_dim+stimuli[i]] = 1
        if include_question:
            observation[self.num_features*self.feature_dim+question_type] = 1
            observation[self.num_features*self.feature_dim+self.question_space_dim+question_value] = 1
        return observation
    
    def _convert_action(self, action):
        """
        convert integer action to the format of the stimuli, i.e. a list of numbers with length num_features, 
            the value of each number is within [0, feature_dim)
        if the action is "no action" or "stop", return the corresponding string
        the encoding of the action is small-edian, i.e. after converting action to base feature_dim, 
            the first feature corresponds to the last digit of the action
        """
        if action == self.action_space.n - 2:
            return ["no_action"]
        elif action == self.action_space.n - 1:
            return ["stop"]
        else:
            # change action base to feature_dim
            action_base = np.zeros(self.num_features, dtype=int)
            for i in range(self.num_features):
                action_base[i] = action % self.feature_dim
                action = action // self.feature_dim
            return action_base
    
    def _check_action(self, action):
        """
        check if action is correct
        action should be a list of numbers with length num_features, the value of each number is within [0, feature_dim)
        """
        # if action is "no action" or "stop", return False
        # check stop in "step" function
        if action[0] == "no_action" or action[0] == "stop":
            return False
        for correct_action in self.memory_sequence[self.correct_answers_index]:
            if np.all(action == correct_action):
                return True
        return False
