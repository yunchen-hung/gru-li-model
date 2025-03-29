import itertools
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FreeRecall(gym.Env):
    def __init__(self,
                 num_features=4,
                 feature_dim=2,
                 sequence_len=4,
                 retrieve_time_limit=None,
                 reset_state_before_test=True,

                 correct_reward=1.0,
                 wrong_reward=-1.0,
                 no_action_reward=-1.0,
                 
                 one_hot_stimuli=True,              # whether to use one-hot encoding for the stimuli, otherwise use feature-wise encoding
                 one_hot_action=True,               # whether to use one-hot encoding for the action, otherwise use feature-wise encoding

                 seed=None,
                 **kwargs):
                 
        np.random.seed(None)

        self.num_features = num_features
        self.feature_dim = feature_dim 
        self.vocabulary_size = feature_dim ** num_features
        self.sequence_len = sequence_len
        self.retrieve_time_limit = retrieve_time_limit if retrieve_time_limit is not None else sequence_len

        self.reset_state_before_test = reset_state_before_test
        
        self.include_question_during_encode = include_question_during_encode

        self.correct_reward = correct_reward
        self.wrong_reward = wrong_reward
        self.no_action_reward = no_action_reward

        self.action_space_type = action_space_type

        self.one_hot_stimuli = one_hot_stimuli
        if self.one_hot_stimuli:
            obs_space_dim = self.feature_dim ** self.num_features + self.num_features + self.feature_dim + self.num_features
        else:
            obs_space_dim = self.num_features * self.feature_dim + self.num_features + self.feature_dim + self.num_features
        self.obs_shape = obs_space_dim
        self.observation_space = spaces.Box(low=-0.1, high=1.1, shape=(obs_space_dim,), dtype=float)

        self.one_hot_action = one_hot_action
        if self.one_hot_action:
            action_space_dim = [self.vocabulary_size + 1]
        else:
            action_space_dim = [self.feature_dim] * self.num_features + [2]     # the final [2] is for deciding whether to task "no action"

        if self.action_space_type == "feature_wise":
            action_space_dim = [self.feature_dim] * self.num_features + [self.answer_dim] + [2]
        elif self.action_space_type == "task_wise":
            action_space_dim = [self.feature_dim ** self.num_features + 1] + [self.feature_dim + 1] + [self.answer_dim + 1]
        else:
            raise ValueError("Invalid action space type")
        self.action_shape = np.sum(action_space_dim)
        self.action_space = spaces.MultiDiscrete(action_space_dim)

        self.action_space_mask = np.ones(len(self.action_space.nvec), dtype=bool)
        self.encoding_action_space_mask = np.zeros(len(self.action_space.nvec), dtype=bool)
        if self.action_space_type == "feature_wise":
            self.encoding_action_space_mask[:self.num_features] = True
        elif self.action_space_type == "task_wise":
            self.encoding_action_space_mask[0] = True

        self.all_stimuli = self._generate_all_stimuli()


    def reset(self, memory_sequence_index=None, **kwargs):
        if memory_sequence_index is None:
            self.memory_sequence = self.all_stimuli[np.random.choice(len(self.all_stimuli), self.sequence_len, replace=False)]
        else:
            self.memory_sequence = self.all_stimuli[memory_sequence_index]
        self.memory_sequence_int = self._convert_item_to_int(self.memory_sequence)
        self._generate_condition_features() # will use different method for different tasks

        self.phase = "encoding"     # encoding, recall
        self.timestep = 0
        self.answered = False       # whether all matched items are recalled/the question is answered

        # convert the first observation to concatenated one-hot vectors
        obs = self._generate_observation(self.memory_sequence[0], self.condition_feature, self.condition_value, 
                                         self.query_feature, include_question=self.include_question_during_encode)
        info = {"phase": "encoding"}
        return obs, info


    def step(self, action):
        self.timestep += 1
        if self.phase == "encoding":
            info = {"phase": "encoding", 
                    "gt": self._compute_gt(), "gt_mask": True, 
                    "loss_mask": True, 
                    "correct": 0, "wrong": 0, "not_know": 0,
                    "done": False,
                    "action_space_mask": self.encoding_action_space_mask}
            if self.timestep >= self.sequence_len:
                self.phase = "recall"
                self.timestep = 0
                info["phase"] = "recall"
                if self.reset_state_before_test:    # send signal for the agent to reset its state
                    info["reset_state"] = True
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
                    "gt_mask": False,
                    "loss_mask": True,
                    "correct": 0, "wrong": 0, "not_know": 0,
                    "done": False,
                    "action_space_mask": self.action_space_mask}
            
            # for different task
            # compute the reward and the ground truth
            # update number of correct, wrong, and not_know
            # check if the trial is done
            reward, correct, wrong, not_know = self._compute_reward_and_metrics(action)
            info["correct"] = correct
            info["wrong"] = wrong
            info["not_know"] = not_know

            # default ground truth is the memory sequence (same in free recall task)
            # different tasks have a different _compute_gt method
            info["gt"] = self._compute_gt()

            if self.timestep >= self.retrieve_time_limit:
                info["done"] = True
                
            return obs, reward, False, False, info
    

    def get_ground_truth(self, phase="recall"):
        """
        return the ground truth for the current trial
        """
        if phase == "encoding":
            return self.memory_sequence[self.timestep]
        elif phase == "recall":
            return self.memory_sequence[self.timestep]
    

    def get_trial_data(self):
        """
        used when recording data
        """
        return {
            "memory_sequence": self.memory_sequence,
            "condition_feature": self.condition_feature,
            "condition_value": self.condition_value,
            "query_feature": self.query_feature,
            "answer": self.answer,
            "memory_sequence_int": self.memory_sequence_int,
        }
    

    def compute_accuracy(self, actions):
        """
        given action sequence, compute the accuracy of current trial
        """
        raise NotImplementedError
    

    def _generate_condition_features(self):
        """
        generate the condition features for the current trial
        """
        self.condition_feature = None
        self.condition_value = None
        self.query_feature = None
        self.answer = None 


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
                stimuli_int = self._convert_item_to_int(stimuli.reshape(1, -1))
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
    

    def _compute_gt(self):
        gt = np.zeros(len(self.action_space.nvec))
        if self.action_space_type == "feature_wise":
            gt[:self.num_features] = self.memory_sequence[self.timestep-1]
            gt[-1] = 0
        elif self.action_space_type == "task_wise":
            item_int = self._convert_item_to_int(self.memory_sequence[self.timestep-1].reshape(1, -1))
            gt[0] = item_int[0]
        return gt
    

    def _convert_item_to_int(self, item):
        """
        convert the item with multiple features to an integer
        item: shape (num_item, num_features)
        small-endian encoding of the vectorized item ([1,0,0,0] is 1, [0,0,0,1] is 8)
        """
        return np.sum(item * (self.feature_dim ** np.arange(self.num_features)), axis=1).astype(int)
    

    def _convert_int_to_item(self, item_int):
        """
        convert the integer to the item
        """
        return np.array([item_int // (self.feature_dim**i) % self.feature_dim for i in range(self.num_features)])

    
    def _convert_action_to_item(self, action):
        """
        convert the action to the item
        action: based on the action space type, the action is either a feature-wise action or a task-wise action
        item: shape (num_features), ignore the answer dimensions
        """
        item = np.zeros(self.num_features)
        if self.action_space_type == "feature_wise":
            if action[-1] == 1:
                # considered as no action, item is all -1
                item = np.ones(self.num_features) * -1
            else:
                item = action[:self.num_features]
        elif self.action_space_type == "task_wise":
            if action[0] == self.feature_dim**self.num_features:
                # considered as no action, item is all -1
                item = np.ones(self.num_features) * -1
            else:
                item = self._convert_int_to_item(action[0])
        return item

    
    def _convert_action_to_feature(self, action):
        """
        convert the action to the query feature value
        action: based on the action space type, the action is either a feature-wise action or a task-wise action
        feature: the value of the query feature
        """
        if self.action_space_type == "feature_wise":
            if action[-1] == 1:
                # considered as no action, value = 2
                feature = self.feature_dim
            else:
                feature = action[self.query_feature]
        elif self.action_space_type == "task_wise":
            feature = action[1]
        return feature

    
    def _convert_action_to_answer(self, action):
        """
        convert the action to the answer
        action: based on the action space type, the action is either a feature-wise action or a task-wise action
        answer: the answer
        """
        if self.action_space_type == "feature_wise":
            if action[-1] == 1:
                # considered as no action, action = 2
                answer = 2
            else:
                # action = 0 or 1
                answer = action[self.num_features]
        elif self.action_space_type == "task_wise":
            answer = action[2]
        return answer
    

    def _convert_action_to_observation(self, action):
        """
        convert the action to the observation
        """
        obs = np.zeros(self.action_shape)
        if self.action_space_type == "feature_wise":
            item = self._convert_action_to_item(action)
            for i in range(self.num_features):
                obs[int(i*self.feature_dim+item[i])] = 1 * self.action_space_mask[i]
            obs[int(self.feature_dim*self.num_features+action[-2])] = 1 * self.action_space_mask[-2]
            obs[int(-2+action[-1])] = 1 * self.action_space_mask[-1]
        elif self.action_space_type == "task_wise":
            obs[int(action[0])] = 1 * self.action_space_mask[0]
            obs[int(self.feature_dim**self.num_features + 1 + action[1])] = 1 * self.action_space_mask[1]
            obs[int(self.feature_dim**self.num_features + 1 + self.feature_dim + 1 + action[2])] = 1 * self.action_space_mask[2]
        return obs


    def _generate_task_condition(self):
        """
        generate condition_feature, condition_value, query_feature, answer, num_matched_item
        a different method for different task
        """
        self.condition_feature = None
        self.condition_value = None
        self.query_feature = None
        self.answer = None
