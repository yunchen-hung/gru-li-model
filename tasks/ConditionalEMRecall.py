import math
import itertools
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .base import BaseEMTask


class ConditionalEMRecall(BaseEMTask):
    def __init__(self, num_features=2, feature_dim=5, sequence_len=8, retrieve_time_limit=None, 
                 correct_reward=1.0, wrong_reward=-1.0, no_action_reward=0.0, early_stop_reward=-8.0,
                 include_question_during_encode=False, reset_state_before_test=True, no_early_stop=False,
                 question_space=["choice"], has_question=True, one_hot_stimuli=False,
                 sum_feature_placeholder=False, seed=None):
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
            reset_state_before_test: whether to reset the state of the network before testing
            include_question_during_encode: whether to give the question during encoding phase
            has_question: if False, no question will be given, and the agent will be asked to recall all stimuli
            question_space:
                choice: given one of the feature equals to a particular value
                max: given the max value of all the features
                min: given the min value of all the features
            one_hot_stimuli: whether to convert stimuli to one-hot vector
        Observation space:
            stimuli: num_features * [feature_dim one-hot vector]
            question: 
                question_space: the number of possible questions
                question_value: the value within the question, dim = feature_dim
                    e.g. given the value of the 1st feature as x, x is the question_value
        Action space:
            feature_dim ^ num_features + 1 no_action dim + 1 stop dim, a one-hot vector overall
        """
        super().__init__(reset_state_before_test=reset_state_before_test, seed=seed)
        self.num_features = num_features
        self.feature_dim = feature_dim
        self.sequence_len = sequence_len
        self.retrieve_time_limit = retrieve_time_limit if retrieve_time_limit is not None else sequence_len
        self.include_question_during_encode = include_question_during_encode
        self.has_question = has_question
        self.one_hot_stimuli = one_hot_stimuli
        self.no_early_stop = no_early_stop
        self.vocabulary_num = feature_dim ** num_features
        self.sum_feature_placeholder = sum_feature_placeholder

        self.correct_reward = correct_reward
        self.wrong_reward = wrong_reward
        self.no_action_reward = no_action_reward
        self.early_stop_reward = early_stop_reward

        self.question_space_dim = len(question_space)
        if "choice" in question_space:
            self.question_space_dim += num_features - 1
        self.question_space = question_space
        self.question_type_dict = self._generate_question_type_dict()

        if self.one_hot_stimuli:
            obs_space_list = [self.feature_dim ** self.num_features]
        else:
            obs_space_list = [self.feature_dim for _ in range(num_features)]
        obs_space_list.extend([self.question_space_dim, feature_dim])
        if self.sum_feature_placeholder:
            obs_space_list.append(self.question_space_dim)
        # self.observation_space = spaces.MultiDiscrete(obs_space_list)
        self.obs_shape = np.sum(obs_space_list)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.obs_shape,), dtype=np.float32)
        # print(self.observation_space)
        self.action_space = spaces.Discrete(feature_dim ** num_features + 2)
        
        self.all_stimuli = self._generate_all_stimuli()

    def reset(self, batch_size=1):
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
        if self.has_question:
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
        else:
            # no questions, recall all stimuli
            self.num_answers = self.sequence_len
            self.correct_answers_index = np.arange(self.sequence_len)

        self.phase = "encoding"     # encoding, recall
        self.timestep = 0
        self.correct_answer_num = 0
        self.answered = np.zeros(self.num_answers, dtype=bool) # whether each answer has been outputted
        self.done = False

        # convert the first observation to concatenated one-hot vectors
        obs = self._generate_observation(self.memory_sequence[0], self.question_type, self.question_value, include_question=self.include_question_during_encode)
        if self.include_question_during_encode and not self._check_action(self.memory_sequence[self.timestep]):
            gt = self.action_space.n - 2
        else:
            gt = self.convert_stimuli_to_action(self.memory_sequence[self.timestep])
        info = {"phase": "encoding", "gt": gt}
        return obs, info

    def step(self, action):
        """
        action: a MultiDiscrete vector of length num_features
        """
        try:
            action = action[0].item()
        except:
            pass
        if self.done:
            obs = self._generate_observation(None, self.question_type, self.question_value, include_question=self.has_question)
            info = {"phase": "recall", 
                    "gt": self.action_space.n - 1, "gt_mask": False, "loss_mask": False,
                    "correct": 0, "wrong": 0, "not_know": 0,
                    "done": True}
            return obs, 0.0, False, False, info
        self.timestep += 1
        if self.phase == "encoding":
            if self.timestep >= self.sequence_len:
                # first timestep of recall phase
                self.phase = "recall"
                obs = self._generate_observation(None, self.question_type, self.question_value, include_question=self.has_question)
                if self.include_question_during_encode and not self._check_action(self.memory_sequence[self.timestep-1]):
                    gt = self.action_space.n - 2
                else:
                    gt = self.convert_stimuli_to_action(self.memory_sequence[self.timestep-1])
                self.timestep = 0
                info = {"phase": "recall", "reset_state": self.reset_state_before_test,
                        "gt": gt, "gt_mask": True, "loss_mask": True,
                        "correct": 0, "wrong": 0, "not_know": 0,
                        "done": False}
                return obs, 0.0, False, False, info
            else:
                # encoding phase
                obs = self._generate_observation(self.memory_sequence[self.timestep], self.question_type, self.question_value, 
                                                include_question=self.include_question_during_encode)
                if self.include_question_during_encode and not self._check_action(self.memory_sequence[self.timestep]):
                    gt = self.action_space.n - 2
                else:
                    gt = self.convert_stimuli_to_action(self.memory_sequence[self.timestep-1])
                info = {"phase": "encoding", 
                        "gt": gt, "gt_mask": True, "loss_mask": True,
                        "correct": 0, "wrong": 0, "not_know": 0,
                        "done": False}
                return obs, 0.0, False, False, info
        elif self.phase == "recall":
            obs = self._generate_observation(None, self.question_type, self.question_value, include_question=self.has_question)
            if not self._check_action(self.memory_sequence[self.timestep-1]):
                gt = self.action_space.n - 2
            else:
                gt = self.convert_stimuli_to_action(self.memory_sequence[self.timestep-1])
            info = {"phase": "recall",
                    "gt": gt, "gt_mask": True, "loss_mask": True,
                    "correct": 0, "wrong": 0, "not_know": 0,
                    "done": False}

            converted_action = self.convert_action_to_stimuli(action)
            action_correct = self._check_action(converted_action)

            if action_correct or (not self.no_early_stop and converted_action[0] == "stop" and self.correct_answer_num == self.num_answers):
                reward = self.correct_reward
                self.correct_answer_num += 1
                info["correct"] = 1
            elif converted_action[0] == "no_action" or (self.no_early_stop and converted_action[0] == "stop" 
                                                        and self.correct_answer_num == self.num_answers):
                reward = self.no_action_reward
                info["not_know"] = 1
            elif converted_action[0] == "stop":
                reward = self.early_stop_reward
                info["wrong"] = 1
            else:
                reward = self.wrong_reward
                info["wrong"] = 1

            if (not self.no_early_stop and converted_action[0] == "stop") or self.timestep >= self.retrieve_time_limit:
                self.done = True
            info["done"] = self.done
            
            return obs, reward, False, False, info

    def render(self, mode='human'):
        print("memory sequence:", self.memory_sequence)
        print("question type:", self.question_type)
        print("question value:", self.question_value)
        print("correct answers:", self.memory_sequence[self.correct_answers_index])
    
    def compute_accuracy(self, actions):
        """
        compute the accuracy of a sequence of actions during recall phase
        """
        correct_actions = 0
        wrong_actions = 0
        no_actions = 0
        answered = np.zeros(self.num_answers, dtype=bool)
        for action in actions[self.sequence_len:]:
            if action == self.action_space.n - 2:
                # no action
                no_actions += 1
            elif correct_actions == self.num_answers:
                # all correct answers have been outputted, all actions are wrong except for "stop"
                if action == self.action_space.n - 1:
                    # stop
                    correct_actions += 1
                else:
                    wrong_actions += 1
            else:
                # check if an action is correct
                if action == self.action_space.n - 1:
                    wrong_actions += 1
                else:
                    converted_action = self.convert_action_to_stimuli(action)
                    correct_one_action = False
                    for i, correct_action in enumerate(self.memory_sequence[self.correct_answers_index]):
                        if np.all(converted_action == correct_action) and not answered[i]:
                            answered[i] = True
                            correct_one_action = True
                            break
                    if correct_one_action:
                        correct_actions += 1
                    else:
                        wrong_actions += 1
        return correct_actions, wrong_actions, no_actions
    
    def get_ground_truth(self, phase='recall'):
        """
        get expected actions of a trial
        """
        if self.include_question_during_encode:
            gt_enc = np.array([self.convert_stimuli_to_action(self.memory_sequence[i]) \
                                if i in self.correct_answers_index else self.action_space.n - 2 \
                                    for i in range(self.sequence_len)])
        else:
            gt_enc = np.array([self.convert_stimuli_to_action(self.memory_sequence[i]) for i in range(self.sequence_len)])
        gt_rec = np.array([self.convert_stimuli_to_action(self.memory_sequence[i]) \
                            if i in self.correct_answers_index else self.action_space.n - 2 \
                                for i in range(self.sequence_len)])
        gt = np.concatenate([gt_enc, gt_rec])

        if phase == 'encoding':
            mask = np.array([1 if i < self.sequence_len else 0 for i in range(self.sequence_len*2)])
        elif phase == 'recall':
            mask = np.array([1 if i >= self.sequence_len else 0 for i in range(self.sequence_len*2)])
        else:
            mask = np.ones(self.sequence_len*2)

        # print(gt.shape, mask.shape)

        return gt.reshape(1, -1), mask.reshape(1, -1)

    def get_trial_data(self):
        """
        get trial data, including memory sequence, question type, question value, and correct answers
        """
        return {"memory_sequence": self.memory_sequence, "question_type": self.question_type, 
                "question_value": self.question_value, "correct_answers": self.memory_sequence[self.correct_answers_index],
                "memory_sequence_int": np.array([self.convert_stimuli_to_action(m) for m in self.memory_sequence])}

    def convert_action_to_stimuli(self, action):
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
        
    def convert_stimuli_to_action(self, stimuli):
        """
        convert stimuli to action, i.e. a integer action
        the encoding of the action is small-edian, i.e. after converting action to base feature_dim, 
            the first feature corresponds to the last digit of the action
        """
        action = 0
        for i in range(self.num_features):
            action += stimuli[i] * (self.feature_dim ** i)
        return action
    
    def convert_stimuli_to_one_hot(self, stimuli):
        """
        convert stimuli to one-hot vector, corresponding to the action
        """
        action = self.convert_stimuli_to_action(stimuli)
        stimuli_one_hot = np.zeros(self.feature_dim ** self.num_features)
        stimuli_one_hot[action] = 1
        return stimuli_one_hot
        
    def convert_action_to_observation(self, action):
        """
        convert action to observation space, make no action and stop actions as two additional dimensions
        """
        action = self.convert_action_to_stimuli(action)
        action_obs = np.zeros(self.feature_dim*self.num_features+2)
        if action[0] == "no_action":
            action_obs[-2] = 1
        elif action[0] == "stop":
            action_obs[-1] = 1
        else:
            for i in range(self.num_features):
                action_obs[i*self.feature_dim+action[i]] = 1
        return action_obs

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
            observation[question_offset+question_type] = 1
            observation[question_offset+self.question_space_dim+question_value] = 1
        return observation
    
    def _check_action(self, action):
        """
        check if action is correct
        action should be a list of numbers with length num_features, the value of each number is within [0, feature_dim)
        """
        # if action is "no action" or "stop", return False
        # check stop in "step" function
        if action[0] == "no_action" or action[0] == "stop":
            return False
        for i, correct_action in enumerate(self.memory_sequence[self.correct_answers_index]):
            if np.all(action == correct_action) and not self.answered[i]:
                if self.phase == "recall":
                    self.answered[i] = True
                return True
        return False
    

if __name__ == "__main__":
    from . import MetaLearningEnv

    env = ConditionalEMRecall(include_question_during_encode=True, has_question=False)
    env = MetaLearningEnv(env)
    obs, info = env.reset()
    print('memory_sequence:', env.memory_sequence)
    print('question_type:', env.question_type)
    print('question_value:', env.question_value)
    print('correct_answers:', env.memory_sequence[env.correct_answers_index])

    gt = env.get_ground_truth()
    actions = np.random.choice(gt, 9)
    correct_actions, wrong_actions, not_know_actions = env.compute_accuracy(actions) 
    print('gt:', gt)
    print('actions:', actions)
    print('correct_actions:', correct_actions)
    print('wrong_actions:', wrong_actions)
    print('not_know_actions:', not_know_actions)

    actions = env.memory_sequence[env.correct_answers_index]
    actions_int = [0 for _ in range(8)]
    for action in actions:
        actions_int.append(action[0]+action[1]*5)
        actions_int.append(action[0]+action[1]*5)
    actions_int.append(26)
    actions_int.append(27)
    actions_int = np.array(actions_int)

    print(obs, info)
    cnt = 0
    while True:
        # action = env.action_space.sample()
        action = actions_int[cnt]
        cnt += 1
        print("action:", action, env.convert_action_to_stimuli(action))
        obs, reward, done, info = env.step(action)
        print(obs, reward, done, info)
        if done:
            break
