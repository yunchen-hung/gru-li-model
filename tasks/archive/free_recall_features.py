import math
import random
from copy import deepcopy
import numpy as np
import gym


class FreeRecallWithFeatures(gym.Env):
    def __init__(self, vocabulary_num, memory_num, feature_num, feature_dim, retrieve_time_limit, true_reward=1.0, false_reward=-0.1,
    not_know_reward=-0.1, reset_state_before_test=False, start_recall_cue=False, encode_reward_weight=0.1,
    return_action=False, return_reward=False, forward_smooth=0, backward_smooth=0, dt=10, tau=10, batch_size=1):
        self.vocabulary_num = vocabulary_num
        self.memory_num = memory_num
        self.feature_num = feature_num
        self.feature_dim = feature_dim
        self.retrieve_time_limit = max(retrieve_time_limit, memory_num)
        self.true_reward = true_reward
        self.false_reward = false_reward
        self.not_know_reward = not_know_reward
        self.reset_state_before_test = reset_state_before_test
        self.start_recall_cue = start_recall_cue
        self.batch_size = batch_size
        self.encode_reward_weight = encode_reward_weight
        self.return_action = return_action
        self.return_reward = return_reward

        self.steps_each_item = int(tau / dt)

        self.forward_smooth = forward_smooth
        self.backward_smooth = backward_smooth
        assert self.forward_smooth >= 0 and self.forward_smooth <= 1
        assert self.backward_smooth >= 0 and self.backward_smooth <= 1
        self.smooth_matrix = self.generate_smooth_matrix()

        self.feature_map = self.generate_feature_map()

        self.memory_sequence, self.feature_cue, self.feature_sequence = self.generate_sequence()
        self.fixed_feature_sequence = deepcopy(self.feature_sequence)
        self.stimuli = self.generate_stimuli()
        self.current_timestep = 0
        self.current_step_within_item = 0
        self.testing = False    # false: presenting sequence, true: testing
        self.reported_memory = np.zeros(self.batch_size)

    def generate_smooth_matrix(self):
        smooth_matrix = np.eye(self.memory_num)
        for i in range(self.memory_num - 1):
            for j in range(self.memory_num - i - 1):
                smooth_matrix[j+i+1][j] = math.pow(self.forward_smooth, i+1)
                smooth_matrix[j][j+i+1] = math.pow(self.backward_smooth, i+1)
        return smooth_matrix
    
    def generate_feature_map(self):
        """
        feature_map[item][feature_cue] = feature
        """
        feature_map = np.random.randint(1, self.feature_dim+1, size=(self.vocabulary_num, self.feature_num))
        return feature_map

    def generate_sequence(self, batch_size=None):
        """
        memory_sequence: batch size x length of sequence, each item represented by an integer
        feature_cue: batch size x length of sequence, each cue represented by an integer
        feature_sequence: batch size x length of sequence
        """
        if batch_size is not None:
            self.batch_size = batch_size
        rand_index = np.repeat(np.arange(0, self.vocabulary_num).reshape(1, -1), self.batch_size, axis=0)
        for i in range(self.batch_size):
            np.random.shuffle(rand_index[i])
        memory_sequence = rand_index[:, :self.memory_num]
        feature_cue = np.random.randint(0, self.feature_num, size=(self.batch_size))
        feature_sequence = self.feature_map[memory_sequence, feature_cue]
        return memory_sequence, feature_cue, feature_sequence

    def generate_stimuli(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        assert batch_size <= self.batch_size
        data = np.zeros((batch_size, self.memory_num, self.vocabulary_num))
        for i in range(batch_size):
            data[i, :, :] = np.eye(self.vocabulary_num)[self.memory_sequence[i]]
        data = np.einsum('jk,ikl->ijl', self.smooth_matrix, data).transpose(2, 0, 1)
        data = data / np.linalg.norm(data, axis=0)
        data = data.transpose(1, 2, 0)
        return data

    def increase_timestep(self, set_zero=False):
        self.current_step_within_item += 1
        if self.current_step_within_item == self.steps_each_item:
            self.current_step_within_item = 0
            if set_zero:
                self.current_timestep = 0
            else:
                self.current_timestep += 1
    
    def step(self, action, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        assert batch_size <= self.batch_size
        assert len(action) == batch_size
        start_recall = 0
        if self.return_action:
            if self.current_timestep == 0:
                returned_actions = np.zeros((batch_size, 1))
            else:
                returned_actions = np.array([a.cpu().detach().item() for a in action]).reshape(-1, 1)
        if self.testing:
            if self.current_timestep == 0:
                start_recall = 1
            rewards = np.zeros(batch_size)
            for i in range(batch_size):
                if action[i] == 0:
                    rewards[i] = self.not_know_reward
                # elif int(action[i]) in self.feature_sequence[i]:
                elif action[i] == self.feature_sequence[i][self.current_timestep-self.memory_num]:
                    rewards[i] = self.true_reward
                    # feature_pos = np.where(self.feature_sequence[i] == int(action[i]))[0][0]
                    # self.feature_sequence[i][feature_pos] = 0
                    self.reported_memory[i] += 1
                else:
                    rewards[i] = self.false_reward
                    self.reported_memory[i] += 1
            observations = np.concatenate((np.zeros((batch_size, self.vocabulary_num)), np.repeat(np.eye(self.feature_num)[self.feature_cue].reshape(1, -1), 
                                            batch_size, axis=0)), axis=1)
            info = {"encoding_on": False}
            self.increase_timestep()
            if self.current_timestep >= self.retrieve_time_limit or np.min(self.reported_memory) >= self.memory_num or np.sum(self.feature_sequence) == 0:
                done = True
            else:
                done = False
        else:
            rewards = np.zeros(batch_size)
            done = False
            for i in range(batch_size):
                if action[i] == self.feature_sequence[i][self.current_timestep]:
                    rewards[i] = self.true_reward * self.encode_reward_weight
                else:
                    rewards[i] = self.false_reward * self.encode_reward_weight
            self.increase_timestep()
            if self.current_timestep == self.memory_num:
                self.testing = True
                observations = np.zeros((batch_size, self.vocabulary_num))
                if self.current_step_within_item == self.steps_each_item - 1:
                    self.current_timestep = 0
                info = {"encoding_on": False}
                if self.reset_state_before_test:
                    info["reset_state"] = True
            else:
                observations = self.stimuli[:, self.current_timestep, :]
                info = {"encoding_on": True, "retrieval_off": True}
            observations = np.concatenate((observations, np.zeros((batch_size, self.feature_num))), axis=1)
        if self.start_recall_cue:
            observations = np.concatenate((observations, np.array([start_recall for _ in range(batch_size)]).reshape(-1, 1)), axis=1)
        if self.return_action:
            observations = np.concatenate((observations, returned_actions), axis=1)
        if self.return_reward:
            observations = np.concatenate((observations, rewards.reshape(-1, 1)), axis=1)
        return observations, rewards, done, info

    def reset(self, regenerate_contexts=True, batch_size=None):
        if regenerate_contexts:
            self.memory_sequence, self.feature_cue, self.feature_sequence = self.generate_sequence()
            self.fixed_feature_sequence = deepcopy(self.feature_sequence)
            self.stimuli = self.generate_stimuli()

        self.current_timestep = 0
        self.current_step_within_item = 0
        self.testing = False
        self.reported_memory = np.zeros(self.batch_size)
        info = {"encoding_on": True, "retrieval_off": True}
        observations = np.eye(self.vocabulary_num)[self.memory_sequence[:, self.current_timestep]]
        observations = np.concatenate((observations, np.zeros((self.batch_size, self.feature_num))), axis=1)
        if self.start_recall_cue:
            observations = np.concatenate((observations, np.zeros((self.batch_size, 1))), axis=1)
        if self.return_action:
            observations = np.concatenate((observations, np.zeros((self.batch_size, 1))), axis=1)
        if self.return_reward:
            observations = np.concatenate((observations, np.zeros((self.batch_size, 1))), axis=1)
        return observations, info

    def render(self, mode='human'):
        pass

    def get_batch(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        assert batch_size <= self.batch_size
        data = np.zeros((self.memory_num + self.retrieve_time_limit, batch_size, self.vocabulary_num))
        gt = np.zeros((self.memory_num + self.retrieve_time_limit, batch_size, self.vocabulary_num))
        data[:self.memory_num, :, :] = self.stimuli.transpose(1, 0, 2)
        for i in range(batch_size):
            gt[self.memory_num:self.memory_num*2, i, :] = np.eye(self.vocabulary_num)[self.memory_sequence[i]]
        if self.steps_each_item > 1:
            data = np.repeat(data, self.steps_each_item, axis=0)
            gt = np.repeat(gt, self.steps_each_item, axis=0)
        if self.start_recall_cue:
            cue = np.zeros((self.memory_num + self.retrieve_time_limit, batch_size, 1))
            cue[self.memory_num+1, :, 0] = 1
            data = np.concatenate((data, cue), axis=2)
        return data, gt

    def compute_accuracy(self, actions):
        assert len(actions[0]) <= self.batch_size
        # print("actions: ", actions)
        batch_size = min(len(actions[0]), self.batch_size)
        correct_actions = 0
        wrong_actions = 0
        not_know_actions = 0
        feature_sequence = self.feature_map[self.memory_sequence[:batch_size], self.feature_cue]
        # print("feature sequence: ", feature_sequence)
        for actions_batch in actions[self.memory_num:]:
            for i, a in enumerate(actions_batch):
                action = int(a)
                if action == 0:
                    not_know_actions += 1
                elif action in feature_sequence[i]:
                    correct_actions += 1
                    feature_pos = np.where(feature_sequence[i] == action)[0][0]
                    feature_sequence[i][feature_pos] = 0
                else:
                    wrong_actions += 1
        # print(correct_actions, wrong_actions, not_know_actions)
        return correct_actions, wrong_actions, not_know_actions

    def compute_rewards(self, actions):
        assert len(actions[0]) <= self.batch_size
        batch_size = min(len(actions[0]), self.batch_size)
        rewards = [[] for _ in range(batch_size)]
        feature_sequence = self.feature_map[self.memory_sequence[:batch_size], self.feature_cue]
        for t, action_batch in enumerate(actions[i]):
            for i, action in enumerate(action_batch):
                if t < self.memory_num:
                    rewards[i].append(0.0)
                else:
                    if action == 0:
                        rewards[i].append(0.0)
                    elif action in feature_sequence[i]:
                        rewards[i].append(self.true_reward)
                        feature_pos = np.where(feature_sequence[i] == action[i])[0][0]
                        feature_sequence[i][feature_pos] = 0
                    else:
                        rewards[i].append(self.false_reward)
        return np.array(rewards).transpose(1, 0)
