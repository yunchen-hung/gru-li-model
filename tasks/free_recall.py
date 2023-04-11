import math
import numpy as np
import gym


class FreeRecall(gym.Env):
    def __init__(self, vocabulary_num=100, memory_num=10, retrieve_time_limit=15, true_reward=1.0, false_reward=-0.1, repeat_penalty=-0.1, 
    not_know_reward=-0.1, reset_state_before_test=False, start_recall_cue=False, no_repeat_items=True, encode_reward_weight=0.1,
    return_action=False, return_reward=False, forward_smooth=0, backward_smooth=0, dt=10, tau=10, batch_size=1):
        self.vocabulary_num = vocabulary_num
        self.memory_num = memory_num
        self.true_reward = true_reward
        self.false_reward = false_reward
        self.not_know_reward = not_know_reward
        self.repeat_penalty = repeat_penalty
        self.retrieve_time_limit = max(retrieve_time_limit, memory_num)
        self.reset_state_before_test = reset_state_before_test
        self.start_recall_cue = start_recall_cue
        self.no_repeat_items = no_repeat_items
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

        self.memory_sequence = self.generate_sequence()
        self.stimuli = self.generate_stimuli()
        self.current_timestep = 0
        self.current_step_within_item = 0
        self.testing = False    # false: presenting sequence, true: testing
        self.not_retrieved = np.ones((self.batch_size, self.vocabulary_num+1), dtype=np.bool)
        # for i in range(self.batch_size):
        #     self.not_retrieved[i][self.memory_sequence[i]] = True
        self.reported_memory = np.zeros(self.batch_size)

    def generate_smooth_matrix(self):
        smooth_matrix = np.eye(self.memory_num)
        for i in range(self.memory_num - 1):
            for j in range(self.memory_num - i - 1):
                smooth_matrix[j+i+1][j] = math.pow(self.forward_smooth, i+1)
                smooth_matrix[j][j+i+1] = math.pow(self.backward_smooth, i+1)
        return smooth_matrix

    def generate_sequence(self, batch_size=None):
        if batch_size is not None:
            self.batch_size = batch_size
        rand_index = np.repeat(np.arange(1, self.vocabulary_num+1).reshape(1, -1), self.batch_size, axis=0)
        for i in range(self.batch_size):
            np.random.shuffle(rand_index[i])
        memory_sequence = rand_index[:, :self.memory_num]
        return memory_sequence      # axis 0: batch size, axis 1: length of sequence

    def generate_stimuli(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        assert batch_size <= self.batch_size
        data = np.zeros((batch_size, self.memory_num, self.vocabulary_num+1))
        for i in range(batch_size):
            data[i, :, :] = np.eye(self.vocabulary_num+1)[self.memory_sequence[i]]
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
                returned_actions = np.zeros((batch_size, self.vocabulary_num+1))
            else:
                returned_actions_digit = np.array([a.cpu().detach().item() for a in action])
                returned_actions = np.eye(self.vocabulary_num+1)[returned_actions_digit]
        if self.testing:
            if self.current_timestep == 0:
                start_recall = 1
            rewards = np.zeros(batch_size)
            for i in range(batch_size):
                if action[i] in list(self.memory_sequence[i]):
                    if self.not_retrieved[i][action[i]]:
                        rewards[i] = self.true_reward
                        self.not_retrieved[i][action[i]] = False
                    else:
                        rewards[i] = self.repeat_penalty
                    self.reported_memory[i] += 1
                elif action[i] == 0:
                    rewards[i] = self.not_know_reward
                else:
                    rewards[i] = self.false_reward
                    self.reported_memory[i] += 1
            observations = np.zeros((batch_size, self.vocabulary_num+1))
            info = {"encoding_on": False}
            self.increase_timestep()
            if self.current_timestep >= self.retrieve_time_limit or np.min(self.reported_memory) >= self.memory_num or np.sum(self.not_retrieved) == 0:
                done = True
            else:
                done = False
        else:
            rewards = np.zeros(batch_size)
            done = False
            for i in range(batch_size):
                if action[i] == self.memory_sequence[i][self.current_timestep]:
                    rewards[i] = self.true_reward * self.encode_reward_weight
                else:
                    rewards[i] = self.false_reward * self.encode_reward_weight
            self.increase_timestep()
            if self.current_timestep == self.memory_num:
                self.testing = True
                observations = np.zeros((batch_size, self.vocabulary_num+1))
                if self.current_step_within_item == self.steps_each_item - 1:
                    self.current_timestep = 0
                info = {"encoding_on": False}
                if self.reset_state_before_test:
                    info["reset_state"] = True
            else:
                observations = self.stimuli[:, self.current_timestep, :]
                info = {"encoding_on": True, "retrieval_off": True}
        if self.start_recall_cue:
            observations = np.concatenate((observations, np.array([start_recall for _ in range(batch_size)]).reshape(-1, 1)), axis=1)
        if self.return_action:
            observations = np.concatenate((observations, returned_actions), axis=1)
        if self.return_reward:
            observations = np.concatenate((observations, rewards.reshape(-1, 1)), axis=1)
        return observations, rewards, done, info

    def reset(self, regenerate_contexts=True, batch_size=None):
        if regenerate_contexts:
            self.memory_sequence = self.generate_sequence(batch_size)
            self.stimuli = self.generate_stimuli()

        self.current_timestep = 0
        self.current_step_within_item
        self.testing = False
        self.not_retrieved = np.ones((self.batch_size, self.vocabulary_num+1), dtype=np.bool)
        self.reported_memory = np.zeros(self.batch_size)
        # self.not_retrieved = np.zeros(self.vocabulary_num+1, dtype=np.bool)
        # self.not_retrieved[self.memory_sequence] = True
        # self.reported_memory = 0
        info = {"encoding_on": True, "retrieval_off": True}
        observations = np.eye(self.vocabulary_num+1)[self.memory_sequence[:, self.current_timestep]]
        if self.start_recall_cue:
            observations = np.concatenate((observations, np.zeros((self.batch_size, 1))), axis=1)
        if self.return_action:
            observations = np.concatenate((observations, np.zeros((self.batch_size, self.vocabulary_num+1))), axis=1)
        if self.return_reward:
            observations = np.concatenate((observations, np.zeros((self.batch_size, 1))), axis=1)
        return observations, info

    def render(self, mode='human'):
        pass

    def get_batch(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        assert batch_size <= self.batch_size
        data = np.zeros((self.memory_num + self.retrieve_time_limit, batch_size, self.vocabulary_num+1))
        gt = np.zeros((self.memory_num + self.retrieve_time_limit, batch_size, self.vocabulary_num+1))
        data[:self.memory_num, :, :] = self.stimuli.transpose(1, 0, 2)
        for i in range(batch_size):
            gt[self.memory_num:self.memory_num*2, i, :] = np.eye(self.vocabulary_num+1)[self.memory_sequence[i]]
        if self.steps_each_item > 1:
            data = np.repeat(data, self.steps_each_item, axis=0)
            gt = np.repeat(gt, self.steps_each_item, axis=0)
        if self.start_recall_cue:
            cue = np.zeros((self.memory_num + self.retrieve_time_limit, batch_size, 1))
            cue[self.memory_num+1, :, 0] = 1
            data = np.concatenate((data, cue), axis=2)
        return data, gt
        # if self.start_recall_cue:
        #     stim = np.eye(self.vocabulary_num+2)[self.memory_sequence]
        #     stim_blank = np.zeros((self.retrieve_time_limit, self.vocabulary_num+2))
        #     stim_blank[0, self.vocabulary_num+1] = 1
        #     data = np.concatenate((stim, stim_blank), axis=0)
        #     gt_blank = np.zeros((self.memory_num, self.vocabulary_num+1))
        #     gt_blank2 = np.zeros((self.retrieve_time_limit - self.memory_num, self.vocabulary_num+1))
        #     gt = np.concatenate((gt_blank, stim[:, 0:self.vocabulary_num+1], gt_blank2), axis=0)
        # else:
        #     stim = np.eye(self.vocabulary_num+1)[self.memory_sequence]
        #     stim_blank = np.zeros((self.retrieve_time_limit, self.vocabulary_num+1))
        #     gt_blank = np.zeros((self.memory_num, self.vocabulary_num+1))
        #     gt_blank2 = np.zeros((self.retrieve_time_limit - self.memory_num, self.vocabulary_num+1))
        #     data = np.concatenate((stim, stim_blank), axis=0)
        #     gt = np.concatenate((gt_blank, stim, gt_blank2), axis=0)

    def compute_accuracy(self, actions):
        assert len(actions[0]) <= self.batch_size
        batch_size = min(len(actions[0]), self.batch_size)
        correct_actions = 0
        wrong_actions = 0
        not_know_actions = 0
        not_retrieved = np.ones((batch_size, self.vocabulary_num+1), dtype=np.bool)
        for actions_batch in actions[self.memory_num:]:
            for i, action in enumerate(actions_batch):
                if action in list(self.memory_sequence[i]) and not_retrieved[i][action]:
                    correct_actions += 1
                    not_retrieved[i][action] = False
                elif action == 0:
                    not_know_actions += 1
                else:
                    wrong_actions += 1
        return correct_actions, wrong_actions, not_know_actions

    def compute_rewards(self, actions):
        assert len(actions[0]) <= self.batch_size
        batch_size = min(len(actions[0]), self.batch_size)
        rewards = [[] for _ in range(batch_size)]
        not_retrieved = np.ones((batch_size, self.vocabulary_num+1), dtype=np.bool)
        for t, action_batch in enumerate(actions[i]):
            for i, action in enumerate(action_batch):
                if t < self.memory_num:
                    rewards[i].append(0.0)
                else:
                    if action in list(self.memory_sequence[i]):
                        if not_retrieved[i][action]:
                            rewards[i].append(self.true_reward)
                            not_retrieved[i][action] = False
                        else:
                            rewards[i].append(self.repeat_penalty)
                    elif action == 0:
                        rewards[i].append(0.0)
                    else:
                        rewards[i].append(self.false_reward)
        return np.array(rewards).transpose(1, 0)
