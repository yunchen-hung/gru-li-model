import math
import numpy as np
import gym


import numpy as np
import gym


class SequentialMemory(gym.Env):
    def __init__(self, vocabulary_num=100, memory_num=10, retrieve_time_limit=15, true_reward=1.0, false_reward=-0.1, 
    reset_state_before_test=False, start_recall_cue=False):
        self.vocabulary_num = vocabulary_num
        self.memory_num = memory_num
        self.true_reward = true_reward
        self.false_reward = false_reward
        self.retrieve_time_limit = max(retrieve_time_limit, memory_num)
        self.reset_state_before_test = reset_state_before_test
        self.start_recall_cue = start_recall_cue
        self.batch_size = 1

        self.memory_sequence = self.generate_sequence()
        self.current_timestep = 0
        self.testing = False    # false: presenting sequence, true: testing
        self.reported_memory = 0

    def generate_sequence(self):
        rand_index = np.arange(1, self.vocabulary_num+1)
        np.random.shuffle(rand_index)
        return rand_index[:self.memory_num]
    
    def step(self, action):
        start_recall = 0
        if self.testing:
            if self.current_timestep == 0:
                start_recall = 1
            if action == self.memory_sequence[self.reported_memory]:
                reward = self.true_reward
                self.reported_memory += 1
            elif action == 0:
                reward = 0.0    # not know
            else:
                reward = self.false_reward
                self.reported_memory += 1
            observation = np.zeros(self.vocabulary_num+1)
            info = {"encoding_on": False}
            self.current_timestep += 1
            if self.current_timestep >= self.retrieve_time_limit or self.reported_memory >= self.memory_num:
                done = True
            else:
                done = False
        else:
            reward = 0.0
            done = False
            self.current_timestep += 1
            if self.current_timestep == self.memory_num:
                self.testing = True
                observation = np.zeros(self.vocabulary_num+1)
                self.current_timestep = 0
                self.reported_memory = 0
                info = {"encoding_on": False}
                if self.reset_state_before_test:
                    info["reset_state"] = True
            else:
                observation = np.eye(self.vocabulary_num+1)[self.memory_sequence[self.current_timestep]]
                info = {"encoding_on": True, "retrieval_off": True}
        if self.start_recall_cue:
            observation = np.concatenate((observation, np.array([start_recall])))
        return observation, reward, done, info

    def reset(self, regenerate_contexts=True):
        if regenerate_contexts:
            self.memory_sequence = self.generate_sequence()
        self.current_timestep = 0
        self.testing = False
        self.reported_memory = 0
        info = {"encoding_on": True, "retrieval_off": True}
        observation = np.eye(self.vocabulary_num+1)[self.memory_sequence[self.current_timestep]]
        if self.start_recall_cue:
            observation = np.concatenate((observation, np.array([0])))
        return observation, info

    def render(self, mode='human'):
        pass
    
    def get_batch(self):
        if self.start_recall_cue:
            stim = np.eye(self.vocabulary_num+2)[self.memory_sequence]
            stim_blank = np.zeros((self.memory_num, self.vocabulary_num+2))
            stim_blank[0, self.vocabulary_num+1] = 1
            data = np.concatenate((stim, stim_blank), axis=0)
            gt_blank = np.zeros((self.memory_num, self.vocabulary_num+1))
            gt = np.concatenate((gt_blank, stim[:, 0:self.vocabulary_num+1]), axis=0)
        else:
            stim = np.eye(self.vocabulary_num+1)[self.memory_sequence]
            blank = np.zeros((self.memory_num, self.vocabulary_num+1))
            data = np.concatenate((stim, blank), axis=0)
            gt = np.concatenate((blank, stim), axis=0)
            # blank_gt = np.zeros(self.memory_num)
            # gt = np.concatenate((blank_gt, self.memory_sequence), axis=0)
        return data, gt

    def compute_accuracy(self, actions):
        correct_actions = 0
        wrong_actions = 0
        not_know_actions = 0
        reported_memory = 0
        for action in actions[self.memory_num:]:
            if action == self.memory_sequence[reported_memory]:
                correct_actions += 1
                reported_memory += 1
            elif action == 0:
                not_know_actions += 1
            else:
                wrong_actions += 1
                reported_memory += 1
            if reported_memory >= self.memory_num:
                break
        return correct_actions, wrong_actions, not_know_actions

    def compute_rewards(self, actions):
        rewards = []
        for t, action in enumerate(actions):
            if t < self.memory_num:
                rewards.append(0.0)
            else:
                if action == self.memory_sequence[t - self.memory_num]:
                    rewards.append(self.true_reward)
                elif action == 0:
                    rewards.append(0.0)
                else:
                    rewards.append(self.false_reward)
        return rewards


# class SequentialMemory(gym.Env):
#     """
#     Gym environment for sequential memory task, memory sequence provided with .
#     """

#     def __init__(self, num_arms, context_len, context_num=0, sequence_len=[5, 10], max_arm_reward_probability=0.9, true_reward=1.0, false_reward=-0.1,
#         seen_before_signal=False):
#         self.num_arms = num_arms
#         self.context_len = context_len
#         self.sequence_len = sequence_len
#         assert max_arm_reward_probability > 0.5 and max_arm_reward_probability <= 1
#         self.max_arm_reward_prob = max_arm_reward_probability
#         self.other_arms_reward_prob = 1 - self.max_arm_reward_prob
#         self.true_reward = true_reward
#         self.false_reward = false_reward
#         self.seen_before_signal = seen_before_signal

#         # sequence_len should be an int or a list
#         if isinstance(self.sequence_len, int):
#             assert self.sequence_len > 1
#         else:
#             assert isinstance(self.sequence_len, list)
#             assert len(self.sequence_len) == 2
#             assert self.sequence_len[0] > 1

#         if context_num == 0:
#             # self.context_num = int(math.pow(2, context_len))
#             self.context_num = self.context_len
#         else:
#             self.context_num = context_num
#             assert self.context_num < self.context_len
#         assert self.context_num > 0

#         self.action_space = gym.spaces.Discrete(num_arms + 1)
#         self.observation_space = gym.spaces.Discrete(context_len)
#         self.reward_range = (-1, 1)

#         self.contexts, self.sequences, self.sequence_lens = self._generate_contexts()
#         self.contexts_seen_before = np.zeros(self.context_num, dtype=bool)

#         self.timestep = 0
#         self.current_seq_len = self._generate_sequence_len()
#         self.current_context_num = np.random.randint(self.context_num)

#     def _generate_contexts(self):
#         # random_index = np.arange(int(math.pow(2, self.context_num)))
#         random_index = np.arange(self.context_len)
#         np.random.shuffle(random_index)
#         contexts_scalar = random_index[:self.context_num]
#         contexts_vector = np.zeros((self.context_num, self.context_len))
#         for i in range(self.context_num):
#             # contexts_vector[i, :] = np.binary_repr(contexts_scalar[i], width=self.context_len)
#             contexts_vector[i, contexts_scalar[i]] = 1
        
#         sequences = np.random.randint(self.num_arms, size=(self.context_num, self.sequence_len[1])) + 1

#         sequence_lens = np.random.randint(self.sequence_len[0], self.sequence_len[1] + 1, size=self.context_num)

#         return contexts_vector, sequences, sequence_lens

#     def _generate_sequence_len(self):
#         if isinstance(self.sequence_len, int):
#             return self.sequence_len
#         else:
#             return np.random.randint(self.sequence_len[0], self.sequence_len[1])

#     def regenerate_contexts(self):
#         self.contexts, self.max_reward_arms = self._generate_contexts()
#         self.contexts_seen_before = np.zeros(self.context_num, dtype=bool)

#     def regenerate_part_of_environment(self, replace_ratio=0.25):
#         replace_num = int(self.context_num * replace_ratio)
#         random_index = np.arange(self.context_len)
#         np.random.shuffle(random_index)
#         to_be_replaced_index = random_index[:replace_num]
#         replaced_max_reward_arms = np.random.randint(self.num_arms, size=replace_num) + 1
#         self.max_reward_arms[to_be_replaced_index] = replaced_max_reward_arms
#         self.contexts_seen_before[to_be_replaced_index] = False

#     def step(self, action):
#         action = int(action)
#         assert self.action_space.contains(action)
#         assert self.timestep < self.current_seq_len
#         self.timestep += 1
#         if action == 0:
#             reward = 0.0
#         elif action == self.max_reward_arms[self.current_context_num]:
#             reward = np.random.choice([self.true_reward, self.false_reward], p=[self.max_arm_reward_prob, 1 - self.max_arm_reward_prob])
#         else:
#             reward = np.random.choice([self.true_reward, self.false_reward], p=[self.other_arms_reward_prob, 1 - self.other_arms_reward_prob])
#         done = self.timestep == self.current_seq_len
#         info = {"encoding_on": self.timestep == self.current_seq_len - 1}
#         if self.seen_before_signal:
#             observation = np.concatenate((self.contexts[self.current_context_num], np.array([0.0])))
#         else:
#             observation = np.zeros(self.context_len)
#         # self.contexts_seen_before[self.current_context_num] = True
#         return observation, reward, done, info

#     def reset(self):
#         self.timestep = 0
#         self.current_seq_len = self._generate_sequence_len()
#         self.current_context_num = np.random.randint(self.context_num)
#         if self.seen_before_signal:
#             observation = np.concatenate((self.contexts[self.current_context_num], np.array([self.contexts_seen_before[self.current_context_num]])))
#         else:
#             observation = self.contexts[self.current_context_num]
#         self.contexts_seen_before[self.current_context_num] = True
#         return observation

#     def render(self, mode='human'):
#         pass

#     def compute_accuracy(self, actions):
#         correct_actions = 0
#         wrong_actions = 0
#         for i in range(self.current_seq_len):
#             if actions[i] == self.max_reward_arms[self.current_context_num]:
#                 correct_actions += 1
#             elif actions[i] != 0:
#                 wrong_actions += 1
#         return correct_actions, wrong_actions
        


