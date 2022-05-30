import math
import numpy as np
import gym


class BarcodeBandits(gym.Env):
    def __init__(self, num_arms, context_len, context_num=0, sequence_len=10, trial_per_context_per_batch=10, 
    max_arm_reward_probability=0.9, true_reward=1.0, false_reward=-0.1) -> None:
        self.num_arms = num_arms
        self.context_len = context_len
        self.context_num = context_num
        self.sequence_len = sequence_len
        self.trial_per_context_per_batch = trial_per_context_per_batch
        self.max_arm_reward_prob = max_arm_reward_probability
        self.other_arms_reward_prob = 1 - max_arm_reward_probability
        self.true_reward = true_reward
        self.false_reward = false_reward

        # sequence_len should be an int or a list
        if isinstance(self.sequence_len, int):
            assert self.sequence_len > 1
        else:
            assert isinstance(self.sequence_len, list)
            assert len(self.sequence_len) == 2
            assert self.sequence_len[0] > 1

        if context_num == 0:
            # self.context_num = int(math.pow(2, context_len))
            self.context_num = self.context_len
        else:
            self.context_num = context_num
            assert self.context_num <= self.context_len
        assert self.context_num > 0
        self.batch_size = self.context_num * self.trial_per_context_per_batch

        self.action_space = gym.spaces.Discrete(num_arms + 1)
        self.observation_space = gym.spaces.Discrete(context_len)

        self.contexts, self.max_reward_arms, self.context_pool = self._generate_contexts()

        self.timestep = 0
        self.current_seq_len = self._generate_sequence_len()
        self.current_context_pool_index = 0
        
    def _generate_contexts(self):
        # random_index = np.arange(int(math.pow(2, self.context_len)))
        random_index = np.arange(self.context_len)
        np.random.shuffle(random_index)
        contexts_scalar = random_index[:self.context_num]
        contexts_vector = np.zeros((self.context_num, self.context_len))
        for i in range(self.context_num):
            # contexts_vector[i, :] = np.binary_repr(contexts_scalar[i], width=self.context_len)
            contexts_vector[i, contexts_scalar[i]] = 1
        
        max_reward_arms = np.random.randint(self.num_arms, size=self.context_num) + 1

        context_pool = np.arange(self.batch_size) % self.context_len
        np.random.shuffle(context_pool)

        return contexts_vector, max_reward_arms, context_pool

    def _generate_sequence_len(self):
        if isinstance(self.sequence_len, int):
            return self.sequence_len
        else:
            return np.random.randint(self.sequence_len[0], self.sequence_len[1])

    def regenerate_contexts(self):
        self.contexts, self.max_reward_arms, self.context_pool = self._generate_contexts()
        self.timestep = 0
        self.current_seq_len = self._generate_sequence_len()
        self.current_context_pool_index = 0
        self.current_context_num = self.context_pool[self.current_context_pool_index]

    def step(self, action):
        action = int(action)
        assert self.action_space.contains(action)
        assert self.timestep < self.current_seq_len
        self.timestep += 1
        if action == 0:
            reward = 0.0
        elif action == self.max_reward_arms[self.current_context_num]:
            reward = np.random.choice([self.true_reward, self.false_reward], p=[self.max_arm_reward_prob, 1 - self.max_arm_reward_prob])
        else:
            reward = np.random.choice([self.true_reward, self.false_reward], p=[self.other_arms_reward_prob, 1 - self.other_arms_reward_prob])
        done = self.timestep == self.current_seq_len
        info = {"encoding_on": self.timestep == self.current_seq_len - 1}
        observation = np.zeros(self.context_len)
        # self.contexts_seen_before[self.current_context_num] = True
        return observation, reward, done, info

    def reset(self, context_index=None):
        """
        context_index: for testing, if None, get the next context from batch pool, else get the specified context
        """
        self.timestep = 0
        self.current_seq_len = self._generate_sequence_len()
        if context_index is None:
            self.current_context_pool_index = (self.current_context_pool_index + 1) % self.batch_size
            self.current_context_num = self.context_pool[self.current_context_pool_index]
        else:
            assert isinstance(context_index, int)
            assert context_index >=0 and context_index < self.context_num
            self.current_context_num = context_index
        observation = self.contexts[self.current_context_num]
        return observation

    def render(self, mode='human'):
        pass

    def compute_accuracy(self, actions):
        correct_actions = 0
        wrong_actions = 0
        for i in range(self.current_seq_len):
            if actions[i] == self.max_reward_arms[self.current_context_num]:
                correct_actions += 1
            elif actions[i] != 0:
                wrong_actions += 1
        return correct_actions, wrong_actions
    


# class BarcodeBandits(gym.Env):
#     """
#     Gym environment for the Barcode Bandits problem.
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

#         self.contexts, self.max_reward_arms = self._generate_contexts()
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
        
#         max_reward_arms = np.random.randint(self.num_arms, size=self.context_num) + 1
#         # max_reward_arms_vector = np.zeros((self.context_num, self.num_arms + 1))
#         # for i in range(self.context_num):
#         #     max_reward_arms_vector[i, max_reward_arms[i]] = 1
#         return contexts_vector, max_reward_arms

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

